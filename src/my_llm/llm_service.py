from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from my_llm.config import settings


class LLMService:
    QUESTION_STOPWORDS = {
        "o",
        "a",
        "os",
        "as",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "e",
        "é",
        "eh",
        "que",
        "qual",
        "quais",
        "quem",
        "como",
        "para",
        "por",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "me",
        "te",
        "se",
        "sigla",
        "significa",
        "defina",
        "explique",
        "sobre",
        "seu",
        "sua",
        "seus",
        "suas",
        "meu",
        "minha",
        "meus",
        "minhas",
        "ola",
        "olá",
        "oi",
        "bom",
        "boa",
        "dia",
        "tarde",
        "noite",
    }

    def __init__(self) -> None:
        self.adapter_active = False
        self.adapter_status = "Adapter desativado."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model,
            local_files_only=settings.offline_mode,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=settings.offline_mode,
        )
        self.model.to(self.device)
        self.model.eval()
        # Deterministic generation: remove sampling-only params to avoid runtime warnings.
        if self.model.generation_config is not None:
            self.model.generation_config.top_k = None
            self.model.generation_config.top_p = None
            self.model.generation_config.temperature = None

        adapter_dir = Path(settings.output_model_dir)
        if not settings.use_adapter:
            self.adapter_status = "Adapter desativado por configuração (USE_ADAPTER=0)."
        elif not adapter_dir.exists() or not any(adapter_dir.iterdir()):
            self.adapter_status = "Adapter não encontrado. Usando apenas modelo base."
        else:
            config_path = adapter_dir / "adapter_config.json"
            if not config_path.exists():
                self.adapter_status = "Adapter ignorado: adapter_config.json não encontrado."
            else:
                try:
                    adapter_cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    adapter_base = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
                    target_modules = adapter_cfg.get("target_modules") or []
                    if isinstance(target_modules, str):
                        target_modules = [target_modules]

                    current_modules = {name.split(".")[-1] for name, _ in self.model.named_modules()}
                    missing_targets = [name for name in target_modules if name not in current_modules]

                    if adapter_base and adapter_base != settings.base_model:
                        self.adapter_status = (
                            "Adapter ignorado: foi treinado para outro modelo base "
                            f"('{adapter_base}')."
                        )
                    elif missing_targets:
                        self.adapter_status = (
                            "Adapter ignorado: target_modules incompatíveis com o modelo atual "
                            f"({missing_targets})."
                        )
                    else:
                        self.model = PeftModel.from_pretrained(self.model, settings.output_model_dir)
                        self.model.to(self.device)
                        self.model.eval()
                        self.adapter_active = True
                        self.adapter_status = "Adapter carregado com sucesso."
                except Exception as exc:
                    self.adapter_status = f"Adapter ignorado: {exc}"

        self.finetune_examples = self._load_finetune_examples()
        self._finetune_mtime: float | None = self._finetune_file_mtime()

    def _finetune_file_mtime(self) -> float | None:
        path = Path(settings.finetune_data_path)
        if not path.exists():
            return None
        return path.stat().st_mtime

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_-]+", lowered)
        return " ".join(tokens)

    def _load_finetune_examples(self) -> list[tuple[str, str]]:
        path = Path(settings.finetune_data_path)
        if not path.exists():
            return []

        examples: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = str(parsed.get("instruction") or parsed.get("input") or "").strip()
            output = str(parsed.get("output") or parsed.get("response") or "").strip()
            if instruction and output:
                examples.append((instruction, output))
        return examples

    def _finetune_fallback(self, user_message: str) -> str | None:
        current_mtime = self._finetune_file_mtime()
        if current_mtime != self._finetune_mtime:
            self.finetune_examples = self._load_finetune_examples()
            self._finetune_mtime = current_mtime

        if not self.finetune_examples:
            return None

        user_norm = self._normalize_text(user_message)
        user_tokens = set(user_norm.split())
        if not user_tokens:
            return None

        # Keep only content tokens to make matching robust across different phrasings.
        user_content_tokens = {
            t for t in user_tokens if len(t) >= 3 and t not in self.QUESTION_STOPWORDS
        }
        key_tokens = user_content_tokens or {t for t in user_tokens if len(t) >= 3}

        best_score = 0.0
        best_output = None
        for instruction, output in self.finetune_examples:
            inst_tokens = set(self._normalize_text(instruction).split())
            if not inst_tokens:
                continue

            inst_content_tokens = {
                t for t in inst_tokens if len(t) >= 3 and t not in self.QUESTION_STOPWORDS
            }
            comparable_user = user_content_tokens or user_tokens
            comparable_inst = inst_content_tokens or inst_tokens

            inter = len(comparable_user & comparable_inst)
            union = len(comparable_user | comparable_inst)
            if union == 0:
                continue
            jaccard = inter / union
            coverage = inter / max(len(comparable_inst), 1)

            key_overlap = 0.0
            if key_tokens:
                key_hits = len(key_tokens & comparable_inst)
                key_overlap = key_hits / len(key_tokens)

            # Blend scores: lexical similarity + instruction coverage + acronym/keyword match.
            score = (0.55 * jaccard) + (0.30 * coverage) + (0.15 * key_overlap)
            if score > best_score:
                best_score = score
                best_output = output

        effective_threshold = settings.finetune_fallback_threshold
        if len(self.finetune_examples) <= 2:
            effective_threshold = min(effective_threshold, 0.30)
        elif len(self.finetune_examples) <= 5:
            effective_threshold = min(effective_threshold, 0.35)

        if best_output and best_score >= effective_threshold:
            return best_output
        return None

    def _build_prompt(self, user_message: str, contexts: List[str], history: list[dict[str, str]] | None = None) -> str:
        context_block = ""
        if contexts:
            joined = "\n\n---\n\n".join(contexts)
            context_block = f"Contexto relevante:\n{joined}\n\n"
        memory_block = ""
        if history:
            turns: list[str] = []
            for item in history[-6:]:
                role = item.get("role", "")
                content = (item.get("content", "") or "").strip()
                if not content:
                    continue
                if role == "user":
                    turns.append(f"Usuário: {content}")
                elif role == "assistant":
                    turns.append(f"Assistente: {content}")
            if turns:
                memory_block = "Histórico recente da conversa:\n" + "\n".join(turns) + "\n\n"

        strict_rule = ""
        if contexts:
            strict_rule = (
                "Regra obrigatória: se houver contexto relevante, responda somente com base nele. "
                "Não use conhecimento externo e não invente fatos. "
                "Se a informação não estiver no contexto, diga explicitamente: "
                "'Não encontrei essa informação no contexto carregado.'\n\n"
            )

        return (
            "Você é um assistente técnico útil, objetivo e claro. "
            "Responda sempre em português do Brasil, salvo pedido explícito por outro idioma.\n\n"
            f"{strict_rule}"
            f"{context_block}"
            f"{memory_block}"
            f"Pergunta do usuário: {user_message}\n"
            "Resposta:"
        )

    def chat(
        self,
        user_message: str,
        contexts: List[str] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        fallback = self._finetune_fallback(user_message)
        if fallback:
            return fallback

        context_list = contexts or []

        prompt = self._build_prompt(user_message, context_list, history)

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente técnico útil e objetivo. "
                        "Responda em português do Brasil. "
                        "Se houver contexto fornecido, use apenas esse contexto e não invente fatos. "
                        "Se não houver informação suficiente, diga que não encontrou no contexto."
                    ),
                }
            ]
            if context_list:
                messages.append(
                    {
                        "role": "system",
                        "content": "Use apenas este contexto quando for relevante:\n\n" + "\n\n---\n\n".join(context_list),
                    }
                )
            if history:
                for item in history[-6:]:
                    role = item.get("role", "")
                    content = (item.get("content", "") or "").strip()
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_message})
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0][prompt_tokens:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if decoded:
            return decoded

        full_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Resposta:" in full_decoded:
            return full_decoded.split("Resposta:", maxsplit=1)[-1].strip()
        return full_decoded.strip()
