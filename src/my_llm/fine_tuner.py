from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from my_llm.config import settings


@dataclass
class FineTuneParams:
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_length: int = 256


class FineTuner:
    def __init__(self) -> None:
        self.base_model = settings.base_model
        self.data_path = settings.finetune_data_path
        self.output_dir = settings.output_model_dir

    def _format_example(self, row: dict) -> str:
        instruction = row.get("instruction") or row.get("input") or ""
        answer = row.get("output") or row.get("response") or ""
        return f"Instruction: {instruction}\nAnswer: {answer}"

    def _infer_lora_target_modules(self, model: AutoModelForCausalLM) -> list[str]:
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

        # Qwen/Llama-like
        qwen_llama_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(module_names):
            return [name for name in qwen_llama_targets if name in module_names]

        # GPT-2 style
        if "c_attn" in module_names:
            return ["c_attn"]

        # Falcon / MPT-like fallbacks
        fallback_targets = ["query_key_value", "dense", "fc_in", "fc_out", "Wqkv"]
        found = [name for name in fallback_targets if name in module_names]
        if found:
            return found

        # Conservative fallback: common projection names only if present.
        generic = ["q_proj", "v_proj"]
        return [name for name in generic if name in module_names]

    def run(self, params: FineTuneParams | None = None) -> str:
        params = params or FineTuneParams()

        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(
                f"Fine-tuning dataset not found at: {self.data_path}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            local_files_only=settings.offline_mode,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            local_files_only=settings.offline_mode,
        )
        target_modules = self._infer_lora_target_modules(model)
        if not target_modules:
            raise ValueError(
                "Não foi possível inferir target_modules para LoRA neste modelo base."
            )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

        dataset = load_dataset("json", data_files=str(data_file), split="train")

        def tokenize_fn(row: dict) -> dict:
            text = self._format_example(row)
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=params.max_length,
                padding="max_length",
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=params.epochs,
            per_device_train_batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            logging_steps=5,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        return self.output_dir
