from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gradio as gr

from my_llm.config import settings
from my_llm.context_store import ContextStore
from my_llm.fine_tuner import FineTuneParams, FineTuner
from my_llm.llm_service import LLMService


context_store: ContextStore | None = None
context_store_error: str | None = None
llm_service: LLMService | None = None
llm_service_error: str | None = None
fine_tuner = FineTuner()


def _new_conversations_state() -> dict[str, Any]:
    conv_id = "conv-1"
    return {
        "counter": 1,
        "active": conv_id,
        "order": [conv_id],
        "items": {
            conv_id: {
                "title": "Conversa 1",
                "history": [],
                "memory": [],
            }
        },
    }


def _ensure_conversations_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state or "items" not in state or not state.get("order"):
        return _new_conversations_state()
    return state


def _conversation_choices(state: dict[str, Any]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for conv_id in state["order"]:
        item = state["items"].get(conv_id, {})
        label = str(item.get("title") or conv_id)
        choices.append((label, conv_id))
    return choices


def _conversation_selector_update(state: dict[str, Any]):
    return gr.update(choices=_conversation_choices(state), value=state["active"])


def _get_active_conversation(state: dict[str, Any]) -> dict[str, Any]:
    state = _ensure_conversations_state(state)
    active = state["active"]
    if active not in state["items"]:
        active = state["order"][0]
        state["active"] = active
    return state["items"][active]


def _dataset_path() -> Path:
    return Path(settings.finetune_data_path)


def _auth_file_path() -> Path:
    return Path(settings.auth_file_path)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _ensure_auth_file() -> None:
    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    if auth_file.exists():
        return

    auth_file.write_text(
        json.dumps(
            {
                "username": settings.auth_username,
                "password_hash": _hash_password(settings.auth_password),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_auth_record() -> dict[str, str]:
    _ensure_auth_file()
    auth_file = _auth_file_path()
    try:
        data = json.loads(auth_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    username = str(data.get("username") or "").strip()
    password_hash = str(data.get("password_hash") or "").strip()
    if not username or not password_hash:
        username = settings.auth_username
        password_hash = _hash_password(settings.auth_password)
        auth_file.write_text(
            json.dumps({"username": username, "password_hash": password_hash}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {"username": username, "password_hash": password_hash}


def _verify_login(username: str, password: str) -> bool:
    record = _load_auth_record()
    return username.strip() == record["username"] and _hash_password(password) == record["password_hash"]


def do_login(username: str, password: str):
    if _verify_login(username, password):
        return (
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            _new_conversations_state(),
        )
    return (
        gr.update(value="Usuário ou senha inválidos.", visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        _new_conversations_state(),
    )


def do_logout():
    return (
        gr.update(value="Sessão encerrada.", visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        _new_conversations_state(),
    )


def save_local_credentials(new_username: str, new_password: str) -> str:
    new_username = (new_username or "").strip()
    new_password = (new_password or "").strip()
    if not new_username or not new_password:
        return "Preencha usuário e senha para salvar."

    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_file.write_text(
        json.dumps(
            {
                "username": new_username,
                "password_hash": _hash_password(new_password),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"Credenciais salvas em {auth_file}."


def _read_finetune_dataset() -> list[dict[str, str]]:
    path = _dataset_path()
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue

        instruction = str(parsed.get("instruction") or parsed.get("input") or "").strip()
        output = str(parsed.get("output") or parsed.get("response") or "").strip()
        if instruction or output:
            rows.append({"instruction": instruction, "output": output})
    return rows


def _format_dataset_preview(limit: int = 30) -> str:
    rows = _read_finetune_dataset()
    if not rows:
        return "Nenhum exemplo no dataset ainda."

    lines = [f"Total de exemplos: {len(rows)}"]
    start = max(0, len(rows) - limit)
    for idx, row in enumerate(rows[start:], start=start + 1):
        lines.append(f"\n[{idx}] Instrução: {row['instruction']}")
        lines.append(f"[{idx}] Resposta: {row['output']}")
    return "\n".join(lines)


def refresh_finetune_dataset() -> str:
    return _format_dataset_preview()


def add_finetune_example(instruction: str, output: str) -> tuple[str, str, str, str]:
    instruction = (instruction or "").strip()
    output = (output or "").strip()

    if not instruction or not output:
        return "Preencha Instrução e Resposta.", instruction, output, _format_dataset_preview()

    path = _dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(record + "\n")

    return "Exemplo adicionado ao dataset.", "", "", _format_dataset_preview()


def _debug_status(retrieved_count: int = 0, scores: list[float] | None = None) -> str:
    service = _get_llm_service()
    adapter_line = "adapter=indisponível"
    if service is not None:
        adapter_line = f"adapter_ativo={service.adapter_active} | {service.adapter_status}"

    context_line = "context_store=ok" if _get_context_store() is not None else f"context_store=erro ({context_store_error})"
    avg_score = "-"
    range_score = "-"
    if scores:
        avg_score = f"{(sum(scores) / len(scores)):.3f}"
        range_score = f"{min(scores):.3f}..{max(scores):.3f}"
    return (
        f"chunks_recuperados={retrieved_count}\n"
        f"score_medio={avg_score} | faixa_score={range_score}\n"
        f"{adapter_line}\n"
        f"{context_line}\n"
        f"base_model={settings.base_model}"
    )


def refresh_debug_info() -> str:
    return _debug_status(0)


def _format_context_files() -> str:
    root = Path(settings.context_dir)
    root.mkdir(parents=True, exist_ok=True)

    context_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"}]
    )
    if not context_files:
        return "Nenhum contexto carregado ainda."

    lines = [f"Total de arquivos de contexto: {len(context_files)}"]
    for file_path in context_files:
        rel = file_path.relative_to(root)
        size = file_path.stat().st_size
        lines.append(f"- {rel} ({size} bytes)")
    return "\n".join(lines)


def refresh_context_files() -> str:
    return _format_context_files()


def create_conversation(state: dict[str, Any] | None):
    state = _ensure_conversations_state(state)
    state["counter"] += 1
    conv_id = f"conv-{state['counter']}"
    title = f"Conversa {state['counter']}"
    state["order"].append(conv_id)
    state["items"][conv_id] = {"title": title, "history": [], "memory": []}
    state["active"] = conv_id
    return state, _conversation_selector_update(state), [], _debug_status(0), ""


def select_conversation(selected_id: str, state: dict[str, Any] | None):
    state = _ensure_conversations_state(state)
    if selected_id in state["items"]:
        state["active"] = selected_id
    conv = _get_active_conversation(state)
    return conv.get("history", []), _debug_status(0), state, _conversation_selector_update(state), ""


def rename_active_conversation(new_title: str, state: dict[str, Any] | None):
    state = _ensure_conversations_state(state)
    title = (new_title or "").strip()
    if not title:
        return state, _conversation_selector_update(state), "Informe um nome para a conversa."
    conv = _get_active_conversation(state)
    conv["title"] = title
    return state, _conversation_selector_update(state), "Conversa renomeada."


def delete_active_conversation(state: dict[str, Any] | None):
    state = _ensure_conversations_state(state)
    active = state["active"]
    if len(state["order"]) == 1:
        conv = _get_active_conversation(state)
        conv["history"] = []
        conv["memory"] = []
        return state, _conversation_selector_update(state), [], _debug_status(0), "Conversa limpa (última conversa não pode ser removida)."

    state["order"] = [cid for cid in state["order"] if cid != active]
    state["items"].pop(active, None)
    state["active"] = state["order"][0]
    conv = _get_active_conversation(state)
    return state, _conversation_selector_update(state), conv.get("history", []), _debug_status(0), "Conversa removida."


def clear_active_conversation(state: dict[str, Any] | None):
    state = _ensure_conversations_state(state)
    conv = _get_active_conversation(state)
    conv["history"] = []
    conv["memory"] = []
    return state, [], _debug_status(0), "Conversa limpa."


def chat(
    message: str,
    state: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], str, str, dict[str, Any]]:
    state = _ensure_conversations_state(state)
    conv = _get_active_conversation(state)
    history = conv.get("history", [])
    session_memory = conv.get("memory", [])
    message = (message or "").strip()
    if not message:
        return history, "", _debug_status(0), state

    retrieved_items: list[dict[str, Any]] = []
    store = _get_context_store()
    if store is not None:
        retrieved_items = store.query(message)
    retrieved_items = [
        item for item in retrieved_items if float(item.get("score", 0.0)) >= settings.context_use_score_floor
    ]

    service = _get_llm_service()
    if service is None:
        error = llm_service_error or "Não foi possível inicializar o modelo de linguagem."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Erro: {error}"})
        conv["history"] = history
        conv["memory"] = session_memory
        return history, "", _debug_status(0), state

    context_texts = [item["text"] for item in retrieved_items]
    response_core = service.chat(message, contexts=context_texts, history=session_memory)
    response = response_core

    if retrieved_items:
        citations = []
        for idx, item in enumerate(retrieved_items[:3], start=1):
            snippet = " ".join(item["text"].split())[: settings.citation_snippet_chars]
            ellipsis = "..." if len(" ".join(item["text"].split())) > settings.citation_snippet_chars else ""
            citations.append(
                f"[{idx}] {item.get('source', 'desconhecido')} | score={item.get('score', 0):.3f} | {snippet}{ellipsis}"
            )
        response = f"{response}\n\nFontes do contexto:\n" + "\n".join(citations)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    session_memory = session_memory + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_core},
    ]
    session_memory = session_memory[-12:]
    conv["history"] = history
    conv["memory"] = session_memory
    scores = [float(item.get("score", 0.0)) for item in retrieved_items]
    return history, "", _debug_status(len(retrieved_items), scores), state


def ingest_context(files: list[Any] | None) -> tuple[str, str]:
    if not files:
        return "Nenhum arquivo enviado.", _format_context_files()

    target = Path(settings.context_dir)
    target.mkdir(parents=True, exist_ok=True)

    for f in files:
        src = Path(f.name)
        dst = target / src.name
        dst.write_bytes(src.read_bytes())

    store = _get_context_store()
    if store is None:
        return (
            f"Contexto indisponível: {context_store_error or 'falha ao inicializar modelo de embeddings.'}",
            _format_context_files(),
        )

    total = store.ingest_directory(str(target))
    return f"Contexto atualizado com sucesso. Chunks indexados: {total}", _format_context_files()


def run_fine_tuning(epochs: int, batch_size: int, learning_rate: float) -> tuple[str, str]:
    params = FineTuneParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    try:
        output_dir = fine_tuner.run(params)
    except Exception as exc:
        return f"Erro no fine-tuning: {exc}", _format_dataset_preview()

    global llm_service, llm_service_error
    llm_service = None
    llm_service_error = None
    return f"Fine-tuning finalizado. Adapter salvo em: {output_dir}", _format_dataset_preview()


def _get_context_store() -> ContextStore | None:
    global context_store, context_store_error

    if context_store is not None:
        return context_store

    if context_store_error is not None:
        return None

    try:
        context_store = ContextStore()
        context_store.load()
        return context_store
    except Exception as exc:
        context_store_error = str(exc)
        return None


def _get_llm_service() -> LLMService | None:
    global llm_service, llm_service_error

    if llm_service is not None:
        return llm_service

    if llm_service_error is not None:
        return None

    try:
        llm_service = LLMService()
        return llm_service
    except Exception as exc:
        llm_service_error = str(exc)
        return None


def build_ui() -> gr.Blocks:
    _ensure_auth_file()

    with gr.Blocks(title="My LLM Studio") as demo:
        gr.Markdown("# My LLM Studio")
        conversations_state = gr.State(_new_conversations_state())

        login_status = gr.Textbox(label="Status de acesso", interactive=False, value="Faça login para continuar.")

        with gr.Column(visible=True) as login_panel:
            with gr.Group():
                gr.Markdown("## Login")
                login_user = gr.Textbox(label="Usuário", placeholder="Digite seu usuário")
                login_pass = gr.Textbox(label="Senha", type="password", placeholder="Digite sua senha")
                login_btn = gr.Button("Entrar", variant="primary")

        with gr.Column(visible=False) as app_panel:
            with gr.Row():
                gr.Markdown("Chat + Contexto (RAG) + Ajuste fino LoRA")
                logout_btn = gr.Button("Sair", variant="secondary", size="sm", min_width=80)

            with gr.Tab("Chat"):
                chat_selector = gr.Dropdown(
                    label="Conversa",
                    choices=_conversation_choices(_new_conversations_state()),
                    value="conv-1",
                )
                with gr.Row():
                    new_chat_btn = gr.Button("Nova conversa")
                    clear_chat_btn = gr.Button("Limpar conversa")
                    delete_chat_btn = gr.Button("Excluir conversa")
                with gr.Row():
                    rename_chat_input = gr.Textbox(label="Renomear conversa", placeholder="Novo nome da conversa")
                    rename_chat_btn = gr.Button("Renomear")
                chat_status = gr.Textbox(label="Status da conversa", interactive=False)

                chatbot = gr.Chatbot(label="Conversa", height=500, type="messages")
                msg = gr.Textbox(label="Mensagem", placeholder="Digite sua pergunta...")
                send = gr.Button("Enviar", variant="primary")
                debug_info = gr.Textbox(label="Depuração", value=refresh_debug_info(), lines=5, interactive=False)
                refresh_debug_btn = gr.Button("Atualizar depuração")

                chat_selector.change(
                    select_conversation,
                    inputs=[chat_selector, conversations_state],
                    outputs=[chatbot, debug_info, conversations_state, chat_selector, chat_status],
                )
                new_chat_btn.click(
                    create_conversation,
                    inputs=[conversations_state],
                    outputs=[conversations_state, chat_selector, chatbot, debug_info, chat_status],
                )
                rename_chat_btn.click(
                    rename_active_conversation,
                    inputs=[rename_chat_input, conversations_state],
                    outputs=[conversations_state, chat_selector, chat_status],
                )
                clear_chat_btn.click(
                    clear_active_conversation,
                    inputs=[conversations_state],
                    outputs=[conversations_state, chatbot, debug_info, chat_status],
                )
                delete_chat_btn.click(
                    delete_active_conversation,
                    inputs=[conversations_state],
                    outputs=[conversations_state, chat_selector, chatbot, debug_info, chat_status],
                )

                send.click(chat, inputs=[msg, conversations_state], outputs=[chatbot, msg, debug_info, conversations_state])
                msg.submit(chat, inputs=[msg, conversations_state], outputs=[chatbot, msg, debug_info, conversations_state])
                refresh_debug_btn.click(refresh_debug_info, outputs=[debug_info])

            with gr.Tab("Contexto"):
                files = gr.File(
                    label="Enviar arquivos .txt, .md ou .json",
                    file_count="multiple",
                )
                ingest_btn = gr.Button("Indexar contexto")
                ingest_status = gr.Textbox(label="Status", interactive=False)
                context_files = gr.Textbox(
                    label="Contextos carregados",
                    value=refresh_context_files(),
                    lines=8,
                    interactive=False,
                )
                refresh_context_btn = gr.Button("Atualizar contextos")

                ingest_btn.click(ingest_context, inputs=[files], outputs=[ingest_status, context_files])
                refresh_context_btn.click(refresh_context_files, outputs=[context_files])

            with gr.Tab("Ajuste fino"):
                gr.Markdown(
                    "Monte seu dataset aqui e rode o treino. Arquivo salvo em "
                    f"`{settings.finetune_data_path}` com campos `instruction` (instrução) e `output` (resposta)."
                )

                instruction = gr.Textbox(label="Instrução", lines=3, placeholder="Ex: Explique score de crédito de forma simples")
                output = gr.Textbox(label="Resposta", lines=5, placeholder="Resposta esperada")
                add_example_btn = gr.Button("Adicionar exemplo")

                dataset_preview = gr.Textbox(
                    label="Dataset atual",
                    value=_format_dataset_preview(),
                    lines=12,
                    interactive=False,
                )
                refresh_dataset_btn = gr.Button("Atualizar visualização")

                epochs = gr.Slider(1, 5, value=1, step=1, label="Épocas")
                batch_size = gr.Slider(1, 8, value=2, step=1, label="Tamanho do lote")
                learning_rate = gr.Number(value=2e-4, label="Taxa de aprendizado")
                train_btn = gr.Button("Iniciar fine-tuning", variant="primary")
                train_status = gr.Textbox(label="Status", interactive=False)

                add_example_btn.click(
                    add_finetune_example,
                    inputs=[instruction, output],
                    outputs=[train_status, instruction, output, dataset_preview],
                )
                refresh_dataset_btn.click(refresh_finetune_dataset, outputs=[dataset_preview])
                train_btn.click(
                    run_fine_tuning,
                    inputs=[epochs, batch_size, learning_rate],
                    outputs=[train_status, dataset_preview],
                )

            with gr.Tab("Como usar"):
                gr.Markdown(
                    "## Guia rápido\n"
                    "1. Abra **Contexto** e envie arquivos `.txt`, `.md` ou `.json`.\n"
                    "2. Clique em **Indexar contexto**.\n"
                    "3. Vá para **Chat**, faça perguntas e acompanhe o painel **Depuração**.\n"
                    "4. Em **Ajuste fino**, adicione exemplos e rode o treino.\n\n"
                    "## Variáveis de ambiente (.env)\n"
                    "- `GRADIO_SERVER_NAME`: host da aplicação (ex.: `0.0.0.0`).\n"
                    "- `GRADIO_SERVER_PORT`: porta do servidor (ex.: `7860`).\n"
                    "- `OFFLINE_MODE`: `1` para somente local, `0` para permitir download.\n"
                    "- `DOWNLOAD_MODELS_ON_STARTUP`: `1` para baixar/verificar modelos no boot quando `OFFLINE_MODE=0`.\n"
                    "- `FORCE_DOWNLOAD_ON_STARTUP`: `1` para forçar novo download em todo boot, `0` para reutilizar cache.\n"
                    "- `BASE_MODEL`: modelo principal de geração.\n"
                    "- `EMBEDDING_MODEL`: modelo de embeddings do RAG.\n"
                    "- `CONTEXT_DIR`: pasta dos arquivos de contexto.\n"
                    "- `VECTOR_STORE_PATH`: caminho do índice FAISS.\n"
                    "- `CHUNKS_PATH`: caminho do arquivo de chunks.\n"
                    "- `TOP_K_CONTEXT`: referência de chunks a recuperar.\n"
                    "- `MIN_CONTEXT_SCORE`: score mínimo para aceitar chunk no RAG.\n"
                    "- `MAX_CONTEXT_CHUNKS`: limite máximo de chunks por pergunta.\n"
                    "- `STRICT_CONTEXT_MODE`: `1` exige evidência no contexto; `0` permite fallback.\n"
                    "- `FINETUNE_DATA_PATH`: caminho do dataset de fine-tuning (`jsonl`).\n"
                    "- `OUTPUT_MODEL_DIR`: pasta onde o adapter LoRA é salvo.\n"
                    "- `USE_ADAPTER`: `1` habilita adapter, `0` usa só modelo base.\n"
                    "- `MAX_NEW_TOKENS`: limite de tokens de resposta.\n"
                    "- `TEMPERATURE`: criatividade da geração (menor = mais determinístico).\n"
                    "- `TOP_P`: amostragem núcleo (quando aplicável).\n"
                    "- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`: cache local Hugging Face.\n"
                    "- `AUTH_FILE_PATH`: arquivo local das credenciais de login.\n"
                    "- `AUTH_USERNAME`, `AUTH_PASSWORD`: credencial inicial para bootstrap.\n\n"
                    "## Dica de operação\n"
                    "- Em produção online no primeiro deploy: `OFFLINE_MODE=0` para baixar modelos.\n"
                    "- Após cachear modelos: mude para `OFFLINE_MODE=1` para operação local."
                )

            with gr.Accordion("Gerenciar acesso", open=False):
                gr.Markdown(f"Salve usuário e senha localmente em `{settings.auth_file_path}`.")
                new_user = gr.Textbox(label="Novo usuário")
                new_pass = gr.Textbox(label="Nova senha", type="password")
                save_auth_btn = gr.Button("Salvar credenciais", variant="primary")
                save_auth_status = gr.Textbox(label="Status", interactive=False)
                save_auth_btn.click(save_local_credentials, inputs=[new_user, new_pass], outputs=[save_auth_status])

        login_btn.click(do_login, inputs=[login_user, login_pass], outputs=[login_status, login_panel, app_panel, conversations_state])
        logout_btn.click(do_logout, outputs=[login_status, login_panel, app_panel, conversations_state])

    return demo


if __name__ == "__main__":
    build_ui().launch()
