import os
from pathlib import Path

from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _download_models_on_startup() -> None:
    # Baixa (ou valida cache) dos modelos no boot para evitar erro no primeiro uso.
    # Em Windows sem privilegio de symlink, usamos pasta local materializada.
    from huggingface_hub import snapshot_download

    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model_home = Path(os.getenv("LOCAL_MODELS_DIR", "data/models"))
    base_local_dir = Path(os.getenv("BASE_MODEL_LOCAL_DIR", str(model_home / "base-model")))
    emb_local_dir = Path(os.getenv("EMBEDDING_MODEL_LOCAL_DIR", str(model_home / "embedding-model")))
    base_local_dir.mkdir(parents=True, exist_ok=True)
    emb_local_dir.mkdir(parents=True, exist_ok=True)
    force_download = _to_bool(os.getenv("FORCE_DOWNLOAD_ON_STARTUP", "0"))

    print(f"[startup] Download de modelos habilitado. local_dir={model_home}")
    print(f"[startup] Baixando/verificando modelo base: {base_model}")
    snapshot_download(
        repo_id=base_model,
        local_dir=str(base_local_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
    )
    print(f"[startup] Baixando/verificando modelo de embeddings: {embedding_model}")
    snapshot_download(
        repo_id=embedding_model,
        local_dir=str(emb_local_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
    )
    os.environ["BASE_MODEL"] = str(base_local_dir)
    os.environ["EMBEDDING_MODEL"] = str(emb_local_dir)
    print("[startup] Modelos prontos no cache local.")

def _prefer_local_models_if_available() -> None:
    model_home = Path(os.getenv("LOCAL_MODELS_DIR", "data/models"))
    base_local_dir = Path(os.getenv("BASE_MODEL_LOCAL_DIR", str(model_home / "base-model")))
    emb_local_dir = Path(os.getenv("EMBEDDING_MODEL_LOCAL_DIR", str(model_home / "embedding-model")))

    base_config = base_local_dir / "config.json"
    emb_config = emb_local_dir / "config.json"
    if base_config.exists():
        os.environ["BASE_MODEL"] = str(base_local_dir)
        print(f"[startup] BASE_MODEL local detectado: {base_local_dir}")
    if emb_config.exists():
        os.environ["EMBEDDING_MODEL"] = str(emb_local_dir)
        print(f"[startup] EMBEDDING_MODEL local detectado: {emb_local_dir}")


if __name__ == "__main__":
    load_dotenv()

    cache_root = Path(os.getenv("HF_HOME", "data/hf-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    offline_mode = os.getenv("OFFLINE_MODE", "1") == "1"
    if offline_mode:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _prefer_local_models_if_available()
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        _prefer_local_models_if_available()

    if not offline_mode and _to_bool(os.getenv("DOWNLOAD_MODELS_ON_STARTUP", "1"), default=True):
        _download_models_on_startup()

    from my_llm.ui import build_ui

    app = build_ui()
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name=host, server_port=port)
