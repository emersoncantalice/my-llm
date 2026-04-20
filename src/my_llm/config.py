from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    base_model: str = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    context_dir: str = os.getenv("CONTEXT_DIR", "data/contexts")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "data/contexts/faiss.index")
    chunks_path: str = os.getenv("CHUNKS_PATH", "data/contexts/chunks.json")
    finetune_data_path: str = os.getenv("FINETUNE_DATA_PATH", "data/finetune/train.jsonl")
    output_model_dir: str = os.getenv("OUTPUT_MODEL_DIR", "data/finetune/adapters")
    use_adapter: bool = os.getenv("USE_ADAPTER", "1") == "1"
    auth_file_path: str = os.getenv("AUTH_FILE_PATH", "data/auth.json")
    auth_username: str = os.getenv("AUTH_USERNAME", "admin")
    auth_password: str = os.getenv("AUTH_PASSWORD", "admin123")
    offline_mode: bool = os.getenv("OFFLINE_MODE", "1") == "1"
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "200"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    top_k_context: int = int(os.getenv("TOP_K_CONTEXT", "3"))
    min_context_score: float = float(os.getenv("MIN_CONTEXT_SCORE", "0.30"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
    strict_context_mode: bool = os.getenv("STRICT_CONTEXT_MODE", "1") == "1"
    context_use_score_floor: float = float(os.getenv("CONTEXT_USE_SCORE_FLOOR", "0.45"))
    citation_snippet_chars: int = int(os.getenv("CITATION_SNIPPET_CHARS", "220"))
    finetune_fallback_threshold: float = float(os.getenv("FINETUNE_FALLBACK_THRESHOLD", "0.40"))


settings = Settings()
