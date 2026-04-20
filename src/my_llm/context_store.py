from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from my_llm.config import settings


class ContextStore:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            local_files_only=settings.offline_mode,
        )
        self.chunk_records: List[dict[str, Any]] = []
        self.index: faiss.IndexFlatIP | None = None

    def load(self) -> None:
        chunks_file = Path(settings.chunks_path)
        index_file = Path(settings.vector_store_path)

        if not chunks_file.exists() or not index_file.exists():
            return

        loaded = json.loads(chunks_file.read_text(encoding="utf-8"))
        if loaded and isinstance(loaded[0], str):
            # Backward compatibility with old format.
            self.chunk_records = [{"text": text, "source": "desconhecido"} for text in loaded]
        else:
            self.chunk_records = loaded

        self.index = faiss.read_index(str(index_file))

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(end - overlap, 0)
        return chunks

    def _read_text_with_fallback(self, file_path: Path) -> str:
        raw_bytes = file_path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    def _dynamic_top_k(self, prompt: str) -> int:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_-]+", prompt)
        length = len(tokens)
        if length <= 5:
            return 2
        if length <= 12:
            return 3
        if length <= 24:
            return 4
        return min(settings.max_context_chunks, 6)

    def _lexical_overlap(self, query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        lowered = text.lower()
        hits = sum(1 for token in query_tokens if token in lowered)
        return hits / max(len(query_tokens), 1)

    def ingest_directory(self, context_dir: str | None = None) -> int:
        root = Path(context_dir or settings.context_dir)
        root.mkdir(parents=True, exist_ok=True)
        ignored_paths = {
            Path(settings.chunks_path).resolve(),
            Path(settings.vector_store_path).resolve(),
        }

        records: List[dict[str, Any]] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() in ignored_paths:
                continue
            if file_path.suffix.lower() not in {".txt", ".md", ".json"}:
                continue

            raw = self._read_text_with_fallback(file_path)
            if file_path.suffix.lower() == ".json":
                try:
                    obj = json.loads(raw)
                    raw = json.dumps(obj, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass

            for chunk in self._chunk_text(raw):
                records.append(
                    {
                        "text": chunk,
                        "source": str(file_path.relative_to(root)).replace("\\", "/"),
                    }
                )

        self.chunk_records = records
        if not self.chunk_records:
            self.index = None
            return 0

        embeddings = self.embedding_model.encode(
            [r["text"] for r in self.chunk_records],
            normalize_embeddings=True,
        )
        vectors = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        Path(settings.chunks_path).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.vector_store_path).parent.mkdir(parents=True, exist_ok=True)

        Path(settings.chunks_path).write_text(
            json.dumps(self.chunk_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        faiss.write_index(self.index, settings.vector_store_path)
        return len(self.chunk_records)

    def query(self, prompt: str, top_k: int | None = None, min_score: float | None = None) -> List[dict[str, Any]]:
        if not prompt.strip() or not self.index or not self.chunk_records:
            return []

        final_k = min(top_k or self._dynamic_top_k(prompt), settings.max_context_chunks, len(self.chunk_records))
        candidate_k = min(max(final_k * 4, final_k), len(self.chunk_records))

        query_embedding = self.embedding_model.encode([prompt], normalize_embeddings=True)
        query_vector = np.asarray(query_embedding, dtype="float32")
        scores, indices = self.index.search(query_vector, candidate_k)

        query_tokens = [
            t for t in re.findall(r"[a-zA-ZÀ-ÿ0-9_-]+", prompt.lower()) if len(t) >= 3
        ]

        results: List[dict[str, Any]] = []
        threshold = min_score if min_score is not None else settings.min_context_score
        for sem_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = self.chunk_records[idx]
            lexical = self._lexical_overlap(query_tokens, record["text"])
            combined = 0.75 * float(sem_score) + 0.25 * lexical
            if combined < threshold:
                continue
            results.append(
                {
                    "text": record["text"],
                    "source": record.get("source", "desconhecido"),
                    "score": round(combined, 4),
                    "semantic_score": round(float(sem_score), 4),
                    "lexical_score": round(lexical, 4),
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:final_k]
