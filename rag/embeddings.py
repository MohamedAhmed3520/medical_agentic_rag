from __future__ import annotations

from typing import Any

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    HuggingFaceEmbeddings = None  # type: ignore[assignment]


class EmbeddingModel:
    """Wrapper around Hugging Face embeddings with a compatibility fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        if HuggingFaceEmbeddings is None:
            self.client = None
        else:
            self.client = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, text: str) -> list[float]:
        if self.client is None:
            return [0.0] * 384
        return self.client.embed_query(text)

    def embed_documents(self, documents: list[Any]) -> list[list[float]]:
        if self.client is None:
            return [[0.0] * 384 for _ in documents]
        return self.client.embed_documents([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents])
