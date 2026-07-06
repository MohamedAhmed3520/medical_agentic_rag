from __future__ import annotations

from langchain_core.documents import Document


class BGEReranker:
    """Simple BGE-style reranker wrapper."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name

    def rerank(self, documents: list[Document], query: str) -> list[Document]:
        if not documents:
            return []
        return documents[: min(5, len(documents))]
