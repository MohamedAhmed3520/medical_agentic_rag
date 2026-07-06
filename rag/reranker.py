from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


class Reranker:
    """Simple reranker wrapper with configurable model placeholder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name

    def rerank(self, documents: list[Document], query: str) -> list[Document]:
        if not documents:
            return []
        scored = sorted(documents, key=lambda doc: len((doc.page_content or "").split()), reverse=False)
        return scored[: min(5, len(scored))]
