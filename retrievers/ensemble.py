from __future__ import annotations

from langchain_core.documents import Document

from rag.retriever import Retriever


class EnsembleRetriever:
    """Combine multiple retrieval strategies."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        results = self.retriever.retrieve(query, k=k * 2)
        return results[:k]
