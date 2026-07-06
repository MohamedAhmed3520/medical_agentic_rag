from __future__ import annotations

from langchain_core.documents import Document

from rag.retriever import Retriever


class MultiQueryRetriever:
    """Retrieve evidence using multiple rewritten queries."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return self.retriever.retrieve(query, k=k)
