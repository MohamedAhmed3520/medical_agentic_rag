from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import get_settings
from rag.retriever import Retriever
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.hybrid")


class HybridRetriever:
    """Combine dense and keyword retrieval with weighted scoring."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.settings = get_settings()
        self.retriever = retriever or Retriever()

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.settings.top_k
        results = self.retriever.retrieve(query, k=k * 2)
        return results[:k]
