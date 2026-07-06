from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import get_settings
from rag.bm25 import BM25Retriever
from vectordb.vector_store import FAISSVectorStore
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.retriever")


class Retriever:
    """Dense + BM25 retrieval orchestrator."""

    def __init__(self, vector_store: FAISSVectorStore | None = None, bm25_retriever: BM25Retriever | None = None) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or FAISSVectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.settings.top_k
        dense_results = self.vector_store.similarity_search(query, k=k)
        bm25_results = self.bm25_retriever.search(query, k=k)
        combined = dense_results + bm25_results
        unique: dict[str, Document] = {}
        for document in combined:
            doc_key = document.metadata.get("chunk_id") or document.metadata.get("source") or document.page_content
            if doc_key not in unique:
                unique[doc_key] = document
        return list(unique.values())[:k]
