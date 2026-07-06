from __future__ import annotations

from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:
    """BM25-based keyword retrieval over document chunks."""

    def __init__(self) -> None:
        self.tokenized_documents: list[list[str]] = []
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        self.tokenized_documents = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        return [self.documents[idx] for idx in ranked]
