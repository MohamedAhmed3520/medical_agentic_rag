from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        documents: list[Document],
        query: str,
        top_k: int = 5,
    ) -> list[Document]:

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked[:top_k]]
