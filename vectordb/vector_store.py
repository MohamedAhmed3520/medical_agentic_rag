from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from config import get_settings
from rag.embeddings import EmbeddingModel
from utils.file_utils import read_json, write_json
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.vector_store")


class FAISSVectorStore:
    """FAISS-backed vector store for dense retrieval."""

    def __init__(self, embedding_model: EmbeddingModel | None = None, path: str | Path | None = None) -> None:
        self.settings = get_settings()
        self.embedding_model = embedding_model or EmbeddingModel(self.settings.embedding_model)
        self.path = Path(path or self.settings.vector_store_path)
        self.metadata_path = Path(self.settings.metadata_path)
        self.index: faiss.Index | None = None
        self.documents: list[Document] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            self.index = faiss.read_index(str(self.path))
            metadata = read_json(self.metadata_path)
            self.documents = [Document(page_content=item["text"], metadata=item["metadata"]) for item in metadata.get("documents", [])]
            logger.info("Loaded existing FAISS index from %s", self.path)

    def build(self, documents: list[Document]) -> None:
        if not documents:
            return
        embeddings = self.embedding_model.embed_documents(documents)
        matrix = np.array(embeddings, dtype="float32")
        self.index = faiss.IndexFlatL2(matrix.shape[1])
        self.index.add(matrix)
        self.documents = documents
        self._persist()

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        if self.index is None:
            self.build(documents)
            return
        embeddings = self.embedding_model.embed_documents(documents)
        matrix = np.array(embeddings, dtype="float32")
        self.index.add(matrix)
        self.documents.extend(documents)
        self._persist()

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if self.index is None or not self.documents:
            return []
        query_vector = np.array([self.embedding_model.embed_query(query)], dtype="float32")
        distances, indices = self.index.search(query_vector, min(k, len(self.documents)))
        results: list[Document] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            if idx < len(self.documents):
                results.append(self.documents[int(idx)])
        return results

    def _persist(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.path))
        payload = {"documents": [{"text": doc.page_content, "metadata": doc.metadata} for doc in self.documents]}
        write_json(self.metadata_path, payload)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        self.index = None
        self.documents = []
