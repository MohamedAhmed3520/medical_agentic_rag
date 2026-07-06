from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import get_settings
from rag.bm25 import BM25Retriever
from rag.cleaner import TextCleaner
from rag.compressor import ContextCompressor
from rag.citation import CitationBuilder
from rag.loader import PDFDocumentLoader
from rag.query_rewriter import QueryRewriter
from rag.reranker import Reranker
from rag.splitter import DocumentSplitter
from rag.embeddings import EmbeddingModel
from vectordb.vector_store import FAISSVectorStore
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.pipeline")


@dataclass(slots=True)
class ProcessingResult:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    answer: str = ""


class RAGPipeline:
    """End-to-end ingestion and retrieval pipeline."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.settings = get_settings()
        self.data_dir = Path(data_dir or self.settings.data_dir)
        self.loader = PDFDocumentLoader(self.data_dir)
        self.cleaner = TextCleaner()
        self.splitter = DocumentSplitter(self.settings.chunk_size, self.settings.chunk_overlap)
        self.embedding_model = EmbeddingModel(self.settings.embedding_model)
        self.vector_store = FAISSVectorStore(self.embedding_model, self.settings.vector_store_path)
        self.bm25_retriever = BM25Retriever()
        self.reranker = Reranker(self.settings.reranker_model)
        self.compressor = ContextCompressor()
        self.citation_builder = CitationBuilder()
        self.query_rewriter = QueryRewriter()

    def ingest_documents(self, file_paths: list[str | Path], rebuild: bool = False) -> ProcessingResult:
        if rebuild:
            self.vector_store.reset()
        raw_documents = []
        for file_path in file_paths:
            raw_documents.extend(self.loader.load(file_path))
        cleaned_documents = self.cleaner.clean_documents(raw_documents)
        chunks = self.splitter.split_documents(cleaned_documents)
        self.vector_store.build(chunks)
        self.bm25_retriever.fit(chunks)
        return ProcessingResult(documents=cleaned_documents, chunks=chunks, citations=[], answer="")

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[Document], list[Any]]:
        rewritten_queries = self.query_rewriter.rewrite(query)
        all_docs: list[Document] = []
        for rewritten_query in rewritten_queries:
            all_docs.extend(self.vector_store.similarity_search(rewritten_query, k=k or self.settings.top_k))
        unique_docs: list[Document] = []
        seen: set[str] = set()
        for doc in all_docs:
            key = doc.metadata.get("source", "") + str(doc.metadata.get("page_number", "")) + doc.page_content
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        reranked = self.reranker.rerank(unique_docs, query)
        compressed = self.compressor.compress(reranked)
        citations = self.citation_builder.build(compressed)
        return compressed, citations

    def answer(self, query: str) -> ProcessingResult:
        documents, citations = self.retrieve(query)
        if not documents:
            answer = "I couldn't find sufficient evidence in the uploaded medical documents."
        else:
            answer = "Based on the retrieved medical evidence, the information below summarizes the available content."
        return ProcessingResult(documents=documents, chunks=documents, citations=citations, answer=answer)
