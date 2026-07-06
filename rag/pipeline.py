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
from rag.generator import MedicalGenerator
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
    """
    End-to-end Medical RAG pipeline.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
    ) -> None:

        self.settings = get_settings()

        self.data_dir = Path(
            data_dir or self.settings.data_dir
        )

        self.loader = PDFDocumentLoader(
            self.data_dir
        )

        self.cleaner = TextCleaner()

        self.splitter = DocumentSplitter(
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        self.embedding_model = EmbeddingModel(
            self.settings.embedding_model
        )

        self.vector_store = FAISSVectorStore(
            self.embedding_model,
            self.settings.vector_store_path,
        )

        self.bm25_retriever = BM25Retriever()

        self.reranker = Reranker(
            self.settings.reranker_model
        )

        self.compressor = ContextCompressor()

        self.citation_builder = CitationBuilder()

        self.query_rewriter = QueryRewriter()

        # NEW
        self.generator = MedicalGenerator()

    ####################################################################
    # INGESTION
    ####################################################################

    def ingest_documents(
        self,
        file_paths: list[str | Path],
        rebuild: bool = False,
    ) -> ProcessingResult:

        if rebuild:
            self.vector_store.reset()

        raw_documents: list[Document] = []

        for file_path in file_paths:
            raw_documents.extend(
                self.loader.load(file_path)
            )

        cleaned_documents = (
            self.cleaner.clean_documents(
                raw_documents
            )
        )

        chunks = self.splitter.split_documents(
            cleaned_documents
        )

        self.vector_store.build(chunks)

        self.bm25_retriever.fit(chunks)

        logger.info(
            "Indexed %d chunks.",
            len(chunks),
        )

        return ProcessingResult(
            documents=cleaned_documents,
            chunks=chunks,
            citations=[],
            answer="",
        )

    ####################################################################
    # RETRIEVAL
    ####################################################################

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> tuple[list[Document], list[Any]]:

        rewritten_queries = (
            self.query_rewriter.rewrite(query)
        )

        all_documents: list[Document] = []

        for rewritten_query in rewritten_queries:

            docs = self.vector_store.similarity_search(
                rewritten_query,
                k=k or self.settings.top_k,
            )

            all_documents.extend(docs)

        unique_documents: list[Document] = []

        seen: set[str] = set()

        for doc in all_documents:

            key = (
                doc.metadata.get("source", "")
                + str(
                    doc.metadata.get(
                        "page_number",
                        "",
                    )
                )
                + doc.page_content
            )

            if key in seen:
                continue

            seen.add(key)

            unique_documents.append(doc)

        reranked_documents = (
            self.reranker.rerank(
                unique_documents,
                query,
            )
        )

        compressed_documents = (
            self.compressor.compress(
                reranked_documents
            )
        )

        citations = (
            self.citation_builder.build(
                compressed_documents
            )
        )

        logger.info(
            "Retrieved %d documents.",
            len(compressed_documents),
        )

        return (
            compressed_documents,
            citations,
        )

    ####################################################################
    # GENERATOR
    ####################################################################

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:

        return self.generator.generate(
            question=question,
            context=context,
        )

    ####################################################################
    # SIMPLE PIPELINE API
    ####################################################################

    def answer(
        self,
        query: str,
    ) -> ProcessingResult:

        documents, citations = self.retrieve(
            query
        )

        if not documents:

            return ProcessingResult(
                documents=[],
                chunks=[],
                citations=[],
                answer=(
                    "I couldn't find sufficient "
                    "evidence in the uploaded "
                    "medical documents."
                ),
            )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        answer = self.generate_answer(
            question=query,
            context=context,
        )

        return ProcessingResult(
            documents=documents,
            chunks=documents,
            citations=citations,
            answer=answer,
        )
