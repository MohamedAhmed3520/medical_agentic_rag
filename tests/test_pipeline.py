from __future__ import annotations

from langchain_core.documents import Document

from medical_agentic_rag.rag.citation import CitationBuilder
from medical_agentic_rag.rag.pipeline import RAGPipeline


def test_pipeline_init() -> None:
    pipeline = RAGPipeline(data_dir=".")
    assert pipeline is not None


def test_citation_builder_dedupes_repeated_source_and_page() -> None:
    builder = CitationBuilder()
    documents = [
        Document(page_content="first evidence", metadata={"file_name": "meningioma_001.pdf", "page_number": 1}),
        Document(page_content="duplicate evidence", metadata={"file_name": "meningioma_001.pdf", "page_number": 1}),
        Document(page_content="another page", metadata={"file_name": "meningioma_001.pdf", "page_number": 2}),
    ]

    citations = builder.build(documents)

    assert [(citation.source, citation.page) for citation in citations] == [
        ("meningioma_001.pdf", 1),
        ("meningioma_001.pdf", 2),
    ]
