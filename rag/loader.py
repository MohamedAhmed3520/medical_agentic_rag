from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.loader")


class PDFDocumentLoader:
    """Load PDF documents from disk."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else Path(__file__).resolve().parent.parent / "data"
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        logger.info("Loading PDF: %s", path)
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        for doc in documents:
            doc.metadata.setdefault("source", str(path.name))
            doc.metadata.setdefault("file_name", str(path.name))
            doc.metadata.setdefault("page_number", 1)
        return documents

    def load_directory(self, directory: str | Path | None = None) -> list[Document]:
        target_dir = Path(directory) if directory else self.directory
        pdf_files = sorted(target_dir.glob("*.pdf"))
        documents: list[Document] = []
        for pdf_path in pdf_files:
            documents.extend(self.load(pdf_path))
        return documents
