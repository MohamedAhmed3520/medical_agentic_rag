from __future__ import annotations

try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentSplitter:
    """Split long documents into smaller chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            page_content = document.page_content or ""
            if not page_content:
                continue
            split_chunks = self.splitter.split_documents([document])
            for chunk in split_chunks:
                metadata = chunk.metadata.copy()
                metadata.setdefault("chunk_id", len(chunks))
                chunk.metadata = metadata
                chunks.append(chunk)
        return chunks
