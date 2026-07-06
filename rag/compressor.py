from __future__ import annotations

from langchain_core.documents import Document


class ContextCompressor:
    """Compress retrieved context into concise evidence chunks."""

    def compress(self, documents: list[Document]) -> list[Document]:
        compressed: list[Document] = []
        for document in documents:
            text = (document.page_content or "").strip()
            if len(text) > 700:
                text = text[:700] + "..."
            compressed.append(Document(page_content=text, metadata=document.metadata.copy()))
        return compressed
