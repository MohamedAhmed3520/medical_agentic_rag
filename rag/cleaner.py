from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document


class TextCleaner:
    """Clean and normalize document text."""

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text.strip()

    def clean_documents(self, documents: Iterable[Document]) -> list[Document]:
        cleaned: list[Document] = []
        for document in documents:
            cleaned_text = self.clean_text(document.page_content or "")
            if not cleaned_text:
                continue
            cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata.copy())
            cleaned.append(cleaned_document)
        return cleaned
