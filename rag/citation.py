from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(slots=True)
class Citation:
    source: str
    page: int | None
    snippet: str


class CitationBuilder:
    """Create citations from retrieved evidence."""

    def build(self, documents: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int]] = set()
        for document in documents:
            metadata = document.metadata or {}
            source = str(metadata.get("file_name") or metadata.get("source") or "unknown")
            page_value = metadata.get("page") or metadata.get("page_number") or 1
            page = int(page_value) + 1 if metadata.get("page") is not None and metadata.get("page_number") is None else int(page_value or 1)
            snippet = (document.page_content or "").strip()
            if not snippet:
                continue
            key = (source.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    source=source,
                    page=page,
                    snippet=snippet[:180],
                )
            )
        return citations
