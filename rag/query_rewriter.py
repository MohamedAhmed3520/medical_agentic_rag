from __future__ import annotations

from typing import List


class QueryRewriter:
    """Rewrite user queries to improve retrieval."""

    def rewrite(self, query: str) -> list[str]:
        cleaned = query.strip()
        if not cleaned:
            return []
        return [cleaned, f"{cleaned} evidence medical documents", f"{cleaned} clinical context"]
