from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphState:

    user_query: str = ""
    rewritten_query: str = ""

    intent: str = "medical"
    should_retrieve: bool = True

    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    reranked_documents: list[dict[str, Any]] = field(default_factory=list)

    medical_context: str = ""

    reflection: str = ""

    validation: str = ""

    citations: list[dict[str, Any]] = field(default_factory=list)

    citation_text: str = ""

    answer: str = ""

    final_response: str = ""

    error: str | None = None
