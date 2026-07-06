from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphState:
    """State container for the LangGraph workflow."""

    # User Input
    user_query: str = ""
    rewritten_query: str = ""

    # Router
    intent: str = "medical"
    should_retrieve: bool = True

    # Retrieval
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    reranked_documents: list[dict[str, Any]] = field(default_factory=list)

    # Research Context
    medical_context: str = ""

    # Reflection
    reflection: str = ""

    # Validation
    validation: str = ""

    # Citations
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_text: str = ""

    # Final Response
    answer: str = ""
    final_response: str = ""

    # Error
    error: str | None = None
