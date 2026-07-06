from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CrewTask:
    """Simple task representation for a CrewAI-style workflow."""

    name: str
    description: str
    expected_output: str


def build_tasks() -> list[CrewTask]:
    return [
        CrewTask(
            name="retrieve_evidence",
            description="Retrieve relevant evidence from the medical knowledge base.",
            expected_output="A ranked list of evidence chunks.",
        ),
        CrewTask(
            name="validate_evidence",
            description="Validate that the retrieved evidence is medically relevant and safe.",
            expected_output="A validation summary.",
        ),
        CrewTask(
            name="generate_answer",
            description="Generate a final evidence-based answer with citations.",
            expected_output="A final response with citations and disclaimer.",
        ),
    ]
