from __future__ import annotations

from typing import Any


class Evaluator:
    """Lightweight evaluation helper placeholder for RAG metrics."""

    def evaluate(self, query: str, answer: str, contexts: list[str]) -> dict[str, float]:
        faithfulness = 1.0 if contexts else 0.0
        relevancy = 1.0 if answer and query else 0.0
        context_precision = 1.0 if contexts else 0.0
        context_recall = 1.0 if contexts else 0.0
        hallucination = 0.0 if contexts else 1.0
        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "hallucination_score": hallucination,
        }
