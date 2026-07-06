from __future__ import annotations

from .base import BaseMemory


class VectorMemory(BaseMemory):
    """Vector-style memory store placeholder."""

    def add(self, role: str, content: str) -> None:
        super().add(role, content)
