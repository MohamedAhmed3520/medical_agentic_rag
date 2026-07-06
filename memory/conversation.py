from __future__ import annotations

from .base import BaseMemory


class ConversationBufferMemory(BaseMemory):
    """Keeps the full conversation history."""


class ConversationSummaryMemory(BaseMemory):
    """Keeps a compact summary of the conversation."""


class ConversationWindowMemory(BaseMemory):
    """Keeps a recent window of conversation turns."""

    def __init__(self, window_size: int = 6) -> None:
        super().__init__()
        self.window_size = window_size

    def add(self, role: str, content: str) -> None:
        super().add(role, content)
        if len(self.entries) > self.window_size:
            self.entries = self.entries[-self.window_size :]
