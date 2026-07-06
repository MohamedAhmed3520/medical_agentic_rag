from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEntry:
    role: str
    content: str


class BaseMemory:
    """Base interface for conversation memory."""

    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(self, role: str, content: str) -> None:
        self.entries.append(MemoryEntry(role=role, content=content))

    def get_history(self) -> list[MemoryEntry]:
        return list(self.entries)
