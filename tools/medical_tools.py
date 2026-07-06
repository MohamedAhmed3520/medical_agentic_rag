from __future__ import annotations

from typing import Any


class MedicalTools:
    """Utility tools exposed to agents and the MCP layer."""

    @staticmethod
    def calculator(expression: str) -> float:
        return eval(expression, {"__builtins__": {}}, {})  # nosec B307

    @staticmethod
    def summarize(text: str) -> str:
        return text[:500] if len(text) > 500 else text

    @staticmethod
    def metadata_lookup(metadata: dict[str, Any]) -> dict[str, Any]:
        return metadata
