from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm.openrouter import OpenRouterClient
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.agents")


@dataclass(slots=True)
class AgentConfig:
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[Any] = field(default_factory=list)


class BaseAgent:
    """Base class for CrewAI-style agents."""

    def __init__(self, config: AgentConfig, llm_client: OpenRouterClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client or OpenRouterClient()

    def run(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        choices = response.get("choices", [])
        if not choices:
            return "No response generated."
        return choices[0].get("message", {}).get("content", "")
