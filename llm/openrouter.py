from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterator

import httpx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.llm")


class OpenRouterClient:
    """OpenRouter-compatible wrapper using requests and optional async support."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openrouter_api_key or ""
        self.base_url = base_url or self.settings.openrouter_base_url
        self.model = model or self.settings.model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, messages: list[dict[str, Any]], *, temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> dict[str, Any]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke(self, messages: list[dict[str, Any]], *, temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> dict[str, Any]:
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, model=model)

    def stream(self, messages: list[dict[str, Any]], *, temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> Iterator[str]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "stream": True,
        }
        with requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=60, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")

    async def achat(self, messages: list[dict[str, Any]], *, temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> dict[str, Any]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def astream(self, messages: list[dict[str, Any]], *, temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> AsyncIterator[str]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
