from __future__ import annotations

import requests

from config import get_settings


class MedicalGenerator:

    def __init__(self):

        self.settings = get_settings()

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        if not context.strip():

            return (
                "I couldn't find enough information "
                "in the uploaded documents."
            )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert medical assistant. "
                        "Answer ONLY using the provided medical context. "
                        "Do not hallucinate. "
                        "If the answer is not in the context, clearly say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Medical Context:\n\n{context}\n\n"
                        f"Question:\n{question}"
                    ),
                },
            ],
        }

        response = requests.post(
            f"{self.settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()
