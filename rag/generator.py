from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings


class MedicalGenerator:
    """Generate grounded medical answers using OpenRouter."""

    def __init__(self) -> None:
        settings = get_settings()

        self.llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        if not context.strip():
            return (
                "I couldn't find enough information "
                "in the uploaded medical documents."
            )

        messages = [
            SystemMessage(
                content="""
You are an expert Medical AI Assistant.

Rules:

1. Answer ONLY from the supplied context.
2. Never invent medical facts.
3. If the answer is not contained in the context,
   clearly say so.
4. Be concise.
5. Use bullet points whenever appropriate.
6. Explain medical terms in simple language.
7. Mention when the evidence comes from the uploaded
   medical documents.
"""
            ),
            HumanMessage(
                content=f"""
Medical Context

{context}

-------------------------

Question

{question}
"""
            ),
        ]

        response = self.llm.invoke(messages)

        return response.content.strip()