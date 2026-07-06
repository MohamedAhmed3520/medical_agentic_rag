from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Task

from agents.crew_agents import create_agent_registry
from utils.logging import setup_logging
from utils.prompt_loader import load_prompt

logger = setup_logging("medical_agentic_rag.crew")


class MedicalCrew:
    """CrewAI-based orchestration for the medical workflow."""

    def __init__(self) -> None:
        self.agent_registry = create_agent_registry()
        self.agents = self._build_agents()
        self.tasks = self._build_tasks()

    def _build_agents(self) -> list[Agent]:
        coordinator_prompt = load_prompt("coordinator_prompt")
        generator_prompt = load_prompt("generator_prompt")
        validator_prompt = load_prompt("validation_prompt")

        return [
            Agent(
                role="Coordinator",
                goal="Coordinate the workflow",
                backstory=coordinator_prompt,
                verbose=False,
            ),
            Agent(
                role="Answer Generator",
                goal="Generate the final evidence-based answer",
                backstory=generator_prompt,
                verbose=False,
            ),
            Agent(
                role="Medical Validator",
                goal="Validate medical evidence",
                backstory=validator_prompt,
                verbose=False,
            ),
        ]

    def _build_tasks(self) -> list[Task]:
        return [
            Task(
                description="""
Coordinate the medical workflow.

User Question:
{query}

Retrieved Context:
{context}

Your job is to ensure the retrieved evidence is sufficient
before sending it to the Answer Generator.
""",
                expected_output="A structured retrieval and reasoning plan.",
                agent=self.agents[0],
            ),
            Task(
                description="""
You are an evidence-based medical assistant.

Question:
{query}

Retrieved Context:
{context}

Instructions:

- Answer ONLY using the retrieved context.
- Never invent medical facts.
- If the answer is not contained in the retrieved context,
  explicitly state that.
- Include citations when possible.
- End with a medical disclaimer.
""",
                expected_output="A complete evidence-based medical answer.",
                agent=self.agents[1],
            ),
            Task(
                description="""
Review the generated answer.

Question:
{query}

Retrieved Context:
{context}

Verify:

- Medical accuracy
- Evidence support
- Missing citations
- Hallucinations

If unsupported claims exist, remove them.
""",
                expected_output="A validated evidence-based response.",
                agent=self.agents[2],
            ),
        ]

    def run(self, query: str, retrieved_docs: list) -> dict[str, Any]:
        logger.info("Running CrewAI workflow")

        context = "\n\n".join(
            f"""Source: {doc.source}
Page: {doc.page}

{doc.text}
"""
            for doc in retrieved_docs
        )

        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )

        result = crew.kickoff(
            inputs={
                "query": query,
                "context": context,
            }
        )

        return {
            "query": query,
            "answer": str(result),
            "retrieved_docs": retrieved_docs,
        }
