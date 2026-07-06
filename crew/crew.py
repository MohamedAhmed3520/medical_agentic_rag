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
                description="Coordinate retrieval and evidence validation for the user query.",
                expected_output="A structured plan for evidence retrieval.",
                agent=self.agents[0],
            ),
            Task(
                description="Generate the final answer with citations and disclaimer.",
                expected_output="A polished evidence-based answer.",
                agent=self.agents[1],
            ),
            Task(
                description="Validate the medical accuracy and evidence support of the answer.",
                expected_output="A validation note.",
                agent=self.agents[2],
            ),
        ]

    def run(self, query: str) -> dict[str, Any]:
        logger.info("Running CrewAI workflow for query: %s", query)
        crew = Crew(agents=self.agents, tasks=self.tasks)
        result = crew.kickoff(inputs={"query": query})
        return {"query": query, "result": str(result), "status": "completed"}
