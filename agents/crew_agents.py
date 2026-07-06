from __future__ import annotations

from .base import AgentConfig, BaseAgent


def create_agent_registry() -> dict[str, BaseAgent]:
    registry: dict[str, BaseAgent] = {}
    agent_specs = [
        ("coordinator", "Coordinator Agent", "Coordinate the medical RAG workflow.", "You coordinate retrieval, evidence validation, and answer generation."),
        ("retrieval", "Retrieval Agent", "Retrieve relevant evidence from the medical document store.", "You retrieve evidence carefully from the uploaded documents."),
        ("medical_research", "Medical Research Agent", "Research and explain the medical context.", "You explain medical concepts using only the retrieved evidence."),
        ("query_rewrite", "Query Rewrite Agent", "Rewrite user queries for better retrieval.", "You rewrite questions to improve retrieval quality."),
        ("reranker", "Reranker Agent", "Rerank retrieved evidence.", "You prioritize the most relevant evidence."),
        ("medical_validator", "Medical Validator Agent", "Validate evidence quality.", "You validate evidence for relevance and safety."),
        ("hallucination", "Hallucination Detection Agent", "Detect unsupported claims.", "You flag hallucinations and unsupported statements."),
        ("citation", "Citation Agent", "Create citations for evidence.", "You provide precise citations."),
        ("reflection", "Reflection Agent", "Reflect on the response quality.", "You improve the response based on evidence."),
        ("answer_generator", "Answer Generator Agent", "Generate the final answer.", "You generate an evidence-based answer."),
    ]
    for key, name, goal, backstory in agent_specs:
        registry[key] = BaseAgent(AgentConfig(name=key, role=name, goal=goal, backstory=backstory))
    return registry
