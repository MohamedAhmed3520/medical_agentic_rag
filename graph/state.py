from dataclasses import dataclass, field

@dataclass(slots=False)
class GraphState:
    user_query: str = ""
    rewritten_query: str = ""
    intent: str = ""
    should_retrieve: bool = True
    retrieved_documents: list = field(default_factory=list)
    reranked_documents: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    medical_context: str = ""
    reflection: str = ""
    validation: str = ""
    answer: str = ""
    final_response: str = ""
