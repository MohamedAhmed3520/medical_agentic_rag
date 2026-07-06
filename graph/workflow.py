from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from rag.pipeline import RAGPipeline
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.graph")


class MedicalRAGGraph:
    """LangGraph workflow for the medical RAG pipeline."""

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
        "good afternoon",
    }

    THANKS = {
        "thanks",
        "thank you",
        "thx",
    }

    GOODBYES = {
        "bye",
        "goodbye",
        "see you",
    }

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("router", self._router)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("hybrid_search", self._hybrid_search)
        workflow.add_node("reranker", self._reranker)
        workflow.add_node("medical_research", self._medical_research)
        workflow.add_node("reflection", self._reflection)
        workflow.add_node("validation", self._validation)
        workflow.add_node("citation", self._citation)
        workflow.add_node("generator", self._generator)
        workflow.add_node("final_answer", self._final_answer)

        workflow.set_entry_point("router")

        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "hybrid_search")
        workflow.add_edge("hybrid_search", "reranker")
        workflow.add_edge("reranker", "medical_research")
        workflow.add_edge("medical_research", "reflection")
        workflow.add_edge("reflection", "validation")
        workflow.add_edge("validation", "citation")
        workflow.add_edge("citation", "generator")
        workflow.add_edge("generator", "final_answer")
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    def _router(self, state: GraphState) -> GraphState:
        """
        Detect the user intent before running retrieval.
        """

        query = state.user_query.strip()
        lowered = query.lower()

        state.rewritten_query = query

        if lowered in self.GREETINGS:
            state.intent = "greeting"
            state.should_retrieve = False

        elif lowered in self.THANKS:
            state.intent = "thanks"
            state.should_retrieve = False

        elif lowered in self.GOODBYES:
            state.intent = "goodbye"
            state.should_retrieve = False

        else:
            state.intent = "medical"
            state.should_retrieve = True

        logger.info(
            "Intent=%s | Retrieve=%s",
            state.intent,
            state.should_retrieve,
        )

        return state
    def _retriever(self, state: GraphState) -> GraphState:
        if not state.should_retrieve:
            logger.info("Skipping retrieval.")
                return state
    
            documents, citations = self.pipeline.retrieve(
                state.rewritten_query
            )
    
            if not documents:
                logger.warning("No documents retrieved.")
    
                state.retrieved_documents = []
                state.citations = []
    
                return state
    
            state.retrieved_documents = [
                doc.metadata.copy() | {
                    "content": doc.page_content
                }
                for doc in documents
            ]
    
            state.citations = [
                {
                    "source": c.source,
                    "page": c.page,
                    "snippet": c.snippet,
                }
                for c in citations
            ]
    
            logger.info(
                "Retrieved %d documents.",
                len(state.retrieved_documents),
            )
    
        return state

    def _hybrid_search(self, state: GraphState) -> GraphState:
        """
        Placeholder.

        Hybrid retrieval will later merge
        dense search + BM25.
        """
        return state

    def _reranker(self, state: GraphState) -> GraphState:

        state.reranked_documents = (
            state.retrieved_documents
        )

        return state

    def _medical_research(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Build the medical context.
        """

        docs = (
            state.reranked_documents
            or state.retrieved_documents
        )

        state.medical_context = "\n\n".join(
            doc["content"]
            for doc in docs[:5]
        )

        return state

    def _reflection(
        self,
        state: GraphState,
    ) -> GraphState:

        state.reflection = (
            "Evidence reviewed successfully."
        )

        return state

    def _validation(
        self,
        state: GraphState,
    ) -> GraphState:

        if state.retrieved_documents:

            state.validation = (
                "Evidence found."
            )

        else:

            state.validation = (
                "No supporting evidence."
            )

        return state

    def _citation(
        self,
        state: GraphState,
    ) -> GraphState:

        return state

    def _generator(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Generate the final answer.
        """

        if state.intent == "greeting":

            state.answer = (
                "Hello! 👋 How can I help you today?"
            )

            return state

        if state.intent == "thanks":

            state.answer = (
                "You're welcome! 😊"
            )

            return state

        if state.intent == "goodbye":

            state.answer = (
                "Goodbye! Take care."
            )

            return state

        if not state.retrieved_documents:

            state.answer = (
                "I couldn't find relevant medical "
                "evidence in the uploaded documents."
            )

            return state

        # Temporary answer.
        # Will be replaced by the LLM generator.

        state.answer = (
            "Based on the retrieved medical evidence, "
            "the relevant information has been gathered. "
            "The next step is to let the language model "
            "generate the final response using this context."
        )

        return state

    def _final_answer(
        self,
        state: GraphState,
    ) -> GraphState:

        state.final_response = state.answer

        return state

    def run(
        self,
        query: str,
    ) -> GraphState:

        initial_state = GraphState(
            user_query=query
        )

        return self.graph.invoke(
            initial_state
        )
