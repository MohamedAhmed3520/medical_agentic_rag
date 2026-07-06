from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from rag.pipeline import RAGPipeline
from utils.logging import setup_logging

logger = setup_logging("medical_agentic_rag.graph")


class MedicalRAGGraph:
    """
    LangGraph workflow for the Medical Agentic RAG system.
    """

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
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

    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
    ) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.graph = self._build_graph()

    ###########################################################
    # GRAPH
    ###########################################################

    def _build_graph(self):
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

    ###########################################################
    # ROUTER
    ###########################################################

    def _router(
        self,
        state: GraphState,
    ) -> GraphState:
        query = state.user_query.strip()
        lowered = query.lower()

        state.rewritten_query = query

        # Greeting
        if any(word in lowered for word in self.GREETINGS):
            state.intent = "greeting"
            state.should_retrieve = False

        # Thanks
        elif any(word in lowered for word in self.THANKS):
            state.intent = "thanks"
            state.should_retrieve = False

        # Goodbye
        elif any(word in lowered for word in self.GOODBYES):
            state.intent = "goodbye"
            state.should_retrieve = False

        # Medical
        else:
            state.intent = "medical"
            state.should_retrieve = True

        logger.info(
            "Intent=%s | Retrieve=%s",
            state.intent,
            state.should_retrieve,
        )

        return state

    ###########################################################
    # RETRIEVER
    ###########################################################

    def _retriever(
        self,
        state: GraphState,
    ) -> GraphState:
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
            doc.metadata.copy()
            | {
                "content": doc.page_content,
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

    ###########################################################
    # HYBRID SEARCH
    ###########################################################

    def _hybrid_search(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Placeholder for future FAISS + BM25 hybrid retrieval.
        """
        return state

    ###########################################################
    # RERANKER
    ###########################################################

    def _reranker(
        self,
        state: GraphState,
    ) -> GraphState:
        state.reranked_documents = state.retrieved_documents
        return state

    ###########################################################
    # MEDICAL RESEARCH
    ###########################################################

    def _medical_research(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Build a rich medical context for the LLM.
        """

        docs = (
            state.reranked_documents
            or state.retrieved_documents
        )

        if not docs:
            state.medical_context = ""
            return state

        context_parts = []

        for doc in docs[:5]:
            source = doc.get(
                "source",
                "Unknown Source",
            )

            page = (
                doc.get("page")
                or doc.get("page_number")
                or "Unknown"
            )

            content = doc.get(
                "content",
                "",
            )

            context_parts.append(
                f"""Source:
{source}

Page:
{page}

Content:
{content}"""
            )

        state.medical_context = "\n\n".join(context_parts)

        logger.info(
            "Medical context built with %d documents.",
            len(docs[:5]),
        )

        return state

    ###########################################################
    # REFLECTION
    ###########################################################

    def _reflection(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Placeholder for self-reflection.
        Later this can ask the LLM if more retrieval is needed.
        """

        if state.retrieved_documents:
            state.reflection = (
                "Sufficient evidence appears available."
            )
        else:
            state.reflection = "No evidence retrieved."

        logger.info(state.reflection)

        return state

    ###########################################################
    # VALIDATION
    ###########################################################

    def _validation(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Validate whether evidence exists before generation.
        """

        if state.retrieved_documents:
            state.validation = "Evidence validated."
        else:
            state.validation = "No supporting evidence."

        logger.info(state.validation)

        return state

    ###########################################################
    # CITATION
    ###########################################################

    def _citation(
        self,
        state: GraphState,
    ) -> GraphState:
        if not state.citations:
            state.citation_text = ""
            return state

        references = []

        for index, citation in enumerate(
            state.citations,
            start=1,
        ):
            references.append(
                f"[{index}] "
                f"{citation['source']} "
                f"(Page {citation['page']})"
            )

        state.citation_text = "\n".join(references)

        logger.info(
            "Prepared %d citations.",
            len(state.citations),
        )

        return state

    ###########################################################
    # GENERATOR
    ###########################################################

    def _generator(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Generate the final answer using OpenRouter.
        """

        # Greeting
        if state.intent == "greeting":
            state.answer = (
                "Hello! 👋 I'm your Medical AI assistant.\n\n"
                "Ask me anything about the uploaded medical documents."
            )
            return state

        # Thanks
        if state.intent == "thanks":
            state.answer = (
                "You're welcome! 😊\n\n"
                "If you have another medical question, feel free to ask."
            )
            return state

        # Goodbye
        if state.intent == "goodbye":
            state.answer = (
                "Goodbye! 👋\n"
                "Take care and stay healthy."
            )
            return state

        # No evidence
        if not state.retrieved_documents:
            state.answer = (
                "I couldn't find relevant information "
                "in the uploaded medical documents."
            )
            return state

        logger.info("Generating answer using OpenRouter...")

        state.answer = self.pipeline.generate_answer(
            question=state.user_query,
            context=state.medical_context,
        )

        return state

    ###########################################################
    # FINAL ANSWER
    ###########################################################

    def _final_answer(
        self,
        state: GraphState,
    ) -> GraphState:
        if state.citation_text:
            state.final_response = (
                f"{state.answer}\n\n"
                f"### Sources\n\n"
                f"{state.citation_text}"
            )
        else:
            state.final_response = state.answer

        logger.info("Final answer generated.")

        return state

    ###########################################################
    # RUN
    ###########################################################

    def run(
        self,
        query: str,
    ) -> GraphState:
        logger.info(
            "User Query: %s",
            query,
        )

        initial_state = GraphState(
            user_query=query,
        )

        return self.graph.invoke(initial_state)
