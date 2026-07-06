from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PACKAGE_ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from config import get_settings
from graph.workflow import MedicalRAGGraph
from rag.pipeline import RAGPipeline


def _ensure_index(pipeline: RAGPipeline, settings) -> None:
    pdf_files = sorted(settings.data_dir.glob("*.pdf"))
    if not pdf_files:
        return
    if pipeline.vector_store.index is None or not pipeline.vector_store.documents:
        pipeline.ingest_documents(pdf_files, rebuild=True)


def main() -> None:
    st.set_page_config(page_title="Medical Agentic RAG", layout="wide")
    settings = get_settings()
    pipeline = RAGPipeline(settings.data_dir)
    workflow = MedicalRAGGraph(pipeline)
    _ensure_index(pipeline, settings)

    st.title("Medical Agentic RAG")
    st.markdown("Evidence-based medical assistance from uploaded PDFs.")

    with st.sidebar:
        st.header("Settings")
        uploaded_files = st.file_uploader("Upload medical PDFs", type=["pdf"], accept_multiple_files=True)
        rebuild = st.checkbox("Rebuild index", value=False)
        if st.button("Build / Rebuild Index"):
            files = []
            for uploaded in uploaded_files or []:
                target = settings.data_dir / uploaded.name
                target.write_bytes(uploaded.getvalue())
                files.append(target)
            if files:
                pipeline.ingest_documents(files, rebuild=rebuild)
                st.success("Index built successfully.")
            else:
                st.warning("Upload at least one PDF.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a medical question based on the uploaded documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            result = workflow.run(prompt)
            answer_text = result.get("final_response", "") or "I couldn't find sufficient evidence in the uploaded medical documents."
            st.markdown(answer_text)
            st.caption("Retrieved Evidence")
            if result.get("citations"):
                for citation in result["citations"]:
                    source = citation.get("source") or "unknown"
                    page = citation.get("page")
                    snippet = citation.get("snippet") or ""
                    if snippet:
                        st.write(f"- {source} (page {page})")
                        st.caption(snippet)
                    else:
                        st.write(f"- {source} (page {page})")
            st.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
    main()
