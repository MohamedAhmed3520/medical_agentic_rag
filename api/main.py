from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from graph.workflow import MedicalRAGGraph
from rag.pipeline import RAGPipeline

app = FastAPI(title="Medical Agentic RAG API")

pipeline = RAGPipeline()
workflow = MedicalRAGGraph(pipeline)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, object]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    result = workflow.run(request.query)
    return QueryResponse(answer=result.get("final_response", ""), citations=result.get("citations", []))
