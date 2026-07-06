from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Application configuration loaded from environment variables and .env."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "logs")
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "prompts")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "models")
    vector_store_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache" / "faiss.index")
    metadata_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache" / "metadata.json")
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-large"
    temperature: float = 0.1
    max_tokens: int = 700
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    hybrid_alpha: float = 0.6
    enable_async: bool = True

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    settings = Settings(
        project_root=Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent)),
        data_dir=Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "data")),
        cache_dir=Path(os.getenv("CACHE_DIR", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache")),
        logs_dir=Path(os.getenv("LOGS_DIR", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "logs")),
        prompts_dir=Path(os.getenv("PROMPTS_DIR", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "prompts")),
        models_dir=Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "models")),
        vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache" / "faiss.index")),
        metadata_path=Path(os.getenv("METADATA_PATH", Path(__file__).resolve().parent.parent / "medical_agentic_rag" / "cache" / "metadata.json")),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MAX_TOKENS", "700")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "8")),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.6")),
        enable_async=os.getenv("ENABLE_ASYNC", "true").lower() == "true",
    )
    settings.project_root.mkdir(parents=True, exist_ok=True)
    return settings
