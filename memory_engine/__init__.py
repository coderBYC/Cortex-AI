"""Local-first memory engine for AI agents (Phase 1)."""

from memory_engine.db import MemoryDB
from memory_engine.extraction import (
    JARO_WINKLER_THRESHOLD,
    LlamaCppBackend,
    LocalEmbedder,
    MemoryExtractor,
    MemoryStack,
    MockLLM,
    create_llm,
    get_local_embedder,
    require_local_embedder,
    resolve_model_path,
)
from memory_engine.retrieval import HybridRetriever

__all__ = [
    "MemoryDB",
    "MemoryExtractor",
    "MemoryStack",
    "MockLLM",
    "LlamaCppBackend",
    "LocalEmbedder",
    "HybridRetriever",
    "JARO_WINKLER_THRESHOLD",
    "create_llm",
    "get_local_embedder",
    "require_local_embedder",
    "resolve_model_path",
]

__version__ = "0.1.0"
