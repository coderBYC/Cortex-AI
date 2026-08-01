"""Local-first memory engine for AI agents (Phase 1)."""

from memory_engine.db import MemoryDB
from memory_engine.extraction import (
    JARO_WINKLER_THRESHOLD,
    LlamaCppBackend,
    LocalEmbedder,
    MemoryExtractor,
    MockLLM,
    create_llm,
    get_local_embedder,
)
from memory_engine.retrieval import HybridRetriever

__all__ = [
    "MemoryDB",
    "MemoryExtractor",
    "MockLLM",
    "LlamaCppBackend",
    "LocalEmbedder",
    "HybridRetriever",
    "JARO_WINKLER_THRESHOLD",
    "create_llm",
    "get_local_embedder",
]

__version__ = "0.1.0"
