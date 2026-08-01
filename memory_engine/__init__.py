"""Local-first memory engine for AI agents (Phase 1)."""

from memory_engine.db import MemoryDB
from memory_engine.extraction import MemoryExtractor, MockLLM, JARO_WINKLER_THRESHOLD
from memory_engine.retrieval import HybridRetriever

__all__ = [
    "MemoryDB",
    "MemoryExtractor",
    "MockLLM",
    "HybridRetriever",
    "JARO_WINKLER_THRESHOLD",
]

__version__ = "0.1.0"
