"""Local-first knowledge-layer memory engine (Cortex-AI)."""

from memory_engine.cortex import AskResult, CortexEngine, IngestResult
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
from memory_engine.knowledge_db import KnowledgeDB
from memory_engine.retrieval import HybridRetriever

__all__ = [
    "CortexEngine",
    "IngestResult",
    "AskResult",
    "KnowledgeDB",
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

__version__ = "0.2.0"
