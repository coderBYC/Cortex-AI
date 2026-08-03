"""Hybrid BM25 + vector KNN retrieval with RRF and exponential time-decay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np

from memory_engine.db import MemoryDB
from memory_engine.extraction import hashing_embed, get_local_embedder

# Spec defaults (k=15 sharpens top-rank weight vs classic k=60)
RRF_K = 15
DEFAULT_LAMBDA = 0.00001  # decay per hour; permanent prefs use λ = 0


def _parse_iso(ts: str) -> datetime:
    # Accept both "...Z" and offset-aware ISO strings.
    cleaned = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since(created_at: str, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    created = _parse_iso(created_at)
    return max(0.0, (now - created).total_seconds() / 3600.0)


def time_decay(hours: float, lam: float) -> float:
    """e^{-λ · Δt_hours}. λ = 0 ⇒ no decay (permanent preference)."""
    if lam <= 0.0:
        return 1.0
    return math.exp(-lam * hours)


def rrf_score(rank_vec: Optional[float], rank_bm25: Optional[float], k: int = RRF_K) -> float:
    """
    Reciprocal Rank Fusion over the two channels.
    Missing ranks contribute 0 (document not retrieved by that channel).

    Lower k (15 vs 60) increases score gap between rank-1 and long-tail hits,
    so noisy lower ranks contribute less to the fused ranking / prompt.
    """
    score = 0.0
    if rank_vec is not None:
        score += 1.0 / (k + rank_vec)
    if rank_bm25 is not None:
        score += 1.0 / (k + rank_bm25)
    return score


@dataclass
class ScoredMemory:
    id: int
    entity: str
    attribute: str
    value: str
    confidence: float
    created_at: str
    is_permanent: bool
    rank_vec: Optional[float]
    rank_bm25: Optional[float]
    rrf: float
    decay: float
    final_score: float
    parent_context: Optional[str] = None  # parent session snippet for prompts

    def prompt_text(self) -> str:
        """Context shown to the answer model (parent snippet preferred)."""
        if self.parent_context and self.parent_context.strip():
            return self.parent_context.strip()
        return f"{self.entity}'s {self.attribute} is {self.value}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "is_permanent": self.is_permanent,
            "rank_vec": self.rank_vec,
            "rank_bm25": self.rank_bm25,
            "rrf": self.rrf,
            "decay": self.decay,
            "final_score": self.final_score,
            "parent_context": self.parent_context,
            "fact": f"{self.entity}'s {self.attribute} is {self.value}",
            "prompt_text": self.prompt_text(),
        }


class HybridRetriever:
    """
    Parallel BM25 (FTS5) + vector KNN → RRF → exponential time-decay → top-K.

    Final Score = (1/(15 + Rank_vec) + 1/(15 + Rank_bm25)) × e^{-λ (t_now − t_created)}

    Child facts are indexed for retrieval; parent_context (session snippet) is
    expanded into the answer prompt when present.
    """

    def __init__(
        self,
        db: MemoryDB,
        *,
        lam: float = DEFAULT_LAMBDA,
        rrf_k: int = RRF_K,
        embed_fn: Optional[Any] = None,
        require_fastembed: bool = True,
    ) -> None:
        self.db = db
        self.lam = lam
        self.rrf_k = rrf_k
        if embed_fn is not None:
            self.embed_fn = embed_fn
        elif require_fastembed:
            embedder = get_local_embedder()
            self.embed_fn = lambda text: embedder.embed(text, db.dim)
        else:
            # Explicit opt-out only (tests). Production path uses fastembed.
            self.embed_fn = lambda text: hashing_embed(text, db.dim)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_limit: int = 20,
        now: Optional[datetime] = None,
        query_embedding: Optional[Sequence[float] | np.ndarray] = None,
    ) -> list[ScoredMemory]:
        now = now or datetime.now(timezone.utc)
        emb = (
            np.asarray(query_embedding, dtype=np.float32)
            if query_embedding is not None
            else np.asarray(self.embed_fn(query), dtype=np.float32)
        )

        bm25_hits = self.db.bm25_search(query, limit=candidate_limit)
        vec_hits = self.db.vector_search(emb, limit=candidate_limit)

        bm25_rank = {mid: rank for mid, rank in bm25_hits}
        vec_rank = {mid: rank for mid, rank in vec_hits}
        candidate_ids = set(bm25_rank) | set(vec_rank)

        scored: list[ScoredMemory] = []
        for mid in candidate_ids:
            row = self.db.get_memory(mid)
            if row is None or row["invalidated_at"] is not None:
                continue

            rv = vec_rank.get(mid)
            rb = bm25_rank.get(mid)
            rrf = rrf_score(rv, rb, self.rrf_k)

            is_permanent = bool(row["is_permanent"])
            lam = 0.0 if is_permanent else self.lam
            decay = time_decay(hours_since(row["created_at"], now), lam)
            final = rrf * decay

            parent = row["parent_context"] if "parent_context" in row.keys() else None
            scored.append(
                ScoredMemory(
                    id=mid,
                    entity=row["entity"],
                    attribute=row["attribute"],
                    value=row["value"],
                    confidence=float(row["confidence"]),
                    created_at=row["created_at"],
                    is_permanent=is_permanent,
                    rank_vec=rv,
                    rank_bm25=rb,
                    rrf=rrf,
                    decay=decay,
                    final_score=final,
                    parent_context=parent,
                )
            )

        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored[:top_k]
