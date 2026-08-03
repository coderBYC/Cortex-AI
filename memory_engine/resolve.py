"""Entity resolution: alias + Jaro-Winkler + FTS + embedding fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import jellyfish

from memory_engine.extraction import LocalEmbedder
from memory_engine.knowledge_db import KnowledgeDB
from memory_engine.mentions import Mention

JW_THRESHOLD = 0.88
EMBED_THRESHOLD = 0.72
# Reject JW/FTS merges when normalized lengths differ a lot (apple vs AppleInc).
MIN_LENGTH_RATIO = 0.8


@dataclass
class ResolvedMention:
    mention: str
    type: str
    entity_id: str
    canonical_name: str
    score: float
    method: str  # exact|alias|norm|jw|fts|embedding|created

    def as_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "type": self.type,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "score": self.score,
            "method": self.method,
        }


def normalize_name(name: str) -> str:
    """Alphanumeric fold: 'Py-Torch' / 'Pytorch' → 'pytorch'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _jw(a: str, b: str) -> float:
    return float(jellyfish.jaro_winkler_similarity(a.lower(), b.lower()))


def _length_compatible(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    return (min(len(na), len(nb)) / max(len(na), len(nb))) >= MIN_LENGTH_RATIO


def name_similarity(a: str, b: str) -> float:
    """
    Similarity used for merge decisions.

    - Normalized exact match → 1.0 (handles PyTorch / py-torch / Pytorch)
    - Otherwise JW on normalized forms, but capped below threshold when
      length ratio is low (keeps apple ≠ AppleInc despite JW ~0.92)
    """
    na, nb = normalize_name(a), normalize_name(b)
    if na and na == nb:
        return 1.0
    score = float(jellyfish.jaro_winkler_similarity(na, nb)) if na and nb else 0.0
    if not _length_compatible(a, b):
        return min(score, JW_THRESHOLD - 0.01)
    return score


class EntityResolver:
    """
    Resolve surface mentions to canonical entities.

    Scoring blend (context-aware):
      1. exact / alias match
      2. normalized alias match (hyphen/case fold)
      3. Jaro-Winkler > 0.88 (with length-ratio guard)
      4. FTS / substring name match (length-safe)
      5. embedding cosine > 0.72
      else create a new entity
    """

    def __init__(self, db: KnowledgeDB, embedder: LocalEmbedder) -> None:
        self.db = db
        self.embedder = embedder

    def _pick_typed(self, hits: list[dict[str, Any]], mention_type: str) -> dict[str, Any]:
        typed = [h for h in hits if h["type"] == mention_type]
        return typed[0] if typed else hits[0]

    def resolve(
        self,
        mention: Mention,
        *,
        context: str = "",
    ) -> ResolvedMention:
        name = mention.mention.strip()
        if not name:
            raise ValueError("empty mention")

        # 1) Exact / alias
        hits = self.db.find_entities_by_alias(name)
        if hits:
            chosen = self._pick_typed(hits, mention.type)
            self.db.add_alias(chosen["id"], name)
            return ResolvedMention(
                name, mention.type, chosen["id"], chosen["canonical_name"], 1.0, "exact"
            )

        # 2) Normalized alias (PyTorch ≡ py-torch)
        norm = normalize_name(name)
        for eid, canon, alias, etype, _emb in self.db.all_entity_names():
            if normalize_name(alias) == norm and norm:
                if etype != mention.type:
                    # Allow if only hit; prefer type match by continuing scan
                    continue
                self.db.add_alias(eid, name)
                return ResolvedMention(name, mention.type, eid, canon, 1.0, "norm")
        # Second pass: accept cross-type only if unique norm match
        norm_hits = [
            (eid, canon, etype)
            for eid, canon, alias, etype, _emb in self.db.all_entity_names()
            if normalize_name(alias) == norm and norm
        ]
        if len({h[0] for h in norm_hits}) == 1:
            eid, canon, _ = norm_hits[0]
            self.db.add_alias(eid, name)
            return ResolvedMention(name, mention.type, eid, canon, 1.0, "norm")

        # 3) Jaro-Winkler over aliases (length-guarded)
        best: Optional[tuple[float, dict[str, Any], str]] = None
        for eid, canon, alias, etype, _emb in self.db.all_entity_names():
            score = name_similarity(name, alias)
            if etype == mention.type:
                score = min(1.0, score + 0.02)
            if context and canon.lower() in context.lower():
                score = min(1.0, score + 0.03)
            if best is None or score > best[0]:
                best = (score, {"id": eid, "canonical_name": canon, "type": etype}, "jw")
        if best and best[0] >= JW_THRESHOLD and _length_compatible(name, best[1]["canonical_name"]):
            self.db.add_alias(best[1]["id"], name)
            return ResolvedMention(
                name,
                mention.type,
                best[1]["id"],
                best[1]["canonical_name"],
                min(1.0, best[0]),
                "jw",
            )

        # 4) FTS / substring — only when length-compatible
        fts_hits = self.db.search_entities_fts(name, limit=5)
        fts_hits = [
            h
            for h in fts_hits
            if _length_compatible(name, h["canonical_name"])
            or normalize_name(name) == normalize_name(h["canonical_name"])
        ]
        if fts_hits:
            typed = [h for h in fts_hits if h["type"] == mention.type]
            chosen = typed[0] if typed else fts_hits[0]
            score = name_similarity(name, chosen["canonical_name"])
            if score >= 0.75:
                self.db.add_alias(chosen["id"], name)
                return ResolvedMention(
                    name,
                    mention.type,
                    chosen["id"],
                    chosen["canonical_name"],
                    max(score, 0.8),
                    "fts",
                )

        # 5) Embedding similarity (mention + light context)
        emb_text = f"{name} ({mention.type})"
        if context:
            emb_text = f"{name}. Context: {context[:200]}"
        qemb = self.embedder.embed(emb_text, self.db.dim)
        emb_hits = self.db.search_entities_embedding(qemb, limit=5)
        for ent, sim in emb_hits:
            if not _length_compatible(name, ent["canonical_name"]):
                continue
            if ent["type"] == mention.type and sim >= EMBED_THRESHOLD:
                self.db.add_alias(ent["id"], name)
                return ResolvedMention(
                    name,
                    mention.type,
                    ent["id"],
                    ent["canonical_name"],
                    float(sim),
                    "embedding",
                )
            if sim >= EMBED_THRESHOLD + 0.05:
                self.db.add_alias(ent["id"], name)
                return ResolvedMention(
                    name,
                    mention.type,
                    ent["id"],
                    ent["canonical_name"],
                    float(sim),
                    "embedding",
                )

        # 6) Create new entity
        emb = self.embedder.embed(f"{name} ({mention.type})", self.db.dim)
        eid = self.db.create_entity(
            canonical_name=name,
            entity_type=mention.type,
            aliases=[name, normalize_name(name)] if normalize_name(name) != name.lower() else [name],
            embedding=emb,
        )
        return ResolvedMention(name, mention.type, eid, name, 1.0, "created")

    def resolve_all(
        self, mentions: list[Mention], *, context: str = ""
    ) -> list[ResolvedMention]:
        return [self.resolve(m, context=context) for m in mentions]
