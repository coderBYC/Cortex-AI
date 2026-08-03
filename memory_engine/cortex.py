"""CortexEngine — full knowledge-layer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from memory_engine.extraction import (
    LocalEmbedder,
    MockLLM,
    create_llm,
    require_local_embedder,
    resolve_model_path,
)
from memory_engine.extractors import KnowledgeExtractor
from memory_engine.knowledge_db import KnowledgeDB
from memory_engine.mentions import MentionDetector
from memory_engine.query import (
    Answerer,
    BuiltContext,
    ContextBuilder,
    IntentRouter,
    RoutedQuery,
    format_answer_with_entities,
)
from memory_engine.resolve import EntityResolver, ResolvedMention
from memory_engine.state_engine import StateTransitionEngine


@dataclass
class IngestResult:
    memory_id: str
    timestamp: str
    mentions: list[dict[str, Any]] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "mentions": self.mentions,
            "resolved": self.resolved,
            "facts": self.facts,
            "events": self.events,
            "actions": self.actions,
        }


@dataclass
class AskResult:
    query: str
    intent: str
    context: BuiltContext
    answer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "answer": self.answer,
            "context_blocks": self.context.blocks,
            "states": self.context.states,
            "facts": self.context.facts,
            "events": [
                {k: v for k, v in e.items() if k != "embedding"}
                for e in self.context.events
            ],
        }


class CortexEngine:
    """
    USER INPUT
      → RAW MEMORY STORE (immutable)
      → ENTITY MENTION DETECTION (Local LLM + GBNF)
      → ENTITY RESOLUTION (Alias + JW + FTS + Embedding)
      → FACT + EVENT EXTRACTION
      → KNOWLEDGE LAYER + State Transition Engine

    QUERY
      → Intent Router (State / Fact / Event)
      → Context Builder
      → Local LLM → Answer
    """

    def __init__(
        self,
        db_path: str | Path = "memories.db",
        *,
        model_path: Optional[str | Path] = None,
        allow_mock: bool = False,
        llm: Any = None,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        self.db = KnowledgeDB(db_path)
        self.embedder = embedder or require_local_embedder()
        if llm is not None:
            self.llm = llm
        elif allow_mock:
            # Explicit offline/test path — do not load GGUF even if present.
            self.llm = MockLLM()
        else:
            resolved = resolve_model_path(model_path)
            if resolved is None:
                raise FileNotFoundError(
                    "No GGUF model found. Pass model_path or allow_mock=True."
                )
            self.llm = create_llm(model_path=model_path, allow_mock=False)

        self.mentions = MentionDetector(self.llm)
        self.resolver = EntityResolver(self.db, self.embedder)
        self.extractor = KnowledgeExtractor(self.llm)
        self.state = StateTransitionEngine(self.db)
        self.router = IntentRouter()
        self.context_builder = ContextBuilder(self.db)
        self.answerer = Answerer(self.llm, db=self.db)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "CortexEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- ingest

    def remember(
        self, text: str, *, timestamp: Optional[str] = None
    ) -> IngestResult:
        """Full write path: raw → mentions → resolve → facts/events → state."""
        emb = self.embedder.embed(text, self.db.dim)
        mid = self.db.add_memory(text, timestamp=timestamp, embedding=emb)
        mem = self.db.get_memory(mid)
        assert mem is not None

        # 1) Mention detection
        mention_list = self.mentions.detect(text)

        # 2) Entity resolution
        resolved: list[ResolvedMention] = self.resolver.resolve_all(
            mention_list, context=text
        )
        for r in resolved:
            self.db.insert_mention(
                mid,
                r.mention,
                r.type,
                resolved_entity_id=r.entity_id,
                resolve_score=r.score,
                resolve_method=r.method,
            )

        # 3) Fact + event extraction
        facts, events = self.extractor.extract(text, resolved)

        # 4) Knowledge layer + state transitions
        actions: list[dict[str, Any]] = []
        actions.extend(self.state.apply_facts(facts, memory_id=mid))
        actions.extend(self.state.apply_events(events, memory_id=mid))

        return IngestResult(
            memory_id=mid,
            timestamp=str(mem["timestamp"]),
            mentions=[m.as_dict() for m in mention_list],
            resolved=[r.as_dict() for r in resolved],
            facts=[f.as_dict() for f in facts],
            events=[e.as_dict() for e in events],
            actions=actions,
        )

    # ----------------------------------------------------------------- query

    def ask(self, query: str) -> AskResult:
        routed: RoutedQuery = self.router.route(query)
        ctx = self.context_builder.build(routed)
        answer = self.answerer.answer(query, ctx)
        answer = format_answer_with_entities(answer, self.db)
        # Also resolve entity ids that slipped into heuristic answers
        if answer.startswith("Current school:") or "ENTITY_" in answer:
            answer = format_answer_with_entities(answer, self.db)
        return AskResult(
            query=query,
            intent=routed.intent.value,
            context=ctx,
            answer=answer,
        )

    def summary(self) -> dict[str, Any]:
        return self.db.dump_summary()
