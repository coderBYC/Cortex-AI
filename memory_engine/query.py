"""Query path: intent router → context builder → local LLM answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from memory_engine.knowledge_db import KnowledgeDB


class Intent(str, Enum):
    STATE = "state"  # current attributes → SQL on states
    FACT = "fact"  # SPO / relationships → facts graph/table
    EVENT = "event"  # transitions / when → hybrid events + memories
    GENERAL = "general"  # blend of all


STATE_CUES = re.compile(
    r"(?i)\b(where (do|does|am|is)|what (is|are) my|currently|now|live|lives|"
    r"school|university|major|occupation|work|job|study|studying)\b"
)
EVENT_CUES = re.compile(
    r"(?i)\b(when|transfer|moved|started|before|after|year|happened|"
    r"change|switched|left|joined)\b"
)
FACT_CUES = re.compile(
    r"(?i)\b(who|what|which|uses|use|know|about|friend|related|"
    r"studied|interested)\b"
)


@dataclass
class RoutedQuery:
    intent: Intent
    query: str
    hints: list[str] = field(default_factory=list)


@dataclass
class BuiltContext:
    intent: Intent
    blocks: list[str]
    states: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        if not self.blocks:
            return "(no relevant knowledge)"
        return "\n".join(self.blocks)


class IntentRouter:
    """Route a natural-language query to State / Fact / Event / General."""

    def route(self, query: str) -> RoutedQuery:
        q = query.strip()
        hints: list[str] = []
        scores = {
            Intent.STATE: 0,
            Intent.FACT: 0,
            Intent.EVENT: 0,
            Intent.GENERAL: 1,
        }
        if STATE_CUES.search(q):
            scores[Intent.STATE] += 3
            hints.append("state_cues")
        if EVENT_CUES.search(q):
            scores[Intent.EVENT] += 3
            hints.append("event_cues")
        if FACT_CUES.search(q):
            scores[Intent.FACT] += 2
            hints.append("fact_cues")

        # Prefer the highest non-general score; ties → general blend
        ranked = sorted(
            ((s, i) for i, s in scores.items() if i != Intent.GENERAL),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 3:
            # If state and event both strong, keep GENERAL blend
            if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
                intent = Intent.GENERAL
            else:
                intent = ranked[0][1]
        else:
            intent = Intent.GENERAL
        return RoutedQuery(intent=intent, query=q, hints=hints)


def _human_entity(db: KnowledgeDB, ref: str) -> str:
    if not ref or ref == "USER":
        return "USER"
    ent = db.get_entity(ref)
    if ent:
        return str(ent["canonical_name"])
    return ref


class ContextBuilder:
    """Assemble grounded context from the knowledge layer for the answer LLM."""

    def __init__(self, db: KnowledgeDB) -> None:
        self.db = db

    def build(self, routed: RoutedQuery, *, limit: int = 8) -> BuiltContext:
        intent = routed.intent
        q = routed.query
        states: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        blocks: list[str] = []

        want_state = intent in {Intent.STATE, Intent.GENERAL}
        want_fact = intent in {Intent.FACT, Intent.GENERAL, Intent.STATE}
        want_event = intent in {Intent.EVENT, Intent.GENERAL}
        want_mem = intent in {Intent.EVENT, Intent.GENERAL, Intent.FACT}

        if want_state:
            states = self.db.list_states("USER")
            if states:
                blocks.append("## Current State")
                for s in states:
                    val = _human_entity(self.db, s["value"])
                    blocks.append(f"- USER.{s['key']} = {val}")

        if want_fact:
            facts = self.db.search_facts(q, limit=limit)
            if not facts and intent == Intent.STATE:
                facts = self.db.active_facts(subject="USER")[:limit]
            if facts:
                blocks.append("## Facts")
                for f in facts:
                    subj = _human_entity(self.db, f["subject"])
                    obj = _human_entity(self.db, f["object"])
                    time = f" [{f['time_expr']}]" if f.get("time_expr") else ""
                    blocks.append(f"- ({subj}, {f['predicate']}, {obj}){time}")

        if want_event:
            events = self.db.search_events(q, limit=limit)
            if not events and intent == Intent.EVENT:
                events = self.db.list_events(limit=limit)
            if events:
                blocks.append("## Events")
                for e in events:
                    payload = e.get("payload") or {}
                    fr = _human_entity(self.db, str(payload.get("from", "")))
                    to = _human_entity(self.db, str(payload.get("to", "")))
                    year = payload.get("year") or e.get("event_time") or ""
                    blocks.append(
                        f"- {e['event_type']}: {fr} → {to}"
                        + (f" ({year})" if year else "")
                    )

        if want_mem:
            memories = self.db.search_memories_bm25(q, limit=min(3, limit))
            if memories:
                blocks.append("## Raw Memories")
                for m in memories:
                    blocks.append(f"- [{m['id']} @ {m['timestamp']}] {m['text']}")

        return BuiltContext(
            intent=intent,
            blocks=blocks,
            states=states,
            facts=facts,
            events=events,
            memories=memories,
        )


ANSWER_PROMPT = """\
You are a precise personal memory assistant.
Answer the question using ONLY the knowledge context below.
If the context is insufficient, say you don't know.
Prefer current state for "now/currently" questions; use events for "when/transfer".
Be concise (1-3 sentences).

Knowledge:
{context}

Question: {question}
Answer:"""


class Answerer:
    """Local LLM answer head over built context."""

    def __init__(self, llm: Any = None, db: Optional[KnowledgeDB] = None) -> None:
        self.llm = llm
        self.db = db

    def answer(self, question: str, context: BuiltContext) -> str:
        block = context.as_prompt_block()
        if block == "(no relevant knowledge)":
            return "I don't have enough stored knowledge to answer that."

        from memory_engine.extraction import MockLLM

        if self.llm is None or isinstance(self.llm, MockLLM):
            return self._heuristic_answer(question, context)

        prompt = ANSWER_PROMPT.format(context=block, question=question)
        try:
            raw = self.llm.complete(prompt)
            return (raw or "").strip() or self._heuristic_answer(question, context)
        except Exception:
            return self._heuristic_answer(question, context)

    def _label(self, ref: str) -> str:
        if self.db is None:
            return ref
        return _human_entity(self.db, ref)

    def _heuristic_answer(self, question: str, context: BuiltContext) -> str:
        q = question.lower()
        for s in context.states:
            key = s["key"]
            if key == "studied_at" and re.search(r"school|universit|stud", q):
                return f"Current school: {self._label(s['value'])}."
            if key == "lives_in" and re.search(r"live|where", q):
                return f"Lives in {self._label(s['value'])}."
            if key == "major" and "major" in q:
                return f"Major: {s['value']}."
            if key == "uses" and re.search(r"software|use|tool|pytorch|library", q):
                return f"Uses: {self._label(s['value'])}."

        if context.events and re.search(r"transfer|when|moved|switch", q):
            e = context.events[0]
            payload = e.get("payload") or {}
            fr = self._label(str(payload.get("from", "")))
            to = self._label(str(payload.get("to", "")))
            year = payload.get("year") or e.get("event_time") or ""
            return (
                f"Event {e['event_type']}: {fr} → {to}"
                + (f" ({year})" if year else "")
                + "."
            )

        # Prefer facts that match query tokens (uses / major / etc.)
        if context.facts:
            for f in context.facts:
                if f["predicate"] in q or any(
                    t in f["predicate"] for t in q.split() if len(t) > 3
                ):
                    return (
                        f"{self._label(f['subject'])}'s {f['predicate']} "
                        f"is {self._label(f['object'])}."
                    )
            f = context.facts[0]
            return (
                f"{self._label(f['subject'])}'s {f['predicate']} "
                f"is {self._label(f['object'])}."
            )

        if context.memories:
            return f"From memory: {context.memories[0]['text']}"

        return "I don't have enough stored knowledge to answer that."


def format_answer_with_entities(text: str, db: KnowledgeDB) -> str:
    """Replace ENTITY_xxx ids in a string with canonical names when present."""
    def repl(m: re.Match[str]) -> str:
        eid = m.group(0)
        ent = db.get_entity(eid)
        return ent["canonical_name"] if ent else eid

    return re.sub(r"ENTITY_\d+", repl, text)
