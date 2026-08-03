"""State transition engine — conflict handling over facts/events → current state."""

from __future__ import annotations

from typing import Any

from memory_engine.extractors import ExtractedEvent, ExtractedFact
from memory_engine.knowledge_db import KnowledgeDB


# Predicates that are single-valued "current state" keys (newer replaces older).
STATEFUL_PREDICATES = {
    "studied_at",
    "studies_at",  # alias form of studied_at
    "lives_in",
    "works_as",
    "works_at",
    "major",
    "location",
    "occupation",
    "relationship_status",
}

# Normalize synonymous predicates onto one state key.
PREDICATE_ALIASES = {
    "studies_at": "studied_at",
    "location": "lives_in",
}


class StateTransitionEngine:
    """
    Apply newly extracted facts/events into the knowledge layer.

    Conflict policy:
      - For stateful predicates: invalidate prior active facts for (subject, predicate),
        write the new fact as current, upsert states[subject][predicate] = object.
      - For multi-valued predicates (uses, interested_in): append without invalidating.
      - For transfer_school / moved events: update from→to state and record event.
    """

    def __init__(self, db: KnowledgeDB) -> None:
        self.db = db

    def apply_facts(
        self, facts: list[ExtractedFact], *, memory_id: str
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for fact in facts:
            predicate = PREDICATE_ALIASES.get(fact.predicate, fact.predicate)
            replace = predicate in STATEFUL_PREDICATES or fact.predicate in STATEFUL_PREDICATES
            if replace:
                self.db.invalidate_facts(fact.subject, predicate)
                if predicate != fact.predicate:
                    self.db.invalidate_facts(fact.subject, fact.predicate)

            fid = self.db.insert_fact(
                subject=fact.subject,
                predicate=predicate,
                obj=fact.object,
                time_expr=fact.time or None,
                memory_id=memory_id,
            )
            if replace or predicate in {"uses", "interested_in", "major"}:
                self.db.upsert_state(
                    fact.subject,
                    predicate,
                    fact.object,
                    source_fact_id=fid,
                )
            actions.append(
                {
                    "action": "fact_upsert" if replace else "fact_add",
                    "fact_id": fid,
                    "subject": fact.subject,
                    "predicate": predicate,
                    "object": fact.object,
                    "time": fact.time,
                }
            )
        return actions

    def apply_events(
        self, events: list[ExtractedEvent], *, memory_id: str
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for ev in events:
            payload = {
                "from": ev.from_id,
                "to": ev.to_id,
                "year": ev.year,
            }
            eid = self.db.insert_event(
                ev.event_type,
                payload,
                memory_id=memory_id,
                event_time=ev.year or None,
            )
            # Prefer existing facts for SPO history; events mainly update state.
            if ev.event_type == "transfer_school" and ev.to_id:
                active = self.db.active_facts(subject="USER", predicate="studied_at")
                matching = [f for f in active if f["object"] == ev.to_id]
                if matching:
                    fid = matching[0]["id"]
                    self.db.invalidate_facts("USER", "studied_at", except_id=fid)
                else:
                    self.db.invalidate_facts("USER", "studied_at")
                    fid = self.db.insert_fact(
                        "USER",
                        "studied_at",
                        ev.to_id,
                        time_expr=f"{ev.year}-present" if ev.year else "present",
                        memory_id=memory_id,
                    )
                self.db.upsert_state(
                    "USER",
                    "studied_at",
                    ev.to_id,
                    source_fact_id=fid,
                    source_event_id=eid,
                )
            elif ev.event_type == "moved" and ev.to_id:
                active = self.db.active_facts(subject="USER", predicate="lives_in")
                matching = [f for f in active if f["object"] == ev.to_id]
                if matching:
                    fid = matching[0]["id"]
                    self.db.invalidate_facts("USER", "lives_in", except_id=fid)
                else:
                    self.db.invalidate_facts("USER", "lives_in")
                    fid = self.db.insert_fact(
                        "USER",
                        "lives_in",
                        ev.to_id,
                        time_expr=f"{ev.year}-present" if ev.year else "present",
                        memory_id=memory_id,
                    )
                self.db.upsert_state(
                    "USER",
                    "lives_in",
                    ev.to_id,
                    source_fact_id=fid,
                    source_event_id=eid,
                )

            actions.append(
                {
                    "action": "event_add",
                    "event_id": eid,
                    **ev.as_dict(),
                }
            )
        return actions
