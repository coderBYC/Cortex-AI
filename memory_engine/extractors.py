"""Fact + event extraction with GBNF (knowledge-layer triples / events)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from memory_engine.extraction import LlamaCppBackend, MockLLM
from memory_engine.resolve import ResolvedMention

FACT_EVENT_GBNF = r"""
root ::= "{" ws "\"" "facts" "\"" ws ":" ws "[" ws (fact ("," ws fact)*)? ws "]" ws "," ws "\"" "events" "\"" ws ":" ws "[" ws (event ("," ws event)*)? ws "]" ws "}"
fact ::= "{" ws "\"" "subject" "\"" ws ":" ws string "," ws "\"" "predicate" "\"" ws ":" ws string "," ws "\"" "object" "\"" ws ":" ws string "," ws "\"" "time" "\"" ws ":" ws string ws "}"
event ::= "{" ws "\"" "event_type" "\"" ws ":" ws string "," ws "\"" "from" "\"" ws ":" ws string "," ws "\"" "to" "\"" ws ":" ws string "," ws "\"" "year" "\"" ws ":" ws string ws "}"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
"""

FACT_EVENT_PROMPT = """\
You extract durable knowledge from a user memory, using the resolved entity IDs provided.
Return ONLY JSON of the form:
{
  "facts": [{"subject":"USER"|"ENTITY_xxx", "predicate":"snake_case", "object":"ENTITY_xxx"|literal, "time":"..."}],
  "events": [{"event_type":"snake_case", "from":"ENTITY_xxx"|\"\", "to":"ENTITY_xxx"|\"\", "year":"YYYY"|\"\"}]
}
Rules:
- Prefer ENTITY_xxx ids from the resolved list for organizations/locations/software when possible.
- subject is usually "USER" for personal facts.
- predicates like studied_at, major, uses, lives_in, works_as.
- events capture transitions (transfer_school, moved, started_job).
- If none, return empty arrays.
"""


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    time: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "time": self.time,
        }


@dataclass
class ExtractedEvent:
    event_type: str
    from_id: str = ""
    to_id: str = ""
    year: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "from": self.from_id,
            "to": self.to_id,
            "year": self.year,
        }


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_facts_events(raw: str) -> tuple[list[ExtractedFact], list[ExtractedEvent]]:
    data = _parse_json_object(raw)
    facts: list[ExtractedFact] = []
    events: list[ExtractedEvent] = []
    for item in data.get("facts") or []:
        if not isinstance(item, dict):
            continue
        try:
            facts.append(
                ExtractedFact(
                    subject=str(item.get("subject", "USER")).strip() or "USER",
                    predicate=str(item["predicate"]).strip().lower().replace(" ", "_"),
                    object=str(item["object"]).strip(),
                    time=str(item.get("time", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    for item in data.get("events") or []:
        if not isinstance(item, dict):
            continue
        try:
            events.append(
                ExtractedEvent(
                    event_type=str(item.get("event_type", "")).strip().lower().replace(" ", "_"),
                    from_id=str(item.get("from", "")).strip(),
                    to_id=str(item.get("to", "")).strip(),
                    year=str(item.get("year", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return [f for f in facts if f.predicate and f.object], [
        e for e in events if e.event_type
    ]


def _find_resolved(resolved: list[ResolvedMention], surface: str) -> Optional[ResolvedMention]:
    surface_l = surface.lower()
    for r in resolved:
        if r.mention.lower() == surface_l or r.canonical_name.lower() == surface_l:
            return r
    # Substring fallback (longer names first)
    for r in sorted(resolved, key=lambda x: len(x.mention), reverse=True):
        if surface_l in r.mention.lower() or r.mention.lower() in surface_l:
            return r
    return None


def heuristic_facts_events(
    text: str, resolved: list[ResolvedMention]
) -> tuple[list[ExtractedFact], list[ExtractedEvent]]:
    """Deterministic fallback extractor for demos without a strong LLM."""
    facts: list[ExtractedFact] = []
    events: list[ExtractedEvent] = []
    by_type: dict[str, list[ResolvedMention]] = {}
    for r in resolved:
        by_type.setdefault(r.type, []).append(r)

    year_m = re.search(r"\b(20\d{2})\b", text)

    # Explicit past → present school transfer patterns (order from text, not mention list)
    past_m = re.search(
        r"(?i)\b(?:used to|previously)\s+(?:study|studied|studying)\s+at\s+([^.,;]+?)(?:\s+in\s+[^.,;]+)?(?=[.,;]|$)",
        text,
    )
    school_to_m = re.search(
        r"(?i)\b(?:transferred|transfer)\s+to\s+(?:the\s+)?([^.,;]+?)(?:\s+for\s|\s+in\s|[.,;]|$)",
        text,
    )

    past_org = _find_resolved(resolved, past_m.group(1).strip()) if past_m else None
    to_org = _find_resolved(resolved, school_to_m.group(1).strip()) if school_to_m else None

    if past_org and to_org and past_org.entity_id != to_org.entity_id:
        facts.append(ExtractedFact("USER", "studied_at", past_org.entity_id, "past"))
        facts.append(
            ExtractedFact(
                "USER",
                "studied_at",
                to_org.entity_id,
                f"{year_m.group(1)}-present" if year_m else "present",
            )
        )
        events.append(
            ExtractedEvent(
                "transfer_school",
                past_org.entity_id,
                to_org.entity_id,
                year_m.group(1) if year_m else "",
            )
        )
    elif re.search(r"(?i)\b(studied|study|studying)\s+at\b", text):
        study_m = re.search(
            r"(?i)\b(?:studied|study|studying)\s+at\s+(?:the\s+)?([^.,;]+?)(?=[.,;]|$)",
            text,
        )
        org = _find_resolved(resolved, study_m.group(1).strip()) if study_m else None
        if org is None:
            orgs = by_type.get("organization", [])
            org = orgs[0] if orgs else None
        if org:
            facts.append(
                ExtractedFact("USER", "studies_at", org.entity_id, "present")
            )

    # works_at
    work_m = re.search(
        r"(?i)\b(?:work|works|working|worked)\s+at\s+(?:the\s+)?([^.,;]+?)(?=[.,;]|$)",
        text,
    )
    if work_m:
        org = _find_resolved(resolved, work_m.group(1).strip())
        if org is None and by_type.get("organization"):
            org = by_type["organization"][0]
        if org:
            facts.append(ExtractedFact("USER", "works_at", org.entity_id, "present"))

    # relationship_status
    rel_m = re.search(
        r"(?i)\b(?:i(?:'m| am)|relationship status(?:\s+is)?)\s+"
        r"(single|married|engaged|divorced|in a relationship|dating)\b",
        text,
    )
    if rel_m:
        facts.append(
            ExtractedFact("USER", "relationship_status", rel_m.group(1).lower(), "present")
        )

    for r in by_type.get("field", []):
        if r.mention.lower() in text.lower():
            pred = "interested_in" if r.mention.lower() == "robotics" else "major"
            facts.append(ExtractedFact("USER", pred, r.mention, ""))

    for r in by_type.get("software", []):
        facts.append(ExtractedFact("USER", "uses", r.entity_id, "present"))

    # Residence: "I live in X" / "I moved to X"
    live_m = re.search(
        r"(?i)\b(?:live|lives|living|based)\s+in\s+([A-Za-z][A-Za-z\s]+?)(?=[.,;]|$)",
        text,
    )
    moved_m = re.search(
        r"(?i)\bmoved\s+to\s+([A-Za-z][A-Za-z\s]+?)(?=[.,;]|$)",
        text,
    )
    if moved_m:
        dest = _find_resolved(resolved, moved_m.group(1).strip())
        if dest is None and by_type.get("location"):
            # Prefer location whose mention appears in the capture
            cap = moved_m.group(1).strip().lower()
            for loc in by_type["location"]:
                if loc.mention.lower() in cap or cap in loc.mention.lower():
                    dest = loc
                    break
            dest = dest or by_type["location"][0]
        if dest:
            src = None
            # optional: previous lives_in not known here; event from=""
            events.append(
                ExtractedEvent(
                    "moved",
                    src.entity_id if src else "",
                    dest.entity_id,
                    year_m.group(1) if year_m else "",
                )
            )
            facts.append(
                ExtractedFact("USER", "lives_in", dest.entity_id, "present")
            )
    elif live_m:
        loc = _find_resolved(resolved, live_m.group(1).strip())
        if loc is None and by_type.get("location"):
            cap = live_m.group(1).strip().lower()
            for candidate in by_type["location"]:
                if candidate.mention.lower() in cap or cap in candidate.mention.lower():
                    loc = candidate
                    break
            loc = loc or by_type["location"][0]
        if loc:
            facts.append(ExtractedFact("USER", "lives_in", loc.entity_id, "present"))

    return facts, events


class KnowledgeExtractor:
    """Extract SPO facts + events given resolved entity IDs."""

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm or MockLLM()

    def extract(
        self, text: str, resolved: list[ResolvedMention]
    ) -> tuple[list[ExtractedFact], list[ExtractedEvent]]:
        entity_lines = "\n".join(
            f"- {r.mention} => {r.entity_id} ({r.type}, canonical={r.canonical_name})"
            for r in resolved
        ) or "(none)"
        prompt = (
            f"{FACT_EVENT_PROMPT}\n\n"
            f"Resolved entities:\n{entity_lines}\n\n"
            f"Memory text:\n{text}\n\n"
            f"JSON:"
        )

        if isinstance(self.llm, MockLLM):
            return heuristic_facts_events(text, resolved)

        try:
            if isinstance(self.llm, LlamaCppBackend):
                raw = self.llm.complete(prompt, grammar=FACT_EVENT_GBNF)
            else:
                raw = self.llm.complete(prompt, grammar=FACT_EVENT_GBNF)
            facts, events = parse_facts_events(raw)
            if facts or events:
                return facts, events
        except Exception:
            pass
        return heuristic_facts_events(text, resolved)
