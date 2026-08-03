"""Entity mention detection via local LLM + GBNF (with heuristic fallback)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from memory_engine.extraction import LlamaCppBackend, MockLLM

# Single-line GBNF (llama.cpp rejects multiline RHS continuations)
MENTION_GBNF = r"""
root ::= "[" ws (mention ("," ws mention)*)? ws "]"
mention ::= "{" ws "\"" "mention" "\"" ws ":" ws string "," ws "\"" "type" "\"" ws ":" ws string ws "}"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
"""

MENTION_PROMPT = """\
Extract named entity mentions from the text.
Return ONLY a JSON array. Each element:
  mention (string) — the surface form as written
  type (string) — one of: person, organization, location, field, software, product, other
Skip pronouns and generic words. If none, return [].
"""

KNOWN_TYPES = {
    "person",
    "organization",
    "location",
    "field",
    "software",
    "product",
    "other",
}


@dataclass
class Mention:
    mention: str
    type: str

    def as_dict(self) -> dict[str, str]:
        return {"mention": self.mention, "type": self.type}


def _parse_mentions(raw: str) -> list[Mention]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[Mention] = []
    seen: set[str] = set()
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("mention", "")).strip()
        typ = str(item.get("type", "other")).strip().lower()
        if not name or name.lower() in seen:
            continue
        if typ not in KNOWN_TYPES:
            typ = "other"
        seen.add(name.lower())
        out.append(Mention(name, typ))
    return out


_HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(University of [A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)\b"), "organization"),
    (re.compile(r"\b(AppleInc|Apple Inc)\b", re.I), "organization"),
    (re.compile(r"\b([A-Z]{2,}\b)"), "organization"),  # NTU, MIT, NYU
    (re.compile(r"\b(Py[- ]?Torch|TensorFlow|Python|JavaScript|Rust|Go)\b", re.I), "software"),
    (re.compile(r"\b(Mechanical Engineering|Computer Science|Electrical Engineering|robotics|AI|ML)\b", re.I), "field"),
    (
        re.compile(
            r"\b(Taiwan|Singapore|Ann Arbor|Michigan|NYC|New York|San Francisco|SF|"
            r"Chicago|Boston|Miami|Seattle|Austin|London|Tokyo)\b",
            re.I,
        ),
        "location",
    ),
]


def heuristic_mentions(text: str) -> list[Mention]:
    found: list[Mention] = []
    seen: set[str] = set()
    # Longer patterns first so "University of Michigan" wins over "Michigan"
    patterns = sorted(_HEURISTIC_PATTERNS, key=lambda p: p[0].pattern.count("("), reverse=True)
    for pattern, typ in patterns:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            key = name.lower()
            if key in seen or len(name) < 2:
                continue
            # Skip if this mention is a substring of an already found longer mention
            if any(key != s and key in s for s in seen):
                continue
            seen.add(key)
            found.append(Mention(name, typ))
    return found


class MentionDetector:
    """Detect entity mentions with GBNF-constrained local LLM."""

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm or MockLLM()

    def detect(self, text: str) -> list[Mention]:
        if isinstance(self.llm, MockLLM) or not hasattr(self.llm, "complete"):
            return heuristic_mentions(text)

        prompt = f"{MENTION_PROMPT}\n\nText:\n{text}\n\nJSON:"
        try:
            if isinstance(self.llm, LlamaCppBackend):
                raw = self.llm.complete(prompt, grammar=MENTION_GBNF)
            else:
                raw = self.llm.complete(prompt, grammar=MENTION_GBNF)
            mentions = _parse_mentions(raw)
            if mentions:
                return mentions
        except Exception:
            pass
        return heuristic_mentions(text)
