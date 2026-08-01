"""GBNF grammar definitions, local LLM extraction, and Jaro-Winkler deduplication."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import jellyfish
import numpy as np

from memory_engine.db import MemoryDB

# Spec: merge when Jaro-Winkler similarity > 0.88
JARO_WINKLER_THRESHOLD = 0.88

# ---------------------------------------------------------------------------
# GBNF grammar — forces the LLM to emit a JSON array of memory objects.
# Compatible with llama-cpp-python's `grammar=` / LlamaGrammar.from_string().
# ---------------------------------------------------------------------------

MEMORY_JSON_GBNF = r"""
root        ::= "[" ws memory ("," ws memory)* ws "]"
memory      ::= "{" ws
                "\"entity\""    ws ":" ws string "," ws
                "\"attribute\"" ws ":" ws string "," ws
                "\"value\""     ws ":" ws string "," ws
                "\"confidence\"" ws ":" ws number "," ws
                "\"is_permanent\"" ws ":" ws boolean
                ws "}"
string      ::= "\"" chars "\""
chars       ::= char*
char        ::= [^"\\] | "\\" escape
escape      ::= ["\\/bfnrt] | "u" hex hex hex hex
hex         ::= [0-9a-fA-F]
number      ::= "0" | [1-9] [0-9]? ("." [0-9]+)?
boolean     ::= "true" | "false"
ws          ::= [ \t\n\r]*
"""

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable personal memory facts from a conversation.
Return ONLY a JSON array. Each element must have:
  entity (string), attribute (string), value (string),
  confidence (0-1 number), is_permanent (boolean).
Skip greetings, ephemeral chit-chat, and anything not worth remembering.
Prefer canonical entity names (e.g. "Bryan" not "he").
"""


@dataclass
class ExtractedFact:
    entity: str
    attribute: str
    value: str
    confidence: float = 1.0
    is_permanent: bool = False

    def normalized_entity(self) -> str:
        return self.entity.strip()


class LLMBackend(Protocol):
    """Minimal interface so MockLLM and llama-cpp-python share a call site."""

    def complete(self, prompt: str, *, grammar: Optional[str] = None) -> str: ...


class MockLLM:
    """
    Deterministic, zero-dependency extractor for demos and tests.
    Pulls simple \"X's Y is Z\" / \"My name is X\" style facts with regex.
    """

    _PATTERNS: list[tuple[re.Pattern[str], Any]] = [
        (
            re.compile(
                r"(?i)\b(?:my name is|i(?:'m| am))\s+([A-Z][a-zA-Z\-']+)"
            ),
            lambda m: ExtractedFact(m.group(1), "name", m.group(1), 0.95, True),
        ),
        (
            re.compile(
                r"(?i)\b(?:i live in|i(?:'m| am) based in)\s+"
                r"([A-Za-z][A-Za-z0-9\s]+?)(?=\s+and\b|[.,;]|$)"
            ),
            lambda m: ExtractedFact(
                "user", "location", m.group(1).strip(), 0.9, False
            ),
        ),
        (
            re.compile(
                r"(?i)\b(?:i work as|i(?:'m| am) a(?:n)?)\s+"
                r"([A-Za-z][A-Za-z\s\-]+?)(?=\s+and\b|[.,;]|$)"
            ),
            lambda m: ExtractedFact(
                "user", "occupation", m.group(1).strip(), 0.85, False
            ),
        ),
        (
            re.compile(
                r"(?i)\bi prefer\s+([^.]+?)(?=\s+and\b|[.,;]|$)"
            ),
            lambda m: ExtractedFact(
                "user", "preference", m.group(1).strip(), 0.9, True
            ),
        ),
        (
            # "Alice's dog is named Nimbus" before the generic "is ..." pattern
            re.compile(
                r"(?i)\b([A-Z][a-zA-Z\-']+)'s\s+(\w+)\s+is\s+named\s+"
                r"([A-Za-z][A-Za-z\-']*)"
            ),
            lambda m: ExtractedFact(
                m.group(1), m.group(2).lower(), m.group(3), 0.9, False
            ),
        ),
        (
            re.compile(
                r"(?i)\b([A-Z][a-zA-Z\-']+)'s\s+(\w+)\s+is\s+"
                r"([A-Za-z0-9][^.;]*?)(?=(?:\s+and\s+[A-Z])|[.;]|$)"
            ),
            lambda m: ExtractedFact(
                m.group(1),
                m.group(2).lower(),
                m.group(3).strip(),
                0.85,
                False,
            ),
        ),
        (
            re.compile(r"(?i)\bi (?:like|love|enjoy)\s+([^.]+?)(?=\s+and\b|[.,;]|$)"),
            lambda m: ExtractedFact(
                "user", "likes", m.group(1).strip(), 0.8, True
            ),
        ),
    ]

    def complete(self, prompt: str, *, grammar: Optional[str] = None) -> str:
        # The conversation text is expected after the last "Conversation:" marker.
        text = prompt
        if "Conversation:" in prompt:
            text = prompt.rsplit("Conversation:", 1)[-1]

        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str]] = set()
        for pattern, builder in self._PATTERNS:
            for match in pattern.finditer(text):
                fact = builder(match)
                key = (fact.entity.lower(), fact.attribute.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(fact)

        payload = [
            {
                "entity": f.entity,
                "attribute": f.attribute,
                "value": f.value,
                "confidence": f.confidence,
                "is_permanent": f.is_permanent,
            }
            for f in facts
        ]
        return json.dumps(payload)

    def embed(self, text: str, dim: int = 384) -> np.ndarray:
        """Hashing trick embedding — stable, local, no model required."""
        return hashing_embed(text, dim)


class LlamaCppBackend:
    """
    Thin adapter around llama-cpp-python with GBNF-constrained JSON output.
    Instantiated only when a GGUF model path is provided.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4) -> None:
        from llama_cpp import Llama, LlamaGrammar  # type: ignore

        self._LlamaGrammar = LlamaGrammar
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
            embedding=True,
        )

    def complete(self, prompt: str, *, grammar: Optional[str] = None) -> str:
        g = self._LlamaGrammar.from_string(grammar) if grammar else None
        out = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            grammar=g,
            stop=["</s>", "```"],
        )
        return out["choices"][0]["text"].strip()

    def embed(self, text: str, dim: int = 384) -> np.ndarray:
        raw = self.llm.create_embedding(text)["data"][0]["embedding"]
        vec = np.asarray(raw, dtype=np.float32)
        if vec.shape[0] == dim:
            return vec
        # Pad / truncate to the DB's configured dimensionality.
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, vec.shape[0])
        out[:n] = vec[:n]
        return out


def hashing_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Local bag-of-tokens hashing embedding (signed hashing trick).
    Good enough for prototype vector search without a cloud model.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = hash(tok)
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def parse_facts(raw: str) -> list[ExtractedFact]:
    """Parse LLM JSON (possibly fenced) into ExtractedFact list."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Recover the outermost JSON array if the model added prose.
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        return []

    facts: list[ExtractedFact] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            facts.append(
                ExtractedFact(
                    entity=str(item["entity"]).strip(),
                    attribute=str(item["attribute"]).strip(),
                    value=str(item["value"]).strip(),
                    confidence=float(item.get("confidence", 1.0)),
                    is_permanent=bool(item.get("is_permanent", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return facts


def jaro_winkler(a: str, b: str) -> float:
    return float(jellyfish.jaro_winkler_similarity(a.lower(), b.lower()))


def find_duplicate_entity(
    entity: str,
    existing: Sequence[tuple[int, str]],
    threshold: float = JARO_WINKLER_THRESHOLD,
) -> Optional[int]:
    """
    Return the memory id of the best-matching existing entity if similarity
    exceeds `threshold`, else None.
    """
    best_id: Optional[int] = None
    best_score = 0.0
    for mid, name in existing:
        score = jaro_winkler(entity, name)
        if score > best_score:
            best_score = score
            best_id = mid
    if best_id is not None and best_score > threshold:
        return best_id
    return None


class MemoryExtractor:
    """
    Deduplicating ingestion pipeline:
      conversation → (GBNF) JSON facts → Jaro-Winkler entity merge → SQLite.
    """

    def __init__(
        self,
        db: MemoryDB,
        llm: Optional[LLMBackend] = None,
        *,
        threshold: float = JARO_WINKLER_THRESHOLD,
    ) -> None:
        self.db = db
        self.llm = llm or MockLLM()
        self.threshold = threshold

    def _embed(self, text: str) -> np.ndarray:
        embed_fn = getattr(self.llm, "embed", None)
        if callable(embed_fn):
            return np.asarray(embed_fn(text, self.db.dim), dtype=np.float32)
        return hashing_embed(text, self.db.dim)

    def extract_from_text(self, conversation: str) -> list[ExtractedFact]:
        prompt = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\nConversation:\n{conversation}\n\nJSON:"
        )
        raw = self.llm.complete(prompt, grammar=MEMORY_JSON_GBNF)
        return parse_facts(raw)

    def ingest(self, conversation: str) -> list[dict[str, Any]]:
        """
        Extract facts and write/merge them into the DB.
        Returns a list of action records for observability.
        """
        facts = self.extract_from_text(conversation)
        existing = self.db.list_active_entities()
        actions: list[dict[str, Any]] = []

        for fact in facts:
            if not fact.entity or not fact.attribute:
                continue

            emb = self._embed(
                self.db.fact_text(fact.entity, fact.attribute, fact.value)
            )
            dup_id = find_duplicate_entity(
                fact.normalized_entity(), existing, self.threshold
            )

            if dup_id is not None:
                # Merge into the matched entity row (update attribute/value).
                matched = self.db.get_memory(dup_id)
                # Prefer the canonical (existing) entity spelling on merge.
                canon_entity = matched["entity"] if matched else fact.entity
                # If the matched row is a different attribute, insert a sibling
                # under the canonical entity name rather than overwriting.
                if matched and matched["attribute"].lower() != fact.attribute.lower():
                    new_id = self.db.insert_memory(
                        entity=canon_entity,
                        attribute=fact.attribute,
                        value=fact.value,
                        confidence=fact.confidence,
                        embedding=emb,
                        is_permanent=fact.is_permanent,
                    )
                    existing.append((new_id, canon_entity))
                    actions.append(
                        {
                            "action": "insert_under_canonical",
                            "id": new_id,
                            "matched_entity_id": dup_id,
                            "entity": canon_entity,
                            "attribute": fact.attribute,
                            "value": fact.value,
                        }
                    )
                else:
                    self.db.update_memory(
                        dup_id,
                        entity=canon_entity,
                        attribute=fact.attribute,
                        value=fact.value,
                        confidence=fact.confidence,
                        embedding=emb,
                        is_permanent=fact.is_permanent,
                    )
                    actions.append(
                        {
                            "action": "merge",
                            "id": dup_id,
                            "entity": canon_entity,
                            "attribute": fact.attribute,
                            "value": fact.value,
                            "similarity": jaro_winkler(
                                fact.entity, canon_entity
                            ),
                        }
                    )
            else:
                new_id = self.db.insert_memory(
                    entity=fact.entity,
                    attribute=fact.attribute,
                    value=fact.value,
                    confidence=fact.confidence,
                    embedding=emb,
                    is_permanent=fact.is_permanent,
                )
                existing.append((new_id, fact.entity))
                actions.append(
                    {
                        "action": "insert",
                        "id": new_id,
                        "entity": fact.entity,
                        "attribute": fact.attribute,
                        "value": fact.value,
                    }
                )

        return actions
