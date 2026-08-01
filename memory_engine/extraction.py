"""GBNF grammar, local llama.cpp extraction, real embeddings, Jaro-Winkler dedup."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import jellyfish
import numpy as np

from memory_engine.db import DEFAULT_DIM, MemoryDB

# Spec: merge when Jaro-Winkler similarity > 0.88
JARO_WINKLER_THRESHOLD = 0.88

# Default local embedding model (384-dim, ONNX via fastembed — no cloud).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# GBNF grammar — forces llama.cpp to emit a JSON array of memory objects.
# IMPORTANT: llama.cpp GBNF does NOT allow multiline RHS continuations; each
# rule must be on a single line or the parser reports "expecting name".
# Passed to llama-cpp-python as LlamaGrammar.from_string(MEMORY_JSON_GBNF).
# ---------------------------------------------------------------------------

MEMORY_JSON_GBNF = r"""
root ::= "[" ws (memory ("," ws memory)*)? ws "]"
memory ::= "{" ws "\"" "entity" "\"" ws ":" ws string "," ws "\"" "attribute" "\"" ws ":" ws string "," ws "\"" "value" "\"" ws ":" ws string "," ws "\"" "confidence" "\"" ws ":" ws number "," ws "\"" "is_permanent" "\"" ws ":" ws boolean ws "}"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= ("0" | "1") ("." [0-9]+)?
boolean ::= "true" | "false"
ws ::= [ \t\n\r]*
"""

# Equivalent JSON Schema (optional path via LlamaGrammar.from_json_schema).
MEMORY_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "attribute": {"type": "string"},
            "value": {"type": "string"},
            "confidence": {"type": "number"},
            "is_permanent": {"type": "boolean"},
        },
        "required": [
            "entity",
            "attribute",
            "value",
            "confidence",
            "is_permanent",
        ],
    },
}

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable personal memory facts from a conversation.
Return ONLY a JSON array. Each element must have:
  entity (string), attribute (string), value (string),
  confidence (0-1 number), is_permanent (boolean).
Skip greetings, ephemeral chit-chat, and anything not worth remembering.
Prefer canonical entity names (e.g. "Bryan" not "he").
If there are no memorable facts, return [].
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
    """Minimal interface so MockLLM and LlamaCppBackend share a call site."""

    def complete(self, prompt: str, *, grammar: Optional[str] = None) -> str: ...


class Embedder(Protocol):
    """Local embedding interface (fastembed or llama.cpp)."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str, dim: Optional[int] = None) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Real local embeddings (replaces hashing-trick mock vectors)
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    ONNX sentence embeddings via fastembed — fully local, zero API calls.
    Default model: BAAI/bge-small-en-v1.5 (384-dim).
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe dimensionality once.
        probe = next(self._model.embed(["dimension probe"]))
        self._dim = int(np.asarray(probe, dtype=np.float32).shape[0])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str, dim: Optional[int] = None) -> np.ndarray:
        target = dim or self._dim
        raw = next(self._model.embed([text]))
        vec = np.asarray(raw, dtype=np.float32).ravel()
        if vec.shape[0] == target:
            return _l2_normalize(vec)
        out = np.zeros(target, dtype=np.float32)
        n = min(target, vec.shape[0])
        out[:n] = vec[:n]
        return _l2_normalize(out)


_EMBEDDER_SINGLETON: Optional[LocalEmbedder] = None


def get_local_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> LocalEmbedder:
    """Process-wide LocalEmbedder (model load is relatively expensive)."""
    global _EMBEDDER_SINGLETON
    if (
        _EMBEDDER_SINGLETON is None
        or _EMBEDDER_SINGLETON.model_name != model_name
    ):
        _EMBEDDER_SINGLETON = LocalEmbedder(model_name=model_name)
    return _EMBEDDER_SINGLETON


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec


def hashing_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """
    Fallback bag-of-tokens hashing embedding when fastembed is unavailable.
    Prefer LocalEmbedder for production / demo quality.
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
    return _l2_normalize(vec)


# ---------------------------------------------------------------------------
# Mock LLM (regex) — kept for offline / no-GGUF runs
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Regex extractor for offline unit tests only (--mock).
    Production path uses LlamaCppBackend + MEMORY_JSON_GBNF.
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
        # grammar is ignored; MockLLM is not constrained by GBNF.
        _ = grammar
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


# ---------------------------------------------------------------------------
# llama-cpp-python + GBNF-constrained JSON extraction
# ---------------------------------------------------------------------------

class LlamaCppBackend:
    """
    Local GGUF inference via llama-cpp-python.

    JSON extraction always runs under MEMORY_JSON_GBNF so the model can only
    emit a schema-valid memory array (deterministic structure, no prose).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
    ) -> None:
        from llama_cpp import Llama, LlamaGrammar  # type: ignore

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"GGUF model not found: {path}\n"
                "  Pass a real .gguf path to --model, set CORTEX_MODEL, "
                "or place the model at ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf.\n"
                "  Example download (Hugging Face):\n"
                "    mkdir -p ~/models && cd ~/models &&\n"
                "    curl -L -o Llama-3.2-1B-Instruct-Q4_K_M.gguf \\\n"
                "      'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
                "/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'"
            )

        self.model_path = path
        self._LlamaGrammar = LlamaGrammar
        # Compile / bind the GBNF once; reuse on every extraction call.
        # (Actual parse happens on first generate; keep rules single-line.)
        self._json_grammar = LlamaGrammar.from_string(MEMORY_JSON_GBNF)

        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": n_ctx,
            "n_threads": n_threads or max(1, (os.cpu_count() or 4) // 2),
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
            "embedding": False,  # vectors come from LocalEmbedder / fastembed
        }
        # Prefer auto-detected chat template from GGUF metadata when present.
        if chat_format:
            kwargs["chat_format"] = chat_format

        self.llm = Llama(**kwargs)

    def _resolve_grammar(self, grammar: Optional[str] = None) -> Any:
        if grammar and grammar.strip() != MEMORY_JSON_GBNF.strip():
            return self._LlamaGrammar.from_string(grammar)
        return self._json_grammar

    def complete(self, prompt: str, *, grammar: Optional[str] = None) -> str:
        """
        Run constrained generation under GBNF.

        Prefer extract_memories() for Instruct models (uses chat template).
        """
        gbnf = self._resolve_grammar(grammar)
        out = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            grammar=gbnf,
            stop=["</s>", "<|eot_id|>", "<|im_end|>", "```"],
        )
        return out["choices"][0]["text"].strip()

    def extract_memories(self, conversation: str) -> str:
        """Chat-templated extraction with MEMORY_JSON_GBNF applied."""
        gbnf = self._json_grammar
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation:\n{conversation}\n\n"
                    "Return ONLY the JSON array of memory objects."
                ),
            },
        ]
        try:
            out = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.0,
                top_p=1.0,
                grammar=gbnf,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            # Fallback if chat_format is incompatible with this GGUF.
            prompt = (
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"Conversation:\n{conversation}\n\n"
                f"JSON:"
            )
            return self.complete(prompt, grammar=MEMORY_JSON_GBNF)


def create_llm(
    model_path: Optional[str | Path] = None,
    *,
    allow_mock: bool = False,
    **llama_kwargs: Any,
) -> LLMBackend:
    """
    Factory for the local extraction backend.

    Default: LlamaCppBackend (GGUF + GBNF). Pass allow_mock=True only for
    offline unit tests without a model file.
    """
    path = resolve_model_path(model_path)
    if path is not None:
        return LlamaCppBackend(path, **llama_kwargs)
    if allow_mock:
        return MockLLM()
    raise FileNotFoundError(
        "No GGUF model found. Set --model / CORTEX_MODEL, or place a model at "
        f"{DEFAULT_MODEL_CANDIDATES[0]}.\n"
        "Download example:\n"
        "  mkdir -p ~/models && curl -L -o "
        "~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \\\n"
        "    'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
        "/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'"
    )


# Default local GGUF search order (first existing file wins).
DEFAULT_MODEL_CANDIDATES = (
    Path.home() / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    Path.home() / "models" / "llama.gguf",
)


def resolve_model_path(model_path: Optional[str | Path] = None) -> Optional[Path]:
    """Resolve a GGUF path from arg, $CORTEX_MODEL, or well-known defaults."""
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env = os.environ.get("CORTEX_MODEL")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(DEFAULT_MODEL_CANDIDATES)
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def require_local_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> LocalEmbedder:
    """Load fastembed or raise a clear install error (no silent mock vectors)."""
    try:
        return get_local_embedder(model_name)
    except ImportError as exc:
        raise ImportError(
            "fastembed is required for local vector generation. "
            "Install with: pip install fastembed"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local embedder '{model_name}': {exc}"
        ) from exc


class MemoryStack:
    """
    Wired local memory system:
      fastembed vectors + llama-cpp GBNF extraction + SQLite hybrid retrieval.
    """

    def __init__(
        self,
        db_path: str | Path = "memories.db",
        *,
        model_path: Optional[str | Path] = None,
        allow_mock: bool = False,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        # Lazy import avoids extraction ↔ retrieval circular dependency.
        from memory_engine.retrieval import HybridRetriever

        self.embedder = require_local_embedder(embed_model)
        self.llm = create_llm(model_path, allow_mock=allow_mock)
        self.db = MemoryDB(db_path, dim=self.embedder.dim)
        self.extractor = MemoryExtractor(
            self.db,
            self.llm,
            embedder=self.embedder,
            require_fastembed=True,
        )
        self.retriever = HybridRetriever(
            self.db,
            embed_fn=lambda text: self.embedder.embed(text, self.db.dim),
            require_fastembed=True,
        )

    def remember(self, conversation: str) -> list[dict[str, Any]]:
        return self.extractor.ingest(conversation)

    def recall(self, query: str, *, top_k: int = 5) -> list[Any]:
        return self.retriever.retrieve(query, top_k=top_k)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MemoryStack":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Parsing + Jaro-Winkler dedup
# ---------------------------------------------------------------------------

def parse_facts(raw: str) -> list[ExtractedFact]:
    """Parse LLM JSON (possibly fenced) into ExtractedFact list."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

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
            conf = float(item.get("confidence", 1.0))
            conf = min(1.0, max(0.0, conf))
            facts.append(
                ExtractedFact(
                    entity=str(item["entity"]).strip(),
                    attribute=str(item["attribute"]).strip(),
                    value=str(item["value"]).strip(),
                    confidence=conf,
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


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """
    Deduplicating ingestion pipeline:
      conversation → llama-cpp (GBNF) JSON facts → Jaro-Winkler merge → SQLite
      with fastembed vectors written alongside each fact.
    """

    def __init__(
        self,
        db: MemoryDB,
        llm: Optional[LLMBackend] = None,
        *,
        embedder: Optional[Embedder] = None,
        threshold: float = JARO_WINKLER_THRESHOLD,
        require_fastembed: bool = True,
        allow_mock_llm: bool = False,
    ) -> None:
        self.db = db
        if llm is None:
            self.llm = create_llm(allow_mock=allow_mock_llm)
        else:
            self.llm = llm
        self.threshold = threshold

        if embedder is not None:
            self.embedder = embedder
        elif require_fastembed:
            self.embedder = require_local_embedder()
            if self.embedder.dim != self.db.dim:
                n = self.db.conn.execute(
                    "SELECT COUNT(*) AS c FROM memories_meta"
                ).fetchone()["c"]
                if int(n) == 0:
                    self.db.dim = self.embedder.dim
        else:
            self.embedder = None

    def _embed(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError(
                "No embedder configured. LocalEmbedder / fastembed is required."
            )
        return np.asarray(self.embedder.embed(text, self.db.dim), dtype=np.float32)

    def extract_from_text(self, conversation: str) -> list[ExtractedFact]:
        # Prefer chat-templated GBNF extraction on LlamaCppBackend.
        extract_fn = getattr(self.llm, "extract_memories", None)
        if callable(extract_fn):
            raw = extract_fn(conversation)
        else:
            prompt = (
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"Conversation:\n{conversation}\n\n"
                f"JSON:"
            )
            # Always pass the GBNF string — LlamaCppBackend applies it;
            # MockLLM (tests only) ignores it.
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
                matched = self.db.get_memory(dup_id)
                canon_entity = matched["entity"] if matched else fact.entity
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
