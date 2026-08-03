#!/usr/bin/env python3
"""
LoCoMo benchmark runner for Cortex-AI (local memory engine).

Pipeline
--------
1. Ingest each LoCoMo dialog turn into Cortex (fastembed vectors + FTS5),
   stamped with the session datetime so exponential time-decay applies.
2. Optionally run llama-cpp GBNF fact extraction on each turn (--extract).
3. For every QA item: hybrid retrieve (BM25 ∥ vectors → RRF × e^{-λΔt}),
   then answer from the retrieved memories (local llama or extractive).
4. Score with LoCoMo-style token F1, broken out by category:

   Cat 1  multi_hop     — RRF should surface related facts across sessions
   Cat 2  temporal      — time-decay should prefer fresher / dated facts
   Cat 3  open_domain   — world-ish inference over stored context
   Cat 4  single_hop    — BM25 should nail exact names / IDs / phrases
   Cat 5  adversarial   — should abstain ("no information available")

Usage
-----
  # Smoke test (1 conversation, 40 questions)
  .venv/bin/python -m evals.run_locomo --max-samples 1 --max-qa 40

  # Full 10-conversation benchmark (slow with --extract)
  .venv/bin/python -m evals.run_locomo --out results/locomo_cortex.json

  # GBNF extraction on every turn (much slower, fuller Cortex path)
  .venv/bin/python -m evals.run_locomo --extract --max-samples 1 --max-qa 20
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory_engine.db import MemoryDB
from memory_engine.extraction import (
    MEMORY_JSON_GBNF,
    LlamaCppBackend,
    LocalEmbedder,
    MemoryExtractor,
    create_llm,
    get_local_embedder,
    resolve_model_path,
)
from memory_engine.retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# LoCoMo category map (paper + evaluation.py comments)
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

WATCH_CATEGORIES = ("temporal", "single_hop", "multi_hop")

# ---------------------------------------------------------------------------
# LoCoMo-compatible scoring (ported from locomo/task_eval/evaluation.py)
# ---------------------------------------------------------------------------

try:
    from nltk.stem import PorterStemmer

    _STEMMER = PorterStemmer()
except Exception:  # pragma: no cover
    _STEMMER = None


def _normalize_answer(s: str) -> str:
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _stem_tokens(text: str) -> list[str]:
    tokens = _normalize_answer(text).split()
    if _STEMMER is None:
        return tokens
    return [_STEMMER.stem(w) for w in tokens]


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = _stem_tokens(prediction)
    gold = _stem_tokens(ground_truth)
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    golds = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not golds:
        return 0.0
    return float(
        np.mean(
            [max(f1_score(p, g) for p in preds) for g in golds]
        )
    )


def score_qa(category: int, prediction: str, answer: Any) -> float:
    """Return LoCoMo-compatible accuracy contribution for one QA."""
    pred = (prediction or "").strip()
    if category == 5:
        low = pred.lower()
        if (
            "no information available" in low
            or "not mentioned" in low
            or "no information" in low
        ):
            return 1.0
        return 0.0

    gold = "" if answer is None else str(answer)
    if category == 3:
        gold = gold.split(";")[0].strip()

    if category == 1:
        return multi_answer_f1(pred, gold)
    if category in (2, 3, 4):
        return f1_score(pred, gold)
    return 0.0


# ---------------------------------------------------------------------------
# Session datetime parsing → ISO (feeds time-decay)
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %B %Y",
    "%H:%M on %d %B, %Y",
    "%d %B, %Y",
    "%B %d, %Y",
)


def parse_session_datetime(raw: Optional[str]) -> str:
    """Parse LoCoMo session timestamps like '1:56 pm on 8 May, 2023' → ISO UTC."""
    if not raw:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    text = unicodedata.normalize("NFKC", raw.strip())
    text = text.replace("  ", " ")
    # Normalize am/pm casing for %p
    text = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), text, flags=re.I)
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    # Fallback: keep a stable but non-parsed stamp
    return datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def iter_sessions(conversation: dict[str, Any]) -> list[tuple[int, str, list[dict]]]:
    """Return [(session_num, iso_created_at, turns), ...] in chronological order."""
    sessions: list[tuple[int, str, list[dict]]] = []
    for key, value in conversation.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if not m or not isinstance(value, list) or not value:
            continue
        num = int(m.group(1))
        raw_dt = conversation.get(f"session_{num}_date_time", "")
        sessions.append((num, parse_session_datetime(raw_dt), value))
    sessions.sort(key=lambda x: x[0])
    return sessions


def turn_value(turn: dict[str, Any]) -> str:
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        return f"{text} [image: {caption}]".strip()
    return text


def build_parent_snippet(
    sess_num: int,
    raw_dt: str,
    turns: list[dict[str, Any]],
    center_idx: int,
    *,
    window: int = 2,
) -> str:
    """
    Parent session snippet around a child turn.

    Example:
      [Session 4 - 1:56 pm on 8 May, 2023]
      Caroline: I just signed my lease in SF!
      Melanie: Wow, so you're officially leaving NYC?
    """
    lo = max(0, center_idx - window)
    hi = min(len(turns), center_idx + window + 1)
    header = f"[Session {sess_num}"
    if raw_dt:
        header += f" - {raw_dt}"
    header += "]"
    lines = [header]
    for t in turns[lo:hi]:
        sp = (t.get("speaker") or "?").strip()
        tv = turn_value(t)
        if tv:
            lines.append(f"{sp}: {tv}")
    return "\n".join(lines)


def ingest_conversation(
    db: MemoryDB,
    embedder: LocalEmbedder,
    conversation: dict[str, Any],
    *,
    extractor: Optional[MemoryExtractor] = None,
    extract: bool = False,
    parent_window: int = 1,
) -> dict[str, int]:
    """
    Write every dialog turn into Cortex memories (parent–child layout).

    Child (FTS5 + vector): compact dated utterance for precise matching.
    Parent (prompt only): ±window neighboring turns in the same session.

    Session timestamps are embedded into the child text so temporal
    questions can hit BM25 / vectors; created_at drives e^{-λΔt}.
    """
    stats = {"turns": 0, "memories": 0, "extracted": 0}
    for sess_num, created_at, turns in iter_sessions(conversation):
        raw_dt = conversation.get(f"session_{sess_num}_date_time", "") or ""
        for i, turn in enumerate(turns):
            speaker = (turn.get("speaker") or "unknown").strip()
            dia_id = (turn.get("dia_id") or f"S{sess_num}").strip()
            value = turn_value(turn)
            if not value:
                continue

            # Child fact — what gets embedded / BM25-indexed.
            dated_value = f"(on {raw_dt}) {value}" if raw_dt else value
            index_text = (
                f"{speaker} said on {raw_dt}: {value}"
                if raw_dt
                else f"{speaker}: {value}"
            )
            parent = build_parent_snippet(
                sess_num, raw_dt, turns, i, window=parent_window
            )

            emb = embedder.embed(index_text, dim=db.dim)
            db.insert_memory(
                entity=speaker,
                attribute=dia_id,
                value=dated_value,
                confidence=1.0,
                embedding=emb,
                is_permanent=False,
                created_at=created_at,
                parent_context=parent,
            )
            stats["turns"] += 1
            stats["memories"] += 1

            if extract and extractor is not None:
                before = {r["id"] for r in db.dump_active()}
                actions = extractor.ingest(
                    f"[{raw_dt}] {speaker}: {value}" if raw_dt else f"{speaker}: {value}"
                )
                stats["extracted"] += len(actions)
                after_ids = {r["id"] for r in db.dump_active()} - before
                for mid in after_ids:
                    db.conn.execute(
                        "UPDATE memories_meta SET created_at = ?, parent_context = ? WHERE id = ?",
                        (created_at, parent, mid),
                    )
                if after_ids:
                    db.conn.commit()
                    stats["memories"] += len(after_ids)
    return stats


def select_qa(
    qa_list: list[dict[str, Any]],
    *,
    categories: Optional[list[int]],
    max_qa: Optional[int],
    balanced: bool,
) -> list[dict[str, Any]]:
    """Filter / subsample QA items, optionally balancing across categories."""
    items = list(qa_list)
    if categories:
        want = set(categories)
        items = [q for q in items if q.get("category") in want]
    if max_qa is None or max_qa <= 0 or len(items) <= max_qa:
        return items
    if not balanced:
        return items[:max_qa]

    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for q in items:
        by_cat[int(q["category"])].append(q)
    cats = sorted(by_cat.keys())
    if not cats:
        return []
    # Round-robin until max_qa
    out: list[dict[str, Any]] = []
    idx = {c: 0 for c in cats}
    while len(out) < max_qa:
        progressed = False
        for c in cats:
            i = idx[c]
            if i < len(by_cat[c]):
                out.append(by_cat[c][i])
                idx[c] = i + 1
                progressed = True
                if len(out) >= max_qa:
                    break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

ANSWER_GBNF = r"""
root ::= answer
answer ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt]
"""


def format_context(hits: list[Any], limit: int = 8, *, max_chars: int = 3500) -> str:
    """Prefer parent session snippets when present (parent–child expansion)."""
    lines = []
    used = 0
    for i, h in enumerate(hits[:limit], 1):
        body = h.prompt_text() if hasattr(h, "prompt_text") else h.value
        # Keep individual snippets bounded so local models stay under n_ctx.
        if len(body) > 700:
            body = body[:700].rstrip() + "…"
        block = f"[{i}] (score={h.final_score:.4f}, {h.created_at})\n{body}"
        if used + len(block) > max_chars and lines:
            break
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines) if lines else "(no memories retrieved)"


def answer_extractive(hits: list[Any]) -> str:
    """Fallback: return top parent/child text concatenated."""
    if not hits:
        return "no information available"
    tops = hits[:3]
    parts = []
    for h in tops:
        if getattr(h, "parent_context", None):
            parts.append(h.parent_context)
        else:
            parts.append(h.value)
    return " | ".join(parts)


def answer_prompt_messages(
    question: str, hits: list[Any], *, category: int
) -> list[dict[str, str]]:
    context = format_context(hits)
    abstain_hint = (
        'If the answer is not in the memories, reply exactly: "no information available".'
        if category == 5
        else (
            "Answer with a few words. Prefer exact phrases, names, and dates "
            "from the memories when possible."
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the retrieved memories below. "
                "Reply with a short answer only — no preamble. "
                + abstain_hint
            ),
        },
        {
            "role": "user",
            "content": (
                f"Memories:\n{context}\n\n"
                f"Question: {question}\n"
                f"Short answer:"
            ),
        },
    ]


def answer_with_llama(
    llm: LlamaCppBackend,
    question: str,
    hits: list[Any],
    *,
    category: int,
) -> str:
    messages = answer_prompt_messages(question, hits, category=category)
    try:
        out = llm.llm.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=0.0,
        )
        text = out["choices"][0]["message"]["content"].strip()
    except Exception:
        context = format_context(hits)
        prompt = (
            f"Memories:\n{context}\n\nQuestion: {question}\nShort answer:"
        )
        text = llm.complete(prompt).strip()

    text = text.strip().strip('"').strip("'")
    if not text:
        return answer_extractive(hits)
    return text


def answer_with_openai(
    client: Any,
    model: str,
    question: str,
    hits: list[Any],
    *,
    category: int,
) -> str:
    """Same retrieval context as local llama — isolates answer-model quality."""
    messages = answer_prompt_messages(question, hits, category=category)
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )
    text = (out.choices[0].message.content or "").strip()
    text = text.strip().strip('"').strip("'")
    if not text:
        return answer_extractive(hits)
    return text


def answer_question(
    retriever: HybridRetriever,
    question: str,
    *,
    category: int,
    top_k: int,
    llm: Optional[LlamaCppBackend] = None,
    openai_client: Any = None,
    openai_model: Optional[str] = None,
) -> tuple[str, list[dict[str, Any]]]:
    hits = retriever.retrieve(question, top_k=top_k)
    hit_dicts = [h.as_dict() for h in hits]

    if openai_client is not None and openai_model:
        pred = answer_with_openai(
            openai_client, openai_model, question, hits, category=category
        )
    elif isinstance(llm, LlamaCppBackend):
        pred = answer_with_llama(llm, question, hits, category=category)
    else:
        pred = answer_extractive(hits)
        if category == 5 and not hits:
            pred = "no information available"
    return pred, hit_dicts


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        name = CATEGORY_NAMES.get(r["category"], str(r["category"]))
        by_cat[name].append(float(r["f1"]))

    summary = {
        "n": len(rows),
        "overall_f1": float(np.mean([r["f1"] for r in rows])) if rows else 0.0,
        "by_category": {},
    }
    for name, scores in sorted(by_cat.items()):
        summary["by_category"][name] = {
            "n": len(scores),
            "f1": float(np.mean(scores)),
        }
    return summary


def print_watch_report(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("  Cortex-AI × LoCoMo — category watchlist")
    print("=" * 64)
    print(f"  Overall F1 ({summary['n']} QA): {summary['overall_f1']:.4f}")
    print()
    watch_notes = {
        "temporal": (
            "Time-decay e^{-λΔt} should prefer fresher / dated facts "
            "(Mem0 temporal ≈ 55.5% is the reference bar)."
        ),
        "single_hop": (
            "FTS5 BM25 should catch exact names, IDs, and phrases that "
            "pure vector search often misses."
        ),
        "multi_hop": (
            "RRF should fuse BM25 + vector ranks across sessions so related "
            "facts surface without dumping the full conversation."
        ),
    }
    by = summary.get("by_category", {})
    for name in WATCH_CATEGORIES:
        block = by.get(name)
        if not block:
            print(f"  • {name:12s}  (no items in this run)")
            continue
        print(
            f"  • {name:12s}  F1={block['f1']:.4f}  n={block['n']}"
        )
        print(f"      → {watch_notes[name]}")
    # Also print the rest briefly
    for name, block in by.items():
        if name in WATCH_CATEGORIES:
            continue
        print(f"  • {name:12s}  F1={block['f1']:.4f}  n={block['n']}")
    print("=" * 64)


def run_benchmark(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"LoCoMo data not found: {data_path}", file=sys.stderr)
        return 1

    samples = json.load(open(data_path))
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
    if args.max_samples:
        samples = samples[: args.max_samples]

    model_path = resolve_model_path(args.model)
    use_openai = bool(args.openai_model)
    need_local_llm = args.extract or (args.llm_answer and not use_openai)
    if need_local_llm and model_path is None:
        print(
            "No GGUF model found (needed for --extract / --llm-answer). "
            "Pass --model or place Llama at ~/models/...",
            file=sys.stderr,
        )
        return 1

    print("Loading fastembed…")
    embedder = get_local_embedder()
    llm: Optional[LlamaCppBackend] = None
    openai_client = None
    if use_openai:
        import os

        from openai import OpenAI

        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "OpenAI key required: set OPENAI_API_KEY or pass --openai-api-key",
                file=sys.stderr,
            )
            return 1
        openai_client = OpenAI(api_key=api_key)
        print(f"Answer model     : OpenAI {args.openai_model}")
        print("Retrieval        : Cortex (fastembed + FTS5 BM25 + RRF + decay)")
    if need_local_llm:
        print(f"Loading llama-cpp GGUF: {model_path}")
        backend = create_llm(model_path)
        if not isinstance(backend, LlamaCppBackend):
            print("Expected LlamaCppBackend", file=sys.stderr)
            return 1
        llm = backend
        if not use_openai and args.llm_answer:
            print("Answer model     : local LlamaCppBackend")
            print("Retrieval        : Cortex (fastembed + FTS5 BM25 + RRF + decay)")

    out_rows: list[dict[str, Any]] = []
    per_sample_out: list[dict[str, Any]] = []

    for sample in samples:
        sid = sample["sample_id"]
        db_path = Path(args.db_dir) / f"locomo_{sid}.db"
        if db_path.exists() and args.overwrite:
            for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
                if p.exists():
                    p.unlink()

        print(f"\n=== {sid} ===")
        db = MemoryDB(db_path, dim=embedder.dim)
        extractor = None
        if args.extract and llm is not None:
            extractor = MemoryExtractor(db, llm, embedder=embedder)

        if args.overwrite or db.conn.execute(
            "SELECT COUNT(*) AS c FROM memories_meta"
        ).fetchone()["c"] == 0:
            stats = ingest_conversation(
                db,
                embedder,
                sample["conversation"],
                extractor=extractor,
                extract=args.extract,
            )
            print(
                f"  ingested turns={stats['turns']} "
                f"memories={stats['memories']} extracted={stats['extracted']}"
            )
        else:
            n = db.conn.execute(
                "SELECT COUNT(*) AS c FROM memories_meta"
            ).fetchone()["c"]
            print(f"  reusing existing DB ({n} memories) at {db_path}")

        retriever = HybridRetriever(
            db,
            embed_fn=lambda t, _e=embedder, _d=db: _e.embed(t, _d.dim),
            require_fastembed=True,
        )

        qa_list = select_qa(
            sample["qa"],
            categories=args.categories,
            max_qa=args.max_qa,
            balanced=args.balanced,
        )

        sample_qa_out = []
        for i, qa in enumerate(qa_list, 1):
            q = qa["question"]
            cat = int(qa["category"])
            gold = qa.get("answer")
            pred, hits = answer_question(
                retriever,
                q,
                category=cat,
                top_k=args.top_k,
                llm=llm if (args.llm_answer and not use_openai) else None,
                openai_client=openai_client if use_openai else None,
                openai_model=args.openai_model if use_openai else None,
            )
            f1 = score_qa(cat, pred, gold)
            row = {
                "sample_id": sid,
                "question": q,
                "answer": gold,
                "prediction": pred,
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, str(cat)),
                "f1": round(f1, 4),
                "evidence": qa.get("evidence", []),
                "retrieved": [
                    {
                        "entity": h["entity"],
                        "attribute": h["attribute"],
                        "value": h["value"][:160],
                        "parent_context": (h.get("parent_context") or "")[:240],
                        "final_score": h["final_score"],
                        "rank_bm25": h["rank_bm25"],
                        "rank_vec": h["rank_vec"],
                        "decay": h["decay"],
                    }
                    for h in hits[: args.top_k]
                ],
            }
            out_rows.append(row)
            sample_qa_out.append(row)
            if args.verbose or i % 10 == 0 or i == len(qa_list):
                print(
                    f"  [{i}/{len(qa_list)}] cat={CATEGORY_NAMES.get(cat)} "
                    f"f1={f1:.3f} | {q[:60]}"
                )

        per_sample_out.append({"sample_id": sid, "qa": sample_qa_out})
        db.close()

    summary = aggregate_scores(out_rows)
    print_watch_report(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": str(data_path),
            "top_k": args.top_k,
            "extract": args.extract,
            "llm_answer": args.llm_answer,
            "openai_model": args.openai_model,
            "max_samples": args.max_samples,
            "max_qa": args.max_qa,
            "model": str(model_path) if model_path else None,
            "embed_model": embedder.model_name,
            "gbnf_chars": len(MEMORY_JSON_GBNF),
            "answer_backend": (
                f"openai:{args.openai_model}"
                if use_openai
                else ("llama-cpp" if args.llm_answer else "extractive")
            ),
        },
        "summary": summary,
        "samples": per_sample_out,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LoCoMo benchmark on Cortex-AI")
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "locomo" / "data" / "locomo10.json",
        help="Path to locomo10.json",
    )
    p.add_argument(
        "--db-dir",
        type=Path,
        default=ROOT / "results" / "locomo_dbs",
        help="Directory for per-conversation SQLite files",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "locomo_cortex.json",
        help="Results JSON path",
    )
    p.add_argument("--model", type=Path, default=None, help="GGUF path")
    p.add_argument("--max-samples", type=int, default=None, help="Limit conversations")
    p.add_argument("--sample-id", type=str, default=None, help="Run a single sample_id")
    p.add_argument("--max-qa", type=int, default=None, help="Limit QA per conversation")
    p.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate these category ids (1=multi_hop 2=temporal 4=single_hop)",
    )
    p.add_argument("--top-k", type=int, default=5, help="Retrieval depth")
    p.add_argument(
        "--extract",
        action="store_true",
        help="Run llama-cpp GBNF extraction on each turn (slow)",
    )
    p.add_argument(
        "--llm-answer",
        action="store_true",
        help="Generate answers with local llama over retrieved memories",
    )
    p.add_argument(
        "--openai-model",
        type=str,
        default=None,
        help="Answer with OpenAI model over Cortex retrieval "
             "(e.g. gpt-4o-mini). Isolates answer quality vs retrieval.",
    )
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (prefer OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild conversation DBs from scratch",
    )
    p.add_argument(
        "--balanced",
        action="store_true",
        help="When --max-qa is set, round-robin across categories "
             "(ensures temporal / single_hop / multi_hop all appear)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.db_dir.mkdir(parents=True, exist_ok=True)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
