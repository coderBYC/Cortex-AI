#!/usr/bin/env python3
"""
Runnable Phase 1 demonstration of the local-first memory engine.

Usage:
    python -m memory_engine.main
    python -m memory_engine.main --db /tmp/demo_memories.db
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from memory_engine.db import MemoryDB
from memory_engine.extraction import (
    JARO_WINKLER_THRESHOLD,
    MEMORY_JSON_GBNF,
    MemoryExtractor,
    MockLLM,
    jaro_winkler,
)
from memory_engine.retrieval import DEFAULT_LAMBDA, HybridRetriever


DEMO_CONVERSATIONS = [
    "Hi! My name is Bryan. I live in Ann Arbor and I'm a software engineer. "
    "I prefer dark mode editors and I love espresso.",
    # "Brian" ≈ "Bryan" (Jaro-Winkler > 0.88) → merge into existing entity.
    "Just a reminder — Brian's occupation is staff engineer. "
    "Bryan's location is Ann Arbor, Michigan.",
    "Alice's birthday is March 3rd. Alice's dog is named Nimbus.",
]


DEMO_QUERIES = [
    "Where does Bryan live?",
    "What does Bryan do for work?",
    "Tell me about Alice's dog",
    "editor preferences",
]


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def run_demo(db_path: Path, *, keep_db: bool = False) -> int:
    banner("Local-First Memory Engine — Phase 1 Demo")
    print(f"DB path          : {db_path}")
    print(f"JW threshold     : {JARO_WINKLER_THRESHOLD}")
    print(f"Decay λ (hrs)    : {DEFAULT_LAMBDA}")
    print(f"GBNF grammar     : {len(MEMORY_JSON_GBNF)} chars (JSON-constrained)")

    with MemoryDB(db_path) as db:
        print(f"sqlite-vec       : {'enabled' if db.vec_enabled else 'numpy fallback'}")

        llm = MockLLM()
        extractor = MemoryExtractor(db, llm)
        retriever = HybridRetriever(db)

        # --- Ingestion -------------------------------------------------------
        banner("1. Deduplicating Ingestion")
        for i, convo in enumerate(DEMO_CONVERSATIONS, 1):
            print(f"\n--- conversation {i} ---")
            print(f"  {convo[:100]}{'...' if len(convo) > 100 else ''}")
            actions = extractor.ingest(convo)
            for a in actions:
                print(f"  → {a['action']:24s} id={a['id']}  "
                      f"{a['entity']}.{a['attribute']} = {a['value']}"
                      + (f"  (sim={a['similarity']:.3f})" if "similarity" in a else ""))

        # Show that Bryan ≈ Brian was merged via Jaro-Winkler
        sim = jaro_winkler("Bryan", "Brian")
        print(f"\n  Jaro-Winkler('Bryan','Brian') = {sim:.4f} "
              f"(threshold {JARO_WINKLER_THRESHOLD} → "
              f"{'MERGE' if sim > JARO_WINKLER_THRESHOLD else 'INSERT'})")

        banner("2. Active memories in SQLite")
        for row in db.dump_active():
            perm = "permanent" if row["is_permanent"] else "decaying"
            print(
                f"  [{row['id']:3d}] {row['entity']:12s} "
                f"{row['attribute']:24s} = {row['value']!r:30s}  ({perm})"
            )

        # --- Retrieval -------------------------------------------------------
        banner("3. Hybrid RRF + Time-Decay Retrieval")
        for q in DEMO_QUERIES:
            print(f"\n  query: {q!r}")
            hits = retriever.retrieve(q, top_k=3)
            if not hits:
                print("    (no hits)")
                continue
            for h in hits:
                print(
                    f"    score={h.final_score:.6f}  "
                    f"rrf={h.rrf:.5f} decay={h.decay:.4f}  "
                    f"vec#{h.rank_vec} bm25#{h.rank_bm25}  "
                    f"→ {h.entity}'s {h.attribute} is {h.value}"
                )

        banner("4. Machine-readable top hit for first query")
        top = retriever.retrieve(DEMO_QUERIES[0], top_k=1)
        print(json.dumps([h.as_dict() for h in top], indent=2))

    if not keep_db and db_path.exists() and str(db_path).startswith(tempfile.gettempdir()):
        db_path.unlink(missing_ok=True)
        # sqlite WAL sidecars
        Path(str(db_path) + "-wal").unlink(missing_ok=True)
        Path(str(db_path) + "-shm").unlink(missing_ok=True)

    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 local memory engine demo")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: temp file, deleted after run)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the DB file after the demo finishes",
    )
    args = parser.parse_args(argv)

    if args.db is None:
        tmp = tempfile.NamedTemporaryFile(prefix="memories_", suffix=".db", delete=False)
        db_path = Path(tmp.name)
        tmp.close()
        keep = args.keep
    else:
        db_path = args.db
        keep = True

    return run_demo(db_path, keep_db=keep)


if __name__ == "__main__":
    sys.exit(main())
