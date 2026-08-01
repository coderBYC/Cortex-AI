#!/usr/bin/env python3
"""
Living local AI memory system (fastembed + llama-cpp GBNF).

Interactive REPL (default):
    python -m memory_engine.main
    python -m memory_engine.main --db ./memories.db

One-shot:
    python -m memory_engine.main remember "My name is Bryan. I live in Ann Arbor."
    python -m memory_engine.main ask "Where does Bryan live?"
    python -m memory_engine.main list

Scripted smoke demo (still uses real llama + fastembed):
    python -m memory_engine.main --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memory_engine.extraction import (
    JARO_WINKLER_THRESHOLD,
    MEMORY_JSON_GBNF,
    MemoryStack,
    resolve_model_path,
)
from memory_engine.retrieval import DEFAULT_LAMBDA


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_actions(actions: list[dict]) -> None:
    if not actions:
        print("  (no durable facts extracted)")
        return
    for a in actions:
        extra = f"  (sim={a['similarity']:.3f})" if "similarity" in a else ""
        print(
            f"  → {a['action']:24s} id={a['id']}  "
            f"{a['entity']}.{a['attribute']} = {a['value']}{extra}"
        )


def print_hits(hits: list) -> None:
    if not hits:
        print("  (no hits)")
        return
    for h in hits:
        print(
            f"  score={h.final_score:.6f}  "
            f"rrf={h.rrf:.5f} decay={h.decay:.4f}  "
            f"vec#{h.rank_vec} bm25#{h.rank_bm25}  "
            f"→ {h.entity}'s {h.attribute} is {h.value}"
        )


def print_stack_info(stack: MemoryStack) -> None:
    model = getattr(stack.llm, "model_path", None)
    print(f"DB path          : {stack.db.db_path}")
    print(f"JW threshold     : {JARO_WINKLER_THRESHOLD}")
    print(f"Decay λ (hrs)    : {DEFAULT_LAMBDA}")
    print(f"GBNF grammar     : {len(MEMORY_JSON_GBNF)} chars (JSON-constrained)")
    print(
        f"embeddings       : fastembed "
        f"({stack.embedder.model_name}, dim={stack.embedder.dim})"
    )
    print(f"extractor LLM    : {type(stack.llm).__name__}"
          + (f" ({model})" if model else ""))
    print(
        f"sqlite-vec       : "
        f"{'enabled' if stack.db.vec_enabled else 'numpy fallback'}"
    )


def run_repl(stack: MemoryStack) -> int:
    banner("Cortex — Local AI Memory (REPL)")
    print_stack_info(stack)
    print(
        "\nCommands:\n"
        "  remember <text>   ingest conversation → GBNF extract → SQLite\n"
        "  ask <query>       hybrid BM25 + vector recall\n"
        "  list              show active memories\n"
        "  help              show this help\n"
        "  quit              exit\n"
    )
    while True:
        try:
            line = input("cortex> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"quit", "exit", "q"}:
            break
        if line in {"help", "?"}:
            print(
                "  remember <text> | ask <query> | list | quit"
            )
            continue
        if line == "list":
            rows = stack.db.dump_active()
            if not rows:
                print("  (empty)")
            for row in rows:
                perm = "permanent" if row["is_permanent"] else "decaying"
                print(
                    f"  [{row['id']:3d}] {row['entity']:12s} "
                    f"{row['attribute']:24s} = {row['value']!r}  ({perm})"
                )
            continue

        if line.startswith("remember "):
            text = line[len("remember "):].strip()
            if not text:
                print("  usage: remember <conversation text>")
                continue
            print_actions(stack.remember(text))
            continue

        if line.startswith("ask "):
            query = line[len("ask "):].strip()
            if not query:
                print("  usage: ask <query>")
                continue
            print_hits(stack.recall(query, top_k=5))
            continue

        # Bare text defaults to remember (living conversation capture).
        print_actions(stack.remember(line))

    return 0


def run_oneshot(stack: MemoryStack, command: str, text: str) -> int:
    if command == "remember":
        print_actions(stack.remember(text))
        return 0
    if command == "ask":
        hits = stack.recall(text, top_k=5)
        print_hits(hits)
        print(json.dumps([h.as_dict() for h in hits], indent=2))
        return 0
    if command == "list":
        print(json.dumps(stack.db.dump_active(), indent=2))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


def run_demo(stack: MemoryStack) -> int:
    """Scripted smoke test — still real llama-cpp + fastembed, not mocks."""
    samples = [
        "Hi! My name is Bryan. I live in Ann Arbor and I'm a software engineer. "
        "I prefer dark mode editors and I love espresso.",
        "Just a reminder — Brian's occupation is staff engineer. "
        "Bryan's location is Ann Arbor, Michigan.",
        "Alice's birthday is March 3rd. Alice's dog is named Nimbus.",
    ]
    queries = [
        "Where does Bryan live?",
        "What does Bryan do for work?",
        "Tell me about Alice's dog",
        "editor preferences",
    ]

    banner("Cortex — Local AI Memory (demo)")
    print_stack_info(stack)

    banner("1. Deduplicating Ingestion (llama-cpp + GBNF)")
    for i, convo in enumerate(samples, 1):
        print(f"\n--- conversation {i} ---")
        print(f"  {convo[:100]}{'...' if len(convo) > 100 else ''}")
        print_actions(stack.remember(convo))

    banner("2. Active memories in SQLite")
    for row in stack.db.dump_active():
        perm = "permanent" if row["is_permanent"] else "decaying"
        print(
            f"  [{row['id']:3d}] {row['entity']:12s} "
            f"{row['attribute']:24s} = {row['value']!r:30s}  ({perm})"
        )

    banner("3. Hybrid RRF + Time-Decay Retrieval (fastembed vectors)")
    for q in queries:
        print(f"\n  query: {q!r}")
        print_hits(stack.recall(q, top_k=3))

    banner("4. Machine-readable top hit")
    top = stack.recall(queries[0], top_k=1)
    print(json.dumps([h.as_dict() for h in top], indent=2))
    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local-first AI memory (fastembed + llama-cpp GBNF)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("memories.db"),
        help="SQLite path (default: ./memories.db)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="GGUF path for llama-cpp extraction "
             "(default: $CORTEX_MODEL or ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run scripted smoke demo with real local models",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Allow MockLLM fallback (tests only — not for living use)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("remember", "ask", "list"),
        help="Optional one-shot command",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="",
        help="Text for remember/ask",
    )
    args = parser.parse_args(argv)

    resolved = resolve_model_path(args.model)
    if resolved is None and not args.mock:
        print(
            "No GGUF model found. Download one or pass --model.\n"
            f"  Expected: {Path.home() / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'}",
            file=sys.stderr,
        )
        return 1

    with MemoryStack(
        args.db,
        model_path=args.model,
        allow_mock=args.mock,
    ) as stack:
        if args.demo:
            return run_demo(stack)
        if args.command:
            if args.command in {"remember", "ask"} and not args.text:
                print(f"usage: ... {args.command} <text>", file=sys.stderr)
                return 2
            return run_oneshot(stack, args.command, args.text)
        return run_repl(stack)


if __name__ == "__main__":
    sys.exit(main())
