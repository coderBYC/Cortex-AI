#!/usr/bin/env python3
"""
Cortex-AI — knowledge-layer memory CLI.

Interactive REPL (default):
    python -m memory_engine.main
    python -m memory_engine.main --db ./memories.db --mock

One-shot:
    python -m memory_engine.main remember "I used to study at NTU..."
    python -m memory_engine.main ask "Where do I study now?"
    python -m memory_engine.main list

Scripted smoke demo:
    python -m memory_engine.main --demo --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memory_engine.cortex import CortexEngine, IngestResult
from memory_engine.extraction import resolve_model_path


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_ingest(result: IngestResult) -> None:
    print(f"  raw memory     : {result.memory_id} @ {result.timestamp}")
    if result.mentions:
        print("  mentions:")
        for m in result.mentions:
            print(f"    - {m['mention']!r} ({m['type']})")
    if result.resolved:
        print("  resolved:")
        for r in result.resolved:
            print(
                f"    - {r['mention']!r} → {r['entity_id']} "
                f"[{r['canonical_name']}] via {r['method']} ({r['score']:.2f})"
            )
    if result.facts:
        print("  facts:")
        for f in result.facts:
            print(
                f"    - ({f['subject']}, {f['predicate']}, {f['object']})"
                + (f" [{f['time']}]" if f.get("time") else "")
            )
    if result.events:
        print("  events:")
        for e in result.events:
            print(
                f"    - {e['event_type']}: {e.get('from')} → {e.get('to')}"
                + (f" ({e['year']})" if e.get("year") else "")
            )
    if result.actions:
        print("  state actions:")
        for a in result.actions:
            print(f"    → {a.get('action')}: {a}")


def print_engine_info(engine: CortexEngine) -> None:
    model = getattr(engine.llm, "model_path", None)
    print(f"DB path          : {engine.db.db_path}")
    print(f"embeddings       : {engine.embedder.model_name} (dim={engine.embedder.dim})")
    print(
        f"extractor LLM    : {type(engine.llm).__name__}"
        + (f" ({model})" if model else "")
    )
    print(f"pipeline         : raw → mention → resolve → fact/event → state")
    print(f"query            : intent → context → answer")


def run_repl(engine: CortexEngine) -> int:
    banner("Cortex — Knowledge Layer (REPL)")
    print_engine_info(engine)
    print(
        "\nCommands:\n"
        "  remember <text>   ingest → mentions → entities → facts/events → state\n"
        "  ask <query>       intent router → context → answer\n"
        "  list              dump states / facts / events / raw\n"
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
            print("  remember <text> | ask <query> | list | quit")
            continue
        if line == "list":
            _print_list(engine)
            continue
        if line.startswith("remember "):
            text = line[len("remember ") :].strip()
            if not text:
                print("  usage: remember <text>")
                continue
            print_ingest(engine.remember(text))
            continue
        if line.startswith("ask "):
            query = line[len("ask ") :].strip()
            if not query:
                print("  usage: ask <query>")
                continue
            result = engine.ask(query)
            print(f"  intent : {result.intent}")
            print(f"  answer : {result.answer}")
            if result.context.blocks:
                print("  context:")
                for b in result.context.blocks:
                    print(f"    {b}")
            continue
        print_ingest(engine.remember(line))
    return 0


def _print_list(engine: CortexEngine) -> None:
    summary = engine.summary()
    print(f"  summary: {summary}")
    print("  states:")
    for s in engine.db.list_states():
        print(f"    {s['subject']}.{s['key']} = {s['value']}")
    print("  active facts:")
    for f in engine.db.active_facts():
        print(
            f"    ({f['subject']}, {f['predicate']}, {f['object']})"
            + (f" [{f['time_expr']}]" if f.get("time_expr") else "")
        )
    print("  events:")
    for e in engine.db.list_events(limit=20):
        print(f"    {e['event_type']} {e.get('payload')}")
    print("  raw memories:")
    for m in engine.db.list_memories(limit=20):
        print(f"    [{m['id']}] {m['text'][:80]}")


def run_oneshot(engine: CortexEngine, command: str, text: str) -> int:
    if command == "remember":
        result = engine.remember(text)
        print_ingest(result)
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if command == "ask":
        result = engine.ask(text)
        print(f"intent: {result.intent}")
        print(f"answer: {result.answer}")
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0
    if command == "list":
        _print_list(engine)
        print(json.dumps(engine.summary(), indent=2))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


def run_demo(engine: CortexEngine) -> int:
    sample = (
        "I used to study at NTU in Taiwan. "
        "In 2026 I transferred to the University of Michigan "
        "for Mechanical Engineering. I use PyTorch for robotics."
    )
    queries = [
        "Where do I study now?",
        "Did I transfer schools?",
        "What is my major?",
        "What software do I use?",
    ]

    banner("Cortex — Knowledge Layer Demo")
    print_engine_info(engine)

    banner("1. Ingest (raw → mention → resolve → fact/event → state)")
    print(f"  input: {sample}")
    print_ingest(engine.remember(sample, timestamp="2026-08-03T00:00:00+00:00"))

    banner("2. Knowledge layer snapshot")
    _print_list(engine)

    banner("3. Query (intent → context → answer)")
    for q in queries:
        result = engine.ask(q)
        print(f"\n  Q: {q}")
        print(f"  intent: {result.intent}")
        print(f"  A: {result.answer}")

    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cortex-AI knowledge-layer memory (local LLM + SQLite)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("cortex.db"),
        help="SQLite path (default: ./cortex.db)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="GGUF path (default: $CORTEX_MODEL or ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run NTU→Michigan knowledge-layer smoke demo",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Allow MockLLM + heuristic extractors (tests / offline)",
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
            "No GGUF model found. Download one or pass --model / --mock.\n"
            f"  Expected: {Path.home() / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'}",
            file=sys.stderr,
        )
        return 1

    with CortexEngine(
        args.db,
        model_path=args.model,
        allow_mock=args.mock,
    ) as engine:
        if args.demo:
            return run_demo(engine)
        if args.command:
            if args.command in {"remember", "ask"} and not args.text:
                print(f"usage: ... {args.command} <text>", file=sys.stderr)
                return 2
            return run_oneshot(engine, args.command, args.text)
        return run_repl(engine)


if __name__ == "__main__":
    sys.exit(main())
