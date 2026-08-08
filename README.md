# Cortex-AI: Local-First Memory Layer

A lightweight local memory daemon for AI agents.

Cortex-AI stores and retrieves long-term memories on-device using:
- `SQLite` (`FTS5` lexical search + optional `sqlite-vec`)
- `fastembed` for local CPU embeddings
- `llama-cpp-python` with strict `GBNF` JSON extraction
- hybrid retrieval (`BM25` + vector) with `RRF` and exponential time-decay

## Why this exists

Most agent memory stacks either:
- push data to cloud vector DBs,
- lose precision on exact-string recall,
- or don't model memory freshness over time.

Cortex-AI is designed to be:
- **local-first** (no required cloud memory infra)
- **deterministic at extraction boundaries** (GBNF-constrained JSON)
- **hybrid and temporal** (BM25 + vector + decay)

## Core architecture

```text
USER INPUT
  → RAW MEMORY STORE (immutable memory_store)
  → ENTITY MENTION DETECTION (Local LLM + GBNF)
  → ENTITY RESOLUTION (Alias + Jaro-Winkler + FTS + Embedding)
  → FACT EXTRACTION + EVENT EXTRACTION
  → KNOWLEDGE LAYER (Entity Graph / Fact / Event / State)
  → State Transition Engine (conflict handling)

QUERY
  → Intent Router (State → SQL | Fact → table | Event → hybrid)
  → Context Builder
  → Local LLM → Answer
```

Entry point: `CortexEngine` in `memory_engine/cortex.py`.

### Write path

1. **Raw memory** — append-only `memory_store` (`id`, `text`, `timestamp`)
2. **Mentions** — `mentions.py` GBNF array of `{mention, type}`
3. **Resolve** — `resolve.py` blends exact/alias → JW (>0.88) → FTS → embedding → create
4. **Facts / events** — `extractors.py` SPO triples + life events (GBNF)
5. **State** — `state_engine.py` invalidates superseded stateful predicates (`studied_at`, `lives_in`, …)

### Query path

`query.py`: intent cues route to states / facts / events / raw memories, then answer head.

### Legacy Phase-1 path

`MemoryStack` + `HybridRetriever` (BM25 ∥ vector → RRF `k=15` × time-decay) remain for LoCoMo evals (`evals/run_locomo.py`).

## Repository layout

```text
memory_engine/
  cortex.py        # CortexEngine (remember + retrieve)
  knowledge_db.py  # memory_store / entities / facts / events / states
  mentions.py      # mention detection (GBNF + generic fallback)
  resolve.py       # entity resolution (injectable gazetteer)
  gazetteer.py     # optional alias-group loader
  extractors.py    # fact + event extraction
  state_engine.py  # conflict handling → current state
  query.py         # intent router + context builder
  main.py          # REPL + one-shot CLI
  db.py / retrieval.py / extraction.py  # legacy LoCoMo path

mcp_server/        # stdio MCP tools for any host model
tests/
  fixtures/        # scenario aliases + mention patterns (not prod)
  support.py       # make_test_engine()
evals/
  run_locomo.py
```

## Requirements

- Python `3.11+` (recommended)
- macOS/Linux
- enough RAM for chosen GGUF (7B models need significantly more than 1B)

Install:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Models

### Embedding model (local CPU)

Default: `BAAI/bge-small-en-v1.5` via `fastembed`

### Local extraction / answer model (GGUF)

Place a model at:

```text
~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

or pass `--model`.

Example download:

```bash
mkdir -p ~/models
cd ~/models
curl -L -o Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
```

## Usage

### Interactive REPL

```bash
.venv/bin/python -m memory_engine.main
# offline heuristics (no GGUF load):
.venv/bin/python -m memory_engine.main --mock
```

Commands:
- `remember <text>` — raw → mentions → resolve → facts/events → state
- `ask <query>` — intent → context → answer
- `list` — dump states / facts / events / raw memories
- `quit`

Bare text is treated as `remember`.

### One-shot

```bash
.venv/bin/python -m memory_engine.main remember \
  "I used to study at NTU in Taiwan. In 2026 I transferred to the University of Michigan."
.venv/bin/python -m memory_engine.main ask "Where do I study now?"
.venv/bin/python -m memory_engine.main list
```

### Scripted demo

```bash
.venv/bin/python -m memory_engine.main --demo --mock   # heuristics
.venv/bin/python -m memory_engine.main --demo          # real GGUF
```

### Layered unit tests

```bash
.venv/bin/python -m unittest tests.test_layers -v
```

- **Layer 1a** — entity merge: `PyTorch` / `py-torch` / `Pytorch` same; `AppleInc` ≠ `apple`
- **Layer 1b** — fact extraction emits `(subject, predicate, object)`
- **Layer 1c** — state transitions for `lives_in`, `works_at`, `studies_at`, `relationship_status`
- **Layer 2** — 2024 Chicago → 2025 Boston → 2026 Miami; ask “Where do I live now?”

### 11-turn life-story scenario

```bash
.venv/bin/python -m unittest tests.test_scenario_11 -v
```

Covers grew_up_in → NTU → major → interests → PyTorch → Foxconn internship →
move to Ann Arbor → transfer to UMich (+ UMich alias) → Let Him Cook founded → SwiftUI/FastAPI.

Scenario-specific alias groups and mention regexes live in `tests/fixtures/`
(not in production `memory_engine/`). Tests wire them via `tests.support.make_test_engine`.

## MCP server (any model searches memory)

Cortex exposes tools over stdio MCP. The **host model** answers; Cortex only stores/retrieves.

```bash
# Use the project venv (macOS system `python` is often missing — use this):
.venv/bin/python -m mcp_server --mock --db ./cortex.db
```

Tools: `memory_remember`, `memory_retrieve`, `memory_get_state`, `memory_search_facts`,
`memory_search_events`, `memory_search_memories`, `memory_resolve_entity`, `memory_get`,
`memory_summary`.

Cursor `mcp.json` snippet:

```json
{
  "mcpServers": {
    "cortex-memory": {
      "command": "/ABS/PATH/SQLitememo/.venv/bin/python",
      "args": ["-m", "mcp_server", "--mock"],
      "env": { "CORTEX_DB": "/ABS/PATH/SQLitememo/cortex.db" }
    }
  }
}
```

Prefer `memory_retrieve` / `memory_get_state` over any local answer head.

### Unified benchmark CLI (Ingest → Search → Evaluate)

Use the project venv (`python` alone often fails on macOS):

```bash
# Full LoCoMo (10 conversations, gpt-4o answer + judge)
.venv/bin/python run_benchmark.py \
  --benchmark locomo \
  --project-name memloom-eval-locomo \
  --answerer-model gpt-4o \
  --judge-model gpt-4o \
  --provider openai

# LongMemEval (downloads dataset on first run; --mock if no local GGUF)
.venv/bin/python run_benchmark.py \
  --benchmark longmemeval \
  --project-name memloom-eval-longmem \
  --answerer-model gpt-4o \
  --judge-model gpt-4o \
  --provider openai \
  --max-questions 20 --mock
```

Requires `OPENAI_API_KEY`. Cortex handles ingest/search; OpenAI handles answer + judge.
Run one benchmark at a time — parallel gpt-4o runs hit org TPM limits (~30k).

Note: `memory-benchmarks/run_benchmark.py` is a Mem0-suite dispatcher (needs Mem0 cloud/OSS). Prefer the root `run_benchmark.py` for Cortex.

## LoCoMo benchmark

LoCoMo repo is expected at `./locomo` with data at `locomo/data/locomo10.json`.

### Smoke run (watch categories)

```bash
.venv/bin/python -m evals.run_locomo \
  --max-samples 1 \
  --categories 1 2 4 \
  --max-qa 30 \
  --balanced \
  --top-k 5 \
  --llm-answer \
  --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --out results/locomo_llama1b_opt.json
```

### A/B answer heads with same retrieval

Local Qwen 7B:

```bash
.venv/bin/python -m evals.run_locomo \
  --max-samples 1 --categories 1 2 4 --max-qa 30 --balanced \
  --top-k 5 --llm-answer \
  --model ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --out results/locomo_qwen7b.json
```

OpenAI answer head over Cortex retrieval:

```bash
export OPENAI_API_KEY="<your-key>"
.venv/bin/python -m evals.run_locomo \
  --max-samples 1 --categories 1 2 4 --max-qa 30 --balanced \
  --top-k 5 --openai-model gpt-4o-mini \
  --out results/locomo_gpt4o_mini_opt.json
```

## LoCoMo results (full, Aug 2026)

Setup: hybrid retrieval (BM25 ∥ vector → RRF × time-decay), **no** GBNF `--extract`, answerer + judge **gpt-4o**, `top_k=10`.  
Artifacts: `results/memloom-eval-locomo_locomo.json`, `results/memloom-eval-locomo_locomo_judged.json`.

### Cortex scores (1986 QA)

| Metric | Score |
|---|---:|
| Overall token F1 | **0.605** |
| LLM judge (gpt-4o) | **0.543** (1079 / 1986) |

| Category | n | F1 |
|---|---:|---:|
| adversarial | 446 | 0.821 |
| single_hop | 841 | 0.682 |
| temporal | 321 | 0.460 |
| multi_hop | 282 | 0.326 |
| open_domain | 96 | 0.246 |

### Compared to other memory software

Published LoCoMo tables usually report **LLM-judge %** on categories 1–4 (often excluding adversarial). Numbers below are from the Mem0 paper / Memobase writeups and vendor corrections — treat as **directional**, not a controlled bake-off (prompts, judge, and category inclusion differ).

| System | Overall (LLM judge, approx.) |
|---|---:|
| Full conversation in context | ~73% |
| Memobase / Zep (claimed) | ~75% |
| Mem0 Graph | ~68% |
| Mem0 | ~67% |
| LangMem | ~58% |
| **Cortex (this run)** | **~54%** |
| OpenAI memory baseline | ~53% |

On **token F1**, Memobase’s own published artifact reported overall F1 ≈ **0.51** (with LLM ≈ 0.76); Cortex F1 **0.61** is stronger on overlap while the judge is stricter / lower.

**Reading the gap**
- Cortex is competitive as **local hybrid RAG** (strong adversarial / solid single-hop).
- Behind graph-memory stacks on **multi-hop** and **temporal** (Mem0 temporal bar often cited ~55%).
- Closing the gap means better cross-session evidence (knowledge layer / extraction), not only a stronger answer model — gpt-4o is already used here.

### Earlier A/B smoke (conv-26, 30 QA, cats 1/2/4)

| Answer model | Overall F1 | Temporal | Single-hop | Multi-hop |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 0.308 | 0.282 | 0.386 | 0.256 |
| Qwen2.5-7B | 0.393 | 0.249 | 0.540 | 0.392 |
| gpt-4o-mini (same retrieval) | 0.494 | 0.527 | 0.601 | 0.355 |

## Notes

- If `sqlite-vec` is unavailable in your SQLite build, Cortex still runs using local numpy cosine fallback.
- `MockLLM` exists for offline tests only (`--mock`), not production quality.
- If you expose API keys in logs/chats, rotate them immediately.

## License / attribution

- LoCoMo dataset and benchmark code are from Snap Research: <https://github.com/snap-research/locomo>
