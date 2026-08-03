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

### 1) Ingestion (memory extraction + dedup)

`memory_engine/extraction.py`

- Extraction model: local GGUF via `LlamaCppBackend` (`llama-cpp-python`)
- Structured output: `MEMORY_JSON_GBNF` enforces JSON schema:
  - `entity`, `attribute`, `value`, `confidence`, `is_permanent`
- Dedup: Jaro-Winkler entity merge with threshold `0.88`
- Embeddings: `LocalEmbedder` (`fastembed`, default `BAAI/bge-small-en-v1.5`)

### 2) Storage

`schema.sql`, `memory_engine/db.py`

- `memories_meta`: source of truth
  - child fact fields (`entity`, `attribute`, `value`)
  - `created_at`, `is_permanent`, optional `parent_context`
- `memories_fts` (`FTS5`): lexical search (BM25)
- `memories_vec` (`sqlite-vec`, optional): vector KNN
- graceful fallback to numpy cosine over BLOB embeddings if `sqlite-vec` unavailable

### 3) Retrieval + ranking

`memory_engine/retrieval.py`

Final score:

\[
\text{Final Score} = \left(\frac{1}{15 + \text{Rank}_{vec}} + \frac{1}{15 + \text{Rank}_{bm25}}\right) \times e^{-\lambda \Delta t}
\]

- `RRF_K = 15` (top-rank emphasis)
- default decay `lambda = 1e-5` per hour
- permanent memories use `lambda = 0`
- parent-child expansion: retrieved child facts can surface `parent_context` session snippets in prompts

## Repository layout

```text
memory_engine/
  __init__.py
  db.py
  extraction.py
  retrieval.py
  main.py          # REPL + one-shot CLI

evals/
  run_locomo.py    # LoCoMo benchmark harness

schema.sql
requirements.txt
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
```

Commands:
- `remember <text>`
- `ask <query>`
- `list`
- `quit`

Bare text is treated as `remember`.

### One-shot

```bash
.venv/bin/python -m memory_engine.main remember "My name is Bryan. I live in Ann Arbor."
.venv/bin/python -m memory_engine.main ask "Where does Bryan live?"
.venv/bin/python -m memory_engine.main list
```

### Scripted demo

```bash
.venv/bin/python -m memory_engine.main --demo
```

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

## Current benchmark snapshot (conv-26, 30 QA, balanced, cats 1/2/4)

| Answer model | Overall F1 | Temporal | Single-hop | Multi-hop |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 0.308 | 0.282 | 0.386 | 0.256 |
| Qwen2.5-7B | 0.393 | 0.249 | 0.540 | 0.392 |
| gpt-4o-mini (same retrieval) | 0.494 | 0.527 | 0.601 | 0.355 |

Interpretation:
- Temporal is close to the ~0.555 reference band when using a strong answer model.
- Single-hop is solid with BM25 support.
- Multi-hop still needs better cross-session evidence retrieval / synthesis.

## Notes

- If `sqlite-vec` is unavailable in your SQLite build, Cortex still runs using local numpy cosine fallback.
- `MockLLM` exists for offline tests only (`--mock`), not production quality.
- If you expose API keys in logs/chats, rotate them immediately.

## License / attribution

- LoCoMo dataset and benchmark code are from Snap Research: <https://github.com/snap-research/locomo>
