-- Local-First Memory Engine — Phase 1 schema
-- Applied by memory_engine.db.init_db(); kept here as the canonical reference.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Metadata for each memory fact
CREATE TABLE IF NOT EXISTS memories_meta (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity          TEXT    NOT NULL,
    attribute       TEXT    NOT NULL,
    value           TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL,          -- ISO-8601 UTC
    invalidated_at  TEXT,                      -- NULL = active
    is_permanent    INTEGER NOT NULL DEFAULT 0, -- 1 => λ = 0 (no decay)
    embedding       BLOB                       -- fallback when sqlite-vec unavailable
);

CREATE INDEX IF NOT EXISTS idx_memories_meta_entity
    ON memories_meta(entity);
CREATE INDEX IF NOT EXISTS idx_memories_meta_active
    ON memories_meta(invalidated_at);

-- BM25 / lexical search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    fact,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);

-- Semantic vector search (sqlite-vec). Created at runtime once the
-- extension is loaded; dimension is configurable (default 384).
-- CREATE VIRTUAL TABLE memories_vec USING vec0(
--     memory_id INTEGER PRIMARY KEY,
--     embedding float[384]
-- );
