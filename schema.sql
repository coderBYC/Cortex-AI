-- Cortex-AI Knowledge Layer schema
-- Immutable raw memories + resolved entities + facts/events/state

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- =====================================================================
-- RAW MEMORY STORE (immutable)
-- =====================================================================
CREATE TABLE IF NOT EXISTS memory_store (
    id          TEXT PRIMARY KEY,          -- mem_001
    text        TEXT NOT NULL,
    timestamp   TEXT NOT NULL,             -- ISO-8601
    embedding   BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_store_fts USING fts5(
    text,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);

-- =====================================================================
-- ENTITY GRAPH
-- =====================================================================
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,      -- ENTITY_001
    canonical_name  TEXT NOT NULL,
    type            TEXT NOT NULL,         -- organization|location|field|software|person|...
    embedding       BLOB,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    UNIQUE(entity_id, alias)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias
    ON entity_aliases(alias);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    entity_id UNINDEXED,
    name,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);

-- Mentions detected in a raw memory (before/after resolution)
CREATE TABLE IF NOT EXISTS mentions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id           TEXT NOT NULL REFERENCES memory_store(id) ON DELETE CASCADE,
    mention             TEXT NOT NULL,
    type                TEXT NOT NULL,
    resolved_entity_id  TEXT REFERENCES entities(id),
    resolve_score       REAL,
    resolve_method      TEXT               -- exact|alias|jw|fts|embedding|created
);

CREATE INDEX IF NOT EXISTS idx_mentions_memory
    ON mentions(memory_id);

-- =====================================================================
-- FACT STORE (subject-predicate-object)
-- =====================================================================
CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,      -- fact_001
    subject         TEXT NOT NULL,         -- USER | ENTITY_xxx
    predicate       TEXT NOT NULL,         -- studied_at | major | uses | ...
    object          TEXT NOT NULL,         -- ENTITY_xxx | literal
    time_expr       TEXT,                  -- "before 2026", "2026-present"
    memory_id       TEXT REFERENCES memory_store(id),
    confidence      REAL NOT NULL DEFAULT 1.0,
    valid_from      TEXT,
    valid_to        TEXT,                  -- NULL = currently valid
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_subject
    ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_predicate
    ON facts(predicate);
CREATE INDEX IF NOT EXISTS idx_facts_active
    ON facts(valid_to);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);

-- =====================================================================
-- EVENT STORE
-- =====================================================================
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,      -- event_001
    event_type      TEXT NOT NULL,         -- transfer_school | moved | ...
    payload_json    TEXT NOT NULL,         -- {"from":"...","to":"...","year":2026}
    memory_id       TEXT REFERENCES memory_store(id),
    event_time      TEXT,
    embedding       BLOB,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON events(event_type);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    event_id UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);

-- =====================================================================
-- STATE STORE (current world model)
-- =====================================================================
CREATE TABLE IF NOT EXISTS states (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject         TEXT NOT NULL,         -- USER | ENTITY_xxx
    key             TEXT NOT NULL,         -- studied_at | major | location | ...
    value           TEXT NOT NULL,
    source_fact_id  TEXT REFERENCES facts(id),
    source_event_id TEXT REFERENCES events(id),
    updated_at      TEXT NOT NULL,
    UNIQUE(subject, key)
);

CREATE INDEX IF NOT EXISTS idx_states_subject
    ON states(subject);

-- Legacy Phase-1 tables kept for migration compatibility (unused by CortexEngine)
CREATE TABLE IF NOT EXISTS memories_meta (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity          TEXT    NOT NULL,
    attribute       TEXT    NOT NULL,
    value           TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL,
    invalidated_at  TEXT,
    is_permanent    INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    parent_context  TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    fact,
    content='',
    contentless_delete=1,
    tokenize='porter unicode61'
);
