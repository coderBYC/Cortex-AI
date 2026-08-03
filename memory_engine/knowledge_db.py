"""Knowledge-layer SQLite store: raw memories, entities, facts, events, states."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from memory_engine.db import (
    DEFAULT_DIM,
    SCHEMA_PATH,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class KnowledgeDB:
    """
    Cortex knowledge layer over a single SQLite file.

    Layers:
      memory_store  — immutable raw utterances
      entities      — resolved entity graph (+ aliases + FTS)
      facts         — SPO triples with temporal validity
      events        — discrete life events
      states        — current attribute snapshot per subject
    """

    def __init__(self, db_path: str | Path = "memories.db", dim: int = DEFAULT_DIM) -> None:
        self.db_path = Path(db_path)
        self.dim = dim
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.vec_enabled = False
        self._load_extensions()
        self._apply_schema()
        self._counters = self._load_counters()

    # ------------------------------------------------------------------ setup

    def _load_extensions(self) -> None:
        enable = getattr(self.conn, "enable_load_extension", None)
        if not callable(enable):
            self.vec_enabled = False
            return
        try:
            enable(True)
            import sqlite_vec

            sqlite_vec.load(self.conn)
            self.vec_enabled = True
        except Exception:
            self.vec_enabled = False
        finally:
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass

    def _apply_schema(self) -> None:
        if SCHEMA_PATH.exists():
            self.conn.executescript(SCHEMA_PATH.read_text())
        self.conn.commit()

    def _load_counters(self) -> dict[str, int]:
        def _max_suffix(table: str, prefix: str) -> int:
            rows = self.conn.execute(f"SELECT id FROM {table}").fetchall()
            best = 0
            for r in rows:
                sid = str(r["id"])
                if sid.startswith(prefix):
                    try:
                        best = max(best, int(sid.split("_", 1)[1]))
                    except (IndexError, ValueError):
                        continue
            return best

        return {
            "mem": _max_suffix("memory_store", "mem_"),
            "ent": _max_suffix("entities", "ENTITY_"),
            "fact": _max_suffix("facts", "fact_"),
            "event": _max_suffix("events", "event_"),
        }

    def _next_id(self, kind: str) -> str:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        n = self._counters[kind]
        if kind == "ent":
            return f"ENTITY_{n:03d}"
        if kind == "mem":
            return f"mem_{n:03d}"
        if kind == "fact":
            return f"fact_{n:03d}"
        if kind == "event":
            return f"event_{n:03d}"
        raise ValueError(kind)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KnowledgeDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------- raw memories

    def _rowid_for_memory(self, memory_id: str) -> int:
        # Map text id → stable integer via hash (FTS5 needs integer rowid).
        return abs(hash(memory_id)) % (2**63 - 1) or 1

    def add_memory(
        self,
        text: str,
        *,
        timestamp: Optional[str] = None,
        embedding: Optional[Sequence[float] | np.ndarray] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Append an immutable raw memory and FTS index it. Returns mem_xxx."""
        mid = memory_id or self._next_id("mem")
        ts = timestamp or _utc_now_iso()
        blob = pack_embedding(embedding) if embedding is not None else None
        self.conn.execute(
            "INSERT INTO memory_store (id, text, timestamp, embedding) VALUES (?, ?, ?, ?)",
            (mid, text, ts, blob),
        )
        rid = self._rowid_for_memory(mid)
        try:
            self.conn.execute(
                "INSERT INTO memory_store_fts(rowid, text) VALUES (?, ?)",
                (rid, text),
            )
        except sqlite3.IntegrityError:
            self.conn.execute("DELETE FROM memory_store_fts WHERE rowid = ?", (rid,))
            self.conn.execute(
                "INSERT INTO memory_store_fts(rowid, text) VALUES (?, ?)",
                (rid, text),
            )
        self.conn.commit()
        return mid

    def get_memory(self, memory_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM memory_store WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, text, timestamp FROM memory_store ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_memories_bm25(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        tokens = [t for t in query.replace('"', " ").split() if len(t) > 1]
        if not tokens:
            return []
        # Hash-rowid FTS join is awkward; use token overlap over the small store.
        all_rows = self.conn.execute(
            "SELECT id, text, timestamp, embedding FROM memory_store"
        ).fetchall()
        qtoks = {t.lower() for t in tokens}
        scored: list[tuple[float, dict]] = []
        for r in all_rows:
            ttoks = set(r["text"].lower().split())
            overlap = len(qtoks & ttoks)
            if overlap:
                scored.append((float(overlap), dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    # --------------------------------------------------------------- entities

    def create_entity(
        self,
        canonical_name: str,
        entity_type: str,
        *,
        aliases: Optional[Sequence[str]] = None,
        embedding: Optional[Sequence[float] | np.ndarray] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        eid = entity_id or self._next_id("ent")
        now = _utc_now_iso()
        blob = pack_embedding(embedding) if embedding is not None else None
        self.conn.execute(
            """
            INSERT INTO entities (id, canonical_name, type, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (eid, canonical_name, entity_type, blob, now, now),
        )
        alias_set = {canonical_name}
        for a in aliases or []:
            if a and a.strip():
                alias_set.add(a.strip())
        for a in alias_set:
            self.conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                (eid, a),
            )
        # FTS index
        rid = self._rowid_for_memory(eid)
        name_blob = " ".join(sorted(alias_set))
        try:
            self.conn.execute(
                "INSERT INTO entities_fts(rowid, entity_id, name) VALUES (?, ?, ?)",
                (rid, eid, name_blob),
            )
        except sqlite3.IntegrityError:
            self.conn.execute("DELETE FROM entities_fts WHERE rowid = ?", (rid,))
            self.conn.execute(
                "INSERT INTO entities_fts(rowid, entity_id, name) VALUES (?, ?, ?)",
                (rid, eid, name_blob),
            )
        self.conn.commit()
        return eid

    def add_alias(self, entity_id: str, alias: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
            (entity_id, alias.strip()),
        )
        self.conn.execute(
            "UPDATE entities SET updated_at = ? WHERE id = ?",
            (_utc_now_iso(), entity_id),
        )
        self.conn.commit()

    def get_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        aliases = self.conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        d["aliases"] = [a["alias"] for a in aliases]
        return d

    def list_entities(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, canonical_name, type FROM entities ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def find_entities_by_alias(self, name: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT e.id, e.canonical_name, e.type, e.embedding
            FROM entity_aliases a
            JOIN entities e ON e.id = a.entity_id
            WHERE lower(a.alias) = lower(?)
            """,
            (name.strip(),),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_entity_names(self) -> list[tuple[str, str, str]]:
        """Return (entity_id, canonical_name, alias) rows for JW scan."""
        rows = self.conn.execute(
            """
            SELECT e.id, e.canonical_name, a.alias, e.type, e.embedding
            FROM entities e
            JOIN entity_aliases a ON a.entity_id = e.id
            """
        ).fetchall()
        return [(r["id"], r["canonical_name"], r["alias"], r["type"], r["embedding"]) for r in rows]

    def search_entities_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        # Python fallback over aliases (reliable without FTS join issues)
        q = query.lower()
        q_norm = "".join(c for c in q if c.isalnum())
        scored: list[tuple[float, dict]] = []
        for r in self.conn.execute(
            """
            SELECT e.id, e.canonical_name, e.type, a.alias, e.embedding
            FROM entities e JOIN entity_aliases a ON a.entity_id = e.id
            """
        ).fetchall():
            alias = r["alias"].lower()
            a_norm = "".join(c for c in alias if c.isalnum())
            if q_norm and a_norm and q_norm == a_norm:
                scored.append((3.0, dict(r)))
            elif q == alias:
                scored.append((2.0, dict(r)))
            elif q_norm and a_norm and (
                (q_norm in a_norm or a_norm in q_norm)
                and min(len(q_norm), len(a_norm)) / max(len(q_norm), len(a_norm)) >= 0.8
            ):
                scored.append((1.0, dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        # dedupe by id
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for _, d in scored:
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def search_entities_embedding(
        self, query_emb: np.ndarray, limit: int = 10
    ) -> list[tuple[dict[str, Any], float]]:
        rows = self.conn.execute(
            "SELECT id, canonical_name, type, embedding FROM entities WHERE embedding IS NOT NULL"
        ).fetchall()
        scored: list[tuple[dict[str, Any], float]] = []
        for r in rows:
            other = unpack_embedding(r["embedding"])
            scored.append((dict(r), cosine_similarity(query_emb, other)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # --------------------------------------------------------------- mentions

    def insert_mention(
        self,
        memory_id: str,
        mention: str,
        mention_type: str,
        *,
        resolved_entity_id: Optional[str] = None,
        resolve_score: Optional[float] = None,
        resolve_method: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO mentions
                (memory_id, mention, type, resolved_entity_id, resolve_score, resolve_method)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                mention,
                mention_type,
                resolved_entity_id,
                resolve_score,
                resolve_method,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_mentions(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM mentions WHERE memory_id = ? ORDER BY id", (memory_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ facts

    def insert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        time_expr: Optional[str] = None,
        memory_id: Optional[str] = None,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> str:
        fid = fact_id or self._next_id("fact")
        now = _utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO facts
                (id, subject, predicate, object, time_expr, memory_id,
                 confidence, valid_from, valid_to, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fid,
                subject,
                predicate,
                obj,
                time_expr,
                memory_id,
                confidence,
                valid_from or now,
                valid_to,
                now,
            ),
        )
        text = f"{subject} {predicate} {obj}" + (f" ({time_expr})" if time_expr else "")
        rid = self._rowid_for_memory(fid)
        try:
            self.conn.execute(
                "INSERT INTO facts_fts(rowid, fact_id, text) VALUES (?, ?, ?)",
                (rid, fid, text),
            )
        except sqlite3.IntegrityError:
            self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (rid,))
            self.conn.execute(
                "INSERT INTO facts_fts(rowid, fact_id, text) VALUES (?, ?, ?)",
                (rid, fid, text),
            )
        self.conn.commit()
        return fid

    def invalidate_facts(
        self, subject: str, predicate: str, *, except_id: Optional[str] = None
    ) -> int:
        """Soft-close currently valid facts for (subject, predicate)."""
        now = _utc_now_iso()
        if except_id:
            cur = self.conn.execute(
                """
                UPDATE facts SET valid_to = ?
                WHERE subject = ? AND predicate = ? AND valid_to IS NULL AND id != ?
                """,
                (now, subject, predicate, except_id),
            )
        else:
            cur = self.conn.execute(
                """
                UPDATE facts SET valid_to = ?
                WHERE subject = ? AND predicate = ? AND valid_to IS NULL
                """,
                (now, subject, predicate),
            )
        self.conn.commit()
        return int(cur.rowcount)

    def active_facts(
        self, *, subject: Optional[str] = None, predicate: Optional[str] = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM facts WHERE valid_to IS NULL"
        params: list[Any] = []
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        if predicate:
            sql += " AND predicate = ?"
            params.append(predicate)
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def search_facts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q = query.lower()
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL ORDER BY created_at DESC"
        ).fetchall()
        scored: list[tuple[int, dict]] = []
        for r in rows:
            text = f"{r['subject']} {r['predicate']} {r['object']} {r['time_expr'] or ''}".lower()
            hits = sum(1 for t in q.split() if t in text)
            if hits:
                scored.append((hits, dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    # ----------------------------------------------------------------- events

    def insert_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        memory_id: Optional[str] = None,
        event_time: Optional[str] = None,
        embedding: Optional[Sequence[float] | np.ndarray] = None,
        event_id: Optional[str] = None,
    ) -> str:
        eid = event_id or self._next_id("event")
        now = _utc_now_iso()
        blob = pack_embedding(embedding) if embedding is not None else None
        payload_json = json.dumps(payload, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO events
                (id, event_type, payload_json, memory_id, event_time, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (eid, event_type, payload_json, memory_id, event_time, blob, now),
        )
        text = f"{event_type} {payload_json}"
        rid = self._rowid_for_memory(eid)
        try:
            self.conn.execute(
                "INSERT INTO events_fts(rowid, event_id, text) VALUES (?, ?, ?)",
                (rid, eid, text),
            )
        except sqlite3.IntegrityError:
            self.conn.execute("DELETE FROM events_fts WHERE rowid = ?", (rid,))
            self.conn.execute(
                "INSERT INTO events_fts(rowid, event_id, text) VALUES (?, ?, ?)",
                (rid, eid, text),
            )
        self.conn.commit()
        return eid

    def search_events(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q = query.lower()
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC"
        ).fetchall()
        scored: list[tuple[int, dict]] = []
        for r in rows:
            text = f"{r['event_type']} {r['payload_json']}".lower()
            hits = sum(1 for t in q.split() if t in text)
            if hits:
                d = dict(r)
                d["payload"] = json.loads(d["payload_json"])
                scored.append((hits, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload_json"])
            out.append(d)
        return out

    # ------------------------------------------------------------------ state

    def upsert_state(
        self,
        subject: str,
        key: str,
        value: str,
        *,
        source_fact_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> None:
        now = _utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO states (subject, key, value, source_fact_id, source_event_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject, key) DO UPDATE SET
                value = excluded.value,
                source_fact_id = excluded.source_fact_id,
                source_event_id = excluded.source_event_id,
                updated_at = excluded.updated_at
            """,
            (subject, key, value, source_fact_id, source_event_id, now),
        )
        self.conn.commit()

    def get_state(self, subject: str, key: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM states WHERE subject = ? AND key = ?", (subject, key)
        ).fetchone()
        return dict(row) if row else None

    def list_states(self, subject: Optional[str] = None) -> list[dict[str, Any]]:
        if subject:
            rows = self.conn.execute(
                "SELECT * FROM states WHERE subject = ? ORDER BY key", (subject,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM states ORDER BY subject, key"
            ).fetchall()
        return [dict(r) for r in rows]

    def dump_summary(self) -> dict[str, Any]:
        return {
            "memories": self.conn.execute("SELECT COUNT(*) c FROM memory_store").fetchone()["c"],
            "entities": self.conn.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"],
            "facts": self.conn.execute(
                "SELECT COUNT(*) c FROM facts WHERE valid_to IS NULL"
            ).fetchone()["c"],
            "events": self.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"],
            "states": self.conn.execute("SELECT COUNT(*) c FROM states").fetchone()["c"],
        }
