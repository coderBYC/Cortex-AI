"""SQLite initialization: FTS5 + sqlite-vec (with numpy fallback)."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Default embedding dimensionality (matches common MiniLM-style local models).
DEFAULT_DIM = 384

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pack_embedding(vec: Sequence[float] | np.ndarray) -> bytes:
    """Serialize a float vector to little-endian float32 bytes."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return arr.tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    """Deserialize little-endian float32 bytes to a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MemoryDB:
    """Thin wrapper around a single SQLite file holding meta + FTS5 + vectors."""

    def __init__(self, db_path: str | Path = "memories.db", dim: int = DEFAULT_DIM) -> None:
        self.db_path = Path(db_path)
        self.dim = dim
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.vec_enabled = False
        self._load_extensions()
        self._apply_schema()

    # ------------------------------------------------------------------ setup

    def _load_extensions(self) -> None:
        # macOS system Python often ships SQLite without loadable-extension
        # support. Prefer sqlite-vec when available; otherwise score in numpy.
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
        else:
            self._apply_inline_schema()

        # Migrate older DBs that predate parent_context.
        cols = {
            r[1]
            for r in self.conn.execute("PRAGMA table_info(memories_meta)").fetchall()
        }
        if "parent_context" not in cols:
            self.conn.execute(
                "ALTER TABLE memories_meta ADD COLUMN parent_context TEXT"
            )

        if self.vec_enabled:
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding float[{self.dim}]
                )
                """
            )
        self.conn.commit()

    def _apply_inline_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

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

            CREATE INDEX IF NOT EXISTS idx_memories_meta_entity
                ON memories_meta(entity);
            CREATE INDEX IF NOT EXISTS idx_memories_meta_active
                ON memories_meta(invalidated_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                fact,
                content='',
                contentless_delete=1,
                tokenize='porter unicode61'
            );
            """
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- queries

    def list_active_entities(self) -> list[tuple[int, str]]:
        """Return (id, entity) for all non-invalidated memories."""
        rows = self.conn.execute(
            """
            SELECT id, entity FROM memories_meta
            WHERE invalidated_at IS NULL
            ORDER BY id
            """
        ).fetchall()
        return [(int(r["id"]), str(r["entity"])) for r in rows]

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM memories_meta WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def fact_text(self, entity: str, attribute: str, value: str) -> str:
        return f"{entity}'s {attribute} is {value}"

    # ---------------------------------------------------------------- writes

    def insert_memory(
        self,
        entity: str,
        attribute: str,
        value: str,
        confidence: float,
        embedding: Sequence[float] | np.ndarray,
        *,
        is_permanent: bool = False,
        created_at: Optional[str] = None,
        parent_context: Optional[str] = None,
    ) -> int:
        """
        Insert a child memory into meta + FTS + vec.

        `value` is the child fact used for BM25 / vector indexing.
        `parent_context` is the optional parent session snippet expanded into
        the answer prompt at retrieval time (not separately vectorized).
        """
        created = created_at or _utc_now_iso()
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(f"embedding dim {emb.shape[0]} != expected {self.dim}")

        cur = self.conn.execute(
            """
            INSERT INTO memories_meta
                (entity, attribute, value, confidence, created_at, invalidated_at,
                 is_permanent, embedding, parent_context)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                entity,
                attribute,
                value,
                float(confidence),
                created,
                1 if is_permanent else 0,
                pack_embedding(emb),
                parent_context,
            ),
        )
        memory_id = int(cur.lastrowid)
        # Index the compact child fact only (parent stays out of FTS to avoid noise).
        fact = self.fact_text(entity, attribute, value)
        self.conn.execute(
            "INSERT INTO memories_fts(rowid, fact) VALUES (?, ?)",
            (memory_id, fact),
        )
        if self.vec_enabled:
            self.conn.execute(
                "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
                (memory_id, pack_embedding(emb)),
            )
        self.conn.commit()
        return memory_id

    def update_memory(
        self,
        memory_id: int,
        *,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        value: Optional[str] = None,
        confidence: Optional[float] = None,
        embedding: Optional[Sequence[float] | np.ndarray] = None,
        is_permanent: Optional[bool] = None,
        parent_context: Optional[str] = None,
        bump_created_at: bool = True,
    ) -> None:
        """Merge into an existing memory (used by Jaro-Winkler dedup)."""
        row = self.get_memory(memory_id)
        if row is None:
            raise KeyError(f"memory id {memory_id} not found")

        new_entity = entity if entity is not None else row["entity"]
        new_attr = attribute if attribute is not None else row["attribute"]
        new_value = value if value is not None else row["value"]
        new_conf = float(confidence) if confidence is not None else float(row["confidence"])
        new_perm = (
            int(is_permanent)
            if is_permanent is not None
            else int(row["is_permanent"])
        )
        created_at = _utc_now_iso() if bump_created_at else row["created_at"]
        if parent_context is not None:
            new_parent = parent_context
        elif "parent_context" in row.keys():
            new_parent = row["parent_context"]
        else:
            new_parent = None

        emb_blob = row["embedding"]
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).ravel()
            if emb.shape[0] != self.dim:
                raise ValueError(f"embedding dim {emb.shape[0]} != expected {self.dim}")
            emb_blob = pack_embedding(emb)

        self.conn.execute(
            """
            UPDATE memories_meta
            SET entity = ?, attribute = ?, value = ?, confidence = ?,
                created_at = ?, is_permanent = ?, embedding = ?,
                parent_context = ?, invalidated_at = NULL
            WHERE id = ?
            """,
            (
                new_entity,
                new_attr,
                new_value,
                new_conf,
                created_at,
                new_perm,
                emb_blob,
                new_parent,
                memory_id,
            ),
        )

        fact = self.fact_text(new_entity, new_attr, new_value)
        # FTS5 contentless: delete + reinsert to refresh the indexed fact.
        self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
        self.conn.execute(
            "INSERT INTO memories_fts(rowid, fact) VALUES (?, ?)",
            (memory_id, fact),
        )

        if self.vec_enabled and embedding is not None:
            self.conn.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,)
            )
            self.conn.execute(
                "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
                (memory_id, emb_blob),
            )

        self.conn.commit()

    def invalidate(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories_meta SET invalidated_at = ? WHERE id = ?",
            (_utc_now_iso(), memory_id),
        )
        self.conn.commit()

    # ----------------------------------------------------------- search helpers

    @staticmethod
    def _fts_tokens(query: str) -> list[str]:
        # Strip punctuation; drop ultra-common stopwords so OR queries stay focused.
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
            "on", "for", "and", "or", "what", "where", "when", "who", "whom",
            "which", "does", "do", "did", "me", "my", "about", "tell", "please",
        }
        raw = re.findall(r"[A-Za-z0-9]+", query.lower())
        tokens = [t for t in raw if t not in stop and len(t) > 1]
        return tokens

    def bm25_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """
        Return (memory_id, bm25_rank) for active memories.
        Rank is 1-based position after ordering by BM25 (lower bm25 score = better).
        Uses an OR of content tokens so natural-language questions still hit.
        """
        tokens = self._fts_tokens(query)
        if not tokens:
            return []
        # OR-combine quoted tokens (FTS5: space = AND, OR must be explicit).
        match = " OR ".join(f'"{t}"' for t in tokens)

        try:
            rows = self.conn.execute(
                """
                SELECT m.id AS id, bm25(memories_fts) AS score
                FROM memories_fts
                JOIN memories_meta m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.invalidated_at IS NULL
                ORDER BY score
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r["id"]), float(i + 1)) for i, r in enumerate(rows)]

    def vector_search(
        self, query_embedding: Sequence[float] | np.ndarray, limit: int = 20
    ) -> list[tuple[int, float]]:
        """
        Return (memory_id, vec_rank) for active memories.
        Prefers sqlite-vec KNN; falls back to numpy cosine over BLOBs.
        """
        emb = np.asarray(query_embedding, dtype=np.float32).ravel()
        if emb.shape[0] != self.dim:
            raise ValueError(f"query embedding dim {emb.shape[0]} != expected {self.dim}")

        if self.vec_enabled:
            return self._vector_search_sqlite_vec(emb, limit)
        return self._vector_search_numpy(emb, limit)

    def _vector_search_sqlite_vec(
        self, emb: np.ndarray, limit: int
    ) -> list[tuple[int, float]]:
        blob = pack_embedding(emb)
        rows = self.conn.execute(
            """
            SELECT v.memory_id AS id, v.distance AS distance
            FROM memories_vec v
            JOIN memories_meta m ON m.id = v.memory_id
            WHERE v.embedding MATCH ?
              AND k = ?
              AND m.invalidated_at IS NULL
            ORDER BY distance
            """,
            (blob, limit),
        ).fetchall()
        return [(int(r["id"]), float(i + 1)) for i, r in enumerate(rows)]

    def _vector_search_numpy(
        self, emb: np.ndarray, limit: int
    ) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            """
            SELECT id, embedding FROM memories_meta
            WHERE invalidated_at IS NULL AND embedding IS NOT NULL
            """
        ).fetchall()
        scored: list[tuple[int, float]] = []
        for r in rows:
            other = unpack_embedding(r["embedding"])
            scored.append((int(r["id"]), cosine_similarity(emb, other)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(mid, float(i + 1)) for i, (mid, _) in enumerate(scored[:limit])]

    def dump_active(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, entity, attribute, value, confidence, created_at, is_permanent
            FROM memories_meta
            WHERE invalidated_at IS NULL
            ORDER BY id
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def __repr__(self) -> str:
        return (
            f"MemoryDB(path={self.db_path!s}, dim={self.dim}, "
            f"sqlite_vec={'on' if self.vec_enabled else 'fallback'})"
        )
