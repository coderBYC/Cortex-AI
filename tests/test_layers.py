"""Layered tests for Cortex knowledge pipeline.

Layer 1a — Entity resolution (aliases / hard negatives)
Layer 1b — Fact extraction (subject, predicate, object)
Layer 1c — State transitions (lives_in, works_at, studies_at, relationship_status)
Layer 2  — Temporal conversational history (Chicago → Boston → Miami)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_engine.cortex import CortexEngine
from memory_engine.extraction import MockLLM, require_local_embedder
from memory_engine.extractors import ExtractedFact, KnowledgeExtractor
from memory_engine.knowledge_db import KnowledgeDB
from memory_engine.mentions import Mention
from memory_engine.resolve import EntityResolver, name_similarity, normalize_name
from memory_engine.state_engine import StateTransitionEngine


def _tmp_db() -> Path:
    return Path(tempfile.mkstemp(suffix=".db")[1])


class Layer1aEntityResolution(unittest.TestCase):
    """PyTorch variants collapse; AppleInc ≠ apple."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = require_local_embedder()

    def setUp(self) -> None:
        self.db_path = _tmp_db()
        self.db = KnowledgeDB(self.db_path)
        self.resolver = EntityResolver(self.db, self.embedder)

    def tearDown(self) -> None:
        self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            p.unlink(missing_ok=True)

    def test_normalize_pytorch_variants(self) -> None:
        self.assertEqual(normalize_name("PyTorch"), "pytorch")
        self.assertEqual(normalize_name("py-torch"), "pytorch")
        self.assertEqual(normalize_name("Pytorch"), "pytorch")
        self.assertEqual(name_similarity("PyTorch", "py-torch"), 1.0)

    def test_pytorch_aliases_same_entity(self) -> None:
        a = self.resolver.resolve(Mention("PyTorch", "software"))
        b = self.resolver.resolve(Mention("py-torch", "software"))
        c = self.resolver.resolve(Mention("Pytorch", "software"))
        self.assertEqual(a.entity_id, b.entity_id)
        self.assertEqual(b.entity_id, c.entity_id)
        self.assertEqual(len(self.db.list_entities()), 1)

    def test_appleinc_vs_apple_different(self) -> None:
        # High JW (~0.92) but length ratio should keep them apart
        self.assertLess(name_similarity("AppleInc", "apple"), 0.88)
        apple_inc = self.resolver.resolve(Mention("AppleInc", "organization"))
        apple = self.resolver.resolve(Mention("apple", "organization"))
        self.assertNotEqual(apple_inc.entity_id, apple.entity_id)
        self.assertEqual(len(self.db.list_entities()), 2)


class Layer1bFactExtraction(unittest.TestCase):
    """Extractor returns (subject, predicate, object) triples."""

    def test_fact_list_lives_and_uses(self) -> None:
        extractor = KnowledgeExtractor(MockLLM())
        # Pre-resolved entities as the write path would supply
        from memory_engine.resolve import ResolvedMention

        resolved = [
            ResolvedMention("Chicago", "location", "ENTITY_001", "Chicago", 1.0, "created"),
            ResolvedMention("PyTorch", "software", "ENTITY_002", "PyTorch", 1.0, "created"),
            ResolvedMention("NTU", "organization", "ENTITY_003", "NTU", 1.0, "created"),
        ]
        text = "I live in Chicago. I study at NTU. I use PyTorch."
        facts, _events = extractor.extract(text, resolved)
        triples = {(f.subject, f.predicate, f.object) for f in facts}

        self.assertIn(("USER", "lives_in", "ENTITY_001"), triples)
        self.assertIn(("USER", "uses", "ENTITY_002"), triples)
        # studies_at preferred for present-tense study
        self.assertTrue(
            ("USER", "studies_at", "ENTITY_003") in triples
            or ("USER", "studied_at", "ENTITY_003") in triples
        )

    def test_end_to_end_remember_emits_spo(self) -> None:
        db_path = _tmp_db()
        try:
            with CortexEngine(db_path, allow_mock=True) as engine:
                result = engine.remember("I live in Chicago and I use PyTorch.")
                self.assertTrue(result.facts, "expected at least one SPO fact")
                for f in result.facts:
                    self.assertIn("subject", f)
                    self.assertIn("predicate", f)
                    self.assertIn("object", f)
                    self.assertTrue(f["subject"])
                    self.assertTrue(f["predicate"])
                    self.assertTrue(f["object"])
                preds = {f["predicate"] for f in result.facts}
                self.assertIn("lives_in", preds)
                self.assertIn("uses", preds)
        finally:
            Path(db_path).unlink(missing_ok=True)


class Layer1cStateTransitions(unittest.TestCase):
    """Stateful predicates replace prior values."""

    PREDICATES = ("lives_in", "works_at", "studies_at", "relationship_status")

    def setUp(self) -> None:
        self.db_path = _tmp_db()
        self.db = KnowledgeDB(self.db_path)
        self.engine = StateTransitionEngine(self.db)
        self.mid = self.db.add_memory("seed")

    def tearDown(self) -> None:
        self.db.close()
        self.db_path.unlink(missing_ok=True)

    def test_each_stateful_predicate_replaces(self) -> None:
        first = {
            "lives_in": "Chicago",
            "works_at": "ENTITY_ORG_A",
            "studies_at": "ENTITY_SCHOOL_A",
            "relationship_status": "single",
        }
        second = {
            "lives_in": "Boston",
            "works_at": "ENTITY_ORG_B",
            "studies_at": "ENTITY_SCHOOL_B",
            "relationship_status": "married",
        }
        for pred in self.PREDICATES:
            self.engine.apply_facts(
                [ExtractedFact("USER", pred, first[pred], "past")],
                memory_id=self.mid,
            )
        for pred in self.PREDICATES:
            self.engine.apply_facts(
                [ExtractedFact("USER", pred, second[pred], "present")],
                memory_id=self.mid,
            )

        states = {s["key"]: s["value"] for s in self.db.list_states("USER")}
        # studies_at normalizes to studied_at
        self.assertEqual(states.get("lives_in"), "Boston")
        self.assertEqual(states.get("works_at"), "ENTITY_ORG_B")
        self.assertEqual(states.get("studied_at"), "ENTITY_SCHOOL_B")
        self.assertEqual(states.get("relationship_status"), "married")

        # Prior values soft-closed (valid_to set), only latest active
        for pred, expected in [
            ("lives_in", "Boston"),
            ("works_at", "ENTITY_ORG_B"),
            ("studied_at", "ENTITY_SCHOOL_B"),
            ("relationship_status", "married"),
        ]:
            active = self.db.active_facts(subject="USER", predicate=pred)
            self.assertEqual(len(active), 1, pred)
            self.assertEqual(active[0]["object"], expected)


class Layer2TemporalHistory(unittest.TestCase):
    """2024 Chicago → 2025 Boston → 2026 Miami; current state is Miami."""

    def test_moves_update_current_lives_in(self) -> None:
        db_path = _tmp_db()
        try:
            with CortexEngine(db_path, allow_mock=True) as engine:
                engine.remember(
                    "I live in Chicago",
                    timestamp="2024-06-01T00:00:00+00:00",
                )
                st = engine.db.get_state("USER", "lives_in")
                self.assertIsNotNone(st)
                chicago_id = st["value"]
                chicago = engine.db.get_entity(chicago_id)
                self.assertEqual(chicago["canonical_name"], "Chicago")

                engine.remember(
                    "I moved to Boston",
                    timestamp="2025-06-01T00:00:00+00:00",
                )
                st = engine.db.get_state("USER", "lives_in")
                boston = engine.db.get_entity(st["value"])
                self.assertEqual(boston["canonical_name"], "Boston")

                engine.remember(
                    "I moved to Miami",
                    timestamp="2026-06-01T00:00:00+00:00",
                )
                st = engine.db.get_state("USER", "lives_in")
                miami = engine.db.get_entity(st["value"])
                self.assertEqual(miami["canonical_name"], "Miami")

                # Only one active lives_in fact
                active = engine.db.active_facts(subject="USER", predicate="lives_in")
                self.assertEqual(len(active), 1)
                self.assertEqual(active[0]["object"], st["value"])

                # Historical facts retained (invalidated)
                all_lives = engine.db.conn.execute(
                    "SELECT object, valid_to FROM facts WHERE subject='USER' AND predicate='lives_in'"
                ).fetchall()
                self.assertGreaterEqual(len(all_lives), 3)

                # Query path
                ask = engine.ask("Where do I live now?")
                self.assertEqual(ask.intent, "state")
                self.assertIn("Miami", ask.answer)

                # Move events recorded
                events = engine.db.list_events()
                moved = [e for e in events if e["event_type"] == "moved"]
                self.assertGreaterEqual(len(moved), 2)
        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
