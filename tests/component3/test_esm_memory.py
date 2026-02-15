"""Tests for Component 4 persistent ESM memory."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_esm_memory_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "component3" / "esm_memory.py"
    spec = importlib.util.spec_from_file_location("component3_esm_memory_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ESMMemoryTests(unittest.TestCase):
    def test_initialize_and_write_snapshot_jsonl(self) -> None:
        mod = _load_esm_memory_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runs" / "t41" / "esm_pools.jsonl"
            store = mod.ESMMemoryStore(output_path=path)
            store.initialize_claim(
                claim_id="claim_1",
                triples=[
                    {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.8},
                    {"evidence_id": "s1", "pool": "S", "raw_triple": ["B", "r2", "C"], "rel": 0.35},
                    {"evidence_id": "c1", "pool": "C", "raw_triple": ["C", "r3", "D"], "rel": 0.6},
                ],
            )
            store.write_claim_snapshot("claim_1")

            self.assertTrue(path.exists())
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["claim_id"], "claim_1")
            self.assertEqual(row["pools"]["A"], ["a1"])
            self.assertEqual(row["pools"]["S"], ["s1"])
            self.assertEqual(row["pools"]["C"], ["c1"])
            s_row = next(item for item in row["triples"] if item["evidence_id"] == "s1")
            self.assertEqual(s_row["suspended_at_round"], 0)
            self.assertEqual(s_row["suspended_reason"], "initial_esm_partition")

    def test_recovery_attempt_and_promotion_are_tracked(self) -> None:
        mod = _load_esm_memory_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runs" / "t41" / "esm_pools.jsonl"
            store = mod.ESMMemoryStore(output_path=path)
            store.initialize_claim(
                claim_id="claim_2",
                triples=[
                    {"evidence_id": "s_bridge", "pool": "S", "raw_triple": ["B", "r_bridge", "C"], "rel": 0.42},
                ],
            )

            store.record_recovery_attempt(
                claim_id="claim_2",
                evidence_id="s_bridge",
                round_index=1,
                selected=True,
                bridge_score_rank=1,
                bridge_score_value=0.87,
            )
            store.promote_s_to_active(claim_id="claim_2", evidence_id="s_bridge", round_index=1)

            state = store.get_claim_state("claim_2")
            triple = state.triples["s_bridge"]
            self.assertEqual(triple.current_pool, "A")
            self.assertEqual(triple.promoted_to_active_at_round, 1)
            self.assertEqual(len(triple.recovery_attempts), 1)
            self.assertTrue(triple.recovery_attempts[0].selected)
            self.assertEqual(triple.recovery_attempts[0].bridge_score_rank, 1)

    def test_non_s_promotions_or_attempts_raise(self) -> None:
        mod = _load_esm_memory_module()
        store = mod.ESMMemoryStore(output_path=Path("runs") / "unused" / "esm_pools.jsonl")
        store.initialize_claim(
            claim_id="claim_3",
            triples=[
                {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r", "B"], "rel": 0.9},
                {"evidence_id": "c1", "pool": "C", "raw_triple": ["A", "r", "C"], "rel": 0.7},
            ],
        )

        with self.assertRaises(ValueError):
            store.promote_s_to_active(claim_id="claim_3", evidence_id="a1", round_index=1)
        with self.assertRaises(ValueError):
            store.record_recovery_attempt(
                claim_id="claim_3",
                evidence_id="c1",
                round_index=1,
                selected=False,
            )

    def test_initialize_claim_rejects_duplicate_evidence_ids(self) -> None:
        mod = _load_esm_memory_module()
        store = mod.ESMMemoryStore(output_path=Path("runs") / "unused" / "esm_pools.jsonl")
        with self.assertRaises(ValueError):
            store.initialize_claim(
                claim_id="claim_dup",
                triples=[
                    {"evidence_id": "dup", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.9},
                    {"evidence_id": "dup", "pool": "S", "raw_triple": ["B", "r2", "C"], "rel": 0.4},
                ],
            )


if __name__ == "__main__":
    unittest.main()
