"""Tests for Component 3 bridge rescue integration."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_pv_bridge_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "component3" / "pv_bridge.py"
    spec = importlib.util.spec_from_file_location("component3_pv_bridge_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PVBridgeTests(unittest.TestCase):
    def test_trigger_requires_low_margin_and_disconnected_graph(self) -> None:
        mod = _load_pv_bridge_module()
        self.assertTrue(mod.should_trigger_bridge_rescue(margin=0.14, is_disconnected=True, margin_threshold=0.15))
        self.assertFalse(mod.should_trigger_bridge_rescue(margin=0.16, is_disconnected=True, margin_threshold=0.15))
        self.assertFalse(mod.should_trigger_bridge_rescue(margin=0.10, is_disconnected=False, margin_threshold=0.15))

    def test_ranking_uses_canonical_priority_with_salience_tiebreak(self) -> None:
        mod = _load_pv_bridge_module()
        engine = mod.PVBridgeRecoveryEngine(max_backtrack_rounds=2, backtrack_k=3)

        active = [
            {"evidence_id": "a1", "raw_triple": ["A", "r_left", "B"], "pool": "A", "p_ent": 0.8, "p_con": 0.1},
            {"evidence_id": "a2", "raw_triple": ["C", "r_right", "D"], "pool": "A", "p_ent": 0.7, "p_con": 0.1},
        ]
        suspended = [
            # Connects components, lower bridge bonus.
            {"evidence_id": "s_bridge_low", "raw_triple": ["B", "r_bridge", "C"], "pool": "S", "p_ent": 0.6, "p_con": 0.1},
            # Connects components, higher bridge bonus.
            {"evidence_id": "s_bridge_high", "raw_triple": ["A", "r_bridge2", "C"], "pool": "S", "p_ent": 0.55, "p_con": 0.15},
            # High rel/bonus but does not connect components, must rank after connecters.
            {"evidence_id": "s_noise", "raw_triple": ["A", "r_noise", "B"], "pool": "S", "p_ent": 0.95, "p_con": 0.03},
        ]

        ranked = engine.rank_suspended_candidates(
            active_triples=active,
            suspended_triples=suspended,
            bridge_bonus_by_id={"s_bridge_low": 0.4, "s_bridge_high": 0.8, "s_noise": 0.95},
            salience_by_id={"s_bridge_low": 0.2, "s_bridge_high": 0.4, "s_noise": 0.9},
            k=3,
        )

        self.assertEqual([row.evidence_id for row in ranked], ["s_bridge_high", "s_bridge_low", "s_noise"])

    def test_toy_bridge_rescue_changes_prediction(self) -> None:
        mod = _load_pv_bridge_module()
        engine = mod.PVBridgeRecoveryEngine(max_backtrack_rounds=2, backtrack_k=1, margin_threshold=0.15)

        active = [
            {"evidence_id": "a1", "raw_triple": ["A", "r_left", "B"], "pool": "A", "p_ent": 0.8, "p_con": 0.1},
            {"evidence_id": "a2", "raw_triple": ["C", "r_right", "D"], "pool": "A", "p_ent": 0.7, "p_con": 0.1},
        ]
        suspended = [
            {"evidence_id": "s_bridge", "raw_triple": ["B", "r_bridge", "C"], "pool": "S", "p_ent": 0.55, "p_con": 0.08},
            {"evidence_id": "s_noise", "raw_triple": ["A", "r_noise", "B"], "pool": "S", "p_ent": 0.95, "p_con": 0.01},
        ]

        def predictor(current_active):
            ids = {row["evidence_id"] for row in current_active}
            disconnected = "s_bridge" not in ids
            if disconnected:
                return {
                    "prediction": "REFUTED",
                    "margin": 0.05,
                    "is_disconnected": True,
                    "bridge_bonus_by_id": {"s_bridge": 0.9, "s_noise": 0.95},
                    "salience_by_id": {"s_bridge": 0.7, "s_noise": 0.1},
                }
            return {
                "prediction": "SUPPORTED",
                "margin": 0.35,
                "is_disconnected": False,
                "bridge_bonus_by_id": {"s_bridge": 0.9, "s_noise": 0.95},
                "salience_by_id": {"s_bridge": 0.7, "s_noise": 0.1},
            }

        result = engine.run(
            claim_id="toy_claim",
            active_triples=active,
            suspended_triples=suspended,
            predictor=predictor,
            anchors=["A", "C"],
        )

        self.assertTrue(result.triggered)
        self.assertEqual(result.rounds_run, 1)
        self.assertEqual(result.promoted_evidence_ids, ("s_bridge",))
        self.assertEqual(result.initial_prediction["prediction"], "REFUTED")
        self.assertEqual(result.final_prediction["prediction"], "SUPPORTED")
        self.assertEqual(result.actions[0].selected_evidence_ids, ("s_bridge",))

    def test_budget_constraints_are_enforced(self) -> None:
        mod = _load_pv_bridge_module()
        with self.assertRaises(ValueError):
            mod.PVBridgeRecoveryEngine(max_backtrack_rounds=3, backtrack_k=3)
        with self.assertRaises(ValueError):
            mod.PVBridgeRecoveryEngine(max_backtrack_rounds=2, backtrack_k=4)


if __name__ == "__main__":
    unittest.main()
