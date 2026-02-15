"""Tests for Component 4 rule-based backtracking controller."""

from __future__ import annotations

import importlib.util
import importlib
import sys
import unittest
from pathlib import Path


def _load_backtracking_module():
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    module_path = repo_root / "src" / "component3" / "backtracking.py"
    spec = importlib.util.spec_from_file_location("component3_backtracking_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _MemorySpy:
    def __init__(self):
        self.attempt_calls = []
        self.promote_calls = []

    def record_recovery_attempt(self, **kwargs):
        self.attempt_calls.append(dict(kwargs))

    def promote_s_to_active(self, **kwargs):
        self.promote_calls.append(dict(kwargs))


class BacktrackingTests(unittest.TestCase):
    def test_trigger_fires_when_any_rule_is_true(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(
            max_backtrack_rounds=2,
            backtrack_k=3,
            margin_threshold=0.15,
            min_a=2,
            rel_threshold=0.3,
        )
        active_dense = [
            {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.9},
            {"evidence_id": "a2", "pool": "A", "raw_triple": ["B", "r2", "C"], "rel": 0.8},
        ]
        active_starved = [
            {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.2},
        ]

        low_conf = controller.evaluate_trigger(
            prediction={"probabilities": [0.55, 0.45], "is_disconnected": False},
            active_triples=active_dense,
        )
        self.assertTrue(low_conf.should_trigger)
        self.assertTrue(low_conf.low_confidence)

        low_conn = controller.evaluate_trigger(
            prediction={"probabilities": [0.90, 0.10], "is_disconnected": True},
            active_triples=active_dense,
        )
        self.assertTrue(low_conn.should_trigger)
        self.assertTrue(low_conn.low_connectivity)

        hunger = controller.evaluate_trigger(
            prediction={"probabilities": [0.90, 0.10], "is_disconnected": False},
            active_triples=active_starved,
        )
        self.assertTrue(hunger.should_trigger)
        self.assertTrue(hunger.evidence_hunger)

        none = controller.evaluate_trigger(
            prediction={"probabilities": [0.90, 0.10], "is_disconnected": False},
            active_triples=active_dense,
        )
        self.assertFalse(none.should_trigger)

    def test_ranking_uses_bridge_priority_order(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(max_backtrack_rounds=2, backtrack_k=3)

        active = [
            {"evidence_id": "a1", "raw_triple": ["A", "r_left", "B"], "pool": "A", "p_ent": 0.8, "p_con": 0.1},
            {"evidence_id": "a2", "raw_triple": ["C", "r_right", "D"], "pool": "A", "p_ent": 0.7, "p_con": 0.1},
        ]
        suspended = [
            {"evidence_id": "s_bridge_low", "raw_triple": ["B", "r_bridge", "C"], "pool": "S", "p_ent": 0.6, "p_con": 0.1},
            {"evidence_id": "s_bridge_high", "raw_triple": ["A", "r_bridge2", "C"], "pool": "S", "p_ent": 0.55, "p_con": 0.15},
            {"evidence_id": "s_noise", "raw_triple": ["A", "r_noise", "B"], "pool": "S", "p_ent": 0.95, "p_con": 0.03},
        ]

        ranked = controller.rank_suspended(
            active_triples=active,
            suspended_triples=suspended,
            bridge_bonus_by_id={"s_bridge_low": 0.4, "s_bridge_high": 0.8, "s_noise": 0.95},
            salience_by_id={"s_bridge_low": 0.2, "s_bridge_high": 0.4, "s_noise": 0.9},
        )

        self.assertEqual([row.evidence_id for row in ranked], ["s_bridge_high", "s_bridge_low", "s_noise"])

    def test_toy_recovery_changes_prediction_and_updates_memory(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(
            max_backtrack_rounds=2,
            backtrack_k=1,
            margin_threshold=0.15,
            min_a=1,
            rel_threshold=0.3,
        )
        memory = _MemorySpy()

        active = [
            {"evidence_id": "a1", "raw_triple": ["A", "r_left", "B"], "pool": "A", "rel": 0.8, "p_ent": 0.8, "p_con": 0.1},
            {"evidence_id": "a2", "raw_triple": ["C", "r_right", "D"], "pool": "A", "rel": 0.7, "p_ent": 0.7, "p_con": 0.1},
        ]
        suspended = [
            {"evidence_id": "s_bridge", "raw_triple": ["B", "r_bridge", "C"], "pool": "S", "rel": 0.55, "p_ent": 0.55, "p_con": 0.08},
            {"evidence_id": "s_noise", "raw_triple": ["A", "r_noise", "B"], "pool": "S", "rel": 0.96, "p_ent": 0.95, "p_con": 0.01},
        ]

        def predictor(current_active):
            ids = {row["evidence_id"] for row in current_active}
            disconnected = "s_bridge" not in ids
            if disconnected:
                return {
                    "prediction": "REFUTED",
                    "probabilities": [0.53, 0.47],
                    "is_disconnected": True,
                    "bridge_bonus_by_id": {"s_bridge": 0.92, "s_noise": 0.95},
                    "salience_by_id": {"s_bridge": 0.8, "s_noise": 0.1},
                }
            return {
                "prediction": "SUPPORTED",
                "probabilities": [0.82, 0.18],
                "is_disconnected": False,
                "bridge_bonus_by_id": {"s_bridge": 0.92, "s_noise": 0.95},
                "salience_by_id": {"s_bridge": 0.8, "s_noise": 0.1},
            }

        result = controller.run(
            claim_id="toy_claim",
            active_triples=active,
            suspended_triples=suspended,
            predictor=predictor,
            anchors=["A", "C"],
            memory_store=memory,
        )

        self.assertTrue(result.triggered)
        self.assertEqual(result.rounds_run, 1)
        self.assertEqual(result.promoted_evidence_ids, ("s_bridge",))
        self.assertEqual(result.initial_prediction["prediction"], "REFUTED")
        self.assertEqual(result.final_prediction["prediction"], "SUPPORTED")
        self.assertEqual(result.actions[0].selected_evidence_ids, ("s_bridge",))
        self.assertGreater(len(memory.attempt_calls), 0)
        self.assertEqual(len(memory.promote_calls), 1)
        self.assertEqual(memory.promote_calls[0]["evidence_id"], "s_bridge")

    def test_budget_constraints_are_enforced(self) -> None:
        mod = _load_backtracking_module()
        with self.assertRaises(ValueError):
            mod.RuleBasedBacktrackingController(max_backtrack_rounds=3, backtrack_k=3)
        with self.assertRaises(ValueError):
            mod.RuleBasedBacktrackingController(max_backtrack_rounds=2, backtrack_k=4)

    def test_evaluate_trigger_respects_explicit_disconnect_flag(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(min_a=1)

        decision = controller.evaluate_trigger(
            prediction={"probabilities": [0.9, 0.1], "is_disconnected": False},
            active_triples=[{"pool": "A"}],  # No triple payload; should not be parsed.
        )

        self.assertFalse(decision.low_connectivity)

    def test_missing_evidence_ids_use_stable_fallbacks_during_promotion(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(
            max_backtrack_rounds=1,
            backtrack_k=1,
            margin_threshold=0.15,
            min_a=1,
            rel_threshold=0.3,
        )

        active = [
            {"evidence_id": "a0", "pool": "A", "raw_triple": ["X", "r", "Y"], "rel": 0.9, "p_ent": 0.9, "p_con": 0.05},
        ]
        suspended = [
            {"pool": "S", "raw_triple": ["Y", "r1", "A"], "rel": 0.4, "p_ent": 0.4, "p_con": 0.1},
            {"pool": "S", "raw_triple": ["A", "r2", "C"], "rel": 0.45, "p_ent": 0.45, "p_con": 0.1},
        ]

        def predictor(_current_active):
            return {
                "probabilities": [0.51, 0.49],  # Trigger low-confidence path.
                "is_disconnected": False,
                "bridge_bonus_by_id": {"triple_0": 0.1, "triple_1": 0.9},
                "salience_by_id": {"triple_0": 0.1, "triple_1": 0.9},
            }

        result = controller.run(
            claim_id="claim_missing_ids",
            active_triples=active,
            suspended_triples=suspended,
            predictor=predictor,
        )
        self.assertEqual(result.promoted_evidence_ids, ("triple_0",))

    def test_conservative_gate_blocks_non_bridge_promotions(self) -> None:
        mod = _load_backtracking_module()
        controller = mod.RuleBasedBacktrackingController(
            max_backtrack_rounds=1,
            backtrack_k=1,
            margin_threshold=0.15,
            min_a=1,
            rel_threshold=0.3,
        )

        active = [
            {"evidence_id": "a0", "pool": "A", "raw_triple": ["X", "r", "Y"], "rel": 0.9, "p_ent": 0.9, "p_con": 0.05},
        ]
        # Both suspended edges are in unseen nodes only, so neither bridges the active component.
        suspended = [
            {"evidence_id": "s0", "pool": "S", "raw_triple": ["A", "r1", "B"], "rel": 0.7, "p_ent": 0.7, "p_con": 0.1},
            {"evidence_id": "s1", "pool": "S", "raw_triple": ["B", "r2", "C"], "rel": 0.6, "p_ent": 0.6, "p_con": 0.1},
        ]

        def predictor(_current_active):
            return {
                "probabilities": [0.51, 0.49],
                "is_disconnected": False,
                "bridge_bonus_by_id": {"s0": 0.8, "s1": 0.7},
                "salience_by_id": {"s0": 0.2, "s1": 0.2},
            }

        result = controller.run(
            claim_id="claim_gate_block",
            active_triples=active,
            suspended_triples=suspended,
            predictor=predictor,
        )
        self.assertFalse(result.triggered)
        self.assertEqual(result.promoted_evidence_ids, ())

    def test_component3_package_exposes_core_api_without_optional_deps(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        src_root = repo_root / "src"
        package_key = "component3"
        existing = sys.modules.pop(package_key, None)
        sys.path.insert(0, str(src_root))
        try:
            package = importlib.import_module(package_key)
            self.assertTrue(hasattr(package, "RuleBasedBacktrackingController"))
            self.assertTrue(hasattr(package, "ESMMemoryStore"))
            self.assertTrue(hasattr(package, "compute_component4_diagnostics"))
        finally:
            sys.path.pop(0)
            sys.modules.pop(package_key, None)
            if existing is not None:
                sys.modules[package_key] = existing


if __name__ == "__main__":
    unittest.main()
