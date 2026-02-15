"""Tests for Component 5 learned controller and custom losses."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


def _load_controller_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "component3" / "controller.py"
    spec = importlib.util.spec_from_file_location("component3_controller_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ScriptedPolicy:
    def __init__(self, module):
        self._mod = module
        self._decisions = []

    def queue_decision(self, action_index: int):
        self._decisions.append(action_index)

    def decide(self, features):
        if not self._decisions:
            action_index = self._mod.ACTION_NO_OP
        else:
            action_index = self._decisions.pop(0)
        probs = [0.0, 0.0, 0.0]
        probs[action_index] = 1.0
        return self._mod.ControllerDecision(
            action_index=action_index,
            action_name=self._mod.ACTION_LABELS[action_index],
            recover_k=self._mod.ACTION_RECOVER_K[action_index],
            probabilities=tuple(probs),
            features=features,
        )


class ControllerTests(unittest.TestCase):
    def test_feature_extraction_builds_six_feature_vector(self) -> None:
        mod = _load_controller_module()
        active = [
            {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.5, "p_con": 0.1},
            {"evidence_id": "c1", "pool": "C", "raw_triple": ["B", "r2", "C"], "rel": 0.9, "p_con": 0.7},
        ]
        suspended = [{"evidence_id": "s1", "pool": "S", "raw_triple": ["C", "r3", "D"], "rel": 0.4, "p_con": 0.2}]
        prediction = {"probabilities": [0.6, 0.4], "is_disconnected": False, "esi_score": 0.75}

        features = mod.extract_controller_features(
            prediction=prediction,
            active_triples=active,
            suspended_triples=suspended,
            anchors=["A", "C"],
        )

        self.assertAlmostEqual(features.top2_margin, 0.2, places=6)
        self.assertAlmostEqual(features.connectivity, 1.0, places=6)
        self.assertAlmostEqual(features.counter_mass, 0.7, places=6)
        self.assertAlmostEqual(features.active_mass, 1.4, places=6)
        self.assertAlmostEqual(features.esi_score, 0.75, places=6)
        self.assertGreater(features.prediction_entropy, 0.65)
        self.assertLess(features.prediction_entropy, 0.70)
        self.assertEqual(len(features.as_tuple()), 6)

    def test_controller_decision_maps_to_three_actions(self) -> None:
        mod = _load_controller_module()
        controller = mod.ControllerMLP()

        def _fixed_logits(_features):
            return (0.0, 1.0, 3.0)

        controller.logits = _fixed_logits  # type: ignore[assignment]
        features = mod.ControllerFeatures(0.9, 0.05, 0.0, 1.2, 0.2, 0.1)
        decision = controller.decide(features)

        self.assertEqual(decision.action_name, "recover_top_3")
        self.assertEqual(decision.recover_k, 3)
        self.assertAlmostEqual(sum(decision.probabilities), 1.0, places=6)
        self.assertGreater(decision.probabilities[2], decision.probabilities[1])
        self.assertGreater(decision.probabilities[1], decision.probabilities[0])

    def test_learned_controller_recovery_changes_prediction(self) -> None:
        mod = _load_controller_module()
        policy = _ScriptedPolicy(mod)
        policy.queue_decision(mod.ACTION_RECOVER_TOP1)
        policy.queue_decision(mod.ACTION_NO_OP)
        controller = mod.LearnedBacktrackingController(
            policy=policy,
            max_backtrack_rounds=2,
            max_recover_k=3,
            min_a=1,
        )

        active = [
            {"evidence_id": "a1", "pool": "A", "raw_triple": ["A", "r1", "B"], "rel": 0.7, "p_ent": 0.7, "p_con": 0.1},
            {"evidence_id": "a2", "pool": "A", "raw_triple": ["C", "r2", "D"], "rel": 0.7, "p_ent": 0.7, "p_con": 0.1},
        ]
        suspended = [
            {"evidence_id": "s_bridge", "pool": "S", "raw_triple": ["B", "r3", "C"], "rel": 0.55, "p_ent": 0.55, "p_con": 0.1},
            {"evidence_id": "s_noise", "pool": "S", "raw_triple": ["A", "r4", "B"], "rel": 0.95, "p_ent": 0.95, "p_con": 0.01},
        ]

        def predictor(current_active):
            ids = {row["evidence_id"] for row in current_active}
            if "s_bridge" in ids:
                return {
                    "prediction": "SUPPORTED",
                    "probabilities": [0.84, 0.16],
                    "is_disconnected": False,
                    "bridge_bonus_by_id": {"s_bridge": 0.9, "s_noise": 0.99},
                    "salience_by_id": {"s_bridge": 0.8, "s_noise": 0.1},
                    "esi_score": 0.6,
                }
            return {
                "prediction": "REFUTED",
                "probabilities": [0.52, 0.48],
                "is_disconnected": True,
                "bridge_bonus_by_id": {"s_bridge": 0.9, "s_noise": 0.99},
                "salience_by_id": {"s_bridge": 0.8, "s_noise": 0.1},
                "esi_score": 0.2,
            }

        result = controller.run(
            claim_id="claim_controller_flip",
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
        self.assertEqual(result.actions[0].decision.action_name, "recover_top_1")

    def test_starvation_counter_and_recovery_losses_match_formulas(self) -> None:
        mod = _load_controller_module()

        starv = mod.starvation_penalty_loss(0.1, lambda_starv=0.1, tau_starv=0.3)
        counter = mod.counter_evidence_preservation_loss(0.9, 0.4, lambda_counter=0.1)
        recovery = mod.recovery_utility_loss(0.8, 0.5, lambda_recovery=0.1)

        self.assertAlmostEqual(starv, 0.02, places=6)
        self.assertAlmostEqual(counter, 0.05, places=6)
        self.assertAlmostEqual(recovery, 0.03, places=6)
        self.assertAlmostEqual(mod.starvation_penalty_loss(0.5), 0.0, places=6)
        self.assertAlmostEqual(mod.counter_evidence_preservation_loss(0.2, 0.8), 0.0, places=6)
        self.assertAlmostEqual(mod.recovery_utility_loss(0.3, 0.7), 0.0, places=6)

    def test_synthetic_masking_recovery_loss_masks_gold_triples(self) -> None:
        mod = _load_controller_module()
        active = [
            {"evidence_id": "g1", "pool": "A", "rel": 0.9},
            {"evidence_id": "g2", "pool": "A", "rel": 0.8},
            {"evidence_id": "n1", "pool": "A", "rel": 0.2},
        ]

        def predictor(rows):
            ids = {row["evidence_id"] for row in rows}
            if "g1" in ids and "g2" in ids:
                return {"probability": 0.90}
            if "g1" in ids or "g2" in ids:
                return {"probability": 0.65}
            return {"probability": 0.20}

        loss, steps = mod.synthetic_masking_recovery_loss(
            active_triples=active,
            gold_evidence_ids=["g1", "g2"],
            predictor=predictor,
            lambda_recovery=0.1,
            mask_sizes=(1, 2, 3),
        )

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].mask_size, 1)
        self.assertEqual(steps[1].mask_size, 2)
        self.assertGreater(steps[0].loss_recovery, 0.0)
        self.assertGreater(steps[1].loss_recovery, steps[0].loss_recovery)
        self.assertTrue(math.isclose(loss, (steps[0].loss_recovery + steps[1].loss_recovery) / 2.0, rel_tol=1e-6))

    def test_joint_loss_matches_component5_formula(self) -> None:
        mod = _load_controller_module()
        total, breakdown = mod.compute_component5_joint_loss(
            loss_bce=0.4,
            loss_evidence=0.2,
            mean_rel_a=0.1,
            max_contra_pool=0.9,
            max_contra_used=0.4,
            prob_restored=0.8,
            prob_masked=0.5,
            lambda_starv=0.1,
            tau_starv=0.3,
            lambda_counter=0.1,
            lambda_recovery=0.1,
        )

        self.assertAlmostEqual(total, 0.7, places=6)
        self.assertAlmostEqual(breakdown.loss_starvation, 0.02, places=6)
        self.assertAlmostEqual(breakdown.loss_counter, 0.05, places=6)
        self.assertAlmostEqual(breakdown.loss_recovery, 0.03, places=6)
        self.assertAlmostEqual(breakdown.total_loss, 0.7, places=6)

    def test_loss_ablation_reports_marginal_contributions(self) -> None:
        mod = _load_controller_module()
        report = mod.run_loss_ablation(
            loss_bce=0.4,
            loss_evidence=0.2,
            mean_rel_a=0.1,
            max_contra_pool=0.9,
            max_contra_used=0.4,
            prob_restored=0.8,
            prob_masked=0.5,
        )
        by_variant = {row["variant"]: row for row in report}

        self.assertIn("full", by_variant)
        self.assertIn("no_starvation", by_variant)
        self.assertIn("no_counter", by_variant)
        self.assertIn("no_recovery", by_variant)
        self.assertIn("bce_plus_evidence_only", by_variant)
        self.assertAlmostEqual(by_variant["full"]["delta_vs_full"], 0.0, places=6)
        self.assertAlmostEqual(by_variant["no_starvation"]["delta_vs_full"], 0.02, places=6)
        self.assertAlmostEqual(by_variant["no_counter"]["delta_vs_full"], 0.05, places=6)
        self.assertAlmostEqual(by_variant["no_recovery"]["delta_vs_full"], 0.03, places=6)
        self.assertAlmostEqual(by_variant["bce_plus_evidence_only"]["total_loss"], 0.6, places=6)

    def test_component5_joint_loss_adapter_works_with_torch(self) -> None:
        mod = _load_controller_module()
        try:
            import torch  # type: ignore
        except Exception:
            self.skipTest("torch is not available in this runtime")

        class _ModelStub:
            def __init__(self):
                self.latest_edge_logits = torch.tensor([0.2, -0.1], dtype=torch.float32)
                self.latest_edge_labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
                self.latest_component5_context = {
                    "mean_rel_a": torch.tensor(0.1, dtype=torch.float32),
                    "max_contra_pool": torch.tensor(0.8, dtype=torch.float32),
                    "max_contra_used": torch.tensor(0.3, dtype=torch.float32),
                    "prob_restored": torch.tensor(0.9, dtype=torch.float32),
                    "prob_masked": torch.tensor(0.4, dtype=torch.float32),
                }

        model = _ModelStub()
        criterion = mod.Component5JointLossAdapter(
            model,
            lambda_evidence=1.0,
            lambda_starv=0.1,
            tau_starv=0.3,
            lambda_counter=0.1,
            lambda_recovery=0.1,
            loss_ablation_variant="full",
        )
        logits = torch.tensor([0.1, -0.2], dtype=torch.float32)
        labels = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss_value = criterion(logits, labels)
        self.assertGreater(float(loss_value.detach().item()), 0.0)
        self.assertGreater(criterion.last_starvation, 0.0)
        self.assertGreater(criterion.last_counter, 0.0)
        self.assertGreater(criterion.last_recovery, 0.0)

        no_rec = mod.Component5JointLossAdapter(model, loss_ablation_variant="no_recovery")
        _ = no_rec(logits, labels)
        self.assertAlmostEqual(no_rec.last_recovery, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

