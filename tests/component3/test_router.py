"""Tests for Component 6 claim-type router and model routing pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_router_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    module_path = src_dir / "component3" / "router.py"
    spec = importlib.util.spec_from_file_location("component3_router_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RouterComponent6Tests(unittest.TestCase):
    def test_t61_router_classifier_meets_target_accuracy(self) -> None:
        mod = _load_router_module()
        train_rows, dev_rows, _ = mod.build_synthetic_component6_data()

        classifier, report = mod.train_router_classifier(
            train_rows,
            dev_rows,
            config=mod.RouterClassifierConfig(backend="naive_bayes", min_dev_accuracy=0.80),
        )

        self.assertEqual(classifier.backend_name, "naive_bayes")
        self.assertGreaterEqual(report.dev_accuracy, 0.80)
        self.assertTrue(report.meets_target)
        self.assertEqual(report.n_train, len(train_rows))
        self.assertEqual(report.n_dev, len(dev_rows))

    def test_t62_difficulty_scoring_uses_all_features(self) -> None:
        mod = _load_router_module()
        hard_features = mod.DifficultyFeatures(
            claim_type="multi hop",
            prediction_margin=0.05,
            graph_connectivity=0.30,
            evidence_count=1,
        )
        easy_features = mod.DifficultyFeatures(
            claim_type="existence",
            prediction_margin=0.80,
            graph_connectivity=0.98,
            evidence_count=6,
        )

        hard_score = mod.score_claim_difficulty(hard_features)
        easy_score = mod.score_claim_difficulty(easy_features)

        self.assertTrue(hard_score.is_hard)
        self.assertFalse(easy_score.is_hard)
        self.assertGreater(hard_score.hard_score, easy_score.hard_score)
        self.assertGreater(hard_score.margin_signal, 0.0)
        self.assertGreater(hard_score.connectivity_signal, 0.0)
        self.assertGreater(hard_score.evidence_signal, 0.0)

    def test_t63_threshold_tuning_maximizes_routed_accuracy(self) -> None:
        mod = _load_router_module()
        _, _, routing_rows = mod.build_synthetic_component6_data()

        tuning = mod.tune_routing_threshold(
            routing_rows,
            threshold_grid=(0.40, 0.55, 0.70),
            easy_margin_threshold=0.35,
            easy_connectivity_threshold=0.90,
        )

        by_threshold = {row.threshold: row.accuracy for row in tuning.sweep}
        self.assertIn(tuning.best_threshold, (0.40, 0.55, 0.70))
        self.assertGreaterEqual(tuning.best_accuracy, by_threshold[0.40])
        self.assertGreaterEqual(tuning.best_accuracy, by_threshold[0.70])
        self.assertGreaterEqual(tuning.best_accuracy, 0.80)

    def test_t63_model_routing_selects_expected_model_paths(self) -> None:
        mod = _load_router_module()
        router = mod.Component6ModelRouter(
            config=mod.ModelRouterConfig(
                routing_threshold=0.55,
                easy_margin_threshold=0.35,
                easy_connectivity_threshold=0.90,
            )
        )

        easy_decision = router.route(
            mod.RouteInput(
                claim_id="easy_1",
                claim_type="existence",
                prediction_margin=0.80,
                graph_connectivity=0.98,
                evidence_count=5,
            )
        )
        hard_decision = router.route(
            mod.RouteInput(
                claim_id="hard_1",
                claim_type="negation",
                prediction_margin=0.07,
                graph_connectivity=0.35,
                evidence_count=1,
            )
        )

        self.assertEqual(easy_decision.selected_model, mod.MODEL_BERT_BASELINE)
        self.assertIsNone(easy_decision.budget)
        self.assertEqual(hard_decision.selected_model, mod.MODEL_PV_QAGNN)
        self.assertIsNotNone(hard_decision.budget)
        self.assertEqual(hard_decision.reason, "hard_profile")

    def test_t64_budget_mapping_scales_with_complexity(self) -> None:
        mod = _load_router_module()
        simple_budget = mod.get_pv_route_budget("existence", hard_score=0.40)
        hard_budget = mod.get_pv_route_budget("multi hop", hard_score=0.95)

        self.assertEqual(simple_budget.backtrack_rounds, 0)
        self.assertEqual(simple_budget.gnn_layers, 1)
        self.assertLess(simple_budget.mask_alpha, hard_budget.mask_alpha)
        self.assertLess(simple_budget.max_active, hard_budget.max_active)
        self.assertEqual(hard_budget.gnn_layers, 2)
        self.assertLessEqual(hard_budget.backtrack_rounds, 2)

    def test_component6_pipeline_writes_config_and_metrics(self) -> None:
        mod = _load_router_module()
        train_rows, dev_rows, routing_rows = mod.build_synthetic_component6_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = mod.Component6Config(
                run_id="component6_test_run",
                output_root=tmp_dir,
                classifier=mod.RouterClassifierConfig(
                    backend="naive_bayes",
                    min_dev_accuracy=0.80,
                ),
            )
            run_dir, report = mod.run_component6_pipeline(
                config,
                train_examples=train_rows,
                dev_examples=dev_rows,
                routing_dev_examples=routing_rows,
            )

            metrics_path = run_dir / "metrics.json"
            config_path = run_dir / "config.yaml"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(config_path.exists())

            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["task"], "T6")
            self.assertIn("t61_router_classifier", loaded)
            self.assertIn("t62_difficulty_features", loaded)
            self.assertIn("t63_threshold_tuning", loaded)
            self.assertIn("t64_budget_mapping", loaded)
            self.assertEqual(report["t61_router_classifier"]["backend"], "naive_bayes")
            self.assertGreaterEqual(report["t61_router_classifier"]["dev_accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()
