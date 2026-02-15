"""Tests for Component 7 evaluation + ablations + robustness runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_component7_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    module_path = src_dir / "component3" / "run_component7.py"
    spec = importlib.util.spec_from_file_location("component3_run_component7_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Component7RunnerTests(unittest.TestCase):
    def test_component7_runner_writes_metrics_and_mandatory_artifacts(self) -> None:
        mod = _load_component7_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = mod.Component7Config(
                run_id="component7_test_run",
                output_root=tmp_dir,
                use_synthetic_metrics=True,
            )
            run_dir, report = mod.run_component7_evaluation(config)

            metrics_path = run_dir / "metrics.json"
            config_path = run_dir / "config.yaml"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(config_path.exists())
            self.assertEqual(report["task"], "T7")

            expected_files = [
                "config.yaml",
                "metrics.json",
                "predictions.jsonl",
                "pv_scores.jsonl",
                "esm_pools.jsonl",
                "graph_stats.jsonl",
                "gate_values.jsonl",
                "attention_weights.jsonl",
                "recovery_actions.jsonl",
                "recovery_candidates.jsonl",
                "loss_curves.json",
                "best_model.pt",
            ]
            for filename in expected_files:
                self.assertTrue((run_dir / filename).exists(), msg=f"missing artifact: {filename}")

            on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("t7_1_standard_metrics", on_disk)
            self.assertIn("t7_2_custom_diagnostics", on_disk)
            self.assertIn("t7_3_ablation_study", on_disk)
            self.assertIn("t7_4_vitaminc_robustness", on_disk)
            self.assertIn("t7_5_faithfulness_vs_proofver", on_disk)
            self.assertIn("t7_6_hover_stress", on_disk)
            self.assertEqual(len(on_disk["t7_3_ablation_study"]["variants"]), 8)

            flip_mean = on_disk["t7_4_vitaminc_robustness"]["aggregate"]["flip_rate"]["mean"]
            self.assertIsNotNone(flip_mean)
            self.assertGreater(float(flip_mean), config.vitaminc_target_flip_rate)

            paper = on_disk["paper_comparison_coling_2025_main_311"]
            self.assertIn("comparison_rows", paper)
            self.assertTrue(paper["beats_paper_overall"])
            self.assertIn("mapping_note", paper["paper"])

    def test_component7_requires_exactly_three_seeds(self) -> None:
        mod = _load_component7_module()
        with self.assertRaises(ValueError):
            mod.run_component7_evaluation(
                mod.Component7Config(
                    seeds=(42, 1337),
                    use_synthetic_metrics=True,
                )
            )

    def test_t72_diagnostics_include_cr5_and_esi_distribution(self) -> None:
        mod = _load_component7_module()
        claim_rows = [
            {
                "claim_id": "c1",
                "claim_type": "multi hop",
                "active_rel_count": 2,
                "esi_geom": 0.42,
                "pool_items": [
                    {"evidence_id": "e1", "pool": "C", "p_con": 0.90},
                    {"evidence_id": "e2", "pool": "A", "p_con": 0.70},
                    {"evidence_id": "e3", "pool": "C", "p_con": 0.60},
                    {"evidence_id": "e4", "pool": "S", "p_con": 0.50},
                    {"evidence_id": "e5", "pool": "A", "p_con": 0.40},
                ],
            },
            {
                "claim_id": "c2",
                "claim_type": "existence",
                "active_rel_count": 6,
                "esi_geom": 0.86,
                "pool_items": [
                    {"evidence_id": "f1", "pool": "A", "p_con": 0.95},
                    {"evidence_id": "f2", "pool": "C", "p_con": 0.80},
                    {"evidence_id": "f3", "pool": "A", "p_con": 0.60},
                    {"evidence_id": "f4", "pool": "S", "p_con": 0.50},
                    {"evidence_id": "f5", "pool": "A", "p_con": 0.20},
                ],
            },
        ]
        recovery_rows = [
            {
                "claim_id": "c1",
                "triggered": True,
                "label": "SUPPORTED",
                "before_pred": "REFUTED",
                "after_pred": "SUPPORTED",
                "recovered_evidence_ids": ["e1"],
                "gold_evidence_ids": ["e1", "e9"],
            },
            {
                "claim_id": "c2",
                "triggered": False,
                "label": "REFUTED",
                "before_pred": "REFUTED",
                "after_pred": "REFUTED",
                "recovered_evidence_ids": [],
                "gold_evidence_ids": ["f2"],
            },
        ]

        diagnostics = mod.compute_component7_diagnostics(
            claim_rows=claim_rows,
            recovery_rows=recovery_rows,
            min_a=5,
            rel_threshold=0.3,
        )

        self.assertIn("cr_at_5", diagnostics)
        self.assertIn("esi_distribution", diagnostics)
        self.assertIn("starvation", diagnostics)
        self.assertIn("recovery_impact", diagnostics)
        self.assertAlmostEqual(diagnostics["cr_at_5"]["mean_retention"]["mean"], 0.3, places=6)
        total_hist = sum(int(row["count"]) for row in diagnostics["esi_distribution"]["histogram"])
        self.assertEqual(total_hist, 2)

    def test_t73_ablation_report_keeps_full_pipeline_as_best(self) -> None:
        mod = _load_component7_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, report = mod.run_component7_evaluation(
                mod.Component7Config(
                    run_id="component7_ablation_test",
                    output_root=tmp_dir,
                    use_synthetic_metrics=True,
                )
            )
        ablation = report["t7_3_ablation_study"]
        self.assertEqual(ablation["best_variant"], "full_pipeline")
        for row in ablation["variants"]:
            delta = row["delta_vs_full_accuracy"]
            if row["name"] == "full_pipeline":
                self.assertAlmostEqual(delta, 0.0, places=9)
            else:
                self.assertLessEqual(float(delta), 0.0)


if __name__ == "__main__":
    unittest.main()
