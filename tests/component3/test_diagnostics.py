"""Tests for Component 4 diagnostics metrics and artifact writing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_diagnostics_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "component3" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("component3_diagnostics_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DiagnosticsTests(unittest.TestCase):
    def test_component4_metrics_are_computed(self) -> None:
        mod = _load_diagnostics_module()

        claim_rows = [
            {"claim_id": "c1", "active_rel_count": 4},
            {"claim_id": "c2", "active_rel_count": 1},
            {"claim_id": "c3", "active_rel_count": 3},
        ]
        recovery_rows = [
            {
                "claim_id": "c1",
                "triggered": True,
                "label": "SUPPORTED",
                "before_pred": "REFUTED",
                "after_pred": "SUPPORTED",
                "recovered_evidence_ids": ["e1", "e2"],
                "gold_evidence_ids": ["e2", "e9"],
            },
            {
                "claim_id": "c2",
                "triggered": False,
                "label": "REFUTED",
                "before_pred": "REFUTED",
                "after_pred": "REFUTED",
                "recovered_evidence_ids": [],
                "gold_evidence_ids": ["x1"],
            },
            {
                "claim_id": "c3",
                "triggered": True,
                "before_correct": False,
                "after_correct": True,
                "recovered_gold_count": 1,
                "recovered_total": 1,
            },
        ]

        report = mod.compute_component4_diagnostics(
            claim_rows=claim_rows,
            recovery_rows=recovery_rows,
            min_a=3,
            rel_threshold=0.3,
        )
        metrics = report["metrics"]

        self.assertAlmostEqual(metrics["starvation_rate"]["rate"], 1.0 / 3.0, places=6)
        self.assertEqual(metrics["starvation_rate"]["starved_claims"], 1)

        self.assertAlmostEqual(metrics["recovered_gold_rate"]["rate"], 2.0 / 3.0, places=6)
        self.assertEqual(metrics["recovered_gold_rate"]["recovered_total"], 3)

        impact = metrics["recovery_impact"]
        self.assertAlmostEqual(impact["accuracy_before"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(impact["accuracy_after"], 1.0, places=6)
        self.assertAlmostEqual(impact["delta_accuracy"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(impact["trigger_rate"], 2.0 / 3.0, places=6)

    def test_run_component4_diagnostics_writes_file(self) -> None:
        mod = _load_diagnostics_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path, report = mod.run_component4_diagnostics(
                run_id="diag_smoke",
                claim_rows=[{"claim_id": "c1", "active_rel_count": 1}],
                recovery_rows=[{"claim_id": "c1", "triggered": True, "before_correct": False, "after_correct": True}],
                output_root=tmp_dir,
                min_a=2,
            )
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "diagnostics.json")

            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["component"], "component4")
            self.assertEqual(on_disk["metrics"], report["metrics"])

    def test_recovery_impact_parses_string_booleans(self) -> None:
        mod = _load_diagnostics_module()
        impact = mod.compute_recovery_impact(
            [
                {
                    "claim_id": "c_str_bool",
                    "triggered": "False",
                    "before_correct": "False",
                    "after_correct": "False",
                }
            ]
        )
        self.assertEqual(impact["triggered_claims"], 0)
        self.assertAlmostEqual(impact["trigger_rate"], 0.0, places=6)
        self.assertAlmostEqual(impact["accuracy_before"], 0.0, places=6)
        self.assertAlmostEqual(impact["accuracy_after"], 0.0, places=6)
        self.assertAlmostEqual(impact["delta_accuracy"], 0.0, places=6)

    def test_recovered_gold_rate_parses_selected_and_is_gold_strings(self) -> None:
        mod = _load_diagnostics_module()
        metrics = mod.compute_recovered_gold_rate(
            [
                {
                    "claim_id": "c_str_gold",
                    "recovered_evidence": [
                        {"selected": "False", "is_gold": True},
                        {"selected": "True", "is_gold": "False"},
                    ],
                }
            ]
        )
        self.assertEqual(metrics["recovered_total"], 1)
        self.assertEqual(metrics["recovered_gold"], 0)
        self.assertAlmostEqual(metrics["rate"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
