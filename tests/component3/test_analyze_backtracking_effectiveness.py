"""Unit tests for backtracking efficacy analysis script."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "analyze_backtracking_effectiveness.py"
    spec = importlib.util.spec_from_file_location("analyze_backtracking_effectiveness_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeBacktrackingEffectivenessTests(unittest.TestCase):
    def test_computes_trigger_useful_and_flip_metrics(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            predictions = [
                {
                    "claim_id": "test_1",
                    "label": 1,
                    "pred_before_backtracking": 0,
                    "backtracking_triggered": True,
                },
                {
                    "claim_id": "test_2",
                    "label": 0,
                    "pred_before_backtracking": 0,
                    "backtracking_triggered": False,
                },
            ]
            actions = [
                {
                    "claim_id": "test_1",
                    "triggered": True,
                    "margin_before": 0.10,
                    "margin_after": 0.10,
                    "pred_before": 0,
                    "pred_after": 0,
                    "tentative_pred": 1,
                    "label": 1,
                    "reverted": True,
                }
            ]
            (run_dir / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions),
                encoding="utf-8",
            )
            (run_dir / "recovery_actions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in actions),
                encoding="utf-8",
            )

            report = mod.analyze_run(run_dir=run_dir, useful_delta=0.01, default_split="test")
            split = report["splits"]["test"]
            self.assertEqual(split["claims_total"], 2)
            self.assertEqual(split["triggered_claims"], 1)
            self.assertAlmostEqual(split["trigger_rate"], 0.5, places=6)
            self.assertEqual(split["wrong_before_triggered"], 1)
            self.assertAlmostEqual(split["wrong_before_triggered_rate"], 1.0, places=6)
            self.assertEqual(split["useful_recovery_claims"], 0)
            self.assertAlmostEqual(split["useful_recovery_rate"], 0.0, places=6)
            self.assertEqual(split["flip_to_correct_claims"], 0)
            self.assertAlmostEqual(split["flip_to_correct_rate"], 0.0, places=6)
            self.assertEqual(split["tentative_flips_total"], 1)
            self.assertEqual(split["tentative_flip_to_correct_total"], 1)
            self.assertEqual(split["tentative_flip_to_correct_but_reverted_total"], 1)
            self.assertEqual(split["rounds_reverted"], 1)


if __name__ == "__main__":
    unittest.main()
