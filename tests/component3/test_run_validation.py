"""Tests for Component 3 full validation runner (T3.6)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_validation_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    module_path = src_dir / "component3" / "run_validation.py"
    spec = importlib.util.spec_from_file_location("component3_run_validation_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunValidationTests(unittest.TestCase):
    def test_full_validation_writes_metrics_with_mean_std(self) -> None:
        mod = _load_validation_module()

        def fake_evaluator(model_key, seed, config):
            base = {"bert_baseline": 0.93, "qagnn_baseline": 0.90, "pv_qagnn": 0.94}[model_key]
            adj = (seed % 10) / 1000.0
            metrics = {
                "overall": {
                    "accuracy": base + adj,
                    "loss": 1.0 - (base + adj),
                    "precision": base - 0.002 + adj,
                    "recall": base - 0.001 + adj,
                    "f1": base - 0.0015 + adj,
                }
            }
            for claim_type in mod.CLAIM_TYPES:
                metrics[claim_type] = {
                    "accuracy": base + adj - 0.01,
                    "precision": base + adj - 0.012,
                    "recall": base + adj - 0.011,
                    "f1": base + adj - 0.0115,
                }
            return metrics

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = mod.ValidationConfig(
                run_id="t36_test_run",
                output_root=tmp_dir,
                use_synthetic_metrics=False,
            )
            run_dir, report = mod.run_full_validation(config, evaluator=fake_evaluator)

            metrics_path = run_dir / "metrics.json"
            config_path = run_dir / "config.yaml"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(config_path.exists())

            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["task"], "T3.6")
            self.assertEqual(loaded["model_order"], ["bert_baseline", "qagnn_baseline", "pv_qagnn"])
            self.assertEqual(len(loaded["seeds"]), 3)
            self.assertEqual(loaded["summary"]["best_model"], "pv_qagnn")

            pv_overall = loaded["models"]["pv_qagnn"]["aggregate"]["overall"]["accuracy"]
            self.assertIn("mean", pv_overall)
            self.assertIn("std", pv_overall)
            self.assertEqual(len(pv_overall["values"]), 3)

            self.assertIn("per_type", loaded["models"]["bert_baseline"]["aggregate"])
            self.assertIn("multi hop", loaded["models"]["bert_baseline"]["aggregate"]["per_type"])
            self.assertEqual(report["summary"]["best_model"], "pv_qagnn")

    def test_validation_requires_three_seeds(self) -> None:
        mod = _load_validation_module()
        config = mod.ValidationConfig(seeds=(42, 1337), use_synthetic_metrics=True)
        with self.assertRaises(ValueError):
            mod.run_full_validation(config)


if __name__ == "__main__":
    unittest.main()
