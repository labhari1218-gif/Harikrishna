"""Integration test for Component 2 M1 smoke runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from component2.run_m1 import run_smoke


class RunM1Tests(unittest.TestCase):
    def test_run_smoke_writes_config_snapshot_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_smoke(input_jsonl=None, output_root=output_root, max_claims=5)

            self.assertTrue(run_dir.exists())
            run_config_path = run_dir / "run_config.json"
            preds_path = run_dir / "predictions.jsonl"

            self.assertTrue(run_config_path.exists())
            self.assertTrue(preds_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertEqual(run_config["runtime"]["max_claims"], 5)
            self.assertEqual(run_config["component2_config"]["hidden_dim"], 256)
            self.assertEqual(run_config["component2_config"]["num_heads"], 4)
            self.assertEqual(run_config["component2_config"]["rel_emb_dim"], 64)
            self.assertIn("anchors", run_config["runtime"])
            self.assertIn("source_counts", run_config["runtime"]["anchors"])

            prediction_rows = [
                json.loads(line)
                for line in preds_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(prediction_rows), 5)
            self.assertIn("anchor_source", prediction_rows[0])
            self.assertIn("anchors_hash", prediction_rows[0])

    def test_run_smoke_rejects_non_positive_max_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_smoke(input_jsonl=None, output_root=Path(tmp_dir), max_claims=0)

    def test_run_smoke_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                run_smoke(
                    input_jsonl=Path(tmp_dir) / "missing.jsonl",
                    output_root=Path(tmp_dir),
                    max_claims=1,
                )

    def test_cli_mask_overrides_are_reflected_in_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(__file__).resolve().parents[2]
            output_root = Path(tmp_dir) / "m1_logs"

            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "component2.run_m1",
                    "--max-claims",
                    "1",
                    "--output-root",
                    str(output_root),
                    "--mask-sup-alpha",
                    "2.5",
                    "--mask-sup-beta",
                    "-1.0",
                    "--mask-ref-alpha",
                    "3.5",
                    "--mask-ref-beta",
                    "-1.25",
                ],
                cwd=repo_root,
                env=env,
                text=True,
            )

            run_dir = Path(output.strip().split(": ", 1)[1])
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["runtime"]["mask_formulas"]["support"], "sigmoid(2.5*p_ent-1)")
            self.assertEqual(run_config["runtime"]["mask_formulas"]["refute"], "sigmoid(3.5*p_con-1.25)")


if __name__ == "__main__":
    unittest.main()
