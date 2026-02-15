"""Integration test for M2 smoke runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from component2.run_m2 import run_smoke_m2


class RunM2Tests(unittest.TestCase):
    def test_run_smoke_m2_writes_recovery_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_smoke_m2(input_jsonl=None, output_root=output_root, max_claims=1)

            run_config_path = run_dir / "run_config.json"
            preds_path = run_dir / "predictions.jsonl"
            summary_path = run_dir / "summary.json"

            self.assertTrue(run_config_path.exists())
            self.assertTrue(preds_path.exists())
            self.assertTrue(summary_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertEqual(run_config["runtime"]["recovery_policy"]["recovery_top_k"], 3)
            self.assertEqual(run_config["component2_config"]["ppr_epsilon"], 0.001)
            self.assertEqual(run_config["component2_config"]["neutral_cap_gamma"], 2.0)
            self.assertIn("anchors", run_config["runtime"])
            self.assertIn("claims_missing_pickle_mapping", run_config["runtime"]["anchors"])

            prediction_rows = [
                json.loads(line)
                for line in preds_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(prediction_rows), 1)
            row = prediction_rows[0]
            self.assertTrue(row["should_recover"])
            self.assertGreater(row["delta_conn_bridge_at_k"], 0.0)
            self.assertEqual(row["rpi_rel_at_k"], 0.0)
            self.assertNotEqual(row["prob_supported_before"], row["prob_supported_after"])
            self.assertTrue(row["moved_evidence_ids"])
            self.assertIn("ppr_diagnostics", row)
            self.assertIn("converged", row["ppr_diagnostics"])
            self.assertIn("has_disconnected_anchors", row["ppr_diagnostics"])
            self.assertIn("anchor_source", row)
            self.assertIn("anchors_hash", row)
            self.assertIn("esi_geom_before", row)
            self.assertIn("esi_geom_after", row)
            self.assertAlmostEqual(row["esi_geom"], row["esi_geom_after"], places=12)
            self.assertGreaterEqual(row["esi_geom_before"], 0.0)
            self.assertLessEqual(row["esi_geom_before"], 1.0)
            self.assertGreaterEqual(row["esi_geom_after"], 0.0)
            self.assertLessEqual(row["esi_geom_after"], 1.0)
            self.assertGreater(row["esi_geom_after"], row["esi_geom_before"])

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("num_ppr_not_converged", summary)
            self.assertIn("num_disconnected_anchor_graphs", summary)
            self.assertIn("claims_missing_pickle_mapping", summary)

    def test_run_smoke_m2_uses_ac_esi_for_recovery_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            claims_path.write_text(
                json.dumps(
                    {
                        "claim_id": "ac_high",
                        "claim_text": "A relates to C",
                        "label": "REFUTED",
                        "entity_set": ["A", "C"],
                        "triples": [
                            {
                                "evidence_id": "a1",
                                "raw_triple": ["A", "r_left", "B"],
                                "pool": "A",
                                "p_ent": 0.05,
                                "p_con": 0.05,
                                "p_neu": 0.90,
                            },
                            {
                                "evidence_id": "s_bridge",
                                "raw_triple": ["B", "r_bridge", "C"],
                                "pool": "S",
                                "p_ent": 0.55,
                                "p_con": 0.10,
                                "p_neu": 0.35,
                            },
                            {
                                "evidence_id": "c1",
                                "raw_triple": ["B", "r_refute", "C"],
                                "pool": "C",
                                "p_ent": 0.05,
                                "p_con": 0.90,
                                "p_neu": 0.05,
                            },
                        ],
                        # Deliberately stale/low; run_m2 should recompute from A+C.
                        "sufficiency": {"esi_geom": 0.01},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            run_dir = run_smoke_m2(
                input_jsonl=claims_path,
                output_root=base / "component2_logs",
                max_claims=1,
            )
            rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertGreater(row["esi_geom_before"], 0.30)
            self.assertFalse(row["should_recover"])
            self.assertEqual(row["moved_evidence_ids"], [])

    def test_run_smoke_m2_rejects_non_positive_max_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_smoke_m2(input_jsonl=None, output_root=Path(tmp_dir), max_claims=0)

    def test_run_smoke_m2_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                run_smoke_m2(
                    input_jsonl=Path(tmp_dir) / "missing.jsonl",
                    output_root=Path(tmp_dir),
                    max_claims=1,
                )

    def test_cli_mask_overrides_are_reflected_in_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(__file__).resolve().parents[2]
            output_root = Path(tmp_dir) / "m2_logs"

            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "component2.run_m2",
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
