"""Integration test for M3 smoke runner."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from component2.config import DEFAULT_CONFIG
from component2.run_m3 import run_smoke_m3


class RunM3Tests(unittest.TestCase):
    def test_run_smoke_m3_writes_selective_and_salience_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_smoke_m3(input_jsonl=None, output_root=output_root, max_claims=2)

            run_config_path = run_dir / "run_config.json"
            preds_path = run_dir / "predictions.jsonl"
            summary_path = run_dir / "summary.json"
            rc_path = run_dir / "risk_coverage.json"

            self.assertTrue(run_config_path.exists())
            self.assertTrue(preds_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(rc_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertEqual(run_config["component2_config"]["abstain_alpha"], 0.5)
            self.assertEqual(run_config["component2_config"]["rationale_top_n"], 10)
            self.assertEqual(run_config["runtime"]["salience"]["top_n"], 10)
            self.assertEqual(run_config["runtime"]["input_source"], "synthetic")
            self.assertIn("anchors", run_config["runtime"])
            self.assertIn("claims_missing_pickle_mapping", run_config["runtime"]["anchors"])

            rows = [
                json.loads(line)
                for line in preds_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            first = rows[0]
            self.assertIn("abstain_score", first)
            self.assertIn("top_rationale_edges", first)
            self.assertLessEqual(len(first["top_rationale_edges"]), 10)
            self.assertIn("faithfulness_loo", first)
            self.assertIsInstance(first["faithfulness_loo"], dict)
            self.assertIn("anchor_source", first)
            self.assertIn("anchors_hash", first)
            self.assertIn("salience_results", first)
            self.assertIsInstance(first["salience_results"], dict)
            self.assertIn("faithfulness_drop", first["salience_results"])
            self.assertIn("ppr_diagnostics", first)
            self.assertIn("converged", first["ppr_diagnostics"])
            self.assertIn("recovery_metrics", first)
            self.assertIn("should_recover", first["recovery_metrics"])
            self.assertIn("esi_geom_before", first)
            self.assertIn("esi_geom_after", first)
            self.assertAlmostEqual(first["esi_geom"], first["esi_geom_after"], places=12)
            self.assertGreaterEqual(first["esi_geom_before"], 0.0)
            self.assertLessEqual(first["esi_geom_before"], 1.0)
            self.assertGreaterEqual(first["esi_geom_after"], 0.0)
            self.assertLessEqual(first["esi_geom_after"], 1.0)
            max_prob_after = max(first["prob_supported_after"], 1.0 - first["prob_supported_after"])
            expected_abstain = 0.5 * (1.0 - max_prob_after) + 0.5 * (1.0 - first["esi_geom_after"])
            self.assertAlmostEqual(first["abstain_score"], expected_abstain, places=12)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("aurc", summary)
            self.assertIn("num_faithful_claims", summary)
            self.assertIn("num_ppr_not_converged", summary)
            self.assertIn("num_disconnected_anchor_graphs", summary)
            self.assertIn("num_salience_checked", summary)
            self.assertIn("claims_missing_pickle_mapping", summary)

            rc_payload = json.loads(rc_path.read_text(encoding="utf-8"))
            self.assertIn("aurc", rc_payload)
            self.assertIn("points", rc_payload)

    def test_run_smoke_m3_rejects_non_positive_max_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_smoke_m3(input_jsonl=None, output_root=Path(tmp_dir), max_claims=0)

    def test_run_smoke_m3_supports_fixed_tau_operating_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            config = replace(
                DEFAULT_CONFIG,
                tau_abstain=0.55,
                compute_risk_coverage_sweep=False,
            )
            run_dir = run_smoke_m3(input_jsonl=None, output_root=output_root, max_claims=3, config=config)

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            rc_payload = json.loads((run_dir / "risk_coverage.json").read_text(encoding="utf-8"))

            self.assertIsNone(summary["aurc"])
            self.assertAlmostEqual(summary["tau_abstain"], 0.55, places=12)
            self.assertGreaterEqual(summary["coverage_at_tau"], 0.0)
            self.assertLessEqual(summary["coverage_at_tau"], 1.0)
            self.assertGreaterEqual(summary["risk_at_tau"], 0.0)
            self.assertLessEqual(summary["risk_at_tau"], 1.0)
            self.assertGreaterEqual(summary["accuracy_on_answered_at_tau"], 0.0)
            self.assertLessEqual(summary["accuracy_on_answered_at_tau"], 1.0)
            self.assertGreaterEqual(summary["abstain_rate_at_tau"], 0.0)
            self.assertLessEqual(summary["abstain_rate_at_tau"], 1.0)

            self.assertIsNone(rc_payload["aurc"])
            self.assertEqual(rc_payload["points"], [])
            self.assertIsInstance(rc_payload["operating_point"], dict)
            self.assertAlmostEqual(rc_payload["operating_point"]["tau_abstain"], 0.55, places=12)

    def test_cli_mask_overrides_are_reflected_in_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(__file__).resolve().parents[2]
            output_root = Path(tmp_dir) / "m3_logs"
            claims_path = Path(tmp_dir) / "claims.jsonl"
            claims_path.write_text(
                json.dumps(
                    {
                        "claim_id": "m3_cli_0",
                        "claim_text": "A relates to C",
                        "label": "SUPPORTED",
                        "entity_set": ["A", "C"],
                        "triples": [
                            {
                                "evidence_id": "a1",
                                "raw_triple": ["A", "r_left", "B"],
                                "pool": "A",
                                "p_ent": 0.7,
                                "p_con": 0.1,
                                "p_neu": 0.2,
                            },
                            {
                                "evidence_id": "s1",
                                "raw_triple": ["B", "r_bridge", "C"],
                                "pool": "S",
                                "p_ent": 0.55,
                                "p_con": 0.1,
                                "p_neu": 0.35,
                            },
                            {
                                "evidence_id": "c1",
                                "raw_triple": ["A", "r_refute", "Z"],
                                "pool": "C",
                                "p_ent": 0.05,
                                "p_con": 0.75,
                                "p_neu": 0.2,
                            },
                        ],
                        "sufficiency": {"esi_geom": 0.4},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "component2.run_m3",
                    "--input-jsonl",
                    str(claims_path),
                    "--max-claims",
                    "1",
                    "--output-root",
                    str(output_root),
                    "--no-risk-coverage-sweep",
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
