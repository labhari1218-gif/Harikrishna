"""Smoke tests for Component 2 ablation modes A0-A4."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from component2.ablation_runner import run_ablation_study


class AblationModesSmokeTests(unittest.TestCase):
    def _write_claims(self, path: Path, count: int) -> None:
        rows = []
        for idx in range(count):
            rows.append(
                {
                    "claim_id": f"smoke_{idx}",
                    "claim_text": "A connects to C",
                    "label": "TRUE" if idx % 2 == 0 else "FALSE",
                    "entity_set": ["A", "C"],
                    "sufficiency": {"esi_geom": 0.2 if idx % 3 == 0 else 0.7},
                    "triples": [
                        {
                            "evidence_id": f"{idx}_a1",
                            "raw_triple": ["A", "r_left", "B"],
                            "pool": "A",
                            "p_ent": 0.7,
                            "p_con": 0.1,
                            "p_neu": 0.2,
                        },
                        {
                            "evidence_id": f"{idx}_a2",
                            "raw_triple": ["X", "r_right", "C"],
                            "pool": "A",
                            "p_ent": 0.65,
                            "p_con": 0.1,
                            "p_neu": 0.25,
                        },
                        {
                            "evidence_id": f"{idx}_s1",
                            "raw_triple": ["B", "r_bridge", "C"],
                            "pool": "S",
                            "p_ent": 0.55,
                            "p_con": 0.1,
                            "p_neu": 0.35,
                        },
                        {
                            "evidence_id": f"{idx}_s2",
                            "raw_triple": ["B", "r_noise", "Y"],
                            "pool": "S",
                            "p_ent": 0.8,
                            "p_con": 0.05,
                            "p_neu": 0.15,
                        },
                        {
                            "evidence_id": f"{idx}_c1",
                            "raw_triple": ["A", "r_refute", "Z"],
                            "pool": "C",
                            "p_ent": 0.05,
                            "p_con": 0.75,
                            "p_neu": 0.2,
                        },
                    ],
                }
            )

        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_a0_to_a4_smoke_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            self._write_claims(claims_path, count=12)

            run_dir = run_ablation_study(
                split="val",
                claims_jsonl=claims_path,
                n_claims=10,
                seed=13,
                tau_abstain=0.60,
                output_root=base / "logs" / "ablations" / "component2",
            )

            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertTrue((run_dir / "selected_claim_ids.txt").exists())
            self.assertTrue((run_dir / "summary_table.md").exists())

            summary_text = (run_dir / "summary_table.md").read_text(encoding="utf-8")
            self.assertIn("| A0 ", summary_text)
            self.assertIn("| A1 ", summary_text)
            self.assertIn("| A2 ", summary_text)
            self.assertIn("| A3 ", summary_text)
            self.assertIn("| A4 ", summary_text)

            for ablation_id in ("A0", "A1", "A2", "A3", "A4"):
                preds_path = run_dir / f"predictions_{ablation_id}.jsonl"
                metrics_path = run_dir / f"metrics_{ablation_id}.json"
                self.assertTrue(preds_path.exists())
                self.assertTrue(metrics_path.exists())

                pred_lines = [line for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(pred_lines), 10)

                # Risk-coverage artifact exists as png or csv depending on matplotlib availability.
                risk_png = run_dir / "plots" / f"risk_coverage_{ablation_id}.png"
                risk_csv = run_dir / "plots" / f"risk_coverage_{ablation_id}.csv"
                self.assertTrue(risk_png.exists() or risk_csv.exists())

            metrics_a4 = json.loads((run_dir / "metrics_A4.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics_a4["disable_pv_masks"])

            bridge_hist_png = run_dir / "plots" / "histogram_bridgebonus_A2.png"
            bridge_hist_csv = run_dir / "plots" / "histogram_bridgebonus_A2.csv"
            self.assertTrue(bridge_hist_png.exists() or bridge_hist_csv.exists())

            delta_hist_png = run_dir / "plots" / "histogram_delta_conn_A3.png"
            delta_hist_csv = run_dir / "plots" / "histogram_delta_conn_A3.csv"
            self.assertTrue(delta_hist_png.exists() or delta_hist_csv.exists())

    def test_tuning_smoke_writes_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            self._write_claims(claims_path, count=6)

            run_dir = run_ablation_study(
                split="val",
                claims_jsonl=claims_path,
                n_claims=5,
                seed=13,
                tau_abstain=0.60,
                output_root=base / "logs" / "ablations" / "component2",
                ablation_ids=("A1", "A3"),
                tune_mask_coeffs=True,
                mask_tune_strategy="two_stage",
                mask_fit_mode="A1",
                mask_validate_mode="A3",
                mask_grid_size=3,
                mask_tune_claims=5,
                mask_local_refine=False,
            )

            self.assertTrue((run_dir / "best_mask_params.json").exists())
            self.assertTrue((run_dir / "mask_tuning_report.json").exists())

            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            best_mask_params = json.loads((run_dir / "best_mask_params.json").read_text(encoding="utf-8"))
            self.assertIn("mask_tuning", run_config)
            self.assertTrue(run_config["mask_tuning"]["enabled"])
            self.assertEqual(run_config["mask_tuning"]["strategy"], "two_stage")
            self.assertEqual(run_config["mask_tuning"]["best_params"], best_mask_params)

            tunables = run_config["component2_tunables"]
            for key in ("mask_sup_alpha", "mask_sup_beta", "mask_ref_alpha", "mask_ref_beta"):
                self.assertAlmostEqual(float(tunables[key]), float(best_mask_params[key]), places=12)

            for ablation_id in ("A1", "A3"):
                metrics = json.loads((run_dir / f"metrics_{ablation_id}.json").read_text(encoding="utf-8"))
                self.assertIn("mask_coefficients", metrics)
                coeffs = metrics["mask_coefficients"]
                for key in ("mask_sup_alpha", "mask_sup_beta", "mask_ref_alpha", "mask_ref_beta"):
                    self.assertAlmostEqual(float(coeffs[key]), float(best_mask_params[key]), places=12)


if __name__ == "__main__":
    unittest.main()
