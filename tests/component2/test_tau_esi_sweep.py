"""Smoke tests for T16 tau_esi sensitivity sweep."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from component2.ablation_runner import run_tau_esi_sweep


class TauEsiSweepSmokeTests(unittest.TestCase):
    def _write_claims(self, path: Path, count: int) -> None:
        rows = []
        for idx in range(count):
            rows.append(
                {
                    "claim_id": f"tau_{idx}",
                    "claim_text": "A connects to C",
                    "label": "TRUE" if idx % 2 == 0 else "FALSE",
                    "entity_set": ["A", "C"],
                    "sufficiency": {"esi_geom": 0.2 if idx % 2 == 0 else 0.7},
                    "triples": [
                        {
                            "evidence_id": f"{idx}_a1",
                            "raw_triple": ["A", "r_left", "B"],
                            "pool": "A",
                            "p_ent": 0.70,
                            "p_con": 0.10,
                            "p_neu": 0.20,
                        },
                        {
                            "evidence_id": f"{idx}_a2",
                            "raw_triple": ["X", "r_right", "C"],
                            "pool": "A",
                            "p_ent": 0.66,
                            "p_con": 0.11,
                            "p_neu": 0.23,
                        },
                        {
                            "evidence_id": f"{idx}_s_bridge",
                            "raw_triple": ["B", "r_bridge", "C"],
                            "pool": "S",
                            "p_ent": 0.57,
                            "p_con": 0.09,
                            "p_neu": 0.34,
                        },
                        {
                            "evidence_id": f"{idx}_s_noise",
                            "raw_triple": ["B", "r_noise", "Y"],
                            "pool": "S",
                            "p_ent": 0.82,
                            "p_con": 0.04,
                            "p_neu": 0.14,
                        },
                        {
                            "evidence_id": f"{idx}_c1",
                            "raw_triple": ["A", "r_refute", "Z"],
                            "pool": "C",
                            "p_ent": 0.06,
                            "p_con": 0.74,
                            "p_neu": 0.20,
                        },
                    ],
                }
            )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_tau_esi_sweep_outputs_tradeoff_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            self._write_claims(claims_path, count=8)

            run_dir = run_tau_esi_sweep(
                split="val",
                claims_jsonl=claims_path,
                n_claims=6,
                seed=13,
                tau_abstain=0.60,
                output_root=base / "logs" / "ablations" / "component2",
                tau_esi_values=(0.2, 0.3, 0.4),
                mode_id="A3",
            )

            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertTrue((run_dir / "tau_esi_sweep.json").exists())
            self.assertTrue((run_dir / "tau_esi_sweep.md").exists())

            sweep_payload = json.loads((run_dir / "tau_esi_sweep.json").read_text(encoding="utf-8"))
            rows = sweep_payload["rows"]
            self.assertEqual([round(float(row["tau_esi"]), 2) for row in rows], [0.2, 0.3, 0.4])

            for row in rows:
                self.assertIn("recovery_trigger_rate", row)
                self.assertIn("accuracy", row)
                self.assertIn("risk", row)
                self.assertIn("coverage", row)
                self.assertIn("accuracy_on_answered", row)
                tag = str(row["output_tag"])
                metrics_path = run_dir / f"metrics_{tag}.json"
                preds_path = run_dir / f"predictions_{tag}.jsonl"
                self.assertTrue(metrics_path.exists())
                self.assertTrue(preds_path.exists())
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertAlmostEqual(
                    float(metrics_payload["recovery_tau_esi"]),
                    float(row["tau_esi"]),
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()
