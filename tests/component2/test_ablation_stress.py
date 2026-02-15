"""Stress-mode smoke tests for Component 2 ablation runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from component2.ablation_runner import run_ablation_study


class AblationStressSmokeTests(unittest.TestCase):
    def _write_claims(self, path: Path, count: int) -> None:
        rows = []
        for idx in range(count):
            rows.append(
                {
                    "claim_id": f"stress_{idx}",
                    "claim_text": "A connects to C through a bridge",
                    "label": "TRUE" if idx % 2 == 0 else "FALSE",
                    "entity_set": ["A", "C"],
                    "sufficiency": {"esi_geom": 0.2},
                    "triples": [
                        {
                            "evidence_id": f"{idx}_a1",
                            "raw_triple": ["A", "r_left", "B"],
                            "pool": "A",
                            "p_ent": 0.75,
                            "p_con": 0.05,
                            "p_neu": 0.20,
                        },
                        {
                            "evidence_id": f"{idx}_a2",
                            "raw_triple": ["X", "r_right", "C"],
                            "pool": "A",
                            "p_ent": 0.68,
                            "p_con": 0.07,
                            "p_neu": 0.25,
                        },
                        {
                            "evidence_id": f"{idx}_s_bridge",
                            "raw_triple": ["B", "r_bridge", "C"],
                            "pool": "S",
                            "p_ent": 0.58,
                            "p_con": 0.10,
                            "p_neu": 0.32,
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
                            "p_ent": 0.08,
                            "p_con": 0.74,
                            "p_neu": 0.18,
                        },
                    ],
                }
            )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def _load_jsonl(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def test_a5_drop_bridge_stress_removes_bridge_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            self._write_claims(claims_path, count=6)

            run_dir = run_ablation_study(
                split="val",
                claims_jsonl=claims_path,
                n_claims=5,
                seed=7,
                tau_abstain=0.60,
                output_root=base / "logs" / "ablations" / "component2",
                ablation_ids=("A5",),
            )

            pred_rows = self._load_jsonl(run_dir / "predictions_A5.jsonl")
            self.assertEqual(len(pred_rows), 5)
            for row in pred_rows:
                self.assertGreaterEqual(int(row["num_stress_removed_bridge_triples"]), 1)
                self.assertEqual(int(row["num_stress_added_distractors"]), 0)
                removed = row["stress_removed_bridge_ids"]
                self.assertTrue(isinstance(removed, list))
                self.assertTrue(any(str(evidence_id).endswith("_s_bridge") for evidence_id in removed))

            metrics = json.loads((run_dir / "metrics_A5.json").read_text(encoding="utf-8"))
            stress = metrics["stress"]
            self.assertTrue(stress["drop_bridge_triples"])
            self.assertEqual(stress["add_distractors_per_claim"], 0)
            self.assertGreater(stress["claims_with_stress_mutation"], 0)

    def test_a6_add_distractors_stress_injects_expected_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            self._write_claims(claims_path, count=5)

            run_dir = run_ablation_study(
                split="val",
                claims_jsonl=claims_path,
                n_claims=5,
                seed=11,
                tau_abstain=0.60,
                output_root=base / "logs" / "ablations" / "component2",
                ablation_ids=("A6",),
            )

            pred_rows = self._load_jsonl(run_dir / "predictions_A6.jsonl")
            self.assertEqual(len(pred_rows), 5)
            for row in pred_rows:
                self.assertEqual(int(row["num_stress_removed_bridge_triples"]), 0)
                self.assertEqual(int(row["num_stress_added_distractors"]), 3)
                added = row["stress_added_distractor_ids"]
                self.assertTrue(isinstance(added, list))
                self.assertEqual(len(added), 3)
                self.assertTrue(all("__stress_distractor_" in str(evidence_id) for evidence_id in added))

            metrics = json.loads((run_dir / "metrics_A6.json").read_text(encoding="utf-8"))
            stress = metrics["stress"]
            self.assertFalse(stress["drop_bridge_triples"])
            self.assertEqual(stress["add_distractors_per_claim"], 3)
            self.assertAlmostEqual(float(stress["mean_added_distractors"]), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
