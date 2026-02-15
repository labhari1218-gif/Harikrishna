"""Tests for M3 Component 1 log input adaptation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from component2.run_m3 import run_smoke_m3


class RunM3Component1InputTests(unittest.TestCase):
    def test_component1_input_requires_pool_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            pairs_path = base / "pairs.jsonl"

            claims_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c0",
                                "claim_text": "A connects to C",
                                "label": "TRUE",
                                "counts": {"A": 1, "S": 1, "C": 0},
                                "sufficiency": {"esi_geom": 0.2},
                                "top_evidence": {"by_rel": [], "by_contra": []},
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pairs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c0",
                                "evidence_id": "c0_e1",
                                "raw_triple": ["A", "r1", "B"],
                                "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                run_smoke_m3(
                    input_jsonl=claims_path,
                    output_root=base / "component2_logs",
                    max_claims=1,
                )

    def test_component1_input_with_pairs_and_pools_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            pairs_path = base / "pairs.jsonl"

            claims_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c0",
                                "claim_text": "A connects to C",
                                "label": "TRUE",
                                "counts": {"A": 1, "S": 1, "C": 0},
                                "sufficiency": {"esi_geom": 0.2},
                                "top_evidence": {"by_rel": [], "by_contra": []},
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pairs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c0",
                                "evidence_id": "c0_e1",
                                "raw_triple": ["A", "r1", "B"],
                                "pool": "A",
                                "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                            }
                        ),
                        json.dumps(
                            {
                                "claim_id": "c0",
                                "evidence_id": "c0_e2",
                                "raw_triple": ["B", "r2", "C"],
                                "pool": "S",
                                "probs": {"entail": 0.6, "contra": 0.2, "neutral": 0.2},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run_dir = run_smoke_m3(
                input_jsonl=claims_path,
                output_root=base / "component2_logs",
                max_claims=1,
            )
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["runtime"]["input_source"], "component1_logs")

            rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["claim_id"], "c0")
            self.assertIn("ppr_diagnostics", rows[0])

    def test_component1_input_accepts_pool_alias_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"
            pairs_path = base / "pairs.jsonl"

            claims_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c_alias",
                                "claim_text": "A connects to C",
                                "label": "TRUE",
                                "counts": {"A": 1, "S": 1, "C": 0},
                                "sufficiency": {"esi_geom": 0.2},
                                "top_evidence": {"by_rel": [], "by_contra": []},
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pairs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "claim_id": "c_alias",
                                "evidence_id": "c_alias_e1",
                                "raw_triple": ["A", "r1", "B"],
                                "pool": "active",
                                "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                            }
                        ),
                        json.dumps(
                            {
                                "claim_id": "c_alias",
                                "evidence_id": "c_alias_e2",
                                "raw_triple": ["B", "r2", "C"],
                                "pool": "suspended",
                                "probs": {"entail": 0.6, "contra": 0.2, "neutral": 0.2},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run_dir = run_smoke_m3(
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
            self.assertEqual(rows[0]["claim_id"], "c_alias")


if __name__ == "__main__":
    unittest.main()
