"""Deterministic subset selection tests for Component 2 ablation runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from component2.ablation_utils import load_claims_for_ablation, select_deterministic_subset


class AblationSubsetDeterminismTests(unittest.TestCase):
    def test_same_seed_yields_same_claim_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            claims_path = base / "claims.jsonl"

            rows = []
            for idx in range(25):
                rows.append(
                    {
                        "claim_id": f"c_{idx}",
                        "claim_text": f"Claim {idx}",
                        "label": "TRUE" if idx % 2 == 0 else "FALSE",
                        "entity_set": ["A", "C"],
                        "sufficiency": {"esi_geom": 0.2 if idx % 2 == 0 else 0.8},
                        "triples": [
                            {
                                "evidence_id": f"{idx}_a",
                                "raw_triple": ["A", "r1", "B"],
                                "pool": "A",
                                "p_ent": 0.8,
                                "p_con": 0.1,
                                "p_neu": 0.1,
                            },
                            {
                                "evidence_id": f"{idx}_s",
                                "raw_triple": ["B", "r2", "C"],
                                "pool": "S",
                                "p_ent": 0.6,
                                "p_con": 0.2,
                                "p_neu": 0.2,
                            },
                        ],
                    }
                )

            claims_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            claims, source = load_claims_for_ablation(claims_path)
            self.assertEqual(source, "triples_jsonl")

            s1 = select_deterministic_subset(claims=claims, n_claims=10, seed=13)
            s2 = select_deterministic_subset(claims=claims, n_claims=10, seed=13)
            s3 = select_deterministic_subset(claims=claims, n_claims=10, seed=99)

            ids_1 = [claim.claim_id for claim in s1.claims]
            ids_2 = [claim.claim_id for claim in s2.claims]
            ids_3 = [claim.claim_id for claim in s3.claims]

            self.assertEqual(ids_1, ids_2)
            self.assertNotEqual(ids_1, ids_3)


if __name__ == "__main__":
    unittest.main()
