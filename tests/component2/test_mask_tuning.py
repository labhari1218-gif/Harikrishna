"""Tests for Component 2 mask tuning helpers."""

from __future__ import annotations

import unittest

from component2.config import DEFAULT_CONFIG
from component2.mask_tuning import MaskTuningSettings, tune_mask_coefficients
from component2.types import claim_from_dict


def _synthetic_claims(num_claims: int = 6):
    rows = []
    for idx in range(num_claims):
        rows.append(
            {
                "claim_id": f"tune_{idx}",
                "claim_text": "A connects to C",
                "label": "SUPPORTED" if idx % 2 == 0 else "REFUTED",
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
    return [claim_from_dict(row) for row in rows]


class MaskTuningTests(unittest.TestCase):
    def test_two_stage_tuning_is_deterministic(self) -> None:
        claims = _synthetic_claims(6)
        settings = MaskTuningSettings(
            strategy="two_stage",
            grid_size=3,
            tune_claims=5,
            local_refine=False,
        )

        first = tune_mask_coefficients(claims=claims, config=DEFAULT_CONFIG, settings=settings)
        second = tune_mask_coefficients(claims=claims, config=DEFAULT_CONFIG, settings=settings)

        self.assertEqual(first["best_params"], second["best_params"])
        self.assertEqual(first["report"]["best_params"], second["report"]["best_params"])

    def test_two_stage_candidate_counts_match_expected_defaults(self) -> None:
        claims = _synthetic_claims(6)
        settings = MaskTuningSettings(
            strategy="two_stage",
            grid_size=3,
            tune_claims=5,
            local_refine=False,
        )

        result = tune_mask_coefficients(claims=claims, config=DEFAULT_CONFIG, settings=settings)
        report = result["report"]

        self.assertEqual(report["claims_used"], 5)
        self.assertEqual(report["candidates_evaluated"], 20)
        self.assertEqual(report["stages"]["fit_independent"]["candidates_evaluated"], 18)
        self.assertEqual(report["stages"]["validate"]["candidates_evaluated"], 2)

    def test_joint_strategy_candidate_count_matches_grid(self) -> None:
        claims = _synthetic_claims(6)
        settings = MaskTuningSettings(
            strategy="joint",
            fit_mode="A1",
            grid_size=2,
            tune_claims=4,
            local_refine=False,
        )

        result = tune_mask_coefficients(claims=claims, config=DEFAULT_CONFIG, settings=settings)
        report = result["report"]

        # joint search explores (|alpha|*|beta|)^2 coefficient tuples
        self.assertEqual(report["claims_used"], 4)
        self.assertEqual(report["candidates_evaluated"], 16)
        self.assertIn("fit_joint", report["stages"])
        self.assertEqual(report["stages"]["fit_joint"]["candidates_evaluated"], 16)


if __name__ == "__main__":
    unittest.main()
