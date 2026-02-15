"""Tests for M3 selective prediction metrics."""

from __future__ import annotations

import unittest

from component2.selective import (
    abstain_score,
    compute_aurc,
    evaluate_selective_prediction,
    risk_coverage_curve,
    selective_operating_point,
)


class SelectivePredictionTests(unittest.TestCase):
    def test_abstain_score_matches_locked_formula(self) -> None:
        score = abstain_score(probs=[0.80, 0.20], esi_geom=0.30, alpha=0.5)
        self.assertAlmostEqual(score, 0.45, places=12)

    def test_risk_coverage_curve_and_aurc(self) -> None:
        points = risk_coverage_curve(
            abstain_scores=[0.10, 0.20, 0.30, 0.40],
            is_correct=[True, False, True, False],
        )
        self.assertEqual(len(points), 4)
        self.assertAlmostEqual(points[0].coverage, 0.25, places=12)
        self.assertAlmostEqual(points[1].risk, 0.5, places=12)
        self.assertAlmostEqual(points[-1].coverage, 1.0, places=12)
        self.assertAlmostEqual(points[-1].risk, 0.5, places=12)

        aurc = compute_aurc(points)
        self.assertAlmostEqual(aurc, 0.2708333333333333, places=12)

    def test_evaluate_returns_serializable_points(self) -> None:
        result = evaluate_selective_prediction(
            abstain_scores=[0.1, 0.2],
            is_correct=[True, False],
        )
        self.assertIn("aurc", result)
        self.assertEqual(result["num_examples"], 2)
        self.assertEqual(len(result["points"]), 2)
        self.assertIn("coverage", result["points"][0])

    def test_abstain_score_validates_esi_range(self) -> None:
        with self.assertRaises(ValueError):
            abstain_score(probs=[0.7, 0.3], esi_geom=1.2, alpha=0.5)

    def test_selective_operating_point_metrics(self) -> None:
        metrics = selective_operating_point(
            abstain_scores=[0.10, 0.20, 0.80, 0.90],
            is_correct=[True, False, True, False],
            tau_abstain=0.20,
        )
        self.assertEqual(metrics.answered, 2)
        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.errors, 1)
        self.assertAlmostEqual(metrics.coverage, 0.5, places=12)
        self.assertAlmostEqual(metrics.risk, 0.5, places=12)
        self.assertAlmostEqual(metrics.accuracy_on_answered, 0.5, places=12)
        self.assertAlmostEqual(metrics.abstain_rate, 0.5, places=12)

    def test_evaluate_supports_fixed_tau_without_sweep(self) -> None:
        result = evaluate_selective_prediction(
            abstain_scores=[0.10, 0.20, 0.80, 0.90],
            is_correct=[True, False, True, False],
            tau_abstain=0.20,
            compute_curve=False,
        )
        self.assertIsNone(result["aurc"])
        self.assertEqual(result["points"], [])
        self.assertIsInstance(result["operating_point"], dict)
        self.assertAlmostEqual(result["operating_point"]["coverage"], 0.5, places=12)
        self.assertAlmostEqual(result["operating_point"]["risk"], 0.5, places=12)
        self.assertAlmostEqual(result["operating_point"]["accuracy_on_answered"], 0.5, places=12)
        self.assertAlmostEqual(result["operating_point"]["abstain_rate"], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
