"""Tests for M2 bridge rescue scoring."""

from __future__ import annotations

import unittest

from component2.bridge_rescue import BridgeRescuePPR
from component2.config import Component2Config, DEFAULT_CONFIG
from component2.graph_builder import GraphBuilder
from component2.types import evidence_from_dict


class BridgeRescueTests(unittest.TestCase):
    def test_bridge_bonus_uses_epsilon_and_neutral_cap(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "r1", "B"],
                    "pool": "A",
                    "p_ent": 0.70,
                    "p_con": 0.05,
                    "p_neu": 0.25,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "a2",
                    "raw_triple": ["C", "r2", "D"],
                    "pool": "A",
                    "p_ent": 0.60,
                    "p_con": 0.05,
                    "p_neu": 0.35,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_low_rel",
                    "raw_triple": ["B", "bridge_low", "C"],
                    "pool": "S",
                    "p_ent": 0.35,
                    "p_con": 0.05,
                    "p_neu": 0.10,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_high_rel",
                    "raw_triple": ["B", "bridge_high", "C"],
                    "pool": "S",
                    "p_ent": 0.55,
                    "p_con": 0.05,
                    "p_neu": 0.10,
                }
            ),
        ]

        graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_m2",
            triples=triples,
            anchors=["A", "C"],
        )
        scorer = BridgeRescuePPR(config=DEFAULT_CONFIG)
        scores = scorer.compute_bridge_scores(graph)
        by_id = {row.evidence_id: row for row in scores}

        low = by_id["s_low_rel"]
        high = by_id["s_high_rel"]
        self.assertGreater(high.weight, low.weight)
        self.assertGreater(high.bridge_bonus, low.bridge_bonus)
        self.assertGreater(high.bridge_bonus_capped, low.bridge_bonus_capped)

        for row in scores:
            expected_bonus = row.p_source * row.weight * row.p_target
            expected_capped = expected_bonus * ((1.0 - next(t.p_neu for t in triples if t.evidence_id == row.evidence_id)) ** 2.0)
            self.assertAlmostEqual(row.bridge_bonus, expected_bonus, places=12)
            self.assertAlmostEqual(row.bridge_bonus_capped, expected_capped, places=12)

    def test_diagnostics_flags_disconnected_anchor_graph(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "r1", "B"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "a2",
                    "raw_triple": ["C", "r2", "D"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
        ]
        graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_disconnected",
            triples=triples,
            anchors=["A", "C"],
        )
        scorer = BridgeRescuePPR(config=DEFAULT_CONFIG)
        _, diagnostics = scorer.compute_node_scores_with_diagnostics(graph)

        self.assertTrue(diagnostics.has_disconnected_anchors)
        self.assertEqual(diagnostics.anchor_pairs_total, 1)
        self.assertEqual(diagnostics.anchor_pairs_connected, 0)

    def test_diagnostics_marks_non_convergence_when_iteration_budget_is_tiny(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "r1", "B"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "a2",
                    "raw_triple": ["B", "r2", "C"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
        ]
        graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_tiny_budget",
            triples=triples,
            anchors=["A", "C"],
        )
        tiny_budget_cfg = Component2Config(ppr_max_iters=1, ppr_tolerance=0.0)
        scorer = BridgeRescuePPR(config=tiny_budget_cfg)
        _, diagnostics = scorer.compute_node_scores_with_diagnostics(graph)

        self.assertFalse(diagnostics.converged)
        self.assertEqual(diagnostics.iterations, 1)
        self.assertGreaterEqual(diagnostics.residual_l1, 0.0)

    def test_default_solver_converges_for_single_anchor_chain(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "r1", "B"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "a2",
                    "raw_triple": ["B", "r2", "C"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
        ]
        graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_default_budget",
            triples=triples,
            anchors=["A"],
        )
        scorer = BridgeRescuePPR(config=DEFAULT_CONFIG)
        _, diagnostics = scorer.compute_node_scores_with_diagnostics(graph)

        self.assertTrue(diagnostics.converged)
        self.assertLessEqual(diagnostics.residual_l1, DEFAULT_CONFIG.ppr_tolerance)


if __name__ == "__main__":
    unittest.main()
