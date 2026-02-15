"""Tests for M2 recovery predictor and S->A policy."""

from __future__ import annotations

import unittest

from component2.bridge_rescue import BridgeRescuePPR
from component2.config import DEFAULT_CONFIG
from component2.graph_builder import GraphBuilder
from component2.recovery import BridgeRecoveryPolicy, connectivity_for_active
from component2.types import evidence_from_dict


class RecoveryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "left", "B"],
                    "pool": "A",
                    "p_ent": 0.70,
                    "p_con": 0.05,
                    "p_neu": 0.25,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "a2",
                    "raw_triple": ["C", "right", "D"],
                    "pool": "A",
                    "p_ent": 0.62,
                    "p_con": 0.06,
                    "p_neu": 0.32,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_bad1",
                    "raw_triple": ["B", "noise1", "X"],
                    "pool": "S",
                    "p_ent": 0.83,
                    "p_con": 0.05,
                    "p_neu": 0.12,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_bad2",
                    "raw_triple": ["B", "noise2", "Y"],
                    "pool": "S",
                    "p_ent": 0.82,
                    "p_con": 0.05,
                    "p_neu": 0.13,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_bad3",
                    "raw_triple": ["D", "noise3", "W"],
                    "pool": "S",
                    "p_ent": 0.81,
                    "p_con": 0.05,
                    "p_neu": 0.14,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_bridge",
                    "raw_triple": ["B", "bridge", "C"],
                    "pool": "S",
                    "p_ent": 0.52,
                    "p_con": 0.08,
                    "p_neu": 0.12,
                }
            ),
        ]
        self.anchors = ["A", "C"]

    def test_bridge_policy_improves_connectivity_when_rel_policy_fails(self) -> None:
        full_graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_recovery",
            triples=self.triples,
            anchors=self.anchors,
        )
        bridge_scores = BridgeRescuePPR(config=DEFAULT_CONFIG).compute_bridge_scores(full_graph)
        policy = BridgeRecoveryPolicy(config=DEFAULT_CONFIG)

        metrics = policy.evaluate(
            triples=self.triples,
            anchors=self.anchors,
            bridge_scores=bridge_scores,
            esi_geom=0.20,
        )

        self.assertEqual(metrics.conn_a, 0.0)
        self.assertEqual(metrics.rpi_rel_at_k, 0.0)
        self.assertGreater(metrics.rpi_bridge_at_k, 0.0)
        self.assertEqual(metrics.selected_rel_ids, ("s_bad1", "s_bad2", "s_bad3"))
        self.assertIn("s_bridge", metrics.selected_bridge_ids)
        self.assertTrue(metrics.should_recover)

        recovered = policy.apply_recovery(self.triples, metrics.selected_bridge_ids)
        self.assertIn("s_bridge", recovered.moved_evidence_ids)
        conn_after = connectivity_for_active(recovered.recovered_triples, self.anchors)
        self.assertGreater(conn_after, metrics.conn_a)

    def test_recovery_not_triggered_when_esi_is_high(self) -> None:
        full_graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_recovery",
            triples=self.triples,
            anchors=self.anchors,
        )
        bridge_scores = BridgeRescuePPR(config=DEFAULT_CONFIG).compute_bridge_scores(full_graph)
        policy = BridgeRecoveryPolicy(config=DEFAULT_CONFIG)

        metrics = policy.evaluate(
            triples=self.triples,
            anchors=self.anchors,
            bridge_scores=bridge_scores,
            esi_geom=0.75,
        )
        self.assertFalse(metrics.should_recover)

    def test_connectivity_marks_missing_anchor_as_disconnected(self) -> None:
        conn = connectivity_for_active(self.triples, anchors=["A", "Missing_Anchor"])
        self.assertEqual(conn, 0.0)

    def test_recovery_can_add_missing_anchor_into_active_graph(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "a1",
                    "raw_triple": ["A", "left", "B"],
                    "pool": "A",
                    "p_ent": 0.70,
                    "p_con": 0.05,
                    "p_neu": 0.25,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "s_bridge",
                    "raw_triple": ["B", "bridge", "C"],
                    "pool": "S",
                    "p_ent": 0.65,
                    "p_con": 0.05,
                    "p_neu": 0.10,
                }
            ),
        ]
        anchors = ["A", "C"]
        full_graph = GraphBuilder(include_pools=("A", "S", "C")).build(
            claim_id="claim_missing_anchor_bridge",
            triples=triples,
            anchors=anchors,
        )
        bridge_scores = BridgeRescuePPR(config=DEFAULT_CONFIG).compute_bridge_scores(full_graph)
        policy = BridgeRecoveryPolicy(config=DEFAULT_CONFIG)

        metrics = policy.evaluate(
            triples=triples,
            anchors=anchors,
            bridge_scores=bridge_scores,
            esi_geom=0.2,
        )

        self.assertEqual(metrics.conn_a, 0.0)
        self.assertGreater(metrics.delta_conn_bridge_at_k, 0.0)
        self.assertTrue(metrics.should_recover)

        recovered = policy.apply_recovery(triples, metrics.selected_bridge_ids)
        self.assertIn("s_bridge", recovered.moved_evidence_ids)
        conn_after = connectivity_for_active(recovered.recovered_triples, anchors)
        self.assertGreater(conn_after, metrics.conn_a)


if __name__ == "__main__":
    unittest.main()
