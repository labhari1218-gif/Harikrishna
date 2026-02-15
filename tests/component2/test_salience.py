"""Tests for M3 rationale extraction and faithfulness checks."""

from __future__ import annotations

import unittest

import numpy as np

from component2.graph_builder import GraphBuilder
from component2.salience import (
    extract_top_rationale_edges,
    grouped_edge_removal_faithfulness,
    leave_one_out_faithfulness,
)
from component2.types import evidence_from_dict


class _DummyReasoner:
    def __init__(self) -> None:
        self.weights = {
            "e1": 0.12,
            "e2": 0.08,
            "e3": 0.04,
        }

    def forward_graph(self, graph):
        score = sum(self.weights.get(edge.evidence_id, 0.0) for edge in graph.edges)
        prob_supported = min(0.99, max(0.01, 0.5 + 0.5 * score))
        return {"probs": np.array([prob_supported, 1.0 - prob_supported], dtype=np.float64)}


class SalienceTests(unittest.TestCase):
    def _graph(self):
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "e1",
                    "raw_triple": ["A", "r1", "B"],
                    "pool": "A",
                    "p_ent": 0.70,
                    "p_con": 0.10,
                    "p_neu": 0.20,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e2",
                    "raw_triple": ["B", "r2", "C"],
                    "pool": "A",
                    "p_ent": 0.30,
                    "p_con": 0.10,
                    "p_neu": 0.60,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e3",
                    "raw_triple": ["C", "r3", "D"],
                    "pool": "C",
                    "p_ent": 0.50,
                    "p_con": 0.20,
                    "p_neu": 0.30,
                }
            ),
        ]
        return GraphBuilder(include_pools=("A", "C")).build(
            claim_id="claim_sal",
            triples=triples,
            anchors=["A", "D"],
        )

    def test_extract_top_rationale_edges_uses_attention_times_pv(self) -> None:
        graph = self._graph()
        reasoner_output = {
            "salience": [
                {"evidence_id": "e1", "attention_sup": 0.20, "attention_ref": 0.10},
                {"evidence_id": "e2", "attention_sup": 0.50, "attention_ref": 0.20},
                {"evidence_id": "e3", "attention_sup": 0.30, "attention_ref": 0.10},
            ]
        }

        top = extract_top_rationale_edges(graph=graph, reasoner_output=reasoner_output, top_n=2)
        self.assertEqual([row["evidence_id"] for row in top], ["e3", "e2"])
        self.assertGreater(top[0]["salience_attention_pv"], top[1]["salience_attention_pv"])

    def test_extract_top_rationale_edges_can_use_predicted_stream(self) -> None:
        graph = self._graph()
        reasoner_output = {
            "salience": [
                {"evidence_id": "e1", "attention_sup": 0.60, "attention_ref": 0.05},
                {"evidence_id": "e2", "attention_sup": 0.25, "attention_ref": 0.70},
                {"evidence_id": "e3", "attention_sup": 0.20, "attention_ref": 0.65},
            ]
        }

        top_sup = extract_top_rationale_edges(
            graph=graph,
            reasoner_output=reasoner_output,
            top_n=1,
            target_class_idx=0,
        )
        top_ref = extract_top_rationale_edges(
            graph=graph,
            reasoner_output=reasoner_output,
            top_n=1,
            target_class_idx=1,
        )
        self.assertEqual(top_sup[0]["target_class_idx"], 0)
        self.assertEqual(top_ref[0]["target_class_idx"], 1)
        self.assertNotEqual(top_sup[0]["evidence_id"], top_ref[0]["evidence_id"])

    def test_leave_one_out_reports_positive_drop_for_informative_edges(self) -> None:
        graph = self._graph()
        reasoner = _DummyReasoner()
        rationale = [
            {"evidence_id": "e1"},
            {"evidence_id": "e2"},
        ]

        result = leave_one_out_faithfulness(
            reasoner=reasoner,
            graph=graph,
            rationale_edges=rationale,
            predicted_class_idx=0,
            max_edges=2,
        )

        self.assertEqual(result["num_tested_edges"], 2)
        self.assertTrue(result["is_faithful"])
        self.assertGreater(result["mean_confidence_drop"], 0.0)
        self.assertEqual(len(result["edge_drops"]), 2)

    def test_grouped_edge_removal_reports_joint_faithfulness_drop(self) -> None:
        graph = self._graph()
        reasoner = _DummyReasoner()
        rationale = [{"evidence_id": "e1"}, {"evidence_id": "e2"}]

        result = grouped_edge_removal_faithfulness(
            reasoner=reasoner,
            graph=graph,
            rationale_edges=rationale,
            predicted_class_idx=0,
            max_edges=2,
        )

        self.assertEqual(result["num_removed_edges"], 2)
        self.assertTrue(result["is_faithful"])
        self.assertGreater(result["faithfulness_drop"], 0.0)
        self.assertEqual(result["removed_edge_ids"], ["e1", "e2"])


if __name__ == "__main__":
    unittest.main()
