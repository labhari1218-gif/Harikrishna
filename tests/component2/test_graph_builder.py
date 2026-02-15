"""Tests for Component 2 graph builder."""

from __future__ import annotations

import unittest

from component2.graph_builder import GraphBuilder
from component2.types import evidence_from_dict


class GraphBuilderTests(unittest.TestCase):
    def test_build_graph_uses_a_union_c_with_features(self) -> None:
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
                    "evidence_id": "s1",
                    "raw_triple": ["B", "r2", "C"],
                    "pool": "S",
                    "p_ent": 0.4,
                    "p_con": 0.2,
                    "p_neu": 0.4,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "c1",
                    "raw_triple": ["B", "r3", "D"],
                    "pool": "C",
                    "p_ent": 0.05,
                    "p_con": 0.8,
                    "p_neu": 0.15,
                }
            ),
        ]

        builder = GraphBuilder(include_pools=("A", "C"))
        graph = builder.build("claim_1", triples, anchors=["A", "D"])

        self.assertEqual(graph.claim_id, "claim_1")
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual({edge.evidence_id for edge in graph.edges}, {"a1", "c1"})

        a_edge = next(edge for edge in graph.edges if edge.evidence_id == "a1")
        c_edge = next(edge for edge in graph.edges if edge.evidence_id == "c1")

        self.assertAlmostEqual(a_edge.rel, 0.8)
        self.assertAlmostEqual(a_edge.pol, 0.6)
        self.assertAlmostEqual(c_edge.rel, 0.85)
        self.assertAlmostEqual(c_edge.pol, -0.75)

        self.assertIn("A", graph.node_to_idx)
        self.assertIn("D", graph.node_to_idx)
        self.assertTrue(graph.anchor_indices)

    def test_relation_vocab_is_shared_across_graphs(self) -> None:
        builder = GraphBuilder(include_pools=("A", "C"))

        g1 = builder.build(
            "claim_1",
            [
                evidence_from_dict(
                    {
                        "evidence_id": "a1",
                        "raw_triple": ["A", "r1", "B"],
                        "pool": "A",
                        "p_ent": 0.6,
                        "p_con": 0.2,
                        "p_neu": 0.2,
                    }
                )
            ],
            anchors=["A"],
        )
        g2 = builder.build(
            "claim_2",
            [
                evidence_from_dict(
                    {
                        "evidence_id": "a2",
                        "raw_triple": ["X", "r1", "Y"],
                        "pool": "A",
                        "p_ent": 0.6,
                        "p_con": 0.2,
                        "p_neu": 0.2,
                    }
                ),
                evidence_from_dict(
                    {
                        "evidence_id": "a3",
                        "raw_triple": ["Y", "r2", "Z"],
                        "pool": "C",
                        "p_ent": 0.1,
                        "p_con": 0.7,
                        "p_neu": 0.2,
                    }
                ),
            ],
            anchors=["Y"],
        )

        rel_ids_1 = {edge.relation: edge.relation_id for edge in g1.edges}
        rel_ids_2 = {edge.relation: edge.relation_id for edge in g2.edges}

        self.assertEqual(rel_ids_1["r1"], rel_ids_2["r1"])
        self.assertNotEqual(rel_ids_2["r1"], rel_ids_2["r2"])


if __name__ == "__main__":
    unittest.main()
