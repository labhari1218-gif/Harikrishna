"""Tests for Component 2 anchor selection."""

from __future__ import annotations

import unittest

from component2.anchor_selector import AnchorSelector
from component2.config import DEFAULT_CONFIG
from component2.types import evidence_from_dict


class AnchorSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = AnchorSelector(config=DEFAULT_CONFIG)
        self.triples = [
            evidence_from_dict(
                {
                    "evidence_id": "e1",
                    "raw_triple": ["Barack_Obama", "birthPlace", "Honolulu"],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e2",
                    "raw_triple": ["Honolulu", "isPartOf", "Hawaii"],
                    "pool": "A",
                    "p_ent": 0.6,
                    "p_con": 0.1,
                    "p_neu": 0.3,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e3",
                    "raw_triple": ["Barack_Obama", "spouse", "Michelle_Obama"],
                    "pool": "C",
                    "p_ent": 0.1,
                    "p_con": 0.7,
                    "p_neu": 0.2,
                }
            ),
        ]

    def test_seed_anchors_filtered_to_graph_entities(self) -> None:
        anchors = self.selector.select_anchors(
            claim_text="irrelevant",
            triples=self.triples,
            seed_entities=["Barack_Obama", "Mars", "Hawaii"],
        )

        self.assertEqual(anchors, ["Barack_Obama", "Hawaii"])

    def test_text_fallback_when_seed_missing(self) -> None:
        anchors = self.selector.select_anchors(
            claim_text="Barack Obama was born in Hawaii.",
            triples=self.triples,
            seed_entities=["Unknown_Entity"],
        )

        self.assertIn("Barack_Obama", anchors)
        self.assertIn("Hawaii", anchors)

    def test_single_seed_anchor_is_augmented_to_enable_pairs(self) -> None:
        anchors = self.selector.select_anchors(
            claim_text="No matching entities here",
            triples=self.triples,
            seed_entities=["Barack_Obama"],
        )

        self.assertGreaterEqual(len(anchors), 2)
        self.assertEqual(anchors[0], "Barack_Obama")

    def test_degree_fallback_when_no_seed_and_no_text_match(self) -> None:
        anchors = self.selector.select_anchors(
            claim_text="No matching entities here",
            triples=self.triples,
            seed_entities=[],
        )

        self.assertEqual(len(anchors), DEFAULT_CONFIG.anchor_fallback_degree_k)
        self.assertEqual(anchors[0], "Barack_Obama")

    def test_blank_entities_are_not_selected_as_anchors(self) -> None:
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "b1",
                    "raw_triple": ["A", "rel", ""],
                    "pool": "A",
                    "p_ent": 0.7,
                    "p_con": 0.1,
                    "p_neu": 0.2,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "b2",
                    "raw_triple": ["A", "rel2", "B"],
                    "pool": "A",
                    "p_ent": 0.6,
                    "p_con": 0.1,
                    "p_neu": 0.3,
                }
            ),
        ]
        anchors = self.selector.select_anchors(
            claim_text="A relates to B",
            triples=triples,
            seed_entities=["A"],
        )

        self.assertNotIn("", anchors)
        self.assertIn("B", anchors)


if __name__ == "__main__":
    unittest.main()
