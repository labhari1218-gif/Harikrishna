"""Tests for Component 2 type normalization and validation."""

from __future__ import annotations

import unittest

from component2.types import claim_from_dict, evidence_from_dict


class TypesValidationTests(unittest.TestCase):
    def test_evidence_from_dict_requires_probability_fields(self) -> None:
        with self.assertRaises(ValueError):
            evidence_from_dict(
                {
                    "evidence_id": "e1",
                    "raw_triple": ["A", "r", "B"],
                    "pool": "A",
                }
            )

    def test_evidence_from_dict_rejects_out_of_range_probability(self) -> None:
        with self.assertRaises(ValueError):
            evidence_from_dict(
                {
                    "evidence_id": "e1",
                    "raw_triple": ["A", "r", "B"],
                    "pool": "A",
                    "p_ent": 1.2,
                    "p_con": 0.0,
                    "p_neu": 0.0,
                }
            )

    def test_claim_from_dict_rejects_duplicate_evidence_ids(self) -> None:
        with self.assertRaises(ValueError):
            claim_from_dict(
                {
                    "claim_id": "c0",
                    "claim_text": "A relates to B",
                    "triples": [
                        {
                            "evidence_id": "dup",
                            "raw_triple": ["A", "r1", "B"],
                            "pool": "A",
                            "p_ent": 0.7,
                            "p_con": 0.2,
                            "p_neu": 0.1,
                        },
                        {
                            "evidence_id": "dup",
                            "raw_triple": ["B", "r2", "C"],
                            "pool": "A",
                            "p_ent": 0.6,
                            "p_con": 0.2,
                            "p_neu": 0.2,
                        },
                    ],
                }
            )

    def test_claim_from_dict_requires_claim_id(self) -> None:
        with self.assertRaises(ValueError):
            claim_from_dict(
                {
                    "claim_text": "Missing id",
                    "triples": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
