"""Tests for Component 1 pair-log ingestion into Component 3 format."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pd = None  # type: ignore[assignment]

try:
    from component3.c1_pairs_loader import load_component1_evidence_rows, load_component1_evidence_rows_fever
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    load_component1_evidence_rows = None  # type: ignore[assignment]
    load_component1_evidence_rows_fever = None  # type: ignore[assignment]


@unittest.skipUnless(
    pd is not None and load_component1_evidence_rows is not None and load_component1_evidence_rows_fever is not None,
    "pandas deps unavailable",
)
class C1PairsLoaderTests(unittest.TestCase):
    def _claims_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sentence": ["c0", "c1", "c2"],
                "Label": [[1], [0], [1]],
            }
        )

    def _subgraphs_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "walked": [
                    {"connected": [["A", "r1", "B"]], "walkable": [["B", "r2", "C"]]},
                    {"connected": [["X", "r", "Y"]], "walkable": []},
                    {"connected": [], "walkable": [["M", "r3", "N"]]},
                ]
            }
        )

    def _write_pairs(self, root: Path) -> None:
        split_dir = root / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        pairs_file = split_dir / "pairs.jsonl"
        rows = [
            {
                "claim_id": "train_0",
                "evidence_id": "e0",
                "pool": "a",
                "raw_triple": ["A", "r1", "B"],
                "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                "derived": {"rel": 0.9},
            },
            {
                "claim_id": "train_0",
                "evidence_id": "e1",
                "pool": "C",
                "raw_triple": ["B", "r2", "C"],
                "probs": {"entail": 0.1, "contra": 0.8, "neutral": 0.1},
                "derived": {"rel": 0.9},
            },
            {
                # Missing `pool`; falls back to evidence_assignment and rel to p_ent+p_con.
                "claim_id": "train_1",
                "evidence_id": "e2",
                "evidence_assignment": "s",
                "raw_triple": ["X", "r", "Y"],
                "probs": {"entail": "0.25", "contra": "0.50", "neutral": "0.25"},
                "derived": {},
            },
        ]
        with pairs_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _write_pairs_with_empty_sentinel(self, root: Path) -> None:
        split_dir = root / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        pairs_file = split_dir / "pairs.jsonl"
        rows = [
            {
                "schema_version": 2,
                "record_type": "pair",
                "claim_id": "train_0",
                "evidence_id": "e0",
                "pool": "A",
                "raw_triple": ["A", "r1", "B"],
                "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                "derived": {"rel": 0.9},
            },
            {
                "schema_version": 2,
                "record_type": "claim_sentinel",
                "empty_evidence_sentinel": True,
                "claim_id": "train_1",
                "evidence_id": "train_1_empty_sentinel",
                "pool": "S",
                "raw_triple": None,
                "raw_sentence": "__EMPTY_EVIDENCE_SENTINEL__",
                "probs": {"entail": 0.0, "contra": 0.0, "neutral": 1.0},
                "derived": {"rel": 0.0},
            },
        ]
        with pairs_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _fever_claims_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "claim_id": ["train_101", "train_102"],
                "Sentence": ["Roman Atwood is a content creator.", "Tilda Swinton is a vegan."],
                "Label": [1, 0],
                "evidence_rows": [
                    [
                        {
                            "evidence_id": "train_101_ev_0",
                            "text": "Roman Atwood is an American YouTube personality.",
                            "pool": "A",
                            "is_gold": True,
                        }
                    ],
                    [
                        {
                            "evidence_id": "train_102_ev_0",
                            "text": "No evidence sentence available for this claim.",
                            "pool": "A",
                            "is_gold": True,
                        }
                    ],
                ],
            }
        )

    def _write_fever_pairs(self, root: Path) -> None:
        split_dir = root / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        pairs_file = split_dir / "pairs.jsonl"
        rows = [
            {
                "claim_id": "train_101",
                "evidence_id": "ev_sentence_1",
                "pool": "a",
                "raw_sentence": "Roman Atwood is an American YouTube personality.",
                "sentence_page": "Roman_Atwood",
                "sentence_line": 1,
                "probs": {"entail": 0.81, "contra": 0.09, "neutral": 0.10},
                "derived": {"rel": 0.90},
            }
        ]
        with pairs_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_grouping_and_parsing_with_hybrid_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_root = Path(tmp_dir) / "logs" / "component1"
            self._write_pairs(logs_root)

            rows, coverage = load_component1_evidence_rows(
                split="train",
                claims_df=self._claims_df(),
                subgraphs_df=self._subgraphs_df(),
                logs_root=logs_root,
                missing_policy="hybrid_fallback",
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 2)
            self.assertEqual(len(rows[1]), 1)
            self.assertGreaterEqual(len(rows[2]), 1)  # Missing claim uses subgraph fallback

            first = rows[0][0]
            self.assertEqual(first["evidence_id"], "e0")
            self.assertEqual(first["raw_triple"], ["A", "r1", "B"])
            self.assertEqual(first["pool"], "A")  # normalized
            self.assertAlmostEqual(float(first["p_ent"]), 0.8, places=6)
            self.assertAlmostEqual(float(first["p_con"]), 0.1, places=6)
            self.assertAlmostEqual(float(first["rel"]), 0.9, places=6)

            second_claim_row = rows[1][0]
            self.assertEqual(second_claim_row["pool"], "S")
            self.assertAlmostEqual(float(second_claim_row["rel"]), 0.75, places=6)  # p_ent + p_con fallback

            self.assertEqual(coverage["claims_total"], 3)
            self.assertEqual(coverage["claims_with_component1_pairs"], 2)
            self.assertEqual(coverage["claims_missing_component1_pairs"], 1)
            self.assertEqual(coverage["claims_using_fallback"], 1)
            self.assertEqual(coverage["pair_rows_parsed"], 3)
            self.assertGreater(coverage["pair_rows_nonzero_pv"], 0)

    def test_missing_claims_are_empty_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_root = Path(tmp_dir) / "logs" / "component1"
            self._write_pairs(logs_root)

            rows, coverage = load_component1_evidence_rows(
                split="train",
                claims_df=self._claims_df(),
                subgraphs_df=self._subgraphs_df(),
                logs_root=logs_root,
                missing_policy="strict",
            )

            self.assertEqual(len(rows[2]), 0)  # missing claim not backfilled in strict mode
            self.assertEqual(coverage["claims_using_fallback"], 0)
            self.assertEqual(coverage["claims_with_empty_evidence"], 1)

    def test_empty_evidence_sentinel_avoids_fallback_and_yields_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_root = Path(tmp_dir) / "logs" / "component1"
            self._write_pairs_with_empty_sentinel(logs_root)

            rows, coverage = load_component1_evidence_rows(
                split="train",
                claims_df=self._claims_df(),
                subgraphs_df=self._subgraphs_df(),
                logs_root=logs_root,
                missing_policy="hybrid_fallback",
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 1)
            self.assertEqual(len(rows[1]), 0)  # claim_sentinel parses but emits no graph triples
            self.assertGreaterEqual(len(rows[2]), 1)  # truly missing claim still fallbacks
            self.assertEqual(coverage["claims_with_component1_pairs"], 2)
            self.assertEqual(coverage["claims_using_fallback"], 1)
            self.assertEqual(coverage["claims_with_empty_sentinel"], 1)
            self.assertEqual(coverage["pair_rows_sentinel"], 1)

    def test_fever_loader_parses_sentence_pairs_and_fallbacks_to_evidence_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_root = Path(tmp_dir) / "logs" / "component1_fever"
            self._write_fever_pairs(logs_root)
            claims_df = self._fever_claims_df()

            rows, coverage = load_component1_evidence_rows_fever(
                split="train",
                claims_df=claims_df,
                logs_root=logs_root,
                missing_policy="hybrid_fallback",
            )

            self.assertEqual(len(rows), 2)
            # First claim comes from pair logs and contains parsed sentence payload.
            self.assertEqual(rows[0][0]["evidence_id"], "ev_sentence_1")
            self.assertEqual(rows[0][0]["raw_sentence"], "Roman Atwood is an American YouTube personality.")
            self.assertIsNone(rows[0][0]["raw_triple"])
            self.assertEqual(rows[0][0]["page"], "Roman_Atwood")
            self.assertEqual(rows[0][0]["line"], 1)
            # Second claim has no pair log row; fallback uses FEVER evidence_rows from dataframe.
            self.assertEqual(rows[1][0]["evidence_id"], "train_102_ev_0")
            self.assertEqual(rows[1][0]["raw_sentence"], "No evidence sentence available for this claim.")
            self.assertTrue(rows[1][0]["is_fallback"])

            self.assertEqual(coverage["claims_total"], 2)
            self.assertEqual(coverage["claims_with_component1_pairs"], 1)
            self.assertEqual(coverage["claims_using_fallback"], 1)


if __name__ == "__main__":
    unittest.main()
