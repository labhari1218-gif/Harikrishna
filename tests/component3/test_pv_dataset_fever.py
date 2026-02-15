"""Tests for FEVER PV sentence graph dataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pd = None  # type: ignore[assignment]

try:
    from component3.pv_dataset_fever import FeverPVDatasetGraph
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    FeverPVDatasetGraph = None  # type: ignore[assignment]


def _sentence_encoder(text: str):
    base = float(len(text))
    return [base / 10.0, (base + 1.0) / 10.0, (base + 2.0) / 10.0, (base + 3.0) / 10.0]


def _claim_sentence_encoder(claim: str, sentence: str):
    base = float(len(claim) + len(sentence))
    return [
        base / 100.0,
        (base + 1.0) / 100.0,
        (base + 2.0) / 100.0,
        (base + 3.0) / 100.0,
        (base + 4.0) / 100.0,
        (base + 5.0) / 100.0,
    ]


@unittest.skipUnless(pd is not None and FeverPVDatasetGraph is not None, "pandas/torch deps unavailable")
class FeverPVDatasetGraphTests(unittest.TestCase):
    def test_builds_sentence_graph_with_expected_contract(self) -> None:
        df = pd.DataFrame(
            {
                "claim_id": ["train_1"],
                "Sentence": ["Roman Atwood is a content creator."],
                "Label": [1],
            }
        )
        evidence_rows = [
            [
                {
                    "evidence_id": "ev_a",
                    "text": "Roman Atwood is an American YouTube personality.",
                    "pool": "A",
                    "p_ent": 0.82,
                    "p_con": 0.07,
                    "p_neu": 0.11,
                    "rel": 0.89,
                    "is_gold": True,
                },
                {
                    "evidence_id": "ev_c",
                    "raw_sentence": "Tilda Swinton is a vegan.",
                    "pool": "C",
                    "p_ent": 0.05,
                    "p_con": 0.81,
                    "p_neu": 0.14,
                    "rel": 0.86,
                    "is_gold": False,
                },
            ]
        ]

        dataset = FeverPVDatasetGraph(
            df=df,
            evidence=evidence_rows,
            sentence_encoder=_sentence_encoder,
            claim_sentence_encoder=_claim_sentence_encoder,
            sentence_dim=4,
            claim_sentence_dim=6,
            auto_precompute=False,
        )

        claim, graph, label = dataset[0]
        self.assertEqual(claim, "Roman Atwood is a content creator.")
        self.assertEqual(label, 1)

        # Two evidence nodes -> full directed graph has 4 edges.
        self.assertEqual(graph.x.shape, (2, 4))
        # edge_attr = claim_sentence(6) + PV tail(5)
        self.assertEqual(graph.edge_attr.shape, (4, 11))

        # Source-node pool IDs should expand across outgoing edges.
        pool_ids = graph.edge_attr[:, -1].tolist()
        self.assertEqual(pool_ids, [0.0, 0.0, 2.0, 2.0])

        # Gold labels follow source evidence row metadata.
        self.assertEqual(graph.edge_is_gold.tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_empty_rows_return_dummy_graph(self) -> None:
        df = pd.DataFrame(
            {
                "claim_id": ["train_2"],
                "Sentence": ["No evidence claim."],
                "Label": [0],
            }
        )

        dataset = FeverPVDatasetGraph(
            df=df,
            evidence=[[]],
            sentence_encoder=_sentence_encoder,
            claim_sentence_encoder=_claim_sentence_encoder,
            sentence_dim=4,
            claim_sentence_dim=6,
            auto_precompute=False,
        )

        _claim, graph, _label = dataset[0]
        self.assertEqual(tuple(graph.x.shape), (1, 4))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 1))
        self.assertEqual(tuple(graph.edge_attr.shape), (1, 11))

    def test_cache_invalidates_when_sentence_text_changes_for_same_evidence_id(self) -> None:
        df = pd.DataFrame(
            {
                "claim_id": ["train_3"],
                "Sentence": ["Roman Atwood is a content creator."],
                "Label": [1],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sentence_cache_path = Path(tmp_dir) / "sentence_cache.pkl"
            claim_sentence_cache_path = Path(tmp_dir) / "claim_sentence_cache.pkl"

            evidence_rows_v1 = [
                [
                    {
                        "evidence_id": "ev_stable",
                        "text": "Roman Atwood is an American YouTube personality.",
                        "pool": "A",
                    }
                ]
            ]
            dataset_v1 = FeverPVDatasetGraph(
                df=df,
                evidence=evidence_rows_v1,
                sentence_cache_path=sentence_cache_path,
                claim_sentence_cache_path=claim_sentence_cache_path,
                sentence_encoder=_sentence_encoder,
                claim_sentence_encoder=_claim_sentence_encoder,
                sentence_dim=4,
                claim_sentence_dim=6,
                auto_precompute=True,
            )
            vec_v1 = np.asarray(dataset_v1.sentence_embeddings["ev_stable"], dtype=np.float32).copy()
            pair_vec_v1 = np.asarray(
                dataset_v1.claim_sentence_embeddings[("train_3", "ev_stable")],
                dtype=np.float32,
            ).copy()

            evidence_rows_v2 = [
                [
                    {
                        "evidence_id": "ev_stable",
                        "text": "Roman Bernard Atwood is an American internet personality and prankster.",
                        "pool": "A",
                    }
                ]
            ]
            dataset_v2 = FeverPVDatasetGraph(
                df=df,
                evidence=evidence_rows_v2,
                sentence_cache_path=sentence_cache_path,
                claim_sentence_cache_path=claim_sentence_cache_path,
                sentence_encoder=_sentence_encoder,
                claim_sentence_encoder=_claim_sentence_encoder,
                sentence_dim=4,
                claim_sentence_dim=6,
                auto_precompute=False,
            )
            _claim, _graph, _label = dataset_v2[0]

            vec_v2 = np.asarray(dataset_v2.sentence_embeddings["ev_stable"], dtype=np.float32)
            pair_vec_v2 = np.asarray(
                dataset_v2.claim_sentence_embeddings[("train_3", "ev_stable")],
                dtype=np.float32,
            )
            self.assertFalse(np.allclose(vec_v1, vec_v2))
            self.assertFalse(np.allclose(pair_vec_v1, pair_vec_v2))

    def test_batched_precompute_is_deterministic_across_batch_sizes(self) -> None:
        df = pd.DataFrame(
            {
                "claim_id": ["c1", "c2"],
                "Sentence": ["Claim A", "Claim B"],
                "Label": [1, 0],
            }
        )
        evidence_rows = [
            [
                {"evidence_id": "ev1", "text": "Sentence one", "pool": "A"},
                {"evidence_id": "ev2", "text": "Sentence two", "pool": "C"},
            ],
            [
                {"evidence_id": "ev3", "text": "Sentence three", "pool": "A"},
                {"evidence_id": "ev4", "text": "Sentence four", "pool": "C"},
            ],
        ]

        ds_batch1 = FeverPVDatasetGraph(
            df=df,
            evidence=evidence_rows,
            sentence_encoder=_sentence_encoder,
            claim_sentence_encoder=_claim_sentence_encoder,
            sentence_dim=4,
            claim_sentence_dim=6,
            auto_precompute=False,
            precompute_batch_size=1,
        )
        ds_batch1.precompute_sentence_embeddings(force=True)
        ds_batch1.precompute_claim_sentence_embeddings(force=True)

        ds_batch3 = FeverPVDatasetGraph(
            df=df,
            evidence=evidence_rows,
            sentence_encoder=_sentence_encoder,
            claim_sentence_encoder=_claim_sentence_encoder,
            sentence_dim=4,
            claim_sentence_dim=6,
            auto_precompute=False,
            precompute_batch_size=3,
        )
        ds_batch3.precompute_sentence_embeddings(force=True)
        ds_batch3.precompute_claim_sentence_embeddings(force=True)

        self.assertEqual(list(ds_batch1.sentence_embeddings.keys()), list(ds_batch3.sentence_embeddings.keys()))
        self.assertEqual(
            list(ds_batch1.claim_sentence_embeddings.keys()),
            list(ds_batch3.claim_sentence_embeddings.keys()),
        )
        for key in ds_batch1.sentence_embeddings:
            self.assertTrue(np.allclose(ds_batch1.sentence_embeddings[key], ds_batch3.sentence_embeddings[key]))
        for key in ds_batch1.claim_sentence_embeddings:
            self.assertTrue(
                np.allclose(ds_batch1.claim_sentence_embeddings[key], ds_batch3.claim_sentence_embeddings[key])
            )


if __name__ == "__main__":
    unittest.main()
