"""Tests for Component 3 PV-aware dataset."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pd = None  # type: ignore[assignment]

try:
    from component3.pv_dataset import FactKGPVDatasetGraph
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    FactKGPVDatasetGraph = None  # type: ignore[assignment]


def _toy_encoder(claim_text: str, subject: str, relation: str, object_: str):
    """Deterministic tiny encoder for tests."""
    base = len(claim_text) + len(subject) + len(relation) + len(object_)
    return [float(base + i) / 100.0 for i in range(6)]


@unittest.skipUnless(pd is not None and FactKGPVDatasetGraph is not None, "pandas/torch deps unavailable")
class FactKGPVDatasetGraphTests(unittest.TestCase):
    def _build_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "claim_id": ["c1", "c2"],
                "Sentence": ["A relates to D", "B relates to C"],
                "Label": [[1], [0]],
            }
        )

    def _build_evidence(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "triples": [
                    [
                        {
                            "evidence_id": "e1",
                            "raw_triple": ["A", "r1", "B"],
                            "pool": "a",  # lower-case should normalize to A
                            "p_ent": 0.80,
                            "p_con": 0.10,
                            "p_neu": 0.10,
                            "rel": 0.90,
                        },
                        {
                            "evidence_id": "e2",
                            "raw_triple": ["B", "r2", "C"],
                            "pool": "S",
                            "p_ent": 0.50,
                            "p_con": 0.30,
                            "p_neu": 0.20,
                            "rel": 0.80,
                        },
                        {
                            "evidence_id": "e3",
                            "raw_triple": ["C", "r3", "D"],
                            "pool": "C",
                            "p_ent": 0.15,
                            "p_con": 0.75,
                            "p_neu": 0.10,
                            "rel": 0.90,
                        },
                    ],
                    [
                        {
                            "evidence_id": "e4",
                            "raw_triple": ["B", "r4", "C"],
                            "pool": "A",
                            "p_ent": 0.72,
                            "p_con": 0.14,
                            "p_neu": 0.14,
                            "rel": 0.86,
                        }
                    ],
                ]
            }
        )

    def _write_embeddings(self, path: Path) -> None:
        embeddings = {
            "A": [0.1, 0.2, 0.3, 0.4],
            "B": [0.5, 0.6, 0.7, 0.8],
            "C": [0.9, 1.0, 1.1, 1.2],
            "D": [1.3, 1.4, 1.5, 1.6],
        }
        with path.open("wb") as fp:
            pickle.dump(embeddings, fp)

    def test_missing_embeddings_file_raises_error(self) -> None:
        df = self._build_df().iloc[:1]
        evidence = self._build_evidence().iloc[:1]
        missing_path = Path("/tmp/does_not_exist_component3_embeddings.pkl")

        with self.assertRaises(FileNotFoundError):
            FactKGPVDatasetGraph(
                df=df,
                evidence=evidence,
                embeddings_path=missing_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                auto_precompute=False,
            )

    def test_graph_contract_includes_edge_is_gold_reverse_edges_and_rich_pv_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            embeddings_path = base / "embeddings.pkl"
            cache_path = base / "claim_triple_embeddings.pkl"
            self._write_embeddings(embeddings_path)

            dataset = FactKGPVDatasetGraph(
                df=self._build_df().iloc[:1],
                evidence=self._build_evidence().iloc[:1],
                embeddings_path=embeddings_path,
                claim_triple_cache_path=cache_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                add_reverse_edges=True,
                auto_precompute=True,
            )

            claim, graph, label = dataset[0]
            self.assertEqual(claim, "A relates to D")
            self.assertEqual(label, 1)

            # Only A and C triples remain (S is excluded): 2 logical edges.
            # Reverse edge mode duplicates each -> 4 edges total.
            self.assertEqual(graph.edge_index.shape[1], 4)
            # edge_attr = [claim_triple_embed(6), p_ent, p_con, p_neu, rel, pool_id]
            self.assertEqual(graph.edge_attr.shape[1], 11)
            self.assertEqual(graph.x.shape[1], 4)

            # pool IDs should map A->0 and C->2, duplicated by reverse edges.
            pool_ids = graph.edge_attr[:, -1].tolist()
            self.assertEqual(pool_ids, [0.0, 0.0, 2.0, 2.0])

            # A edges are gold (1), C edges are non-gold (0), duplicated by reverse edges.
            self.assertEqual(graph.edge_is_gold.tolist(), [1.0, 1.0, 0.0, 0.0])

            # C1-backed edge metadata should not be all zeros.
            self.assertGreater(float(graph.edge_attr[:, -5].abs().sum().item()), 0.0)  # p_ent
            self.assertGreater(float(graph.edge_attr[:, -4].abs().sum().item()), 0.0)  # p_con
            self.assertGreater(float(graph.edge_attr[:, -2].abs().sum().item()), 0.0)  # rel

            self.assertTrue(cache_path.exists())
            with cache_path.open("rb") as fp:
                cache = pickle.load(fp)
            self.assertEqual(len(cache), 2)

    def test_missing_entity_embedding_raises_without_hash_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            embeddings_path = base / "embeddings.pkl"
            cache_path = base / "claim_triple_embeddings.pkl"
            self._write_embeddings(embeddings_path)

            # Remove one required entity to assert strict no-fallback behavior.
            with embeddings_path.open("rb") as fp:
                emb = pickle.load(fp)
            emb.pop("D")
            with embeddings_path.open("wb") as fp:
                pickle.dump(emb, fp)

            dataset = FactKGPVDatasetGraph(
                df=self._build_df().iloc[:1],
                evidence=self._build_evidence().iloc[:1],
                embeddings_path=embeddings_path,
                claim_triple_cache_path=cache_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                auto_precompute=False,
            )

            with self.assertRaises(KeyError):
                _claim, _graph, _label = dataset[0]

    def test_require_pv_metadata_rejects_fallback_list_triples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            embeddings_path = base / "embeddings.pkl"
            self._write_embeddings(embeddings_path)

            df = pd.DataFrame(
                {
                    "claim_id": ["c1"],
                    "Sentence": ["A relates to B"],
                    "Label": [[1]],
                }
            )
            # List triple has no PV fields and should fail strict mode.
            evidence = [["A", "r", "B"]]

            dataset = FactKGPVDatasetGraph(
                df=df,
                evidence=[evidence],
                embeddings_path=embeddings_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                require_pv_metadata=True,
                auto_precompute=False,
            )

            with self.assertRaises(ValueError):
                _claim, _graph, _label = dataset[0]

    def test_require_claim_triple_cache_raises_on_missing_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            embeddings_path = base / "embeddings.pkl"
            cache_path = base / "claim_triple_embeddings.pkl"
            self._write_embeddings(embeddings_path)

            dataset = FactKGPVDatasetGraph(
                df=self._build_df().iloc[:1],
                evidence=self._build_evidence().iloc[:1],
                embeddings_path=embeddings_path,
                claim_triple_cache_path=cache_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                auto_precompute=False,
                require_claim_triple_cache=True,
            )

            with self.assertRaises(RuntimeError):
                _claim, _graph, _label = dataset[0]

    def test_precompute_claim_triple_embeddings_writes_deterministic_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            embeddings_path = base / "embeddings.pkl"
            cache_path = base / "claim_triple_embeddings.pkl"
            self._write_embeddings(embeddings_path)

            dataset = FactKGPVDatasetGraph(
                df=self._build_df(),
                evidence=self._build_evidence(),
                embeddings_path=embeddings_path,
                claim_triple_cache_path=cache_path,
                claim_triple_encoder=_toy_encoder,
                claim_triple_dim=6,
                auto_precompute=False,
                precompute_batch_size=1,
            )
            dataset.precompute_claim_triple_embeddings(force=True)

            with cache_path.open("rb") as fp:
                cache = pickle.load(fp)
            keys = list(cache.keys())
            self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
