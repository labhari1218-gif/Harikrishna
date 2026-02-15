"""Tests for FactKG anchor adapter injection and bridge-score impact."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
import warnings

from component2.anchor_adapter_factkg import FactKGAnchorAdapter
from component2.anchor_selector import AnchorSelector
from component2.bridge_rescue import BridgeRescuePPR
from component2.config import DEFAULT_CONFIG
from component2.graph_builder import GraphBuilder
from component2.types import claim_from_dict


def _write_factkg_train_pickle(path: Path) -> None:
    rows = {
        "Alpha links to Gamma": {
            "Label": [True],
            "Entity_set": ["Beta", "Delta"],
            "Evidence": {},
            "types": ["onehop"],
        }
    }
    with path.open("wb") as handle:
        pickle.dump(rows, handle)


def _claim_payload_without_entity_set() -> dict:
    return {
        "claim_id": "train_0",
        "claim_text": "Alpha links to Gamma",
        "label": "SUPPORTED",
        "triples": [
            {
                "evidence_id": "a1",
                "raw_triple": ["Alpha", "r_left", "Beta"],
                "pool": "A",
                "p_ent": 0.75,
                "p_con": 0.10,
                "p_neu": 0.15,
            },
            {
                "evidence_id": "a2",
                "raw_triple": ["Gamma", "r_right", "Delta"],
                "pool": "A",
                "p_ent": 0.74,
                "p_con": 0.10,
                "p_neu": 0.16,
            },
            {
                "evidence_id": "s_bridge",
                "raw_triple": ["Beta", "r_bridge", "Gamma"],
                "pool": "S",
                "p_ent": 0.60,
                "p_con": 0.10,
                "p_neu": 0.30,
            },
        ],
        "sufficiency": {"esi_geom": 0.2},
    }


class FactKGAnchorAdapterTests(unittest.TestCase):
    def _make_adapter(self, base_dir: Path) -> FactKGAnchorAdapter:
        return FactKGAnchorAdapter(
            factkg_dir=base_dir,
            split_pickles={"train": "factkg_train.pickle"},
        )

    def test_injects_entity_set_from_factkg_pickle_by_claim_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            factkg_dir = Path(tmp_dir)
            _write_factkg_train_pickle(factkg_dir / "factkg_train.pickle")
            adapter = self._make_adapter(factkg_dir)

            raw = {"claim_id": "train_0", "claim_text": "Alpha links to Gamma"}
            injected = adapter.inject_anchors(raw)

            self.assertEqual(injected["_anchor_source"], "pickle")
            self.assertEqual(injected["entity_set"], ["Beta", "Delta"])
            self.assertEqual(injected["Entity_set"], ["Beta", "Delta"])

            claim = claim_from_dict(_claim_payload_without_entity_set())
            injected_claim = adapter.inject_claim(claim)
            self.assertEqual(injected_claim.anchor_source, "pickle")
            self.assertEqual(injected_claim.claim.entity_set, ("Beta", "Delta"))

    def test_missing_pickle_mapping_falls_back_to_heuristic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            factkg_dir = Path(tmp_dir)
            _write_factkg_train_pickle(factkg_dir / "factkg_train.pickle")
            adapter = self._make_adapter(factkg_dir)

            raw = {"claim_id": "train_999", "claim_text": "Unknown claim text"}
            injected = adapter.inject_anchors(raw)

            self.assertEqual(injected["_anchor_source"], "heuristic")
            self.assertEqual(injected["entity_set"], [])

    def test_missing_factkg_pickle_emits_visibility_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            factkg_dir = Path(tmp_dir)
            adapter = self._make_adapter(factkg_dir)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                injected = adapter.inject_anchors({"claim_id": "train_0", "claim_text": "Unknown claim text"})
                # A second call should not emit another warning because loading is cached.
                _ = adapter.inject_anchors({"claim_id": "train_1", "claim_text": "Another claim text"})

            self.assertEqual(injected["_anchor_source"], "heuristic")
            self.assertEqual(len(caught), 1)
            message = str(caught[0].message)
            self.assertIn("FactKGAnchorAdapter load issues detected", message)
            self.assertIn("factkg_train.pickle", message)

    def test_bridge_scores_change_when_factkg_anchors_are_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            factkg_dir = Path(tmp_dir)
            _write_factkg_train_pickle(factkg_dir / "factkg_train.pickle")
            adapter = self._make_adapter(factkg_dir)

            claim = claim_from_dict(_claim_payload_without_entity_set())
            selector = AnchorSelector(config=DEFAULT_CONFIG)
            builder = GraphBuilder(include_pools=("A", "S", "C"))
            scorer = BridgeRescuePPR(config=DEFAULT_CONFIG)

            heuristic_anchors = selector.select_anchors(
                claim_text=claim.claim_text,
                triples=claim.triples,
                seed_entities=claim.entity_set,
            )
            injected_claim = adapter.inject_claim(claim).claim
            injected_anchors = selector.select_anchors(
                claim_text=injected_claim.claim_text,
                triples=injected_claim.triples,
                seed_entities=injected_claim.entity_set,
            )

            self.assertNotEqual(heuristic_anchors, injected_anchors)

            graph_heuristic = builder.build(
                claim_id=claim.claim_id,
                triples=claim.triples,
                anchors=heuristic_anchors,
            )
            graph_injected = builder.build(
                claim_id=claim.claim_id,
                triples=claim.triples,
                anchors=injected_anchors,
            )

            by_id_heuristic = {row.evidence_id: row for row in scorer.compute_bridge_scores(graph_heuristic)}
            by_id_injected = {row.evidence_id: row for row in scorer.compute_bridge_scores(graph_injected)}

            self.assertNotAlmostEqual(
                by_id_heuristic["s_bridge"].bridge_bonus_capped,
                by_id_injected["s_bridge"].bridge_bonus_capped,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
