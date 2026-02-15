"""Tests for Component 2 hybrid masked dual-stream reasoner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from component2.config import Component2Config, DEFAULT_CONFIG
from component2.graph_builder import GraphBuilder
from component2.reasoner import HybridMaskedDualStreamReasoner
from component2.types import evidence_from_dict


class ReasonerTests(unittest.TestCase):
    def _build_graph(self, relation_name: str = "birthPlace", builder: GraphBuilder | None = None):
        triples = [
            evidence_from_dict(
                {
                    "evidence_id": "e1",
                    "raw_triple": ["Barack_Obama", relation_name, "Honolulu"],
                    "pool": "A",
                    "p_ent": 0.72,
                    "p_con": 0.06,
                    "p_neu": 0.22,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e2",
                    "raw_triple": ["Honolulu", "isPartOf", "Hawaii"],
                    "pool": "A",
                    "p_ent": 0.64,
                    "p_con": 0.05,
                    "p_neu": 0.31,
                }
            ),
            evidence_from_dict(
                {
                    "evidence_id": "e3",
                    "raw_triple": ["Barack_Obama", "birthPlace", "Kenya"],
                    "pool": "C",
                    "p_ent": 0.05,
                    "p_con": 0.75,
                    "p_neu": 0.20,
                }
            ),
        ]

        builder = builder or GraphBuilder(include_pools=("A", "C"))
        return builder.build(claim_id="claim_1", triples=triples, anchors=["Barack_Obama", "Hawaii"])

    def test_mask_formulas_match_locked_defaults(self) -> None:
        reasoner = HybridMaskedDualStreamReasoner(config=DEFAULT_CONFIG)

        p_ent = np.array([0.0, 0.5, 1.0])
        p_con = np.array([0.0, 0.5, 1.0])

        expected_sup = 1.0 / (1.0 + np.exp(-(4.0 * p_ent - 2.0)))
        expected_ref = 1.0 / (1.0 + np.exp(-(4.0 * p_con - 2.0)))

        np.testing.assert_allclose(reasoner.support_mask(p_ent), expected_sup)
        np.testing.assert_allclose(reasoner.refute_mask(p_con), expected_ref)

    def test_mask_formulas_use_config_coefficients(self) -> None:
        config = Component2Config(
            mask_sup_scale=2.0,
            mask_sup_bias=-1.0,
            mask_ref_scale=3.0,
            mask_ref_bias=-1.5,
        )
        reasoner = HybridMaskedDualStreamReasoner(config=config)
        p_ent = np.array([0.1, 0.9], dtype=np.float64)
        p_con = np.array([0.2, 0.8], dtype=np.float64)

        expected_sup = 1.0 / (1.0 + np.exp(-(2.0 * p_ent - 1.0)))
        expected_ref = 1.0 / (1.0 + np.exp(-(3.0 * p_con - 1.5)))

        np.testing.assert_allclose(reasoner.support_mask(p_ent), expected_sup)
        np.testing.assert_allclose(reasoner.refute_mask(p_con), expected_ref)

    def test_disable_pv_masks_returns_unity_masks(self) -> None:
        config = Component2Config(disable_pv_masks=True)
        reasoner = HybridMaskedDualStreamReasoner(config=config)
        p_ent = np.array([0.1, 0.9], dtype=np.float64)
        p_con = np.array([0.2, 0.8], dtype=np.float64)

        np.testing.assert_allclose(reasoner.support_mask(p_ent), np.ones_like(p_ent))
        np.testing.assert_allclose(reasoner.refute_mask(p_con), np.ones_like(p_con))

    def test_mask_alpha_beta_aliases_match_scale_bias(self) -> None:
        config = Component2Config(
            mask_sup_scale=2.5,
            mask_sup_bias=-0.75,
            mask_ref_scale=3.25,
            mask_ref_bias=-1.5,
        )

        self.assertEqual(config.mask_sup_alpha, config.mask_sup_scale)
        self.assertEqual(config.mask_sup_beta, config.mask_sup_bias)
        self.assertEqual(config.mask_ref_alpha, config.mask_ref_scale)
        self.assertEqual(config.mask_ref_beta, config.mask_ref_bias)

        snapshot = config.to_snapshot_dict()
        self.assertEqual(snapshot["mask_sup_alpha"], config.mask_sup_scale)
        self.assertEqual(snapshot["mask_sup_beta"], config.mask_sup_bias)
        self.assertEqual(snapshot["mask_ref_alpha"], config.mask_ref_scale)
        self.assertEqual(snapshot["mask_ref_beta"], config.mask_ref_bias)

    def test_forward_batch_outputs_probs_salience_and_loss(self) -> None:
        reasoner = HybridMaskedDualStreamReasoner(config=DEFAULT_CONFIG)
        graphs = [self._build_graph() for _ in range(5)]
        labels = [0, 1, 0, 1, 0]

        outputs = reasoner.forward_batch(graphs, labels=labels)

        self.assertEqual(outputs["probs"].shape, (5, 2))
        self.assertEqual(outputs["logits"].shape, (5, 2))
        self.assertTrue(math.isfinite(outputs["loss"]))

        for graph_salience in outputs["salience"]:
            self.assertEqual(len(graph_salience), 3)
            for edge_row in graph_salience:
                self.assertIn("evidence_id", edge_row)
                self.assertIn("salience", edge_row)

    def test_relation_conditioned_attention_changes_scores(self) -> None:
        reasoner = HybridMaskedDualStreamReasoner(config=DEFAULT_CONFIG)
        shared_builder = GraphBuilder(include_pools=("A", "C"))
        graph_a = self._build_graph(relation_name="birthPlace", builder=shared_builder)
        graph_b = self._build_graph(relation_name="almaMater", builder=shared_builder)

        out_a = reasoner.forward_graph(graph_a)
        out_b = reasoner.forward_graph(graph_b)

        probs_a = np.array(out_a["probs"], dtype=np.float64)
        probs_b = np.array(out_b["probs"], dtype=np.float64)

        self.assertFalse(np.allclose(probs_a, probs_b))

    def test_forward_batch_rejects_empty_graph_list(self) -> None:
        reasoner = HybridMaskedDualStreamReasoner(config=DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            reasoner.forward_batch(graphs=[])

    def test_entity_hash_vectors_are_stable_across_python_hash_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cmd = [
            sys.executable,
            "-c",
            (
                "import json;"
                "from component2.config import DEFAULT_CONFIG;"
                "from component2.reasoner import HybridMaskedDualStreamReasoner;"
                "reasoner = HybridMaskedDualStreamReasoner(config=DEFAULT_CONFIG);"
                "vec = reasoner._entity_hash_vector('Barack_Obama')[:8];"
                "print(json.dumps([round(float(x), 12) for x in vec.tolist()]))"
            ),
        ]

        def run(seed: int) -> list[float]:
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            env["PYTHONHASHSEED"] = str(seed)
            output = subprocess.check_output(cmd, cwd=repo_root, env=env, text=True)
            return json.loads(output.strip())

        self.assertEqual(run(1), run(2))


if __name__ == "__main__":
    unittest.main()
