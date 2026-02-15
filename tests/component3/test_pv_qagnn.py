"""Tests for Component 3 PV_QAGNN model."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Batch = None  # type: ignore[assignment]
    Data = None  # type: ignore[assignment]

try:
    from component3 import PV_QAGNN
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    PV_QAGNN = None  # type: ignore[assignment]


_NN_BASE = nn.Module if nn is not None else object


class _DummyBert(_NN_BASE):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.base_model = nn.Module()
        self.base_model.pooler = nn.Linear(hidden_size, hidden_size)
        self.base_model.encoder = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("`input_ids` is required for dummy forward pass.")
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, hidden_size)
        ramp = torch.arange(hidden_size, dtype=torch.float32).view(1, 1, hidden_size)
        return SimpleNamespace(last_hidden_state=(base / 100.0) + ramp)


def _dummy_get_bert_model(*args, **kwargs):
    return _DummyBert(hidden_size=16)


def _make_graph(hidden_size: int = 16) -> Data:
    x = torch.tensor(
        [
            [0.1 * (i + 1) for i in range(hidden_size)],
            [0.2 * (i + 1) for i in range(hidden_size)],
            [0.3 * (i + 1) for i in range(hidden_size)],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    # [p_ent, p_con, p_neu, rel, pool_id] for each edge.
    edge_attr = torch.tensor(
        [
            [0.75, 0.10, 0.15, 0.80, 0.0],
            [0.70, 0.12, 0.18, 0.75, 0.0],
            [0.62, 0.20, 0.18, 0.65, 0.0],
            [0.58, 0.25, 0.17, 0.60, 0.0],
        ],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@unittest.skipUnless(
    torch is not None and nn is not None and Batch is not None and Data is not None and PV_QAGNN is not None,
    "torch/torch_geometric deps unavailable",
)
class PVQAGNNTests(unittest.TestCase):
    @patch("models.get_bert_model", side_effect=_dummy_get_bert_model)
    def test_mask_params_are_trainable_with_locked_defaults(self, _mock_get_bert_model) -> None:
        model = PV_QAGNN(model_name="pv_qagnn_test")

        self.assertAlmostEqual(float(model.mask_sup_alpha.item()), 4.0)
        self.assertAlmostEqual(float(model.mask_sup_beta.item()), -2.0)
        self.assertAlmostEqual(float(model.mask_ref_alpha.item()), 4.0)
        self.assertAlmostEqual(float(model.mask_ref_beta.item()), -2.0)

        self.assertTrue(model.mask_sup_alpha.requires_grad)
        self.assertTrue(model.mask_sup_beta.requires_grad)
        self.assertTrue(model.mask_ref_alpha.requires_grad)
        self.assertTrue(model.mask_ref_beta.requires_grad)

    @patch("models.get_bert_model", side_effect=_dummy_get_bert_model)
    def test_support_and_refute_streams_have_independent_parameters(self, _mock_get_bert_model) -> None:
        model = PV_QAGNN(model_name="pv_qagnn_test")

        self.assertEqual(len(model.gnn_layers_sup), 2)
        self.assertEqual(len(model.gnn_layers_ref), 2)

        for sup_layer, ref_layer in zip(model.gnn_layers_sup, model.gnn_layers_ref):
            self.assertIsNot(sup_layer, ref_layer)
            sup_first_param = next(sup_layer.parameters())
            ref_first_param = next(ref_layer.parameters())
            self.assertNotEqual(sup_first_param.data_ptr(), ref_first_param.data_ptr())

    @patch("models.get_bert_model", side_effect=_dummy_get_bert_model)
    def test_forward_pass_exposes_latest_stream_tensors(self, _mock_get_bert_model) -> None:
        model = PV_QAGNN(
            model_name="pv_qagnn_test",
            n_gnn_layers=2,
            gnn_hidden_dim=32,
            gnn_out_features=16,
        )

        graphs = [_make_graph() for _ in range(5)]
        batch_graph = Batch.from_data_list(graphs)
        claim_tokens = {
            "input_ids": torch.randint(0, 20, (5, 6), dtype=torch.long),
            "attention_mask": torch.ones((5, 6), dtype=torch.long),
        }

        logits = model(claim_tokens, batch_graph)
        self.assertEqual(tuple(logits.shape), (5,))
        self.assertTrue(torch.isfinite(logits).all().item())

        self.assertIsNotNone(model.latest_x_sup)
        self.assertIsNotNone(model.latest_x_ref)
        self.assertIsNotNone(model.latest_edge_index)
        self.assertEqual(tuple(model.latest_edge_index.shape), tuple(batch_graph.edge_index.shape))

    @patch("models.get_bert_model", side_effect=_dummy_get_bert_model)
    def test_edge_attr_mismatch_raises_explicit_error(self, _mock_get_bert_model) -> None:
        model = PV_QAGNN(model_name="pv_qagnn_test")

        bad_graph = _make_graph()
        bad_graph.edge_attr = bad_graph.edge_attr[:3]  # edge_index has 4 edges
        batch_graph = Batch.from_data_list([bad_graph])

        claim_tokens = {
            "input_ids": torch.randint(0, 20, (1, 6), dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
        }

        with self.assertRaises(ValueError) as ctx:
            model(claim_tokens, batch_graph)
        self.assertIn("edge_attr/edge_index row mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
