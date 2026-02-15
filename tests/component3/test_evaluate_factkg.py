"""Tests for FactKG evaluation alignment safeguards."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pd = None  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

try:
    from evaluate import evaluate_on_test_set
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    evaluate_on_test_set = None  # type: ignore[assignment]


class _TokenBatch(dict):
    def to(self, device):
        return _TokenBatch({k: v.to(device) for k, v in self.items()})


class _DummyGraph:
    def to(self, _device):
        return self


if torch is not None:
    class _DummyQAGNN(torch.nn.Module):
        def forward(self, claim_tokens, _graph):
            batch_size = int(claim_tokens["input_ids"].shape[0])
            return torch.zeros((batch_size,), dtype=torch.float32, device=claim_tokens["input_ids"].device)


class _TinyQAGNNDataset(Dataset):
    def __init__(self, labels, claim_types=None):
        self.labels = [float(v) for v in labels]
        self.claim_types = claim_types

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return "claim", _DummyGraph(), self.labels[idx]


def _collate_qagnn(batch):
    labels = torch.tensor([row[2] for row in batch], dtype=torch.float32)
    batch_size = len(batch)
    tokens = _TokenBatch(
        {
            "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }
    )
    return tokens, _DummyGraph(), labels


@unittest.skipUnless(
    torch is not None and DataLoader is not None and pd is not None and evaluate_on_test_set is not None,
    "torch/pandas/evaluate deps unavailable",
)
class EvaluateFactKGAlignmentTests(unittest.TestCase):
    def test_raises_when_canonical_factkg_order_does_not_match_loader(self) -> None:
        model = _DummyQAGNN()
        loader = DataLoader(
            _TinyQAGNNDataset(labels=[0.0, 1.0], claim_types=None),
            batch_size=2,
            shuffle=False,
            collate_fn=_collate_qagnn,
        )

        canonical_df = pd.DataFrame(
            {
                "Label": [[1], [0]],
                "types": [["existence"], ["negation"]],
            }
        )
        with patch("evaluate.get_df", return_value=canonical_df):
            with self.assertRaises(ValueError):
                evaluate_on_test_set(
                    qa_gnn=True,
                    model=model,
                    test_loader=loader,
                    criterion=torch.nn.BCEWithLogitsLoss(),
                    dataset_name="factkg",
                )

    def test_uses_dataset_claim_types_without_loading_canonical_df(self) -> None:
        model = _DummyQAGNN()
        loader = DataLoader(
            _TinyQAGNNDataset(labels=[0.0, 1.0], claim_types=[["existence"], ["negation"]]),
            batch_size=2,
            shuffle=False,
            collate_fn=_collate_qagnn,
        )

        with patch("evaluate.get_df", side_effect=AssertionError("get_df should not be called")):
            metrics = evaluate_on_test_set(
                qa_gnn=True,
                model=model,
                test_loader=loader,
                criterion=torch.nn.BCEWithLogitsLoss(),
                dataset_name="factkg",
            )

        self.assertIn("overall", metrics)
        self.assertIn("existence", metrics)
        self.assertIn("single hop", metrics)


if __name__ == "__main__":
    unittest.main()
