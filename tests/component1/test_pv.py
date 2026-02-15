"""Tests for component1.pv throughput-safe inference changes."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from component1.pv import PVConfig, PVScorer


class _DummyTokenizer:
    def __call__(self, evidence_texts, claim_texts, **kwargs):
        del claim_texts, kwargs
        batch = len(evidence_texts)
        return {
            "input_ids": torch.arange(batch * 4, dtype=torch.long).reshape(batch, 4),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class _DummyModel:
    def __call__(self, **inputs):
        batch = int(inputs["input_ids"].shape[0])
        logits = torch.tensor(
            [[0.1, 0.2, 0.7] for _ in range(batch)],
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)


def _build_test_scorer(non_blocking: bool = False) -> PVScorer:
    scorer = PVScorer.__new__(PVScorer)
    scorer.config = PVConfig(non_blocking_transfers=non_blocking)
    scorer.device = torch.device("cpu")
    scorer.model = _DummyModel()
    scorer.tokenizer = _DummyTokenizer()
    scorer.use_autocast = False
    scorer.label_map = {"contra": 0, "neutral": 1, "entail": 2}
    return scorer


def test_score_batch_uses_inference_mode():
    scorer = _build_test_scorer(non_blocking=False)
    with patch("component1.pv.torch.inference_mode", return_value=contextlib.nullcontext()) as patched_ctx:
        results = scorer._score_batch("claim", ["ev1", "ev2"])
    patched_ctx.assert_called_once()
    assert len(results) == 2
    assert all(result.rel > 0.0 for result in results)


def test_non_blocking_flag_is_safe_on_cpu():
    scorer = _build_test_scorer(non_blocking=True)
    moved = scorer._move_inputs_to_device(
        {"input_ids": torch.ones((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
    )
    assert moved["input_ids"].device.type == "cpu"
    assert moved["attention_mask"].device.type == "cpu"
