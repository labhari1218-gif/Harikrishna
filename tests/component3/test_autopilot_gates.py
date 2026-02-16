"""Tests for scripts/autopilot_gates.py gate logic."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "autopilot_gates.py"
    spec = importlib.util.spec_from_file_location("autopilot_gates_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class AutopilotGateTests(unittest.TestCase):
    def test_safety_gate_allows_sentinel_exception_path(self) -> None:
        mod = _load_module()
        self.assertTrue(
            bool(
                mod._compute_safety_gate(
                    strict_clean=False,
                    sentinel_exception_confirmed=True,
                    embeddings_ok=True,
                    cache_ok=True,
                )
            )
        )

    def test_safety_gate_requires_embeddings_and_cache(self) -> None:
        mod = _load_module()
        self.assertFalse(
            bool(
                mod._compute_safety_gate(
                    strict_clean=True,
                    sentinel_exception_confirmed=True,
                    embeddings_ok=False,
                    cache_ok=True,
                )
            )
        )
        self.assertFalse(
            bool(
                mod._compute_safety_gate(
                    strict_clean=True,
                    sentinel_exception_confirmed=False,
                    embeddings_ok=True,
                    cache_ok=False,
                )
            )
        )

    def test_stage1_gate_passes_with_train_acc_threshold(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cwd = Path.cwd()
            try:
                os.chdir(root)
                run_dir = root / "runs" / "r1"
                _write_json(
                    run_dir / "metrics.json",
                    {
                        "stages": [{"train_accuracy_best": 0.96}],
                        "runtime_flags": {"backtracking_enabled": False},
                        "test_metrics": {"overall": {"accuracy": 0.1}},
                    },
                )
                (run_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
                (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "run_console.log").write_text("ok\n", encoding="utf-8")

                report = mod.run_gate(
                    SimpleNamespace(
                        stage="stage1_overfit200",
                        run_id="r1",
                        run_a_id="",
                        run_b_id="",
                        out_json="",
                    )
                )
                self.assertTrue(bool(report["pass"]))
                self.assertAlmostEqual(float(report["train_acc"]), 0.96, places=6)
            finally:
                os.chdir(cwd)

    def test_stage2_gate_fails_when_model_mode_not_claim_only(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cwd = Path.cwd()
            try:
                os.chdir(root)
                run_dir = root / "runs" / "r2"
                _write_json(
                    run_dir / "metrics.json",
                    {
                        "stages": [{"train_accuracy_best": 0.97}],
                        "runtime_flags": {
                            "backtracking_enabled": False,
                            "model_mode": "pv_qagnn",
                        },
                        "test_metrics": {"overall": {"accuracy": 0.71}},
                    },
                )
                (run_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
                (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "run_console.log").write_text("ok\n", encoding="utf-8")

                report = mod.run_gate(
                    SimpleNamespace(
                        stage="stage2_claim_only",
                        run_id="r2",
                        run_a_id="",
                        run_b_id="",
                        out_json="",
                    )
                )
                self.assertFalse(bool(report["pass"]))
                reasons = "\n".join(report.get("reasons", []))
                self.assertIn("model_mode", reasons)
            finally:
                os.chdir(cwd)

    def test_stage4_gate_uses_flip_condition_when_delta_small(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cwd = Path.cwd()
            try:
                os.chdir(root)
                run_a = root / "runs" / "a"
                run_b = root / "runs" / "b"

                _write_json(
                    run_a / "metrics.json",
                    {
                        "runtime_flags": {"backtracking_enabled": False, "model_mode": "pv_qagnn"},
                        "test_metrics": {"overall": {"accuracy": 0.6500}},
                    },
                )
                _write_json(
                    run_b / "metrics.json",
                    {
                        "runtime_flags": {"backtracking_enabled": True, "model_mode": "pv_qagnn"},
                        "test_metrics": {"overall": {"accuracy": 0.6505}},
                    },
                )

                for run_dir in (run_a, run_b):
                    (run_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
                    (run_dir / "run_console.log").write_text("ok\n", encoding="utf-8")

                _write_jsonl(
                    run_a / "predictions.jsonl",
                    [
                        {"claim_id": "test_0", "label": 1, "pred": 0},
                        {"claim_id": "test_1", "label": 1, "pred": 1},
                    ],
                )
                _write_jsonl(
                    run_b / "predictions.jsonl",
                    [
                        {"claim_id": "test_0", "label": 1, "pred": 1},
                        {"claim_id": "test_1", "label": 1, "pred": 1},
                    ],
                )
                (run_b / "recovery_actions.jsonl").write_text("{}\n", encoding="utf-8")
                (run_b / "recovery_candidates.jsonl").write_text("{}\n", encoding="utf-8")
                _write_json(
                    run_b / "backtracking_effectiveness_0p002.json",
                    {
                        "splits": {
                            "test": {
                                "flip_to_correct_claims": 10,
                            }
                        }
                    },
                )

                report = mod.run_gate(
                    SimpleNamespace(
                        stage="stage4_bt_helping",
                        run_id="",
                        run_a_id="a",
                        run_b_id="b",
                        out_json="",
                    )
                )
                self.assertTrue(bool(report["pass"]))
                self.assertFalse(bool(report["condition_delta_acc_ge_0p001"]))
                self.assertTrue(bool(report["condition_flip_to_correct_and_net_not_worse"]))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
