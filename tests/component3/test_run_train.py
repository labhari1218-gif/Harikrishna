"""Tests for Component 3 training orchestration helpers."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


def _load_run_train_module():
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    module_path = repo_root / "src" / "component3" / "run_train.py"
    spec = importlib.util.spec_from_file_location("component3_run_train_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModel:
    def __init__(self) -> None:
        self.params = {
            "base_model.bert.base_model.encoder.layer0.weight": _FakeParam(),
            "base_model.bert.base_model.pooler.weight": _FakeParam(),
            "base_model.classifier.weight": _FakeParam(),
        }

    def named_parameters(self):
        return self.params.items()


class _LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return int(self.labels[idx])


if torch is not None and nn is not None:
    class _TinyBaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.name = "tiny_base"
            self.gnn_out_features = 4

            self.bert = nn.Module()
            self.bert.base_model = nn.Module()
            self.bert.base_model.encoder = nn.Linear(4, 4)
            self.bert.base_model.pooler = nn.Linear(4, 4)

            self.classifier = nn.Linear(4, 1)

            self.latest_x_sup = None
            self.latest_x_ref = None
            self.latest_edge_index = None

        def forward(self, claim_tokens, data_graph):
            batch_size = int(claim_tokens["input_ids"].shape[0])
            x = data_graph.x
            if x.size(1) < self.gnn_out_features:
                pad = torch.zeros((x.size(0), self.gnn_out_features - x.size(1)), dtype=x.dtype)
                x = torch.cat((x, pad), dim=1)
            x = x[:, : self.gnn_out_features]

            self.latest_x_sup = x
            self.latest_x_ref = x
            self.latest_edge_index = data_graph.edge_index
            return torch.zeros((batch_size,), dtype=torch.float32)


@unittest.skipUnless(torch is not None and DataLoader is not None, "torch deps unavailable")
class RunTrainHelperTests(unittest.TestCase):
    def test_default_stage_plan_matches_locked_schedule(self) -> None:
        mod = _load_run_train_module()
        cfg = mod.Component3TrainConfig()
        stages = mod.build_stage_plan(cfg)

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0].name, "stage1_freeze_bert")
        self.assertEqual(stages[0].epochs, 5)
        self.assertAlmostEqual(stages[0].learning_rate, 1.0e-5, places=12)
        self.assertFalse(stages[0].unfreeze_pooler)

        self.assertEqual(stages[1].name, "stage2_unfreeze_pooler")
        self.assertEqual(stages[1].epochs, 5)
        self.assertAlmostEqual(stages[1].learning_rate, 2.0e-6, places=12)
        self.assertTrue(stages[1].unfreeze_pooler)

    def test_config_to_dict_contains_multitask_and_resource_defaults(self) -> None:
        mod = _load_run_train_module()
        payload = mod.config_to_dict(mod.Component3TrainConfig())

        self.assertEqual(payload["task"], "T3.3")
        self.assertEqual(payload["batch_size"], 8)
        self.assertEqual(payload["max_seq_len"], 256)
        self.assertEqual(payload["train_subset_size"], 0)
        self.assertEqual(payload["val_subset_size"], 0)
        self.assertEqual(payload["subset_sampling"], "stratified")
        self.assertEqual(payload["loss"]["lambda_evidence"], 0.1)
        self.assertEqual(payload["loader_num_workers"], 0)
        self.assertFalse(payload["loader_pin_memory"])
        self.assertFalse(payload["loader_persistent_workers"])
        self.assertEqual(payload["loader_prefetch_factor"], 2)
        self.assertFalse(payload["non_blocking_transfers"])
        self.assertEqual(payload["factkg_claim_triple_cache_path"], "data/claim_triple_embeddings.pkl")
        self.assertFalse(payload["factkg_require_claim_triple_cache"])
        self.assertEqual(payload["factkg_precompute_batch_size"], 32)
        self.assertFalse(payload["deterministic_mode"])
        self.assertAlmostEqual(payload["no_collapse_precision_accuracy_gap_min"], 0.005, places=12)
        self.assertEqual(len(payload["stages"]), 2)

    def test_no_collapse_gate_uses_configurable_gap_threshold(self) -> None:
        mod = _load_run_train_module()
        metrics = {
            "overall": {
                "accuracy": 0.6024,
                "precision": 0.5925,
                "recall": 0.5850,
            }
        }
        stages = [{"best_val_accuracy": 60.0}]

        gate_loose = mod._evaluate_no_collapse_gate(
            metrics,
            stages,
            precision_accuracy_gap_min=0.005,
        )
        gate_strict = mod._evaluate_no_collapse_gate(
            metrics,
            stages,
            precision_accuracy_gap_min=0.02,
        )

        self.assertTrue(gate_loose["checks"]["precision_accuracy_gap_ge_min"])
        self.assertFalse(gate_strict["checks"]["precision_accuracy_gap_ge_min"])

    def test_subset_dataloader_preserves_prefetch_factor(self) -> None:
        mod = _load_run_train_module()
        dataset = _LabelDataset([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3,
        )
        subset_loader = mod._subset_dataloader(
            loader,
            max_samples=4,
            subset_sampling="prefix",
            seed=123,
        )
        self.assertEqual(getattr(subset_loader, "num_workers", 0), 1)
        self.assertTrue(bool(getattr(subset_loader, "pin_memory", False)))
        self.assertTrue(bool(getattr(subset_loader, "persistent_workers", False)))
        self.assertEqual(getattr(subset_loader, "prefetch_factor", None), 3)

    def test_write_config_yaml_persists_expected_keys(self) -> None:
        mod = _load_run_train_module()
        payload = mod.config_to_dict(mod.Component3TrainConfig())

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            mod.write_config_yaml(path, payload)
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("task:", text)
            self.assertIn("pipeline:", text)
            self.assertIn("lambda_evidence:", text)
            self.assertIn("stages:", text)

    def test_configure_trainable_parameters_applies_stage_rules(self) -> None:
        mod = _load_run_train_module()
        model = _FakeModel()

        mod.configure_trainable_parameters(model, unfreeze_pooler=False)
        self.assertFalse(model.params["base_model.bert.base_model.encoder.layer0.weight"].requires_grad)
        self.assertFalse(model.params["base_model.bert.base_model.pooler.weight"].requires_grad)
        self.assertTrue(model.params["base_model.classifier.weight"].requires_grad)

        mod.configure_trainable_parameters(model, unfreeze_pooler=True)
        self.assertFalse(model.params["base_model.bert.base_model.encoder.layer0.weight"].requires_grad)
        self.assertTrue(model.params["base_model.bert.base_model.pooler.weight"].requires_grad)
        self.assertTrue(model.params["base_model.classifier.weight"].requires_grad)

    def test_subset_size_zero_keeps_full_loader(self) -> None:
        mod = _load_run_train_module()
        dataset = _LabelDataset([0, 1, 0, 1, 0, 1])
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        out_train = mod._subset_dataloader(loader, max_samples=0, subset_sampling="stratified", seed=123)
        out_val = mod._subset_dataloader(loader, max_samples=0, subset_sampling="prefix", seed=123)

        self.assertIs(out_train, loader)
        self.assertIs(out_val, loader)

    def test_stratified_subsetting_preserves_label_mix_and_is_deterministic(self) -> None:
        mod = _load_run_train_module()
        labels = ([0] * 80) + ([1] * 20)
        loader = DataLoader(_LabelDataset(labels), batch_size=8, shuffle=False)

        subset_loader_a = mod._subset_dataloader(loader, max_samples=30, subset_sampling="stratified", seed=57)
        subset_loader_b = mod._subset_dataloader(loader, max_samples=30, subset_sampling="stratified", seed=57)

        idx_a = list(subset_loader_a.dataset.indices)
        idx_b = list(subset_loader_b.dataset.indices)
        self.assertEqual(idx_a, idx_b)

        subset_labels = [labels[i] for i in idx_a]
        full_pos_rate = sum(labels) / len(labels)
        subset_pos_rate = sum(subset_labels) / len(subset_labels)

        self.assertAlmostEqual(subset_pos_rate, full_pos_rate, delta=0.10)

    def test_loader_dispatch_uses_dataset_name(self) -> None:
        mod = _load_run_train_module()

        with patch.object(mod, "_build_factkg_loaders", return_value=("f_train", "f_val", "f_test", {})) as fact, \
             patch.object(mod, "_build_fever_loaders", return_value=("v_train", "v_val", "v_test", {})) as fever:
            out_factkg = mod._build_default_loaders(mod.Component3TrainConfig(dataset_name="factkg"))
            out_fever = mod._build_default_loaders(mod.Component3TrainConfig(dataset_name="fever"))

        fact.assert_called_once()
        fever.assert_called_once()
        self.assertEqual(out_factkg[0], "f_train")
        self.assertEqual(out_fever[0], "v_train")

    def test_loader_dispatch_rejects_unknown_dataset_name(self) -> None:
        mod = _load_run_train_module()
        cfg = mod.Component3TrainConfig(dataset_name="unknown_dataset")
        with self.assertRaises(ValueError):
            mod._build_default_loaders(cfg)

    def test_ensure_binary_labels_rejects_non_binary_values(self) -> None:
        mod = _load_run_train_module()
        with self.assertRaises(ValueError):
            mod._ensure_binary_labels([0, 1, 2], split="train", dataset_name="FEVER")

    def test_component1_coverage_validation_rejects_missing_pairs_file(self) -> None:
        mod = _load_run_train_module()
        cfg = mod.Component3TrainConfig(use_component1_pairs=True)
        coverage = {
            "train": {"pairs_file_exists": False, "pairs_file": "logs/component1/train/pairs.jsonl"},
            "val": {"pairs_file_exists": True, "pair_rows_total": 10, "pair_rows_parsed": 10},
            "test": {"pairs_file_exists": True, "pair_rows_total": 10, "pair_rows_parsed": 10},
        }
        with self.assertRaises(FileNotFoundError):
            mod._validate_component1_pair_coverage(cfg, coverage)

    def test_component1_coverage_validation_rejects_unparseable_pairs(self) -> None:
        mod = _load_run_train_module()
        cfg = mod.Component3TrainConfig(use_component1_pairs=True)
        coverage = {
            "train": {"pairs_file_exists": True, "pair_rows_total": 10, "pair_rows_parsed": 0},
            "val": {"pairs_file_exists": True, "pair_rows_total": 10, "pair_rows_parsed": 10},
            "test": {"pairs_file_exists": True, "pair_rows_total": 10, "pair_rows_parsed": 10},
        }
        with self.assertRaises(RuntimeError):
            mod._validate_component1_pair_coverage(cfg, coverage)

    def test_run_training_rejects_empty_train_loader(self) -> None:
        mod = _load_run_train_module()

        class _TinyDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return idx

        empty_train_loader = DataLoader(_TinyDataset(), batch_size=8, shuffle=False, drop_last=True)
        non_empty_loader = DataLoader(_TinyDataset(), batch_size=1, shuffle=False, drop_last=False)

        cfg = mod.Component3TrainConfig(
            run_id="unit_test_empty_train_loader",
            output_root=tempfile.gettempdir(),
            evaluate_after_training=False,
            enforce_no_collapse_gate=False,
        )

        with self.assertRaises(ValueError):
            mod.run_training(
                cfg,
                model=_TinyBaseModel(),
                train_loader=empty_train_loader,
                val_loader=non_empty_loader,
                test_loader=non_empty_loader,
            )

    def test_optimizer_state_is_preserved_across_stages(self) -> None:
        mod = _load_run_train_module()

        class _DummyDataset(Dataset):
            def __init__(self, n: int):
                self.labels = [0] * n
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return idx

        loader = DataLoader(_DummyDataset(4), batch_size=2, shuffle=False)

        optimizer_ids = []
        state_call_counts = []

        def _fake_train(model, criterion, optimizer, qa_gnn, train_loader, val_loader=None, n_epochs=1,
                        scheduler=None, grad_accum_steps=1, n_early_stop=None, save_models=True,
                        device=None, non_blocking=False, verbose=0):
            optimizer_ids.append(id(optimizer))
            first_param = optimizer.param_groups[0]["params"][0]
            state = optimizer.state.setdefault(first_param, {})
            state["calls"] = int(state.get("calls", 0)) + 1
            state_call_counts.append(int(state["calls"]))
            history = {
                "best_epoch": 1,
                "best_val_loss": 0.1,
                "best_val_accuracy": 60.0,
                "model_name": "dummy",
            }
            return history, {"best_model_state_dict": model.state_dict()}

        fake_metrics = {
            "overall": {
                "accuracy": 0.60,
                "precision": 0.57,
                "recall": 0.80,
                "f1": 0.67,
            }
        }

        cfg = mod.Component3TrainConfig(
            run_id="unit_test_optimizer_persist",
            output_root=tempfile.gettempdir(),
            train_subset_size=0,
            val_subset_size=0,
            stage1_epochs=1,
            stage2_epochs=1,
            evaluate_after_training=True,
            gradient_accumulation_steps=2,
            enforce_no_collapse_gate=False,
        )

        with patch("train.train", side_effect=_fake_train), patch(
            "evaluate.evaluate_on_test_set", return_value=fake_metrics
        ):
            _run_dir, results = mod.run_training(
                cfg,
                model=_TinyBaseModel(),
                train_loader=loader,
                val_loader=loader,
                test_loader=loader,
            )

        self.assertEqual(len(optimizer_ids), 2)
        self.assertEqual(len(set(optimizer_ids)), 1)
        self.assertEqual(state_call_counts, [1, 2])
        self.assertEqual(results["stages"][0]["optimizer_id"], results["stages"][1]["optimizer_id"])


if __name__ == "__main__":
    unittest.main()
