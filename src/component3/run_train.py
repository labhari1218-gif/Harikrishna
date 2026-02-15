"""Component 3 training entrypoint using existing QA-GNN infrastructure."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any

import torch.nn as nn  # type: ignore


@dataclass(frozen=True)
class StageConfig:
    """A single training stage in the two-stage warm-up schedule."""

    name: str
    epochs: int
    learning_rate: float
    unfreeze_pooler: bool


@dataclass(frozen=True)
class Component3TrainConfig:
    """Runtime configuration for Component 3 training."""

    run_id: str | None = None
    output_root: str = "runs"
    model_name: str = "pv_qagnn_component3"
    dataset_name: str = "factkg"
    subgraph_type: str = "direct_filled"
    fever_data_root: str = "data/fever"
    fever_component1_logs_root: str = "logs/component1_fever"
    fever_include_s_pool: bool = False
    fever_auto_precompute_embeddings: bool = False
    loader_num_workers: int = 0
    loader_pin_memory: bool = False
    loader_persistent_workers: bool = False
    loader_prefetch_factor: int = 2
    non_blocking_transfers: bool = False
    factkg_claim_triple_cache_path: str = "data/claim_triple_embeddings.pkl"
    factkg_require_claim_triple_cache: bool = False
    factkg_precompute_batch_size: int = 32
    deterministic_mode: bool = False
    no_collapse_precision_accuracy_gap_min: float = 0.005
    batch_size: int = 8
    max_seq_len: int = 256
    train_subset_size: int = 0
    val_subset_size: int = 0
    subset_sampling: str = "stratified"
    seed: int = 57
    gnn_hidden_dim: int = 256
    gnn_out_features: int = 256
    n_gnn_layers: int = 2
    gnn_dropout: float = 0.3
    classifier_dropout: float = 0.2
    lm_layer_dropout: float = 0.4
    gnn_batch_norm: bool = True
    lambda_evidence: float = 0.1
    enable_component5: bool = False
    component5_evidence_weight: float = 1.0
    lambda_starv: float = 0.1
    tau_starv: float = 0.3
    lambda_counter: float = 0.1
    lambda_recovery: float = 0.1
    loss_ablation_variant: str = "full"
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    stage1_lr: float = 1.0e-5
    stage2_lr: float = 2.0e-6
    warmup_steps: int = 50
    n_early_stop: int = 3
    evaluate_after_training: bool = True
    online_embeddings: bool = False
    mix_graphs: bool = False
    use_roberta: bool = False
    use_component1_pairs: bool = True
    component1_logs_root: str = "logs/component1"
    missing_pv_policy: str = "hybrid_fallback"
    gradient_accumulation_steps: int = 4
    enforce_no_collapse_gate: bool = True


def build_stage_plan(config: Component3TrainConfig) -> list[StageConfig]:
    """Build the locked two-stage schedule from config."""

    return [
        StageConfig(
            name="stage1_freeze_bert",
            epochs=config.stage1_epochs,
            learning_rate=config.stage1_lr,
            unfreeze_pooler=False,
        ),
        StageConfig(
            name="stage2_unfreeze_pooler",
            epochs=config.stage2_epochs,
            learning_rate=config.stage2_lr,
            unfreeze_pooler=True,
        ),
    ]


def config_to_dict(config: Component3TrainConfig) -> dict[str, Any]:
    """Serialize config and stage plan to a JSON/YAML-friendly dictionary."""

    payload = asdict(config)
    payload["stages"] = [asdict(stage) for stage in build_stage_plan(config)]
    payload["pipeline"] = "pv_enhanced_qa_gnn"
    if config.enable_component5:
        payload["task"] = "T5.5"
        payload["loss"] = {
            "name": "component5_joint_loss",
            "formula": (
                "L_total = L_BCE + w_evidence*L_evidence + L_starv + L_counter + L_recovery"
            ),
            "component5_evidence_weight": config.component5_evidence_weight,
            "lambda_starv": config.lambda_starv,
            "tau_starv": config.tau_starv,
            "lambda_counter": config.lambda_counter,
            "lambda_recovery": config.lambda_recovery,
            "loss_ablation_variant": config.loss_ablation_variant,
        }
    else:
        payload["task"] = "T3.3"
        payload["loss"] = {
            "name": "multitask_bce_plus_dual_evidence",
            "formula": (
                "L_total = L_BCE(claim_pred,label) + "
                "lambda * mean(L_support(edge->A), L_counter(edge->C))"
            ),
            "lambda_evidence": config.lambda_evidence,
        }
    return payload


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text:
        return json.dumps(text)
    return text


def _to_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, inner in value.items():
            if isinstance(inner, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(inner, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(inner)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def write_config_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Persist configuration to YAML, without requiring PyYAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return
    except Exception:
        pass

    yaml_text = "\n".join(_to_yaml_lines(payload)) + "\n"
    path.write_text(yaml_text, encoding="utf-8")


def _is_bert_parameter(param_name: str) -> bool:
    return param_name.startswith("base_model.bert.") or param_name.startswith("bert.")


def _is_pooler_parameter(param_name: str) -> bool:
    return (
        ".bert.base_model.pooler." in param_name
        or ".bert.pooler." in param_name
        or param_name.startswith("bert.base_model.pooler.")
        or param_name.startswith("bert.pooler.")
    )


def configure_trainable_parameters(model: Any, unfreeze_pooler: bool) -> None:
    """Freeze BERT encoder and optionally unfreeze pooler; keep non-BERT trainable."""

    for name, parameter in model.named_parameters():
        if _is_bert_parameter(name):
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    if not unfreeze_pooler:
        return

    for name, parameter in model.named_parameters():
        if _is_pooler_parameter(name):
            parameter.requires_grad = True


def _summarize_trainable_parameters(model: Any) -> dict[str, int]:
    summary = {
        "total": 0,
        "trainable": 0,
        "bert_total": 0,
        "bert_trainable": 0,
        "pooler_total": 0,
        "pooler_trainable": 0,
    }
    for name, parameter in model.named_parameters():
        summary["total"] += int(parameter.numel())
        if parameter.requires_grad:
            summary["trainable"] += int(parameter.numel())
        if _is_bert_parameter(name):
            summary["bert_total"] += int(parameter.numel())
            if parameter.requires_grad:
                summary["bert_trainable"] += int(parameter.numel())
        if _is_pooler_parameter(name):
            summary["pooler_total"] += int(parameter.numel())
            if parameter.requires_grad:
                summary["pooler_trainable"] += int(parameter.numel())
    return summary


def _build_loader_kwargs(
    config: Component3TrainConfig,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    collate_fn: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "collate_fn": collate_fn,
        "num_workers": max(0, int(config.loader_num_workers)),
        "pin_memory": bool(config.loader_pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = max(1, int(config.loader_prefetch_factor))
        if config.loader_persistent_workers:
            kwargs["persistent_workers"] = True
    return kwargs


def _enable_deterministic_mode(seed: int) -> None:
    import numpy as np
    import torch  # type: ignore

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MultiTaskLossAdapter:
    """Criterion adapter computing claim BCE + support/counter edge BCE."""

    def __init__(self, model: Any, lambda_evidence: float = 0.1):
        import torch.nn as nn  # type: ignore

        self.model = model
        self.lambda_evidence = float(lambda_evidence)
        self._bce = nn.BCEWithLogitsLoss()
        self.last_bce = 0.0
        self.last_evidence = 0.0
        self.last_total = 0.0

    def __call__(self, claim_logits, labels):
        import torch  # type: ignore

        labels = labels.float()
        claim_loss = self._bce(claim_logits, labels)

        support_logits = getattr(self.model, "latest_edge_logits_support", None)
        support_labels = getattr(self.model, "latest_edge_labels_support", None)
        counter_logits = getattr(self.model, "latest_edge_logits_counter", None)
        counter_labels = getattr(self.model, "latest_edge_labels_counter", None)

        evidence_terms = []
        if support_logits is not None and support_labels is not None:
            evidence_terms.append(
                self._bce(support_logits.view(-1), support_labels.float().view(-1))
            )
        if counter_logits is not None and counter_labels is not None:
            evidence_terms.append(
                self._bce(counter_logits.view(-1), counter_labels.float().view(-1))
            )

        if evidence_terms:
            evidence_loss = sum(evidence_terms) / float(len(evidence_terms))
        else:
            edge_logits = getattr(self.model, "latest_edge_logits", None)
            edge_labels = getattr(self.model, "latest_edge_labels", None)
            if edge_logits is None or edge_labels is None:
                evidence_loss = torch.zeros((), device=claim_logits.device, dtype=claim_logits.dtype)
            else:
                evidence_loss = self._bce(edge_logits.view(-1), edge_labels.float().view(-1))

        total_loss = claim_loss + (self.lambda_evidence * evidence_loss)
        self.last_bce = float(claim_loss.detach().item())
        self.last_evidence = float(evidence_loss.detach().item())
        self.last_total = float(total_loss.detach().item())
        return total_loss


class MultiTaskPVModel(nn.Module):
    """Wrapper adding auxiliary support/counter evidence heads around PV_QAGNN outputs."""

    def __init__(self, base_model: Any, evidence_head_in_dim: int | None = None):
        super().__init__()
        self.base_model = base_model
        inferred_dim = int(evidence_head_in_dim or getattr(base_model, "gnn_out_features", 256))
        self.evidence_head_support = nn.Linear(inferred_dim, 1)
        self.evidence_head_counter = nn.Linear(inferred_dim, 1)
        # Backward-compatible alias used by older utilities.
        self.evidence_head = self.evidence_head_support
        self.name = getattr(self.base_model, "name", "pv_qagnn_component3")

        self.latest_edge_logits = None
        self.latest_edge_labels = None
        self.latest_edge_logits_support = None
        self.latest_edge_labels_support = None
        self.latest_edge_logits_counter = None
        self.latest_edge_labels_counter = None
        self.latest_component5_context = {}

    def forward(self, claim_tokens, data_graph):
        import torch  # type: ignore

        claim_logits = self.base_model(claim_tokens, data_graph)

        self.latest_edge_labels_support = getattr(
            data_graph, "edge_is_support", getattr(data_graph, "edge_is_gold", None)
        )
        self.latest_edge_labels_counter = getattr(data_graph, "edge_is_counter", None)
        self.latest_edge_labels = self.latest_edge_labels_support
        self.latest_edge_logits = None
        self.latest_edge_logits_support = None
        self.latest_edge_logits_counter = None

        x_sup = getattr(self.base_model, "latest_x_sup", None)
        x_ref = getattr(self.base_model, "latest_x_ref", None)
        edge_index = getattr(self.base_model, "latest_edge_index", None)

        if x_sup is not None and x_ref is not None and edge_index is not None:
            src = edge_index[0]
            edge_repr = x_sup[src] + x_ref[src]
            if edge_repr.dim() != 2:
                raise ValueError(
                    f"Expected edge_repr to be rank-2, got shape {tuple(edge_repr.shape)}."
                )
            if int(edge_repr.size(1)) != int(self.evidence_head_support.in_features):
                raise ValueError(
                    "Evidence head input mismatch: "
                    f"expected {self.evidence_head_support.in_features}, got {int(edge_repr.size(1))}."
                )
            self.latest_edge_logits_support = self.evidence_head_support(edge_repr).view(-1)
            self.latest_edge_logits_counter = self.evidence_head_counter(edge_repr).view(-1)
            self.latest_edge_logits = self.latest_edge_logits_support
        elif self.latest_edge_labels_support is not None:
            self.latest_edge_logits_support = torch.zeros(
                self.latest_edge_labels_support.view(-1).shape[0],
                device=claim_logits.device,
                dtype=claim_logits.dtype,
            )
            self.latest_edge_logits_counter = torch.zeros_like(self.latest_edge_logits_support)
            self.latest_edge_logits = self.latest_edge_logits_support

        if self.latest_edge_labels_support is None and self.latest_edge_logits_support is not None:
            self.latest_edge_labels_support = torch.zeros_like(self.latest_edge_logits_support)
            self.latest_edge_labels = self.latest_edge_labels_support
        if self.latest_edge_labels_counter is None and self.latest_edge_logits_counter is not None:
            self.latest_edge_labels_counter = torch.zeros_like(self.latest_edge_logits_counter)

        edge_attr = getattr(data_graph, "edge_attr", None)
        zero = torch.zeros((), device=claim_logits.device, dtype=claim_logits.dtype)
        prob_restored = torch.sigmoid(claim_logits.view(-1)).mean() if claim_logits.numel() > 0 else zero
        self.latest_component5_context = {
            "mean_rel_a": zero,
            "max_contra_pool": zero,
            "max_contra_used": zero,
            "prob_restored": prob_restored,
            "prob_masked": prob_restored,
        }

        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) >= 5:
            p_con = edge_attr[:, -4]
            rel = edge_attr[:, -2]
            mean_rel_a = rel.mean() if rel.numel() > 0 else zero
            max_contra_pool = p_con.max() if p_con.numel() > 0 else zero

            if self.latest_edge_logits_counter is not None and self.latest_edge_logits_counter.numel() > 0:
                selected_mask = torch.sigmoid(self.latest_edge_logits_counter.view(-1)) > 0.5
                if bool(selected_mask.any()):
                    max_contra_used = p_con[selected_mask].max()
                else:
                    max_contra_used = zero
            else:
                max_contra_used = zero

            prob_masked = prob_restored
            if (
                self.latest_edge_labels_support is not None
                and rel.shape[0] == self.latest_edge_labels_support.view(-1).shape[0]
            ):
                gold_mask = self.latest_edge_labels_support.view(-1).float() > 0.5
                if bool(gold_mask.any()):
                    synthetic_drop = (rel[gold_mask].mean() * 0.1).clamp(min=0.0)
                    prob_masked = torch.clamp(prob_restored - synthetic_drop, min=0.0, max=1.0)

            self.latest_component5_context = {
                "mean_rel_a": mean_rel_a,
                "max_contra_pool": max_contra_pool,
                "max_contra_used": max_contra_used,
                "prob_restored": prob_restored,
                "prob_masked": prob_masked,
            }

        return claim_logits


def _flatten_label(label_value: Any) -> int:
    if isinstance(label_value, (list, tuple)):
        if len(label_value) == 0:
            return 0
        return int(label_value[0])
    return int(label_value)


def _ensure_binary_labels(labels: list[Any], *, split: str, dataset_name: str) -> None:
    flattened = sorted({_flatten_label(v) for v in labels})
    invalid = [value for value in flattened if value not in (0, 1)]
    if invalid:
        raise ValueError(
            f"{dataset_name} split `{split}` contains unsupported labels {invalid}. "
            "Current Component 3 training path is binary-only. "
            "Use FEVER binary artifacts (exclude NEI) or implement a 3-way classifier head."
        )


def _extract_labels_from_dataset(dataset: Any) -> list[int] | None:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [_flatten_label(v) for v in labels]

    base_dataset = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if base_dataset is not None and indices is not None:
        base_labels = _extract_labels_from_dataset(base_dataset)
        if base_labels is None:
            return None
        return [int(base_labels[int(i)]) for i in indices]
    return None


def _deterministic_stratified_indices(labels: list[int], max_samples: int, seed: int) -> list[int]:
    from collections import defaultdict

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)

    rng = random.Random(seed)
    for class_indices in by_label.values():
        rng.shuffle(class_indices)

    total = len(labels)
    if total <= 0:
        return []

    exact_targets = {
        label: (len(indices) * float(max_samples) / float(total)) for label, indices in by_label.items()
    }
    selected_counts = {label: int(target) for label, target in exact_targets.items()}

    remaining = max_samples - sum(selected_counts.values())
    if remaining > 0:
        fractional = sorted(
            (
                (exact_targets[label] - selected_counts[label], label)
                for label in by_label
                if selected_counts[label] < len(by_label[label])
            ),
            reverse=True,
        )
        ptr = 0
        while remaining > 0 and fractional:
            _, label = fractional[ptr % len(fractional)]
            if selected_counts[label] < len(by_label[label]):
                selected_counts[label] += 1
                remaining -= 1
            ptr += 1
            if ptr > (10 * len(labels)):
                break

    selected_indices: list[int] = []
    for label, class_indices in by_label.items():
        take = min(selected_counts.get(label, 0), len(class_indices))
        selected_indices.extend(class_indices[:take])

    if len(selected_indices) < max_samples:
        leftovers: list[int] = []
        for label, class_indices in by_label.items():
            take = min(selected_counts.get(label, 0), len(class_indices))
            leftovers.extend(class_indices[take:])
        rng.shuffle(leftovers)
        selected_indices.extend(leftovers[: max_samples - len(selected_indices)])

    selected_indices = selected_indices[:max_samples]
    selected_indices.sort()
    return selected_indices


def _subset_dataloader(
    dataloader: Any,
    max_samples: int,
    subset_sampling: str = "stratified",
    seed: int = 57,
):
    """Create a subset dataloader preserving collate/runtime flags."""

    if max_samples <= 0:
        return dataloader
    if len(dataloader.dataset) <= max_samples:
        return dataloader

    from torch.utils.data import DataLoader, Subset  # type: ignore

    if subset_sampling not in {"stratified", "prefix"}:
        raise ValueError("Argument `subset_sampling` must be one of: stratified, prefix.")

    if subset_sampling == "prefix":
        selected_indices = list(range(max_samples))
    else:
        labels = _extract_labels_from_dataset(dataloader.dataset)
        if labels is None:
            selected_indices = list(range(max_samples))
        else:
            selected_indices = _deterministic_stratified_indices(labels=labels, max_samples=max_samples, seed=seed)

    subset = Subset(dataloader.dataset, selected_indices)

    kwargs: dict[str, Any] = {
        "batch_size": dataloader.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": dataloader.collate_fn,
        "num_workers": getattr(dataloader, "num_workers", 0),
        "pin_memory": getattr(dataloader, "pin_memory", False),
    }
    if hasattr(dataloader, "timeout"):
        kwargs["timeout"] = getattr(dataloader, "timeout")
    if hasattr(dataloader, "worker_init_fn"):
        kwargs["worker_init_fn"] = getattr(dataloader, "worker_init_fn")
    if hasattr(dataloader, "persistent_workers") and int(kwargs.get("num_workers", 0)) > 0:
        kwargs["persistent_workers"] = getattr(dataloader, "persistent_workers")
    if hasattr(dataloader, "prefetch_factor") and int(kwargs.get("num_workers", 0)) > 0:
        prefetch_factor = getattr(dataloader, "prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(subset, **kwargs)


def _build_model(config: Component3TrainConfig):
    from component3.pv_qagnn import PV_QAGNN

    return PV_QAGNN(
        model_name=config.model_name,
        n_gnn_layers=config.n_gnn_layers,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_out_features=config.gnn_out_features,
        gnn_batch_norm=config.gnn_batch_norm,
        freeze_base_model=False,
        freeze_up_to_pooler=False,
        gnn_dropout=config.gnn_dropout,
        classifier_dropout=config.classifier_dropout,
        lm_layer_dropout=config.lm_layer_dropout,
        use_roberta=config.use_roberta,
    )


def _build_factkg_loaders(config: Component3TrainConfig):
    from component3.c1_pairs_loader import load_component1_evidence_rows
    from component3.pv_dataset import FactKGPVDatasetGraph
    from datasets import GraphCollateFunc, get_df, get_subgraphs
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if config.online_embeddings:
        raise ValueError(
            "Component 3 training requires precomputed entity embeddings. "
            "`--online-embeddings` is not supported with the PV dataset loader."
        )

    tokenizer_name = "roberta-base" if config.use_roberta else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    graph_collate = GraphCollateFunc(tokenizer, max_length=config.max_seq_len)

    coverage: dict[str, Any] = {}

    def _build_split_loader(split: str, *, batch_size: int, shuffle: bool, drop_last: bool):
        claims_df = get_df(split).reset_index(drop=True)
        subgraphs_df = get_subgraphs(split, config.subgraph_type).reset_index(drop=True)

        if config.use_component1_pairs:
            evidence_rows, split_coverage = load_component1_evidence_rows(
                split=split,
                claims_df=claims_df,
                subgraphs_df=subgraphs_df,
                logs_root=config.component1_logs_root,
                missing_policy=config.missing_pv_policy,
            )
        else:
            evidence_rows = subgraphs_df
            split_coverage = {
                "split": split,
                "source": "raw_subgraphs",
                "claims_total": len(claims_df),
                "claims_using_fallback": 0,
                "claims_using_fallback_pct": 0.0,
                "pair_rows_total": 0,
                "pair_rows_parsed": 0,
            }

        coverage[split] = split_coverage
        dataset = FactKGPVDatasetGraph(
            df=claims_df,
            evidence=evidence_rows,
            claim_triple_cache_path=config.factkg_claim_triple_cache_path,
            claim_max_length=config.max_seq_len,
            require_pv_metadata=False,
            add_reverse_edges=True,
            auto_precompute=False,
            require_claim_triple_cache=config.factkg_require_claim_triple_cache,
            precompute_batch_size=config.factkg_precompute_batch_size,
        )
        return DataLoader(dataset, **_build_loader_kwargs(
            config,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=graph_collate,
        ))

    train_loader = _build_split_loader(
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = _build_split_loader(
        "val",
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = _build_split_loader(
        "test",
        batch_size=max(1, config.batch_size // 2),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, coverage


def _load_fever_split_df(data_root: Path, split: str):
    import pandas as pd

    split_to_file = {
        "train": "fever_train.pkl",
        "val": "fever_dev.pkl",
        "test": "fever_test.pkl",
    }
    if split not in split_to_file:
        raise ValueError(f"Unsupported FEVER split `{split}`.")

    path = data_root / split_to_file[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FEVER artifact: {path}. "
            "Run `python scripts/prepare_fever_dataset.py` first."
        )
    return pd.read_pickle(path)


def _build_fever_loaders(config: Component3TrainConfig):
    from component3.c1_pairs_loader import load_component1_evidence_rows_fever
    from component3.pv_dataset_fever import FeverPVDatasetGraph
    from datasets import GraphCollateFunc
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if config.online_embeddings:
        raise ValueError(
            "Component 3 FEVER training requires precomputed sentence embeddings via the FEVER PV dataset. "
            "`--online-embeddings` is not supported."
        )

    tokenizer_name = "roberta-base" if config.use_roberta else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    graph_collate = GraphCollateFunc(tokenizer, max_length=config.max_seq_len)

    data_root = Path(config.fever_data_root)
    cache_tag = f"{tokenizer_name}_len{int(config.max_seq_len)}"
    cache_tag = cache_tag.replace("/", "_").replace("\\", "_").replace("-", "_")
    coverage: dict[str, Any] = {}

    def _build_split_loader(split: str, *, batch_size: int, shuffle: bool, drop_last: bool):
        claims_df = _load_fever_split_df(data_root, split).reset_index(drop=True)
        _ensure_binary_labels(
            list(claims_df["Label"]) if "Label" in claims_df.columns else [],
            split=split,
            dataset_name="FEVER",
        )

        if config.use_component1_pairs:
            evidence_rows, split_coverage = load_component1_evidence_rows_fever(
                split=split,
                claims_df=claims_df,
                logs_root=config.fever_component1_logs_root,
                missing_policy=config.missing_pv_policy,
            )
        else:
            evidence_rows = claims_df["evidence_rows"].tolist() if "evidence_rows" in claims_df.columns else []
            split_coverage = {
                "split": split,
                "source": "fever_raw_evidence_rows",
                "claims_total": len(claims_df),
                "claims_using_fallback": 0,
                "claims_using_fallback_pct": 0.0,
                "pair_rows_total": 0,
                "pair_rows_parsed": 0,
            }

        coverage[split] = split_coverage
        dataset = FeverPVDatasetGraph(
            df=claims_df,
            evidence=evidence_rows,
            sentence_cache_path=data_root / f"sentence_embeddings_{cache_tag}.pkl",
            claim_sentence_cache_path=data_root / f"claim_sentence_embeddings_{cache_tag}.pkl",
            sentence_model_name=tokenizer_name,
            max_seq_len=config.max_seq_len,
            include_s_pool=config.fever_include_s_pool,
            auto_precompute=config.fever_auto_precompute_embeddings,
            precompute_batch_size=config.factkg_precompute_batch_size,
        )
        return DataLoader(dataset, **_build_loader_kwargs(
            config,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=graph_collate,
        ))

    train_loader = _build_split_loader(
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = _build_split_loader(
        "val",
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = _build_split_loader(
        "test",
        batch_size=max(1, config.batch_size // 2),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, coverage


def _build_default_loaders(config: Component3TrainConfig):
    dataset_name = str(config.dataset_name).strip().lower()
    if dataset_name == "factkg":
        return _build_factkg_loaders(config)
    if dataset_name == "fever":
        return _build_fever_loaders(config)
    raise ValueError("Argument `dataset_name` must be either `factkg` or `fever`.")


def _validate_component1_pair_coverage(config: Component3TrainConfig, coverage_stats: dict[str, Any]) -> None:
    """Fail fast when Component 1 pair logs are missing/unusable."""
    if not config.use_component1_pairs:
        return
    if not isinstance(coverage_stats, dict) or not coverage_stats:
        raise RuntimeError(
            "Component 1 pair-log coverage statistics are missing. "
            "Refusing to continue with implicit fallback evidence."
        )

    for split in ("train", "val", "test"):
        split_cov = coverage_stats.get(split)
        if not isinstance(split_cov, dict):
            raise RuntimeError(
                f"Missing coverage entry for split `{split}`. "
                "Cannot validate Component 1 pair-log integrity."
            )
        if not bool(split_cov.get("pairs_file_exists", False)):
            if str(config.dataset_name).strip().lower() == "fever":
                hint = "Run scripts/run_fever_component1.py"
            else:
                hint = "Run scripts/run_component1_pv_esm.py"
            raise FileNotFoundError(
                f"Missing Component 1 pair log for split `{split}`: {split_cov.get('pairs_file')}. "
                f"{hint} or set --no-use-component1-pairs explicitly."
            )

        pair_rows_total = int(split_cov.get("pair_rows_total", 0))
        pair_rows_parsed = int(split_cov.get("pair_rows_parsed", 0))
        if pair_rows_total <= 0:
            raise RuntimeError(
                f"Component 1 pair log for split `{split}` is empty ({split_cov.get('pairs_file')}). "
                "Refusing to continue with silent fallback rows."
            )
        if pair_rows_parsed <= 0:
            raise RuntimeError(
                f"Component 1 pair log for split `{split}` has zero parseable rows "
                f"({split_cov.get('pairs_file')}). The file may be malformed."
            )


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _evaluate_no_collapse_gate(
    test_metrics: dict[str, Any],
    stage_results: list[dict[str, Any]],
    *,
    precision_accuracy_gap_min: float,
) -> dict[str, Any]:
    overall = test_metrics.get("overall", {}) if isinstance(test_metrics, dict) else {}
    recall = float(overall.get("recall", 0.0))
    precision = float(overall.get("precision", 0.0))
    accuracy = float(overall.get("accuracy", 0.0))

    best_val_accuracy = max(
        float(stage.get("best_val_accuracy", float("-inf")))
        for stage in stage_results
    ) if stage_results else float("-inf")

    checks = {
        "recall_lt_0_99": recall < 0.99,
        "precision_accuracy_gap_ge_min": abs(precision - accuracy) >= float(precision_accuracy_gap_min),
        "best_val_accuracy_ge_55": best_val_accuracy >= 55.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "overall_recall": recall,
            "overall_precision": precision,
            "overall_accuracy": accuracy,
            "precision_accuracy_gap": abs(precision - accuracy),
            "precision_accuracy_gap_min": float(precision_accuracy_gap_min),
            "best_val_accuracy": best_val_accuracy,
        },
    }


def run_training(
    config: Component3TrainConfig,
    *,
    model: Any | None = None,
    train_loader: Any | None = None,
    val_loader: Any | None = None,
    test_loader: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run two-stage training with existing train/eval infrastructure."""

    import torch  # type: ignore
    import transformers  # type: ignore
    try:
        from evaluate import evaluate_on_test_set
        from train import train
        from utils import seed_everything
    except ModuleNotFoundError:
        import sys

        # Robust fallback when executed via module/file entrypoints where repo root
        # is not already on sys.path (evaluate.py/train.py live at repository root).
        repo_root = Path(__file__).resolve().parents[2]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from evaluate import evaluate_on_test_set
        from train import train
        from utils import seed_everything

    if config.gradient_accumulation_steps <= 0:
        raise ValueError("Argument `gradient_accumulation_steps` must be positive.")
    if config.loader_num_workers < 0:
        raise ValueError("Argument `loader_num_workers` must be >= 0.")
    if config.loader_prefetch_factor < 1:
        raise ValueError("Argument `loader_prefetch_factor` must be >= 1.")
    if config.factkg_precompute_batch_size < 1:
        raise ValueError("Argument `factkg_precompute_batch_size` must be >= 1.")
    if config.no_collapse_precision_accuracy_gap_min < 0.0:
        raise ValueError("Argument `no_collapse_precision_accuracy_gap_min` must be >= 0.")

    seed_everything(config.seed)
    if config.deterministic_mode:
        _enable_deterministic_mode(config.seed)

    run_id = config.run_id or f"component3_t33_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_yaml(run_dir / "config.yaml", config_to_dict(config))

    if model is None:
        model = _build_model(config)
    wrapped_model = model if isinstance(model, MultiTaskPVModel) else MultiTaskPVModel(
        model,
        evidence_head_in_dim=config.gnn_out_features,
    )

    coverage_stats: dict[str, Any] = {}
    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader, coverage_stats = _build_default_loaders(config)
        _validate_component1_pair_coverage(config, coverage_stats)

    train_loader = _subset_dataloader(
        train_loader,
        max_samples=config.train_subset_size,
        subset_sampling=config.subset_sampling,
        seed=config.seed,
    )
    val_loader = _subset_dataloader(
        val_loader,
        max_samples=config.val_subset_size,
        subset_sampling=config.subset_sampling,
        seed=config.seed,
    )
    if len(train_loader) <= 0:
        raise ValueError(
            "Training loader is empty after batching/subsetting. "
            "Increase split size, reduce batch size, or disable drop_last for tiny debug runs."
        )

    stages = build_stage_plan(config)
    configure_trainable_parameters(wrapped_model, unfreeze_pooler=stages[0].unfreeze_pooler)

    optimizer = torch.optim.AdamW(
        wrapped_model.parameters(),
        lr=stages[0].learning_rate,
    )

    stage_results: list[dict[str, Any]] = []
    for stage in stages:
        configure_trainable_parameters(wrapped_model, unfreeze_pooler=stage.unfreeze_pooler)
        _set_optimizer_lr(optimizer, stage.learning_rate)

        if config.enable_component5:
            from component3.controller import Component5JointLossAdapter

            criterion = Component5JointLossAdapter(
                wrapped_model,
                lambda_evidence=config.component5_evidence_weight,
                lambda_starv=config.lambda_starv,
                tau_starv=config.tau_starv,
                lambda_counter=config.lambda_counter,
                lambda_recovery=config.lambda_recovery,
                loss_ablation_variant=config.loss_ablation_variant,
            )
        else:
            criterion = MultiTaskLossAdapter(wrapped_model, lambda_evidence=config.lambda_evidence)

        steps_per_epoch = max(1, math.ceil(len(train_loader) / float(config.gradient_accumulation_steps)))
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=steps_per_epoch * stage.epochs,
        )

        history, models_dict = train(
            model=wrapped_model,
            criterion=criterion,
            optimizer=optimizer,
            qa_gnn=True,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=stage.epochs,
            scheduler=scheduler,
            grad_accum_steps=config.gradient_accumulation_steps,
            n_early_stop=config.n_early_stop,
            save_models=False,
            non_blocking=bool(config.non_blocking_transfers and config.loader_pin_memory),
            verbose=0,
        )
        best_state_dict = models_dict.get("best_model_state_dict")
        if best_state_dict is not None:
            wrapped_model.load_state_dict(best_state_dict)

        stage_results.append(
            {
                "stage": stage.name,
                "epochs": stage.epochs,
                "learning_rate": stage.learning_rate,
                "unfreeze_pooler": stage.unfreeze_pooler,
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss"),
                "best_val_accuracy": history.get("best_val_accuracy"),
                "last_loss_bce": criterion.last_bce,
                "last_loss_evidence": criterion.last_evidence,
                "last_loss_starvation": getattr(criterion, "last_starvation", 0.0),
                "last_loss_counter": getattr(criterion, "last_counter", 0.0),
                "last_loss_recovery": getattr(criterion, "last_recovery", 0.0),
                "last_loss_total": criterion.last_total,
                "trainable_summary": _summarize_trainable_parameters(wrapped_model),
                "optimizer_id": id(optimizer),
                "optimizer_state_size": len(optimizer.state),
            }
        )

    results: dict[str, Any] = {
        "run_id": run_id,
        "stages": stage_results,
        "data_contract": {
            "component1_coverage": coverage_stats,
            "missing_pv_policy": config.missing_pv_policy,
            "use_component1_pairs": config.use_component1_pairs,
        },
    }

    if config.evaluate_after_training:
        claim_only_criterion = torch.nn.BCEWithLogitsLoss()
        metrics = evaluate_on_test_set(
            qa_gnn=True,
            model=wrapped_model,
            test_loader=test_loader,
            criterion=claim_only_criterion,
            dataset_name=config.dataset_name,
        )
        results["test_metrics"] = metrics
        if str(config.dataset_name).strip().lower() == "factkg":
            gate = _evaluate_no_collapse_gate(
                metrics,
                stage_results,
                precision_accuracy_gap_min=config.no_collapse_precision_accuracy_gap_min,
            )
        else:
            gate = {
                "passed": True,
                "skipped": True,
                "reason": "no_collapse_gate is calibrated for FactKG binary metrics only.",
            }
        results["no_collapse_gate"] = gate

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if (
        config.evaluate_after_training
        and config.enforce_no_collapse_gate
        and (not results.get("no_collapse_gate", {}).get("passed", False))
    ):
        raise RuntimeError(
            "No-collapse gate failed. See metrics.json -> no_collapse_gate for details."
        )

    return run_dir, results


def _parse_args() -> Component3TrainConfig:
    parser = argparse.ArgumentParser(description="Component 3 two-stage trainer")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--model-name", type=str, default="pv_qagnn_component3")
    parser.add_argument("--dataset-name", type=str, choices=["factkg", "fever"], default="factkg")
    parser.add_argument("--subgraph-type", type=str, default="direct_filled")
    parser.add_argument("--fever-data-root", type=str, default="data/fever")
    parser.add_argument("--fever-component1-logs-root", type=str, default="logs/component1_fever")
    parser.add_argument("--fever-include-s-pool", action="store_true")
    parser.add_argument("--fever-auto-precompute-embeddings", action="store_true")
    parser.add_argument("--loader-num-workers", type=int, default=0)
    parser.add_argument("--loader-pin-memory", action="store_true")
    parser.add_argument("--loader-persistent-workers", action="store_true")
    parser.add_argument("--loader-prefetch-factor", type=int, default=2)
    parser.add_argument("--non-blocking-transfers", action="store_true")
    parser.add_argument("--factkg-claim-triple-cache-path", type=str, default="data/claim_triple_embeddings.pkl")
    parser.add_argument("--factkg-require-claim-triple-cache", action="store_true")
    parser.add_argument("--factkg-precompute-batch-size", type=int, default=32)
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--no-collapse-precision-accuracy-gap-min", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--train-subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--subset-sampling", type=str, choices=["stratified", "prefix"], default="stratified")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--lambda-evidence", type=float, default=0.1)
    parser.add_argument("--enable-component5", action="store_true")
    parser.add_argument("--component5-evidence-weight", type=float, default=1.0)
    parser.add_argument("--lambda-starv", type=float, default=0.1)
    parser.add_argument("--tau-starv", type=float, default=0.3)
    parser.add_argument("--lambda-counter", type=float, default=0.1)
    parser.add_argument("--lambda-recovery", type=float, default=0.1)
    parser.add_argument("--loss-ablation-variant", type=str, default="full")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=1.0e-5)
    parser.add_argument("--stage2-lr", type=float, default=2.0e-6)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--n-early-stop", type=int, default=3)
    parser.add_argument("--online-embeddings", action="store_true")
    parser.add_argument("--mix-graphs", action="store_true")
    parser.add_argument("--use-roberta", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--component1-logs-root", type=str, default="logs/component1")
    parser.add_argument("--missing-pv-policy", type=str, default="hybrid_fallback")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--use-component1-pairs", dest="use_component1_pairs", action="store_true")
    parser.add_argument("--no-use-component1-pairs", dest="use_component1_pairs", action="store_false")
    parser.set_defaults(use_component1_pairs=True)

    parser.add_argument("--no-enforce-no-collapse-gate", dest="enforce_no_collapse_gate", action="store_false")
    parser.set_defaults(enforce_no_collapse_gate=True)

    args = parser.parse_args()

    return Component3TrainConfig(
        run_id=args.run_id,
        output_root=args.output_root,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        subgraph_type=args.subgraph_type,
        fever_data_root=args.fever_data_root,
        fever_component1_logs_root=args.fever_component1_logs_root,
        fever_include_s_pool=args.fever_include_s_pool,
        fever_auto_precompute_embeddings=args.fever_auto_precompute_embeddings,
        loader_num_workers=args.loader_num_workers,
        loader_pin_memory=args.loader_pin_memory,
        loader_persistent_workers=args.loader_persistent_workers,
        loader_prefetch_factor=args.loader_prefetch_factor,
        non_blocking_transfers=args.non_blocking_transfers,
        factkg_claim_triple_cache_path=args.factkg_claim_triple_cache_path,
        factkg_require_claim_triple_cache=args.factkg_require_claim_triple_cache,
        factkg_precompute_batch_size=args.factkg_precompute_batch_size,
        deterministic_mode=args.deterministic_mode,
        no_collapse_precision_accuracy_gap_min=args.no_collapse_precision_accuracy_gap_min,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        subset_sampling=args.subset_sampling,
        seed=args.seed,
        lambda_evidence=args.lambda_evidence,
        enable_component5=args.enable_component5,
        component5_evidence_weight=args.component5_evidence_weight,
        lambda_starv=args.lambda_starv,
        tau_starv=args.tau_starv,
        lambda_counter=args.lambda_counter,
        lambda_recovery=args.lambda_recovery,
        loss_ablation_variant=args.loss_ablation_variant,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        warmup_steps=args.warmup_steps,
        n_early_stop=args.n_early_stop,
        online_embeddings=args.online_embeddings,
        mix_graphs=args.mix_graphs,
        use_roberta=args.use_roberta,
        evaluate_after_training=(not args.skip_eval),
        use_component1_pairs=args.use_component1_pairs,
        component1_logs_root=args.component1_logs_root,
        missing_pv_policy=args.missing_pv_policy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enforce_no_collapse_gate=args.enforce_no_collapse_gate,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    run_dir, run_results = run_training(cfg)
    print(f"Run directory: {run_dir}")
    print(json.dumps(run_results, indent=2))
