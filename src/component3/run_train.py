"""Component 3 training entrypoint using existing QA-GNN infrastructure."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

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
    model_mode: str = "pv_qagnn"
    encoder_tune: str = "none"
    unfreeze_last_n: int = 2
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
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
    factkg_require_pv_metadata: bool = True
    factkg_precompute_batch_size: int = 32
    factkg_include_s_pool: bool = True
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
    edge_message_dim: int = 64
    edge_encoder_dropout: float = 0.1
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
    enable_backtracking: bool = True
    backtracking_margin_threshold: float = 0.15
    backtracking_min_a: int = 5
    backtracking_rel_threshold: float = 0.3
    backtracking_promotion_gamma: float = 0.0
    backtracking_directional_delta: float = 0.0
    backtracking_challenge_mode: bool = False
    backtracking_hunger_mode: str = "percentile"
    backtracking_hunger_percentile: float = 0.9
    backtracking_max_rounds: int = 2
    backtracking_top_k: int = 3
    backtracking_do_no_harm_eps: float = 0.002
    backtracking_flip_conf_min: float = 0.1
    backtracking_flip_abslogit_eps: float = 0.002
    backtracking_candidate_log_top_n: int = 25
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


def _infer_edge_feature_dim_from_loader(loader: Any) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError("Cannot infer edge feature dim: loader has no dataset.")
    if len(dataset) <= 0:
        raise ValueError("Cannot infer edge feature dim: dataset is empty.")

    sample = dataset[0]
    if not isinstance(sample, (list, tuple)) or len(sample) < 2:
        raise ValueError("Cannot infer edge feature dim: expected dataset item (claim_text, graph, label).")
    graph = sample[1]
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is None or getattr(edge_attr, "dim", lambda: 0)() != 2:
        raise ValueError("Cannot infer edge feature dim: graph missing rank-2 `edge_attr`.")
    edge_dim = int(edge_attr.size(1))
    if edge_dim < 5:
        raise ValueError(
            "Cannot infer edge feature dim: expected trailing PV metadata "
            f"[p_ent,p_con,p_neu,rel,pool_id], got edge_dim={edge_dim}."
        )
    return edge_dim


class ClaimOnlyBinaryModel(nn.Module):
    """Claim-only binary verifier that ignores graph inputs but keeps label/metric path identical."""

    def __init__(
        self,
        *,
        model_name: str,
        classifier_dropout: float,
        use_roberta: bool,
    ) -> None:
        super().__init__()
        from models import get_bert_model

        self.name = model_name
        self.bert = get_bert_model(
            model_name=f"bert_{model_name}",
            include_classifier=False,
            freeze_base_model=False,
            freeze_up_to_pooler=False,
            use_roberta=use_roberta,
        )
        self.classifier_dropout_layer = nn.Dropout(float(classifier_dropout))
        self.classifier = nn.Linear(int(self.bert.config.hidden_size), 1)

    def forward(self, claim_tokens, _data_graph):
        claim_outputs = self.bert(**claim_tokens)
        claim_embeddings = claim_outputs.last_hidden_state[:, 0]
        claim_embeddings = self.classifier_dropout_layer(claim_embeddings)
        logits = self.classifier(claim_embeddings)
        return logits.squeeze(1)


def _build_claim_only_model(config: Component3TrainConfig) -> ClaimOnlyBinaryModel:
    return ClaimOnlyBinaryModel(
        model_name=str(config.model_name),
        classifier_dropout=float(config.classifier_dropout),
        use_roberta=bool(config.use_roberta),
    )


def _apply_lora_adapter(model: Any, config: Component3TrainConfig):
    if str(config.encoder_tune).strip().lower() != "lora":
        return model
    if not hasattr(model, "bert"):
        raise ValueError("LoRA tuning requires a model with a `bert` encoder attribute.")
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "LoRA requested but `peft` is unavailable. Install with `pip install peft` in the active env."
        ) from exc

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=int(config.lora_r),
        lora_alpha=float(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        bias="none",
        target_modules=["query", "key", "value"],
    )
    model.bert = get_peft_model(model.bert, lora_cfg)
    return model


def _collect_encoder_layer_ids(model: Any) -> list[int]:
    layer_ids: set[int] = set()
    pattern = re.compile(r"\.encoder\.layer\.(\d+)\.")
    for name, _ in model.named_parameters():
        match = pattern.search(str(name))
        if match is None:
            continue
        layer_ids.add(int(match.group(1)))
    return sorted(layer_ids)


def _is_lora_parameter(param_name: str) -> bool:
    lowered = str(param_name).lower()
    return ("lora_" in lowered) or (".lora." in lowered)


def _apply_encoder_tune_trainable_parameters(
    model: Any,
    *,
    encoder_tune: str,
    unfreeze_last_n: int,
) -> None:
    mode = str(encoder_tune).strip()
    if mode == "none":
        return
    if mode == "lora":
        for name, parameter in model.named_parameters():
            if _is_lora_parameter(name):
                parameter.requires_grad = True
        return
    if mode != "unfreeze_lastN":
        raise ValueError("Argument `encoder_tune` must be one of: none, lora, unfreeze_lastN.")

    layer_ids = _collect_encoder_layer_ids(model)
    if not layer_ids:
        return
    keep = max(1, int(unfreeze_last_n))
    target_layers = set(layer_ids[-keep:])
    for name, parameter in model.named_parameters():
        if not _is_bert_parameter(name):
            continue
        match = re.search(r"\.encoder\.layer\.(\d+)\.", str(name))
        if match is None:
            continue
        if int(match.group(1)) in target_layers:
            parameter.requires_grad = True


def _build_model(config: Component3TrainConfig, *, edge_feature_dim: int):
    model_mode = str(config.model_mode).strip().lower()
    if model_mode == "claim_only":
        return _build_claim_only_model(config)
    if model_mode != "pv_qagnn":
        raise ValueError("Argument `model_mode` must be one of: pv_qagnn, claim_only.")

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
        edge_feature_dim=int(edge_feature_dim),
        edge_message_dim=int(config.edge_message_dim),
        edge_encoder_dropout=float(config.edge_encoder_dropout),
        use_roberta=config.use_roberta,
        promotion_gamma=config.backtracking_promotion_gamma,
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
            require_pv_metadata=bool(config.use_component1_pairs and config.factkg_require_pv_metadata),
            include_s_pool=config.factkg_include_s_pool,
            add_reverse_edges=True,
            auto_precompute=False,
            require_claim_triple_cache=config.factkg_require_claim_triple_cache,
            precompute_batch_size=config.factkg_precompute_batch_size,
        )
        if hasattr(dataset, "get_pool_summary"):
            split_coverage = dict(split_coverage)
            split_coverage["pool_summary"] = dataset.get_pool_summary()
            coverage[split] = split_coverage
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

        if str(config.dataset_name).strip().lower() == "factkg":
            pool_summary = split_cov.get("pool_summary")
            if not isinstance(pool_summary, dict):
                raise RuntimeError(
                    f"Missing pool_summary for FactKG split `{split}`. "
                    "Cannot enforce strict artifact validation."
                )
            if "fallback_total" not in pool_summary or "missing_embeddings_total" not in pool_summary:
                raise RuntimeError(
                    f"Pool summary for split `{split}` is missing strict validation fields "
                    "(`fallback_total`, `missing_embeddings_total`)."
                )

            fallback_total = int(pool_summary.get("fallback_total", 0))
            missing_embeddings_total = int(pool_summary.get("missing_embeddings_total", 0))
            claims_using_fallback = int(split_cov.get("claims_using_fallback", 0))
            fallback_edge_rows = int(split_cov.get("fallback_edge_rows", 0))
            pair_rows_missing_schema_version = int(split_cov.get("pair_rows_missing_schema_version", 0))
            pair_rows_non_v2_schema = int(split_cov.get("pair_rows_non_v2_schema", 0))
            if config.factkg_require_pv_metadata and fallback_total > 0:
                raise RuntimeError(
                    f"FactKG split `{split}` contains {fallback_total} fallback evidence rows "
                    "while strict PV metadata mode is enabled."
                )
            if config.factkg_require_pv_metadata and (claims_using_fallback > 0 or fallback_edge_rows > 0):
                raise RuntimeError(
                    f"FactKG split `{split}` used Component1 fallback paths for {claims_using_fallback} claims "
                    f"({fallback_edge_rows} fallback edges) while strict PV metadata mode is enabled."
                )
            if missing_embeddings_total > 0:
                raise RuntimeError(
                    f"FactKG split `{split}` references {missing_embeddings_total} missing entity embeddings. "
                    "Populate entity embeddings before training."
                )
            if config.factkg_require_pv_metadata and pair_rows_missing_schema_version > 0:
                raise RuntimeError(
                    f"FactKG split `{split}` has {pair_rows_missing_schema_version} pair rows without "
                    "schema_version=2. Regenerate Component1 logs before strict training."
                )
            if config.factkg_require_pv_metadata and pair_rows_non_v2_schema > 0:
                raise RuntimeError(
                    f"FactKG split `{split}` has {pair_rows_non_v2_schema} pair rows with non-v2 schema. "
                    "Regenerate Component1 logs before strict training."
                )


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _is_subset_run(config: Component3TrainConfig) -> bool:
    return int(config.train_subset_size) > 0 or int(config.val_subset_size) > 0


def _should_enforce_no_collapse_gate(config: Component3TrainConfig) -> bool:
    # Use no-collapse as a hard gate for full runs only; subset/smoke runs keep
    # the signal in metrics but should not terminate the process.
    return bool(config.enforce_no_collapse_gate and (not _is_subset_run(config)))


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


def _extract_dataset_attribute(dataset: Any, attr_name: str) -> list[Any] | None:
    values = getattr(dataset, attr_name, None)
    if values is not None:
        return list(values)

    base_dataset = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if base_dataset is not None and indices is not None:
        base_values = _extract_dataset_attribute(base_dataset, attr_name)
        if base_values is None:
            return None
        return [base_values[int(i)] for i in indices]

    return None


def _normalize_claim_types(types_value: Any) -> list[str]:
    if isinstance(types_value, (list, tuple, set)):
        return [str(v) for v in types_value]
    if types_value is None:
        return []
    return [str(types_value)]


def _safe_sigmoid(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-float(logit))
        return 1.0 / (1.0 + z)
    z = math.exp(float(logit))
    return z / (1.0 + z)


def _decode_binary_prediction(prediction: Mapping[str, Any]) -> tuple[int, float]:
    if "logit" in prediction and prediction["logit"] is not None:
        prob_one = _safe_sigmoid(float(prediction["logit"]))
        return int(prob_one > 0.5), float(prob_one)

    probabilities = prediction.get("probabilities")
    if isinstance(probabilities, list) and len(probabilities) >= 2:
        prob_one = float(probabilities[0])
        return int(prob_one >= float(probabilities[1])), float(prob_one)
    if isinstance(probabilities, tuple) and len(probabilities) >= 2:
        prob_one = float(probabilities[0])
        return int(prob_one >= float(probabilities[1])), float(prob_one)

    pred_value = prediction.get("pred")
    if pred_value is not None:
        try:
            pred_int = int(pred_value)
            return int(pred_int == 1), float(pred_int == 1)
        except (TypeError, ValueError):
            pass

    return 0, 0.5


def _prediction_logit_value(prediction: Mapping[str, Any]) -> float:
    if "logit" in prediction and prediction["logit"] is not None:
        try:
            return float(prediction["logit"])
        except (TypeError, ValueError):
            return 0.0

    _pred, prob_one = _decode_binary_prediction(prediction)
    clipped = min(1.0 - 1e-6, max(1e-6, float(prob_one)))
    return float(math.log(clipped / (1.0 - clipped)))


def _prediction_edge_mass_map(prediction: Mapping[str, Any], key: str) -> dict[str, float]:
    raw_value = prediction.get(key)
    if not isinstance(raw_value, dict):
        return {}
    normalized: dict[str, float] = {}
    for raw_key, raw_mass in raw_value.items():
        try:
            normalized[str(raw_key)] = float(raw_mass)
        except (TypeError, ValueError):
            continue
    return normalized


def _slice_single_claim_tokens(claim_tokens: Any, index: int, device: Any) -> dict[str, Any]:
    return {
        key: value[index: index + 1].to(device)
        for key, value in claim_tokens.items()
    }


def _evaluate_factkg_with_backtracking(
    *,
    config: Component3TrainConfig,
    model: Any,
    test_loader: Any,
    criterion: Any,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch  # type: ignore
    from torch_geometric.data import Batch  # type: ignore
    from evaluate import _evaluate_factkg_per_type, _safe_prf
    from component3.backtracking import RuleBasedBacktrackingController

    dataset = getattr(test_loader, "dataset", None)
    if dataset is None:
        raise ValueError("Test loader dataset is unavailable for FactKG evaluation.")

    claim_ids = _extract_dataset_attribute(dataset, "claim_ids")
    if claim_ids is None:
        claim_ids = [str(i) for i in range(len(dataset))]
    claim_types = _extract_dataset_attribute(dataset, "claim_types")
    if claim_types is None:
        raise ValueError("FactKG evaluation requires claim_types metadata.")
    claim_types = [_normalize_claim_types(v) for v in claim_types]

    if len(claim_types) != len(dataset):
        raise ValueError(
            "FactKG claim_types metadata length mismatch with test dataset size: "
            f"{len(claim_types)} vs {len(dataset)}."
        )

    supports_recovery = (
        bool(config.enable_backtracking)
        and hasattr(dataset, "get_recovery_triplets")
        and hasattr(dataset, "build_graph_from_recovery_rows")
    )
    controller = None
    if supports_recovery:
        controller = RuleBasedBacktrackingController(
            max_backtrack_rounds=int(config.backtracking_max_rounds),
            backtrack_k=int(config.backtracking_top_k),
            margin_threshold=float(config.backtracking_margin_threshold),
            min_a=int(config.backtracking_min_a),
            rel_threshold=float(config.backtracking_rel_threshold),
            directional_delta=float(config.backtracking_directional_delta),
            challenge_mode=bool(config.backtracking_challenge_mode),
            hunger_mode=str(config.backtracking_hunger_mode),
            hunger_rel_percentile=float(config.backtracking_hunger_percentile),
            do_no_harm_margin_eps=float(config.backtracking_do_no_harm_eps),
            flip_conf_min=float(config.backtracking_flip_conf_min),
            flip_abslogit_eps=float(config.backtracking_flip_abslogit_eps),
        )

    total_loss = 0.0
    total_correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_samples = 0
    claim_offset = 0

    backtracking_claims_triggered = 0
    backtracking_claims_promoted = 0
    backtracking_promotions_total = 0
    backtracking_attempted_claims = 0
    claims_with_s_pool = 0
    recovery_candidates_ranked_total = 0
    recovery_candidates_logged_total = 0
    backtracking_rounds_total = 0
    backtracking_rounds_reverted = 0
    backtracking_rounds_top1_fallback = 0
    backtracking_rounds_accepted = 0

    predictions_path = run_dir / "predictions.jsonl"
    recovery_actions_path = run_dir / "recovery_actions.jsonl"
    recovery_candidates_path = run_dir / "recovery_candidates.jsonl"
    candidate_log_top_n = max(0, int(config.backtracking_candidate_log_top_n))

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model.eval()

    with (
        predictions_path.open("w", encoding="utf-8") as predictions_fp,
        recovery_actions_path.open("w", encoding="utf-8") as recovery_actions_fp,
        recovery_candidates_path.open("w", encoding="utf-8") as recovery_candidates_fp,
        torch.no_grad(),
    ):
        for claim_tokens, _graph_batch, labels in test_loader:
            batch_size = int(labels.shape[0])
            labels = labels.to(model_device).view(-1)
            graph_list = _graph_batch.to_data_list()

            for local_idx in range(batch_size):
                global_idx = claim_offset + local_idx
                if global_idx >= len(dataset):
                    break

                single_tokens = _slice_single_claim_tokens(claim_tokens, local_idx, model_device)
                label_value = int(labels[local_idx].item())
                claim_id = str(claim_ids[global_idx]) if global_idx < len(claim_ids) else str(global_idx)
                metadata = claim_types[global_idx]

                final_prediction: dict[str, Any] | None = None
                backtracking_result = None
                promoted_ids: tuple[str, ...] = ()

                if controller is not None:
                    active_rows, suspended_rows = dataset.get_recovery_triplets(global_idx)
                    if suspended_rows:
                        claims_with_s_pool += 1
                    backtracking_attempted_claims += 1

                    add_reverse_edges = bool(getattr(dataset, "add_reverse_edges", True))

                    def _edge_evidence_ids_from_active_rows(current_active_rows: Sequence[Any]) -> list[str]:
                        edge_evidence_ids: list[str] = []
                        for row_idx, row in enumerate(current_active_rows):
                            if isinstance(row, dict):
                                pool = str(row.get("pool", "A")).upper()
                                evidence_id = str(row.get("evidence_id", f"triple_{row_idx}"))
                            else:
                                pool = str(getattr(row, "pool", "A")).upper()
                                evidence_id = str(getattr(row, "evidence_id", f"triple_{row_idx}"))
                            if pool not in {"A", "C"}:
                                continue
                            edge_evidence_ids.append(evidence_id)
                            if add_reverse_edges:
                                edge_evidence_ids.append(evidence_id)
                        return edge_evidence_ids

                    def _predictor(current_active_rows):
                        rebuilt_graph = dataset.build_graph_from_recovery_rows(
                            index=global_idx,
                            active_rows=current_active_rows,
                        )
                        rebuilt_batch = Batch.from_data_list([rebuilt_graph]).to(model_device)
                        claim_logit_tensor = model(single_tokens, rebuilt_batch).view(-1)
                        logit_value = float(claim_logit_tensor[0].item())
                        prob_value = _safe_sigmoid(logit_value)
                        base_model = getattr(model, "base_model", model)
                        raw_sup = getattr(base_model, "latest_edge_weight_sup", None)
                        raw_ref = getattr(base_model, "latest_edge_weight_ref", None)
                        stream_mass_sup = 0.0
                        stream_mass_ref = 0.0
                        edge_mass_sup_by_id: dict[str, float] = {}
                        edge_mass_ref_by_id: dict[str, float] = {}
                        if raw_sup is not None and raw_ref is not None:
                            sup_values = raw_sup.view(-1).abs().detach().cpu()
                            ref_values = raw_ref.view(-1).abs().detach().cpu()
                            stream_mass_sup = float(sup_values.sum().item())
                            stream_mass_ref = float(ref_values.sum().item())
                            edge_evidence_ids = _edge_evidence_ids_from_active_rows(current_active_rows)
                            if len(edge_evidence_ids) == int(sup_values.shape[0]):
                                for edge_idx, evidence_id in enumerate(edge_evidence_ids):
                                    sup_mass = float(sup_values[edge_idx].item())
                                    ref_mass = float(ref_values[edge_idx].item())
                                    edge_mass_sup_by_id[evidence_id] = edge_mass_sup_by_id.get(evidence_id, 0.0) + sup_mass
                                    edge_mass_ref_by_id[evidence_id] = edge_mass_ref_by_id.get(evidence_id, 0.0) + ref_mass
                        return {
                            "logit": logit_value,
                            "probabilities": [prob_value, 1.0 - prob_value],
                            "margin": abs((2.0 * prob_value) - 1.0),
                            "stream_mass_sup": stream_mass_sup,
                            "stream_mass_ref": stream_mass_ref,
                            "edge_mass_sup_by_id": edge_mass_sup_by_id,
                            "edge_mass_ref_by_id": edge_mass_ref_by_id,
                        }

                    backtracking_result = controller.run(
                        claim_id=claim_id,
                        active_triples=active_rows,
                        suspended_triples=suspended_rows,
                        predictor=_predictor,
                    )
                    final_prediction = dict(backtracking_result.final_prediction)
                    promoted_ids = tuple(backtracking_result.promoted_evidence_ids)
                else:
                    graph_obj = graph_list[local_idx]
                    single_graph_batch = Batch.from_data_list([graph_obj]).to(model_device)
                    claim_logit_tensor = model(single_tokens, single_graph_batch).view(-1)
                    logit_value = float(claim_logit_tensor[0].item())
                    prob_value = _safe_sigmoid(logit_value)
                    final_prediction = {
                        "logit": logit_value,
                        "probabilities": [prob_value, 1.0 - prob_value],
                        "margin": abs((2.0 * prob_value) - 1.0),
                    }

                if final_prediction is None or "logit" not in final_prediction:
                    raise RuntimeError(f"Missing final prediction logit for claim_id={claim_id}.")

                logit = float(final_prediction["logit"])
                prob = _safe_sigmoid(logit)
                pred = int(prob > 0.5)
                margin = abs((2.0 * prob) - 1.0)

                logit_tensor = torch.tensor([logit], device=model_device, dtype=torch.float32)
                label_tensor = labels[local_idx: local_idx + 1].float()
                loss = criterion(logit_tensor, label_tensor)

                total_loss += float(loss.item())
                total_correct += int(pred == label_value)
                all_preds.append(pred)
                all_labels.append(label_value)
                n_samples += 1

                if backtracking_result is not None:
                    if bool(backtracking_result.triggered):
                        backtracking_claims_triggered += 1
                    if promoted_ids:
                        backtracking_claims_promoted += 1
                        backtracking_promotions_total += len(promoted_ids)

                    for action in backtracking_result.actions:
                        backtracking_rounds_total += 1
                        if bool(action.do_no_harm_reverted):
                            backtracking_rounds_reverted += 1
                        else:
                            backtracking_rounds_accepted += 1
                            if str(action.selection_mode) == "top1_fallback":
                                backtracking_rounds_top1_fallback += 1
                        ranked_count = len(action.ranked_candidates)
                        logged_count = min(ranked_count, candidate_log_top_n)
                        recovery_candidates_ranked_total += ranked_count

                        ranked_connects_components = 0
                        ranked_rel_min: float | None = None
                        ranked_rel_max: float | None = None
                        for candidate in action.ranked_candidates:
                            if bool(candidate.connects_components):
                                ranked_connects_components += 1
                            rel_value = float(candidate.rel)
                            if ranked_rel_min is None or rel_value < ranked_rel_min:
                                ranked_rel_min = rel_value
                            if ranked_rel_max is None or rel_value > ranked_rel_max:
                                ranked_rel_max = rel_value

                        reason_tokens: list[str] = []
                        if bool(action.decision.low_confidence):
                            reason_tokens.append("low_confidence")
                        if bool(action.decision.low_connectivity):
                            reason_tokens.append("low_connectivity")
                        if bool(action.decision.evidence_hunger):
                            reason_tokens.append("evidence_hunger")

                        before_sup_map = _prediction_edge_mass_map(action.prediction_before, "edge_mass_sup_by_id")
                        before_ref_map = _prediction_edge_mass_map(action.prediction_before, "edge_mass_ref_by_id")
                        after_sup_map = _prediction_edge_mass_map(action.prediction_after, "edge_mass_sup_by_id")
                        after_ref_map = _prediction_edge_mass_map(action.prediction_after, "edge_mass_ref_by_id")
                        selected_ids_list = list(action.selected_evidence_ids)
                        promoted_edge_influence: list[dict[str, float | str]] = []
                        promoted_sup_before = 0.0
                        promoted_sup_after = 0.0
                        promoted_ref_before = 0.0
                        promoted_ref_after = 0.0
                        for evidence_id in selected_ids_list:
                            sup_before = float(before_sup_map.get(evidence_id, 0.0))
                            sup_after = float(after_sup_map.get(evidence_id, 0.0))
                            ref_before = float(before_ref_map.get(evidence_id, 0.0))
                            ref_after = float(after_ref_map.get(evidence_id, 0.0))
                            promoted_sup_before += sup_before
                            promoted_sup_after += sup_after
                            promoted_ref_before += ref_before
                            promoted_ref_after += ref_after
                            promoted_edge_influence.append(
                                {
                                    "evidence_id": str(evidence_id),
                                    "edge_weight_sup_before": sup_before,
                                    "edge_weight_sup_after": sup_after,
                                    "edge_weight_ref_before": ref_before,
                                    "edge_weight_ref_after": ref_after,
                                }
                            )
                        stream_mass_sup_before = float(action.prediction_before.get("stream_mass_sup", 0.0))
                        stream_mass_sup_after = float(action.prediction_after.get("stream_mass_sup", 0.0))
                        stream_mass_ref_before = float(action.prediction_before.get("stream_mass_ref", 0.0))
                        stream_mass_ref_after = float(action.prediction_after.get("stream_mass_ref", 0.0))
                        promoted_share_sup_before = (
                            promoted_sup_before / stream_mass_sup_before
                            if stream_mass_sup_before > 0.0 else 0.0
                        )
                        promoted_share_sup_after = (
                            promoted_sup_after / stream_mass_sup_after
                            if stream_mass_sup_after > 0.0 else 0.0
                        )
                        promoted_share_ref_before = (
                            promoted_ref_before / stream_mass_ref_before
                            if stream_mass_ref_before > 0.0 else 0.0
                        )
                        promoted_share_ref_after = (
                            promoted_ref_after / stream_mass_ref_after
                            if stream_mass_ref_after > 0.0 else 0.0
                        )

                        action_row = {
                            "claim_id": claim_id,
                            "triggered": bool(backtracking_result.triggered),
                            "round_index": int(action.round_index),
                            "selected_evidence_ids": selected_ids_list,
                            "promoted_s_to_a": selected_ids_list,
                            "margin_before": float(action.margin_before),
                            "tentative_margin": float(action.tentative_margin),
                            "tentative_margin_delta": float(action.tentative_margin - action.margin_before),
                            "margin_after": float(action.margin_after),
                            "margin_delta": float(action.margin_after - action.margin_before),
                            "final_margin": float(action.margin_after),
                            "final_margin_delta": float(action.margin_after - action.margin_before),
                            "tentative_logit": float(_prediction_logit_value(action.tentative_prediction)),
                            "final_logit": float(_prediction_logit_value(action.prediction_after)),
                            "low_confidence": bool(action.decision.low_confidence),
                            "low_connectivity": bool(action.decision.low_connectivity),
                            "evidence_hunger": bool(action.decision.evidence_hunger),
                            "active_high_rel_count": int(action.decision.active_high_rel_count),
                            "active_high_rel_required": int(action.decision.active_high_rel_required),
                            "hunger_rel_threshold": float(action.decision.hunger_rel_threshold),
                            "hunger_mode": str(action.decision.hunger_mode),
                            "recovery_mode": str(action.recovery_mode),
                            "selection_mode": str(action.selection_mode),
                            "do_no_harm_reverted": bool(action.do_no_harm_reverted),
                            "reverted": bool(action.do_no_harm_reverted),
                            "directional_delta": float(config.backtracking_directional_delta),
                            "challenge_mode": bool(config.backtracking_challenge_mode),
                            "promotion_gamma": float(config.backtracking_promotion_gamma),
                            "flip_conf_min": float(config.backtracking_flip_conf_min),
                            "flip_abslogit_eps": float(config.backtracking_flip_abslogit_eps),
                            "stream_mass_sup_before": stream_mass_sup_before,
                            "stream_mass_sup_after": stream_mass_sup_after,
                            "stream_mass_ref_before": stream_mass_ref_before,
                            "stream_mass_ref_after": stream_mass_ref_after,
                            "promoted_mass_sup_before": promoted_sup_before,
                            "promoted_mass_sup_after": promoted_sup_after,
                            "promoted_mass_ref_before": promoted_ref_before,
                            "promoted_mass_ref_after": promoted_ref_after,
                            "promoted_share_sup_before": promoted_share_sup_before,
                            "promoted_share_sup_after": promoted_share_sup_after,
                            "promoted_share_ref_before": promoted_share_ref_before,
                            "promoted_share_ref_after": promoted_share_ref_after,
                            "promoted_edge_influence": promoted_edge_influence,
                            "reason": "+".join(reason_tokens) if reason_tokens else "trigger_policy",
                            "ranked_candidates_total": int(ranked_count),
                            "ranked_candidates_logged": int(logged_count),
                            "logged_count": int(logged_count),
                            "ranked_candidates_connects_components_total": int(ranked_connects_components),
                            "bridge_count": int(ranked_connects_components),
                            "ranked_candidates_rel_min": ranked_rel_min,
                            "rel_min": ranked_rel_min,
                            "ranked_candidates_rel_max": ranked_rel_max,
                            "rel_max": ranked_rel_max,
                        }
                        pred_before, prob_before = _decode_binary_prediction(action.prediction_before)
                        pred_tentative, prob_tentative = _decode_binary_prediction(action.tentative_prediction)
                        pred_after, prob_after = _decode_binary_prediction(action.prediction_after)
                        action_row.update(
                            {
                                "label": int(label_value),
                                "pred_before": int(pred_before),
                                "pred_after": int(pred_after),
                                "prob_before": float(prob_before),
                                "prob_after": float(prob_after),
                                "tentative_pred": int(pred_tentative),
                                "tentative_prob": float(prob_tentative),
                                "final_pred": int(pred_after),
                                "final_prob": float(prob_after),
                                "tentative_correct": bool(pred_tentative == int(label_value)),
                                "final_correct": bool(pred_after == int(label_value)),
                                "correct_before": bool(pred_before == int(label_value)),
                                "correct_after": bool(pred_after == int(label_value)),
                                "flipped_label_tentative": bool(pred_before != pred_tentative),
                                "flipped_to_correct_tentative": bool(
                                    (pred_before != pred_tentative)
                                    and (pred_tentative == int(label_value))
                                    and (pred_before != int(label_value))
                                ),
                                "flipped_label": bool(pred_before != pred_after),
                                "flipped_to_correct": bool(
                                    (pred_before != pred_after)
                                    and (pred_after == int(label_value))
                                    and (pred_before != int(label_value))
                                ),
                                "tentative_flip_to_correct_but_reverted": bool(
                                    (pred_before != pred_tentative)
                                    and (pred_tentative == int(label_value))
                                    and (pred_before != int(label_value))
                                    and bool(action.do_no_harm_reverted)
                                ),
                            }
                        )
                        recovery_actions_fp.write(json.dumps(action_row) + "\n")

                        selected_set = set(action.selected_evidence_ids)
                        for rank_idx, candidate in enumerate(action.ranked_candidates[:logged_count], start=1):
                            candidate_row = {
                                "claim_id": claim_id,
                                "round_index": int(action.round_index),
                                "rank": int(rank_idx),
                                "evidence_id": str(candidate.evidence_id),
                                "connects_components": bool(candidate.connects_components),
                                "bridge_bonus": float(candidate.bridge_bonus),
                                "rel": float(candidate.rel),
                                "salience_x_pv": float(candidate.salience_x_pv),
                                "p_ent": float(candidate.p_ent),
                                "p_con": float(candidate.p_con),
                                "pv_confidence": float(candidate.pv_confidence),
                                "semantic_score": float(candidate.semantic_score),
                                "bridge_semantic_score": float(candidate.bridge_semantic_score),
                                "directional_support_score": float(candidate.p_ent - candidate.p_con),
                                "directional_refute_score": float(candidate.p_con - candidate.p_ent),
                                "selected": bool(candidate.evidence_id in selected_set),
                                "ranked_candidates_total": int(ranked_count),
                                "logged_count": int(logged_count),
                                "bridge_count": int(ranked_connects_components),
                                "rel_min": ranked_rel_min,
                                "rel_max": ranked_rel_max,
                            }
                            recovery_candidates_fp.write(json.dumps(candidate_row) + "\n")
                            recovery_candidates_logged_total += 1

                predictions_row = {
                    "claim_id": claim_id,
                    "label": int(label_value),
                    "pred": int(pred),
                    "logit": float(logit),
                    "confidence": float(prob),
                    "margin": float(margin),
                    "claim_type": metadata,
                    "backtracking_triggered": bool(backtracking_result.triggered) if backtracking_result is not None else False,
                    "promoted_evidence_ids": list(promoted_ids),
                }
                if backtracking_result is not None:
                    pred_before_claim, prob_before_claim = _decode_binary_prediction(backtracking_result.initial_prediction)
                    after_sup_map = _prediction_edge_mass_map(backtracking_result.final_prediction, "edge_mass_sup_by_id")
                    after_ref_map = _prediction_edge_mass_map(backtracking_result.final_prediction, "edge_mass_ref_by_id")
                    promoted_sup_after = float(sum(after_sup_map.get(eid, 0.0) for eid in promoted_ids))
                    promoted_ref_after = float(sum(after_ref_map.get(eid, 0.0) for eid in promoted_ids))
                    stream_sup_before = float(backtracking_result.initial_prediction.get("stream_mass_sup", 0.0))
                    stream_ref_before = float(backtracking_result.initial_prediction.get("stream_mass_ref", 0.0))
                    stream_sup_after = float(backtracking_result.final_prediction.get("stream_mass_sup", 0.0))
                    stream_ref_after = float(backtracking_result.final_prediction.get("stream_mass_ref", 0.0))
                    predictions_row.update(
                        {
                            "pred_before_backtracking": int(pred_before_claim),
                            "prob_before_backtracking": float(prob_before_claim),
                            "correct_before_backtracking": bool(pred_before_claim == int(label_value)),
                            "correct_after_backtracking": bool(pred == int(label_value)),
                            "flipped_by_backtracking": bool(pred_before_claim != pred),
                            "flipped_to_correct": bool(
                                (pred_before_claim != pred)
                                and (pred == int(label_value))
                                and (pred_before_claim != int(label_value))
                            ),
                            "stream_mass_sup_before": stream_sup_before,
                            "stream_mass_ref_before": stream_ref_before,
                            "stream_mass_sup_after": stream_sup_after,
                            "stream_mass_ref_after": stream_ref_after,
                            "promoted_mass_sup_after": promoted_sup_after,
                            "promoted_mass_ref_after": promoted_ref_after,
                            "promoted_share_sup_after": (
                                promoted_sup_after / stream_sup_after if stream_sup_after > 0.0 else 0.0
                            ),
                            "promoted_share_ref_after": (
                                promoted_ref_after / stream_ref_after if stream_ref_after > 0.0 else 0.0
                            ),
                        }
                    )
                predictions_fp.write(json.dumps(predictions_row) + "\n")

            claim_offset += batch_size

    if n_samples <= 0:
        raise ValueError("Evaluation received an empty test loader.")

    overall_accuracy = total_correct / float(n_samples)
    overall_loss = total_loss / float(n_samples)
    precision, recall, f1, _ = _safe_prf(all_labels, all_preds, average="binary")
    metrics_dict = _evaluate_factkg_per_type(
        test_types=claim_types[:n_samples],
        all_preds=all_preds,
        all_labels=all_labels,
    )
    metrics_dict["overall"] = {
        "accuracy": float(overall_accuracy),
        "loss": float(overall_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    backtracking_summary = {
        "enabled": bool(controller is not None),
        "attempted_claims": int(backtracking_attempted_claims),
        "claims_with_s_pool": int(claims_with_s_pool),
        "claims_triggered": int(backtracking_claims_triggered),
        "claims_promoted": int(backtracking_claims_promoted),
        "promotions_total": int(backtracking_promotions_total),
        "rounds_total": int(backtracking_rounds_total),
        "rounds_reverted": int(backtracking_rounds_reverted),
        "rounds_top1_fallback": int(backtracking_rounds_top1_fallback),
        "rounds_accepted": int(backtracking_rounds_accepted),
        "promotion_gamma": float(config.backtracking_promotion_gamma),
        "directional_delta": float(config.backtracking_directional_delta),
        "challenge_mode": bool(config.backtracking_challenge_mode),
        "flip_conf_min": float(config.backtracking_flip_conf_min),
        "flip_abslogit_eps": float(config.backtracking_flip_abslogit_eps),
        "candidate_log_top_n": int(candidate_log_top_n),
        "candidates_ranked_total": int(recovery_candidates_ranked_total),
        "candidates_logged_total": int(recovery_candidates_logged_total),
        "recovery_actions_path": str(recovery_actions_path),
        "recovery_candidates_path": str(recovery_candidates_path),
    }
    return metrics_dict, backtracking_summary


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
    if config.edge_message_dim <= 0:
        raise ValueError("Argument `edge_message_dim` must be positive.")
    if config.edge_encoder_dropout < 0.0 or config.edge_encoder_dropout > 1.0:
        raise ValueError("Argument `edge_encoder_dropout` must be in [0, 1].")
    if config.backtracking_margin_threshold < 0.0:
        raise ValueError("Argument `backtracking_margin_threshold` must be >= 0.")
    if config.backtracking_min_a < 1:
        raise ValueError("Argument `backtracking_min_a` must be >= 1.")
    if config.backtracking_rel_threshold < 0.0:
        raise ValueError("Argument `backtracking_rel_threshold` must be >= 0.")
    if config.backtracking_promotion_gamma < 0.0:
        raise ValueError("Argument `backtracking_promotion_gamma` must be >= 0.")
    if config.backtracking_directional_delta < 0.0:
        raise ValueError("Argument `backtracking_directional_delta` must be >= 0.")
    if str(config.backtracking_hunger_mode).strip().lower() not in {"absolute", "percentile"}:
        raise ValueError("Argument `backtracking_hunger_mode` must be `absolute` or `percentile`.")
    if config.backtracking_hunger_percentile <= 0.0 or config.backtracking_hunger_percentile > 1.0:
        raise ValueError("Argument `backtracking_hunger_percentile` must be in (0, 1].")
    if config.backtracking_max_rounds < 1 or config.backtracking_max_rounds > 2:
        raise ValueError("Argument `backtracking_max_rounds` must be in [1, 2].")
    if config.backtracking_top_k < 1 or config.backtracking_top_k > 3:
        raise ValueError("Argument `backtracking_top_k` must be in [1, 3].")
    if config.backtracking_do_no_harm_eps < 0.0:
        raise ValueError("Argument `backtracking_do_no_harm_eps` must be >= 0.")
    if config.backtracking_flip_conf_min < 0.0:
        raise ValueError("Argument `backtracking_flip_conf_min` must be >= 0.")
    if config.backtracking_flip_abslogit_eps < 0.0:
        raise ValueError("Argument `backtracking_flip_abslogit_eps` must be >= 0.")
    if config.backtracking_candidate_log_top_n < 0:
        raise ValueError("Argument `backtracking_candidate_log_top_n` must be >= 0.")
    if str(config.model_mode).strip() not in {"pv_qagnn", "claim_only"}:
        raise ValueError("Argument `model_mode` must be one of: pv_qagnn, claim_only.")
    if str(config.encoder_tune).strip() not in {"none", "lora", "unfreeze_lastN"}:
        raise ValueError("Argument `encoder_tune` must be one of: none, lora, unfreeze_lastN.")
    if int(config.unfreeze_last_n) < 1:
        raise ValueError("Argument `unfreeze_last_n` must be >= 1.")
    if int(config.lora_r) < 1:
        raise ValueError("Argument `lora_r` must be >= 1.")
    if float(config.lora_alpha) <= 0.0:
        raise ValueError("Argument `lora_alpha` must be > 0.")
    if float(config.lora_dropout) < 0.0 or float(config.lora_dropout) > 1.0:
        raise ValueError("Argument `lora_dropout` must be in [0, 1].")
    if str(config.model_mode).strip() == "claim_only" and bool(config.enable_component5):
        raise ValueError("Argument `enable_component5` is not supported with `model_mode=claim_only`.")

    seed_everything(config.seed)
    if config.deterministic_mode:
        _enable_deterministic_mode(config.seed)

    run_id = config.run_id or f"component3_t33_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_yaml(run_dir / "config.yaml", config_to_dict(config))

    coverage_stats: dict[str, Any] = {}
    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader, coverage_stats = _build_default_loaders(config)
        _validate_component1_pair_coverage(config, coverage_stats)

    if model is None:
        if str(config.model_mode).strip() == "pv_qagnn":
            inferred_edge_feature_dim = _infer_edge_feature_dim_from_loader(train_loader)
            model = _build_model(config, edge_feature_dim=inferred_edge_feature_dim)
            model = _apply_lora_adapter(model, config)
        else:
            model = _build_claim_only_model(config)
            model = _apply_lora_adapter(model, config)

    claim_only_mode = str(config.model_mode).strip() == "claim_only"
    if claim_only_mode:
        wrapped_model = model
    else:
        wrapped_model = model if isinstance(model, MultiTaskPVModel) else MultiTaskPVModel(
            model,
            evidence_head_in_dim=config.gnn_out_features,
        )

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
        _apply_encoder_tune_trainable_parameters(
            wrapped_model,
            encoder_tune=str(config.encoder_tune).strip(),
            unfreeze_last_n=int(config.unfreeze_last_n),
        )
        _set_optimizer_lr(optimizer, stage.learning_rate)

        if claim_only_mode:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif config.enable_component5:
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

        train_acc_series = history.get("train_class_accuracy") or []
        train_loss_series = history.get("train_class_loss") or []
        train_acc_last_pct = float(train_acc_series[-1]) if train_acc_series else 0.0
        train_acc_best_pct = float(max(train_acc_series)) if train_acc_series else 0.0
        stage_last_total = float(train_loss_series[-1]) if train_loss_series else float(getattr(criterion, "last_total", 0.0))

        stage_results.append(
            {
                "stage": stage.name,
                "epochs": stage.epochs,
                "learning_rate": stage.learning_rate,
                "unfreeze_pooler": stage.unfreeze_pooler,
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss"),
                "best_val_accuracy": history.get("best_val_accuracy"),
                "train_accuracy_last": float(train_acc_last_pct / 100.0),
                "train_accuracy_best": float(train_acc_best_pct / 100.0),
                "train_accuracy_last_pct": float(train_acc_last_pct),
                "train_accuracy_best_pct": float(train_acc_best_pct),
                "last_loss_bce": float(getattr(criterion, "last_bce", 0.0)),
                "last_loss_evidence": float(getattr(criterion, "last_evidence", 0.0)),
                "last_loss_starvation": float(getattr(criterion, "last_starvation", 0.0)),
                "last_loss_counter": float(getattr(criterion, "last_counter", 0.0)),
                "last_loss_recovery": float(getattr(criterion, "last_recovery", 0.0)),
                "last_loss_total": float(stage_last_total),
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
        "runtime_flags": {
            "used_s_pool": bool(
                (coverage_stats.get("test", {}) or {}).get("pool_summary", {}).get("s_pool_retained_total", 0) > 0
            ),
            "used_backtracking": False,
            "no_collapse_gate_enforced": bool(_should_enforce_no_collapse_gate(config)),
            "backtracking_enabled": bool(config.enable_backtracking),
            "backtracking_triggered": False,
            "backtracking_attempted": False,
            "used_router": False,
            "used_component5": bool(config.enable_component5),
            "model_mode": str(config.model_mode).strip(),
            "encoder_tune": str(config.encoder_tune).strip(),
        },
    }

    if config.evaluate_after_training:
        claim_only_criterion = torch.nn.BCEWithLogitsLoss()
        dataset_name = str(config.dataset_name).strip().lower()
        factkg_dataset = getattr(test_loader, "dataset", None)
        can_run_factkg_backtracking = (
            dataset_name == "factkg"
            and factkg_dataset is not None
            and _extract_dataset_attribute(factkg_dataset, "claim_types") is not None
        )
        if can_run_factkg_backtracking:
            metrics, backtracking_summary = _evaluate_factkg_with_backtracking(
                config=config,
                model=wrapped_model,
                test_loader=test_loader,
                criterion=claim_only_criterion,
                run_dir=run_dir,
            )
            results["backtracking_summary"] = backtracking_summary
            backtracking_triggered = bool(backtracking_summary.get("claims_triggered", 0) > 0)
            backtracking_attempted = bool(backtracking_summary.get("attempted_claims", 0) > 0)
            results["runtime_flags"]["used_backtracking"] = backtracking_triggered
            results["runtime_flags"]["backtracking_triggered"] = backtracking_triggered
            results["runtime_flags"]["backtracking_attempted"] = backtracking_attempted
        else:
            metrics = evaluate_on_test_set(
                qa_gnn=True,
                model=wrapped_model,
                test_loader=test_loader,
                criterion=claim_only_criterion,
                dataset_name=config.dataset_name,
            )
        results["test_metrics"] = metrics
        if dataset_name == "factkg":
            gate = _evaluate_no_collapse_gate(
                metrics,
                stage_results,
                precision_accuracy_gap_min=config.no_collapse_precision_accuracy_gap_min,
            )
            if _is_subset_run(config):
                gate = dict(gate)
                gate["enforcement_skipped_for_subset_run"] = True
                gate["subset_context"] = {
                    "train_subset_size": int(config.train_subset_size),
                    "val_subset_size": int(config.val_subset_size),
                }
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
        and _should_enforce_no_collapse_gate(config)
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
    parser.add_argument("--model-mode", type=str, choices=["pv_qagnn", "claim_only"], default="pv_qagnn")
    parser.add_argument("--encoder_tune", type=str, choices=["none", "lora", "unfreeze_lastN"], default="none")
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
    parser.add_argument("--factkg-require-pv-metadata", dest="factkg_require_pv_metadata", action="store_true")
    parser.add_argument("--factkg-no-require-pv-metadata", dest="factkg_require_pv_metadata", action="store_false")
    parser.set_defaults(factkg_require_pv_metadata=True)
    parser.add_argument("--factkg-precompute-batch-size", type=int, default=32)
    parser.add_argument("--factkg-include-s-pool", dest="factkg_include_s_pool", action="store_true")
    parser.add_argument("--factkg-no-include-s-pool", dest="factkg_include_s_pool", action="store_false")
    parser.set_defaults(factkg_include_s_pool=True)
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--no-collapse-precision-accuracy-gap-min", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--train-subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--subset-sampling", type=str, choices=["stratified", "prefix"], default="stratified")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--edge-message-dim", type=int, default=64)
    parser.add_argument("--edge-encoder-dropout", type=float, default=0.1)
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
    parser.add_argument("--enable-backtracking", dest="enable_backtracking", action="store_true")
    parser.add_argument("--disable-backtracking", dest="enable_backtracking", action="store_false")
    parser.set_defaults(enable_backtracking=True)
    parser.add_argument("--backtracking-margin-threshold", type=float, default=0.15)
    parser.add_argument("--backtracking-min-a", type=int, default=5)
    parser.add_argument("--backtracking-rel-threshold", type=float, default=0.3)
    parser.add_argument("--backtracking-promotion-gamma", type=float, default=0.0)
    parser.add_argument("--backtracking-directional-delta", type=float, default=0.0)
    parser.add_argument("--backtracking-challenge-mode", dest="backtracking_challenge_mode", action="store_true")
    parser.add_argument("--backtracking-no-challenge-mode", dest="backtracking_challenge_mode", action="store_false")
    parser.set_defaults(backtracking_challenge_mode=False)
    parser.add_argument("--backtracking-hunger-mode", type=str, choices=["absolute", "percentile"], default="percentile")
    parser.add_argument("--backtracking-hunger-percentile", type=float, default=0.9)
    parser.add_argument("--backtracking-max-rounds", type=int, default=2)
    parser.add_argument("--backtracking-top-k", type=int, default=3)
    parser.add_argument("--backtracking-do-no-harm-eps", type=float, default=0.002)
    parser.add_argument("--backtracking-flip-conf-min", type=float, default=0.1)
    parser.add_argument("--backtracking-flip-abslogit-eps", type=float, default=0.002)
    parser.add_argument("--backtracking-candidate-log-top-n", type=int, default=25)
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
        model_mode=args.model_mode,
        encoder_tune=args.encoder_tune,
        unfreeze_last_n=args.unfreeze_last_n,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
        factkg_require_pv_metadata=args.factkg_require_pv_metadata,
        factkg_precompute_batch_size=args.factkg_precompute_batch_size,
        factkg_include_s_pool=args.factkg_include_s_pool,
        deterministic_mode=args.deterministic_mode,
        no_collapse_precision_accuracy_gap_min=args.no_collapse_precision_accuracy_gap_min,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        subset_sampling=args.subset_sampling,
        seed=args.seed,
        edge_message_dim=args.edge_message_dim,
        edge_encoder_dropout=args.edge_encoder_dropout,
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
        enable_backtracking=args.enable_backtracking,
        backtracking_margin_threshold=args.backtracking_margin_threshold,
        backtracking_min_a=args.backtracking_min_a,
        backtracking_rel_threshold=args.backtracking_rel_threshold,
        backtracking_promotion_gamma=args.backtracking_promotion_gamma,
        backtracking_directional_delta=args.backtracking_directional_delta,
        backtracking_challenge_mode=args.backtracking_challenge_mode,
        backtracking_hunger_mode=args.backtracking_hunger_mode,
        backtracking_hunger_percentile=args.backtracking_hunger_percentile,
        backtracking_max_rounds=args.backtracking_max_rounds,
        backtracking_top_k=args.backtracking_top_k,
        backtracking_do_no_harm_eps=args.backtracking_do_no_harm_eps,
        backtracking_flip_conf_min=args.backtracking_flip_conf_min,
        backtracking_flip_abslogit_eps=args.backtracking_flip_abslogit_eps,
        backtracking_candidate_log_top_n=args.backtracking_candidate_log_top_n,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enforce_no_collapse_gate=args.enforce_no_collapse_gate,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    run_dir, run_results = run_training(cfg)
    print(f"Run directory: {run_dir}")
    print(json.dumps(run_results, indent=2))
