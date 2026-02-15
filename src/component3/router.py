"""Component 6 router: claim-type classifier, difficulty scoring, and model routing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence


CLAIM_TYPES = (
    "existence",
    "substitution",
    "multi hop",
    "multi claim",
    "negation",
    "single hop",
)
EASY_CLAIM_TYPES = frozenset({"existence", "substitution", "single hop"})
HARD_CLAIM_TYPES = frozenset({"multi hop", "multi claim", "negation"})

MODEL_BERT_BASELINE = "bert_baseline"
MODEL_PV_QAGNN = "pv_qagnn_backtracking"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_CLAIM_TYPE_ALIASES = {
    "one hop": "single hop",
    "one-hop": "single hop",
    "multi-hop": "multi hop",
    "multihop": "multi hop",
    "conjunction": "multi claim",
}


def _normalize_claim_type(claim_type: str) -> str:
    raw = str(claim_type or "").strip().lower()
    raw = _CLAIM_TYPE_ALIASES.get(raw, raw)
    if raw not in CLAIM_TYPES:
        raise ValueError(f"Unsupported claim type: {claim_type!r}. Expected one of {list(CLAIM_TYPES)}.")
    return raw


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


@dataclass(frozen=True)
class RouterTrainingExample:
    claim_text: str
    claim_type: str
    claim_id: str = ""


@dataclass(frozen=True)
class RouterClassifierConfig:
    """Configuration for T6.1 claim-type router training."""

    backend: str = "auto"
    model_name: str = "bert-base-uncased"
    max_seq_len: int = 256
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 2.0e-5
    warmup_steps: int = 0
    min_dev_accuracy: float = 0.80
    enforce_min_accuracy: bool = True
    allow_fallback_to_naive_bayes: bool = True


@dataclass(frozen=True)
class RouterTrainingReport:
    backend: str
    dev_accuracy: float
    target_accuracy: float
    meets_target: bool
    n_train: int
    n_dev: int


class ClaimTypeClassifier(Protocol):
    """Protocol for claim-type classifier backends."""

    backend_name: str

    def fit(self, train_examples: Sequence[RouterTrainingExample]) -> None:
        ...

    def predict(self, claim_text: str) -> str:
        ...

    def predict_proba(self, claim_text: str) -> dict[str, float]:
        ...


class NaiveBayesClaimTypeClassifier:
    """Dependency-light classifier used as a deterministic fallback in constrained runtimes."""

    backend_name = "naive_bayes"

    def __init__(self) -> None:
        self._class_doc_counts: Counter[str] = Counter()
        self._class_token_totals: Counter[str] = Counter()
        self._class_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._vocab: set[str] = set()
        self._n_examples = 0

    def fit(self, train_examples: Sequence[RouterTrainingExample]) -> None:
        self._class_doc_counts.clear()
        self._class_token_totals.clear()
        self._class_token_counts.clear()
        self._vocab.clear()
        self._n_examples = 0

        if not train_examples:
            raise ValueError("`train_examples` must not be empty.")

        for row in train_examples:
            claim_type = _normalize_claim_type(row.claim_type)
            tokens = _tokenize(row.claim_text)
            self._class_doc_counts[claim_type] += 1
            self._class_token_totals[claim_type] += len(tokens)
            self._class_token_counts[claim_type].update(tokens)
            self._vocab.update(tokens)
            self._n_examples += 1

        for claim_type in CLAIM_TYPES:
            self._class_doc_counts.setdefault(claim_type, 0)
            self._class_token_totals.setdefault(claim_type, 0)
            self._class_token_counts.setdefault(claim_type, Counter())

    def _log_probability_by_class(self, claim_text: str) -> dict[str, float]:
        if self._n_examples <= 0:
            raise RuntimeError("Classifier has not been fitted.")

        tokens = _tokenize(claim_text)
        vocab_size = max(1, len(self._vocab))
        denom_docs = self._n_examples + len(CLAIM_TYPES)
        log_prob: dict[str, float] = {}
        for claim_type in CLAIM_TYPES:
            prior = (self._class_doc_counts[claim_type] + 1.0) / float(denom_docs)
            value = math.log(prior)
            denom_tokens = self._class_token_totals[claim_type] + vocab_size
            if denom_tokens <= 0:
                denom_tokens = vocab_size
            class_counts = self._class_token_counts[claim_type]
            for token in tokens:
                value += math.log((class_counts.get(token, 0) + 1.0) / float(denom_tokens))
            log_prob[claim_type] = value
        return log_prob

    def predict_proba(self, claim_text: str) -> dict[str, float]:
        log_prob = self._log_probability_by_class(claim_text)
        max_log_prob = max(log_prob.values())
        exp_probs = {key: math.exp(value - max_log_prob) for key, value in log_prob.items()}
        total = sum(exp_probs.values()) or 1.0
        return {key: float(value / total) for key, value in exp_probs.items()}

    def predict(self, claim_text: str) -> str:
        probs = self.predict_proba(claim_text)
        return max(probs.items(), key=lambda item: item[1])[0]


class BertClaimTypeClassifier:
    """
    Optional BERT backend for T6.1.

    This backend is only used if torch + transformers are available. It keeps
    batch size and max sequence length aligned with the 8GB constraints.
    """

    backend_name = "bert_finetune"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_len: int = 256,
        batch_size: int = 8,
        epochs: int = 1,
        learning_rate: float = 2.0e-5,
        warmup_steps: int = 0,
    ) -> None:
        self.model_name = model_name
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.warmup_steps = int(warmup_steps)
        self._device = None
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(CLAIM_TYPES),
            )
        if self._device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def fit(self, train_examples: Sequence[RouterTrainingExample]) -> None:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
        from transformers import get_linear_schedule_with_warmup  # type: ignore

        if not train_examples:
            raise ValueError("`train_examples` must not be empty.")

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        texts = [row.claim_text for row in train_examples]
        labels = [CLAIM_TYPES.index(_normalize_claim_type(row.claim_type)) for row in train_examples]
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        dataset = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)
        total_steps = max(1, len(loader) * max(1, self.epochs))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(self.warmup_steps, total_steps - 1),
            num_training_steps=total_steps,
        )

        self._model.train()
        for _ in range(max(1, self.epochs)):
            for batch in loader:
                input_ids, attention_mask, labels_tensor = [item.to(self._device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor,
                )
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()

    def predict_proba(self, claim_text: str) -> dict[str, float]:
        import torch  # type: ignore

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        encoded = self._tokenizer(
            [claim_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().view(-1).tolist()
        return {claim_type: float(probs[idx]) for idx, claim_type in enumerate(CLAIM_TYPES)}

    def predict(self, claim_text: str) -> str:
        probs = self.predict_proba(claim_text)
        return max(probs.items(), key=lambda item: item[1])[0]


def _backend_dependencies_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def evaluate_claim_type_accuracy(
    classifier: ClaimTypeClassifier,
    dev_examples: Sequence[RouterTrainingExample],
) -> float:
    if not dev_examples:
        raise ValueError("`dev_examples` must not be empty.")
    matches = 0
    for row in dev_examples:
        pred = _normalize_claim_type(classifier.predict(row.claim_text))
        truth = _normalize_claim_type(row.claim_type)
        matches += int(pred == truth)
    return float(matches / len(dev_examples))


def train_router_classifier(
    train_examples: Sequence[RouterTrainingExample],
    dev_examples: Sequence[RouterTrainingExample],
    *,
    config: RouterClassifierConfig | None = None,
) -> tuple[ClaimTypeClassifier, RouterTrainingReport]:
    """
    Train T6.1 router classifier.

    Backend selection:
    - `bert`: BERT fine-tuning path (requires torch + transformers)
    - `naive_bayes`: deterministic fallback
    - `auto`: use BERT if available, else fallback
    """

    config = config or RouterClassifierConfig()
    backend = config.backend.lower().strip()
    if backend not in {"auto", "bert", "naive_bayes"}:
        raise ValueError("`backend` must be one of {'auto','bert','naive_bayes'}.")
    if config.batch_size < 1 or config.batch_size > 8:
        raise ValueError("For Component 6 on 8GB GPU, `batch_size` must be in [1, 8].")
    if config.max_seq_len < 1 or config.max_seq_len > 256:
        raise ValueError("For Component 6, `max_seq_len` must be in [1, 256].")

    if backend == "auto":
        backend = "bert" if _backend_dependencies_available() else "naive_bayes"

    if backend == "bert":
        if not _backend_dependencies_available():
            if not config.allow_fallback_to_naive_bayes:
                raise RuntimeError("BERT backend requested but torch/transformers are unavailable.")
            backend = "naive_bayes"
        else:
            classifier: ClaimTypeClassifier = BertClaimTypeClassifier(
                model_name=config.model_name,
                max_seq_len=config.max_seq_len,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
            )
            classifier.fit(train_examples)
            dev_acc = evaluate_claim_type_accuracy(classifier, dev_examples)
            report = RouterTrainingReport(
                backend=classifier.backend_name,
                dev_accuracy=dev_acc,
                target_accuracy=config.min_dev_accuracy,
                meets_target=(dev_acc >= config.min_dev_accuracy),
                n_train=len(train_examples),
                n_dev=len(dev_examples),
            )
            if config.enforce_min_accuracy and not report.meets_target:
                raise RuntimeError(
                    f"T6.1 target miss: dev_accuracy={report.dev_accuracy:.4f} < {report.target_accuracy:.4f}"
                )
            return classifier, report

    classifier = NaiveBayesClaimTypeClassifier()
    classifier.fit(train_examples)
    dev_acc = evaluate_claim_type_accuracy(classifier, dev_examples)
    report = RouterTrainingReport(
        backend=classifier.backend_name,
        dev_accuracy=dev_acc,
        target_accuracy=config.min_dev_accuracy,
        meets_target=(dev_acc >= config.min_dev_accuracy),
        n_train=len(train_examples),
        n_dev=len(dev_examples),
    )
    if config.enforce_min_accuracy and not report.meets_target:
        raise RuntimeError(f"T6.1 target miss: dev_accuracy={dev_acc:.4f} < {config.min_dev_accuracy:.4f}")
    return classifier, report


@dataclass(frozen=True)
class DifficultyFeatures:
    """T6.2 feature set for easy-vs-hard difficulty prediction."""

    claim_type: str
    prediction_margin: float
    graph_connectivity: float
    evidence_count: int


@dataclass(frozen=True)
class DifficultyConfig:
    margin_hard_threshold: float = 0.15
    connectivity_hard_threshold: float = 0.85
    evidence_low_threshold: int = 3
    hard_score_threshold: float = 0.55
    type_weight: float = 0.40
    margin_weight: float = 0.30
    connectivity_weight: float = 0.20
    evidence_weight: float = 0.10


@dataclass(frozen=True)
class DifficultyScore:
    claim_type: str
    hard_score: float
    is_hard: bool
    type_signal: float
    margin_signal: float
    connectivity_signal: float
    evidence_signal: float


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_claim_difficulty(
    features: DifficultyFeatures,
    *,
    config: DifficultyConfig | None = None,
) -> DifficultyScore:
    """Score claim difficulty using claim type + margin + connectivity + evidence count."""

    config = config or DifficultyConfig()
    claim_type = _normalize_claim_type(features.claim_type)
    type_signal = 1.0 if claim_type in HARD_CLAIM_TYPES else 0.0

    margin_threshold = max(1e-6, float(config.margin_hard_threshold))
    margin_signal = 1.0 - _clamp_unit(float(features.prediction_margin) / margin_threshold)

    connectivity_threshold = max(1e-6, float(config.connectivity_hard_threshold))
    connectivity_signal = 1.0 - _clamp_unit(float(features.graph_connectivity) / connectivity_threshold)

    evidence_threshold = max(1, int(config.evidence_low_threshold))
    evidence_signal = _clamp_unit((evidence_threshold - max(0, int(features.evidence_count))) / evidence_threshold)

    total_weight = (
        float(config.type_weight)
        + float(config.margin_weight)
        + float(config.connectivity_weight)
        + float(config.evidence_weight)
    )
    if total_weight <= 0.0:
        raise ValueError("Difficulty weights must sum to a positive value.")

    hard_score = (
        (float(config.type_weight) * type_signal)
        + (float(config.margin_weight) * margin_signal)
        + (float(config.connectivity_weight) * connectivity_signal)
        + (float(config.evidence_weight) * evidence_signal)
    ) / total_weight
    hard_score = _clamp_unit(hard_score)
    return DifficultyScore(
        claim_type=claim_type,
        hard_score=float(hard_score),
        is_hard=bool(hard_score >= float(config.hard_score_threshold)),
        type_signal=float(type_signal),
        margin_signal=float(margin_signal),
        connectivity_signal=float(connectivity_signal),
        evidence_signal=float(evidence_signal),
    )


@dataclass(frozen=True)
class RoutingDevExample:
    claim_id: str
    claim_type: str
    prediction_margin: float
    graph_connectivity: float
    evidence_count: int
    bert_correct: bool
    pv_correct: bool


@dataclass(frozen=True)
class RoutingThresholdPoint:
    threshold: float
    accuracy: float
    pv_route_rate: float


@dataclass(frozen=True)
class RoutingThresholdResult:
    best_threshold: float
    best_accuracy: float
    best_pv_route_rate: float
    sweep: tuple[RoutingThresholdPoint, ...]


def _model_for_features(
    features: DifficultyFeatures,
    *,
    threshold: float,
    difficulty_config: DifficultyConfig,
    easy_margin_threshold: float,
    easy_connectivity_threshold: float,
) -> tuple[str, DifficultyScore]:
    score = score_claim_difficulty(features, config=difficulty_config)
    claim_type = score.claim_type

    is_easy_profile = (
        claim_type in EASY_CLAIM_TYPES
        and float(features.prediction_margin) >= float(easy_margin_threshold)
        and float(features.graph_connectivity) >= float(easy_connectivity_threshold)
        and float(score.hard_score) < float(threshold)
    )
    if is_easy_profile:
        return MODEL_BERT_BASELINE, score

    is_hard_profile = (
        claim_type in HARD_CLAIM_TYPES
        or float(features.prediction_margin) < float(difficulty_config.margin_hard_threshold)
        or float(features.graph_connectivity) < float(difficulty_config.connectivity_hard_threshold)
        or float(score.hard_score) >= float(threshold)
    )
    if is_hard_profile:
        return MODEL_PV_QAGNN, score
    return MODEL_BERT_BASELINE, score


def tune_routing_threshold(
    dev_examples: Sequence[RoutingDevExample],
    *,
    difficulty_config: DifficultyConfig | None = None,
    threshold_grid: Sequence[float] | None = None,
    easy_margin_threshold: float = 0.35,
    easy_connectivity_threshold: float = 0.90,
) -> RoutingThresholdResult:
    """T6.3 threshold tuning to maximize routed accuracy on dev."""

    if not dev_examples:
        raise ValueError("`dev_examples` must not be empty.")
    difficulty_config = difficulty_config or DifficultyConfig()
    if threshold_grid is None:
        threshold_grid = tuple(x / 100.0 for x in range(35, 91, 5))
    if not threshold_grid:
        raise ValueError("`threshold_grid` must not be empty.")
    for value in threshold_grid:
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError("Each routing threshold must be in [0.0, 1.0].")

    sweep_rows: list[RoutingThresholdPoint] = []
    for threshold in threshold_grid:
        correct = 0
        pv_count = 0
        for row in dev_examples:
            features = DifficultyFeatures(
                claim_type=row.claim_type,
                prediction_margin=float(row.prediction_margin),
                graph_connectivity=float(row.graph_connectivity),
                evidence_count=int(row.evidence_count),
            )
            model_key, _ = _model_for_features(
                features,
                threshold=float(threshold),
                difficulty_config=difficulty_config,
                easy_margin_threshold=easy_margin_threshold,
                easy_connectivity_threshold=easy_connectivity_threshold,
            )
            if model_key == MODEL_PV_QAGNN:
                pv_count += 1
                correct += int(bool(row.pv_correct))
            else:
                correct += int(bool(row.bert_correct))
        accuracy = float(correct / len(dev_examples))
        pv_rate = float(pv_count / len(dev_examples))
        sweep_rows.append(
            RoutingThresholdPoint(
                threshold=float(threshold),
                accuracy=accuracy,
                pv_route_rate=pv_rate,
            )
        )

    best = max(sweep_rows, key=lambda row: (row.accuracy, -row.pv_route_rate, row.threshold))
    return RoutingThresholdResult(
        best_threshold=float(best.threshold),
        best_accuracy=float(best.accuracy),
        best_pv_route_rate=float(best.pv_route_rate),
        sweep=tuple(sweep_rows),
    )


@dataclass(frozen=True)
class RouterBudget:
    """T6.4 budget mapping for the PV route."""

    max_active: int
    backtrack_rounds: int
    mask_alpha: float
    gnn_layers: int


DEFAULT_PV_BUDGETS: Mapping[str, RouterBudget] = {
    "existence": RouterBudget(max_active=16, backtrack_rounds=0, mask_alpha=3.2, gnn_layers=1),
    "substitution": RouterBudget(max_active=18, backtrack_rounds=0, mask_alpha=3.4, gnn_layers=1),
    "single hop": RouterBudget(max_active=20, backtrack_rounds=0, mask_alpha=3.8, gnn_layers=1),
    "multi hop": RouterBudget(max_active=28, backtrack_rounds=2, mask_alpha=4.8, gnn_layers=2),
    "multi claim": RouterBudget(max_active=26, backtrack_rounds=2, mask_alpha=4.6, gnn_layers=2),
    "negation": RouterBudget(max_active=24, backtrack_rounds=2, mask_alpha=4.7, gnn_layers=2),
}


def get_pv_route_budget(
    claim_type: str,
    *,
    hard_score: float,
    budget_table: Mapping[str, RouterBudget] | None = None,
) -> RouterBudget:
    claim_type = _normalize_claim_type(claim_type)
    budget_table = budget_table or DEFAULT_PV_BUDGETS
    base = budget_table[claim_type]

    if hard_score < 0.80 or claim_type not in HARD_CLAIM_TYPES:
        return base

    # High-hardness escalation remains bounded by the locked constraints.
    return RouterBudget(
        max_active=int(base.max_active + 4),
        backtrack_rounds=min(2, int(base.backtrack_rounds + 1)),
        mask_alpha=float(base.mask_alpha + 0.3),
        gnn_layers=max(2, int(base.gnn_layers)),
    )


@dataclass(frozen=True)
class RouteInput:
    claim_id: str
    claim_type: str
    prediction_margin: float
    graph_connectivity: float
    evidence_count: int


@dataclass(frozen=True)
class RouteDecision:
    claim_id: str
    selected_model: str
    difficulty: DifficultyScore
    budget: RouterBudget | None
    reason: str


@dataclass(frozen=True)
class ModelRouterConfig:
    difficulty_config: DifficultyConfig = field(default_factory=DifficultyConfig)
    routing_threshold: float = 0.55
    easy_margin_threshold: float = 0.35
    easy_connectivity_threshold: float = 0.90


class Component6ModelRouter:
    """Router that dispatches claims to BERT baseline vs PV-QA-GNN path."""

    def __init__(
        self,
        *,
        config: ModelRouterConfig | None = None,
        budget_table: Mapping[str, RouterBudget] | None = None,
    ) -> None:
        self.config = config or ModelRouterConfig()
        self.budget_table = budget_table or DEFAULT_PV_BUDGETS

    def route(self, route_input: RouteInput) -> RouteDecision:
        features = DifficultyFeatures(
            claim_type=route_input.claim_type,
            prediction_margin=float(route_input.prediction_margin),
            graph_connectivity=float(route_input.graph_connectivity),
            evidence_count=int(route_input.evidence_count),
        )
        model_key, score = _model_for_features(
            features,
            threshold=self.config.routing_threshold,
            difficulty_config=self.config.difficulty_config,
            easy_margin_threshold=self.config.easy_margin_threshold,
            easy_connectivity_threshold=self.config.easy_connectivity_threshold,
        )

        if model_key == MODEL_BERT_BASELINE:
            return RouteDecision(
                claim_id=route_input.claim_id,
                selected_model=MODEL_BERT_BASELINE,
                difficulty=score,
                budget=None,
                reason="easy_profile",
            )

        budget = get_pv_route_budget(
            score.claim_type,
            hard_score=score.hard_score,
            budget_table=self.budget_table,
        )
        return RouteDecision(
            claim_id=route_input.claim_id,
            selected_model=MODEL_PV_QAGNN,
            difficulty=score,
            budget=budget,
            reason="hard_profile",
        )


def evaluate_routed_accuracy(
    dev_examples: Sequence[RoutingDevExample],
    router: Component6ModelRouter,
) -> float:
    if not dev_examples:
        raise ValueError("`dev_examples` must not be empty.")
    correct = 0
    for row in dev_examples:
        decision = router.route(
            RouteInput(
                claim_id=row.claim_id,
                claim_type=row.claim_type,
                prediction_margin=row.prediction_margin,
                graph_connectivity=row.graph_connectivity,
                evidence_count=row.evidence_count,
            )
        )
        if decision.selected_model == MODEL_PV_QAGNN:
            correct += int(bool(row.pv_correct))
        else:
            correct += int(bool(row.bert_correct))
    return float(correct / len(dev_examples))


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


@dataclass(frozen=True)
class Component6Config:
    run_id: str | None = None
    output_root: str = "runs"
    classifier: RouterClassifierConfig = field(default_factory=RouterClassifierConfig)
    router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    threshold_grid: tuple[float, ...] = tuple(x / 100.0 for x in range(35, 91, 5))


def build_synthetic_component6_data() -> tuple[list[RouterTrainingExample], list[RouterTrainingExample], list[RoutingDevExample]]:
    """Deterministic synthetic data for smoke tests when full ML stack/dataset is unavailable."""

    train_examples = [
        RouterTrainingExample("is there a city named oslo in norway", "existence", "tr_1"),
        RouterTrainingExample("does a species called polar bear exist", "existence", "tr_2"),
        RouterTrainingExample("replace actor with singer in this claim", "substitution", "tr_3"),
        RouterTrainingExample("substitute capital paris with lyon", "substitution", "tr_4"),
        RouterTrainingExample("if a leads to b and b leads to c then c holds", "multi hop", "tr_5"),
        RouterTrainingExample("chain of two relations needed for this proof", "multi hop", "tr_6"),
        RouterTrainingExample("both conditions must hold together for this claim", "multi claim", "tr_7"),
        RouterTrainingExample("requires two independent evidences to verify", "multi claim", "tr_8"),
        RouterTrainingExample("it is not true that the city is in germany", "negation", "tr_9"),
        RouterTrainingExample("claim denies that a relation exists", "negation", "tr_10"),
        RouterTrainingExample("single relation says person born in city", "single hop", "tr_11"),
        RouterTrainingExample("one step relation verifies the statement", "single hop", "tr_12"),
    ]
    dev_examples = [
        RouterTrainingExample("does river nile exist in africa", "existence", "dv_1"),
        RouterTrainingExample("swap author with painter in sentence", "substitution", "dv_2"),
        RouterTrainingExample("this requires a chain over two edges", "multi hop", "dv_3"),
        RouterTrainingExample("both evidence pieces are required", "multi claim", "dv_4"),
        RouterTrainingExample("claim is not correct according to graph", "negation", "dv_5"),
        RouterTrainingExample("one edge relation proves this", "single hop", "dv_6"),
    ]
    routing_examples = [
        RoutingDevExample("r_1", "existence", 0.82, 1.0, 5, bert_correct=True, pv_correct=False),
        RoutingDevExample("r_2", "substitution", 0.74, 0.95, 4, bert_correct=True, pv_correct=True),
        RoutingDevExample("r_3", "single hop", 0.61, 0.93, 4, bert_correct=True, pv_correct=True),
        RoutingDevExample("r_4", "multi hop", 0.07, 0.32, 2, bert_correct=False, pv_correct=True),
        RoutingDevExample("r_5", "multi claim", 0.11, 0.58, 2, bert_correct=False, pv_correct=True),
        RoutingDevExample("r_6", "negation", 0.09, 0.40, 1, bert_correct=False, pv_correct=True),
    ]
    return train_examples, dev_examples, routing_examples


def run_component6_pipeline(
    config: Component6Config,
    *,
    train_examples: Sequence[RouterTrainingExample],
    dev_examples: Sequence[RouterTrainingExample],
    routing_dev_examples: Sequence[RoutingDevExample],
) -> tuple[Path, dict[str, Any]]:
    """Run T6.1-T6.4 end-to-end and persist run artifacts."""

    run_id = config.run_id or f"component6_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _classifier, train_report = train_router_classifier(
        train_examples=train_examples,
        dev_examples=dev_examples,
        config=config.classifier,
    )
    threshold_result = tune_routing_threshold(
        routing_dev_examples,
        difficulty_config=config.router.difficulty_config,
        threshold_grid=config.threshold_grid,
        easy_margin_threshold=config.router.easy_margin_threshold,
        easy_connectivity_threshold=config.router.easy_connectivity_threshold,
    )
    router = Component6ModelRouter(
        config=ModelRouterConfig(
            difficulty_config=config.router.difficulty_config,
            routing_threshold=threshold_result.best_threshold,
            easy_margin_threshold=config.router.easy_margin_threshold,
            easy_connectivity_threshold=config.router.easy_connectivity_threshold,
        )
    )

    routed_accuracy = evaluate_routed_accuracy(routing_dev_examples, router)
    decisions = []
    for row in routing_dev_examples:
        decision = router.route(
            RouteInput(
                claim_id=row.claim_id,
                claim_type=row.claim_type,
                prediction_margin=row.prediction_margin,
                graph_connectivity=row.graph_connectivity,
                evidence_count=row.evidence_count,
            )
        )
        decisions.append(
            {
                "claim_id": row.claim_id,
                "claim_type": row.claim_type,
                "selected_model": decision.selected_model,
                "reason": decision.reason,
                "hard_score": decision.difficulty.hard_score,
                "budget": asdict(decision.budget) if decision.budget is not None else None,
            }
        )

    metrics = {
        "task": "T6",
        "run_id": run_id,
        "t61_router_classifier": asdict(train_report),
        "t62_difficulty_features": {
            "fields": ["claim_type", "prediction_margin", "graph_connectivity", "evidence_count"],
            "hard_threshold": config.router.difficulty_config.hard_score_threshold,
        },
        "t63_threshold_tuning": {
            "best_threshold": threshold_result.best_threshold,
            "best_accuracy": threshold_result.best_accuracy,
            "best_pv_route_rate": threshold_result.best_pv_route_rate,
            "sweep": [asdict(row) for row in threshold_result.sweep],
            "routed_accuracy_at_best": routed_accuracy,
        },
        "t64_budget_mapping": {
            claim_type: asdict(get_pv_route_budget(claim_type, hard_score=0.9))
            for claim_type in CLAIM_TYPES
        },
        "routing_decisions": decisions,
    }

    config_payload = asdict(config)
    config_payload["task"] = "T6"
    config_payload["claim_types"] = list(CLAIM_TYPES)
    config_payload["model_keys"] = [MODEL_BERT_BASELINE, MODEL_PV_QAGNN]
    config_payload["threshold_grid"] = list(config.threshold_grid)
    config_payload["router"]["routing_threshold"] = threshold_result.best_threshold
    write_config_yaml(run_dir / "config.yaml", config_payload)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return run_dir, metrics


def _parse_args() -> Component6Config:
    parser = argparse.ArgumentParser(description="Run Component 6 router pipeline.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "bert", "naive_bayes"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--min-dev-accuracy", type=float, default=0.80)
    parser.add_argument("--routing-threshold", type=float, default=0.55)
    parser.add_argument("--easy-margin-threshold", type=float, default=0.35)
    parser.add_argument("--easy-connectivity-threshold", type=float, default=0.90)
    args = parser.parse_args()

    return Component6Config(
        run_id=args.run_id,
        output_root=args.output_root,
        classifier=RouterClassifierConfig(
            backend=args.backend,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            min_dev_accuracy=args.min_dev_accuracy,
        ),
        router=ModelRouterConfig(
            routing_threshold=args.routing_threshold,
            easy_margin_threshold=args.easy_margin_threshold,
            easy_connectivity_threshold=args.easy_connectivity_threshold,
        ),
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train_rows, dev_rows, routing_rows = build_synthetic_component6_data()
    run_path, report = run_component6_pipeline(
        cfg,
        train_examples=train_rows,
        dev_examples=dev_rows,
        routing_dev_examples=routing_rows,
    )
    print(f"Run directory: {run_path}")
    print(json.dumps(report["t63_threshold_tuning"], indent=2))
