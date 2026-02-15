"""Learned controller and custom losses for Component 5."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .backtracking import RuleBasedBacktrackingController, is_disconnected_graph, margin_from_prediction
except ImportError:  # pragma: no cover - fallback for direct module execution
    try:
        from component3.backtracking import RuleBasedBacktrackingController, is_disconnected_graph, margin_from_prediction
    except ImportError:  # pragma: no cover - fallback for spec-based loading in tests
        backtracking_path = Path(__file__).resolve().parent / "backtracking.py"
        spec = importlib.util.spec_from_file_location("component3_backtracking_fallback", backtracking_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        RuleBasedBacktrackingController = module.RuleBasedBacktrackingController
        is_disconnected_graph = module.is_disconnected_graph
        margin_from_prediction = module.margin_from_prediction

try:
    from ._graph_utils import (
        triple_evidence_id as _triple_evidence_id,
        triple_p_con as _triple_p_con,
        triple_pool as _triple_pool,
        triple_rel as _triple_rel,
        triple_with_pool as _triple_with_pool,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from component3._graph_utils import (
        triple_evidence_id as _triple_evidence_id,
        triple_p_con as _triple_p_con,
        triple_pool as _triple_pool,
        triple_rel as _triple_rel,
        triple_with_pool as _triple_with_pool,
    )


ACTION_NO_OP = 0
ACTION_RECOVER_TOP1 = 1
ACTION_RECOVER_TOP3 = 2

ACTION_LABELS = {
    ACTION_NO_OP: "no_op",
    ACTION_RECOVER_TOP1: "recover_top_1",
    ACTION_RECOVER_TOP3: "recover_top_3",
}
ACTION_RECOVER_K = {
    ACTION_NO_OP: 0,
    ACTION_RECOVER_TOP1: 1,
    ACTION_RECOVER_TOP3: 3,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _scalar(value: Any) -> float:
    if _is_tensor(value):
        try:
            return float(value.detach().item())
        except Exception:  # pragma: no cover - defensive fallback
            return float(value.detach().mean().item())
    return _as_float(value, default=0.0)


def _safe_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    values = [_as_float(value, default=0.0) for value in logits]
    if not values:
        return ()
    max_logit = max(values)
    exp_values = [math.exp(value - max_logit) for value in values]
    total = sum(exp_values)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in exp_values)


def _extract_probabilities(prediction: Mapping[str, Any]) -> tuple[float, ...]:
    probabilities = prediction.get("probabilities")
    if _is_sequence(probabilities):
        values = [_as_float(item, default=0.0) for item in probabilities]
        if values:
            total = sum(values)
            if total > 0:
                return tuple(value / total for value in values)

    if "prob" in prediction:
        prob = min(1.0, max(0.0, _as_float(prediction.get("prob"), default=0.5)))
        return (prob, 1.0 - prob)

    if "confidence" in prediction:
        prob = min(1.0, max(0.0, _as_float(prediction.get("confidence"), default=0.5)))
        return (prob, 1.0 - prob)

    if "logit" in prediction:
        logit = _as_float(prediction.get("logit"), default=0.0)
        prob = 1.0 / (1.0 + math.exp(-logit))
        return (prob, 1.0 - prob)

    return (0.5, 0.5)


def prediction_entropy(probabilities: Sequence[float]) -> float:
    probs = [_as_float(value, default=0.0) for value in probabilities]
    total = sum(probs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for prob in probs:
        p = prob / total
        if p <= 0:
            continue
        entropy -= p * math.log(p)
    return float(entropy)


def compute_top2_margin(probabilities: Sequence[float]) -> float:
    probs = [_as_float(value, default=0.0) for value in probabilities]
    if not probs:
        return 1.0
    if len(probs) == 1:
        p = min(1.0, max(0.0, probs[0]))
        return abs((2.0 * p) - 1.0)
    top_two = sorted(probs, reverse=True)[:2]
    return float(max(0.0, top_two[0] - top_two[1]))


def compute_active_mass(triples: Sequence[Any]) -> float:
    return float(sum(_triple_rel(triple) for triple in triples if _triple_pool(triple) in {"A", "C"}))


def compute_counter_mass(triples: Sequence[Any]) -> float:
    counter_rows = [triple for triple in triples if _triple_pool(triple) == "C"]
    if not counter_rows:
        counter_rows = list(triples)
    return float(sum(_triple_p_con(triple) for triple in counter_rows))


def compute_mean_rel_active(triples: Sequence[Any]) -> float:
    rel_values = [_triple_rel(triple) for triple in triples if _triple_pool(triple) == "A"]
    if not rel_values:
        return 0.0
    return float(sum(rel_values) / len(rel_values))


@dataclass(frozen=True)
class ControllerFeatures:
    """Input feature vector for learned Component 5 backtracking decisions."""

    prediction_entropy: float
    top2_margin: float
    connectivity: float
    counter_mass: float
    active_mass: float
    esi_score: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.prediction_entropy),
            float(self.top2_margin),
            float(self.connectivity),
            float(self.counter_mass),
            float(self.active_mass),
            float(self.esi_score),
        )


@dataclass(frozen=True)
class ControllerDecision:
    """Controller output action over {no-op, recover top-1, recover top-3}."""

    action_index: int
    action_name: str
    recover_k: int
    probabilities: tuple[float, float, float]
    features: ControllerFeatures


def extract_controller_features(
    *,
    prediction: Mapping[str, Any],
    active_triples: Sequence[Any],
    suspended_triples: Sequence[Any] | None = None,
    anchors: Sequence[str] | None = None,
    esi_score: float | None = None,
) -> ControllerFeatures:
    """Build the fixed 6-feature input vector for the controller MLP."""

    probabilities = _extract_probabilities(prediction)
    entropy = prediction_entropy(probabilities)

    if "margin" in prediction and prediction["margin"] is not None:
        margin = _as_float(prediction["margin"], default=0.0)
    else:
        margin = float(margin_from_prediction(prediction))
    margin = max(0.0, margin)

    if "connectivity" in prediction and prediction["connectivity"] is not None:
        connectivity = min(1.0, max(0.0, _as_float(prediction["connectivity"], default=0.0)))
    elif "is_disconnected" in prediction:
        connectivity = 0.0 if bool(prediction["is_disconnected"]) else 1.0
    else:
        connectivity = 0.0 if is_disconnected_graph(active_triples, anchors=anchors) else 1.0

    all_triples: list[Any] = list(active_triples)
    if suspended_triples is not None:
        all_triples.extend(suspended_triples)

    if "counter_mass" in prediction and prediction["counter_mass"] is not None:
        counter_mass = _as_float(prediction["counter_mass"], default=0.0)
    else:
        counter_mass = compute_counter_mass(all_triples)

    if "active_mass" in prediction and prediction["active_mass"] is not None:
        active_mass = _as_float(prediction["active_mass"], default=0.0)
    else:
        active_mass = compute_active_mass(active_triples)

    if esi_score is None:
        if "esi_score" in prediction and prediction["esi_score"] is not None:
            esi = _as_float(prediction["esi_score"], default=1.0)
        else:
            esi = 1.0
    else:
        esi = _as_float(esi_score, default=1.0)

    return ControllerFeatures(
        prediction_entropy=float(entropy),
        top2_margin=float(margin if probabilities else compute_top2_margin((0.5, 0.5))),
        connectivity=float(connectivity),
        counter_mass=float(counter_mass),
        active_mass=float(active_mass),
        esi_score=float(esi),
    )


class ControllerMLP:
    """Two-layer MLP policy over fixed controller features."""

    def __init__(
        self,
        *,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 3,
    ) -> None:
        if int(input_dim) != 6:
            raise ValueError("Component 5 controller expects exactly 6 input features.")
        if int(output_dim) != 3:
            raise ValueError("Component 5 controller expects exactly 3 actions.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        self._torch = None
        self._module = None
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            self._torch = torch
            self._module = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        except Exception:  # pragma: no cover - optional dependency
            self._torch = None
            self._module = None

    @property
    def trainable(self) -> bool:
        return bool(self._module is not None)

    def parameters(self):
        if self._module is None:
            return ()
        return self._module.parameters()

    def named_parameters(self):
        if self._module is None:
            return ()
        return self._module.named_parameters()

    def train(self, mode: bool = True):
        if self._module is not None:
            self._module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        if self._module is not None:
            self._module.to(*args, **kwargs)
        return self

    def _feature_values(self, features: ControllerFeatures | Sequence[float] | Mapping[str, Any]) -> list[float]:
        if isinstance(features, ControllerFeatures):
            return list(features.as_tuple())
        if _is_mapping(features):
            return list(
                ControllerFeatures(
                    prediction_entropy=_as_float(features.get("prediction_entropy"), default=0.0),
                    top2_margin=_as_float(features.get("top2_margin"), default=0.0),
                    connectivity=_as_float(features.get("connectivity"), default=0.0),
                    counter_mass=_as_float(features.get("counter_mass"), default=0.0),
                    active_mass=_as_float(features.get("active_mass"), default=0.0),
                    esi_score=_as_float(features.get("esi_score"), default=1.0),
                ).as_tuple()
            )
        if _is_sequence(features) and len(features) == self.input_dim:
            return [_as_float(value, default=0.0) for value in features]
        raise ValueError(
            "Controller features must be `ControllerFeatures`, a mapping, or a 6-length sequence."
        )

    def logits(self, features: ControllerFeatures | Sequence[float] | Mapping[str, Any]):
        feature_values = self._feature_values(features)
        if self._module is not None and self._torch is not None:
            tensor = self._torch.tensor(feature_values, dtype=self._torch.float32).view(1, -1)
            logits = self._module(tensor).view(-1)
            return logits

        # Fallback heuristic logits for environments without torch.
        entropy, margin, connectivity, counter_mass, active_mass, esi = feature_values
        logits = (
            (margin * 3.0) + (connectivity * 1.5) - entropy,
            entropy + (1.0 - margin) + (1.0 - connectivity) + (0.2 * counter_mass),
            (1.2 * entropy)
            + (1.5 * (1.0 - margin))
            + (1.2 * (1.0 - connectivity))
            + (0.2 * counter_mass)
            + (0.1 * max(0.0, 1.0 - active_mass))
            + (0.4 * max(0.0, 1.0 - esi)),
        )
        return logits

    def predict_proba(
        self,
        features: ControllerFeatures | Sequence[float] | Mapping[str, Any],
    ) -> tuple[float, float, float]:
        logits = self.logits(features)
        if _is_tensor(logits):
            if self._torch is None:  # pragma: no cover - defensive fallback
                raise RuntimeError("Torch-backed logits produced without torch runtime.")
            probs = self._torch.softmax(logits.detach(), dim=-1)
            return tuple(float(value.item()) for value in probs)
        return tuple(_safe_softmax(logits))

    def decide(
        self,
        features: ControllerFeatures | Sequence[float] | Mapping[str, Any],
    ) -> ControllerDecision:
        if isinstance(features, ControllerFeatures):
            feature_row = features
        else:
            values = self._feature_values(features)
            feature_row = ControllerFeatures(
                prediction_entropy=values[0],
                top2_margin=values[1],
                connectivity=values[2],
                counter_mass=values[3],
                active_mass=values[4],
                esi_score=values[5],
            )
        probs = self.predict_proba(feature_row)
        action_index = int(max(range(len(probs)), key=lambda idx: probs[idx]))
        return ControllerDecision(
            action_index=action_index,
            action_name=ACTION_LABELS[action_index],
            recover_k=ACTION_RECOVER_K[action_index],
            probabilities=(float(probs[0]), float(probs[1]), float(probs[2])),
            features=feature_row,
        )


@dataclass(frozen=True)
class LearnedBacktrackingAction:
    """Single backtracking round driven by controller action."""

    round_index: int
    decision: ControllerDecision
    selected_evidence_ids: tuple[str, ...]
    margin_before: float
    margin_after: float
    prediction_before: Any
    prediction_after: Any


@dataclass(frozen=True)
class LearnedBacktrackingResult:
    """Run summary for learned controller backtracking."""

    claim_id: str
    triggered: bool
    rounds_run: int
    initial_prediction: Any
    final_prediction: Any
    promoted_evidence_ids: tuple[str, ...]
    actions: tuple[LearnedBacktrackingAction, ...]


class LearnedBacktrackingController:
    """Component 5 learned controller replacing rule-based trigger logic."""

    def __init__(
        self,
        *,
        policy: ControllerMLP | None = None,
        max_backtrack_rounds: int = 2,
        max_recover_k: int = 3,
        rel_threshold: float = 0.3,
        min_a: int = 5,
    ) -> None:
        if max_backtrack_rounds < 1 or max_backtrack_rounds > 2:
            raise ValueError("`max_backtrack_rounds` must be in [1, 2] for Component 5.")
        if max_recover_k < 1 or max_recover_k > 3:
            raise ValueError("`max_recover_k` must be in [1, 3] for Component 5.")

        self.policy = policy or ControllerMLP()
        self.max_backtrack_rounds = int(max_backtrack_rounds)
        self.max_recover_k = int(max_recover_k)
        self.rel_threshold = float(rel_threshold)
        self.min_a = int(min_a)
        self._ranker = RuleBasedBacktrackingController(
            max_backtrack_rounds=max_backtrack_rounds,
            backtrack_k=max_recover_k,
            min_a=min_a,
            rel_threshold=rel_threshold,
        )

    def decide(
        self,
        *,
        prediction: Mapping[str, Any],
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        anchors: Sequence[str] | None = None,
        esi_score: float | None = None,
    ) -> ControllerDecision:
        features = extract_controller_features(
            prediction=prediction,
            active_triples=active_triples,
            suspended_triples=suspended_triples,
            anchors=anchors,
            esi_score=esi_score,
        )
        return self.policy.decide(features)

    def run(
        self,
        *,
        claim_id: str,
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        predictor: Callable[[Sequence[Any]], Mapping[str, Any]],
        anchors: Sequence[str] | None = None,
        rebuild_graph_fn: Callable[[Sequence[Any]], Any] | None = None,
        rerun_model_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> LearnedBacktrackingResult:
        current_active = list(active_triples)
        remaining_s = [triple for triple in suspended_triples if _triple_pool(triple) == "S"]
        promoted_ids: list[str] = []
        actions: list[LearnedBacktrackingAction] = []

        prediction = dict(predictor(current_active))
        initial_prediction = dict(prediction)
        rounds_run = 0

        for round_idx in range(1, self.max_backtrack_rounds + 1):
            decision = self.decide(
                prediction=prediction,
                active_triples=current_active,
                suspended_triples=remaining_s,
                anchors=anchors,
                esi_score=prediction.get("esi_score"),
            )
            recover_k = min(int(decision.recover_k), self.max_recover_k)
            if recover_k <= 0:
                break
            if not remaining_s:
                break

            ranked_rows = self._ranker.rank_suspended(
                active_triples=current_active,
                suspended_triples=remaining_s,
                bridge_bonus_by_id=prediction.get("bridge_bonus_by_id", {}),
                salience_by_id=prediction.get("salience_by_id", {}),
            )
            if not ranked_rows:
                break
            selected_rows = ranked_rows[:recover_k]
            selected_ids = {row.evidence_id for row in selected_rows}
            if not selected_ids:
                break

            next_remaining_s: list[Any] = []
            for idx, triple in enumerate(remaining_s):
                evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
                if evidence_id in selected_ids:
                    current_active.append(_triple_with_pool(triple, "A"))
                    promoted_ids.append(evidence_id)
                else:
                    next_remaining_s.append(triple)
            remaining_s = next_remaining_s

            prediction_before = dict(prediction)
            if rebuild_graph_fn is not None and rerun_model_fn is not None:
                graph_obj = rebuild_graph_fn(current_active)
                prediction = dict(rerun_model_fn(graph_obj))
            else:
                prediction = dict(predictor(current_active))
            rounds_run = round_idx

            actions.append(
                LearnedBacktrackingAction(
                    round_index=round_idx,
                    decision=decision,
                    selected_evidence_ids=tuple(row.evidence_id for row in selected_rows),
                    margin_before=float(margin_from_prediction(prediction_before)),
                    margin_after=float(margin_from_prediction(prediction)),
                    prediction_before=prediction_before,
                    prediction_after=dict(prediction),
                )
            )

        return LearnedBacktrackingResult(
            claim_id=str(claim_id),
            triggered=bool(actions),
            rounds_run=int(rounds_run),
            initial_prediction=initial_prediction,
            final_prediction=dict(prediction),
            promoted_evidence_ids=tuple(promoted_ids),
            actions=tuple(actions),
        )


def _relu_positive(value: Any):
    if _is_tensor(value):
        return value.clamp(min=0.0)
    return max(0.0, _as_float(value, default=0.0))


def _zero_like(value: Any):
    if _is_tensor(value):
        return value * 0.0
    return 0.0


def starvation_penalty_loss(
    mean_rel_a: Any,
    *,
    lambda_starv: float = 0.1,
    tau_starv: float = 0.3,
):
    """L_starv = lambda_1 * max(0, tau_starv - mean_rel_A)."""

    gap = float(tau_starv) - mean_rel_a
    return float(lambda_starv) * _relu_positive(gap)


def counter_evidence_preservation_loss(
    max_contra_pool: Any,
    max_contra_used: Any,
    *,
    lambda_counter: float = 0.1,
):
    """L_counter = lambda_2 * max(0, max_contra_pool - max_contra_used)."""

    gap = max_contra_pool - max_contra_used
    return float(lambda_counter) * _relu_positive(gap)


def recovery_utility_loss(
    prob_restored: Any,
    prob_masked: Any,
    *,
    lambda_recovery: float = 0.1,
):
    """L_recovery = lambda_3 * max(0, P_restored - P_masked)."""

    improvement = prob_restored - prob_masked
    return float(lambda_recovery) * _relu_positive(improvement)


def _prediction_to_probability(prediction: Any) -> float:
    if _is_mapping(prediction):
        if "probability" in prediction:
            return min(1.0, max(0.0, _as_float(prediction["probability"], default=0.5)))
        if "prob" in prediction:
            return min(1.0, max(0.0, _as_float(prediction["prob"], default=0.5)))
        probs = _extract_probabilities(prediction)
        if probs:
            return float(max(probs))
    if _is_sequence(prediction):
        values = [_as_float(value, default=0.0) for value in prediction]
        if values:
            return float(max(values))
    value = _as_float(prediction, default=0.5)
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class SyntheticMaskingStep:
    mask_size: int
    masked_evidence_ids: tuple[str, ...]
    prob_masked: float
    prob_restored: float
    loss_recovery: float


def synthetic_masking_recovery_loss(
    *,
    active_triples: Sequence[Any],
    gold_evidence_ids: Sequence[str],
    predictor: Callable[[Sequence[Any]], Any],
    lambda_recovery: float = 0.1,
    mask_sizes: Sequence[int] = (1, 2, 3),
) -> tuple[float, tuple[SyntheticMaskingStep, ...]]:
    """Compute recovery-utility loss by masking 1-3 gold triples and restoring."""

    active_rows = list(active_triples)
    if not active_rows:
        return 0.0, ()

    gold_set = {str(item) for item in gold_evidence_ids}
    gold_in_active: list[str] = []
    for idx, triple in enumerate(active_rows):
        evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
        if evidence_id in gold_set:
            gold_in_active.append(evidence_id)

    if not gold_in_active:
        return 0.0, ()

    base_probability = _prediction_to_probability(predictor(active_rows))
    steps: list[SyntheticMaskingStep] = []
    losses: list[float] = []

    for size in mask_sizes:
        n_mask = int(size)
        if n_mask <= 0:
            continue
        masked_ids = tuple(gold_in_active[:n_mask])
        if len(masked_ids) < n_mask:
            continue
        masked_id_set = set(masked_ids)
        masked_rows = [
            row
            for idx, row in enumerate(active_rows)
            if _triple_evidence_id(row, fallback_idx=idx) not in masked_id_set
        ]
        masked_probability = _prediction_to_probability(predictor(masked_rows))
        restored_probability = _prediction_to_probability(predictor(active_rows))
        loss_value = recovery_utility_loss(
            restored_probability,
            masked_probability,
            lambda_recovery=lambda_recovery,
        )
        loss_float = _as_float(loss_value, default=0.0)
        losses.append(loss_float)
        steps.append(
            SyntheticMaskingStep(
                mask_size=n_mask,
                masked_evidence_ids=masked_ids,
                prob_masked=float(masked_probability),
                prob_restored=float(restored_probability if not math.isnan(restored_probability) else base_probability),
                loss_recovery=float(loss_float),
            )
        )

    if not losses:
        return 0.0, tuple(steps)
    return float(sum(losses) / len(losses)), tuple(steps)


@dataclass(frozen=True)
class LossBreakdown:
    """Component 5 loss decomposition."""

    loss_bce: float
    loss_evidence: float
    loss_starvation: float
    loss_counter: float
    loss_recovery: float
    total_loss: float
    enabled_terms: dict[str, bool]


def _coerce_enabled_terms(enabled_terms: Mapping[str, bool] | None = None) -> dict[str, bool]:
    defaults = {"starvation": True, "counter": True, "recovery": True}
    if enabled_terms is None:
        return defaults
    rows = dict(defaults)
    for key in defaults:
        if key in enabled_terms:
            rows[key] = bool(enabled_terms[key])
    return rows


def compute_component5_joint_loss(
    *,
    loss_bce: Any,
    loss_evidence: Any,
    mean_rel_a: Any,
    max_contra_pool: Any,
    max_contra_used: Any,
    prob_restored: Any,
    prob_masked: Any,
    lambda_starv: float = 0.1,
    tau_starv: float = 0.3,
    lambda_counter: float = 0.1,
    lambda_recovery: float = 0.1,
    enabled_terms: Mapping[str, bool] | None = None,
) -> tuple[Any, LossBreakdown]:
    """Compute Component 5 joint loss with optional per-term ablations."""

    enabled = _coerce_enabled_terms(enabled_terms)
    starv = (
        starvation_penalty_loss(
            mean_rel_a,
            lambda_starv=lambda_starv,
            tau_starv=tau_starv,
        )
        if enabled["starvation"]
        else _zero_like(loss_bce)
    )
    counter = (
        counter_evidence_preservation_loss(
            max_contra_pool,
            max_contra_used,
            lambda_counter=lambda_counter,
        )
        if enabled["counter"]
        else _zero_like(loss_bce)
    )
    recovery = (
        recovery_utility_loss(
            prob_restored,
            prob_masked,
            lambda_recovery=lambda_recovery,
        )
        if enabled["recovery"]
        else _zero_like(loss_bce)
    )
    total = loss_bce + loss_evidence + starv + counter + recovery
    breakdown = LossBreakdown(
        loss_bce=_scalar(loss_bce),
        loss_evidence=_scalar(loss_evidence),
        loss_starvation=_scalar(starv),
        loss_counter=_scalar(counter),
        loss_recovery=_scalar(recovery),
        total_loss=_scalar(total),
        enabled_terms=enabled,
    )
    return total, breakdown


LOSS_ABLATION_VARIANTS = {
    "full": {"starvation": True, "counter": True, "recovery": True},
    "no_starvation": {"starvation": False, "counter": True, "recovery": True},
    "no_counter": {"starvation": True, "counter": False, "recovery": True},
    "no_recovery": {"starvation": True, "counter": True, "recovery": False},
    "bce_plus_evidence_only": {"starvation": False, "counter": False, "recovery": False},
}


def resolve_loss_ablation_variant(name: str) -> dict[str, bool]:
    key = str(name).strip().lower()
    if key not in LOSS_ABLATION_VARIANTS:
        raise ValueError(
            "Unsupported loss ablation variant. "
            f"Expected one of {sorted(LOSS_ABLATION_VARIANTS.keys())}, got: {name!r}"
        )
    return dict(LOSS_ABLATION_VARIANTS[key])


def run_loss_ablation(
    *,
    loss_bce: float,
    loss_evidence: float,
    mean_rel_a: float,
    max_contra_pool: float,
    max_contra_used: float,
    prob_restored: float,
    prob_masked: float,
    lambda_starv: float = 0.1,
    tau_starv: float = 0.3,
    lambda_counter: float = 0.1,
    lambda_recovery: float = 0.1,
    variants: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Run per-loss-term ablation and report marginal total-loss deltas."""

    variant_rows = list(variants) if variants is not None else list(LOSS_ABLATION_VARIANTS.keys())
    reports: list[dict[str, Any]] = []

    _, full_breakdown = compute_component5_joint_loss(
        loss_bce=loss_bce,
        loss_evidence=loss_evidence,
        mean_rel_a=mean_rel_a,
        max_contra_pool=max_contra_pool,
        max_contra_used=max_contra_used,
        prob_restored=prob_restored,
        prob_masked=prob_masked,
        lambda_starv=lambda_starv,
        tau_starv=tau_starv,
        lambda_counter=lambda_counter,
        lambda_recovery=lambda_recovery,
        enabled_terms=resolve_loss_ablation_variant("full"),
    )

    for variant in variant_rows:
        enabled = resolve_loss_ablation_variant(str(variant))
        _, breakdown = compute_component5_joint_loss(
            loss_bce=loss_bce,
            loss_evidence=loss_evidence,
            mean_rel_a=mean_rel_a,
            max_contra_pool=max_contra_pool,
            max_contra_used=max_contra_used,
            prob_restored=prob_restored,
            prob_masked=prob_masked,
            lambda_starv=lambda_starv,
            tau_starv=tau_starv,
            lambda_counter=lambda_counter,
            lambda_recovery=lambda_recovery,
            enabled_terms=enabled,
        )
        reports.append(
            {
                "variant": str(variant),
                "enabled_terms": dict(enabled),
                "loss_bce": breakdown.loss_bce,
                "loss_evidence": breakdown.loss_evidence,
                "loss_starvation": breakdown.loss_starvation,
                "loss_counter": breakdown.loss_counter,
                "loss_recovery": breakdown.loss_recovery,
                "total_loss": breakdown.total_loss,
                "delta_vs_full": float(full_breakdown.total_loss - breakdown.total_loss),
            }
        )
    return reports


class Component5JointLossAdapter:
    """Criterion adapter for T5.5 joint training objective."""

    def __init__(
        self,
        model: Any,
        *,
        lambda_evidence: float = 1.0,
        lambda_starv: float = 0.1,
        tau_starv: float = 0.3,
        lambda_counter: float = 0.1,
        lambda_recovery: float = 0.1,
        loss_ablation_variant: str = "full",
    ) -> None:
        import torch.nn as nn  # type: ignore

        self.model = model
        self.lambda_evidence = float(lambda_evidence)
        self.lambda_starv = float(lambda_starv)
        self.tau_starv = float(tau_starv)
        self.lambda_counter = float(lambda_counter)
        self.lambda_recovery = float(lambda_recovery)
        self.enabled_terms = resolve_loss_ablation_variant(loss_ablation_variant)
        self._bce = nn.BCEWithLogitsLoss()

        self.last_breakdown: LossBreakdown | None = None
        self.last_bce = 0.0
        self.last_evidence = 0.0
        self.last_starvation = 0.0
        self.last_counter = 0.0
        self.last_recovery = 0.0
        self.last_total = 0.0

    def _context_value(self, context: Mapping[str, Any], key: str, default: Any) -> Any:
        if key not in context:
            return default
        value = context[key]
        if value is None:
            return default
        return value

    def __call__(self, claim_logits, labels):
        import torch  # type: ignore

        labels = labels.float()
        bce_loss = self._bce(claim_logits, labels)

        edge_logits = getattr(self.model, "latest_edge_logits", None)
        edge_labels = getattr(self.model, "latest_edge_labels", None)
        if edge_logits is None or edge_labels is None:
            evidence_loss = torch.zeros((), device=claim_logits.device, dtype=claim_logits.dtype)
        else:
            evidence_loss = self._bce(edge_logits.view(-1), edge_labels.float().view(-1))
        evidence_loss = evidence_loss * self.lambda_evidence

        context = getattr(self.model, "latest_component5_context", {}) or {}
        mean_rel_a = self._context_value(context, "mean_rel_a", 0.0)
        max_contra_pool = self._context_value(context, "max_contra_pool", 0.0)
        max_contra_used = self._context_value(context, "max_contra_used", 0.0)
        prob_restored = self._context_value(context, "prob_restored", 0.0)
        prob_masked = self._context_value(context, "prob_masked", 0.0)

        total_loss, breakdown = compute_component5_joint_loss(
            loss_bce=bce_loss,
            loss_evidence=evidence_loss,
            mean_rel_a=mean_rel_a,
            max_contra_pool=max_contra_pool,
            max_contra_used=max_contra_used,
            prob_restored=prob_restored,
            prob_masked=prob_masked,
            lambda_starv=self.lambda_starv,
            tau_starv=self.tau_starv,
            lambda_counter=self.lambda_counter,
            lambda_recovery=self.lambda_recovery,
            enabled_terms=self.enabled_terms,
        )

        self.last_breakdown = breakdown
        self.last_bce = breakdown.loss_bce
        self.last_evidence = breakdown.loss_evidence
        self.last_starvation = breakdown.loss_starvation
        self.last_counter = breakdown.loss_counter
        self.last_recovery = breakdown.loss_recovery
        self.last_total = breakdown.total_loss
        return total_loss
