"""Rule-based S->A backtracking controller for Component 4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Mapping, Sequence

try:
    from ._graph_utils import (
        connected_components as _connected_components,
        edge_connects_components as _edge_connects_components,
        is_disconnected_graph,
        triple_evidence_id as _triple_evidence_id,
        triple_p_ent as _triple_p_ent,
        triple_pool as _triple_pool,
        triple_rel as _triple_rel,
        triple_with_pool as _triple_with_pool,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from component3._graph_utils import (
        connected_components as _connected_components,
        edge_connects_components as _edge_connects_components,
        is_disconnected_graph,
        triple_evidence_id as _triple_evidence_id,
        triple_p_ent as _triple_p_ent,
        triple_pool as _triple_pool,
        triple_rel as _triple_rel,
        triple_with_pool as _triple_with_pool,
    )


def top2_margin(probabilities: Sequence[float]) -> float:
    probs = [float(value) for value in probabilities]
    if not probs:
        return 1.0
    if len(probs) == 1:
        p = min(1.0, max(0.0, probs[0]))
        return abs((2.0 * p) - 1.0)
    top_two = sorted(probs, reverse=True)[:2]
    return float(max(0.0, top_two[0] - top_two[1]))


def margin_from_prediction(prediction: Mapping[str, Any]) -> float:
    if "margin" in prediction and prediction["margin"] is not None:
        return float(prediction["margin"])

    if "max_prob" in prediction and "second_prob" in prediction:
        return float(prediction["max_prob"]) - float(prediction["second_prob"])

    probabilities = prediction.get("probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        return top2_margin(probabilities)

    if "logit" in prediction:
        logit = float(prediction["logit"])
        prob = 1.0 / (1.0 + math.exp(-logit))
        return top2_margin([prob, 1.0 - prob])

    return 1.0


def high_rel_active_count(active_triples: Sequence[Any], rel_threshold: float = 0.3) -> int:
    threshold = float(rel_threshold)
    return sum(
        1
        for triple in active_triples
        if _triple_pool(triple) == "A" and _triple_rel(triple) > threshold
    )


@dataclass(frozen=True)
class RankedSuspendedTriple:
    """Ranked suspended triple candidate for S->A promotion."""

    evidence_id: str
    connects_components: bool
    bridge_bonus: float
    rel: float
    salience_x_pv: float


@dataclass(frozen=True)
class TriggerDecision:
    """Per-round trigger decomposition for diagnostics."""

    should_trigger: bool
    margin: float
    low_confidence: bool
    low_connectivity: bool
    evidence_hunger: bool
    active_high_rel_count: int


@dataclass(frozen=True)
class BacktrackingAction:
    """One S->A promotion round."""

    round_index: int
    decision: TriggerDecision
    selected_evidence_ids: tuple[str, ...]
    margin_before: float
    margin_after: float
    prediction_before: Any
    prediction_after: Any
    ranked_candidates: tuple[RankedSuspendedTriple, ...] = ()


@dataclass(frozen=True)
class BacktrackingResult:
    """Full rule-based backtracking run output."""

    claim_id: str
    triggered: bool
    rounds_run: int
    initial_prediction: Any
    final_prediction: Any
    promoted_evidence_ids: tuple[str, ...]
    actions: tuple[BacktrackingAction, ...]


class RuleBasedBacktrackingController:
    """Component 4 rule-based controller with bounded S-only recovery."""

    def __init__(
        self,
        *,
        max_backtrack_rounds: int = 2,
        backtrack_k: int = 3,
        margin_threshold: float = 0.15,
        min_a: int = 5,
        rel_threshold: float = 0.3,
    ) -> None:
        if max_backtrack_rounds < 1 or max_backtrack_rounds > 2:
            raise ValueError("`max_backtrack_rounds` must be in [1, 2] for Component 4.")
        if backtrack_k < 1 or backtrack_k > 3:
            raise ValueError("`backtrack_k` must be in [1, 3] for Component 4.")
        if min_a < 1:
            raise ValueError("`min_a` must be >= 1.")
        self.max_backtrack_rounds = int(max_backtrack_rounds)
        self.backtrack_k = int(backtrack_k)
        self.margin_threshold = float(margin_threshold)
        self.min_a = int(min_a)
        self.rel_threshold = float(rel_threshold)

    def evaluate_trigger(
        self,
        *,
        prediction: Mapping[str, Any],
        active_triples: Sequence[Any],
        anchors: Sequence[str] | None = None,
    ) -> TriggerDecision:
        margin = margin_from_prediction(prediction)
        if "is_disconnected" in prediction and prediction["is_disconnected"] is not None:
            disconnected = bool(prediction["is_disconnected"])
        else:
            disconnected = is_disconnected_graph(active_triples, anchors=anchors)
        high_rel_count = high_rel_active_count(active_triples, rel_threshold=self.rel_threshold)

        low_confidence = bool(margin < self.margin_threshold)
        low_connectivity = bool(disconnected)
        evidence_hunger = bool(high_rel_count < self.min_a)

        return TriggerDecision(
            should_trigger=bool(low_confidence or low_connectivity or evidence_hunger),
            margin=float(margin),
            low_confidence=low_confidence,
            low_connectivity=low_connectivity,
            evidence_hunger=evidence_hunger,
            active_high_rel_count=int(high_rel_count),
        )

    def rank_suspended(
        self,
        *,
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        bridge_bonus_by_id: Mapping[str, float] | None = None,
        salience_by_id: Mapping[str, float] | None = None,
    ) -> list[RankedSuspendedTriple]:
        bridge_bonus_by_id = bridge_bonus_by_id or {}
        salience_by_id = salience_by_id or {}
        node_to_component = _connected_components(active_triples)
        rows: list[tuple[Any, RankedSuspendedTriple]] = []

        for idx, triple in enumerate(suspended_triples):
            if _triple_pool(triple) != "S":
                continue
            evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
            p_ent = _triple_p_ent(triple)
            salience = float(salience_by_id.get(evidence_id, 0.0))
            row = RankedSuspendedTriple(
                evidence_id=evidence_id,
                connects_components=_edge_connects_components(triple, node_to_component),
                bridge_bonus=float(bridge_bonus_by_id.get(evidence_id, 0.0)),
                rel=_triple_rel(triple),
                salience_x_pv=salience * p_ent,
            )
            sort_key = (
                row.connects_components,  # 1) bridge component connector
                row.bridge_bonus,         # 2) PPR bridge bonus
                row.rel,                  # 3) rel tie-breaker
                row.salience_x_pv,        # 4) salience x PV
                row.evidence_id,
            )
            rows.append((sort_key, row))

        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows]

    def _record_attempts(
        self,
        *,
        memory_store: Any | None,
        claim_id: str,
        ranked_rows: Sequence[RankedSuspendedTriple],
        selected_ids: set[str],
        round_index: int,
    ) -> None:
        if memory_store is None:
            return
        for rank, row in enumerate(ranked_rows, start=1):
            memory_store.record_recovery_attempt(
                claim_id=claim_id,
                evidence_id=row.evidence_id,
                round_index=round_index,
                selected=(row.evidence_id in selected_ids),
                bridge_score_rank=rank,
                bridge_score_value=row.bridge_bonus,
                note="rule_based_backtracking",
            )

    def run(
        self,
        *,
        claim_id: str,
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        predictor: Callable[[Sequence[Any]], Dict[str, Any]],
        anchors: Sequence[str] | None = None,
        rebuild_graph_fn: Callable[[Sequence[Any]], Any] | None = None,
        rerun_model_fn: Callable[[Any], Dict[str, Any]] | None = None,
        memory_store: Any | None = None,
    ) -> BacktrackingResult:
        current_active = list(active_triples)
        remaining_s = [triple for triple in suspended_triples if _triple_pool(triple) == "S"]
        promoted_ids: list[str] = []
        actions: list[BacktrackingAction] = []

        prediction = predictor(current_active)
        initial_prediction = dict(prediction)

        rounds_run = 0
        for round_idx in range(1, self.max_backtrack_rounds + 1):
            prediction_before = dict(prediction)
            decision = self.evaluate_trigger(
                prediction=prediction_before,
                active_triples=current_active,
                anchors=anchors,
            )
            if not decision.should_trigger:
                break
            if not remaining_s:
                break

            ranked_rows = self.rank_suspended(
                active_triples=current_active,
                suspended_triples=remaining_s,
                bridge_bonus_by_id=prediction_before.get("bridge_bonus_by_id", {}),
                salience_by_id=prediction_before.get("salience_by_id", {}),
            )
            if not ranked_rows:
                break
            selected_rows = ranked_rows[: self.backtrack_k]
            selected_ids = {row.evidence_id for row in selected_rows}
            if not selected_ids:
                break
            # Conservative execution gate: only promote if the selected set
            # contains at least one candidate that bridges disconnected components.
            if not any(row.connects_components for row in selected_rows):
                self._record_attempts(
                    memory_store=memory_store,
                    claim_id=str(claim_id),
                    ranked_rows=ranked_rows,
                    selected_ids=set(),
                    round_index=round_idx,
                )
                break

            self._record_attempts(
                memory_store=memory_store,
                claim_id=str(claim_id),
                ranked_rows=ranked_rows,
                selected_ids=selected_ids,
                round_index=round_idx,
            )

            next_remaining_s: list[Any] = []
            for idx, triple in enumerate(remaining_s):
                evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
                if evidence_id in selected_ids:
                    current_active.append(_triple_with_pool(triple, "A"))
                    promoted_ids.append(evidence_id)
                    if memory_store is not None:
                        memory_store.promote_s_to_active(
                            claim_id=str(claim_id),
                            evidence_id=evidence_id,
                            round_index=round_idx,
                        )
                else:
                    next_remaining_s.append(triple)
            remaining_s = next_remaining_s

            rounds_run = round_idx
            if rebuild_graph_fn is not None and rerun_model_fn is not None:
                graph_obj = rebuild_graph_fn(current_active)
                prediction = rerun_model_fn(graph_obj)
            else:
                prediction = predictor(current_active)

            margin_after = margin_from_prediction(prediction)
            actions.append(
                BacktrackingAction(
                    round_index=round_idx,
                    decision=decision,
                    selected_evidence_ids=tuple(row.evidence_id for row in selected_rows),
                    margin_before=decision.margin,
                    margin_after=float(margin_after),
                    prediction_before=prediction_before,
                    prediction_after=dict(prediction),
                    ranked_candidates=tuple(ranked_rows),
                )
            )

        return BacktrackingResult(
            claim_id=str(claim_id),
            triggered=(len(actions) > 0),
            rounds_run=rounds_run,
            initial_prediction=initial_prediction,
            final_prediction=dict(prediction),
            promoted_evidence_ids=tuple(promoted_ids),
            actions=tuple(actions),
        )
