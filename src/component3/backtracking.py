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
        triple_p_con as _triple_p_con,
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
        triple_p_con as _triple_p_con,
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


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    q = min(1.0, max(0.0, float(q)))
    idx = int(round((len(sorted_values) - 1) * q))
    idx = min(len(sorted_values) - 1, max(0, idx))
    return float(sorted_values[idx])


def _prediction_support_lean(prediction: Mapping[str, Any] | None) -> bool:
    if prediction is None:
        return True

    pred_value = prediction.get("prediction")
    if isinstance(pred_value, str):
        normalized = pred_value.strip().lower()
        if "support" in normalized:
            return True
        if "refut" in normalized or "contrad" in normalized:
            return False

    explicit_pred = prediction.get("pred")
    if explicit_pred is not None:
        try:
            return int(explicit_pred) == 1
        except (TypeError, ValueError):
            pass

    if "logit" in prediction and prediction["logit"] is not None:
        return float(prediction["logit"]) >= 0.0

    probabilities = prediction.get("probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        probs = [float(v) for v in probabilities]
        if len(probs) >= 2:
            return probs[0] >= probs[1]
        if len(probs) == 1:
            return probs[0] >= 0.5

    return True


def _prediction_binary_probability(prediction: Mapping[str, Any] | None) -> float | None:
    """Best-effort class-1 probability extraction for binary outputs."""

    if prediction is None:
        return None

    probabilities = prediction.get("probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        probs = [float(v) for v in probabilities]
        if len(probs) >= 2:
            return float(probs[0])
        if len(probs) == 1:
            return float(probs[0])

    if "prob" in prediction and prediction["prob"] is not None:
        return float(prediction["prob"])

    if "confidence" in prediction and prediction["confidence"] is not None:
        confidence = float(prediction["confidence"])
        pred_value = prediction.get("pred")
        if pred_value is None:
            pred_is_support = _prediction_support_lean(prediction)
        else:
            try:
                pred_is_support = int(pred_value) == 1
            except (TypeError, ValueError):
                pred_is_support = _prediction_support_lean(prediction)
        return float(confidence if pred_is_support else (1.0 - confidence))

    if "logit" in prediction and prediction["logit"] is not None:
        logit = float(prediction["logit"])
        return float(1.0 / (1.0 + math.exp(-logit)))

    return None


def _prediction_logit(prediction: Mapping[str, Any] | None) -> float:
    """Best-effort binary logit extraction for flip-confidence gating."""

    if prediction is None:
        return 0.0
    if "logit" in prediction and prediction["logit"] is not None:
        return float(prediction["logit"])

    prob = _prediction_binary_probability(prediction)
    if prob is None:
        return 0.0
    clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
    return float(math.log(clipped / (1.0 - clipped)))


@dataclass(frozen=True)
class RankedSuspendedTriple:
    """Ranked suspended triple candidate for S->A promotion."""

    evidence_id: str
    connects_components: bool
    bridge_bonus: float
    rel: float
    salience_x_pv: float
    p_ent: float
    p_con: float
    pv_confidence: float
    semantic_score: float
    bridge_semantic_score: float


@dataclass(frozen=True)
class TriggerDecision:
    """Per-round trigger decomposition for diagnostics."""

    should_trigger: bool
    margin: float
    low_confidence: bool
    low_connectivity: bool
    evidence_hunger: bool
    active_high_rel_count: int
    active_high_rel_required: int
    hunger_rel_threshold: float
    hunger_mode: str


@dataclass(frozen=True)
class BacktrackingAction:
    """One S->A promotion round."""

    round_index: int
    decision: TriggerDecision
    selected_evidence_ids: tuple[str, ...]
    margin_before: float
    tentative_margin: float
    margin_after: float
    prediction_before: Any
    tentative_prediction: Any
    prediction_after: Any
    recovery_mode: str = "bridge"
    selection_mode: str = "top_k"
    do_no_harm_reverted: bool = False
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
        hunger_mode: str = "percentile",
        hunger_rel_percentile: float = 0.9,
        directional_delta: float = 0.0,
        challenge_mode: bool = False,
        enable_do_no_harm_gate: bool = True,
        do_no_harm_margin_eps: float = 0.002,
        flip_conf_min: float = 0.1,
        flip_abslogit_eps: float = 0.002,
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
        normalized_hunger_mode = str(hunger_mode).strip().lower()
        if normalized_hunger_mode not in {"absolute", "percentile"}:
            raise ValueError("`hunger_mode` must be `absolute` or `percentile`.")
        if hunger_rel_percentile <= 0.0 or hunger_rel_percentile > 1.0:
            raise ValueError("`hunger_rel_percentile` must be in (0, 1].")
        if directional_delta < 0.0:
            raise ValueError("`directional_delta` must be >= 0.")
        if do_no_harm_margin_eps < 0.0:
            raise ValueError("`do_no_harm_margin_eps` must be >= 0.")
        if flip_conf_min < 0.0:
            raise ValueError("`flip_conf_min` must be >= 0.")
        if flip_abslogit_eps < 0.0:
            raise ValueError("`flip_abslogit_eps` must be >= 0.")
        self.hunger_mode = normalized_hunger_mode
        self.hunger_rel_percentile = float(hunger_rel_percentile)
        self.directional_delta = float(directional_delta)
        self.challenge_mode = bool(challenge_mode)
        self.enable_do_no_harm_gate = bool(enable_do_no_harm_gate)
        self.do_no_harm_margin_eps = float(do_no_harm_margin_eps)
        self.flip_conf_min = float(flip_conf_min)
        self.flip_abslogit_eps = float(flip_abslogit_eps)

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
        active_rels = [
            _triple_rel(triple)
            for triple in active_triples
            if _triple_pool(triple) == "A"
        ]
        if self.hunger_mode == "percentile":
            hunger_threshold = _quantile(active_rels, self.hunger_rel_percentile) if active_rels else 0.0
            high_rel_count = sum(1 for rel in active_rels if rel >= hunger_threshold)
            # In percentile mode the eligible slice is claim-local; scale required count by graph size.
            required_high_rel = min(
                self.min_a,
                max(1, int(math.ceil(len(active_rels) * (1.0 - self.hunger_rel_percentile)))),
            ) if active_rels else self.min_a
        else:
            hunger_threshold = float(self.rel_threshold)
            high_rel_count = sum(1 for rel in active_rels if rel > hunger_threshold)
            required_high_rel = self.min_a

        low_confidence = bool(margin < self.margin_threshold)
        low_connectivity = bool(disconnected)
        evidence_hunger = bool(high_rel_count < required_high_rel)

        return TriggerDecision(
            should_trigger=bool(low_confidence or low_connectivity or evidence_hunger),
            margin=float(margin),
            low_confidence=low_confidence,
            low_connectivity=low_connectivity,
            evidence_hunger=evidence_hunger,
            active_high_rel_count=int(high_rel_count),
            active_high_rel_required=int(required_high_rel),
            hunger_rel_threshold=float(hunger_threshold),
            hunger_mode=self.hunger_mode,
        )

    def rank_suspended(
        self,
        *,
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        bridge_bonus_by_id: Mapping[str, float] | None = None,
        salience_by_id: Mapping[str, float] | None = None,
        prediction: Mapping[str, Any] | None = None,
    ) -> list[RankedSuspendedTriple]:
        bridge_bonus_by_id = bridge_bonus_by_id or {}
        salience_by_id = salience_by_id or {}
        support_lean = _prediction_support_lean(prediction)
        node_to_component = _connected_components(active_triples)
        rows: list[tuple[Any, RankedSuspendedTriple]] = []

        for idx, triple in enumerate(suspended_triples):
            if _triple_pool(triple) != "S":
                continue
            evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
            p_ent = _triple_p_ent(triple)
            p_con = _triple_p_con(triple)
            pv_confidence = max(float(p_ent), float(p_con))
            salience = float(salience_by_id.get(evidence_id, 0.0))
            semantic_score = float(p_ent) if support_lean else float(p_con)
            bridge_bonus = float(bridge_bonus_by_id.get(evidence_id, 0.0))
            bridge_semantic_score = bridge_bonus * (0.5 + pv_confidence)
            row = RankedSuspendedTriple(
                evidence_id=evidence_id,
                connects_components=_edge_connects_components(triple, node_to_component),
                bridge_bonus=bridge_bonus,
                rel=_triple_rel(triple),
                salience_x_pv=salience * p_ent,
                p_ent=float(p_ent),
                p_con=float(p_con),
                pv_confidence=float(pv_confidence),
                semantic_score=float(semantic_score),
                bridge_semantic_score=float(bridge_semantic_score),
            )
            sort_key = (
                row.connects_components,  # 1) bridge component connector
                row.bridge_semantic_score,  # 2) bridge bonus weighted by PV confidence
                row.semantic_score,         # 3) class-lean semantic tie-breaker
                row.rel,                    # 4) rel tie-breaker
                row.pv_confidence,          # 5) PV confidence
                row.salience_x_pv,          # 6) salience x PV
                row.evidence_id,
            )
            rows.append((sort_key, row))

        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows]

    def _rank_rows_for_recovery_mode(
        self,
        *,
        ranked_rows: Sequence[RankedSuspendedTriple],
        recovery_mode: str,
        support_lean: bool,
    ) -> list[RankedSuspendedTriple]:
        mode = str(recovery_mode).strip().lower()
        if mode == "bridge":
            return list(ranked_rows)

        if mode != "polarity":
            raise ValueError(f"Unsupported recovery mode: {recovery_mode}")

        def _polarity_key(row: RankedSuspendedTriple) -> tuple[Any, ...]:
            opposite_score = float(row.p_con) if support_lean else float(row.p_ent)
            # In connected-mode recovery we prioritize opposite-polarity evidence;
            # connectivity remains a soft bonus, not a hard gate.
            return (
                opposite_score,
                float(row.rel),
                float(row.pv_confidence),
                bool(row.connects_components),
                float(row.bridge_semantic_score),
                row.evidence_id,
            )

        return sorted(ranked_rows, key=_polarity_key, reverse=True)

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

        def _apply_selected_ids(
            *,
            base_active: Sequence[Any],
            base_remaining_s: Sequence[Any],
            selected_ids: set[str],
        ) -> tuple[list[Any], list[Any], tuple[str, ...]]:
            next_active = list(base_active)
            next_remaining_s: list[Any] = []
            promoted_in_order: list[str] = []
            for idx, triple in enumerate(base_remaining_s):
                evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
                if evidence_id in selected_ids:
                    promoted_triple = _triple_with_pool(triple, "A")
                    if isinstance(promoted_triple, dict):
                        promoted_triple["is_promoted"] = True
                    next_active.append(promoted_triple)
                    promoted_in_order.append(evidence_id)
                else:
                    next_remaining_s.append(triple)
            return next_active, next_remaining_s, tuple(promoted_in_order)

        def _predict_from_active(rows: Sequence[Any]) -> Dict[str, Any]:
            if rebuild_graph_fn is not None and rerun_model_fn is not None:
                graph_obj = rebuild_graph_fn(rows)
                return dict(rerun_model_fn(graph_obj))
            return dict(predictor(rows))

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

            support_lean = _prediction_support_lean(prediction_before)
            recovery_mode = "bridge" if bool(decision.low_connectivity) else "polarity"
            ranked_rows_base = self.rank_suspended(
                active_triples=current_active,
                suspended_triples=remaining_s,
                bridge_bonus_by_id=prediction_before.get("bridge_bonus_by_id", {}),
                salience_by_id=prediction_before.get("salience_by_id", {}),
                prediction=prediction_before,
            )
            ranked_rows = self._rank_rows_for_recovery_mode(
                ranked_rows=ranked_rows_base,
                recovery_mode=recovery_mode,
                support_lean=support_lean,
            )
            if not ranked_rows:
                break
            directional_delta = float(self.directional_delta)
            if directional_delta > 0.0:
                if recovery_mode == "polarity":
                    if support_lean:
                        eligible_rows = [
                            row for row in ranked_rows
                            if float(row.p_con - row.p_ent) >= directional_delta
                        ]
                    else:
                        eligible_rows = [
                            row for row in ranked_rows
                            if float(row.p_ent - row.p_con) >= directional_delta
                        ]
                else:
                    if support_lean:
                        eligible_rows = [
                            row for row in ranked_rows
                            if float(row.p_ent - row.p_con) >= directional_delta
                        ]
                    else:
                        eligible_rows = [
                            row for row in ranked_rows
                            if float(row.p_con - row.p_ent) >= directional_delta
                        ]
            else:
                eligible_rows = ranked_rows
            selected_mode = "top_k"
            challenge_precomputed: tuple[list[Any], list[Any], tuple[str, ...], Dict[str, Any]] | None = None
            if self.challenge_mode and recovery_mode == "polarity" and eligible_rows:
                support_row = max(
                    eligible_rows,
                    key=lambda row: (float(row.p_ent), float(row.rel), float(row.pv_confidence), row.evidence_id),
                )
                refute_row = max(
                    eligible_rows,
                    key=lambda row: (float(row.p_con), float(row.rel), float(row.pv_confidence), row.evidence_id),
                )
                challenge_options: list[tuple[str, RankedSuspendedTriple]] = [("support", support_row)]
                if refute_row.evidence_id != support_row.evidence_id:
                    challenge_options.append(("refute", refute_row))

                challenge_scored: list[
                    tuple[
                        float,
                        float,
                        str,
                        RankedSuspendedTriple,
                        list[Any],
                        list[Any],
                        tuple[str, ...],
                        Dict[str, Any],
                    ]
                ] = []
                for tag, row in challenge_options:
                    ids = {row.evidence_id}
                    candidate_active_opt, candidate_remaining_s_opt, candidate_promoted_opt = _apply_selected_ids(
                        base_active=current_active,
                        base_remaining_s=remaining_s,
                        selected_ids=ids,
                    )
                    candidate_prediction_opt = _predict_from_active(candidate_active_opt)
                    challenge_scored.append(
                        (
                            abs(float(_prediction_logit(candidate_prediction_opt))),
                            float(row.rel),
                            tag,
                            row,
                            candidate_active_opt,
                            candidate_remaining_s_opt,
                            candidate_promoted_opt,
                            dict(candidate_prediction_opt),
                        )
                    )
                challenge_scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
                _abs_logit, _rel_value, best_tag, best_row, best_active, best_remaining, best_promoted, best_prediction = challenge_scored[0]
                selected_mode = f"challenge_{best_tag}"
                challenge_precomputed = (
                    list(best_active),
                    list(best_remaining),
                    tuple(best_promoted),
                    dict(best_prediction),
                )
                selected_rows = [best_row]
            else:
                selected_rows = eligible_rows[: self.backtrack_k]
            selected_ids = {row.evidence_id for row in selected_rows}
            if not selected_ids:
                self._record_attempts(
                    memory_store=memory_store,
                    claim_id=str(claim_id),
                    ranked_rows=ranked_rows,
                    selected_ids=set(),
                    round_index=round_idx,
                )
                break
            # Conservative bridge gate only applies when graph connectivity is the trigger.
            if recovery_mode == "bridge" and not any(row.connects_components for row in selected_rows):
                self._record_attempts(
                    memory_store=memory_store,
                    claim_id=str(claim_id),
                    ranked_rows=ranked_rows,
                    selected_ids=set(),
                    round_index=round_idx,
                )
                break

            selected_ids_final = set(selected_ids)
            selected_rows_final = list(selected_rows)

            def _passes_do_no_harm_gate(
                *,
                before_prediction: Mapping[str, Any],
                candidate_prediction: Mapping[str, Any],
                margin_before: float,
            ) -> bool:
                if not self.enable_do_no_harm_gate:
                    return True

                before_support = _prediction_support_lean(before_prediction)
                candidate_support = _prediction_support_lean(candidate_prediction)
                if bool(candidate_support == before_support):
                    candidate_margin = margin_from_prediction(candidate_prediction)
                    return bool(
                        float(candidate_margin) >= float(margin_before + self.do_no_harm_margin_eps)
                    )

                before_logit = _prediction_logit(before_prediction)
                candidate_logit = _prediction_logit(candidate_prediction)
                return bool(
                    abs(float(candidate_logit))
                    >= (abs(float(before_logit)) + float(self.flip_abslogit_eps))
                )

            if challenge_precomputed is None:
                candidate_active, candidate_remaining_s, candidate_promoted = _apply_selected_ids(
                    base_active=current_active,
                    base_remaining_s=remaining_s,
                    selected_ids=selected_ids_final,
                )
                candidate_prediction = _predict_from_active(candidate_active)
            else:
                candidate_active, candidate_remaining_s, candidate_promoted, candidate_prediction = challenge_precomputed
            candidate_margin = float(margin_from_prediction(candidate_prediction))

            do_no_harm_reverted = False
            tentative_prediction = dict(candidate_prediction)
            tentative_margin = float(candidate_margin)
            if not _passes_do_no_harm_gate(
                before_prediction=prediction_before,
                candidate_prediction=candidate_prediction,
                margin_before=float(decision.margin),
            ):
                if len(selected_rows) > 1:
                    fallback_rows = selected_rows[:1]
                    fallback_ids = {row.evidence_id for row in fallback_rows}
                    fallback_active, fallback_remaining_s, fallback_promoted = _apply_selected_ids(
                        base_active=current_active,
                        base_remaining_s=remaining_s,
                        selected_ids=fallback_ids,
                    )
                    fallback_prediction = _predict_from_active(fallback_active)
                    fallback_margin = float(margin_from_prediction(fallback_prediction))
                    tentative_prediction = dict(fallback_prediction)
                    tentative_margin = float(fallback_margin)
                    if _passes_do_no_harm_gate(
                        before_prediction=prediction_before,
                        candidate_prediction=fallback_prediction,
                        margin_before=float(decision.margin),
                    ):
                        selected_mode = "top1_fallback"
                        selected_ids_final = set(fallback_ids)
                        selected_rows_final = list(fallback_rows)
                        candidate_active = fallback_active
                        candidate_remaining_s = fallback_remaining_s
                        candidate_promoted = fallback_promoted
                        candidate_prediction = dict(fallback_prediction)
                    else:
                        do_no_harm_reverted = True
                        selected_mode = "reverted"
                        selected_ids_final = set()
                        selected_rows_final = []
                else:
                    do_no_harm_reverted = True
                    selected_mode = "reverted"
                    selected_ids_final = set()
                    selected_rows_final = []

            if do_no_harm_reverted:
                self._record_attempts(
                    memory_store=memory_store,
                    claim_id=str(claim_id),
                    ranked_rows=ranked_rows,
                    selected_ids=set(),
                    round_index=round_idx,
                )
                rounds_run = round_idx
                actions.append(
                    BacktrackingAction(
                        round_index=round_idx,
                        decision=decision,
                        selected_evidence_ids=tuple(),
                        margin_before=decision.margin,
                        tentative_margin=float(tentative_margin),
                        margin_after=float(decision.margin),
                        prediction_before=prediction_before,
                        tentative_prediction=dict(tentative_prediction),
                        prediction_after=dict(prediction_before),
                        recovery_mode=str(recovery_mode),
                        selection_mode=selected_mode,
                        do_no_harm_reverted=True,
                        ranked_candidates=tuple(ranked_rows),
                    )
                )
                break

            self._record_attempts(
                memory_store=memory_store,
                claim_id=str(claim_id),
                ranked_rows=ranked_rows,
                selected_ids=selected_ids_final,
                round_index=round_idx,
            )

            current_active = list(candidate_active)
            remaining_s = list(candidate_remaining_s)
            prediction = dict(candidate_prediction)
            promoted_ids.extend(candidate_promoted)
            for evidence_id in candidate_promoted:
                if memory_store is not None:
                    memory_store.promote_s_to_active(
                        claim_id=str(claim_id),
                        evidence_id=evidence_id,
                        round_index=round_idx,
                    )

            rounds_run = round_idx
            margin_after = margin_from_prediction(prediction)
            actions.append(
                BacktrackingAction(
                    round_index=round_idx,
                    decision=decision,
                    selected_evidence_ids=tuple(row.evidence_id for row in selected_rows_final),
                    margin_before=decision.margin,
                    tentative_margin=float(tentative_margin),
                    margin_after=float(margin_after),
                    prediction_before=prediction_before,
                    tentative_prediction=dict(tentative_prediction),
                    prediction_after=dict(prediction),
                    recovery_mode=str(recovery_mode),
                    selection_mode=selected_mode,
                    do_no_harm_reverted=False,
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
