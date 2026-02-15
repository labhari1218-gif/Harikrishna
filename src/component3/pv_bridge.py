"""Bridge-rescue integration for Component 3 PV-QA-GNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

try:
    from ._graph_utils import (
        connected_components as _connected_components,
        edge_connects_components as _edge_connects_components,
        is_disconnected_graph as _is_disconnected,
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
        is_disconnected_graph as _is_disconnected,
        triple_evidence_id as _triple_evidence_id,
        triple_p_ent as _triple_p_ent,
        triple_pool as _triple_pool,
        triple_rel as _triple_rel,
        triple_with_pool as _triple_with_pool,
    )


def should_trigger_bridge_rescue(
    *,
    margin: float,
    is_disconnected: bool,
    margin_threshold: float = 0.15,
) -> bool:
    """Trigger only when confidence is low and graph is structurally deficient."""

    return bool(float(margin) < float(margin_threshold) and bool(is_disconnected))


@dataclass(frozen=True)
class RankedCandidate:
    """Ranked suspended candidate with diagnostics."""

    evidence_id: str
    connects_components: bool
    bridge_bonus: float
    rel: float
    salience_x_pv: float


@dataclass(frozen=True)
class RecoveryAction:
    """One backtracking action and resulting prediction summary."""

    round_index: int
    selected_evidence_ids: tuple[str, ...]
    margin_before: float
    margin_after: float
    disconnected_before: bool
    disconnected_after: bool


@dataclass(frozen=True)
class BridgeRunResult:
    """Bridge rescue run result with action trace."""

    claim_id: str
    triggered: bool
    rounds_run: int
    initial_prediction: Any
    final_prediction: Any
    promoted_evidence_ids: tuple[str, ...]
    actions: tuple[RecoveryAction, ...]


class PVBridgeRecoveryEngine:
    """S-only ranked bridge recovery with bounded rounds and budget."""

    def __init__(
        self,
        *,
        max_backtrack_rounds: int = 2,
        backtrack_k: int = 3,
        margin_threshold: float = 0.15,
    ) -> None:
        if max_backtrack_rounds < 1 or max_backtrack_rounds > 2:
            raise ValueError("`max_backtrack_rounds` must be in [1, 2] for Component 3.")
        if backtrack_k < 1 or backtrack_k > 3:
            raise ValueError("`backtrack_k` must be in [1, 3] for Component 3.")
        self.max_backtrack_rounds = int(max_backtrack_rounds)
        self.backtrack_k = int(backtrack_k)
        self.margin_threshold = float(margin_threshold)

        # Component 3 wrapper around Component 2 bridge-rescue stack.
        try:
            from component2.bridge_rescue import BridgeRescuePPR
            from component2.config import DEFAULT_CONFIG
            from component2.recovery import BridgeRecoveryPolicy

            self.bridge_rescue = BridgeRescuePPR(config=DEFAULT_CONFIG)
            self.recovery_policy = BridgeRecoveryPolicy(config=DEFAULT_CONFIG)
        except ModuleNotFoundError:
            self.bridge_rescue = None
            self.recovery_policy = None

    def rank_suspended_candidates(
        self,
        *,
        active_triples: Sequence[Any],
        suspended_triples: Sequence[Any],
        bridge_bonus_by_id: Dict[str, float] | None = None,
        salience_by_id: Dict[str, float] | None = None,
        k: int | None = None,
    ) -> list[RankedCandidate]:
        """Rank suspended candidates with canonical bridge-first ordering."""

        bridge_bonus_by_id = bridge_bonus_by_id or {}
        salience_by_id = salience_by_id or {}
        limit = int(k or self.backtrack_k)
        node_to_component = _connected_components(active_triples)

        rows: list[tuple[Any, RankedCandidate]] = []
        for idx, triple in enumerate(suspended_triples):
            if _triple_pool(triple) != "S":
                continue
            evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
            p_ent = _triple_p_ent(triple)
            salience = float(salience_by_id.get(evidence_id, 0.0))
            salience_x_pv = salience * p_ent
            row = RankedCandidate(
                evidence_id=evidence_id,
                connects_components=_edge_connects_components(triple, node_to_component),
                bridge_bonus=float(bridge_bonus_by_id.get(evidence_id, 0.0)),
                rel=_triple_rel(triple),
                salience_x_pv=salience_x_pv,
            )
            sort_key = (
                row.connects_components,  # 1) bridge component connector
                row.bridge_bonus,         # 2) PPR bridge bonus
                row.rel,                 # 3) rel tie-breaker (p_ent + p_con)
                row.salience_x_pv,       # 4) C3 post-GNN factor
                row.evidence_id,
            )
            rows.append((sort_key, row))

        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows[:limit]]

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
    ) -> BridgeRunResult:
        """
        Run bounded S->A bridge recovery rounds.

        `predictor` should return a dict containing at least:
          - `margin` (float)
          - optionally `is_disconnected` (bool)
          - optionally `bridge_bonus_by_id` (dict evidence_id -> float)
          - optionally `salience_by_id` (dict evidence_id -> float)
        """

        current_active = list(active_triples)
        remaining_s = [triple for triple in suspended_triples if _triple_pool(triple) == "S"]
        promoted_ids: list[str] = []
        actions: list[RecoveryAction] = []

        prediction = predictor(current_active)
        initial_prediction = dict(prediction)

        rounds_run = 0
        for round_idx in range(self.max_backtrack_rounds):
            margin_before = float(prediction.get("margin", 1.0))
            disconnected_before = bool(
                prediction.get("is_disconnected", _is_disconnected(current_active, anchors=anchors))
            )
            trigger = should_trigger_bridge_rescue(
                margin=margin_before,
                is_disconnected=disconnected_before,
                margin_threshold=self.margin_threshold,
            )
            if not trigger:
                break

            ranked = self.rank_suspended_candidates(
                active_triples=current_active,
                suspended_triples=remaining_s,
                bridge_bonus_by_id=prediction.get("bridge_bonus_by_id", {}),
                salience_by_id=prediction.get("salience_by_id", {}),
                k=self.backtrack_k,
            )
            if not ranked:
                break

            selected_ids = tuple(row.evidence_id for row in ranked)
            if not selected_ids:
                break

            selected_set = set(selected_ids)
            next_remaining_s: list[Any] = []
            for triple in remaining_s:
                evidence_id = _triple_evidence_id(triple)
                if evidence_id in selected_set:
                    current_active.append(_triple_with_pool(triple, "A"))
                    promoted_ids.append(evidence_id)
                else:
                    next_remaining_s.append(triple)
            remaining_s = next_remaining_s

            rounds_run = round_idx + 1
            if rebuild_graph_fn is not None and rerun_model_fn is not None:
                graph_obj = rebuild_graph_fn(current_active)
                prediction = rerun_model_fn(graph_obj)
            else:
                prediction = predictor(current_active)

            margin_after = float(prediction.get("margin", 1.0))
            disconnected_after = bool(
                prediction.get("is_disconnected", _is_disconnected(current_active, anchors=anchors))
            )
            actions.append(
                RecoveryAction(
                    round_index=rounds_run,
                    selected_evidence_ids=selected_ids,
                    margin_before=margin_before,
                    margin_after=margin_after,
                    disconnected_before=disconnected_before,
                    disconnected_after=disconnected_after,
                )
            )

        return BridgeRunResult(
            claim_id=claim_id,
            triggered=(len(actions) > 0),
            rounds_run=rounds_run,
            initial_prediction=initial_prediction,
            final_prediction=dict(prediction),
            promoted_evidence_ids=tuple(promoted_ids),
            actions=tuple(actions),
        )
