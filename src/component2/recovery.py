"""Policy-aware recovery logic for Component 2 M2."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

from .bridge_rescue import BridgeScore
from .config import Component2Config, DEFAULT_CONFIG
from .types import EvidenceTriple


def _anchor_connectivity(triples: Sequence[EvidenceTriple], anchors: Sequence[str]) -> float:
    """Fraction of anchor pairs connected in an undirected entity graph."""

    adjacency: Dict[str, set[str]] = defaultdict(set)
    nodes = set()

    for triple in triples:
        if triple.pool != "A":
            continue
        subj, _, obj = triple.raw_triple
        nodes.add(subj)
        nodes.add(obj)
        adjacency[subj].add(obj)
        adjacency[obj].add(subj)

    unique_anchors = list(dict.fromkeys(str(anchor) for anchor in anchors))
    if len(unique_anchors) < 2:
        return 1.0

    def is_connected(u: str, v: str) -> bool:
        if u not in nodes or v not in nodes:
            return False
        if u == v:
            return True
        queue = deque([u])
        visited = {u}
        while queue:
            cur = queue.popleft()
            if cur == v:
                return True
            for nxt in adjacency.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    connected = 0
    total = 0
    for i, u in enumerate(unique_anchors):
        for v in unique_anchors[i + 1 :]:
            total += 1
            if is_connected(u, v):
                connected += 1

    return float(connected) / float(total) if total > 0 else 1.0


def _connected_components(triples: Sequence[EvidenceTriple]) -> Dict[str, int]:
    """Component index per node in Active-set undirected graph."""

    adjacency: Dict[str, set[str]] = defaultdict(set)
    nodes = set()
    for triple in triples:
        if triple.pool != "A":
            continue
        subj, _, obj = triple.raw_triple
        nodes.add(subj)
        nodes.add(obj)
        adjacency[subj].add(obj)
        adjacency[obj].add(subj)

    node_to_comp: Dict[str, int] = {}
    comp_idx = 0
    for node in nodes:
        if node in node_to_comp:
            continue
        queue = deque([node])
        node_to_comp[node] = comp_idx
        while queue:
            cur = queue.popleft()
            for nxt in adjacency.get(cur, ()):
                if nxt not in node_to_comp:
                    node_to_comp[nxt] = comp_idx
                    queue.append(nxt)
        comp_idx += 1
    return node_to_comp


def _edge_connects_components(triple: EvidenceTriple, node_to_comp: Dict[str, int]) -> bool:
    subj, _, obj = triple.raw_triple
    comp_subj = node_to_comp.get(subj)
    comp_obj = node_to_comp.get(obj)
    if comp_subj is None and comp_obj is None:
        return False
    if comp_subj is None or comp_obj is None:
        return True
    return comp_subj != comp_obj


@dataclass(frozen=True)
class RecoveryMetrics:
    """Diagnostics for recovery policy and trigger decisions."""

    conn_a: float
    conn_a_plus_rel: float
    conn_a_plus_bridge: float
    rpi_rel_at_k: float
    rpi_bridge_at_k: float
    delta_conn_bridge_at_k: float
    selected_rel_ids: Tuple[str, ...]
    selected_bridge_ids: Tuple[str, ...]
    should_recover: bool


@dataclass(frozen=True)
class RecoveryResult:
    """Recovery action outcome with updated triples."""

    recovered_triples: Tuple[EvidenceTriple, ...]
    moved_evidence_ids: Tuple[str, ...]


class BridgeRecoveryPolicy:
    """One-step S->A recovery using bridge-aware DeltaConn trigger."""

    def __init__(self, config: Component2Config = DEFAULT_CONFIG):
        self.config = config

    def _select_rel_top_k(self, s_triples: Sequence[EvidenceTriple], k: int) -> List[EvidenceTriple]:
        return sorted(
            s_triples,
            key=lambda triple: (triple.rel, -triple.p_neu, triple.evidence_id),
            reverse=True,
        )[:k]

    def _select_bridge_top_k(
        self,
        s_triples: Sequence[EvidenceTriple],
        bridge_scores: Dict[str, BridgeScore],
        node_to_comp: Dict[str, int],
        k: int,
    ) -> List[EvidenceTriple]:
        ranked = []
        for triple in s_triples:
            score = bridge_scores.get(triple.evidence_id)
            bridge_bonus = score.bridge_bonus_capped if score is not None else 0.0
            connects_components = _edge_connects_components(triple, node_to_comp)
            ranked.append((connects_components, bridge_bonus, triple.rel, triple.evidence_id, triple))

        ranked.sort(reverse=True)
        return [row[-1] for row in ranked[:k]]

    def evaluate(
        self,
        triples: Sequence[EvidenceTriple],
        anchors: Sequence[str],
        bridge_scores: Sequence[BridgeScore],
        esi_geom: float,
    ) -> RecoveryMetrics:
        """Evaluate rel-based vs bridge-based recovery and trigger decision."""

        k = int(self.config.recovery_top_k)
        s_triples = [triple for triple in triples if triple.pool == "S"]
        conn_a = _anchor_connectivity(triples, anchors)

        bridge_by_id = {row.evidence_id: row for row in bridge_scores if row.pool == "S"}
        node_to_comp = _connected_components(triples)

        rel_selected = self._select_rel_top_k(s_triples, k=k)
        bridge_selected = self._select_bridge_top_k(
            s_triples=s_triples,
            bridge_scores=bridge_by_id,
            node_to_comp=node_to_comp,
            k=k,
        )

        rel_selected_ids = tuple(triple.evidence_id for triple in rel_selected)
        bridge_selected_ids = tuple(triple.evidence_id for triple in bridge_selected)

        rel_augmented = tuple(triples) + tuple(replace(triple, pool="A") for triple in rel_selected)
        bridge_augmented = tuple(triples) + tuple(replace(triple, pool="A") for triple in bridge_selected)

        conn_a_plus_rel = _anchor_connectivity(rel_augmented, anchors)
        conn_a_plus_bridge = _anchor_connectivity(bridge_augmented, anchors)

        rpi_rel = conn_a_plus_rel - conn_a
        rpi_bridge = conn_a_plus_bridge - conn_a
        should_recover = bool(esi_geom < self.config.recovery_tau_esi and rpi_bridge > 0.0)

        return RecoveryMetrics(
            conn_a=conn_a,
            conn_a_plus_rel=conn_a_plus_rel,
            conn_a_plus_bridge=conn_a_plus_bridge,
            rpi_rel_at_k=rpi_rel,
            rpi_bridge_at_k=rpi_bridge,
            delta_conn_bridge_at_k=rpi_bridge,
            selected_rel_ids=rel_selected_ids,
            selected_bridge_ids=bridge_selected_ids,
            should_recover=should_recover,
        )

    def apply_recovery(
        self,
        triples: Sequence[EvidenceTriple],
        selected_bridge_ids: Iterable[str],
    ) -> RecoveryResult:
        """Move selected S triples into A for one-step rerun."""

        selected_ids = set(selected_bridge_ids)
        moved: List[str] = []
        recovered: List[EvidenceTriple] = []
        for triple in triples:
            if triple.pool == "S" and triple.evidence_id in selected_ids:
                recovered.append(replace(triple, pool="A"))
                moved.append(triple.evidence_id)
            else:
                recovered.append(triple)

        return RecoveryResult(
            recovered_triples=tuple(recovered),
            moved_evidence_ids=tuple(moved),
        )


def connectivity_for_active(
    triples: Sequence[EvidenceTriple],
    anchors: Sequence[str],
) -> float:
    """Public helper for tests and logging."""

    return _anchor_connectivity(triples, anchors)
