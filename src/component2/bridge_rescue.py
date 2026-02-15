"""Bridge rescue scoring with anchor-conditioned weighted PPR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Component2Config, DEFAULT_CONFIG
from .types import EvidenceGraph


@dataclass(frozen=True)
class BridgeScore:
    """Bridge score details for one evidence edge."""

    evidence_id: str
    pool: str
    source: str
    target: str
    p_source: float
    p_target: float
    weight: float
    bridge_bonus: float
    bridge_bonus_capped: float


@dataclass(frozen=True)
class PPRDiagnostics:
    """Diagnostics emitted during weighted PPR bridge scoring."""

    converged: bool
    iterations: int
    residual_l1: float
    num_nodes: int
    num_edges: int
    num_anchors: int
    anchor_pairs_total: int
    anchor_pairs_connected: int
    has_disconnected_anchors: bool
    fallback_reason: Optional[str]


class BridgeRescuePPR:
    """Compute BridgeBonus and neutral-capped BridgeBonus' for graph edges."""

    def __init__(self, config: Component2Config = DEFAULT_CONFIG):
        self.config = config

    def _anchor_restart(self, graph: EvidenceGraph) -> np.ndarray:
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            return np.zeros((0,), dtype=np.float64)

        restart = np.zeros((num_nodes,), dtype=np.float64)
        if graph.anchor_indices:
            value = 1.0 / float(len(graph.anchor_indices))
            for idx in graph.anchor_indices:
                if 0 <= idx < num_nodes:
                    restart[idx] = value
            return restart

        # Fallback for claims with no resolved anchors.
        return np.full((num_nodes,), 1.0 / float(num_nodes), dtype=np.float64)

    def _undirected_transition(self, graph: EvidenceGraph) -> np.ndarray:
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            return np.zeros((0, 0), dtype=np.float64)

        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for edge in graph.edges:
            w = self.config.ppr_epsilon + float(edge.rel)
            adjacency[edge.source, edge.target] += w
            adjacency[edge.target, edge.source] += w

        transition = np.zeros_like(adjacency)
        row_sum = np.sum(adjacency, axis=1)
        for idx in range(num_nodes):
            if row_sum[idx] > 0.0:
                transition[idx] = adjacency[idx] / row_sum[idx]
        return transition

    def _anchor_connectivity_stats(self, graph: EvidenceGraph) -> Tuple[int, int]:
        if len(graph.anchor_indices) < 2:
            return 0, 0

        adjacency = [[] for _ in range(len(graph.nodes))]
        for edge in graph.edges:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)

        def connected(src: int, dst: int) -> bool:
            if src == dst:
                return True
            queue = [src]
            seen = {src}
            while queue:
                cur = queue.pop(0)
                if cur == dst:
                    return True
                for nxt in adjacency[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            return False

        total = 0
        connected_count = 0
        anchors = list(graph.anchor_indices)
        for i, src in enumerate(anchors):
            for dst in anchors[i + 1 :]:
                total += 1
                if connected(src, dst):
                    connected_count += 1
        return total, connected_count

    def compute_node_scores_with_diagnostics(self, graph: EvidenceGraph) -> Tuple[np.ndarray, PPRDiagnostics]:
        """Compute node scores and return convergence/connectivity diagnostics."""

        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        num_anchors = len(graph.anchor_indices)
        anchor_pairs_total, anchor_pairs_connected = self._anchor_connectivity_stats(graph)
        has_disconnected_anchors = anchor_pairs_total > anchor_pairs_connected

        if num_nodes == 0:
            diagnostics = PPRDiagnostics(
                converged=True,
                iterations=0,
                residual_l1=0.0,
                num_nodes=0,
                num_edges=0,
                num_anchors=0,
                anchor_pairs_total=0,
                anchor_pairs_connected=0,
                has_disconnected_anchors=False,
                fallback_reason="empty_graph",
            )
            return np.zeros((0,), dtype=np.float64), diagnostics

        if num_edges == 0:
            diagnostics = PPRDiagnostics(
                converged=True,
                iterations=0,
                residual_l1=0.0,
                num_nodes=num_nodes,
                num_edges=num_edges,
                num_anchors=num_anchors,
                anchor_pairs_total=anchor_pairs_total,
                anchor_pairs_connected=anchor_pairs_connected,
                has_disconnected_anchors=has_disconnected_anchors,
                fallback_reason="no_edges",
            )
            return self._anchor_restart(graph), diagnostics

        alpha = float(self.config.ppr_alpha)
        tol = float(self.config.ppr_tolerance)
        max_iters = int(self.config.ppr_max_iters)
        restart = self._anchor_restart(graph)
        transition = self._undirected_transition(graph)

        p = restart.copy()
        residual = 0.0
        converged = False
        iters = 0
        for iters in range(1, max_iters + 1):
            next_p = alpha * restart + (1.0 - alpha) * (transition.T @ p)
            residual = float(np.sum(np.abs(next_p - p)))
            p = next_p
            if residual <= tol:
                converged = True
                break

        normalizer = float(np.sum(p))
        if normalizer > 0.0:
            p = p / normalizer

        diagnostics = PPRDiagnostics(
            converged=converged,
            iterations=iters if max_iters > 0 else 0,
            residual_l1=residual,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_anchors=num_anchors,
            anchor_pairs_total=anchor_pairs_total,
            anchor_pairs_connected=anchor_pairs_connected,
            has_disconnected_anchors=has_disconnected_anchors,
            fallback_reason=None,
        )
        return p, diagnostics

    def compute_node_scores(self, graph: EvidenceGraph) -> np.ndarray:
        """Compute anchor-conditioned weighted PPR node scores."""

        node_scores, _ = self.compute_node_scores_with_diagnostics(graph)
        return node_scores

    def compute_bridge_scores(
        self,
        graph: EvidenceGraph,
        node_scores: Optional[np.ndarray] = None,
    ) -> List[BridgeScore]:
        """Compute BridgeBonus and BridgeBonus' for each edge in the graph."""

        if node_scores is None:
            node_scores = self.compute_node_scores(graph)
        gamma = float(self.config.neutral_cap_gamma)
        rows: List[BridgeScore] = []

        for edge in graph.edges:
            p_source = float(node_scores[edge.source]) if node_scores.size else 0.0
            p_target = float(node_scores[edge.target]) if node_scores.size else 0.0
            weight = float(self.config.ppr_epsilon + edge.rel)
            bridge_bonus = p_source * weight * p_target
            neutral_multiplier = max(0.0, 1.0 - float(edge.p_neu)) ** gamma
            bridge_bonus_capped = bridge_bonus * neutral_multiplier

            rows.append(
                BridgeScore(
                    evidence_id=edge.evidence_id,
                    pool=edge.pool,
                    source=graph.nodes[edge.source],
                    target=graph.nodes[edge.target],
                    p_source=p_source,
                    p_target=p_target,
                    weight=weight,
                    bridge_bonus=bridge_bonus,
                    bridge_bonus_capped=bridge_bonus_capped,
                )
            )

        return rows

    @staticmethod
    def sort_desc(scores: Sequence[BridgeScore]) -> List[BridgeScore]:
        return sorted(
            scores,
            key=lambda row: (row.bridge_bonus_capped, row.bridge_bonus, row.weight, row.evidence_id),
            reverse=True,
        )
