"""Component 2 helpers for Evidence Sufficiency Index (ESI) estimation."""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Dict, List, Sequence

from .types import EvidenceTriple


def _anchor_connectivity_ac(
    triples_ac: Sequence[EvidenceTriple],
    anchors: Sequence[str],
) -> float:
    """Fraction of anchor pairs connected in undirected A+C graph."""

    adjacency: Dict[str, set[str]] = defaultdict(set)
    nodes = set()
    for triple in triples_ac:
        subj, _, obj = triple.raw_triple
        nodes.add(subj)
        nodes.add(obj)
        adjacency[subj].add(obj)
        adjacency[obj].add(subj)

    unique_anchors = [str(anchor) for anchor in dict.fromkeys(anchors) if str(anchor).strip()]
    present_anchors = [anchor for anchor in unique_anchors if anchor in nodes]
    if len(present_anchors) < 2:
        return 1.0

    def is_connected(src: str, dst: str) -> bool:
        if src == dst:
            return True
        queue: deque[str] = deque([src])
        seen = {src}
        while queue:
            cur = queue.popleft()
            if cur == dst:
                return True
            for nxt in adjacency.get(cur, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    connected = 0
    total = 0
    for idx, src in enumerate(present_anchors):
        for dst in present_anchors[idx + 1 :]:
            total += 1
            if is_connected(src, dst):
                connected += 1
    if total == 0:
        return 1.0
    return float(connected) / float(total)


def compute_esi_metrics_ac(
    triples: Sequence[EvidenceTriple],
    anchors: Sequence[str],
    eps: float = 0.05,
) -> Dict[str, float]:
    """
    Compute ESI metrics on combined A+C pools.

    Keys mirror Component 1 naming for compatibility (`mass_A`, `coverage_A`, ...),
    but values are computed from the combined A+C evidence set.
    """

    triples_ac = [triple for triple in triples if triple.pool in {"A", "C"}]
    if not triples_ac:
        return {
            "esi_prod": 0.0,
            "esi_geom": 0.0,
            "starve_score_prod": 1.0,
            "starve_score_geom": 1.0,
            "mass_A": 0.0,
            "mass_A_normalized": 0.0,
            "coverage_A": 0.0,
            "connectivity_A": 0.0,
        }

    mass_a = float(sum(triple.rel for triple in triples_ac))
    alpha = float(max(1, len(triples_ac)))
    mass_norm = 1.0 - math.exp(-mass_a / alpha)

    unique_anchors = [str(anchor) for anchor in dict.fromkeys(anchors) if str(anchor).strip()]
    if not unique_anchors:
        coverage = 1.0
        connectivity = 1.0
    else:
        nodes = set()
        for triple in triples_ac:
            subj, _, obj = triple.raw_triple
            nodes.add(subj)
            nodes.add(obj)
        unique_anchor_set = set(unique_anchors)
        coverage = float(sum(1 for anchor in unique_anchor_set if anchor in nodes)) / float(len(unique_anchor_set))
        connectivity = _anchor_connectivity_ac(triples_ac=triples_ac, anchors=unique_anchors)

    coverage_s = eps + (1.0 - eps) * coverage
    connectivity_s = eps + (1.0 - eps) * connectivity

    esi_prod = mass_norm * coverage_s * connectivity_s
    esi_geom = max(0.0, min(1.0, esi_prod ** (1.0 / 3.0)))

    return {
        "esi_prod": float(esi_prod),
        "esi_geom": float(esi_geom),
        "starve_score_prod": float(1.0 - esi_prod),
        "starve_score_geom": float(1.0 - esi_geom),
        "mass_A": float(mass_a),
        "mass_A_normalized": float(mass_norm),
        "coverage_A": float(coverage),
        "connectivity_A": float(connectivity),
    }


def estimate_esi_geom_ac(
    triples: Sequence[EvidenceTriple],
    anchors: Sequence[str],
    eps: float = 0.05,
) -> float:
    """Return ESI_geom computed over A+C pools."""

    return float(compute_esi_metrics_ac(triples=triples, anchors=anchors, eps=eps)["esi_geom"])
