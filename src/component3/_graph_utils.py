"""Shared graph/triple utilities for Component 3 modules."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Mapping, Sequence


def triple_field(triple: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(triple, Mapping):
        return triple.get(field_name, default)
    return getattr(triple, field_name, default)


def triple_pool(triple: Any) -> str:
    return str(triple_field(triple, "pool", "A")).upper()


def triple_evidence_id(triple: Any, fallback_idx: int = 0) -> str:
    value = triple_field(triple, "evidence_id", None)
    if value is None:
        return f"triple_{fallback_idx}"
    return str(value)


def triple_raw(triple: Any, *, required: bool = True) -> tuple[str, str, str] | None:
    raw = triple_field(triple, "raw_triple", None)
    if raw is None:
        raw = triple_field(triple, "triple", None)
    if raw is None:
        content = triple_field(triple, "content", None)
        if isinstance(content, Mapping):
            raw = content.get("triple")

    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return str(raw[0]), str(raw[1]), str(raw[2])

    if required:
        raise ValueError(f"Unsupported triple format: {triple!r}")
    return None


def triple_rel(triple: Any) -> float:
    rel = triple_field(triple, "rel", None)
    if rel is not None:
        return float(rel)
    p_ent = float(triple_field(triple, "p_ent", 0.0))
    p_con = float(triple_field(triple, "p_con", 0.0))
    return p_ent + p_con


def triple_p_ent(triple: Any) -> float:
    return float(triple_field(triple, "p_ent", 0.0))


def triple_p_con(triple: Any) -> float:
    return float(triple_field(triple, "p_con", 0.0))


def triple_with_pool(triple: Any, pool: str) -> Any:
    normalized_pool = str(pool).upper()
    if isinstance(triple, dict):
        row = dict(triple)
        row["pool"] = normalized_pool
        return row
    if is_dataclass(triple):
        return replace(triple, pool=normalized_pool)
    if hasattr(triple, "__dict__"):  # pragma: no cover - defensive fallback
        clone = triple.__class__(**triple.__dict__)
        setattr(clone, "pool", normalized_pool)
        return clone
    raise TypeError(f"Cannot set pool for triple type: {type(triple).__name__}")


def active_edges(active_triples: Sequence[Any]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for triple in active_triples:
        if triple_pool(triple) not in {"A", "C"}:
            continue
        subj, _, obj = triple_raw(triple, required=True)
        edges.append((subj, obj))
    return edges


def connected_components(active_triples: Sequence[Any]) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for subj, obj in active_edges(active_triples):
        nodes.add(subj)
        nodes.add(obj)
        adjacency.setdefault(subj, set()).add(obj)
        adjacency.setdefault(obj, set()).add(subj)

    node_to_component: dict[str, int] = {}
    component_idx = 0
    for node in nodes:
        if node in node_to_component:
            continue
        queue = [node]
        node_to_component[node] = component_idx
        while queue:
            current = queue.pop(0)
            for nxt in adjacency.get(current, ()):
                if nxt not in node_to_component:
                    node_to_component[nxt] = component_idx
                    queue.append(nxt)
        component_idx += 1
    return node_to_component


def is_disconnected_graph(active_triples: Sequence[Any], anchors: Sequence[str] | None = None) -> bool:
    node_to_component = connected_components(active_triples)
    if not node_to_component:
        return True

    if anchors:
        anchor_components = {node_to_component.get(str(anchor)) for anchor in anchors}
        if None in anchor_components:
            return True
        return len(anchor_components) > 1

    return len(set(node_to_component.values())) > 1


def edge_connects_components(triple: Any, node_to_component: dict[str, int]) -> bool:
    subj, _, obj = triple_raw(triple, required=True)
    comp_subj = node_to_component.get(subj)
    comp_obj = node_to_component.get(obj)
    if comp_subj is None and comp_obj is None:
        return False
    if comp_subj is None or comp_obj is None:
        return True
    return comp_subj != comp_obj

