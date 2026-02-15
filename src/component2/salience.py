"""Rationale extraction and faithfulness checks for M3."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .types import EvidenceGraph


def _drop_edge(graph: EvidenceGraph, evidence_id: str) -> EvidenceGraph:
    return EvidenceGraph(
        claim_id=graph.claim_id,
        nodes=list(graph.nodes),
        node_to_idx=dict(graph.node_to_idx),
        edges=[edge for edge in graph.edges if edge.evidence_id != evidence_id],
        anchors=list(graph.anchors),
        anchor_indices=list(graph.anchor_indices),
    )


def _drop_edges(graph: EvidenceGraph, evidence_ids: Iterable[str]) -> EvidenceGraph:
    evidence_id_set = set(str(evidence_id) for evidence_id in evidence_ids)
    return EvidenceGraph(
        claim_id=graph.claim_id,
        nodes=list(graph.nodes),
        node_to_idx=dict(graph.node_to_idx),
        edges=[edge for edge in graph.edges if edge.evidence_id not in evidence_id_set],
        anchors=list(graph.anchors),
        anchor_indices=list(graph.anchor_indices),
    )


def extract_top_rationale_edges(
    graph: EvidenceGraph,
    reasoner_output: Dict[str, object],
    top_n: int = 10,
    target_class_idx: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Rank edges by attention x PV relevance and return top-N rows.

    If target_class_idx is provided, use the stream attention for that class:
    - 0 (SUPPORTED): attention_sup
    - 1 (REFUTED): attention_ref
    Otherwise, use max(attention_sup, attention_ref).
    """

    salience_rows = reasoner_output.get("salience", [])
    by_id = {row.get("evidence_id"): row for row in salience_rows if isinstance(row, dict)}

    ranked = []
    for edge in graph.edges:
        row = by_id.get(edge.evidence_id, {})
        attention_sup = float(row.get("attention_sup", 0.0))
        attention_ref = float(row.get("attention_ref", 0.0))
        attention_max = max(attention_sup, attention_ref)
        if target_class_idx == 0:
            attention_used = attention_sup
        elif target_class_idx == 1:
            attention_used = attention_ref
        else:
            attention_used = attention_max
        pv_relevance = float(edge.rel)
        score = attention_used * pv_relevance

        ranked.append(
            {
                "evidence_id": edge.evidence_id,
                "source": graph.nodes[edge.source],
                "target": graph.nodes[edge.target],
                "relation": edge.relation,
                "pool": edge.pool,
                "p_ent": float(edge.p_ent),
                "p_con": float(edge.p_con),
                "p_neu": float(edge.p_neu),
                "pv_relevance": pv_relevance,
                "attention_sup": attention_sup,
                "attention_ref": attention_ref,
                "attention_max": attention_max,
                "attention_used": attention_used,
                "target_class_idx": target_class_idx,
                "salience_attention_pv": score,
            }
        )

    ranked.sort(
        key=lambda row: (
            row["salience_attention_pv"],
            row["attention_used"],
            row["pv_relevance"],
            row["evidence_id"],
        ),
        reverse=True,
    )
    return ranked[: max(0, int(top_n))]


def leave_one_out_faithfulness(
    reasoner,
    graph: EvidenceGraph,
    rationale_edges: Sequence[Dict[str, object]],
    predicted_class_idx: Optional[int] = None,
    max_edges: Optional[int] = None,
    base_output: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run leave-one-out confidence drops for a rationale edge list."""

    if base_output is None:
        base_output = reasoner.forward_graph(graph)

    probs = np.array(base_output["probs"], dtype=np.float64)
    pred_idx = int(np.argmax(probs)) if predicted_class_idx is None else int(predicted_class_idx)
    base_conf = float(probs[pred_idx])

    if max_edges is None:
        selected = list(rationale_edges)
    else:
        selected = list(rationale_edges)[: max(0, int(max_edges))]

    drops = []
    for row in selected:
        evidence_id = str(row["evidence_id"])
        pruned = _drop_edge(graph, evidence_id=evidence_id)
        out = reasoner.forward_graph(pruned)
        conf_after = float(np.array(out["probs"], dtype=np.float64)[pred_idx])
        conf_drop = base_conf - conf_after
        drops.append(
            {
                "evidence_id": evidence_id,
                "confidence_before": base_conf,
                "confidence_after": conf_after,
                "confidence_drop": conf_drop,
            }
        )

    mean_drop = float(np.mean([row["confidence_drop"] for row in drops])) if drops else 0.0
    positive_fraction = (
        float(np.mean([1.0 if row["confidence_drop"] > 0.0 else 0.0 for row in drops])) if drops else 0.0
    )

    return {
        "predicted_class_idx": pred_idx,
        "base_confidence": base_conf,
        "num_tested_edges": len(drops),
        "mean_confidence_drop": mean_drop,
        "positive_drop_fraction": positive_fraction,
        "is_faithful": bool(drops and mean_drop > 0.0),
        "edge_drops": drops,
    }


def grouped_edge_removal_faithfulness(
    reasoner,
    graph: EvidenceGraph,
    rationale_edges: Sequence[Dict[str, object]],
    predicted_class_idx: Optional[int] = None,
    max_edges: Optional[int] = None,
    base_output: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Measure confidence drop after removing multiple top rationale edges at once."""

    if base_output is None:
        base_output = reasoner.forward_graph(graph)

    probs = np.array(base_output["probs"], dtype=np.float64)
    pred_idx = int(np.argmax(probs)) if predicted_class_idx is None else int(predicted_class_idx)
    base_conf = float(probs[pred_idx])

    if max_edges is None:
        selected = list(rationale_edges)
    else:
        selected = list(rationale_edges)[: max(0, int(max_edges))]

    selected_ids = [str(row["evidence_id"]) for row in selected]
    if selected_ids:
        pruned = _drop_edges(graph, evidence_ids=selected_ids)
        out = reasoner.forward_graph(pruned)
        conf_after = float(np.array(out["probs"], dtype=np.float64)[pred_idx])
    else:
        conf_after = base_conf

    drop = base_conf - conf_after
    return {
        "predicted_class_idx": pred_idx,
        "base_confidence": base_conf,
        "confidence_after_group_removal": conf_after,
        "faithfulness_drop": drop,
        "num_removed_edges": len(selected_ids),
        "removed_edge_ids": selected_ids,
        "is_faithful": bool(selected_ids and drop > 0.0),
    }
