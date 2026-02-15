"""Numeric parity checks between C2 reasoner attention and PV-QAGNN single-stream proxy."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

try:
    from component2.config import DEFAULT_CONFIG, Component2Config
    from component2.graph_builder import GraphBuilder
    from component2.reasoner import HybridMaskedDualStreamReasoner
    from component2.types import evidence_from_dict
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from component2.config import DEFAULT_CONFIG, Component2Config
    from component2.graph_builder import GraphBuilder
    from component2.reasoner import HybridMaskedDualStreamReasoner
    from component2.types import evidence_from_dict


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _single_stream_proxy_weight(
    *,
    base_attention_sup: float,
    base_attention_ref: float,
    p_ent: float,
    p_con: float,
    alpha_sup: float = 4.0,
    beta_sup: float = -2.0,
    alpha_ref: float = 4.0,
    beta_ref: float = -2.0,
) -> float:
    """PV-QAGNN single-stream proxy: mean of support/refute weighted attentions."""

    gate_sup = _sigmoid(alpha_sup * p_ent + beta_sup)
    gate_ref = _sigmoid(alpha_ref * p_con + beta_ref)
    w_sup = float(base_attention_sup) * gate_sup
    w_ref = float(base_attention_ref) * gate_ref
    return 0.5 * (w_sup + w_ref)


def _synthetic_triples_for_claim(claim_idx: int) -> list:
    """Build varied A/C triples for parity evaluation."""

    p1 = min(0.95, 0.45 + (0.03 * claim_idx))
    p2 = min(0.90, 0.40 + (0.02 * claim_idx))
    c1 = min(0.90, 0.35 + (0.025 * claim_idx))
    c2 = min(0.88, 0.30 + (0.02 * claim_idx))

    rows = [
        {
            "evidence_id": f"{claim_idx}_a1",
            "raw_triple": [f"A{claim_idx}", "r_left", f"B{claim_idx}"],
            "pool": "A",
            "p_ent": p1,
            "p_con": 0.08,
            "p_neu": max(0.0, 1.0 - p1 - 0.08),
        },
        {
            "evidence_id": f"{claim_idx}_a2",
            "raw_triple": [f"B{claim_idx}", "r_bridge", f"C{claim_idx}"],
            "pool": "A",
            "p_ent": p2,
            "p_con": 0.10,
            "p_neu": max(0.0, 1.0 - p2 - 0.10),
        },
        {
            "evidence_id": f"{claim_idx}_c1",
            "raw_triple": [f"A{claim_idx}", "r_refute", f"D{claim_idx}"],
            "pool": "C",
            "p_ent": 0.08,
            "p_con": c1,
            "p_neu": max(0.0, 1.0 - 0.08 - c1),
        },
        {
            "evidence_id": f"{claim_idx}_c2",
            "raw_triple": [f"D{claim_idx}", "r_refute_2", f"C{claim_idx}"],
            "pool": "C",
            "p_ent": 0.10,
            "p_con": c2,
            "p_neu": max(0.0, 1.0 - 0.10 - c2),
        },
    ]
    return [evidence_from_dict(row) for row in rows]


def run_numeric_parity_check(
    *,
    num_claims: int = 10,
    threshold: float = 0.9,
    config: Component2Config = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Compare C2 dual-stream edge attention to PV-QAGNN single-stream proxy weights.

    T3.5 locked check:
      - alpha/beta fixed at 4.0 / -2.0
      - evaluate 10 claims
      - report Pearson correlation on edge weights
    """

    if num_claims < 1:
        raise ValueError("`num_claims` must be >= 1.")

    locked_cfg = replace(
        config,
        mask_sup_scale=4.0,
        mask_sup_bias=-2.0,
        mask_ref_scale=4.0,
        mask_ref_bias=-2.0,
    )

    builder = GraphBuilder(include_pools=("A", "C"))
    reasoner = HybridMaskedDualStreamReasoner(config=locked_cfg)

    c2_weights: List[float] = []
    c3_proxy_weights: List[float] = []
    per_claim: List[Dict[str, Any]] = []

    for claim_idx in range(num_claims):
        claim_id = f"parity_{claim_idx}"
        triples = _synthetic_triples_for_claim(claim_idx=claim_idx)
        graph = builder.build(
            claim_id=claim_id,
            triples=triples,
            anchors=[f"A{claim_idx}", f"C{claim_idx}"],
        )
        outputs = reasoner.forward_graph(graph)
        salience_rows = outputs["salience"]

        triple_by_id = {triple.evidence_id: triple for triple in triples}
        local_c2: List[float] = []
        local_c3: List[float] = []
        for row in salience_rows:
            evidence_id = str(row["evidence_id"])
            triple = triple_by_id[evidence_id]
            c2_val = 0.5 * (float(row["attention_sup"]) + float(row["attention_ref"]))
            c3_val = _single_stream_proxy_weight(
                base_attention_sup=float(row["base_attention_sup"]),
                base_attention_ref=float(row["base_attention_ref"]),
                p_ent=float(triple.p_ent),
                p_con=float(triple.p_con),
                alpha_sup=4.0,
                beta_sup=-2.0,
                alpha_ref=4.0,
                beta_ref=-2.0,
            )
            c2_weights.append(c2_val)
            c3_proxy_weights.append(c3_val)
            local_c2.append(c2_val)
            local_c3.append(c3_val)

        claim_corr = (
            float(np.corrcoef(local_c2, local_c3)[0, 1])
            if len(local_c2) > 1 and np.std(local_c2) > 0.0 and np.std(local_c3) > 0.0
            else 1.0
        )
        per_claim.append({"claim_id": claim_id, "num_edges": len(local_c2), "pearson_corr": claim_corr})

    if np.std(c2_weights) == 0.0 or np.std(c3_proxy_weights) == 0.0:
        pearson_corr = 1.0 if np.allclose(c2_weights, c3_proxy_weights) else 0.0
    else:
        pearson_corr = float(np.corrcoef(c2_weights, c3_proxy_weights)[0, 1])

    return {
        "num_claims": num_claims,
        "num_edges": len(c2_weights),
        "alpha_beta": {
            "alpha_sup": 4.0,
            "beta_sup": -2.0,
            "alpha_ref": 4.0,
            "beta_ref": -2.0,
        },
        "pearson_corr": pearson_corr,
        "threshold": float(threshold),
        "passed": bool(pearson_corr > float(threshold)),
        "per_claim": per_claim,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Component 3 T3.5 numeric parity check.")
    parser.add_argument("--num-claims", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_numeric_parity_check(num_claims=args.num_claims, threshold=args.threshold)
    print(json.dumps(report, indent=2))
