"""Component 2 M2 smoke runner with bridge rescue and one-step recovery."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np

from .anchor_adapter_factkg import FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .bridge_rescue import BridgeRescuePPR
from .config import Component2Config, DEFAULT_CONFIG
from .esi import compute_esi_metrics_ac
from .graph_builder import GraphBuilder
from .io_utils import (
    ensure_existing_input_path,
    ensure_positive_max_claims,
    load_claims_jsonl,
    make_run_dir,
    write_json,
    write_jsonl,
)
from .reasoner import HybridMaskedDualStreamReasoner
from .recovery import BridgeRecoveryPolicy
from .types import ClaimRecord, claim_from_dict, ensure_non_empty_claims


def _anchor_hash(anchors: List[str]) -> str:
    digest = hashlib.sha1()
    digest.update("||".join(anchors).encode("utf-8"))
    return digest.hexdigest()


def _make_anchor_adapter(config: Component2Config) -> FactKGAnchorAdapter:
    return FactKGAnchorAdapter(
        factkg_dir=Path(config.factkg_data_dir),
        split_pickles={
            "train": config.factkg_train_pickle,
            "val": config.factkg_val_pickle,
            "test": config.factkg_test_pickle,
        },
    )


def _synthetic_claims_m2(num_claims: int = 5) -> List[ClaimRecord]:
    """Generate toy claims with disconnected anchors and bridge candidates in S."""

    rows: List[Dict[str, object]] = []
    for idx in range(num_claims):
        rows.append(
            {
                "claim_id": f"synthetic_m2_{idx}",
                "claim_text": "A is related to C",
                "label": "SUPPORTED",
                "entity_set": ["A", "C"],
                "triples": [
                    {
                        "evidence_id": f"{idx}_a1",
                        "raw_triple": ["A", "r_left", "B"],
                        "pool": "A",
                        "p_ent": 0.65,
                        "p_con": 0.05,
                        "p_neu": 0.30,
                    },
                    {
                        "evidence_id": f"{idx}_a2",
                        "raw_triple": ["C", "r_right", "D"],
                        "pool": "A",
                        "p_ent": 0.62,
                        "p_con": 0.06,
                        "p_neu": 0.32,
                    },
                    {
                        "evidence_id": f"{idx}_s_bad1",
                        "raw_triple": ["B", "r_noise", "X"],
                        "pool": "S",
                        "p_ent": 0.83,
                        "p_con": 0.05,
                        "p_neu": 0.12,
                    },
                    {
                        "evidence_id": f"{idx}_s_bad2",
                        "raw_triple": ["B", "r_noise_2", "Y"],
                        "pool": "S",
                        "p_ent": 0.82,
                        "p_con": 0.05,
                        "p_neu": 0.13,
                    },
                    {
                        "evidence_id": f"{idx}_s_bad3",
                        "raw_triple": ["D", "r_noise_3", "W"],
                        "pool": "S",
                        "p_ent": 0.81,
                        "p_con": 0.05,
                        "p_neu": 0.14,
                    },
                    {
                        "evidence_id": f"{idx}_s_bridge",
                        "raw_triple": ["B", "r_bridge", "C"],
                        "pool": "S",
                        "p_ent": 0.52,
                        "p_con": 0.08,
                        "p_neu": 0.12,
                    },
                    {
                        "evidence_id": f"{idx}_c1",
                        "raw_triple": ["A", "r_refute", "Z"],
                        "pool": "C",
                        "p_ent": 0.05,
                        "p_con": 0.70,
                        "p_neu": 0.25,
                    },
                ],
                "sufficiency": {"esi_geom": 0.20},
            }
        )
    return [claim_from_dict(row) for row in rows]


def run_smoke_m2(
    input_jsonl: Path | None,
    output_root: Path,
    max_claims: int,
    config: Component2Config = DEFAULT_CONFIG,
) -> Path:
    """Run M2 flow and log recovery diagnostics."""

    ensure_positive_max_claims(max_claims)
    ensure_existing_input_path(input_jsonl)
    run_dir = make_run_dir(output_root=output_root, milestone="m2")

    if input_jsonl is not None:
        claims = load_claims_jsonl(input_jsonl, max_claims=max_claims)
    else:
        claims = _synthetic_claims_m2(num_claims=max_claims)
    ensure_non_empty_claims(claims)

    anchor_selector = AnchorSelector(config=config)
    builder_active = GraphBuilder(include_pools=("A", "C"))
    builder_full = GraphBuilder(include_pools=("A", "S", "C"))
    reasoner = HybridMaskedDualStreamReasoner(config=config)
    bridge_rescue = BridgeRescuePPR(config=config)
    recovery_policy = BridgeRecoveryPolicy(config=config)
    anchor_adapter = _make_anchor_adapter(config=config)

    anchor_source_counts: Dict[str, int] = {"provided": 0, "pickle": 0, "heuristic": 0}
    claims_missing_pickle_mapping = 0
    perf_guardrail_rows: List[Dict[str, object]] = []

    prediction_rows = []
    for claim in claims:
        injection = anchor_adapter.inject_claim(claim)
        claim_with_anchors = injection.claim
        anchor_source = injection.anchor_source
        anchor_source_counts[anchor_source] = anchor_source_counts.get(anchor_source, 0) + 1
        if anchor_source == "heuristic":
            claims_missing_pickle_mapping += 1

        anchors = anchor_selector.select_anchors(
            claim_text=claim_with_anchors.claim_text,
            triples=claim_with_anchors.triples,
            seed_entities=claim_with_anchors.entity_set,
        )
        graph_before = builder_active.build(
            claim_id=claim_with_anchors.claim_id,
            triples=claim_with_anchors.triples,
            anchors=anchors,
        )
        out_before = reasoner.forward_graph(graph_before)

        full_graph = builder_full.build(
            claim_id=claim_with_anchors.claim_id,
            triples=claim_with_anchors.triples,
            anchors=anchors,
        )
        if len(full_graph.edges) > int(config.numpy_edge_warn_threshold):
            warnings.warn(
                "Component 2 numpy mode is running on a large graph "
                f"(claim_id={claim_with_anchors.claim_id}, edges={len(full_graph.edges)}). "
                "Consider a future torch mode or limit max_claims for faster runs."
            )
            perf_guardrail_rows.append(
                {
                    "claim_id": claim_with_anchors.claim_id,
                    "num_edges": len(full_graph.edges),
                }
            )
        node_scores, ppr_diag = bridge_rescue.compute_node_scores_with_diagnostics(full_graph)
        bridge_scores = bridge_rescue.compute_bridge_scores(full_graph, node_scores=node_scores)

        esi_before_metrics = compute_esi_metrics_ac(claim_with_anchors.triples, anchors=anchors)
        esi_geom_before = float(esi_before_metrics["esi_geom"])
        claim_with_anchors.sufficiency.update(esi_before_metrics)
        metrics = recovery_policy.evaluate(
            triples=claim_with_anchors.triples,
            anchors=anchors,
            bridge_scores=bridge_scores,
            esi_geom=esi_geom_before,
        )

        recovered = recovery_policy.apply_recovery(
            triples=claim_with_anchors.triples,
            selected_bridge_ids=metrics.selected_bridge_ids,
        )
        triples_after = recovered.recovered_triples if metrics.should_recover else claim_with_anchors.triples
        esi_after_metrics = (
            compute_esi_metrics_ac(triples_after, anchors=anchors) if metrics.should_recover else esi_before_metrics
        )
        esi_geom_after = float(esi_after_metrics["esi_geom"])
        claim_with_anchors.sufficiency.update(esi_after_metrics)
        graph_after = builder_active.build(
            claim_id=claim_with_anchors.claim_id,
            triples=triples_after,
            anchors=anchors,
        )
        out_after = reasoner.forward_graph(graph_after)

        pred_before_idx = int(np.argmax(out_before["probs"]))
        pred_after_idx = int(np.argmax(out_after["probs"]))

        prediction_rows.append(
            {
                "claim_id": claim_with_anchors.claim_id,
                "anchors": anchors,
                "anchors_hash": _anchor_hash(anchors),
                "anchor_source": anchor_source,
                "anchor_seed_entities": list(claim_with_anchors.entity_set),
                "prediction_before": "SUPPORTED" if pred_before_idx == 0 else "REFUTED",
                "prediction_after": "SUPPORTED" if pred_after_idx == 0 else "REFUTED",
                "prob_supported_before": float(out_before["probs"][0]),
                "prob_supported_after": float(out_after["probs"][0]),
                "esi_geom": esi_geom_after,
                "esi_geom_before": esi_geom_before,
                "esi_geom_after": esi_geom_after,
                "rpi_rel_at_k": float(metrics.rpi_rel_at_k),
                "rpi_bridge_at_k": float(metrics.rpi_bridge_at_k),
                "delta_conn_bridge_at_k": float(metrics.delta_conn_bridge_at_k),
                "should_recover": bool(metrics.should_recover),
                "selected_rel_ids": list(metrics.selected_rel_ids),
                "selected_bridge_ids": list(metrics.selected_bridge_ids),
                "moved_evidence_ids": list(recovered.moved_evidence_ids if metrics.should_recover else ()),
                "bridge_scores_s": [
                    {
                        "evidence_id": row.evidence_id,
                        "bridge_bonus": float(row.bridge_bonus),
                        "bridge_bonus_capped": float(row.bridge_bonus_capped),
                    }
                    for row in sorted(
                        [r for r in bridge_scores if r.pool == "S"],
                        key=lambda r: r.bridge_bonus_capped,
                        reverse=True,
                    )
                ],
                "ppr_diagnostics": {
                    "converged": bool(ppr_diag.converged),
                    "iterations": int(ppr_diag.iterations),
                    "residual_l1": float(ppr_diag.residual_l1),
                    "num_nodes": int(ppr_diag.num_nodes),
                    "num_edges": int(ppr_diag.num_edges),
                    "num_anchors": int(ppr_diag.num_anchors),
                    "anchor_pairs_total": int(ppr_diag.anchor_pairs_total),
                    "anchor_pairs_connected": int(ppr_diag.anchor_pairs_connected),
                    "has_disconnected_anchors": bool(ppr_diag.has_disconnected_anchors),
                    "fallback_reason": ppr_diag.fallback_reason,
                },
                "top_salience_after": sorted(
                    out_after["salience"],
                    key=lambda row: row["salience"],
                    reverse=True,
                )[:3],
            }
        )

    run_config = {
        "component2_config": config.to_snapshot_dict(),
        "runtime": {
            "input_jsonl": None if input_jsonl is None else str(input_jsonl),
            "max_claims": max_claims,
            "num_claims": len(claims),
            "output_dir": str(run_dir),
            "reasoner": "HybridMaskedDualStreamReasoner",
            "relation_embeddings": {
                "mode": "fixed_features_numpy",
                "trainable": False,
            },
            "anchors": {
                "source_priority": ["provided", "pickle", "heuristic"],
                "source_counts": anchor_source_counts,
                "claims_missing_pickle_mapping": claims_missing_pickle_mapping,
                "factkg_pickles": {
                    "train": str(Path(config.factkg_data_dir) / config.factkg_train_pickle),
                    "val": str(Path(config.factkg_data_dir) / config.factkg_val_pickle),
                    "test": str(Path(config.factkg_data_dir) / config.factkg_test_pickle),
                },
            },
            "bridge_rescue": {
                "class": "BridgeRescuePPR",
                "graph_pools": ["A", "S", "C"],
                "formula": "BridgeBonus=e_ppr_u*(epsilon+rel)*e_ppr_v",
                "neutral_cap": "BridgeBonus' = BridgeBonus*(1-p_neu)^gamma",
            },
            "recovery_policy": {
                "class": "BridgeRecoveryPolicy",
                "trigger": "esi_geom<tau_esi and delta_conn_bridge_at_k>0",
                "recovery_top_k": config.recovery_top_k,
            },
            "mask_formulas": {
                "support": f"sigmoid({config.mask_sup_scale:g}*p_ent{config.mask_sup_bias:+g})",
                "refute": f"sigmoid({config.mask_ref_scale:g}*p_con{config.mask_ref_bias:+g})",
            },
            "performance_guardrails": {
                "numpy_edge_warn_threshold": int(config.numpy_edge_warn_threshold),
                "num_warnings": len(perf_guardrail_rows),
            },
        },
    }

    write_json(run_dir / "run_config.json", run_config)
    write_json(
        run_dir / "summary.json",
        {
            "num_claims": len(prediction_rows),
            "anchor_source_counts": anchor_source_counts,
            "claims_missing_pickle_mapping": claims_missing_pickle_mapping,
            "num_recovered": sum(1 for row in prediction_rows if row["should_recover"]),
            "avg_delta_conn_bridge_at_k": float(
                np.mean([row["delta_conn_bridge_at_k"] for row in prediction_rows]) if prediction_rows else 0.0
            ),
            "num_ppr_not_converged": sum(1 for row in prediction_rows if not row["ppr_diagnostics"]["converged"]),
            "num_disconnected_anchor_graphs": sum(
                1 for row in prediction_rows if row["ppr_diagnostics"]["has_disconnected_anchors"]
            ),
            "num_perf_guardrail_warnings": len(perf_guardrail_rows),
            "perf_guardrail_claim_ids": [str(row["claim_id"]) for row in perf_guardrail_rows],
            "relation_vocab_size_active": len(builder_active.relation_vocab_snapshot()),
            "relation_vocab_size_full": len(builder_full.relation_vocab_snapshot()),
        },
    )
    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    return run_dir


def _config_with_mask_overrides(
    config: Component2Config,
    *,
    mask_sup_alpha: float | None,
    mask_sup_beta: float | None,
    mask_ref_alpha: float | None,
    mask_ref_beta: float | None,
) -> Component2Config:
    updates = {}
    if mask_sup_alpha is not None:
        updates["mask_sup_scale"] = float(mask_sup_alpha)
    if mask_sup_beta is not None:
        updates["mask_sup_bias"] = float(mask_sup_beta)
    if mask_ref_alpha is not None:
        updates["mask_ref_scale"] = float(mask_ref_alpha)
    if mask_ref_beta is not None:
        updates["mask_ref_bias"] = float(mask_ref_beta)
    if not updates:
        return config
    return replace(config, **updates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component 2 M2 smoke runner")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Optional claim JSONL input")
    parser.add_argument("--max-claims", type=int, default=DEFAULT_CONFIG.smoke_max_claims)
    parser.add_argument("--output-root", type=Path, default=Path("logs/component2"))
    parser.add_argument("--mask-sup-alpha", type=float, default=None)
    parser.add_argument("--mask-sup-beta", type=float, default=None)
    parser.add_argument("--mask-ref-alpha", type=float, default=None)
    parser.add_argument("--mask-ref-beta", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config_with_mask_overrides(
        DEFAULT_CONFIG,
        mask_sup_alpha=args.mask_sup_alpha,
        mask_sup_beta=args.mask_sup_beta,
        mask_ref_alpha=args.mask_ref_alpha,
        mask_ref_beta=args.mask_ref_beta,
    )
    run_dir = run_smoke_m2(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        max_claims=args.max_claims,
        config=config,
    )
    print(f"Component 2 M2 run completed: {run_dir}")


if __name__ == "__main__":
    main()
