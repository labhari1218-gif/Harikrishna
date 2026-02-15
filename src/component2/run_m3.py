"""Component 2 M3 smoke runner with selective prediction and rationale checks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import warnings

import numpy as np

from .anchor_adapter_factkg import FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .bridge_rescue import BridgeRescuePPR
from .config import Component2Config, DEFAULT_CONFIG
from .esi import compute_esi_metrics_ac
from .graph_builder import GraphBuilder
from .io_utils import (
    ensure_positive_max_claims,
    load_claims_jsonl,
    make_run_dir,
    write_json,
    write_jsonl,
)
from .reasoner import HybridMaskedDualStreamReasoner
from .recovery import BridgeRecoveryPolicy
from .salience import extract_top_rationale_edges, grouped_edge_removal_faithfulness, leave_one_out_faithfulness
from .selective import abstain_score, evaluate_selective_prediction
from .types import ClaimRecord, claim_from_dict, claim_label_to_index, ensure_non_empty_claims


DEFAULT_COMPONENT1_CLAIMS_PATH = Path("logs/component1/train/claims.jsonl")


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


def _synthetic_claims_m3(num_claims: int = 5) -> List[ClaimRecord]:
    """Generate toy claims for M3 selective prediction and rationale checks."""

    rows: List[Dict[str, object]] = []
    for idx in range(num_claims):
        rows.append(
            {
                "claim_id": f"synthetic_m3_{idx}",
                "claim_text": "A is related to C",
                "label": "SUPPORTED" if idx % 2 == 0 else "REFUTED",
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
                "sufficiency": {"esi_geom": 0.20 if idx % 2 == 0 else 0.85},
            }
        )
    return [claim_from_dict(row) for row in rows]


def _peek_first_json_row(path: Path) -> Dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return None


def _is_component1_claim_metrics_row(raw_row: Dict[str, Any]) -> bool:
    return "triples" not in raw_row and "top_evidence" in raw_row and "sufficiency" in raw_row


def _normalize_pool(pool_raw: object) -> str:
    int_mapping = {0: "A", 1: "S", 2: "C"}
    if isinstance(pool_raw, int) and pool_raw in int_mapping:
        return int_mapping[pool_raw]

    aliases = {
        "A": "A",
        "ACTIVE": "A",
        "S": "S",
        "SUSPENDED": "S",
        "C": "C",
        "CANDIDATE": "C",
    }
    pool = aliases.get(str(pool_raw).strip().upper(), str(pool_raw).strip().upper())
    if pool not in {"A", "S", "C"}:
        raise ValueError(f"Unsupported pool value '{pool_raw}' in pair logs.")
    return pool


def _load_claims_from_component1_logs(claims_jsonl: Path, max_claims: int) -> List[ClaimRecord]:
    """Load Component 1 claim metrics + pair logs into ClaimRecord rows."""

    pairs_jsonl = claims_jsonl.with_name("pairs.jsonl")
    if not pairs_jsonl.exists():
        raise FileNotFoundError(
            f"Missing companion pair log: {pairs_jsonl}. "
            "Component 1 metrics logs require pairs.jsonl to reconstruct triples."
        )

    selected_claim_rows: List[Dict[str, Any]] = []
    with claims_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            selected_claim_rows.append(json.loads(line))
            if len(selected_claim_rows) >= max_claims:
                break

    if not selected_claim_rows:
        return []

    selected_ids = set()
    for row in selected_claim_rows:
        claim_id = str(row.get("claim_id", "")).strip()
        if not claim_id:
            raise ValueError(f"Claims log row is missing claim_id in {claims_jsonl}")
        selected_ids.add(claim_id)

    triples_by_claim: Dict[str, List[Dict[str, object]]] = {claim_id: [] for claim_id in selected_ids}

    with pairs_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            claim_id = str(row.get("claim_id", "")).strip()
            if claim_id not in triples_by_claim:
                continue

            pool_raw = row.get("pool", row.get("evidence_assignment"))
            if pool_raw is None:
                raise ValueError(
                    f"Pair log row for claim_id={claim_id} is missing pool assignment. "
                    "Cannot reconstruct A/S/C sets. Re-run Component 1 with pair logs that include "
                    "pool labels and use --pair_log_mode all."
                )
            pool = _normalize_pool(pool_raw)

            triple = row.get("raw_triple")
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue

            probs = row.get("probs", {})
            if not isinstance(probs, dict):
                raise ValueError(
                    f"Pair log row for claim_id={claim_id} has non-dict probs field. "
                    f"Expected a probability map, got {type(probs).__name__}."
                )

            evidence_id = row.get("evidence_id")
            if evidence_id is None or str(evidence_id).strip() == "":
                raise ValueError(f"Pair log row for claim_id={claim_id} is missing evidence_id.")

            triples_by_claim[claim_id].append(
                {
                    "evidence_id": str(evidence_id),
                    "raw_triple": [str(triple[0]), str(triple[1]), str(triple[2])],
                    "pool": pool,
                    "probs": probs,
                }
            )

    claims: List[ClaimRecord] = []
    for raw in selected_claim_rows:
        claim_id = str(raw.get("claim_id", "")).strip()
        triples = triples_by_claim.get(claim_id, [])
        if not triples:
            raise ValueError(
                f"No pair-level triples found for claim_id={claim_id} in {pairs_jsonl}. "
                "Component 2 requires pair logs with full per-triple records."
            )

        counts = raw.get("counts", {})
        if isinstance(counts, dict):
            expected_a = int(counts.get("A", 0))
            expected_s = int(counts.get("S", 0))
            expected_c = int(counts.get("C", 0))
            expected_total = expected_a + expected_s + expected_c
            if expected_total > len(triples):
                raise ValueError(
                    f"Claim {claim_id} expects {expected_total} triples from counts(A/S/C), "
                    f"but only {len(triples)} triples were logged in {pairs_jsonl}. "
                    "This usually means pair logs are partial (e.g., A/C only) and recovery on S cannot be evaluated."
                )
            observed_s = sum(1 for row in triples if row["pool"] == "S")
            if expected_s > observed_s:
                raise ValueError(
                    f"Claim {claim_id} expects {expected_s} Suspended triples but only {observed_s} are present "
                    f"in {pairs_jsonl}. Re-run Component 1 with --pair_log_mode all."
                )

        claim_payload = {
            "claim_id": claim_id,
            "claim_text": raw.get("claim_text", ""),
            "label": raw.get("label"),
            "entity_set": raw.get("entity_set", raw.get("Entity_set", [])),
            "triples": triples,
            "sufficiency": raw.get("sufficiency", {}),
        }
        claims.append(claim_from_dict(claim_payload))

    return claims


def _load_claims_for_m3(input_jsonl: Path | None, max_claims: int) -> tuple[List[ClaimRecord], str]:
    """Load claims with explicit source labeling for run metadata."""

    if input_jsonl is None:
        return _synthetic_claims_m3(num_claims=max_claims), "synthetic"

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    first_row = _peek_first_json_row(input_jsonl)
    if isinstance(first_row, dict) and _is_component1_claim_metrics_row(first_row):
        return _load_claims_from_component1_logs(input_jsonl, max_claims=max_claims), "component1_logs"

    return load_claims_jsonl(input_jsonl, max_claims=max_claims), "triples_jsonl"


def run_smoke_m3(
    input_jsonl: Path | None,
    output_root: Path,
    max_claims: int,
    config: Component2Config = DEFAULT_CONFIG,
) -> Path:
    """Run M3 flow and log selective + rationale outputs."""

    ensure_positive_max_claims(max_claims)
    run_dir = make_run_dir(output_root=output_root, milestone="m3")
    claims, input_source = _load_claims_for_m3(input_jsonl=input_jsonl, max_claims=max_claims)
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
    for claim_idx, claim in enumerate(claims):
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
        label_idx = claim_label_to_index(claim_with_anchors.label)
        is_correct = None if label_idx is None else bool(pred_after_idx == label_idx)
        abstain = abstain_score(out_after["probs"], esi_geom_after, alpha=config.abstain_alpha)

        top_rationale = extract_top_rationale_edges(
            graph=graph_after,
            reasoner_output=out_after,
            top_n=config.rationale_top_n,
            target_class_idx=pred_after_idx,
        )

        faithfulness_loo = None
        salience_results = None
        if claim_idx < config.faithfulness_subset_size:
            faithfulness_loo = leave_one_out_faithfulness(
                reasoner=reasoner,
                graph=graph_after,
                rationale_edges=top_rationale,
                predicted_class_idx=pred_after_idx,
                max_edges=config.faithfulness_top_k_edges,
                base_output=out_after,
            )
            salience_results = grouped_edge_removal_faithfulness(
                reasoner=reasoner,
                graph=graph_after,
                rationale_edges=top_rationale,
                predicted_class_idx=pred_after_idx,
                max_edges=config.faithfulness_top_k_edges,
                base_output=out_after,
            )

        prediction_rows.append(
            {
                "claim_id": claim_with_anchors.claim_id,
                "anchors": anchors,
                "anchors_hash": _anchor_hash(anchors),
                "anchor_source": anchor_source,
                "anchor_seed_entities": list(claim_with_anchors.entity_set),
                "prediction_before": "SUPPORTED" if pred_before_idx == 0 else "REFUTED",
                "prediction_after": "SUPPORTED" if pred_after_idx == 0 else "REFUTED",
                "label": claim_with_anchors.label,
                "is_correct": is_correct,
                "prob_supported_before": float(out_before["probs"][0]),
                "prob_supported_after": float(out_after["probs"][0]),
                "esi_geom": esi_geom_after,
                "esi_geom_before": esi_geom_before,
                "esi_geom_after": esi_geom_after,
                "abstain_score": float(abstain),
                "rpi_rel_at_k": float(metrics.rpi_rel_at_k),
                "rpi_bridge_at_k": float(metrics.rpi_bridge_at_k),
                "delta_conn_bridge_at_k": float(metrics.delta_conn_bridge_at_k),
                "should_recover": bool(metrics.should_recover),
                "recovery_metrics": {
                    "rpi_rel_at_k": float(metrics.rpi_rel_at_k),
                    "rpi_bridge_at_k": float(metrics.rpi_bridge_at_k),
                    "delta_conn_bridge_at_k": float(metrics.delta_conn_bridge_at_k),
                    "should_recover": bool(metrics.should_recover),
                    "selected_rel_ids": list(metrics.selected_rel_ids),
                    "selected_bridge_ids": list(metrics.selected_bridge_ids),
                },
                "selected_rel_ids": list(metrics.selected_rel_ids),
                "selected_bridge_ids": list(metrics.selected_bridge_ids),
                "moved_evidence_ids": list(recovered.moved_evidence_ids if metrics.should_recover else ()),
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
                "top_rationale_edges": top_rationale,
                "faithfulness_loo": faithfulness_loo,
                "salience_results": salience_results,
            }
        )

    labeled_rows = [row for row in prediction_rows if row["is_correct"] is not None]
    selective_eval = evaluate_selective_prediction(
        abstain_scores=[float(row["abstain_score"]) for row in labeled_rows],
        is_correct=[bool(row["is_correct"]) for row in labeled_rows],
        tau_abstain=config.tau_abstain,
        compute_curve=config.compute_risk_coverage_sweep,
    )

    run_config = {
        "component2_config": config.to_snapshot_dict(),
        "runtime": {
            "input_jsonl": None if input_jsonl is None else str(input_jsonl),
            "input_source": input_source,
            "max_claims": max_claims,
            "num_claims": len(claims),
            "output_dir": str(run_dir),
            "reasoner": "HybridMaskedDualStreamReasoner",
            "relation_embeddings": {
                "mode": "fixed_features_numpy",
                "trainable": False,
                "note": "Relation embeddings are random-initialized features in numpy inference; no end-to-end training in v1.",
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
            "selective_prediction": {
                "abstain_formula": "alpha*(1-max_prob)+(1-alpha)*(1-esi_geom)",
                "alpha": config.abstain_alpha,
                "metric": "AURC" if config.compute_risk_coverage_sweep else None,
                "compute_risk_coverage_sweep": bool(config.compute_risk_coverage_sweep),
                "tau_abstain": config.tau_abstain,
            },
            "salience": {
                "formula": "attention_for_predicted_class * (p_ent+p_con)",
                "top_n": config.rationale_top_n,
                "faithfulness_subset_size": config.faithfulness_subset_size,
                "faithfulness_top_k_edges": config.faithfulness_top_k_edges,
                "faithfulness_drop_mode": "grouped_top_k_removal",
            },
            "recovery_policy": {
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

    operating_point = selective_eval.get("operating_point")
    write_json(run_dir / "run_config.json", run_config)
    write_json(
        run_dir / "summary.json",
        {
            "num_claims": len(prediction_rows),
            "num_labeled_claims": len(labeled_rows),
            "anchor_source_counts": anchor_source_counts,
            "claims_missing_pickle_mapping": claims_missing_pickle_mapping,
            "num_recovered": sum(1 for row in prediction_rows if row["should_recover"]),
            "aurc": None if selective_eval["aurc"] is None else float(selective_eval["aurc"]),
            "num_faithful_claims": sum(
                1
                for row in prediction_rows
                if isinstance(row["faithfulness_loo"], dict) and row["faithfulness_loo"].get("is_faithful")
            ),
            "num_ppr_not_converged": sum(1 for row in prediction_rows if not row["ppr_diagnostics"]["converged"]),
            "num_disconnected_anchor_graphs": sum(
                1 for row in prediction_rows if row["ppr_diagnostics"]["has_disconnected_anchors"]
            ),
            "num_salience_checked": sum(1 for row in prediction_rows if isinstance(row["salience_results"], dict)),
            "num_perf_guardrail_warnings": len(perf_guardrail_rows),
            "perf_guardrail_claim_ids": [str(row["claim_id"]) for row in perf_guardrail_rows],
            "tau_abstain": None if operating_point is None else float(operating_point["tau_abstain"]),
            "coverage_at_tau": None if operating_point is None else float(operating_point["coverage"]),
            "risk_at_tau": None if operating_point is None else float(operating_point["risk"]),
            "accuracy_on_answered_at_tau": (
                None if operating_point is None else float(operating_point["accuracy_on_answered"])
            ),
            "abstain_rate_at_tau": None if operating_point is None else float(operating_point["abstain_rate"]),
            "relation_vocab_size_active": len(builder_active.relation_vocab_snapshot()),
            "relation_vocab_size_full": len(builder_full.relation_vocab_snapshot()),
        },
    )
    write_json(run_dir / "risk_coverage.json", selective_eval)
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
    parser = argparse.ArgumentParser(description="Component 2 M3 smoke runner")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_COMPONENT1_CLAIMS_PATH,
        help="Claim JSONL input (defaults to Component 1 train claims log)",
    )
    parser.add_argument("--max-claims", type=int, default=DEFAULT_CONFIG.smoke_max_claims)
    parser.add_argument(
        "--tau-abstain",
        type=float,
        default=DEFAULT_CONFIG.tau_abstain,
        help="Optional fixed abstention operating point. If omitted, only risk-coverage sweep is used.",
    )
    parser.add_argument(
        "--no-risk-coverage-sweep",
        action="store_true",
        help="Disable risk-coverage sweep/AURC and compute only fixed operating point (if tau_abstain is set).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("logs/component2"))
    parser.add_argument("--mask-sup-alpha", type=float, default=None)
    parser.add_argument("--mask-sup-beta", type=float, default=None)
    parser.add_argument("--mask-ref-alpha", type=float, default=None)
    parser.add_argument("--mask-ref-beta", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = replace(
        DEFAULT_CONFIG,
        tau_abstain=args.tau_abstain,
        compute_risk_coverage_sweep=not args.no_risk_coverage_sweep,
    )
    config = _config_with_mask_overrides(
        base_config,
        mask_sup_alpha=args.mask_sup_alpha,
        mask_sup_beta=args.mask_sup_beta,
        mask_ref_alpha=args.mask_ref_alpha,
        mask_ref_beta=args.mask_ref_beta,
    )
    run_dir = run_smoke_m3(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        max_claims=args.max_claims,
        config=config,
    )
    print(f"Component 2 M3 run completed: {run_dir}")


if __name__ == "__main__":
    main()
