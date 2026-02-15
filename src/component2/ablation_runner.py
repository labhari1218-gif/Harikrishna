"""Ablation study runner for Component 2 (A0-A6 + tau-sweep)."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .ablation_utils import (
    BinaryMetrics,
    PlotWriter,
    active_component_map,
    baseline_predict_from_component1,
    bridge_edge_connects_different_components,
    compute_binary_metrics,
    estimate_esi_geom,
    git_commit_hash,
    load_claims_for_ablation,
    make_ablation_run_dir,
    pool_counts,
    safe_mean,
    select_deterministic_subset,
    spearman_correlation,
    summarize_bridge_top_values,
    write_selected_claim_ids,
)
from .anchor_adapter_factkg import FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .bridge_rescue import BridgeRescuePPR
from .config import Component2Config, DEFAULT_CONFIG
from .graph_builder import GraphBuilder
from .io_utils import write_json, write_jsonl
from .mask_tuning import MaskTuningSettings, tune_mask_coefficients
from .reasoner import HybridMaskedDualStreamReasoner
from .recovery import BridgeRecoveryPolicy, connectivity_for_active
from .selective import abstain_score, evaluate_selective_prediction
from .types import ClaimRecord, EvidenceTriple, claim_label_to_index


@dataclass(frozen=True)
class AblationMode:
    """One ablation mode controlled by feature switches."""

    ablation_id: str
    description: str
    use_component2: bool
    enable_bridge_scoring: bool
    enable_recovery: bool
    disable_pv_masks: bool = False
    stress_drop_bridge_triples: bool = False
    stress_add_distractors: int = 0


ABLATION_MODES: Dict[str, AblationMode] = {
    "A0": AblationMode(
        ablation_id="A0",
        description="Baseline (pre-Component2 heuristic verifier over Component1 A/C masses)",
        use_component2=False,
        enable_bridge_scoring=False,
        enable_recovery=False,
    ),
    "A1": AblationMode(
        ablation_id="A1",
        description="Component2 without bridge scoring and without recovery",
        use_component2=True,
        enable_bridge_scoring=False,
        enable_recovery=False,
    ),
    "A2": AblationMode(
        ablation_id="A2",
        description="Component2 with bridge scoring and without recovery",
        use_component2=True,
        enable_bridge_scoring=True,
        enable_recovery=False,
    ),
    "A3": AblationMode(
        ablation_id="A3",
        description="Component2 full: bridge scoring plus one-step recovery",
        use_component2=True,
        enable_bridge_scoring=True,
        enable_recovery=True,
    ),
    "A4": AblationMode(
        ablation_id="A4",
        description="Component2 full without PV masks (uniform edge weights)",
        use_component2=True,
        enable_bridge_scoring=True,
        enable_recovery=True,
        disable_pv_masks=True,
    ),
    "A5": AblationMode(
        ablation_id="A5",
        description="Stress test: Component2 full with bridge-like S triples dropped",
        use_component2=True,
        enable_bridge_scoring=True,
        enable_recovery=True,
        stress_drop_bridge_triples=True,
    ),
    "A6": AblationMode(
        ablation_id="A6",
        description="Stress test: Component2 full with injected high-rel S distractors",
        use_component2=True,
        enable_bridge_scoring=True,
        enable_recovery=True,
        stress_add_distractors=3,
    ),
}


def _make_anchor_adapter(config: Component2Config) -> FactKGAnchorAdapter:
    return FactKGAnchorAdapter(
        factkg_dir=Path(config.factkg_data_dir),
        split_pickles={
            "train": config.factkg_train_pickle,
            "val": config.factkg_val_pickle,
            "test": config.factkg_test_pickle,
        },
    )


def _pred_label_from_probs(prob_supported: float) -> str:
    return "SUPPORTED" if float(prob_supported) >= 0.5 else "REFUTED"


def _serialize_recovered_edges(triples_before: Sequence[EvidenceTriple], recovered_ids: Sequence[str]) -> List[Dict[str, object]]:
    by_id = {triple.evidence_id: triple for triple in triples_before}
    rows: List[Dict[str, object]] = []
    for evidence_id in recovered_ids:
        triple = by_id.get(evidence_id)
        if triple is None:
            continue
        rows.append(
            {
                "evidence_id": triple.evidence_id,
                "raw_triple": [
                    str(triple.raw_triple[0]),
                    str(triple.raw_triple[1]),
                    str(triple.raw_triple[2]),
                ],
            }
        )
    return rows


def _triple_connects_different_components(
    triple: EvidenceTriple,
    component_map: Mapping[str, int],
) -> bool:
    src, _, dst = triple.raw_triple
    src_comp = component_map.get(str(src), f"missing:{src}")
    dst_comp = component_map.get(str(dst), f"missing:{dst}")
    return src_comp != dst_comp


def _drop_bridge_like_s_triples(claim: ClaimRecord) -> Tuple[ClaimRecord, Tuple[str, ...]]:
    component_map = active_component_map(claim.triples)
    removed_ids: List[str] = []
    kept: List[EvidenceTriple] = []
    for triple in claim.triples:
        if triple.pool == "S" and _triple_connects_different_components(triple, component_map):
            removed_ids.append(str(triple.evidence_id))
            continue
        kept.append(triple)
    return replace(claim, triples=tuple(kept)), tuple(removed_ids)


def _add_s_distractors(claim: ClaimRecord, count: int) -> Tuple[ClaimRecord, Tuple[str, ...]]:
    if count <= 0:
        return claim, ()

    used_ids = {str(triple.evidence_id) for triple in claim.triples}
    anchor_candidates = [str(entity) for entity in claim.entity_set if str(entity).strip()]
    if not anchor_candidates:
        anchor_candidates = [str(triple.raw_triple[0]) for triple in claim.triples if str(triple.raw_triple[0]).strip()]
    if not anchor_candidates:
        anchor_candidates = ["stress_anchor"]

    triples = list(claim.triples)
    added_ids: List[str] = []
    for idx in range(int(count)):
        source = anchor_candidates[idx % len(anchor_candidates)]
        target = f"{source}__stress_noise_{idx}"
        evidence_id = f"{claim.claim_id}__stress_distractor_{idx}"
        suffix = 1
        while evidence_id in used_ids:
            evidence_id = f"{claim.claim_id}__stress_distractor_{idx}_{suffix}"
            suffix += 1
        used_ids.add(evidence_id)
        triples.append(
            EvidenceTriple(
                evidence_id=evidence_id,
                raw_triple=(source, "r_stress_distractor", target),
                pool="S",
                p_ent=0.90,
                p_con=0.07,
                p_neu=0.03,
                metadata={"stress_distractor": True},
            )
        )
        added_ids.append(evidence_id)

    return replace(claim, triples=tuple(triples)), tuple(added_ids)


def _apply_stress_profile(
    *,
    claim: ClaimRecord,
    mode: AblationMode,
) -> Tuple[ClaimRecord, Tuple[str, ...], Tuple[str, ...]]:
    stressed = claim
    removed_bridge_ids: Tuple[str, ...] = ()
    added_distractor_ids: Tuple[str, ...] = ()

    if mode.stress_drop_bridge_triples:
        stressed, removed_bridge_ids = _drop_bridge_like_s_triples(stressed)
    if mode.stress_add_distractors > 0:
        stressed, added_distractor_ids = _add_s_distractors(stressed, mode.stress_add_distractors)
    return stressed, removed_bridge_ids, added_distractor_ids


def _build_summary_table(
    mode_order: Sequence[str],
    per_mode_metrics: Mapping[str, Mapping[str, object]],
    tau_abstain: float,
) -> str:
    header = [
        "Ablation",
        "Accuracy",
        "AURC (sweep)",
        f"Coverage@tau={tau_abstain:.2f}",
        f"Risk@tau={tau_abstain:.2f}",
        f"Accuracy_on_answered@tau={tau_abstain:.2f}",
        "Recovery trigger rate",
        "Mean delta_conn",
        "Mean ESI before/after",
    ]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for ablation_id in mode_order:
        metrics = per_mode_metrics.get(ablation_id, {})
        selective = metrics.get("selective_prediction", {})
        sweep = selective.get("s1_sweep", {}) if isinstance(selective, dict) else {}
        fixed = selective.get("s2_fixed_tau", {}) if isinstance(selective, dict) else {}
        recovery = metrics.get("recovery", {}) if isinstance(metrics, dict) else {}

        def _fmt(val: object) -> str:
            if val is None:
                return "-"
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        row = [
            ablation_id,
            _fmt(metrics.get("accuracy")),
            _fmt(sweep.get("aurc")),
            _fmt(fixed.get("coverage")),
            _fmt(fixed.get("risk")),
            _fmt(fixed.get("accuracy_on_answered")),
            _fmt(recovery.get("recovery_trigger_rate")),
            _fmt(recovery.get("mean_delta_conn")),
            _fmt(recovery.get("mean_esi_before_after")),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _run_single_mode(
    *,
    mode: AblationMode,
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    tau_abstain: float,
    plot_writer: PlotWriter,
    run_dir: Path,
    output_tag: str | None = None,
) -> Dict[str, object]:
    """Run one ablation mode over selected claims and write outputs."""

    mode_output_id = mode.ablation_id if output_tag is None else str(output_tag)

    anchor_selector = AnchorSelector(config=config)
    anchor_adapter = _make_anchor_adapter(config=config)

    # Fresh per-mode state to avoid cross-mode relation-vocab leakage.
    builder_active = GraphBuilder(include_pools=("A", "C"))
    builder_full = GraphBuilder(include_pools=("A", "S", "C"))
    reasoner_config = replace(config, disable_pv_masks=bool(mode.disable_pv_masks))
    reasoner = HybridMaskedDualStreamReasoner(config=reasoner_config)
    bridge_rescue = BridgeRescuePPR(config=config)
    recovery_policy = BridgeRecoveryPolicy(config=config)

    prediction_rows: List[Dict[str, object]] = []

    bridge_max_values: List[float] = []

    top_bridge_cross_component_total = 0
    top_bridge_cross_component_hits = 0

    for claim in claims:
        claim_for_mode, removed_bridge_ids, added_distractor_ids = _apply_stress_profile(
            claim=claim,
            mode=mode,
        )

        injection = anchor_adapter.inject_claim(claim_for_mode)
        claim_with_anchors = injection.claim
        anchor_source = injection.anchor_source

        anchors = anchor_selector.select_anchors(
            claim_text=claim_with_anchors.claim_text,
            triples=claim_with_anchors.triples,
            seed_entities=claim_with_anchors.entity_set,
        )

        counts = pool_counts(claim_with_anchors.triples)
        conn_before = float(connectivity_for_active(claim_with_anchors.triples, anchors))
        esi_before = claim_with_anchors.sufficiency.get("esi_geom")
        if esi_before is None:
            esi_before = estimate_esi_geom(claim_with_anchors.triples, anchors)
        esi_before = float(esi_before)

        pred_label = "SUPPORTED"
        prob_supported = 0.5
        confidence_before = 0.5
        confidence_after = 0.5

        conn_after = conn_before
        esi_after = esi_before

        bridge_top1 = None
        bridge_top5: List[float] = []
        top_bridge_cross_component = None
        rpi_rel_at_k = None
        rpi_bridge_at_k = None
        delta_conn_bridge_at_k = None
        selected_bridge_ids: List[str] = []
        recovered_edge_ids: List[str] = []
        recovered_edges: List[Dict[str, object]] = []

        if mode.use_component2:
            graph_before = builder_active.build(
                claim_id=claim_with_anchors.claim_id,
                triples=claim_with_anchors.triples,
                anchors=anchors,
            )
            out_before = reasoner.forward_graph(graph_before)
            prob_supported_before = float(out_before["probs"][0])
            confidence_before = float(np.max(out_before["probs"]))

            pred_label = _pred_label_from_probs(prob_supported_before)
            prob_supported = prob_supported_before
            confidence_after = confidence_before

            triples_after: Sequence[EvidenceTriple] = claim_with_anchors.triples

            if mode.enable_bridge_scoring:
                full_graph = builder_full.build(
                    claim_id=claim_with_anchors.claim_id,
                    triples=claim_with_anchors.triples,
                    anchors=anchors,
                )
                node_scores, _ = bridge_rescue.compute_node_scores_with_diagnostics(full_graph)
                bridge_scores = bridge_rescue.compute_bridge_scores(full_graph, node_scores=node_scores)
                scores_s = [row for row in bridge_scores if row.pool == "S"]
                bridge_top1, bridge_top5 = summarize_bridge_top_values(scores_s)

                if bridge_top1 is not None:
                    bridge_max_values.append(float(bridge_top1))

                if scores_s:
                    top_bridge = max(scores_s, key=lambda row: row.bridge_bonus_capped)
                    comp_map = active_component_map(claim_with_anchors.triples)
                    top_bridge_cross_component = bridge_edge_connects_different_components(top_bridge, comp_map)
                    top_bridge_cross_component_total += 1
                    if top_bridge_cross_component:
                        top_bridge_cross_component_hits += 1

                recovery_metrics = recovery_policy.evaluate(
                    triples=claim_with_anchors.triples,
                    anchors=anchors,
                    bridge_scores=bridge_scores,
                    esi_geom=esi_before,
                )
                rpi_rel_at_k = float(recovery_metrics.rpi_rel_at_k)
                rpi_bridge_at_k = float(recovery_metrics.rpi_bridge_at_k)
                delta_conn_bridge_at_k = float(recovery_metrics.delta_conn_bridge_at_k)
                selected_bridge_ids = list(recovery_metrics.selected_bridge_ids)

                if mode.enable_recovery and recovery_metrics.should_recover:
                    recovered = recovery_policy.apply_recovery(
                        triples=claim_with_anchors.triples,
                        selected_bridge_ids=recovery_metrics.selected_bridge_ids,
                    )
                    triples_after = recovered.recovered_triples
                    recovered_edge_ids = list(recovered.moved_evidence_ids)
                    recovered_edges = _serialize_recovered_edges(claim_with_anchors.triples, recovered_edge_ids)

                    conn_after = float(connectivity_for_active(triples_after, anchors))
                    esi_after = float(estimate_esi_geom(triples_after, anchors))

                    graph_after = builder_active.build(
                        claim_id=claim_with_anchors.claim_id,
                        triples=triples_after,
                        anchors=anchors,
                    )
                    out_after = reasoner.forward_graph(graph_after)
                    prob_supported = float(out_after["probs"][0])
                    confidence_after = float(np.max(out_after["probs"]))
                    pred_label = _pred_label_from_probs(prob_supported)

            # A1 or A2 (or A3 no-trigger) keeps before-pass output.

        else:
            pred_label, prob_supported, confidence_after = baseline_predict_from_component1(claim_with_anchors.triples)
            confidence_before = confidence_after

        pred_prob = float(prob_supported if pred_label == "SUPPORTED" else 1.0 - prob_supported)
        final_esi_for_selective = float(esi_after if mode.enable_recovery else esi_before)
        abstain = float(
            abstain_score(
                probs=[float(prob_supported), float(1.0 - prob_supported)],
                esi_geom=final_esi_for_selective,
                alpha=config.abstain_alpha,
            )
        )
        abstained = bool(abstain > float(tau_abstain))

        gold_idx = claim_label_to_index(claim_with_anchors.label)
        pred_idx = claim_label_to_index(pred_label)
        correctness = None if gold_idx is None or pred_idx is None else int(gold_idx == pred_idx)

        prediction_rows.append(
            {
                "claim_id": claim_with_anchors.claim_id,
                "claim_text": claim_with_anchors.claim_text,
                "gold_label": claim_with_anchors.label,
                "pred_label": pred_label,
                "pred_prob": pred_prob,
                "confidence": float(max(pred_prob, 1.0 - pred_prob)),
                "confidence_before": float(confidence_before),
                "confidence_after": float(confidence_after),
                "abstain_score": abstain,
                "abstained": abstained,
                "esi_geom": final_esi_for_selective,
                "esi_geom_before": float(esi_before),
                "esi_geom_after": float(esi_after),
                "conn_a": float(conn_before),
                "conn_after": float(conn_after),
                "anchors": list(anchors),
                "anchor_source": anchor_source,
                "count_A": int(counts["A"]),
                "count_S": int(counts["S"]),
                "count_C": int(counts["C"]),
                "bridge_top1_bonus_capped": bridge_top1,
                "bridge_top5_bonus_capped": bridge_top5,
                "top_bridge_connects_different_anchor_components": top_bridge_cross_component,
                "rpi_rel_at_k": rpi_rel_at_k,
                "rpi_bridge_at_k": rpi_bridge_at_k,
                "delta_conn_bridge_at_k": delta_conn_bridge_at_k,
                "selected_bridge_ids": selected_bridge_ids,
                "recovered_edge_ids": recovered_edge_ids,
                "recovered_edges": recovered_edges,
                "delta_conn": float(conn_after - conn_before),
                "stress_removed_bridge_ids": list(removed_bridge_ids),
                "stress_added_distractor_ids": list(added_distractor_ids),
                "num_stress_removed_bridge_triples": int(len(removed_bridge_ids)),
                "num_stress_added_distractors": int(len(added_distractor_ids)),
                "is_correct": correctness,
            }
        )

    pred_labels = [str(row["pred_label"]) for row in prediction_rows]
    gold_labels = [row.get("gold_label") for row in prediction_rows]
    binary_metrics: BinaryMetrics = compute_binary_metrics(gold_labels=gold_labels, pred_labels=pred_labels)

    labeled_rows = [row for row in prediction_rows if row.get("is_correct") is not None]
    selective_s1 = evaluate_selective_prediction(
        abstain_scores=[float(row["abstain_score"]) for row in labeled_rows],
        is_correct=[bool(row["is_correct"]) for row in labeled_rows],
        tau_abstain=None,
        compute_curve=True,
    )
    selective_s2 = evaluate_selective_prediction(
        abstain_scores=[float(row["abstain_score"]) for row in labeled_rows],
        is_correct=[bool(row["is_correct"]) for row in labeled_rows],
        tau_abstain=float(tau_abstain),
        compute_curve=False,
    )

    risk_cov_points = selective_s1.get("points", [])
    plot_writer.risk_coverage(mode_output_id, risk_cov_points)
    plot_writer.accuracy_vs_coverage(mode_output_id, risk_cov_points)

    if mode.enable_bridge_scoring:
        plot_writer.histogram(mode_output_id, "bridgebonus", bridge_max_values)
    if mode.enable_recovery:
        plot_writer.histogram(mode_output_id, "delta_conn", [float(row["delta_conn"]) for row in prediction_rows])

    if mode.enable_bridge_scoring:
        bridge_correctness_x = [
            float(row["bridge_top1_bonus_capped"])
            for row in labeled_rows
            if row.get("bridge_top1_bonus_capped") is not None
        ]
        bridge_correctness_y = [
            float(row["is_correct"])
            for row in labeled_rows
            if row.get("bridge_top1_bonus_capped") is not None
        ]
        bridge_conf_x = [
            float(row["bridge_top1_bonus_capped"])
            for row in prediction_rows
            if row.get("bridge_top1_bonus_capped") is not None
        ]
        bridge_conf_y = [float(row["confidence"]) for row in prediction_rows if row.get("bridge_top1_bonus_capped") is not None]
        corr_max_bridge_correctness = spearman_correlation(bridge_correctness_x, bridge_correctness_y)
        corr_max_bridge_conf = spearman_correlation(bridge_conf_x, bridge_conf_y)
    else:
        corr_max_bridge_correctness = None
        corr_max_bridge_conf = None

    if mode.enable_recovery:
        delta_corr_x = [
            float(row["delta_conn_bridge_at_k"])
            for row in labeled_rows
            if row.get("delta_conn_bridge_at_k") is not None
        ]
        delta_corr_y = [
            float(row["is_correct"])
            for row in labeled_rows
            if row.get("delta_conn_bridge_at_k") is not None
        ]
        corr_delta_correctness = spearman_correlation(delta_corr_x, delta_corr_y)
    else:
        corr_delta_correctness = None

    bridge_cross_fraction = None
    if top_bridge_cross_component_total > 0:
        bridge_cross_fraction = float(top_bridge_cross_component_hits) / float(top_bridge_cross_component_total)

    metrics_payload: Dict[str, object] = {
        "ablation_id": mode.ablation_id,
        "output_tag": mode_output_id,
        "description": mode.description,
        "recovery_tau_esi": float(config.recovery_tau_esi),
        "disable_pv_masks": bool(mode.disable_pv_masks),
        "mask_coefficients": {
            "mask_sup_alpha": float(config.mask_sup_scale),
            "mask_sup_beta": float(config.mask_sup_bias),
            "mask_ref_alpha": float(config.mask_ref_scale),
            "mask_ref_beta": float(config.mask_ref_bias),
        },
        "num_claims": len(prediction_rows),
        "num_labeled_claims": int(binary_metrics.num_labeled),
        "accuracy": float(binary_metrics.accuracy),
        "macro_f1": float(binary_metrics.macro_f1),
        "selective_prediction": {
            "s1_sweep": {
                "aurc": selective_s1.get("aurc"),
                "num_points": len(risk_cov_points),
            },
            "s2_fixed_tau": selective_s2.get("operating_point"),
        },
        "bridge_correlation": {
            "spearman_max_bridgebonus_vs_correctness": corr_max_bridge_correctness,
            "spearman_max_bridgebonus_vs_confidence": corr_max_bridge_conf,
            "spearman_delta_conn_bridge_at_k_vs_correctness": corr_delta_correctness,
            "top_bridge_connects_different_anchor_components_fraction": bridge_cross_fraction,
            "top_bridge_connects_different_anchor_components_hits": top_bridge_cross_component_hits,
            "top_bridge_connects_different_anchor_components_total": top_bridge_cross_component_total,
        },
        "recovery": {
            "recovery_trigger_rate": (
                float(sum(1 for row in prediction_rows if len(row["recovered_edge_ids"]) > 0)) / float(len(prediction_rows))
                if prediction_rows
                else 0.0
            ),
            "mean_delta_conn": safe_mean(float(row["delta_conn"]) for row in prediction_rows),
            "mean_esi_before": safe_mean(float(row["esi_geom_before"]) for row in prediction_rows),
            "mean_esi_after": safe_mean(float(row["esi_geom_after"]) for row in prediction_rows),
            "mean_esi_before_after": (
                f"{safe_mean(float(row['esi_geom_before']) for row in prediction_rows):.4f}"
                f"/{safe_mean(float(row['esi_geom_after']) for row in prediction_rows):.4f}"
            ),
        },
        "stress": {
            "drop_bridge_triples": bool(mode.stress_drop_bridge_triples),
            "add_distractors_per_claim": int(mode.stress_add_distractors),
            "claims_with_stress_mutation": int(
                sum(
                    1
                    for row in prediction_rows
                    if int(row["num_stress_removed_bridge_triples"]) > 0 or int(row["num_stress_added_distractors"]) > 0
                )
            ),
            "mean_removed_bridge_triples": safe_mean(
                float(row["num_stress_removed_bridge_triples"]) for row in prediction_rows
            ),
            "mean_added_distractors": safe_mean(float(row["num_stress_added_distractors"]) for row in prediction_rows),
        },
    }

    write_jsonl(run_dir / f"predictions_{mode_output_id}.jsonl", prediction_rows)
    write_json(run_dir / f"metrics_{mode_output_id}.json", metrics_payload)
    return metrics_payload


def _parse_tau_values(tau_esi_values: Sequence[float]) -> Tuple[float, ...]:
    parsed: List[float] = []
    for tau in tau_esi_values:
        tau_v = float(tau)
        if not np.isfinite(tau_v):
            raise ValueError(f"tau_esi value must be finite, got {tau}")
        if tau_v <= 0.0 or tau_v >= 1.0:
            raise ValueError(f"tau_esi value must be in (0, 1), got {tau_v}")
        parsed.append(tau_v)
    if not parsed:
        raise ValueError("tau_esi_values must not be empty")
    return tuple(parsed)


def _tau_output_tag(mode_id: str, tau_esi: float) -> str:
    tau_label = f"{float(tau_esi):.2f}".replace(".", "p")
    return f"{mode_id}_tau{tau_label}"


def _build_tau_esi_summary_table(
    rows: Sequence[Mapping[str, object]],
    tau_abstain: float,
) -> str:
    header = [
        "tau_esi",
        "Accuracy",
        "AURC (sweep)",
        f"Coverage@tau={tau_abstain:.2f}",
        f"Risk@tau={tau_abstain:.2f}",
        f"Accuracy_on_answered@tau={tau_abstain:.2f}",
        "Recovery trigger rate",
        "Mean delta_conn",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['tau_esi']):.2f}",
                    f"{float(row['accuracy']):.4f}",
                    f"{float(row['aurc']):.4f}" if row["aurc"] is not None else "-",
                    f"{float(row['coverage']):.4f}" if row["coverage"] is not None else "-",
                    f"{float(row['risk']):.4f}" if row["risk"] is not None else "-",
                    f"{float(row['accuracy_on_answered']):.4f}" if row["accuracy_on_answered"] is not None else "-",
                    f"{float(row['recovery_trigger_rate']):.4f}",
                    f"{float(row['mean_delta_conn']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_tau_esi_sweep(
    *,
    split: str,
    claims_jsonl: Path,
    n_claims: int = 2000,
    seed: int = 13,
    tau_abstain: float = 0.60,
    output_root: Path = Path("logs/ablations/component2"),
    run_id: str | None = None,
    tau_esi_values: Sequence[float] = (0.2, 0.3, 0.4),
    mode_id: str = "A3",
    config: Component2Config = DEFAULT_CONFIG,
) -> Path:
    """Run tau_esi sensitivity sweep and write tradeoff summary artifacts."""

    mode = ABLATION_MODES.get(mode_id)
    if mode is None:
        raise ValueError(f"Unknown mode_id '{mode_id}'. Expected one of: {sorted(ABLATION_MODES)}")
    if not mode.enable_recovery:
        raise ValueError(f"mode_id '{mode_id}' must enable recovery for tau_esi sensitivity sweeps.")

    if n_claims < 1:
        raise ValueError(f"n_claims must be >= 1, got {n_claims}")

    tau_values = _parse_tau_values(tau_esi_values)
    all_claims, input_source = load_claims_for_ablation(claims_jsonl)
    subset = select_deterministic_subset(all_claims, n_claims=n_claims, seed=seed)

    run_dir = make_ablation_run_dir(output_root=output_root, run_id=run_id)
    plots_dir = run_dir / "plots"
    plot_writer = PlotWriter(plots_dir)

    write_selected_claim_ids(run_dir / "selected_claim_ids.txt", subset.claims)
    write_json(
        run_dir / "subset_selection.json",
        {
            "split": split,
            "seed": int(seed),
            "n_claims_requested": int(n_claims),
            "n_claims_selected": len(subset.claims),
            "selected_indices": list(subset.indices),
            "selected_claim_ids_file": "selected_claim_ids.txt",
        },
    )

    repo_root = Path(__file__).resolve().parents[2]
    commit_hash = git_commit_hash(repo_root)
    run_config = {
        "git_commit_hash": commit_hash,
        "experiment": "tau_esi_sweep",
        "mode_id": mode_id,
        "tau_esi_values": [float(tau) for tau in tau_values],
        "split": split,
        "claims_jsonl": str(claims_jsonl),
        "input_source": input_source,
        "n_claims": int(n_claims),
        "n_claims_selected": len(subset.claims),
        "seed": int(seed),
        "tau_abstain_modes": {
            "s1": {"tau_abstain": None, "description": "risk-coverage sweep + AURC"},
            "s2": {"tau_abstain": float(tau_abstain), "description": "fixed operating point"},
        },
        "component2_tunables_base": config.to_snapshot_dict(),
    }
    write_json(run_dir / "run_config.json", run_config)

    sweep_rows: List[Dict[str, object]] = []
    per_tau_metrics: Dict[str, Mapping[str, object]] = {}
    for tau_esi in tau_values:
        tau_config = replace(config, recovery_tau_esi=float(tau_esi))
        output_tag = _tau_output_tag(mode_id=mode_id, tau_esi=tau_esi)
        metrics = _run_single_mode(
            mode=mode,
            claims=subset.claims,
            config=tau_config,
            tau_abstain=tau_abstain,
            plot_writer=plot_writer,
            run_dir=run_dir,
            output_tag=output_tag,
        )
        per_tau_metrics[f"{tau_esi:.2f}"] = metrics

        selective = metrics.get("selective_prediction", {})
        s1 = selective.get("s1_sweep", {}) if isinstance(selective, dict) else {}
        s2 = selective.get("s2_fixed_tau", {}) if isinstance(selective, dict) else {}
        recovery = metrics.get("recovery", {}) if isinstance(metrics, dict) else {}

        sweep_rows.append(
            {
                "tau_esi": float(tau_esi),
                "output_tag": output_tag,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "aurc": s1.get("aurc"),
                "coverage": s2.get("coverage"),
                "risk": s2.get("risk"),
                "accuracy_on_answered": s2.get("accuracy_on_answered"),
                "recovery_trigger_rate": float(recovery.get("recovery_trigger_rate", 0.0)),
                "mean_delta_conn": float(recovery.get("mean_delta_conn", 0.0)),
            }
        )

    sweep_rows = sorted(sweep_rows, key=lambda row: float(row["tau_esi"]))
    best_accuracy_row = max(sweep_rows, key=lambda row: float(row["accuracy"]))
    best_risk_row = min(
        [row for row in sweep_rows if row["risk"] is not None],
        key=lambda row: float(row["risk"]),
        default=None,
    )

    summary_payload = {
        "mode_id": mode_id,
        "tau_esi_values": [float(tau) for tau in tau_values],
        "rows": sweep_rows,
        "best_accuracy": {
            "tau_esi": float(best_accuracy_row["tau_esi"]),
            "accuracy": float(best_accuracy_row["accuracy"]),
        },
        "best_risk": (
            {
                "tau_esi": float(best_risk_row["tau_esi"]),
                "risk": float(best_risk_row["risk"]),
            }
            if best_risk_row is not None
            else None
        ),
    }
    write_json(run_dir / "tau_esi_sweep.json", summary_payload)
    write_json(run_dir / "summary_metrics.json", {"tau_esi_sweep": summary_payload, "per_tau_metrics": per_tau_metrics})

    summary_table = _build_tau_esi_summary_table(rows=sweep_rows, tau_abstain=float(tau_abstain))
    (run_dir / "tau_esi_sweep.md").write_text(summary_table, encoding="utf-8")
    return run_dir


def run_ablation_study(
    *,
    split: str,
    claims_jsonl: Path,
    n_claims: int = 2000,
    seed: int = 13,
    tau_abstain: float = 0.60,
    output_root: Path = Path("logs/ablations/component2"),
    run_id: str | None = None,
    ablation_ids: Sequence[str] = ("A0", "A1", "A2", "A3", "A4"),
    config: Component2Config = DEFAULT_CONFIG,
    tune_mask_coeffs: bool = False,
    mask_tune_strategy: str = "two_stage",
    mask_fit_mode: str = "A1",
    mask_validate_mode: str = "A3",
    mask_grid_size: int = 7,
    mask_alpha_min: float = 1.0,
    mask_alpha_max: float = 7.0,
    mask_beta_min: float = -3.0,
    mask_beta_max: float = 0.0,
    mask_tune_claims: int = 200,
    mask_local_refine: bool = False,
) -> Path:
    """Run selected ablations and write paper-grade outputs under one run directory."""

    selected_modes: List[AblationMode] = []
    for ablation_id in ablation_ids:
        mode = ABLATION_MODES.get(ablation_id)
        if mode is None:
            raise ValueError(f"Unknown ablation_id '{ablation_id}'. Expected one of: {sorted(ABLATION_MODES)}")
        selected_modes.append(mode)

    if n_claims < 1:
        raise ValueError(f"n_claims must be >= 1, got {n_claims}")

    all_claims, input_source = load_claims_for_ablation(claims_jsonl)
    subset = select_deterministic_subset(all_claims, n_claims=n_claims, seed=seed)

    run_dir = make_ablation_run_dir(output_root=output_root, run_id=run_id)
    plots_dir = run_dir / "plots"
    plot_writer = PlotWriter(plots_dir)

    write_selected_claim_ids(run_dir / "selected_claim_ids.txt", subset.claims)
    write_json(
        run_dir / "subset_selection.json",
        {
            "split": split,
            "seed": int(seed),
            "n_claims_requested": int(n_claims),
            "n_claims_selected": len(subset.claims),
            "selected_indices": list(subset.indices),
            "selected_claim_ids_file": "selected_claim_ids.txt",
        },
    )

    effective_config = config
    mask_tuning_report: Dict[str, object] = {"enabled": False}
    if tune_mask_coeffs:
        tuning_settings = MaskTuningSettings(
            strategy=str(mask_tune_strategy),
            fit_mode=str(mask_fit_mode),
            validate_mode=str(mask_validate_mode),
            grid_size=int(mask_grid_size),
            alpha_min=float(mask_alpha_min),
            alpha_max=float(mask_alpha_max),
            beta_min=float(mask_beta_min),
            beta_max=float(mask_beta_max),
            tune_claims=int(mask_tune_claims),
            local_refine=bool(mask_local_refine),
        )
        tuning_output = tune_mask_coefficients(
            claims=subset.claims,
            config=config,
            settings=tuning_settings,
        )
        effective_config = tuning_output["best_config"]
        mask_tuning_report = dict(tuning_output["report"])
        write_json(run_dir / "best_mask_params.json", tuning_output["best_params"])
        write_json(run_dir / "mask_tuning_report.json", tuning_output["report"])

    repo_root = Path(__file__).resolve().parents[2]
    commit_hash = git_commit_hash(repo_root)

    run_config = {
        "git_commit_hash": commit_hash,
        "ablation_id": "all",
        "ablation_ids": [mode.ablation_id for mode in selected_modes],
        "split": split,
        "claims_jsonl": str(claims_jsonl),
        "input_source": input_source,
        "n_claims": int(n_claims),
        "n_claims_selected": len(subset.claims),
        "seed": int(seed),
        "tau_abstain_modes": {
            "s1": {"tau_abstain": None, "description": "risk-coverage sweep + AURC"},
            "s2": {"tau_abstain": float(tau_abstain), "description": "fixed operating point"},
        },
        "component2_tunables": effective_config.to_snapshot_dict(),
        "mask_tuning": mask_tuning_report,
        "baseline": {
            "source": "component1_pool_mass_vote",
            "note": "Repo has no dedicated pre-Component2 inference entrypoint for claim-log subsets; using deterministic Component1 A/C mass baseline.",
        },
    }
    write_json(run_dir / "run_config.json", run_config)

    per_mode_metrics: Dict[str, Mapping[str, object]] = {}
    for mode in selected_modes:
        per_mode_metrics[mode.ablation_id] = _run_single_mode(
            mode=mode,
            claims=subset.claims,
            config=effective_config,
            tau_abstain=tau_abstain,
            plot_writer=plot_writer,
            run_dir=run_dir,
        )

    summary_table = _build_summary_table(
        mode_order=[mode.ablation_id for mode in selected_modes],
        per_mode_metrics=per_mode_metrics,
        tau_abstain=float(tau_abstain),
    )
    (run_dir / "summary_table.md").write_text(summary_table, encoding="utf-8")

    write_json(run_dir / "summary_metrics.json", {"modes": per_mode_metrics})
    return run_dir


__all__ = [
    "ABLATION_MODES",
    "AblationMode",
    "run_ablation_study",
    "run_tau_esi_sweep",
]
