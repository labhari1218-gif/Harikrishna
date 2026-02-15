"""Deterministic mask-coefficient tuning helpers for Component 2 ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .ablation_utils import compute_binary_metrics, estimate_esi_geom
from .anchor_adapter_factkg import FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .bridge_rescue import BridgeRescuePPR
from .config import Component2Config, DEFAULT_CONFIG
from .graph_builder import GraphBuilder
from .recovery import BridgeRecoveryPolicy
from .reasoner import HybridMaskedDualStreamReasoner
from .selective import abstain_score, evaluate_selective_prediction
from .types import ClaimRecord, claim_label_to_index


@dataclass(frozen=True)
class MaskParams:
    """Mask coefficients in alpha/beta naming."""

    mask_sup_alpha: float
    mask_sup_beta: float
    mask_ref_alpha: float
    mask_ref_beta: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mask_sup_alpha": float(self.mask_sup_alpha),
            "mask_sup_beta": float(self.mask_sup_beta),
            "mask_ref_alpha": float(self.mask_ref_alpha),
            "mask_ref_beta": float(self.mask_ref_beta),
        }


@dataclass(frozen=True)
class MaskTuningSettings:
    """Tuning strategy and search-space configuration."""

    strategy: str = "two_stage"
    fit_mode: str = "A1"
    validate_mode: str = "A3"
    grid_size: int = 7
    alpha_min: float = 1.0
    alpha_max: float = 7.0
    beta_min: float = -3.0
    beta_max: float = 0.0
    tune_claims: int = 200
    local_refine: bool = False


def params_from_config(config: Component2Config) -> MaskParams:
    """Extract alpha/beta params from config scale/bias fields."""

    return MaskParams(
        mask_sup_alpha=float(config.mask_sup_scale),
        mask_sup_beta=float(config.mask_sup_bias),
        mask_ref_alpha=float(config.mask_ref_scale),
        mask_ref_beta=float(config.mask_ref_bias),
    )


def config_with_mask_params(config: Component2Config, params: Mapping[str, float]) -> Component2Config:
    """Return a config with updated mask coefficients."""

    return replace(
        config,
        mask_sup_scale=float(params["mask_sup_alpha"]),
        mask_sup_bias=float(params["mask_sup_beta"]),
        mask_ref_scale=float(params["mask_ref_alpha"]),
        mask_ref_bias=float(params["mask_ref_beta"]),
    )


def _make_anchor_adapter(config: Component2Config) -> FactKGAnchorAdapter:
    return FactKGAnchorAdapter(
        factkg_dir=Path(config.factkg_data_dir),
        split_pickles={
            "train": config.factkg_train_pickle,
            "val": config.factkg_val_pickle,
            "test": config.factkg_test_pickle,
        },
    )


def _build_grid(low: float, high: float, size: int) -> tuple[float, ...]:
    if size < 1:
        raise ValueError(f"grid_size must be >= 1, got {size}")
    if high < low:
        raise ValueError(f"invalid range: min {low} must be <= max {high}")
    if size == 1:
        return (float(low),)
    values = np.linspace(float(low), float(high), int(size), dtype=np.float64)
    return tuple(float(value) for value in values.tolist())


def _l1_distance(params: MaskParams, defaults: MaskParams) -> float:
    return float(
        abs(params.mask_sup_alpha - defaults.mask_sup_alpha)
        + abs(params.mask_sup_beta - defaults.mask_sup_beta)
        + abs(params.mask_ref_alpha - defaults.mask_ref_alpha)
        + abs(params.mask_ref_beta - defaults.mask_ref_beta)
    )


def _metrics_for_mode(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    mode: str,
) -> Dict[str, float]:
    if mode not in {"A1", "A3"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: A1, A3")

    anchor_selector = AnchorSelector(config=config)
    anchor_adapter = _make_anchor_adapter(config=config)

    builder_active = GraphBuilder(include_pools=("A", "C"))
    builder_full = GraphBuilder(include_pools=("A", "S", "C"))
    reasoner = HybridMaskedDualStreamReasoner(config=config)
    bridge_rescue = BridgeRescuePPR(config=config)
    recovery_policy = BridgeRecoveryPolicy(config=config)

    pred_labels = []
    gold_labels = []
    abstain_scores = []
    is_correct = []

    for claim in claims:
        injection = anchor_adapter.inject_claim(claim)
        claim_with_anchors = injection.claim

        anchors = anchor_selector.select_anchors(
            claim_text=claim_with_anchors.claim_text,
            triples=claim_with_anchors.triples,
            seed_entities=claim_with_anchors.entity_set,
        )

        esi_before = claim_with_anchors.sufficiency.get("esi_geom")
        if esi_before is None:
            esi_before = estimate_esi_geom(claim_with_anchors.triples, anchors)
        esi_before = float(esi_before)

        graph_before = builder_active.build(
            claim_id=claim_with_anchors.claim_id,
            triples=claim_with_anchors.triples,
            anchors=anchors,
        )
        out_before = reasoner.forward_graph(graph_before)

        prob_supported = float(out_before["probs"][0])
        final_esi = esi_before

        if mode == "A3":
            full_graph = builder_full.build(
                claim_id=claim_with_anchors.claim_id,
                triples=claim_with_anchors.triples,
                anchors=anchors,
            )
            node_scores, _ = bridge_rescue.compute_node_scores_with_diagnostics(full_graph)
            bridge_scores = bridge_rescue.compute_bridge_scores(full_graph, node_scores=node_scores)

            recovery_metrics = recovery_policy.evaluate(
                triples=claim_with_anchors.triples,
                anchors=anchors,
                bridge_scores=bridge_scores,
                esi_geom=esi_before,
            )
            if recovery_metrics.should_recover:
                recovered = recovery_policy.apply_recovery(
                    triples=claim_with_anchors.triples,
                    selected_bridge_ids=recovery_metrics.selected_bridge_ids,
                )
                triples_after = recovered.recovered_triples
                final_esi = float(estimate_esi_geom(triples_after, anchors))

                graph_after = builder_active.build(
                    claim_id=claim_with_anchors.claim_id,
                    triples=triples_after,
                    anchors=anchors,
                )
                out_after = reasoner.forward_graph(graph_after)
                prob_supported = float(out_after["probs"][0])

        pred_label = "SUPPORTED" if prob_supported >= 0.5 else "REFUTED"
        pred_idx = claim_label_to_index(pred_label)
        gold_idx = claim_label_to_index(claim_with_anchors.label)

        pred_labels.append(pred_label)
        gold_labels.append(claim_with_anchors.label)

        if pred_idx is not None and gold_idx is not None:
            is_correct.append(bool(pred_idx == gold_idx))
            abstain_scores.append(
                float(
                    abstain_score(
                        probs=[float(prob_supported), float(1.0 - prob_supported)],
                        esi_geom=float(final_esi),
                        alpha=config.abstain_alpha,
                    )
                )
            )

    binary = compute_binary_metrics(gold_labels=gold_labels, pred_labels=pred_labels)
    selective = evaluate_selective_prediction(
        abstain_scores=abstain_scores,
        is_correct=is_correct,
        tau_abstain=None,
        compute_curve=True,
    )

    aurc_raw = selective.get("aurc")
    aurc = float(aurc_raw) if aurc_raw is not None else float("inf")
    return {
        "accuracy": float(binary.accuracy),
        "aurc": aurc,
        "num_labeled": float(binary.num_labeled),
    }


def _is_better(
    candidate: Dict[str, object],
    best: Dict[str, object] | None,
    defaults: MaskParams,
) -> bool:
    if best is None:
        return True

    eps = 1e-12
    cand_acc = float(candidate["metrics"]["accuracy"])
    best_acc = float(best["metrics"]["accuracy"])
    if cand_acc > best_acc + eps:
        return True
    if cand_acc < best_acc - eps:
        return False

    cand_aurc = float(candidate["metrics"]["aurc"])
    best_aurc = float(best["metrics"]["aurc"])
    if cand_aurc < best_aurc - eps:
        return True
    if cand_aurc > best_aurc + eps:
        return False

    cand_dist = _l1_distance(candidate["params"], defaults)
    best_dist = _l1_distance(best["params"], defaults)
    if cand_dist < best_dist - eps:
        return True
    if cand_dist > best_dist + eps:
        return False

    cand_tuple = (
        candidate["params"].mask_sup_alpha,
        candidate["params"].mask_sup_beta,
        candidate["params"].mask_ref_alpha,
        candidate["params"].mask_ref_beta,
    )
    best_tuple = (
        best["params"].mask_sup_alpha,
        best["params"].mask_sup_beta,
        best["params"].mask_ref_alpha,
        best["params"].mask_ref_beta,
    )
    return cand_tuple < best_tuple


def _evaluate_candidate(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    mode: str,
    params: MaskParams,
) -> Dict[str, object]:
    candidate_config = config_with_mask_params(config, params.to_dict())
    metrics = _metrics_for_mode(claims=claims, config=candidate_config, mode=mode)
    return {
        "params": params,
        "metrics": metrics,
    }


def _run_independent_search(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    defaults: MaskParams,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    mode: str,
) -> Dict[str, object]:
    support_best: Dict[str, object] | None = None
    for alpha, beta in product(alpha_grid, beta_grid):
        candidate = _evaluate_candidate(
            claims=claims,
            config=config,
            mode=mode,
            params=MaskParams(
                mask_sup_alpha=float(alpha),
                mask_sup_beta=float(beta),
                mask_ref_alpha=defaults.mask_ref_alpha,
                mask_ref_beta=defaults.mask_ref_beta,
            ),
        )
        if _is_better(candidate, support_best, defaults=defaults):
            support_best = candidate

    if support_best is None:
        raise RuntimeError("Support-mask tuning failed to produce any candidate.")

    refute_best: Dict[str, object] | None = None
    support_params: MaskParams = support_best["params"]
    for alpha, beta in product(alpha_grid, beta_grid):
        candidate = _evaluate_candidate(
            claims=claims,
            config=config,
            mode=mode,
            params=MaskParams(
                mask_sup_alpha=support_params.mask_sup_alpha,
                mask_sup_beta=support_params.mask_sup_beta,
                mask_ref_alpha=float(alpha),
                mask_ref_beta=float(beta),
            ),
        )
        if _is_better(candidate, refute_best, defaults=defaults):
            refute_best = candidate

    if refute_best is None:
        raise RuntimeError("Refute-mask tuning failed to produce any candidate.")

    return {
        "support_best": {
            "params": support_best["params"].to_dict(),
            "metrics": support_best["metrics"],
        },
        "best": {
            "params": refute_best["params"].to_dict(),
            "metrics": refute_best["metrics"],
        },
        "best_params": refute_best["params"],
        "candidates_evaluated": int((len(alpha_grid) * len(beta_grid)) * 2),
    }


def _run_joint_search(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    defaults: MaskParams,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    mode: str,
) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    for sup_alpha, sup_beta, ref_alpha, ref_beta in product(alpha_grid, beta_grid, alpha_grid, beta_grid):
        candidate = _evaluate_candidate(
            claims=claims,
            config=config,
            mode=mode,
            params=MaskParams(
                mask_sup_alpha=float(sup_alpha),
                mask_sup_beta=float(sup_beta),
                mask_ref_alpha=float(ref_alpha),
                mask_ref_beta=float(ref_beta),
            ),
        )
        if _is_better(candidate, best, defaults=defaults):
            best = candidate

    if best is None:
        raise RuntimeError("Joint mask tuning failed to produce any candidate.")

    return {
        "best": {
            "params": best["params"].to_dict(),
            "metrics": best["metrics"],
        },
        "best_params": best["params"],
        "candidates_evaluated": int((len(alpha_grid) * len(beta_grid)) ** 2),
    }


def _neighborhood(center: float, step: float, low: float, high: float) -> tuple[float, ...]:
    if step <= 0.0:
        return (float(min(max(center, low), high)),)

    values = [
        float(min(max(center - step, low), high)),
        float(min(max(center, low), high)),
        float(min(max(center + step, low), high)),
    ]
    deduped = sorted(dict.fromkeys(values))
    return tuple(deduped)


def _run_local_refinement(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    defaults: MaskParams,
    center: MaskParams,
    mode: str,
    alpha_step: float,
    beta_step: float,
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
) -> Dict[str, object]:
    sup_alphas = _neighborhood(center.mask_sup_alpha, alpha_step, alpha_min, alpha_max)
    sup_betas = _neighborhood(center.mask_sup_beta, beta_step, beta_min, beta_max)
    ref_alphas = _neighborhood(center.mask_ref_alpha, alpha_step, alpha_min, alpha_max)
    ref_betas = _neighborhood(center.mask_ref_beta, beta_step, beta_min, beta_max)

    best: Dict[str, object] | None = None
    candidates_evaluated = 0
    for sup_alpha, sup_beta, ref_alpha, ref_beta in product(sup_alphas, sup_betas, ref_alphas, ref_betas):
        candidate = _evaluate_candidate(
            claims=claims,
            config=config,
            mode=mode,
            params=MaskParams(
                mask_sup_alpha=float(sup_alpha),
                mask_sup_beta=float(sup_beta),
                mask_ref_alpha=float(ref_alpha),
                mask_ref_beta=float(ref_beta),
            ),
        )
        candidates_evaluated += 1
        if _is_better(candidate, best, defaults=defaults):
            best = candidate

    if best is None:
        raise RuntimeError("Local refinement failed to produce any candidate.")

    return {
        "center_params": center.to_dict(),
        "sup_alphas": list(sup_alphas),
        "sup_betas": list(sup_betas),
        "ref_alphas": list(ref_alphas),
        "ref_betas": list(ref_betas),
        "best": {
            "params": best["params"].to_dict(),
            "metrics": best["metrics"],
        },
        "best_params": best["params"],
        "candidates_evaluated": int(candidates_evaluated),
    }


def _select_from_candidates(
    claims: Sequence[ClaimRecord],
    config: Component2Config,
    defaults: MaskParams,
    mode: str,
    candidates: Sequence[MaskParams],
) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    rows = []
    for params in candidates:
        candidate = _evaluate_candidate(claims=claims, config=config, mode=mode, params=params)
        rows.append(
            {
                "params": params.to_dict(),
                "metrics": candidate["metrics"],
            }
        )
        if _is_better(candidate, best, defaults=defaults):
            best = candidate

    if best is None:
        raise RuntimeError("Candidate selection failed to produce any candidate.")

    return {
        "mode": mode,
        "candidates": rows,
        "best": {
            "params": best["params"].to_dict(),
            "metrics": best["metrics"],
        },
        "best_params": best["params"],
        "candidates_evaluated": len(candidates),
    }


def tune_mask_coefficients(
    claims: Sequence[ClaimRecord],
    config: Component2Config = DEFAULT_CONFIG,
    settings: MaskTuningSettings = MaskTuningSettings(),
) -> Dict[str, object]:
    """Tune mask coefficients and return selected params plus diagnostics."""

    if len(claims) == 0:
        raise ValueError("claims must not be empty")
    if settings.tune_claims < 1:
        raise ValueError(f"tune_claims must be >= 1, got {settings.tune_claims}")
    if settings.strategy not in {"two_stage", "independent", "joint"}:
        raise ValueError(
            f"Unsupported strategy '{settings.strategy}'. Expected one of: two_stage, independent, joint"
        )
    if settings.fit_mode not in {"A1", "A3"}:
        raise ValueError(f"Unsupported fit_mode '{settings.fit_mode}'. Expected one of: A1, A3")
    if settings.validate_mode not in {"A1", "A3"}:
        raise ValueError(f"Unsupported validate_mode '{settings.validate_mode}'. Expected one of: A1, A3")

    claims_for_tuning = tuple(claims[: min(len(claims), int(settings.tune_claims))])

    defaults = params_from_config(config)
    alpha_grid = _build_grid(settings.alpha_min, settings.alpha_max, settings.grid_size)
    beta_grid = _build_grid(settings.beta_min, settings.beta_max, settings.grid_size)

    alpha_step = 0.0
    beta_step = 0.0
    if len(alpha_grid) > 1:
        alpha_step = float(alpha_grid[1] - alpha_grid[0])
    if len(beta_grid) > 1:
        beta_step = float(beta_grid[1] - beta_grid[0])

    stages: Dict[str, object] = {}
    candidates_evaluated = 0

    if settings.strategy == "joint":
        joint = _run_joint_search(
            claims=claims_for_tuning,
            config=config,
            defaults=defaults,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            mode=settings.fit_mode,
        )
        candidates_evaluated += int(joint["candidates_evaluated"])
        best_params: MaskParams = joint["best_params"]
        stages["fit_joint"] = {
            "mode": settings.fit_mode,
            "best": joint["best"],
            "candidates_evaluated": int(joint["candidates_evaluated"]),
        }
    else:
        independent = _run_independent_search(
            claims=claims_for_tuning,
            config=config,
            defaults=defaults,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            mode=settings.fit_mode,
        )
        candidates_evaluated += int(independent["candidates_evaluated"])
        best_params = independent["best_params"]
        stages["fit_independent"] = {
            "mode": settings.fit_mode,
            "support_best": independent["support_best"],
            "best": independent["best"],
            "candidates_evaluated": int(independent["candidates_evaluated"]),
        }

        if settings.local_refine:
            refined = _run_local_refinement(
                claims=claims_for_tuning,
                config=config,
                defaults=defaults,
                center=best_params,
                mode=settings.fit_mode,
                alpha_step=alpha_step,
                beta_step=beta_step,
                alpha_min=settings.alpha_min,
                alpha_max=settings.alpha_max,
                beta_min=settings.beta_min,
                beta_max=settings.beta_max,
            )
            candidates_evaluated += int(refined["candidates_evaluated"])
            best_params = refined["best_params"]
            stages["local_refinement"] = {
                "mode": settings.fit_mode,
                "center_params": refined["center_params"],
                "sup_alphas": refined["sup_alphas"],
                "sup_betas": refined["sup_betas"],
                "ref_alphas": refined["ref_alphas"],
                "ref_betas": refined["ref_betas"],
                "best": refined["best"],
                "candidates_evaluated": int(refined["candidates_evaluated"]),
            }

    if settings.strategy == "two_stage":
        validation = _select_from_candidates(
            claims=claims_for_tuning,
            config=config,
            defaults=defaults,
            mode=settings.validate_mode,
            candidates=(defaults, best_params),
        )
        candidates_evaluated += int(validation["candidates_evaluated"])
        stages["validate"] = {
            "mode": validation["mode"],
            "candidates": validation["candidates"],
            "best": validation["best"],
            "candidates_evaluated": int(validation["candidates_evaluated"]),
        }
        selected_params = validation["best_params"]
    else:
        selected_params = best_params

    selected_config = config_with_mask_params(config, selected_params.to_dict())
    report = {
        "enabled": True,
        "strategy": settings.strategy,
        "fit_mode": settings.fit_mode,
        "validate_mode": settings.validate_mode,
        "objective": "validation_accuracy",
        "selection_tiebreak": ["lower_aurc", "lower_l1_to_defaults"],
        "claims_used": len(claims_for_tuning),
        "search_space": {
            "grid_size": int(settings.grid_size),
            "alpha_min": float(settings.alpha_min),
            "alpha_max": float(settings.alpha_max),
            "beta_min": float(settings.beta_min),
            "beta_max": float(settings.beta_max),
            "alpha_grid": list(alpha_grid),
            "beta_grid": list(beta_grid),
            "alpha_step": float(alpha_step),
            "beta_step": float(beta_step),
            "local_refine": bool(settings.local_refine),
        },
        "defaults": defaults.to_dict(),
        "stages": stages,
        "candidates_evaluated": int(candidates_evaluated),
        "best_params": selected_params.to_dict(),
    }

    return {
        "best_params": selected_params.to_dict(),
        "best_config": selected_config,
        "report": report,
    }


__all__ = [
    "MaskParams",
    "MaskTuningSettings",
    "config_with_mask_params",
    "params_from_config",
    "tune_mask_coefficients",
]
