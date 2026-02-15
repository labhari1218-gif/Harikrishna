#!/usr/bin/env python3
"""Run Component 2 ablation study (A0-A6)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from component2.ablation_runner import run_ablation_study, run_tau_esi_sweep
from component2.config import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Component 2 ablation study (A0-A6)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--claims_jsonl",
        type=Path,
        default=None,
        help="Component 1 claims JSONL. Defaults to logs/component1/<split>/claims.jsonl",
    )
    parser.add_argument("--n_claims", type=int, default=2000, help="Subset size (recommended 1000-5000)")
    parser.add_argument("--seed", type=int, default=13, help="Subset shuffle seed")
    parser.add_argument("--tau_abstain", type=float, default=0.60, help="Fixed abstention threshold for S2")
    parser.add_argument(
        "--ablation_ids",
        type=str,
        default="A0,A1,A2,A3,A4",
        help="Comma-separated subset of {A0,A1,A2,A3,A4,A5,A6}",
    )
    parser.add_argument(
        "--tau-esi-sweep",
        action="store_true",
        help="Run T16 sensitivity sweep over tau_esi values (recovery threshold).",
    )
    parser.add_argument(
        "--tau-esi-values",
        type=str,
        default="0.2,0.3,0.4",
        help="Comma-separated tau_esi values, e.g. 0.2,0.3,0.4",
    )
    parser.add_argument(
        "--tau-sweep-mode",
        type=str,
        default="A3",
        help="Recovery-enabled ablation mode for tau_esi sweep (default: A3).",
    )
    parser.add_argument("--output_root", type=Path, default=Path("logs/ablations/component2"))
    parser.add_argument("--run_id", type=str, default=None, help="Optional explicit run folder name")

    # Optional overrides for locked Component2 tunables.
    parser.add_argument("--recovery_top_k", type=int, default=None)
    parser.add_argument("--recovery_tau_esi", type=float, default=None)
    parser.add_argument("--ppr_epsilon", type=float, default=None)
    parser.add_argument("--neutral_cap_gamma", type=float, default=None)

    # Optional mask tuning.
    parser.add_argument("--tune-mask-coeffs", action="store_true")
    parser.add_argument(
        "--mask-tune-strategy",
        type=str,
        default="two_stage",
        choices=["two_stage", "independent", "joint"],
    )
    parser.add_argument("--mask-fit-mode", type=str, default="A1", choices=["A1"])
    parser.add_argument("--mask-validate-mode", type=str, default="A3", choices=["A3"])
    parser.add_argument("--mask-grid-size", type=int, default=7)
    parser.add_argument("--mask-alpha-min", type=float, default=1.0)
    parser.add_argument("--mask-alpha-max", type=float, default=7.0)
    parser.add_argument("--mask-beta-min", type=float, default=-3.0)
    parser.add_argument("--mask-beta-max", type=float, default=0.0)
    parser.add_argument("--mask-tune-claims", type=int, default=200)
    parser.add_argument("--mask-local-refine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    claims_jsonl = args.claims_jsonl
    if claims_jsonl is None:
        claims_jsonl = Path("logs/component1") / args.split / "claims.jsonl"

    if args.n_claims < 1:
        raise ValueError(f"--n_claims must be >= 1, got {args.n_claims}")

    config = replace(
        DEFAULT_CONFIG,
        recovery_top_k=DEFAULT_CONFIG.recovery_top_k if args.recovery_top_k is None else int(args.recovery_top_k),
        recovery_tau_esi=(
            DEFAULT_CONFIG.recovery_tau_esi if args.recovery_tau_esi is None else float(args.recovery_tau_esi)
        ),
        ppr_epsilon=DEFAULT_CONFIG.ppr_epsilon if args.ppr_epsilon is None else float(args.ppr_epsilon),
        neutral_cap_gamma=(
            DEFAULT_CONFIG.neutral_cap_gamma if args.neutral_cap_gamma is None else float(args.neutral_cap_gamma)
        ),
    )

    ablation_ids = [part.strip().upper() for part in args.ablation_ids.split(",") if part.strip()]

    if args.tau_esi_sweep:
        tau_values = [float(part.strip()) for part in str(args.tau_esi_values).split(",") if part.strip()]
        run_dir = run_tau_esi_sweep(
            split=args.split,
            claims_jsonl=claims_jsonl,
            n_claims=int(args.n_claims),
            seed=int(args.seed),
            tau_abstain=float(args.tau_abstain),
            output_root=args.output_root,
            run_id=args.run_id,
            tau_esi_values=tuple(tau_values),
            mode_id=str(args.tau_sweep_mode).strip().upper(),
            config=config,
        )
        print(f"Tau-ESI sweep completed: {run_dir}")
    else:
        run_dir = run_ablation_study(
            split=args.split,
            claims_jsonl=claims_jsonl,
            n_claims=int(args.n_claims),
            seed=int(args.seed),
            tau_abstain=float(args.tau_abstain),
            output_root=args.output_root,
            run_id=args.run_id,
            ablation_ids=tuple(ablation_ids),
            config=config,
            tune_mask_coeffs=bool(args.tune_mask_coeffs),
            mask_tune_strategy=str(args.mask_tune_strategy),
            mask_fit_mode=str(args.mask_fit_mode),
            mask_validate_mode=str(args.mask_validate_mode),
            mask_grid_size=int(args.mask_grid_size),
            mask_alpha_min=float(args.mask_alpha_min),
            mask_alpha_max=float(args.mask_alpha_max),
            mask_beta_min=float(args.mask_beta_min),
            mask_beta_max=float(args.mask_beta_max),
            mask_tune_claims=int(args.mask_tune_claims),
            mask_local_refine=bool(args.mask_local_refine),
        )
        print(f"Ablation study completed: {run_dir}")


if __name__ == "__main__":
    main()
