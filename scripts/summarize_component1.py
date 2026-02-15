#!/usr/bin/env python3
"""
summarize_component1.py - Generate summary statistics for Component 1 runs

Reads claim-level JSONL and generates:
1. Aggregate statistics (mean ESI, CR@k, RPI@k, etc.)
2. Examples dump (highest contradiction, lowest ESI, most neutral)

Usage:
    python scripts/summarize_component1.py --split train
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_claims_jsonl(jsonl_path: Path) -> list:
    """Load all claims from JSONL file."""
    claims = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            claims.append(json.loads(line))
    return claims


def compute_summary_statistics(claims: list) -> dict:
    """
    Compute aggregate statistics across all claims.
    
    Returns:
        Dict with summary statistics
    """
    if not claims:
        logger.warning("No claims to summarize")
        return {}
    
    num_claims = len(claims)
    
    # Set sizes (FIX 2: use .get() for robustness)
    A_sizes = [c.get("counts", {}).get("A", 0) for c in claims]
    S_sizes = [c.get("counts", {}).get("S", 0) for c in claims]
    C_sizes = [c.get("counts", {}).get("C", 0) for c in claims]
    
    # Sufficiency metrics (use ESI_geom as primary, fallback to esi_prod for legacy logs)
    # FIX 2: Robust extraction with .get() to handle missing fields
    esi_geom_values = []
    esi_prod_values = []
    for c in claims:
        suff = c.get("sufficiency", {})
        # Try esi_geom first, fallback to esi_prod
        esi_geom = suff.get("esi_geom")
        if esi_geom is None:
            esi_geom = suff.get("esi_prod", 0.0)
        esi_geom_values.append(esi_geom)
        esi_prod_values.append(suff.get("esi_prod", 0.0))
    
    coverage_vals = [c.get("sufficiency", {}).get("coverage_A", 0.0) for c in claims]
    connectivity_vals = [c.get("sufficiency", {}).get("connectivity_A", 0.0) for c in claims]
    neutral_rate_vals = [c.get("sufficiency", {}).get("neutral_rate_A", 0.0) for c in claims]
    
    # Compute percentiles for data-driven thresholds
    esi_geom_p10 = float(np.percentile(esi_geom_values, 10)) if esi_geom_values else 0.0
    esi_geom_p25 = float(np.percentile(esi_geom_values, 25)) if esi_geom_values else 0.0
    esi_geom_median = float(np.percentile(esi_geom_values, 50)) if esi_geom_values else 0.0
    
    # Starvation rate using both fixed threshold (0.3) and percentile (p10)
    starve_rate_fixed_03 = sum(1 for e in esi_geom_values if e < 0.3) / num_claims if esi_geom_values else 0.0
    starve_rate_p10 = sum(1 for e in esi_geom_values if e < esi_geom_p10) / num_claims if esi_geom_values else 0.0
    
    # Counter retention (FIX 2: use .get() for robustness)
    has_counter = [c.get("counter_retention", {}).get("has_counter", False) for c in claims]
    cr_at_5_vals = [c.get("counter_retention", {}).get("cr_at_5", 0.0) for c in claims]
    cr_at_10_vals = [c.get("counter_retention", {}).get("cr_at_10", 0.0) for c in claims]
    counter_kept = [c.get("counter_retention", {}).get("counter_kept", 0) for c in claims]
    max_contra_C = [
        c.get("counter_retention", {}).get("max_contra_C", 0.0)
        for c in claims if c.get("counter_retention", {}).get("has_counter", False)
    ]
    
    # Recovery potential (FIX 2: use .get() for robustness)
    bridge_count_vals = [c.get("recovery", {}).get("bridge_count_S", 0) for c in claims]
    bridge_rel_mass_vals = [c.get("recovery", {}).get("bridge_rel_mass_S", 0.0) for c in claims]
    rpi_at_1_vals = [c.get("recovery", {}).get("rpi_at_1", 0.0) for c in claims]
    rpi_at_3_vals = [c.get("recovery", {}).get("rpi_at_3", 0.0) for c in claims]
    rpi_at_5_vals = [c.get("recovery", {}).get("rpi_at_5", 0.0) for c in claims]
    
    # Legacy metrics (FIX 2: use .get() for robustness)
    starved_A_vals = [c.get("starvation", {}).get("starved_A", 0) for c in claims]
    active_shortfall_vals = [c.get("starvation", {}).get("active_shortfall", 0) for c in claims]
    weak_A_mass_vals = [c.get("starvation", {}).get("weak_A_mass", 0.0) for c in claims]
    
    # Construct summary
    summary = {
        "num_claims": num_claims,
        
        # Set sizes
        "mean_A": float(np.mean(A_sizes)),
        "mean_S": float(np.mean(S_sizes)),
        "mean_C": float(np.mean(C_sizes)),
        
        # PRIMARY: ESI_geom (geometric mean - recommended)
        "mean_esi_geom": float(np.mean(esi_geom_values)),
        "esi_geom_p10": esi_geom_p10,
        "esi_geom_p25": esi_geom_p25,
        "esi_geom_median": esi_geom_median,
        "starve_rate_geom_fixed_03": float(starve_rate_fixed_03),
        "starve_rate_geom_p10": float(starve_rate_p10),
        
        # LEGACY: ESI_prod (product - very conservative)
        "mean_esi_prod": float(np.mean(esi_prod_values)),
        
        # Other sufficiency components
        "mean_coverage_A": float(np.mean(coverage_vals)),
        "mean_connectivity_A": float(np.mean(connectivity_vals)),
        "mean_neutral_rate_A": float(np.mean(neutral_rate_vals)),
        
        # Counter retention
        "has_counter_rate": float(sum(has_counter) / num_claims),
        "mean_cr_at_5": float(np.mean(cr_at_5_vals)),
        "mean_cr_at_10": float(np.mean(cr_at_10_vals)),
        "counter_kept_mean": float(np.mean(counter_kept)),
        "max_contra_C_mean": float(np.mean(max_contra_C)) if max_contra_C else 0.0,
        
        # Recovery potential
        "mean_bridge_count_S": float(np.mean(bridge_count_vals)),
        "mean_bridge_rel_mass_S": float(np.mean(bridge_rel_mass_vals)),
        "mean_rpi_at_1": float(np.mean(rpi_at_1_vals)),
        "mean_rpi_at_3": float(np.mean(rpi_at_3_vals)),
        "mean_rpi_at_5": float(np.mean(rpi_at_5_vals)),
        
        # Legacy metrics
        "starved_A_rate": float(sum(starved_A_vals) / num_claims),
        "active_shortfall_mean": float(np.mean(active_shortfall_vals)),
        "weak_A_mass_mean": float(np.mean(weak_A_mass_vals))
    }
    
    return summary


def save_examples(claims: list, examples_dir: Path, top_n: int = 10):
    """
    Save top-N examples by category.
    
    Categories:
    - highest_contradiction.json
    - lowest_esi.json
    - most_neutral.json
    """
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    if not claims:
        logger.warning("No claims for examples")
        return
    
    # 1. Highest contradiction
    claims_by_contra = sorted(
        claims,
        key=lambda c: c["counter_retention"]["max_contra_all"],
        reverse=True
    )
    
    highest_contra_file = examples_dir / "highest_contradiction.json"
    with open(highest_contra_file, 'w') as f:
        json.dump(claims_by_contra[:top_n], f, indent=2)
    
    logger.info(f"Saved top-{top_n} highest contradiction examples to {highest_contra_file}")
    
    # 2. Lowest ESI_geom (evidence starvation)
    claims_by_esi = sorted(
        claims,
        key=lambda c: c.get("sufficiency", {}).get("esi_geom", c.get("sufficiency", {}).get("esi_prod", 0.0))
    )
    
    lowest_esi_file = examples_dir / "lowest_esi.json"
    with open(lowest_esi_file, 'w') as f:
        json.dump(claims_by_esi[:top_n], f, indent=2)
    
    logger.info(f"Saved top-{top_n} lowest ESI examples to {lowest_esi_file}")
    
    # 3. Most neutral
    claims_by_neutral = sorted(
        claims,
        key=lambda c: c["sufficiency"]["neutral_rate_A"],
        reverse=True
    )
    
    most_neutral_file = examples_dir / "most_neutral.json"
    with open(most_neutral_file, 'w') as f:
        json.dump(claims_by_neutral[:top_n], f, indent=2)
    
    logger.info(f"Saved top-{top_n} most neutral examples to {most_neutral_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize Component 1 results")
    
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                       help="Dataset split to summarize")
    parser.add_argument("--log_dir", type=str, default="logs/component1",
                       help="Log directory (default: logs/component1)")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Top-N examples per category (default: 10)")
    
    args = parser.parse_args()
    
    # Construct paths
    log_dir = Path(args.log_dir) / args.split
    claims_jsonl = log_dir / "claims.jsonl"
    summary_json = log_dir / "summary.json"
    examples_dir = log_dir / "examples"
    
    if not claims_jsonl.exists():
        logger.error(f"Claims JSONL not found: {claims_jsonl}")
        logger.error("Run run_component1_pv_esm.py first")
        return 1
    
    logger.info(f"Loading claims from {claims_jsonl}")
    claims = load_claims_jsonl(claims_jsonl)
    logger.info(f"Loaded {len(claims)} claims")
    
    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary = compute_summary_statistics(claims)
    summary["split"] = args.split
    
    # Save summary
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_json}")
    
    # Print key metrics
    logger.info("\n" + "="*60)
    logger.info("COMPONENT 1 SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Split: {args.split}")
    logger.info(f"Claims: {summary['num_claims']}")
    logger.info("")
    logger.info("Set Sizes:")
    logger.info(f"  Mean Active (A): {summary['mean_A']:.2f}")
    logger.info(f"  Mean Suspended (S): {summary['mean_S']:.2f}")
    logger.info(f"  Mean Counter (C): {summary['mean_C']:.2f}")
    logger.info("")
    logger.info("Sufficiency Metrics:")
    logger.info(f"  Mean ESI_geom (PRIMARY): {summary['mean_esi_geom']:.4f}")
    logger.info(f"  ESI_geom p10/p25/median: {summary['esi_geom_p10']:.3f} / {summary['esi_geom_p25']:.3f} / {summary['esi_geom_median']:.3f}")
    logger.info(f"  Starvation Rate (ESI_geom<0.3): {summary['starve_rate_geom_fixed_03']:.2%}")
    logger.info(f"  Starvation Rate (ESI_geom<p10): {summary['starve_rate_geom_p10']:.2%}")
    logger.info(f"  Mean ESI_prod (legacy): {summary['mean_esi_prod']:.4f}")
    logger.info(f"  Mean Coverage: {summary['mean_coverage_A']:.4f}")
    logger.info(f"  Mean Connectivity: {summary['mean_connectivity_A']:.4f}")
    logger.info(f"  Mean Neutral Rate: {summary['mean_neutral_rate_A']:.4f}")
    logger.info("")
    logger.info("Counter Retention:")
    logger.info(f"  Has Counter Rate: {summary['has_counter_rate']:.2%}")
    logger.info(f"  Mean CR@5: {summary['mean_cr_at_5']:.4f}")
    logger.info(f"  Mean CR@10: {summary['mean_cr_at_10']:.4f}")
    logger.info("")
    logger.info("Recovery Potential:")
    logger.info(f"  Mean Bridge Count (S): {summary['mean_bridge_count_S']:.2f}")
    logger.info(f"  Mean RPI@1: {summary['mean_rpi_at_1']:.4f}")
    logger.info(f"  Mean RPI@3: {summary['mean_rpi_at_3']:.4f}")
    logger.info(f"  Mean RPI@5: {summary['mean_rpi_at_5']:.4f}")
    logger.info("")
    logger.info("Legacy Metrics:")
    logger.info(f"  Starved A Rate (|A|==0): {summary['starved_A_rate']:.2%}")
    logger.info(f"  Active Shortfall Mean: {summary['active_shortfall_mean']:.2f}")
    logger.info("="*60)
    
    # Generate examples
    logger.info("\nGenerating examples...")
    save_examples(claims, examples_dir, top_n=args.top_n)
    
    logger.info("\nSummary complete!")
    logger.info(f"Summary JSON: {summary_json}")
    logger.info(f"Examples: {examples_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
