"""
logging_utils.py - Claim and pair-level logging for Component 1

Implements JSONL-based logging with:
- Claim-level logs: ESI, CR@k, RPI@k, and legacy metrics
- Pair-level logs: Configurable modes (none/all/A_only/A_and_C/sample)
- Examples dump: Top claims by contradiction, starvation (ESI), neutral mass
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union, TextIO

import numpy as np

from .evidence import EvidenceItem
from .sufficiency_metrics import (
    compute_esi,
    compute_neutral_dominance,
    compute_cr_at_k,
    compute_rpi_at_k
)

logger = logging.getLogger(__name__)


@dataclass
class PairLogConfig:
    """Configuration for pair-level logging."""
    mode: str = "none"  # "none", "all", "A_only", "A_and_C", "sample"
    sample_rate: float = 0.1  # Used when mode="sample"


def write_claim_log(
    claim_id: str,
    claim_text: str,
    label: str,
    claim_entities: List[str],
    A: List[EvidenceItem],
    S: List[EvidenceItem],
    C: List[EvidenceItem],
    pool: List[EvidenceItem],
    min_A: int,
    out_file: Union[str, Path, TextIO]
):
    """
    Write claim-level log entry as single JSON object.
    
    Includes:
    - Sufficiency metrics (ESI, coverage, connectivity, neutral-dominance)
    - Counter retention (CR@k)
    - Recovery potential (RPI@k)
    - Pool statistics
    - Legacy starvation metrics
    
    Args:
        claim_id: Unique claim identifier
        claim_text: Claim text
        label: Ground truth label
        claim_entities: List of claim entity strings
        A, S, C: Evidence sets from ESM
        pool: Full evidence pool
        min_A: Minimum Active size (for shortfall calculation)
        out_file: File path or file handle for JSONL output
    """
    # Compute pool statistics
    rel_vals = [e.pv.rel for e in pool]
    pol_vals = [e.pv.pol for e in pool]
    
    pool_stats = {
        "mean_rel": float(np.mean(rel_vals)) if rel_vals else 0.0,
        "mean_pol": float(np.mean(pol_vals)) if pol_vals else 0.0,
        "p25_rel": float(np.percentile(rel_vals, 25)) if rel_vals else 0.0,
        "p50_rel": float(np.percentile(rel_vals, 50)) if rel_vals else 0.0,
        "p75_rel": float(np.percentile(rel_vals, 75)) if rel_vals else 0.0,
    }
    
    # Top evidence by relevance and by contradiction
    pool_by_rel = sorted(pool, key=lambda e: e.pv.rel, reverse=True)
    pool_by_contra = sorted(pool, key=lambda e: e.pv.p_contra, reverse=True)
    
    top_evidence = {
        "by_rel": [e.evidence_id for e in pool_by_rel[:5]],
        "by_contra": [e.evidence_id for e in pool_by_contra[:5]]
    }
    
    # Compute sufficiency metrics
    esi_metrics = compute_esi(A, claim_entities)
    neutral_dom = compute_neutral_dominance(A)
    cr_metrics = compute_cr_at_k(C, pool, [5, 10])  # FIX: Corrected argument order C, pool
    rpi_metrics = compute_rpi_at_k(A, S, claim_entities, [1, 3, 5])
    
    # Compute legacy starvation metrics
    starved_A = int(len(A) == 0)
    active_shortfall = max(0, min_A - len(A))
    weak_A_mass = sum(e.pv.rel for e in A)
    
    legacy_starvation = {
        "starved_A": starved_A,
        "active_shortfall": active_shortfall,
        "weak_A_mass": weak_A_mass
    }
    
    # Build log entry
    log_entry = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "label": label,
        "counts": {
            "A": len(A),
            "S": len(S),
            "C": len(C)
        },
        "sufficiency": {
            # PRIMARY METRIC: ESI_geom (geometric mean - more robust)
            "esi_geom": esi_metrics["esi_geom"],
            "starve_score_geom": esi_metrics["starve_score_geom"],
            # LEGACY METRIC: ESI_prod (product - very conservative)
            "esi_prod": esi_metrics["esi_prod"],
            "starve_score_prod": esi_metrics["starve_score_prod"],
            # Components
            "mass_A": esi_metrics["mass_A"],
            "mass_A_normalized": esi_metrics["mass_A_normalized"],
            "coverage_A": esi_metrics["coverage_A"],
            "connectivity_A": esi_metrics["connectivity_A"],
            "neutral_rate_A": neutral_dom
        },
        "counter_retention": {
            "has_counter": len(C) > 0,
            "counter_kept": len(C),
            "cr_at_5": cr_metrics["cr_at_5"],
            "cr_at_10": cr_metrics["cr_at_10"],
            "max_contra_all": cr_metrics["max_contra_all"],
            "max_contra_C": cr_metrics["max_contra_C"],
            "contra_mass_C": cr_metrics["contra_mass_C"]
        },
        "recovery": {
            "bridge_count_S": rpi_metrics["bridge_count_S"],
            "bridge_rel_mass_S": rpi_metrics["bridge_rel_mass_S"],
            "rpi_at_1": rpi_metrics["rpi_at_1"],
            "rpi_at_3": rpi_metrics["rpi_at_3"],
            "rpi_at_5": rpi_metrics["rpi_at_5"]
        },
        "pool_stats": pool_stats,
        "top_evidence": top_evidence,
        "starvation": legacy_starvation
    }
    
    # Write to file
    if isinstance(out_file, (str, Path)):
        with open(out_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    else:
        out_file.write(json.dumps(log_entry) + '\n')


def write_pair_log(
    claim_id: str,
    evidence_item: EvidenceItem,
    claim_text: str,
    premise_text: str,
    model_name: str,
    verbalizer_id: str,
    config: PairLogConfig,
    out_file: Union[str, Path, TextIO],
    evidence_assignment: str = "unknown"  # "A", "S", "C", or "unknown"
):
    """
    Conditionally write pair-level log entry based on PairLogConfig.
    
    Logs include raw triple (for audit), verbalized premise, model probs,
    and derived rel/pol scores.
    
    Args:
        claim_id: Claim identifier
        evidence_item: EvidenceItem with PV score
        claim_text: Claim text (hypothesis)
        premise_text: Verbalized evidence text (premise)
        model_name: Model name used for scoring
        verbalizer_id: Verbalization version
        config: PairLogConfig with mode and sample_rate
        out_file: File path or file handle for JSONL output
        evidence_assignment: Which set this evidence belongs to ("A", "S", "C")
    """
    # Check if we should log this pair
    should_log = False
    
    if config.mode == "all":
        should_log = True
    elif config.mode == "A_only" and evidence_assignment == "A":
        should_log = True
    elif config.mode == "A_and_C" and evidence_assignment in ["A", "C"]:
        should_log = True
    elif config.mode == "sample":
        should_log = random.random() < config.sample_rate
    # mode == "none": should_log remains False
    
    if not should_log:
        return
    
    # Extract raw triple for audit (if kg_triple)
    raw_triple = None
    if evidence_item.kind == "kg_triple":
        raw_triple = evidence_item.content.get("triple", [])
    
    # Construct log entry
    log_entry = {
        "claim_id": claim_id,
        "evidence_id": evidence_item.evidence_id,
        "raw_triple": raw_triple,
        "premise_text": premise_text,
        "hypothesis_text": claim_text,
        "model_name": model_name,
        "verbalizer_id": verbalizer_id,
        "probs": {
            "entail": evidence_item.pv.p_entail,
            "contra": evidence_item.pv.p_contra,
            "neutral": evidence_item.pv.p_neutral
        },
        "derived": {
            "rel": evidence_item.pv.rel,
            "pol": evidence_item.pv.pol
        }
    }
    
    # Write to file
    if isinstance(out_file, (str, Path)):
        with open(out_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    else:
        out_file.write(json.dumps(log_entry) + '\n')


def save_examples(
    split: str,
    claims_jsonl: Union[str, Path],
    examples_dir: Union[str, Path],
    top_n: int = 10
):
    """
    Extract and save interesting example claims.
    
    Categories:
    - highest_contradiction.json: Claims with max p_contra in pool
    - lowest_esi.json: Claims with lowest ESI (evidence starvation)
    - most_neutral.json: Claims with highest neutral mass in Active set
    
    Args:
        split: Dataset split name
        claims_jsonl: Path to claim-level JSONL
        examples_dir: Output directory for examples
        top_n: Number of examples per category (default: 10)
    """
    examples_dir = Path(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all claims
    claims = []
    with open(claims_jsonl, 'r') as f:
        for line in f:
            claims.append(json.loads(line))
    
    if not claims:
        logger.warning(f"No claims found in {claims_jsonl}")
        return
    
    # Extract top examples by category
    
    # 1. Highest contradiction (max_contra_all)
    claims_by_contra = sorted(
        claims,
        key=lambda c: c["counter_retention"]["max_contra_all"],
        reverse=True
    )
    
    highest_contra_file = examples_dir / "highest_contradiction.json"
    with open(highest_contra_file, 'w') as f:
        json.dump(claims_by_contra[:top_n], f, indent=2)
    
    logger.info(f"Saved {top_n} highest contradiction examples to {highest_contra_file}")
    
    # 2. Lowest ESI (evidence starvation)
    # Use esi_geom as primary metric (geometric mean - more robust than product)
    claims_by_esi = sorted(
        claims,
        key=lambda c: c["sufficiency"].get("esi_geom", c["sufficiency"].get("esi_prod", 0.0))
    )
    
    lowest_esi_file = examples_dir / "lowest_esi.json"
    with open(lowest_esi_file, 'w') as f:
        json.dump(claims_by_esi[:top_n], f, indent=2)
    
    logger.info(f"Saved {top_n} lowest ESI examples to {lowest_esi_file}")
    
    # 3. Most neutral (highest neutral_rate_A)
    claims_by_neutral = sorted(
        claims,
        key=lambda c: c["sufficiency"]["neutral_rate_A"],
        reverse=True
    )
    
    most_neutral_file = examples_dir / "most_neutral.json"
    with open(most_neutral_file, 'w') as f:
        json.dump(claims_by_neutral[:top_n], f, indent=2)
    
    logger.info(f"Saved {top_n} most neutral examples to {most_neutral_file}")
