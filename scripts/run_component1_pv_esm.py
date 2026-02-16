#!/usr/bin/env python3
"""
run_component1_pv_esm.py - Main runner script for Component 1

Processes claims through the PV+ESM pipeline:
1. Load FactKG claims and subgraph evidence
2. Score evidence using NLI-based PV
3. Partition into A/S/C using ESM
4. Log claim-level and (optionally) pair-level results
5. Cache PV scores for reuse

Usage:
    python scripts/run_component1_pv_esm.py --split train --limit_claims 100
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add repository root to path FIRST (before other imports)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
from tqdm import tqdm

# FIX 3: Import heavy modules directly from submodules (not from __init__)
from component1 import (
    EvidenceItem,
    EvidenceStateManager,
    ESMConfig,
    evidence_to_text,
    VERBALIZER_VERSION,
)
from component1.pv import PVScorer, PVConfig
from component1.cache import PVCache
from component1.evidence import stable_hash
from component1.logging_utils import (
    write_claim_log,
    write_pair_log,
    write_empty_evidence_pair_sentinel,
    PairLogConfig,
)

# Import from project's local datasets.py (not HuggingFace datasets)
import datasets as project_datasets
import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_factkg_data(split: str) -> pd.DataFrame:
    """Load FactKG claims for given split."""
    logger.info(f"Loading FactKG data for split: {split}")
    df = project_datasets.get_df(split)
    logger.info(f"Loaded {len(df)} claims")
    return df


def load_subgraph_data(split: str) -> pd.DataFrame:
    """Load subgraph evidence for given split."""
    logger.info(f"Loading subgraph data for split: {split}")
    df = project_datasets.get_subgraphs(split, subgraph_type="direct_filled")
    logger.info(f"Loaded {len(df)} subgraphs")
    return df


def create_evidence_pool(claim_id: str, subgraph_row: dict) -> list:
    """
    Create evidence pool from subgraph data.
    
    Args:
        claim_id: Unique claim identifier
        subgraph_row: Row from subgraph DataFrame
        
    Returns:
        List of EvidenceItem objects
    """
    evidence_pool = []
    
    # Extract triples from walked['connected'] + walked['walkable'] (deduplicated)
    walked = subgraph_row.get('walked', {})
    connected_triples = walked.get('connected', [])
    walkable_triples = walked.get('walkable', [])

    seen_triples = set()
    merged_triples = []
    for triple in list(connected_triples) + list(walkable_triples):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        triple_key = (str(triple[0]), str(triple[1]), str(triple[2]))
        if triple_key in seen_triples:
            continue
        seen_triples.add(triple_key)
        merged_triples.append([triple_key[0], triple_key[1], triple_key[2]])

    for idx, triple in enumerate(merged_triples):
        if len(triple) != 3:
            continue  # Skip malformed triples
        
        evidence_id = f"{claim_id}_triple_{idx}"
        
        evidence_item = EvidenceItem(
            evidence_id=evidence_id,
            kind="kg_triple",
            content={"triple": triple},
            retrieval_score=0.0,  # Not used yet
            pv=None,  # Will be populated by PV scorer
            meta={}
        )
        
        evidence_pool.append(evidence_item)
    
    return evidence_pool


def process_claim(
    claim_id: str,
    claim_text: str,
    label: str,
    claim_entities: list,
    evidence_pool: list,
    pv_scorer: Optional[PVScorer],
    pv_cache: PVCache,
    esm_config: ESMConfig,
    pv_config: PVConfig,
    pair_log_config: PairLogConfig,
    claim_log_file: Path,
    pair_log_file: Path,
    cache_only: bool = False,
):
    """
    Process a single claim through the PV+ESM pipeline.
    
    Args:
        claim_id: Unique claim identifier
        claim_text: Claim text
        label: Ground truth label
        claim_entities: List of claim entity strings
        evidence_pool: List of EvidenceItem objects
        pv_scorer: Optional PVScorer instance (None when cache_only=True)
        pv_cache: PVCache instance
        esm_config: ESM configuration
        pv_config: PV configuration
        pair_log_config: Pair logging configuration
        claim_log_file: Path to claim-level JSONL
        pair_log_file: Path to pair-level JSONL
        cache_only: If True, require all PV scores to exist in cache (no model scoring)
    """
    if not evidence_pool:
        logger.warning(f"Claim {claim_id} has no evidence triples. Writing sentinel records.")
        write_claim_log(
            claim_id=claim_id,
            claim_text=claim_text,
            label=label,
            claim_entities=claim_entities,
            A=[],
            S=[],
            C=[],
            pool=[],
            min_A=esm_config.min_A,
            contra_tau=esm_config.contra_tau,
            out_file=claim_log_file,
        )
        if pair_log_config.mode != "none":
            write_empty_evidence_pair_sentinel(
                claim_id=claim_id,
                claim_text=claim_text,
                model_name=pv_config.model_name,
                verbalizer_id=VERBALIZER_VERSION,
                out_file=pair_log_file,
            )
        return True

    claim_hash = stable_hash(claim_text)
    
    # Step 1: Score evidence using PV (with caching)
    evidence_texts = []
    premise_texts = []
    
    for item in evidence_pool:
        premise_text, verbalizer_id = evidence_to_text(item)
        evidence_texts.append(item.evidence_id)
        premise_texts.append(premise_text)
    
    # Check cache and score missing items
    to_score_indices = []
    to_score_premises = []
    
    for idx, item in enumerate(evidence_pool):
        premise_text = premise_texts[idx]
        evidence_hash = stable_hash(premise_text)
        
        # Try cache
        cached_pv = pv_cache.get(
            claim_hash,
            evidence_hash,
            pv_config.model_name,
            pv_config.max_length,
            VERBALIZER_VERSION,
            claim_id=claim_id
        )
        
        if cached_pv is not None:
            # Cache hit
            item.pv = cached_pv
        else:
            # Cache miss: add to scoring batch
            to_score_indices.append(idx)
            to_score_premises.append(premise_text)
    
    # Score uncached evidence
    if to_score_premises:
        if cache_only or pv_scorer is None:
            raise RuntimeError(
                f"Cache-only mode missing {len(to_score_premises)} PV scores for claim {claim_id}. "
                "Populate cache first or run without --cache_only."
            )
        pv_results = pv_scorer.pv_score_many(claim_text, to_score_premises)
        
        for i, pv_result in enumerate(pv_results):
            idx = to_score_indices[i]
            item = evidence_pool[idx]
            item.pv = pv_result
            
            # Store in cache
            evidence_hash = stable_hash(to_score_premises[i])
            pv_cache.put(
                claim_hash,
                evidence_hash,
                pv_config.model_name,
                pv_config.max_length,
                VERBALIZER_VERSION,
                pv_result
            )
    
    # Step 2: Partition using ESM
    partition_result = EvidenceStateManager.partition(evidence_pool, esm_config)
    A = partition_result["A"]
    S = partition_result["S"]
    C = partition_result["C"]
    
    # Step 3: Write claim-level log
    write_claim_log(
        claim_id=claim_id,
        claim_text=claim_text,
        label=label,
        claim_entities=claim_entities,
        A=A,
        S=S,
        C=C,
        pool=evidence_pool,
        min_A=esm_config.min_A,
        contra_tau=esm_config.contra_tau,  # FIX 1: Pass ESM's contra_tau
        out_file=claim_log_file
    )
    
    # Step 4: Optionally write pair-level logs
    if pair_log_config.mode != "none":
        # Create assignment map
        assignment_map = {}
        for item in A:
            assignment_map[item.evidence_id] = "A"
        for item in S:
            assignment_map[item.evidence_id] = "S"
        for item in C:
            assignment_map[item.evidence_id] = "C"
        
        for idx, item in enumerate(evidence_pool):
            write_pair_log(
                claim_id=claim_id,
                evidence_item=item,
                claim_text=claim_text,
                premise_text=premise_texts[idx],
                model_name=pv_config.model_name,
                verbalizer_id=VERBALIZER_VERSION,
                config=pair_log_config,
                out_file=pair_log_file,
                evidence_assignment=assignment_map.get(item.evidence_id, "unknown")
            )

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Component 1: PV+ESM pipeline")
    
    # Data arguments
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                       help="Dataset split to process")
    parser.add_argument("--limit_claims", type=int, default=None,
                       help="Limit number of claims to process (for testing)")
    
    # Output arguments
    parser.add_argument("--out_dir", type=str, default="logs/component1",
                       help="Output directory for logs")
    
    # PV scorer arguments
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli",
                       help="HuggingFace NLI model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use (auto=detect, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for PV scoring (conservative for 8GB GPU)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max token length")
    parser.add_argument(
        "--non_blocking_transfers",
        action="store_true",
        help="Enable non-blocking host-to-device PV batch transfer when CUDA is active",
    )
    
    # ESM arguments
    parser.add_argument("--min_A", type=int, default=5,
                       help="Min evidence in Active set")
    parser.add_argument("--max_A", type=int, default=20,
                       help="Max evidence in Active set")
    parser.add_argument("--max_C", type=int, default=5,
                       help="Max evidence in Counter set")
    parser.add_argument("--contra_tau", type=float, default=0.6,
                       help="Contradiction threshold (safer default for strong contradictions)")
    
    # Cache arguments
    parser.add_argument("--cache_dir", type=str, default="cache/pv_sqlite",
                       help="SQLite cache directory")
    
    # Pair logging arguments
    parser.add_argument("--pair_log_mode", type=str, default="none",
                       choices=["none", "all", "A_only", "A_and_C", "sample"],
                       help="Pair-level logging mode")
    parser.add_argument("--pair_log_sample_rate", type=float, default=0.1,
                       help="Sample rate when pair_log_mode=sample")
    parser.add_argument(
        "--cache_only",
        action="store_true",
        help="Use cached PV scores only (skip model loading/scoring and fail on cache misses)",
    )
    
    args = parser.parse_args()
    
    # Create output directories
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    
    claim_log_file = out_dir / "claims.jsonl"
    pair_log_file = out_dir / "pairs.jsonl"

    # Write to temp files first; only promote on fully successful completion.
    claim_log_tmp = out_dir / "claims.jsonl.tmp"
    pair_log_tmp = out_dir / "pairs.jsonl.tmp"
    if claim_log_tmp.exists():
        claim_log_tmp.unlink()
    if pair_log_tmp.exists():
        pair_log_tmp.unlink()
    claim_log_tmp.touch()
    if args.pair_log_mode != "none":
        pair_log_tmp.touch()
    
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Claim log (temp): {claim_log_tmp}")
    if args.pair_log_mode != "none":
        logger.info(f"Pair log (temp): {pair_log_tmp}")
    
    # Initialize PV config
    pv_config = PVConfig(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        non_blocking_transfers=args.non_blocking_transfers,
    )
    pv_scorer: Optional[PVScorer] = None
    if not args.cache_only:
        pv_scorer = PVScorer(pv_config)
    else:
        logger.info("Cache-only mode enabled: skipping PV model initialization.")

    # Initialize cache
    pv_cache = PVCache(args.cache_dir, read_only=args.cache_only)
    
    # Initialize ESM config
    esm_config = ESMConfig(
        min_A=args.min_A,
        max_A=args.max_A,
        max_C=args.max_C,
        contra_tau=args.contra_tau
    )
    
    # Pair log config
    pair_log_config = PairLogConfig(
        mode=args.pair_log_mode,
        sample_rate=args.pair_log_sample_rate
    )
    
    # Load data
    claims_df = load_factkg_data(args.split)
    subgraphs_df = load_subgraph_data(args.split)
    
    # Limit claims if requested
    if args.limit_claims:
        claims_df = claims_df.head(args.limit_claims)
        subgraphs_df = subgraphs_df.head(args.limit_claims)
        logger.info(f"Limited to {len(claims_df)} claims")
    
    # CRITICAL: Reset indices to ensure positional alignment
    claims_df = claims_df.reset_index(drop=True)
    subgraphs_df = subgraphs_df.reset_index(drop=True)
    
    # Verify alignment
    if len(claims_df) != len(subgraphs_df):
        logger.error(f"Claims/subgraphs misalignment: {len(claims_df)} claims vs {len(subgraphs_df)} subgraphs")
        raise ValueError("Claims and subgraphs must have equal length")
    
    # Process claims
    logger.info(f"Processing {len(claims_df)} claims...")
    
    # Process claims with proper cleanup
    claim_error_count = 0
    processed_with_evidence = 0
    skipped_no_evidence = 0
    try:
        # Use positional indexing to guarantee alignment
        for i in tqdm(range(len(claims_df)), desc="Processing claims"):
            claim_row = claims_df.iloc[i]
            subgraph_row = subgraphs_df.iloc[i]
            
            # Generate claim_id from positional index
            claim_id = f"{args.split}_{i}"
            claim_text = claim_row['Sentence']
            
            # Extract label (boolean list -> string)
            label_list = claim_row.get('Label', [None])
            label = "TRUE" if label_list and label_list[0] else "FALSE"
            
            # Extract claim entities
            claim_entities = claim_row.get('Entity_set', [])
            
            # Create evidence pool from aligned subgraph
            evidence_pool = create_evidence_pool(claim_id, subgraph_row)
            
            # Process claim
            try:
                claim_written = process_claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    label=label,
                    claim_entities=claim_entities,
                    evidence_pool=evidence_pool,
                    pv_scorer=pv_scorer,
                    pv_cache=pv_cache,
                    esm_config=esm_config,
                    pv_config=pv_config,
                    pair_log_config=pair_log_config,
                    claim_log_file=claim_log_tmp,
                    pair_log_file=pair_log_tmp,
                    cache_only=args.cache_only,
                )
                if claim_written:
                    processed_with_evidence += 1
                else:
                    skipped_no_evidence += 1
            except Exception as e:
                claim_error_count += 1
                logger.error(f"Error processing claim {claim_id}: {e}", exc_info=True)
                continue
    finally:
        # Ensure cache is flushed and closed for WAL checkpoint
        pv_cache.flush()  # Commit any pending writes
        pv_cache.close()
    
    # Print cache statistics
    cache_stats = pv_cache.get_stats()
    logger.info(f"PV Cache Statistics:")
    logger.info(f"  Hits: {cache_stats['hits']}")
    logger.info(f"  Misses: {cache_stats['misses']}")
    if cache_stats['hits'] + cache_stats['misses'] > 0:
        hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
        logger.info(f"  Hit Rate: {hit_rate:.2%}")

    logger.info(f"Claims written: {processed_with_evidence}")
    logger.info(f"Claims skipped (no evidence): {skipped_no_evidence}")
    logger.info(f"Claim processing errors: {claim_error_count}")

    if claim_error_count > 0:
        logger.error(
            "Run finished with claim-level errors; temp logs kept for inspection and final logs were not replaced."
        )
        logger.error(f"Claim temp log: {claim_log_tmp}")
        if args.pair_log_mode != "none":
            logger.error(f"Pair temp log: {pair_log_tmp}")
        raise RuntimeError(
            f"Component 1 run failed with {claim_error_count} claim processing errors."
        )

    # Promote temp logs to final outputs atomically once successful.
    claim_log_tmp.replace(claim_log_file)
    if args.pair_log_mode != "none":
        pair_log_tmp.replace(pair_log_file)
    elif pair_log_file.exists():
        pair_log_file.unlink()

    logger.info("Processing complete!")
    logger.info(f"Claim-level logs: {claim_log_file}")
    if args.pair_log_mode != "none":
        logger.info(f"Pair-level logs: {pair_log_file}")


if __name__ == "__main__":
    main()
