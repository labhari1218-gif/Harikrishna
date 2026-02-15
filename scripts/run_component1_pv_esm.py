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

# Add repository root to path FIRST (before other imports)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
from tqdm import tqdm

from component1 import (
    EvidenceItem,
    PVScorer,
    PVConfig,
    PVCache,
    EvidenceStateManager,
    ESMConfig,
    evidence_to_text,
    VERBALIZER_VERSION,
)
from component1.evidence import stable_hash
from component1.logging_utils import (
    write_claim_log,
    write_pair_log,
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
    
    # Extract triples from walked['walkable']
    walked = subgraph_row.get('walked', {})
    walkable_triples = walked.get('walkable', [])
    
    for idx, triple in enumerate(walkable_triples):
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
    pv_scorer: PVScorer,
    pv_cache: PVCache,
    esm_config: ESMConfig,
    pv_config: PVConfig,
    pair_log_config: PairLogConfig,
    claim_log_file: Path,
    pair_log_file: Path
):
    """
    Process a single claim through the PV+ESM pipeline.
    
    Args:
        claim_id: Unique claim identifier
        claim_text: Claim text
        label: Ground truth label
        claim_entities: List of claim entity strings
        evidence_pool: List of EvidenceItem objects
        pv_scorer: PVScorer instance
        pv_cache: PVCache instance
        esm_config: ESM configuration
        pv_config: PV configuration
        pair_log_config: Pair logging configuration
        claim_log_file: Path to claim-level JSONL
        pair_log_file: Path to pair-level JSONL
    """
    if not evidence_pool:
        logger.warning(f"Claim {claim_id} has no evidence. Skipping.")
        return
    
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
            claim_id,
            evidence_hash,
            pv_config.model_name,
            pv_config.max_length,
            VERBALIZER_VERSION
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
        pv_results = pv_scorer.pv_score_many(claim_text, to_score_premises)
        
        for i, pv_result in enumerate(pv_results):
            idx = to_score_indices[i]
            item = evidence_pool[idx]
            item.pv = pv_result
            
            # Store in cache
            evidence_hash = stable_hash(to_score_premises[i])
            pv_cache.put(
                claim_id,
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
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base-mnli",
                       help="HuggingFace NLI model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use (auto=detect, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for PV scoring (conservative for 8GB GPU)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max token length")
    
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
    
    args = parser.parse_args()
    
    # Create output directories
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    
    claim_log_file = out_dir / "claims.jsonl"
    pair_log_file = out_dir / "pairs.jsonl"
    
    # Clear existing logs
    if claim_log_file.exists():
        claim_log_file.unlink()
    if pair_log_file.exists():
        pair_log_file.unlink()
    
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Claim log: {claim_log_file}")
    if args.pair_log_mode != "none":
        logger.info(f"Pair log: {pair_log_file}")
    
    # Initialize PV scorer
    pv_config = PVConfig(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    pv_scorer = PVScorer(pv_config)
    
    # Initialize cache
    pv_cache = PVCache(args.cache_dir, read_only=False)
    
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
        logger.info(f"Limited to {len(claims_df)} claims")
    
    # Process claims
    logger.info(f"Processing {len(claims_df)} claims...")
    
    # Process claims with proper cleanup
    try:
        for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
            claim_id = f"{args.split}_{idx}" # Reverted to original claim_id generation
            claim_text = row['Sentence'] # Reverted to original column name
            
            # Extract label (boolean list -> string)
            label_list = row.get('Label', [None]) # Reverted to original column name
            label = "TRUE" if label_list and label_list[0] else "FALSE" # Reverted to original logic
            
            # Extract claim entities
            claim_entities = row.get('Entity_set', []) # Reverted to original column name
            
            # Get subgraph evidence
            if idx >= len(subgraphs_df):
                logger.warning(f"No subgraph for claim {claim_id}. Skipping.")
                continue
            
            subgraph_row = subgraphs_df.iloc[idx]
            evidence_pool = create_evidence_pool(claim_id, subgraph_row) # Reverted to original function call
            
            # Process claim
            try:
                process_claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    label=label, # Reverted to original parameter name
                    claim_entities=claim_entities,
                    evidence_pool=evidence_pool, # Reverted to original parameter name
                    pv_scorer=pv_scorer,
                    pv_cache=pv_cache,
                    esm_config=esm_config,
                    pv_config=pv_config, # Kept original parameter
                    pair_log_config=pair_log_config,
                    claim_log_file=claim_log_file,
                    pair_log_file=pair_log_file
                )
            except Exception as e:
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
    
    logger.info("Processing complete!")
    logger.info(f"Claim-level logs: {claim_log_file}")
    if args.pair_log_mode != "none":
        logger.info(f"Pair-level logs: {pair_log_file}")


if __name__ == "__main__":
    main()
