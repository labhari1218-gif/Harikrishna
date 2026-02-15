"""
esm.py - Evidence State Manager for Component 1

Implements the ESM partitioning policy that divides evidence into:
- Active (A): High-relevance evidence driving sufficiency
- Suspended (S): Neutral-dominant evidence (potential for recovery)
- Counter (C): Strong contradiction candidates

Invariants:
- min_A <= |A| <= max_A (unless pool < min_A)
- |C| <= max_C
- A ∪ S ∪ C = evidence_pool (disjoint partition)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .evidence import EvidenceItem

logger = logging.getLogger(__name__)


@dataclass
class ESMConfig:
    """Configuration for Evidence State Manager."""
    min_A: int = 5   # Minimum evidence in Active set
    max_A: int = 20  # Maximum evidence in Active set
    max_C: int = 5   # Maximum evidence in Counter set
    contra_tau: float = 0.6  # Contradiction threshold for Counter (safer default for strong contradictions)


class EvidenceStateManager:
    """
    Evidence State Manager for A/S/C partitioning.
    
    Partitions evidence pool based on PV scores (rel, pol) into:
    - Active (A): Supporting evidence (high rel, positive pol)
    - Suspended (S): Uncertain/weak evidence
    - Counter (C): Contradictory evidence (high rel, negative pol)
    
    Enforces invariants to ensure model stability:
    - Always maintains min_A evidence in Active (or all available if pool < min_A)
    - Caps Active at max_A
    - Caps Counter at max_C
    """
    
    @staticmethod
    def partition(
        evidence_pool: List[EvidenceItem],
        config: ESMConfig
    ) -> Dict[str, List[EvidenceItem]]:
        """
        Partition evidence pool into Active (A), Suspended (S), and Counter (C) sets.
        
        Partitioning rules:
        1. **Counter (C)**: Select evidence with p_contra >= contra_tau
           - Limited to top max_C items by p_contra (default: max_C=2)
           - Ensures min_A guarantee: won't starve A below min_A
        
        2. **Suspended (S)**: Remaining evidence after selecting C then A
           - FIX 4: Evidence not selected for Counter or Active
           - Typically lower relevance items that didn't make top-max_A cut
           - Can be recovered later via backtracking (future work)
        
        3. **Active (A)**: Top-ranked evidence by relevance
           - Selected after C: top-max_A items by rel from remaining pool
           - Drives sufficiency metrics (ESI, coverage, connectivity)
        
        Min_A Guarantee:
        - If selecting C would cause |A| < min_A, C is shrunk to preserve min_A
        - Prevents complete starvation of Active set
        
        Args:
            evidence_pool: List of EvidenceItem with PV scores
            config: ESMConfig (thresholds, size limits)
            
        Returns:
            Dict with keys "A", "S", "C" mapping to lists of EvidenceItem
        """
        if not evidence_pool:
            logger.warning("Empty evidence pool provided to ESM")
            return {"A": [], "S": [], "C": []}
        
        # Verify all evidence has PV scores
        for item in evidence_pool:
            if item.pv is None:
                raise ValueError(f"Evidence {item.evidence_id} missing PV score")
        
        # Step 1 & 2: Identify and select Counter evidence
        # Counter criterion: strong contradiction probability (p_contra >= contra_tau)
        # NOT polarity-based (which can miss high p_contra cases)
        counter_candidates = [
            e for e in evidence_pool
            if e.pv.p_contra >= config.contra_tau
        ]
        
        # Sort Counter candidates by p_contra descending (strongest contradictions first)
        counter_candidates.sort(key=lambda e: e.pv.p_contra, reverse=True)
        
        # CRITICAL: Don't let C starve A below min_A
        # Shrink C if needed to guarantee |remaining| >= min_A
        max_C_safe = max(0, len(evidence_pool) - config.min_A)
        actual_max_C = min(config.max_C, max_C_safe)
        
        C = counter_candidates[:actual_max_C]
        C_ids = {e.evidence_id for e in C}
        
        logger.debug(f"Selected {len(C)} counter evidence items (max_C={config.max_C})")
        
        # Step 3: Remove Counter from pool
        remaining = [e for e in evidence_pool if e.evidence_id not in C_ids]
        
        # Step 4 & 5: Sort remaining by relevance and select Active
        remaining.sort(key=lambda e: e.pv.rel, reverse=True)
        
        # Step 6: Ensure min_A <= |A| <= max_A
        pool_size = len(remaining)
        
        if pool_size < config.min_A:
            # Not enough evidence: take all remaining
            A = remaining
            S = []
            logger.warning(
                f"Evidence pool ({pool_size}) smaller than min_A ({config.min_A}). "
                f"Using all {len(A)} items as Active."
            )
        else:
            # Enough evidence: enforce min_A and max_A
            num_active = max(config.min_A, min(config.max_A, pool_size))
            A = remaining[:num_active]
            S = remaining[num_active:]
            
            logger.debug(
                f"Active: {len(A)}, Suspended: {len(S)} "
                f"(min_A={config.min_A}, max_A={config.max_A})"
            )
        
        # Verify invariants
        assert len(A) <= config.max_A, f"Active set size {len(A)} exceeds max_A {config.max_A}"
        assert len(C) <= config.max_C, f"Counter set size {len(C)} exceeds max_C {config.max_C}"
        
        if pool_size >= config.min_A:
            assert len(A) >= config.min_A, f"Active set size {len(A)} below min_A {config.min_A}"
        
        # Verify disjoint partition
        total = len(A) + len(S) + len(C)
        assert total == len(evidence_pool), \
            f"Partition not complete: {total} != {len(evidence_pool)}"
        
        return {
            "A": A,
            "S": S,
            "C": C
        }
