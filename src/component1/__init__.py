"""
Component 1: PV Scorer + Evidence State Manager

Production-ready implementation of premise validation scoring,
evidence state management (A/S/C partitioning), and sufficiency metrics.

FIX 3: Lightweight __init__.py - only pure Python dataclasses/utils.
Import PVScorer, PVCache directly from submodules (component1.pv, component1.cache)
to avoid loading transformers at package import time.
"""

__version__ = "1.0.0"
__all__ = [
    "EvidenceItem",
    "PVResult",
    "EvidenceStateManager",
    "ESMConfig",
    "evidence_to_text",
    "VERBALIZER_VERSION",
]

# Only import pure-Python dataclasses and utilities (no heavy dependencies)
from .evidence import EvidenceItem, PVResult, evidence_to_text, VERBALIZER_VERSION
from .esm import EvidenceStateManager, ESMConfig

# For PVScorer, PVConfig, PVCache: import from submodules directly
# Example: from component1.pv import PVScorer, PVConfig
# Example: from component1.cache import PVCache
