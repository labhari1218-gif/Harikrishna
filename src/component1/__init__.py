"""
Component 1: PV Scorer + Evidence State Manager

Production-ready implementation of premise validation scoring,
evidence state management (A/S/C partitioning), and sufficiency metrics.
"""

__version__ = "1.0.0"
__all__ = [
    "EvidenceItem",
    "PVResult",
    "PVScorer",
    "PVConfig",
    "PVCache",
    "EvidenceStateManager",
    "ESMConfig",
    "evidence_to_text",
    "VERBALIZER_VERSION",
]

from .evidence import EvidenceItem, PVResult, evidence_to_text, VERBALIZER_VERSION
from .pv import PVScorer, PVConfig
from .cache import PVCache
from .esm import EvidenceStateManager, ESMConfig
