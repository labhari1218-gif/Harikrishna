"""Component 2 package."""

from .ablation_runner import ABLATION_MODES, run_ablation_study
from .anchor_adapter_factkg import AnchorInjectionResult, FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .bridge_rescue import BridgeRescuePPR, PPRDiagnostics
from .config import Component2Config, DEFAULT_CONFIG
from .graph_builder import GraphBuilder
from .mask_tuning import MaskParams, MaskTuningSettings, tune_mask_coefficients
from .reasoner import HybridMaskedDualStreamReasoner
from .recovery import BridgeRecoveryPolicy
from .salience import extract_top_rationale_edges, grouped_edge_removal_faithfulness, leave_one_out_faithfulness
from .selective import abstain_score, evaluate_selective_prediction, selective_operating_point
from .types import ClaimRecord, EvidenceGraph, EvidenceTriple

__all__ = [
    "ABLATION_MODES",
    "AnchorInjectionResult",
    "FactKGAnchorAdapter",
    "AnchorSelector",
    "abstain_score",
    "BridgeRecoveryPolicy",
    "BridgeRescuePPR",
    "PPRDiagnostics",
    "Component2Config",
    "DEFAULT_CONFIG",
    "evaluate_selective_prediction",
    "GraphBuilder",
    "grouped_edge_removal_faithfulness",
    "HybridMaskedDualStreamReasoner",
    "MaskParams",
    "MaskTuningSettings",
    "extract_top_rationale_edges",
    "leave_one_out_faithfulness",
    "run_ablation_study",
    "ClaimRecord",
    "EvidenceGraph",
    "EvidenceTriple",
    "selective_operating_point",
    "tune_mask_coefficients",
]
