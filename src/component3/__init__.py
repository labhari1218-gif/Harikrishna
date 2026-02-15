"""Component 3 package."""

from .backtracking import RuleBasedBacktrackingController
from .controller import (
    Component5JointLossAdapter,
    ControllerMLP,
    LearnedBacktrackingController,
    compute_component5_joint_loss,
    run_loss_ablation,
)
from .diagnostics import compute_component4_diagnostics, run_component4_diagnostics
from .esm_memory import ESMMemoryStore

__all__ = [
    "Component5JointLossAdapter",
    "ControllerMLP",
    "ESMMemoryStore",
    "LearnedBacktrackingController",
    "RuleBasedBacktrackingController",
    "compute_component5_joint_loss",
    "compute_component4_diagnostics",
    "run_loss_ablation",
    "run_component4_diagnostics",
]

try:
    from .router import (
        CLAIM_TYPES,
        Component6ModelRouter,
        Component6Config,
        DifficultyConfig,
        DifficultyFeatures,
        ModelRouterConfig,
        RouteInput,
        RouterClassifierConfig,
        RouterTrainingExample,
        RoutingDevExample,
        run_component6_pipeline,
        score_claim_difficulty,
        train_router_classifier,
        tune_routing_threshold,
    )
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.extend(
        [
            "CLAIM_TYPES",
            "Component6Config",
            "Component6ModelRouter",
            "DifficultyConfig",
            "DifficultyFeatures",
            "ModelRouterConfig",
            "RouteInput",
            "RouterClassifierConfig",
            "RouterTrainingExample",
            "RoutingDevExample",
            "run_component6_pipeline",
            "score_claim_difficulty",
            "train_router_classifier",
            "tune_routing_threshold",
        ]
    )

# Optional heavyweight exports (e.g., torch-backed components).
try:
    from .pv_bridge import PVBridgeRecoveryEngine
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("PVBridgeRecoveryEngine")

try:
    from .pv_dataset import FactKGPVDatasetGraph
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("FactKGPVDatasetGraph")

try:
    from .c1_pairs_loader import load_component1_evidence_rows
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("load_component1_evidence_rows")

try:
    from .c1_pairs_loader import load_component1_evidence_rows_fever
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("load_component1_evidence_rows_fever")

try:
    from .pv_qagnn import PV_QAGNN
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("PV_QAGNN")

try:
    from .pv_dataset_fever import FeverPVDatasetGraph
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.append("FeverPVDatasetGraph")

try:
    from .run_validation import ValidationConfig, run_full_validation
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.extend(["ValidationConfig", "run_full_validation"])

try:
    from .run_component7 import Component7Config, run_component7_evaluation
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional deps
    pass
else:
    __all__.extend(["Component7Config", "run_component7_evaluation"])
