"""Configuration defaults for Component 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Component2Config:
    """Locked defaults for Component 2 implementation."""

    # Anchor selection
    anchor_max_k: int = 5
    anchor_fallback_degree_k: int = 2
    factkg_data_dir: str = "data/factkg"
    factkg_train_pickle: str = "factkg_train.pickle"
    factkg_val_pickle: str = "factkg_dev.pickle"
    factkg_test_pickle: str = "factkg_test.pickle"

    # Reasoner architecture
    hidden_dim: int = 256
    num_heads: int = 4
    rel_emb_dim: int = 64
    random_seed: int = 13
    message_passing_layers: int = 1

    # Fixed PV masks (locked decision)
    mask_sup_scale: float = 4.0
    mask_sup_bias: float = -2.0
    mask_ref_scale: float = 4.0
    mask_ref_bias: float = -2.0
    disable_pv_masks: bool = False

    # Recovery/bridge defaults (reserved for M2)
    ppr_epsilon: float = 0.001
    neutral_cap_gamma: float = 2.0
    ppr_alpha: float = 0.15
    ppr_max_iters: int = 100
    ppr_tolerance: float = 1e-6
    recovery_tau_esi: float = 0.3
    recovery_tau_rpi_legacy: float = 0.15
    recovery_top_k: int = 3

    # Selective prediction defaults (reserved for M3)
    abstain_alpha: float = 0.5
    tau_abstain: float | None = None
    compute_risk_coverage_sweep: bool = True
    rationale_top_n: int = 10
    faithfulness_subset_size: int = 3
    faithfulness_top_k_edges: int = 10

    # Runner defaults
    smoke_max_claims: int = 5
    numpy_edge_warn_threshold: int = 5000

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """Export a JSON-serializable config snapshot."""

        snapshot = asdict(self)
        snapshot.update(
            {
                "mask_sup_alpha": self.mask_sup_alpha,
                "mask_sup_beta": self.mask_sup_beta,
                "mask_ref_alpha": self.mask_ref_alpha,
                "mask_ref_beta": self.mask_ref_beta,
            }
        )
        return snapshot

    @property
    def mask_sup_alpha(self) -> float:
        """Alias for support-mask scale."""

        return float(self.mask_sup_scale)

    @property
    def mask_sup_beta(self) -> float:
        """Alias for support-mask bias."""

        return float(self.mask_sup_bias)

    @property
    def mask_ref_alpha(self) -> float:
        """Alias for refute-mask scale."""

        return float(self.mask_ref_scale)

    @property
    def mask_ref_beta(self) -> float:
        """Alias for refute-mask bias."""

        return float(self.mask_ref_bias)


DEFAULT_CONFIG = Component2Config()
