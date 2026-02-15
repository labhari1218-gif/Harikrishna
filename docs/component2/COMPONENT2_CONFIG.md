# Component 2 Living Config

This file is the single source of defaults used in code. Update this whenever a default changes.

## Current Defaults (M3 + Stability Updates)

| Key | Value | Source |
|---|---:|---|
| `anchor_max_k` | `5` | `COMPONENT2_DECISIONS.md` (Anchors, locked M1) |
| `anchor_fallback_degree_k` | `2` | `COMPONENT2_DECISIONS.md` (Anchors fallback) |
| `factkg_data_dir` | `"data/factkg"` | `COMPONENT2_DECISIONS.md` (Anchor adapter Option A source) |
| `factkg_train_pickle` | `"factkg_train.pickle"` | `COMPONENT2_DECISIONS.md` (Anchor adapter source path) |
| `factkg_val_pickle` | `"factkg_dev.pickle"` | `COMPONENT2_DECISIONS.md` (Anchor adapter source path) |
| `factkg_test_pickle` | `"factkg_test.pickle"` | `COMPONENT2_DECISIONS.md` (Anchor adapter source path) |
| `hidden_dim` | `256` | `COMPONENT2_DECISIONS.md` (Model sizing Option B) |
| `num_heads` | `4` | `COMPONENT2_DECISIONS.md` (Model sizing Option B) |
| `rel_emb_dim` | `64` | `COMPONENT2_DECISIONS.md` (Model sizing Option B) |
| `random_seed` | `13` | `COMPONENT2_DECISIONS.md` (M1 determinism choice) |
| `message_passing_layers` | `1` | `COMPONENT2_DECISIONS.md` (M1 implementation choice) |
| `mask_sup_scale` | `4.0` | `COMPONENT2_DECISIONS.md` (fixed mask Option A) |
| `mask_sup_bias` | `-2.0` | `COMPONENT2_DECISIONS.md` (fixed mask Option A) |
| `mask_ref_scale` | `4.0` | `COMPONENT2_DECISIONS.md` (fixed mask Option A) |
| `mask_ref_bias` | `-2.0` | `COMPONENT2_DECISIONS.md` (fixed mask Option A) |
| `ppr_epsilon` | `0.001` | `COMPONENT2_DECISIONS.md` (Bridge rescue) |
| `neutral_cap_gamma` | `2` | `COMPONENT2_DECISIONS.md` (Bridge rescue) |
| `ppr_alpha` | `0.15` | `COMPONENT2_DECISIONS.md` (Bridge rescue solver defaults) |
| `ppr_max_iters` | `100` | `COMPONENT2_DECISIONS.md` (Bridge rescue solver defaults) |
| `ppr_tolerance` | `1e-6` | `COMPONENT2_DECISIONS.md` (Bridge rescue solver defaults) |
| `recovery_tau_esi` | `0.3` | `COMPONENT2_DECISIONS.md` (Recovery) |
| `recovery_tau_rpi_legacy` | `0.15` | `COMPONENT2_DECISIONS.md` (Recovery legacy reference) |
| `recovery_top_k` | `3` | `COMPONENT2_DECISIONS.md` (Recovery) |
| `abstain_alpha` | `0.5` | `COMPONENT2_DECISIONS.md` (Abstention) |
| `tau_abstain` | `None` | `COMPONENT2_DECISIONS.md` (optional fixed selective operating point) |
| `compute_risk_coverage_sweep` | `True` | `COMPONENT2_DECISIONS.md` (keep sweep/AURC by default) |
| `rationale_top_n` | `10` | `COMPONENT2_DECISIONS.md` (Rationale Top-N) |
| `faithfulness_subset_size` | `3` | `COMPONENT2_DECISIONS.md` (Rationale validation subset size) |
| `faithfulness_top_k_edges` | `10` | `COMPONENT2_DECISIONS.md` (Leave-one-out edge budget) |
| `smoke_max_claims` | `5` | M1 acceptance requirement (CPU run on 5 claims) |
| `numpy_edge_warn_threshold` | `5000` | `COMPONENT2_DECISIONS.md` (numpy performance guardrail) |
| `ablation_default_n_claims` | `2000` | M4 ablation runner default (`scripts/run_ablation_component2.py`) |
| `ablation_recommended_n_claims_range` | `1000-5000` | M4 ablation study requirement (subset runtime bound) |
| `ablation_default_seed` | `13` | M4 ablation runner deterministic subset seed |
| `ablation_default_tau_abstain_fixed` | `0.60` | M4 selective operating-point default (`S2`) |

## Code Mapping

- Runtime dataclass: `src/component2/config.py` (`Component2Config`)
- Smoke runner snapshot: `logs/component2/.../run_config.json`

## Behavioral Notes (non-numeric guards)

- Anchor source priority is `provided -> pickle -> heuristic` in all runners (`run_m1.py`, `run_m2.py`, `run_m3.py`).
- Anchor adapter mapping keys:
  - `train_{idx}` from `factkg_train.pickle`
  - `val_{idx}` and alias `dev_{idx}` from `factkg_dev.pickle`
  - `test_{idx}` from `factkg_test.pickle`
  - fallback key: exact `claim_text`
- Anchor selection supplements seed anchors with fallback candidates until at least 2 anchors when the graph has at least 2 valid entities.
- Blank entities are ignored during anchor extraction/fallback matching.
- Recovery connectivity treats missing anchors as disconnected (rather than dropping them from pair connectivity checks).
- Selective prediction supports both risk-coverage sweep (AURC) and optional fixed `tau_abstain`.
- Numpy-mode performance warnings are emitted when claim edge count exceeds `numpy_edge_warn_threshold`.
- Ablation runner executes two selective modes per ablation: `S1` sweep (`tau_abstain=None`) and `S2` fixed threshold (`tau_abstain=0.60` by default).

## Change Log

- **2026-02-10:** Created initial living config for M1. Added missing defaults (`anchor_max_k=5`, `anchor_fallback_degree_k=2`, `random_seed=13`, `message_passing_layers=1`) and linked each to decisions.
- **2026-02-10:** Updated for M2 bridge rescue + recovery. Added PPR solver defaults (`ppr_alpha=0.15`, `ppr_max_iters=100`, `ppr_tolerance=1e-6`) and aligned runtime config snapshot with `run_m2.py`.
- **2026-02-10:** Updated for M3 reliability + interpretability. Added `rationale_top_n=10`, `faithfulness_subset_size=3`, and `faithfulness_top_k_edges=10`; aligned runtime snapshot with `run_m3.py`.
- **2026-02-10:** Post-M3 stability audit added behavioral guard notes (anchor augmentation/filtering and missing-anchor recovery semantics) and synchronized docs with latest M2/M3 validation runs.
- **2026-02-10 (A-D fixes patch):** Added FactKG anchor adapter defaults (`factkg_*`), selective fixed-point knobs (`tau_abstain`, `compute_risk_coverage_sweep`), and perf guardrail threshold (`numpy_edge_warn_threshold`). Updated runtime snapshots to log anchor-source diagnostics and fixed-point selective metrics.
- **2026-02-11:** Added M4 ablation runner defaults (`ablation_default_n_claims=2000`, recommended range `1000-5000`, `ablation_default_seed=13`, `ablation_default_tau_abstain_fixed=0.60`) and documented dual selective modes (`S1` sweep + `S2` fixed tau).
