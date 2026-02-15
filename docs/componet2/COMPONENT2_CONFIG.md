# Component 2 Living Config

This file is the single source of defaults used in code. Update this whenever a default changes.

## Current Defaults (M1)

| Key | Value | Source |
|---|---:|---|
| `anchor_max_k` | `5` | `COMPONENT2_DECISIONS.md` (Anchors, locked M1) |
| `anchor_fallback_degree_k` | `2` | `COMPONENT2_DECISIONS.md` (Anchors fallback) |
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
| `recovery_tau_esi` | `0.3` | `COMPONENT2_DECISIONS.md` (Recovery) |
| `recovery_tau_rpi_legacy` | `0.15` | `COMPONENT2_DECISIONS.md` (Recovery legacy reference) |
| `recovery_top_k` | `3` | `COMPONENT2_DECISIONS.md` (Recovery) |
| `abstain_alpha` | `0.5` | `COMPONENT2_DECISIONS.md` (Abstention) |
| `smoke_max_claims` | `5` | M1 acceptance requirement (CPU run on 5 claims) |

## Code Mapping

- Runtime dataclass: `src/component2/config.py` (`Component2Config`)
- Smoke runner snapshot: `logs/component2/.../run_config.json`

## Change Log

- **2026-02-10:** Created initial living config for M1. Added missing defaults (`anchor_max_k=5`, `anchor_fallback_degree_k=2`, `random_seed=13`, `message_passing_layers=1`) and linked each to decisions.
