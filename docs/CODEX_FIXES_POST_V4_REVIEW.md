# Codex Post-v4 Review Fixes - Checklist

**Date**: 2026-02-02  
**Purpose**: Apply final paper-grade hardening to ensure robustness, prevent API misuse, and enhance metrics quality

---

## A. CR@k API - Make Impossible to Misuse

### File: `src/component1/sufficiency_metrics.py`
- [ ] Change `compute_cr_at_k` to keyword-only signature: `def compute_cr_at_k(*, pool_items, counter_items, ks=(5,10))`
  - **Rationale**: Prevents argument order bugs by forcing keyword usage
- [ ] Rename parameters internally: `C` → `counter_items`, `Pool` → `pool_items`
  - **Rationale**: Clear, self-documenting parameter names
- [ ] Fix denominator for small pools: `denom = min(k, len(pool_sorted))`
  - **Rationale**: Handles edge case where pool size < k correctly

### File: `src/component1/logging_utils.py`
- [ ] Update `compute_cr_at_k` call to use keyword arguments: `compute_cr_at_k(pool_items=pool, counter_items=C, ks=[5, 10])`
  - **Rationale**: Complies with new keyword-only API

### File: `tests/component1/test_logging_utils.py`
- [ ] Fix import path: `from component1.evidence import` (remove `src.` prefix)
  - **Rationale**: Tests fail with ModuleNotFoundError for `src` module
- [ ] Fix typo in test data: `" max_contra_all"` → `"max_contra_all"`
  - **Rationale**: Space in key causes KeyError
- [ ] Update test to use keyword args for `compute_cr_at_k`
  - **Rationale**: Tests should validate the new API
- [ ] Add test for positional args rejection: `test_cr_at_k_rejects_positional()`
  - **Rationale**: Ensures keyword-only enforcement works
- [ ] Add test for small pool size: `test_cr_at_k_small_pool()`
  - **Rationale**: Validates denominator fix for pools with size < k

---

## B. save_examples - Robust ESI_geom Ranking

### File: `src/component1/logging_utils.py`
- [ ] Already uses `.get("esi_geom", .get("esi_prod", 0.0))` - verify no changes needed
  - **Rationale**: Falls back gracefully if esi_geom missing

### File: `tests/component1/test_logging_utils.py`
- [ ] Add test case with only `esi_prod` (no `esi_geom`): `test_save_examples_fallback_to_esi_prod()`
  - **Rationale**: Validates fallback logic works correctly

---

## C. Claim/Subgraph Index Alignment

### File: `scripts/run_component1_pv_esm.py`
- [ ] Add `reset_index(drop=True)` after loading claims_df and subgraphs_df
  - **Rationale**: Ensures positional alignment even if original indices are non-sequential
- [ ] Change iteration from `iterrows()` to `range(len(claims_df))`
  - **Rationale**: Explicit positional indexing prevents misalignment
- [ ] Use positional index `i` for claim_id: `claim_id = f"{split}_{i}"`
  - **Rationale**: Consistent with position-based alignment

---

## D. Pool Size & Counter Metrics Enhancement

### File: `src/component1/logging_utils.py`
- [ ] Add `pool_stats` fields:
  - `pool_size_total`: `len(pool)`
  - `pool_size_after_C`: `len(pool) - len(C)`
  - `min_A_feasible`: `min(min_A, len(pool) - len(C))`
  - **Rationale**: Provides context for understanding Active set shortfalls
- [ ] Add counter metrics in A:
  - `counter_in_A`: count of items in A with `p_contra >= contra_tau`
  - `max_contra_A`: max p_contra in A (or 0.0 if empty)
  - **Rationale**: Tracks contradictions that weren't selected for C but ended up in A

---

## E. ESM Documentation Consistency

### File: `src/component1/esm.py`
- [ ] Review Suspended (S) description - currently claims "neutral-dominant" but is just "remaining after A selection"
  - **Rationale**: Documentation should match actual implementation
- [ ] **Decision**: Keep current impl, just clarify in docstring that S = remaining after A/C selection
  - **Rationale**: Avoid changing core logic, just improve clarity

---

## F. Final Integration Validation

- [ ] Run: `conda run -n fact_check_env pytest -q tests/component1`
- [ ] Run: `conda run -n fact_check_env python3 scripts/run_component1_pv_esm.py --split train --limit_claims 50 --pair_log_mode sample --pair_log_sample_rate 0.05`
- [ ] Run: `conda run -n fact_check_env python3 scripts/summarize_component1.py --split train`
- [ ] Verify `logs/component1/train/summary.json` exists and has correct schema

---

## G. Documentation Updates

### File: `DEVLOG_COMPONENT1.md`
- [ ] Add section: "Codex Post-v4 Review Fixes (2026-02-02)"
- [ ] Include: what changed, why, exact commands used
- [ ] Reference this checklist document

---

**Status**: Ready to implement
