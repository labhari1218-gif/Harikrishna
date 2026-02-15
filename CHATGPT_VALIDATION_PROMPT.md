# ChatGPT Validation Prompt: Component 1 Post-v4 + Codex Remaining Fixes

**Date**: 2026-02-02  
**Package**: Component 1 (PV + ESM + Sufficiency Metrics)  
**Version**: Post-v4 Hardening + Codex Remaining Fixes

---

## Objective

Please review the Component 1 codebase for correctness, robustness, and production-readiness. This package has undergone two rounds of hardening:
1. **Post-v4 Hardening** (5 Codex review fixes: CR@k API, index alignment, enhanced metrics, etc.)
2. **Codex Remaining Fixes** (4 additional fixes: contra_tau consistency, legacy compatibility, lightweight imports, ESM docstring)

Focus on validating the most recent **4 Codex remaining fixes** applied today.

---

## Context: What This Code Does

Component 1 implements:
- **Premise Validation (PV)**: NLI-based scoring of evidence triples against claims
- **Evidence State Manager (ESM)**: Partitions evidence into Active (A), Suspended (S), Counter (C) sets
- **Sufficiency Metrics**: ESI (Evidence Sufficiency Index), CR@k (Counter Retention), RPI@k (Recovery Potential)
- **Logging**: Claim/pair-level JSONL logs with comprehensive metrics

**Key invariants**:
- |A| >= min_A (unless pool too small)
- Coverage = 1.0 (all claim entities found in Active set)
- Connectivity = 1.0 (claim entities form connected graph)

---

## Fixes to Validate (Most Recent)

### FIX 1: Counter Threshold Consistency (CORRECTNESS)

**Problem**: `write_claim_log()` used hardcoded `contra_tau=0.5` for `counter_in_A` metric, but ESM uses configurable `contra_tau` (default 0.6).

**Changes Made**:
1. `src/component1/logging_utils.py`:
   - Added `contra_tau: float` parameter to `write_claim_log()` signature (line 47)
   - Removed hardcoded `contra_tau = 0.5` (was line 108)
   - Now uses parameter: `counter_in_A = sum(1 for e in A if e.pv.p_contra >= contra_tau)`

2. `scripts/run_component1_pv_esm.py`:
   - Updated call to pass `contra_tau=esm_config.contra_tau` (line 216)

**Questions to Validate**:
- ✅ Does `write_claim_log()` signature include `contra_tau` parameter?
- ✅ Is the hardcoded `contra_tau = 0.5` removed?
- ✅ Does the call site in `run_component1_pv_esm.py` pass `esm_config.contra_tau`?
- ✅ Is `counter_in_A` calculation consistent with ESM's Counter selection logic?

---

### FIX 2: Legacy Log Compatibility (ROBUSTNESS)

**Problem**: `summarize_component1.py` crashed on logs missing `esi_geom` (only had `esi_prod`).

**Changes Made**:
`scripts/summarize_component1.py`:
- Replaced direct dict access with `.get()` fallbacks (lines ~49-92)
- ESI extraction with fallback:
  ```python
  for c in claims:
      suff = c.get("sufficiency", {})
      esi_geom = suff.get("esi_geom")
      if esi_geom is None:
          esi_geom = suff.get("esi_prod", 0.0)
      esi_geom_values.append(esi_geom)
  ```
- All nested access uses `.get()`: counts, counter_retention, recovery, starvation

**Questions to Validate**:
- ✅ Does ESI extraction have explicit fallback from `esi_geom` to `esi_prod`?
- ✅ Are all nested dict accesses using `.get()` with defaults?
- ✅ Would the code handle a legacy log (only esi_prod, missing esi_geom) without crashing?

---

### FIX 3: Lightweight Package Import (PORTABILITY)

**Problem**: `from component1 import PVScorer` loaded transformers at import time (slow, heavy dependency).

**Changes Made**:
1. `src/component1/__init__.py`:
   - Removed `PVScorer`, `PVConfig`, `PVCache` from imports
   - Only exports pure dataclasses: `EvidenceItem`, `PVResult`, `ESMConfig`, etc.
   - Added comment directing users to import heavy modules from submodules

2. `scripts/run_component1_pv_esm.py`:
   - Changed to direct submodule imports:
     ```python
     from component1.pv import PVScorer, PVConfig
     from component1.cache import PVCache
     ```

**Impact**: Test import time reduced from 2.5s to 0.06s (42× faster)

**Questions to Validate**:
- ✅ Does `__init__.py` avoid importing `pv.py` or `cache.py`?
- ✅ Does `run_component1_pv_esm.py` import from `component1.pv` and `component1.cache` directly?
- ✅ Can someone run `from component1 import EvidenceItem` without loading transformers?

---

### FIX 4: ESM Docstring Accuracy (DOCUMENTATION)

**Problem**: ESM docstring said S = "Neutral-dominant evidence (p_neutral > 0.5)" but implementation is "remaining after C and A selection".

**Changes Made**:
`src/component1/esm.py`:
- Updated Suspended (S) docstring (lines ~61-63):
  - Old: "Neutral-dominant evidence (p_neutral > 0.5)"
  - New: "Remaining evidence after selecting C then A"
  - Added: "Typically lower relevance items that didn't make top-max_A cut"

**Questions to Validate**:
- ✅ Does the ESM `partition()` docstring accurately describe how S is formed?
- ✅ Is the "neutral-dominant" language removed?
- ✅ Does the description match the actual implementation (S = pool - C - A)?

---

## General Code Quality Review

Beyond the 4 specific fixes, please check:

### 1. CR@k API (From Previous Post-v4 Hardening)
- ✅ Is `compute_cr_at_k()` in `sufficiency_metrics.py` keyword-only? (uses `*,` marker)
- ✅ Does small-pool denominator use `min(k, len(pool_sorted))`?
- ✅ Are call sites using keyword args: `compute_cr_at_k(pool_items=..., counter_items=..., ks=...)`?

### 2. Index Alignment (From Previous Post-v4 Hardening)
- ✅ Does `run_component1_pv_esm.py` call `reset_index(drop=True)` on both claims_df and subgraphs_df?
- ✅ Does the loop use `range(len(claims_df))` with `iloc[i]` for positional access?
- ✅ Is there a length verification check before processing?

### 3. Enhanced Metrics (From Previous Post-v4 Hardening)
Check `logging_utils.py` for:
- ✅ `pool_size_total`, `pool_size_after_C`, `min_A_feasible` in pool_stats
- ✅ `counter_in_A`, `max_contra_A` in counter_retention

### 4. Type Safety & Error Handling
- ✅ Are function signatures type-annotated?
- ✅ Are edge cases handled (empty pools, missing fields, etc.)?
- ✅ Are error messages clear and actionable?

---

## Validation Checklist

Please confirm:

**Correctness**:
- [ ] `counter_in_A` uses passed `contra_tau` (not hardcoded 0.5)
- [ ] ESM docstring matches implementation
- [ ] CR@k denominator handles small pools correctly
- [ ] Index alignment guaranteed via `reset_index()` + positional iteration

**Robustness**:
- [ ] Summarize script has `.get()` fallbacks for all nested access
- [ ] ESI extraction falls back to esi_prod when esi_geom missing
- [ ] Edge cases handled (empty pools, missing claim entities, etc.)

**Portability**:
- [ ] `__init__.py` doesn't import heavy dependencies (transformers)
- [ ] Scripts import PVScorer/PVCache from submodules directly
- [ ] Package can be imported without transformers for lightweight usage

**Best Practices**:
- [ ] Keyword-only API for functions prone to argument-order bugs
- [ ] Clear docstrings matching implementation
- [ ] Consistent threshold usage across modules

---

## Test Results to Review

**Unit Tests**:
```
$ PYTHONPATH=src pytest tests/component1 -q
.....                                                [100%]
5 passed in 0.06s
```

**Integration Test** (20 claims):
```
Processing claims: 100%|██████████| 20/20 [00:00<00:00, 1432.73it/s]
✅ Processing complete! (315 cache hits, 0 misses, 100% hit rate)
```

**Key Metrics**:
- Mean ESI_geom: 0.4894
- Has Counter Rate: 55.00%
- Starved A Rate: **0.00%** (invariant maintained)
- Mean Coverage: 1.0000
- Mean Connectivity: 1.0000

---

## Files to Review (Priority Order)

### High Priority (Recent Changes)
1. `src/component1/logging_utils.py` - FIX 1 (contra_tau parameter)
2. `scripts/summarize_component1.py` - FIX 2 (.get() fallbacks)
3. `src/component1/__init__.py` - FIX 3 (lightweight imports)
4. `scripts/run_component1_pv_esm.py` - FIX 1 & FIX 3 (pass contra_tau, direct imports)
5. `src/component1/esm.py` - FIX 4 (docstring update)

### Medium Priority (Previous Hardening)
6. `src/component1/sufficiency_metrics.py` - CR@k keyword-only API
7. `tests/component1/test_logging_utils.py` - Test coverage

### Context/Reference
8. `DEVLOG_COMPONENT1.md` - Full development history
9. `COMPONENT1_TEST_RESULTS.txt` - Detailed test report

---

## Expected Outcome

Please provide:
1. **Validation Report**: Did all 4 fixes achieve their stated goals?
2. **Code Quality Assessment**: Any bugs, anti-patterns, or improvement opportunities?
3. **Correctness Verification**: Are invariants maintained? Logic sound?
4. **Risk Assessment**: Any potential issues in production usage?

---

## Additional Context

- **Environment**: Python 3.10, conda, CPU execution tested
- **Model**: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- **Dataset**: FactKG (86K claims), using subgraphs from Opsahl et al.
- **Use case**: Evidence sufficiency research for fact-checking

---

Thank you for your review! This code will be used in a research paper on evidence gap assessment.
