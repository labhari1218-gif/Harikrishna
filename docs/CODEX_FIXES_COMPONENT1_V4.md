# Component 1 v4 — Fix List (for Codex)
Repo: /home/bs_thesis/shift (Copy)/Fact-or-Fiction

Goal:
Make Component 1 paper-grade and internally consistent by fixing correctness bugs, log schema mismatches, and performance/robustness issues — WITHOUT breaking the baseline pipeline. All changes must remain behind Component-1 scripts/modules.

Non-goals:
- No training changes
- No router/controller/sufficiency loop beyond metrics
- No changes to baseline train.py pipeline behavior

Acceptance criteria:
1) `pytest -q` passes (all component1 tests + new tests you add).
2) `python3 scripts/run_component1_pv_esm.py --split train --limit_claims 100` runs without errors.
3) `python3 scripts/summarize_component1.py --split train` runs without errors and writes examples.
4) Claim logs schema and DEVLOG/README match exactly (no stale fields).

---

## CRITICAL FIXES (correctness)

### Fix 1 — CR@k computed wrong (arguments swapped)
File: src/component1/logging_utils.py
Problem:
- compute_cr_at_k signature is `compute_cr_at_k(C, Pool, k_values)`
- current call is `compute_cr_at_k(pool, C, [5,10])` (WRONG)

Required change:
- Replace call with: `compute_cr_at_k(C, pool, [5, 10])`

Verification:
- Add a small unit test ensuring CR@k behaves as expected (see "New tests").

---

### Fix 2 — save_examples uses missing field "sufficiency.esi"
File: src/component1/logging_utils.py
Problem:
- save_examples() sorts by `c["sufficiency"]["esi"]` but logs store `esi_geom` and `esi_prod`.

Required change:
- Make examples select the canonical metric: `sufficiency.esi_geom`
- If backward compatibility needed: fallback: `get("esi_geom") or get("esi_prod") or 0.0`

Verification:
- Run summarize script and ensure examples files are created.

---

## DOC/SCHEMA CONSISTENCY

### Fix 3 — DEVLOG and artifact contract mismatch
Files:
- DEVLOG_COMPONENT1.md
- README.md (Component 1 section)

Required change:
- Update JSON schema docs to reflect actual output keys:
  - sufficiency.esi_geom
  - sufficiency.esi_prod
  - counter_retention.cr_at_5, cr_at_10, etc.

Verification:
- Grep for "sufficiency.esi" and ensure it is removed or explained as deprecated.

---

### Fix 4 — ESM docstring mismatch
File: src/component1/esm.py
Problem:
- docstring describes polarity-based counter selection but code uses p_contra >= contra_tau.

Required change:
- Update docstring to match implementation and explain rationale.

---

## PERFORMANCE/ROBUSTNESS (paper-grade)

### Fix 5 — SQLite cache commits too frequently
File: src/component1/cache.py
Problem:
- put() commits every insert, which is slow at scale.

Required change:
- Add buffered commits:
  - Maintain an internal counter.
  - Commit every N inserts (e.g., 256 or 1024).
  - Add `flush()` method to force commit.
- Ensure runner calls cache.flush() before exit.

Verification:
- Run 100-claim job; ensure no errors.
- (Optional) log cache stats, confirm speed improvement.

---

### Fix 6 — WAL checkpoint on close
File: src/component1/cache.py
Problem:
- WAL file can grow.

Required change:
- In close(), call:
  - `PRAGMA wal_checkpoint(TRUNCATE)` in try/finally before closing.

Verification:
- Run a small job; ensure close() doesn’t error.

---

## ANALYSIS CLARITY (optional but recommended)

### Fix 7 — Separate “counter exists in pool” vs “counter kept”
Files:
- src/component1/logging_utils.py
- scripts/summarize_component1.py

Problem:
- current has_counter typically equals (len(C)>0), which confounds:
  - evidence contradiction exists in full pool
  - counter evidence kept in C

Required change:
- Log both:
  - has_counter_in_pool = any(p_contra >= contra_tau) over full pool
  - has_counter_kept = (len(C) > 0)
- Summarize both rates.

Verification:
- Summarize output prints both rates.

---

## NEW TESTS (mandatory)

Add:
- tests/component1/test_logging_utils.py

Tests required:
1) test_cr_at_k_correctness:
   - Build a fake pool of EvidenceItems with pv.p_contra values.
   - Choose C and pool such that TopContra@k membership is known.
   - Ensure CR@k matches expected.
2) test_save_examples_uses_esi_geom:
   - Create a tiny temp JSONL with sufficiency.esi_geom values.
   - Call save_examples(), ensure it writes lowest_esi (or lowest_esi_geom) output without KeyError.

---

## REQUIRED VERIFICATION COMMANDS

From repo root:

1) Unit tests:
   conda run -n fact_check_env pytest -q

2) Component 1 run (100 claims):
   conda run -n fact_check_env python3 scripts/run_component1_pv_esm.py --split train --limit_claims 100

3) Summarize + examples:
   conda run -n fact_check_env python3 scripts/summarize_component1.py --split train

4) Sanity check outputs:
   - logs/component1/train/claims.jsonl exists and has 100 lines
   - logs/component1/train/examples/*.json exists (highest_contradiction, lowest_esi_geom, most_neutral)
   - logs/component1/train/summary.json exists

---

## DELIVERABLES
- Code fixes committed on a new branch (do not modify baseline pipeline).
- Updated DEVLOG_COMPONENT1.md + README.md (Component 1 section).
- New tests added and passing.
- Short diff summary appended to DEVLOG.
