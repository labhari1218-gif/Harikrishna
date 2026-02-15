# Codex Runbook — Component 1 v4 Hardening
Repo: /home/bs_thesis/shift (Copy)/Fact-or-Fiction

Purpose:
A repeatable, file-by-file procedure for Codex to:
- find correctness bugs
- patch minimal diffs
- run verification at each step
- update docs + DEVLOG
- keep baseline pipeline unchanged

Authoritative checklist:
- docs/CODEX_FIXES_COMPONENT1_V4.md (MUST follow)

Non-goals:
- No training changes
- No modifications to baseline training behavior
- No router/controller/sufficiency loop changes beyond metrics
- No new heavy dependencies

---

## 0) Setup (branch + sanity checks)

From repo root:

1) Create branch:
   git checkout -b fix/component1-v4-hardening

2) Confirm paths:
   ls -la src/component1
   ls -la scripts
   ls -la docs/CODEX_FIXES_COMPONENT1_V4.md

3) Pre-scan for known breakpoints:
   rg -n 'sufficiency\]\["esi"\]|compute_cr_at_k\(' -S .
   rg -n 'esi_geom|esi_prod' -S src/component1 scripts docs DEVLOG_COMPONENT1.md README.md

4) Compile check (syntax):
   python3 -m compileall src/component1

If any compile errors happen now, fix them first (they block progress).

---

## 1) Fix Order (do in this exact sequence)

### Fix 1 — CR@k call argument order (correctness)
Target file:
- src/component1/logging_utils.py

Steps:
1) Locate call site:
   rg -n 'compute_cr_at_k\(' -S src/component1/logging_utils.py

2) Patch minimal diff:
   - Replace compute_cr_at_k(pool, C, [5,10]) with compute_cr_at_k(C, pool, [5,10])

3) After patch:
   python3 -m compileall src/component1

4) Run tests (if exist yet, otherwise run after Fix 2 when tests are added):
   conda run -n fact_check_env pytest -q

---

### Fix 2 — save_examples uses missing field (correctness)
Target file:
- src/component1/logging_utils.py

Steps:
1) Locate missing field usage:
   rg -n 'sufficiency"\]\["esi"\]' -S src/component1/logging_utils.py

2) Patch:
   - Use sufficiency.esi_geom as canonical
   - Fallback to esi_prod if esi_geom missing
   - Never KeyError

3) After patch:
   python3 -m compileall src/component1

4) Minimal runtime verification:
   - Create a tiny temp JSONL or reuse a small existing claims.jsonl
   - Run summarize script on it (or run full summarize after Fix 4):
     conda run -n fact_check_env python3 scripts/summarize_component1.py --split train

---

### Fix 3 — DEVLOG/README schema consistency (docs)
Targets:
- DEVLOG_COMPONENT1.md
- README.md
- docs/* if needed

Steps:
1) Find stale schema references:
   rg -n 'sufficiency\.esi|sufficiency"\]\["esi"\]' -S DEVLOG_COMPONENT1.md README.md docs

2) Patch:
   - Replace with sufficiency.esi_geom / sufficiency.esi_prod
   - Ensure artifact contract matches actual JSON written by write_claim_log()

3) After patch:
   (no compileall needed, but do a quick grep sanity)
   rg -n 'sufficiency\.esi([^_]|$)|\["esi"\]' -S DEVLOG_COMPONENT1.md README.md docs || true

---

### Fix 4 — ESM docstring mismatch (docs/code clarity)
Target:
- src/component1/esm.py

Steps:
1) Open docstring and ensure it matches implementation:
   - Counter uses p_contra >= contra_tau
   - A/S partition selection rationale

2) After patch:
   python3 -m compileall src/component1

---

### Fix 5 — SQLite buffered commits + flush (performance)
Target:
- src/component1/cache.py
Also update runner to call flush.

Steps:
1) Implement:
   - internal write counter
   - commit every N inserts (e.g. 256/1024)
   - add flush() that commits immediately

2) Ensure runner script calls:
   - cache.flush() in normal completion
   - cache.flush() + cache.close() in finally block

3) After patch:
   python3 -m compileall src/component1

---

### Fix 6 — WAL checkpoint on close (robustness)
Target:
- src/component1/cache.py

Steps:
1) In close():
   - Execute PRAGMA wal_checkpoint(TRUNCATE) in try/finally
   - Then close connection

2) After patch:
   python3 -m compileall src/component1

---

### Fix 7 (optional) — Separate has_counter_in_pool vs has_counter_kept (analysis clarity)
Targets:
- src/component1/logging_utils.py
- scripts/summarize_component1.py

Steps:
1) Log both:
   - has_counter_in_pool = any(p_contra >= contra_tau) over full pool
   - has_counter_kept = len(C) > 0

2) Update summary to print both rates.

3) After patch:
   python3 -m compileall src/component1

---

## 2) Add new tests (mandatory)

Create:
- tests/component1/test_logging_utils.py

Tests required:
1) test_cr_at_k_correctness:
   - Create pool with known p_contra ordering
   - Ensure CR@5 / CR@10 match expected

2) test_save_examples_uses_esi_geom:
   - Create temp jsonl with sufficiency.esi_geom
   - Call save_examples() and ensure it writes files without KeyError

After tests:
- python3 -m compileall src/component1
- conda run -n fact_check_env pytest -q

---

## 3) Full verification (must pass)

From repo root:

1) Unit tests:
   conda run -n fact_check_env pytest -q

2) Component 1 run (100 claims):
   conda run -n fact_check_env python3 scripts/run_component1_pv_esm.py --split train --limit_claims 100

3) Summarize + examples:
   conda run -n fact_check_env python3 scripts/summarize_component1.py --split train

4) Check artifacts:
   test -f logs/component1/train/claims.jsonl
   test -f logs/component1/train/summary.json
   ls -la logs/component1/train/examples

---

## 4) DEVLOG update (required)

Append a new entry to DEVLOG_COMPONENT1.md:
- “v4 hardening”
- list fixes applied (Fix 1..Fix 7)
- files changed
- exact commands run + results

Also ensure README Component 1 section matches the real CLI fields + JSON schema.

---

## 5) Final report (Codex must produce)

At the end, Codex should output:
- files changed
- summary of diffs (short bullets)
- verification commands + outputs (brief)
- any remaining TODOs (if any)
