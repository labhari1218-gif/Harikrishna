# Detailed Project Summary (Till Now)

Generated on: 2026-02-11  
Scope analyzed: code (`src/`, `scripts/`, `tests/`), docs (`*.md`, `docs/`), logs (`logs/`), and repository state (`git`)

## 1) Current Snapshot

- Project folder analyzed: `Fact-or-Fiction`
- Active git branch: `fix/component1-v4-hardening`
- Branch head commit: `1594a17` (`add environment for conda and requirements for python venv`)
- Committed history depth: 43 commits
- Git working tree is heavily ahead locally (not committed yet):
  - Staged changes: 22 entries
  - Unstaged changes: 14 entries
  - Untracked entries (`-uall`): 106 entries
- The repository currently contains both:
  - Older committed baseline work (2024 timeline)
  - Large local 2026 expansions (Component 1 hardening + full Component 2 implementation + M4 ablation runner)

## 2) Historical Timeline (What Was Done Over Time)

### 2.1 Committed baseline timeline (git history)

From the 43 commits (`2024-05-05` to `2024-08-13`):

- Project initialized and baseline structure created.
- Subgraph retrieval implemented and generalized.
- Baseline models and QA-GNN pipeline built and improved.
- Embedding precomputation pipeline added.
- Training/evaluation tooling added and refined.
- ChatGPT prompting/evaluation workflow added.
- README and usage instructions improved repeatedly.
- Environment files (`environment.yaml`, `requirements.txt`) added.

This committed line represents the original "Fact-or-Fiction" project implementation and cleanup.

### 2.2 2026 local development timeline (from dev logs + docs + file dates)

#### Component 1 (major local development wave)

Primary window: `2026-02-02` to `2026-02-11` (based on `DEVLOG_COMPONENT1.md`, docs, script/module mtimes, logs)

- Built Component 1 PV+ESM pipeline modules under `src/component1/`.
- Added runner/summary scripts under `scripts/`.
- Added test suite under `tests/component1/`.
- Iteratively fixed correctness and robustness issues across several internal versions (v1 -> v2 -> v3 -> v4 -> post-v4 hardening).
- Produced large-scale logs for train/val and summaries/examples.

#### Component 2 (full milestone build)

Primary window: `2026-02-10` to `2026-02-11` (from `docs/component2/*`, file mtimes, run logs)

- Implemented milestones M0, M1, M2, M3.
- Applied M3 alignment patch (A-D).
- Started M4 and completed T12/T13 through a full ablation runner (A0/A1/A2/A3).
- Added comprehensive `tests/component2/` suite and multiple run snapshots.

## 3) Codebase Growth and Structure

### 3.1 Component 1 code footprint

`src/component1/`:

- Files: 7
- LOC: 1,694
- Modules:
  - `evidence.py` (dataclasses, verbalization, hashing)
  - `pv.py` (NLI-based PV scorer)
  - `cache.py` (SQLite cache with WAL and buffered commits)
  - `esm.py` (A/S/C partition logic)
  - `sufficiency_metrics.py` (ESI, CR@k, RPI@k)
  - `logging_utils.py` (claim/pair logs + examples)
  - `__init__.py` (lightweight exports)

### 3.2 Component 2 code footprint

`src/component2/`:

- Files: 17
- LOC: 4,374
- Includes:
  - Core M1-M3 stack: `types.py`, `graph_builder.py`, `anchor_selector.py`, `reasoner.py`, `bridge_rescue.py`, `recovery.py`, `selective.py`, `salience.py`
  - Runners: `run_m1.py`, `run_m2.py`, `run_m3.py`
  - M3 patch support: `anchor_adapter_factkg.py`
  - M4 ablation tooling: `ablation_runner.py`, `ablation_utils.py`, `scripts/run_ablation_component2.py`

### 3.3 Test footprint

- `tests/component1/`: 2 files, 310 LOC
- `tests/component2/`: 17 files, 1,900+ LOC
- Total test LOC inspected: 2,210

## 4) Component 1: What Was Implemented and Stabilized

## 4.1 Pipeline behavior implemented

In `scripts/run_component1_pv_esm.py`:

- Loads claims/subgraphs from existing project data loaders.
- Builds evidence pool per claim from walked triples.
- Runs PV scoring with cache lookup/write.
- Partitions evidence into A/S/C using ESM rules.
- Writes:
  - `claims.jsonl` (claim-level metrics)
  - `pairs.jsonl` (pair-level logs, configurable modes)
- Uses temp files and atomic replace on successful completion.
- Adds index alignment guardrails (`reset_index`, positional iteration).

In `scripts/summarize_component1.py`:

- Aggregates claim-level metrics.
- Computes ESI distribution statistics and starve rates.
- Produces examples:
  - highest contradiction
  - lowest ESI
  - most neutral

## 4.2 Key algorithmic additions and fixes (from code + DEVLOG)

- NLI premise/hypothesis order fixed in PV scoring.
- Verbalizer improved (camelCase split, normalized relation text, template phrasing).
- Cache hardened:
  - stable claim hash keying
  - buffered commits
  - WAL mode and checkpoint on close
  - legacy read fallback
- ESM and logging consistency improved (threshold propagation, docstring and schema consistency).
- Sufficiency metrics upgraded:
  - both `esi_prod` and `esi_geom`
  - geometric mean used as primary practical score
  - CR@k keyword-only API to avoid argument order bugs
  - RPI and bridge-related diagnostics included
- Component 2 compatibility fields explicitly logged:
  - `entity_set` / `Entity_set` on claim rows
  - `pool` / `evidence_assignment` on pair rows

## 4.3 Component 1 artifacts present now

From `logs/component1/`:

- `train/claims.jsonl`: 82,083 lines
- `train/pairs.jsonl`: 1,680,348 lines
- `val/claims.jsonl`: 12,491 lines
- `val/pairs.jsonl`: 272,868 lines
- `val/summary.json` exists
- `train_OLD_no_pool_20260211_121858/summary.json` exists (legacy snapshot)

Key metric snapshots from present summaries:

- `logs/component1/val/summary.json`:
  - `num_claims`: 12,491
  - `mean_esi_geom`: 0.3760
  - `mean_coverage_A`: 0.7729
  - `mean_connectivity_A`: 0.4096
  - `has_counter_rate`: 0.4645
- `logs/component1/train_OLD_no_pool_20260211_121858/summary.json`:
  - `num_claims`: 82,083
  - `mean_esi_geom`: 0.3151
  - `mean_coverage_A`: 0.7796
  - `mean_connectivity_A`: 0.3961
  - `has_counter_rate`: 0.4425

Note: current `logs/component1/train/` has huge claims/pairs logs but no `summary.json` at this moment.

## 4.4 Component 1 tests and validation state

Live checks in this workspace:

- `conda run -n fact_check_env python3 -m pytest tests/component1 -q` -> **8 passed**

Environment caveat:

- Running component1 tests with system python + unittest failed earlier because `pytest` was not installed there.
- In the intended conda environment (`fact_check_env`), tests pass.

## 5) Component 2: What Was Implemented

## 5.1 Milestone status from canonical plan

From `docs/component2/COMPONENT2_PLAN.md`:

- M0: complete
- M1: complete
- M2: complete
- M3: complete
- M3 patch (A-D): complete
- M4:
  - T12 complete (no bridge rescue ablation mode)
  - T13 complete (no recovery loop mode)
  - T11 pending
  - T14 pending
  - T15 pending
  - T16 pending

## 5.2 Core architecture implemented in code

- Input contract + normalization: `src/component2/types.py`, `src/component2/io_utils.py`
- Anchor handling:
  - selection and fallbacks: `src/component2/anchor_selector.py`
  - FactKG injection adapter (provided -> pickle -> heuristic): `src/component2/anchor_adapter_factkg.py`
- Graph build:
  - A/C graph and A/S/C graph paths with relation vocab: `src/component2/graph_builder.py`
- Reasoner:
  - hybrid masked dual-stream numpy reasoner: `src/component2/reasoner.py`
- Bridge and recovery:
  - PPR bridge scoring + diagnostics: `src/component2/bridge_rescue.py`
  - policy-aware trigger and one-step recovery: `src/component2/recovery.py`
- Reliability/interpretability:
  - selective prediction / AURC / operating point: `src/component2/selective.py`
  - rationale extraction + faithfulness checks: `src/component2/salience.py`
- Runners and reproducibility snapshots:
  - `run_m1.py`, `run_m2.py`, `run_m3.py`

## 5.3 M3 patch (A-D) reflected in code/docs

Implemented and documented:

- A) Anchor injection from FactKG pickles by default.
- B) Fixed abstention operating point (`tau_abstain`) in addition to sweep.
- C) Explicit relation embedding status (`fixed_features_numpy`, non-trainable currently).
- D) Numpy performance guardrails (`numpy_edge_warn_threshold`) and warning logging.

## 5.4 Component 2 run artifacts present now

`logs/component2/` contains reproducibility snapshots for:

- M1: 2 runs
- M2: 5 runs
- M3: 13 run folders (some partial/empty, others full)

Observed progression in summaries:

- Early M3 run (`20260210_093453_086968`) had `num_ppr_not_converged=100`.
- Later stabilized run (`20260210_094916_968777`) had `num_ppr_not_converged=0`.
- Anchor source logging is present in patch-era runs (e.g., provided vs pickle counts).

## 5.5 M4 ablation runner outputs now available

`logs/ablations/component2/` has 3 completed run directories with full artifacts:

- `20260211_062450_414166`
- `20260211_062856_161382`
- `20260211_062956_534369`

Each includes:

- `run_config.json`
- `selected_claim_ids.txt`
- `predictions_A0..A3.jsonl`
- `metrics_A0..A3.json`
- `summary_metrics.json`
- `summary_table.md`
- `plots/` (PNG/CSV risk-coverage, accuracy-coverage, histograms)

Latest 2000-claim ablation summary (`20260211_062956_534369/summary_table.md`):

- A0: accuracy 0.5315, AURC 0.4674
- A1: accuracy 0.4945, AURC 0.5204
- A2: accuracy 0.4945, AURC 0.5204
- A3: accuracy 0.4945, AURC 0.5225, recovery trigger rate 0.0105, mean ESI improved `0.3789 -> 0.3821`

## 5.6 Component 2 tests and validation state

Live checks in this workspace:

- `conda run -n fact_check_env bash -lc 'PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v'` -> **55 tests passed**
- `conda run -n fact_check_env bash -lc 'PYTHONPATH=src python3 -m unittest discover -s tests -v'` -> **55 tests passed**

(Full `tests` unittest discovery currently covers Component 2 suites; Component 1 is validated via pytest.)

## 6) Documentation and Packaging Work Completed

The workspace contains extensive documentation generated around implementation, validation, packaging, and handoff.

Major sets include:

- Component 1 build/dev/hardening docs:
  - `DEVLOG_COMPONENT1.md`
  - `building.md`
  - `docs/CODEX_FIXES_COMPONENT1_V4.md`
  - `docs/CODEX_FIXES_POST_V4_REVIEW.md`
  - `docs/CODEX_REPORT_V4_HARDENING.md`
  - `COMPONENT1_*` package/readme/presentation files
- Component 2 milestone docs:
  - `docs/component2/COMPONENT2_SPEC_AND_PLAN.md`
  - `docs/component2/COMPONENT2_PLAN.md`
  - `docs/component2/COMPONENT2_DECISIONS.md`
  - `docs/component2/COMPONENT2_CONFIG.md`
  - `docs/component2/DEV_NOTES_M0.md` .. `DEV_NOTES_M3.md`
  - `docs/component2/DEV_NOTES_M3_PATCH.md`
  - `docs/component2/ABLATION_STUDY.md`
- Handoff context:
  - `AGENTS.md` (states continuation from M4)

Also present:

- Legacy typo-mirror docs under `docs/componet2/`.
- TeX fragments for component documentation in `docs/component1/tex/` and `docs/component2/tex/`.

## 7) Important Current Gaps / Pending Work

## 7.1 Component 2 pending milestone tasks

From canonical plan:

- T11: no-PV-mask ablation pending
- T14: stress tests (bridge drop/distractors) pending
- T15: learnable mask coefficients pending
- T16: tau_esi sensitivity sweep pending

## 7.2 Training realism limitation (Component 2)

- Component 2 reasoner is currently numpy inference-style with random-initialized relation features.
- Relation embeddings are documented as non-trainable in current flow.
- PyTorch trainable migration is documented as future work.

## 7.3 Working tree state risk

- A lot of work is local and uncommitted.
- Many docs and source additions exist outside git history.
- Any packaging/handoff should account for this and likely require commit hygiene.

## 7.4 Documentation drift spots

- Some docs are outdated relative to latest code/tests (example: old "known issues" sections in package notes).
- There are duplicated canonical/legacy doc trees (`component2` vs `componet2`).

## 8) What "Done Till Now" Means Practically

At this point, the repository is in a hybrid state:

1. Original project baseline is committed and stable.
2. Component 1 has been significantly built/hardened locally with large train/val artifacts and passing tests in the target conda env.
3. Component 2 M0-M3 + patch is implemented with passing 55-test suite and reproducibility snapshots.
4. M4 has started with a full ablation framework and A0-A3 outputs, but milestone remains incomplete due pending T11/T14/T15/T16.
5. The biggest remaining operational step is to consolidate/commit the local 2026 work and finish pending M4 tasks.

## 9) Key Paths (Quick Index)

- Core project readme: `README.md`
- Component 1 dev log: `DEVLOG_COMPONENT1.md`
- Component 1 build narrative: `building.md`
- Component 1 source: `src/component1/`
- Component 1 runner: `scripts/run_component1_pv_esm.py`
- Component 1 summary script: `scripts/summarize_component1.py`
- Component 1 tests: `tests/component1/`
- Component 1 logs: `logs/component1/`
- Component 2 plan/status: `docs/component2/COMPONENT2_PLAN.md`
- Component 2 source: `src/component2/`
- Component 2 tests: `tests/component2/`
- Component 2 runs: `logs/component2/`
- Component 2 ablations: `logs/ablations/component2/`
- Component 2 run script (M4): `scripts/run_ablation_component2.py`
- Handoff instructions: `AGENTS.md`

