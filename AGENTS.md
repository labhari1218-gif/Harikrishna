# AGENTS.md - Component 3 Recovery and SOTA Plan (Planning Stage)

## Purpose
This file is the execution contract for fixing low Component 3 performance in this repo.
It is intentionally planning-first: establish verified reality, then execute fixes in controlled phases.

Use this when starting a new Codex chat for Component 3 remediation.

## Mode
- Stage: `planning`
- Priority: correctness and traceability over speed
- Rule: every claim about behavior must be tied to code or run artifacts

## Execution anchors (single-source runtime contract)
- `WORKING_BRANCH`: `fix/component1-v4-hardening`
- `BASELINE_COMMIT`: `d7c8cd3`
- `BASELINE_RUN_ID`: `runs/t33_full_seed57_rerun7_tmux_20260215_1755`
- `BASELINE_RUN_CMD` (wrapper entrypoint):
  - `bash scripts/t33_full_recovery_run.sh t33_full_seed57_rerun7_tmux_20260215_1755`
- `BASELINE_TRAIN_CMD` (inside wrapper, must match Phase-0 audit):
  - `python -m component3.run_train --run-id t33_full_seed57_rerun7_tmux_20260215_1755 ...`
- If your local run used a different entrypoint, update these anchors before executing fixes.

## Environment and artifact contract
- Required local artifacts (no silent fallback):
  - `data/embeddings.pkl` is required for Component 3 node features.
  - `logs/component1/<split>/pairs.jsonl` is required when `--use-component1-pairs` is enabled.
  - `data/claim_triple_embeddings.pkl` is required in strict cache mode (`--factkg-require-claim-triple-cache`).
- Failure policy:
  - fail loudly with actionable error if required artifacts are missing.
  - do not fallback to hash embeddings for Component 3.

## Current Reality (Verified from code plus one run on 2026-02-15)

### Run under investigation
- Run: `runs/t33_full_seed57_rerun7_tmux_20260215_1755`
- Config: `enable_component5: false`, `dataset_name: factkg`, `batch_size: 8`, `max_seq_len: 256`
- Test overall: accuracy `0.6021`, F1 `0.5858`
- Weak slices:
  - negation accuracy `0.4056`
  - substitution precision `0.0428`

### What actually runs
1. If `--use-component1-pairs` is enabled, Component 1 pair logs are loaded and aligned by `claim_id` (must be confirmed per run via Phase-0 audit output).
2. FactKG dataset graph uses only `A` and `C` triples.
3. PV-QAGNN does dual streams with trainable gates.
4. Training uses claim BCE + auxiliary edge losses.
5. Backtracking/router are not in the train/eval path of this run.

### BERT usage (two roles)
- Node embeddings:
  - precomputed entity/sentence vectors provided in dataset graphs.
  - these are not re-encoded per batch in PV-QAGNN forward.
- Claim representation and relevance:
  - claim tokens are encoded in model forward and used for relevance/logits.
  - if encoder layers are mostly frozen, hard semantic slices (negation/substitution) usually stagnate.

### Code-level evidence
- S dropped in FactKG dataset path:
  - `src/component3/pv_dataset.py:278` (search: `_parse_ac_triples_for_index`)
  - `src/component3/pv_dataset.py:286` (search: `if triple.pool in {"A", "C"}`)
- FactKG training wrapper uses this dataset:
  - `src/component3/run_train.py:615` (search: `_build_factkg_loaders`)
  - `src/component3/run_train.py:659` (search: `FactKGPVDatasetGraph(`)
- Backtracking exists but is not wired into `run_train.py`/`evaluate.py` flow:
  - controller implementation: `src/component3/backtracking.py:121` (search: `class RuleBasedBacktrackingController`)
  - training loop entry: `src/component3/run_train.py:899` (search: `def run_training(`)
  - eval call used: `src/component3/run_train.py:1065` (search: `evaluate_on_test_set(`)
- Router exists but not used in Component 3 training path:
  - `src/component3/router.py:1` (search: `Component 6 router`)
- Edge features are rich in dataset (`768 + 5`) but GAT message passing uses `edge_dim=1`:
  - edge features built: `src/component3/pv_dataset.py:500` (search: `edge_feature = torch.cat((claim_triple_embedding, pv_meta)`)
  - GAT setup: `src/component3/pv_qagnn.py:102` (search: `edge_dim=1`)
- Model computes edge relevance from claim-node cosine, not from dataset `rel`:
  - `src/component3/pv_qagnn.py:133` (search: `node_relevance = F.cosine_similarity`)
  - `src/component3/pv_qagnn.py:135` (search: `edge_relevance = 0.5 *`)
- C5 losses are disabled in investigated run:
  - config file: `runs/t33_full_seed57_rerun7_tmux_20260215_1755/config.yaml`
  - zeros in metrics: `runs/t33_full_seed57_rerun7_tmux_20260215_1755/metrics.json`

### Important artifact interpretation
- `logs/component1/*/claims.jsonl`:
  - claim-level summary of A/S/C counts and diagnostics from Component 1.
  - proof of partitioning and sufficiency diagnostics, not proof of C3 usage.
- `logs/component1/*/pairs.jsonl`:
  - row-level evidence with `pool`, `probs`, `rel`.
  - direct input source for C3 when `--use-component1-pairs` is enabled.
- `runs/.../train.log`:
  - process output and final JSON dump.
  - does not prove backtracking/router activation unless explicit traces are logged.

## Target Architecture (What we planned)
- Graph reasoner starts from `A + C`.
- Suspended `S` remains available for bounded recovery.
- Triggered backtracking promotes top-k `S -> A`, reruns once/twice.
- C5 losses (starvation/counter/recovery) are optional phase-2 upgrades.
- Router is a later optimization after core baseline is fixed.

## Paper-backed Locks
- QA-GNN is base model family.
- CO-GAT: soft gating, warmup, auxiliary evidence supervision.
- Component 2 bridge rescue: anchor-conditioned weighted PPR and bridge bonus.
  - formula implementation: `src/component2/bridge_rescue.py:216`
  - neutral cap: `src/component2/bridge_rescue.py:217`
- FactKG claim-type taxonomy used for slice reporting.
- VitaminC and ProoFVer are robustness/faithfulness references, not direct training objectives.

## Non-negotiable Engineering Rules
- Keep 8GB-safe defaults (`batch<=8`, `max_seq_len<=256`) unless profiling proves headroom.
- Do not claim SOTA with synthetic metrics.
  - `run_component7.py` currently defaults to synthetic metrics and must be disabled for real claims.
- Do not compare to hardcoded SOTA numbers unless evaluation settings are matched and documented.
- Every phase must emit machine-readable artifacts and pass targeted tests.

## Recovery Program (Phased Plan)

### Phase 0 - Forensic baseline and instrumentation
Goal: make runtime behavior auditable before changing model logic.

Tasks:
1. Add explicit runtime flags to `metrics.json`:
   - `used_s_pool`, `used_backtracking`, `used_router`, `used_component5`.
2. Emit per-run evidence contract summary:
   - A/S/C counts seen by C3 loader.
3. Add a one-page method trace in artifacts:
   - from input logs to final prediction path.

Acceptance:
- You can answer "was S available and used?" from one artifact file.

### Phase 1 - Fix A/S/C data path
Goal: preserve S for recovery while still running initial graph on A+C.

Tasks:
1. FactKG dataset:
   - add `include_s_pool` handling analogous to FEVER design.
   - keep A+C for initial graph edges.
   - preserve S rows per claim as recoverable pool metadata.
2. Keep dual evidence labels:
   - support target from A
   - counter target from C
   - do not train a single "good evidence" head where C is always target `0`.
   - use either two heads (support/counter) or one 3-way edge-role head.
   - locked default objective:
     - `L_evidence = 0.5 * (BCE(support_head, is_A) + BCE(counter_head, is_C))`
   - optional alternative objective:
     - `L_edge_role = CE(edge_role_logits, {support,counter,other})`
3. Add tests:
   - S retention metadata exists.
   - initial graph excludes S.
   - support/counter heads receive correct labels.

Acceptance:
- C3 can access S per claim without contaminating initial graph edges.

### Phase 2 - Integrate inference-time backtracking (minimal viable)
Goal: make backtracking real before training a learned controller.

Tasks:
1. Wire `RuleBasedBacktrackingController` or `PVBridgeRecoveryEngine` into evaluation path.
2. Trigger policy:
   - low margin OR low connectivity OR evidence hunger.
   - conservative execution gate: only perform promotion if ranked S candidates include at least one `connects_components=true`.
3. Ranking policy for S:
   - `connects_components`
   - PPR `bridge_bonus`
   - `rel`
   - `salience x p_ent` tie-breaker
   - compute PPR on Round-0 graph built from A+C edges.
   - seed PPR from claim anchor entities (claim entity set / claim-linked anchors from C1 logs).
   - explicit bridge score for S edge `(u,v)`:
     - `bridge_bonus(u,v) = ppr[u] * (eps + rel_uv) * ppr[v] * neutral_cap(p_neu_uv)`
4. Run bounded recovery:
   - rounds <= 2
   - k <= 3
5. Emit `recovery_actions.jsonl` and `recovery_candidates.jsonl`.

Acceptance:
- At least one real claim in eval shows logged S->A promotion and rerun.

### Phase 3 - Use edge features in message passing
Goal: stop wasting claim-conditioned edge representations.

Tasks:
1. Replace the `edge_dim=1` bottleneck strategy.
2. Default implementation:
   - learned edge scorer MLP `773 -> 1` per stream, fused with PV gates.
3. Optional fallback experiment:
   - full edge attr in GAT (`edge_dim=773`) only if MLP path plateaus and memory allows.
4. Blend relevance signals:
   - dataset `rel` plus semantic relevance (do not drop either without ablation).
5. Add ablation toggles:
   - cosine-only
   - rel-only
   - blended

Acceptance:
- Measurable delta over baseline on validation within small and medium runs.

### Phase 4 - Unfreezing and optimization
Goal: improve semantic slices (negation/substitution) beyond frozen-encoder limits.

Tasks:
1. Add configurable unfreezing (minimal viable default):
   - LoRA on top layers, or unfreeze last 2 transformer layers.
2. Optional sweep:
   - unfreeze last N layers for N in a small bounded grid.
3. Keep warmup and early stop.
4. Add calibration checks per type:
   - precision-recall behavior for substitution and negation.

Acceptance:
- Negation and substitution precision improve without collapse in overall metrics.

### Phase 5 - Fair SOTA comparison
Goal: eliminate apples-to-oranges reporting.

Tasks:
1. Replace hardcoded comparison constants:
   - `scripts/compare_component3_to_sota.py`
2. Require explicit parity metadata for each baseline:
   - dataset split
   - split/source hashes for data artifacts
   - evidence budget/top-k
   - evidence source policy (oracle/retrieved)
   - label space (binary vs 3-way)
   - preprocessing path
   - encoder backbone and tuning policy
   - compute setup (single-model vs multi-stage/LLM)
   - seeds and mean+-std
3. Disable synthetic metrics for publishable tables:
   - `src/component3/run_component7.py` with `use_synthetic_metrics=false`

Acceptance:
- SOTA report includes settings parity table and provenance per number.

### Phase 6 - Explainability contract
Goal: claim-level, auditable rationales.

Output schema per claim:
- `claim_id`, `claim_text`
- `gold_label`, `pred_label`, `prob`, `margin`
- `pool_items[]` with:
  - `evidence_id`, `triple_text`, `pool`
  - `p_ent`, `p_con`, `p_neu`, `rel`
  - stream/gating attribution fields
- `backtracking` block:
  - `triggered`, `reason`, `promoted_s_to_a[]`, `margin_before`, `margin_after`
- diagnostics:
  - `is_disconnected_graph`, `active_high_rel_count`, `cr_at_5`, `esi_geom`

Acceptance:
- You can inspect one JSON line and explain exactly why a claim was support/refute.

## Experiment Matrix (Required)

### Small (smoke)
- Purpose: logic validation, no expensive training.
- Suggested run:
  - `--train-subset-size 1000 --val-subset-size 1000`
  - 1-2 epochs per stage
- Gate:
  - artifacts generated
  - no shape/contract errors

### Medium (directional)
- Purpose: check expected metric direction and slice gains.
- Suggested run:
  - `--train-subset-size 10000 --val-subset-size 2000`
  - full stage schedule
- Gate:
  - improvement on negation/substitution precision over baseline snapshot

### Full (reportable)
- Purpose: final claims and comparisons.
- Suggested run:
  - full train/val/test
  - 3 seeds (`42, 1337, 2026`)
- Gate:
  - mean +- std reported
  - fair SOTA protocol satisfied

## Router decision
- Now: defer router for main accuracy rescue.
- Later: re-enable after Phases 1-4 are stable.
- Reason: current deficits are core data/graph/training path issues, not routing policy.

## Risks to watch
- Synthetic metrics accidentally mixed with real evaluation outputs.
- Backtracking enabled but never triggered due to missing S or too strict trigger.
- Edge feature integration increasing memory without gains.
- Dataset mismatch (binary vs 3-way) in FEVER runs.

## Definition of Done
1. S pool is retained and used for bounded recovery.
2. Backtracking actions are logged and attributable.
3. Edge claim-conditioned features affect message passing.
4. Negation/substitution precision materially improve over run `t33_full_seed57_rerun7_tmux_20260215_1755`.
5. SOTA comparison is fair and reproducible.
6. Per-claim explainability JSONL is complete and stable.

## Copy-Paste Prompt for Codex (Planning-first execution)
Use this prompt in a new Codex chat:

```text
Read first:
1) AGENTS.md
2) PLAN.md
3) docs/BEST_PRACTICES.md
4) docs/TRAINING_SPEC.md
5) runs/t33_full_seed57_rerun7_tmux_20260215_1755/{config.yaml,metrics.json,train.log,sota_comparison.json}
6) src/component3/{run_train.py,pv_dataset.py,pv_qagnn.py,backtracking.py,pv_bridge.py,router.py,run_component7.py}
7) scripts/{t33_full_recovery_run.sh,compare_component3_to_sota.py}

Mode: planning stage first, then phased execution.

Rules:
- Do not start coding before writing a phase plan with acceptance criteria.
- Tie every diagnosis to file:line evidence.
- Keep changes minimal and testable per phase.
- Prefer inference-time backtracking integration before learned controller training.
- Do not use synthetic metrics for any SOTA claim.

Required output format:
1) System Diagram: current reality vs target
2) Severity-ordered root causes with file:line refs
3) Phase plan (0-6) with concrete edits and test commands
4) Ablation matrix (small/medium/full) with stop criteria
5) Explainability JSONL contract

Then execute Phase 0 and Phase 1 in code, run targeted tests, and report artifacts.
```

## Quick command references
- Full orchestrated run:
  - `bash scripts/t33_full_recovery_run.sh <run_id>`
- Direct train:
  - `python src/component3/run_train.py --run-id <run_id> --use-component1-pairs --component1-logs-root logs/component1`
  - or `python -m component3.run_train --run-id <run_id> --use-component1-pairs --component1-logs-root logs/component1`
  - confirm import path/PYTHONPATH behavior in Phase-0 audit.
- SOTA compare helper (to be hardened):
  - `python scripts/compare_component3_to_sota.py --run-metrics runs/<run_id>/metrics.json --baseline-metrics runs/t33_tmux_20260215_095704/metrics.json`
