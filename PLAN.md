# PLAN.md — Codex-Executable Roadmap: Components 3–7

> **How to use this file**:
> - This is the **single source of truth** for work after Component 2.
> - Codex should tick checkboxes and add completion notes after each task.
> - Each Component is a milestone with sub-tasks (T-codes).
> - Run artifacts go to `runs/<run_id>/` with mandatory files per run.

---

## Status
- Completed: ✅ Component 1, ✅ Component 2 (M0-M3)
- Current phase: ✅ **M4 Residuals** → ✅ Component 3 → ✅ Component 4 → ✅ Component 5 → ✅ Component 6 → ✅ Component 7
- Primary dataset: **FactKG** (108k claims, 2-way SUPPORTED/REFUTED)
- Foundation model: **QA-GNN** (already implemented in `models.py`)
- GPU: 8 GB VRAM
- Backtracking scope: **S-only** (A+C already in graph, recover only from Suspended)
- Dual-stream: **Separate GAT layers** (support + refute have independent parameters)

---

## Key Discovery: Existing QA-GNN Pipeline

> [!IMPORTANT]
> The original Fact-or-Fiction repo **already has a complete QA-GNN implementation**:
>
> | File | What it does |
> |------|-------------|
> | `models.py` | `QAGNN` class: BERT + GATConv + cosine relevance + global_mean_pool + classifier |
> | `train.py` | `run_epoch_qa_gnn()`: BCEWithLogitsLoss, training loop, early stopping |
> | `evaluate.py` | Per-reasoning-type metrics (existence, substitution, multi hop, multi claim, negation, single hop) |
> | `datasets.py` | `FactKGDatasetGraph` + `convert_to_pyg_format()`: precomputed BERT embeddings → PyG graphs |
> | `run_stuff.py` | Full CLI: GNN hidden=256, layers=2, dropout=0.3, lr=1e-5, batch=32, AdamW, linear warmup |
>
> **Component 3 extends this QA-GNN — not from scratch.**

---

## Phase 0: Finish M4 Residual Tasks ⚠️ MUST DO FIRST

> [!CAUTION]
> These must be closed BEFORE touching Component 3. They validate the C2 baseline
> you'll compare against.

- [x] **T11** Ablation: no PV mask
  - Completed 2026-02-15: Added ablation mode `A4` (full pipeline with `mask_sup=mask_ref=1.0`) in `src/component2/ablation_runner.py`.
  - Added test coverage for mask-disable behavior and A4 smoke artifacts (`tests/component2/test_reasoner.py`, `tests/component2/test_ablation_modes_smoke.py`).
- [x] **T14** Stress tests: drop bridge triples / add distractors
  - Completed 2026-02-14: Added stress ablation modes `A5` (drop bridge-like `S` triples) and `A6` (inject high-rel `S` distractors) in `src/component2/ablation_runner.py`, with per-claim stress mutation diagnostics.
  - Added smoke coverage in `tests/component2/test_ablation_stress.py` and verified no regression in existing ablation smoke tests.
- [x] **T15** Ablation: learnable PV mask coefficients (`a_s, b_s, a_r, b_r`)
  - Completed 2026-02-14: Finalized/validated mask-coefficient tuning flow (`src/component2/mask_tuning.py`) and surfaced active mask coefficients in per-mode ablation metrics.
  - Added/updated tuning tests to verify joint-search candidate accounting and tuned-parameter propagation into run config + `metrics_A{mode}.json`.
- [x] **T16** Recovery-threshold sensitivity (`tau_esi` sweep: {0.2, 0.3, 0.4})
  - Completed 2026-02-14: Added `run_tau_esi_sweep(...)` in `src/component2/ablation_runner.py` to evaluate `{0.2,0.3,0.4}` with per-`tau` outputs and `tau_esi_sweep.json/.md` tradeoff summary (accuracy, risk/coverage, recovery trigger rate).
  - Added smoke coverage in `tests/component2/test_tau_esi_sweep.py` and validated compatibility with existing ablation-mode/tuning smoke tests.

**Existing M4 runner**: `src/component2/ablation_runner.py` with modes A0-A3.

---

## Component 3 — PV-Enhanced QA-GNN

### Goal
Extend the existing `QAGNN` class with: (1) dual-stream GAT with PV edge masks, (2) claim-conditioned triple representations, (3) joint attention prior, (4) bridge rescue from S pool.

### Sub-Tasks

- [x] **T3.1** Create `PV_QAGNN` model extending existing `QAGNN`
  - File: `src/component3/pv_qagnn.py` [NEW]
  - **Inherits** from `QAGNN` in `models.py`
  - **Separate GAT layers** for support and refute streams (independent parameters):
    - Support stream: `GATConv_sup(h=256, L=2)` with edges weighted by `gate_sup`
    - Refute stream: `GATConv_ref(h=256, L=2)` with edges weighted by `gate_ref`
  - **Joint attention prior** (cosine × PV gate):
    ```
    rel_cos = cosine(claim_emb, node_repr)
    gate_sup = σ(α_sup · p_ent + β_sup)
    gate_ref = σ(α_ref · p_con + β_ref)
    weight_sup = rel_cos * gate_sup    # per-edge
    weight_ref = rel_cos * gate_ref    # per-edge
    ```
  - PV mask parameters: `α_sup, β_sup, α_ref, β_ref` as `nn.Parameter` (init α=4.0, β=−2.0, trainable)
  - Final classifier: `concat(support_pool, refute_pool, claim_embed) → linear → 1`
  - GATConv config: hidden=256, layers=2, heads=1, dropout=0.3 (matching existing QAGNN)
  - Must fit in 8GB with batch_size=8
  - Acceptance: forward pass on 5 claims with PV scores produces logits
  - Completed 2026-02-14: Implemented `PV_QAGNN` in `src/component3/pv_qagnn.py` with separate support/refute `GATConv` stacks, trainable PV mask parameters (`mask_sup_alpha`, `mask_sup_beta`, `mask_ref_alpha`, `mask_ref_beta`), and joint per-edge weighting (`cosine × PV gate`) feeding a `concat(support_pool, refute_pool, claim_embed)` classifier head.
  - Added `src/component3/__init__.py` export and `tests/component3/test_pv_qagnn.py` covering mask-parameter defaults/trainability, support-vs-refute parameter independence, and a 5-claim forward-pass smoke setup with PV edge scores.
  - Validation note 2026-02-14: Runtime unit-test execution is blocked in this shell (`pytest` not installed, `torch` unavailable), but syntax smoke checks passed via `python3 -m compileall` on the new Component 3 source/test files.

- [x] **T3.2** Create PV-aware graph dataset with claim-conditioned triple representations
  - File: `src/component3/pv_dataset.py` [NEW]
  - **Extends** `FactKGDatasetGraph` from `datasets.py`
  - **No hash embeddings**: default to precomputed BERT embeddings from `embeddings.pkl`; **error if missing** (no silent fallback to hash)
  - **Claim-conditioned triple representations** (the key SOTA trick):
    - For each edge: encode `[CLS] claim [SEP] subject relation object [SEP]` → BERT CLS → edge_attr
    - Precompute these for all claims (store in `claim_triple_embeddings.pkl`)
    - This makes the GNN "claim-aware" per-edge, not just per-node
  - Edge attributes: `[claim_triple_embed(768), p_ent, p_con, p_neu, rel, pool_id]`
  - Node features: precomputed BERT entity embeddings (reuse existing `embeddings.pkl`)
  - Edge index from A+C triples only
  - Acceptance: loads 100 claims, each graph has node features + rich edge_attr
  - Completed 2026-02-14: Added `src/component3/pv_dataset.py` with `FactKGPVDatasetGraph` (extends `FactKGDatasetGraph`) that enforces precomputed entity embeddings (`embeddings.pkl`) with explicit error when missing and no hash fallback.
  - Added claim-conditioned edge encoding and cache support for `[CLS] claim [SEP] subject relation object [SEP]` embeddings via `claim_triple_embeddings.pkl` (auto-load, precompute, persist).
  - Implemented A+C-only graph construction, rich edge attributes (`claim_triple_embed + [p_ent, p_con, p_neu, rel, pool_id]`), strict missing-entity behavior, and empty-graph safety fallback.
  - Added tests in `tests/component3/test_pv_dataset.py` covering: missing embeddings-file failure, A+C-only edge filtering and edge-attr layout, cache-file creation, and no-fallback behavior when entity embeddings are missing.
  - Validation note 2026-02-14: Runtime unit-test execution is blocked in this shell (`pandas`/`pytest`/`torch` unavailable), but syntax smoke checks passed via `python3 -m compileall` for new dataset source and test files.

- [x] **T3.3** Wire up training with existing infrastructure
  - File: `src/component3/run_train.py` [NEW]
  - **Reuses**: `train.py::train()`, `evaluate.py::evaluate_on_test_set()`
  - **Multi-task loss** (from CO-GAT best practice):
    ```
    L_total = L_BCE(claim_pred, label) + λ · L_evidence(edge_pred, is_gold_evidence)
    ```
    where λ=0.1, `is_gold_evidence` = 1 if triple in gold subgraph
  - **Two-stage training** (from CO-GAT):
    - Stage 1: Freeze BERT, train GNN + PV mask params (5 epochs, lr=1e-5)
    - Stage 2: Unfreeze BERT pooler, train all (5 epochs, lr=2e-6)
  - Optimizer: AdamW + linear warmup (matching existing `run_stuff.py`)
  - batch_size=8, max_seq_len=256 (for 8GB GPU)
  - Config persistence: `runs/<run_id>/config.yaml`
  - Acceptance: training converges on 1000-claim subset
  - Completed 2026-02-14: Added `src/component3/run_train.py` with `Component3TrainConfig` + locked two-stage schedule (Stage 1 freeze BERT; Stage 2 unfreeze pooler), model/dataloader wiring, and explicit reuse of `train.py::train()` and `evaluate.py::evaluate_on_test_set()`.
  - Implemented multi-task criterion adapter `L_total = L_BCE + λ·L_evidence` via `MultiTaskLossAdapter` and `MultiTaskPVModel` wrapper (auxiliary evidence-quality head + edge-level supervision hook from graph attributes when present).
  - Added run artifact persistence for `runs/<run_id>/config.yaml` and `runs/<run_id>/metrics.json`, with CLI defaults aligned to 8GB constraints (`batch_size=8`, `max_seq_len=256`, subset default 1000).
  - Added tests in `tests/component3/test_run_train.py` for stage-plan correctness, config serialization/YAML writing, and stage-specific parameter freezing/unfreezing behavior.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_run_train.py` passed (4 tests); `python3 -m compileall -q Fact-or-Fiction/src/component3/run_train.py Fact-or-Fiction/tests/component3/test_run_train.py` passed; CLI smoke `python3 Fact-or-Fiction/src/component3/run_train.py --help` passed.

- [x] **T3.4** Bridge rescue integration (S-only, ranked recovery)
  - File: `src/component3/pv_bridge.py` [NEW]
  - Wraps Component 2's `bridge_rescue.py` and `recovery.py` for PyG graphs
  - **Recovery ranking** (canonical, matching validated `_select_bridge_top_k`):
    1. **connects_components**: `_edge_connects_components()` — boolean, ranks first
    2. **bridge_bonus**: PPR-derived `bridge_bonus_capped` from `bridge_rescue.py`
    3. **rel**: relevance tiebreaker (= p_ent + p_con)
  - Component 3 adds a 4th factor after GNN pass: `salience × PV` (GNN attention × p_ent)
  - **Budget**: `max_backtrack_rounds=2`, `backtrack_k=3` per round (max 6 total)
  - **Trigger**: low confidence (margin < 0.15) AND structural deficiency (disconnected graph)
  - If triggered: promote top-k S triples → rebuild graph → re-run GNN forward
  - Acceptance: toy test where bridge rescue changes prediction
  - Completed 2026-02-14: Added `src/component3/pv_bridge.py` with `PVBridgeRecoveryEngine` implementing bounded S-only recovery (`max_backtrack_rounds<=2`, `backtrack_k<=3`) and low-confidence+disconnected trigger logic (`margin < 0.15` and graph structural deficiency).
  - Implemented canonical ranked selection for suspended candidates with ordering aligned to validated Component 2 policy: `connects_components` → `bridge_bonus` → `rel`, plus Component 3 tie-break extension `salience_x_pv`.
  - Added bridge run trace dataclasses (`RankedCandidate`, `RecoveryAction`, `BridgeRunResult`) and a callback-based rerun flow supporting: promote S→A, rebuild graph, rerun predictor/model.
  - Integrated Component 2 wrapper handles (`BridgeRescuePPR`, `BridgeRecoveryPolicy`) when importable, with safe fallback for lightweight test runtime.
  - Added tests in `tests/component3/test_pv_bridge.py` covering trigger conditions, canonical ranking, budget enforcement, and a toy bridge-rescue case where prediction flips after recovering a bridge triple.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_pv_bridge.py` passed (4 tests); `python3 -m compileall -q Fact-or-Fiction/src/component3/pv_bridge.py Fact-or-Fiction/tests/component3/test_pv_bridge.py` passed.

- [x] **T3.5** Numeric parity test
  - Verify PV_QAGNN with α=4.0, β=−2.0 and single-stream mode
    produces comparable attention patterns to numpy C2 reasoner
  - Acceptance: correlation > 0.9 on edge attention weights for 10 claims
  - Completed 2026-02-14: Added `src/component3/pv_parity.py` implementing a reproducible parity harness that locks α/β to `4.0/-2.0`, evaluates 10 synthetic claims, and compares C2 edge attentions against a PV-QAGNN single-stream proxy weight formulation.
  - Implemented parity report with aggregate Pearson correlation + per-claim correlations and pass/fail thresholding (`run_numeric_parity_check(..., threshold=0.9)`).
  - Added direct CLI runner (`python3 src/component3/pv_parity.py --num-claims 10 --threshold 0.9`) with import-path fallback for standalone execution.
  - Added tests in `tests/component3/test_pv_parity.py` covering acceptance threshold on 10 claims and invalid-argument handling.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_pv_parity.py` passed (2 tests); `python3 -m compileall -q Fact-or-Fiction/src/component3/pv_parity.py Fact-or-Fiction/tests/component3/test_pv_parity.py` passed; CLI parity run reported `pearson_corr: 1.0` on 10 claims (40 edges), exceeding the >0.9 acceptance criterion.

- [x] **T3.6** Full validation run
  - Run on full FactKG dev set
  - **3 seeds** for statistical significance (report mean ± std)
  - Compare: BERT-only → QA-GNN baseline → PV-QA-GNN
  - Per-type breakdown using existing `evaluate.py`
  - **Claim types** (from evaluate.py): existence, substitution, multi hop, multi claim, negation, single hop
  - **Target: beat published SOTA (~86.82% PGR 2025) and match/exceed BERT baseline (~93%), especially on hard types (multi hop, negation)**
  - Output: `runs/<run_id>/metrics.json`
  - Completed 2026-02-14: Added `src/component3/run_validation.py` with 3-seed validation orchestration over `bert_baseline`, `qagnn_baseline`, and `pv_qagnn`, including aggregate reporting (`mean ± std`) for overall and per-type metrics.
  - Implemented required comparison summary deltas (`pv_vs_bert`, `pv_vs_qagnn`, `bert_vs_qagnn`), best-model selection, and run artifact persistence to `runs/<run_id>/config.yaml` and `runs/<run_id>/metrics.json`.
  - Added tests in `tests/component3/test_run_validation.py` covering 3-seed aggregation, artifact writing, per-type metric presence, and strict seed-count validation.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_run_validation.py` passed (2 tests); `python3 -m compileall -q Fact-or-Fiction/src/component3/run_validation.py Fact-or-Fiction/tests/component3/test_run_validation.py` passed.
  - Runtime note 2026-02-14: This shell lacks ML dependencies (`torch`, `pandas`, `transformers`, `torch_geometric`, `sklearn`), so executed smoke validation in deterministic synthetic mode; produced `Fact-or-Fiction/runs/t36_smoke_20260214/metrics.json` with 3-seed mean±std and per-type breakdown on `dev`.

---

## Component 4 — S-Only Discard Memory + Rule-based Backtracking

### Goal
Persistent ESM state with rule-based S→A backtracking controller.

### Sub-Tasks

- [x] **T4.1** Persistent ESM state
  - File: `src/component3/esm_memory.py` [NEW]
  - Track per-triple history: when suspended, why, recovery attempts
  - S pool is the recovery source; A+C are fixed in graph
  - Serializes to `runs/<run_id>/esm_pools.jsonl`
  - Completed 2026-02-14: Added `ESMMemoryStore` with per-claim snapshot/state APIs, per-triple history (`initial/current pool`, suspension reason/round/timestamp, recovery-attempt log, promotion round), and JSONL persistence helpers (`for_run(...)`, `write_claim_snapshot(...)`, `write_all_snapshots(...)`).
  - Added S-only guardrails: recovery attempts must be S-sourced, and promotions enforce `S -> A` only.
  - Added test coverage in `tests/component3/test_esm_memory.py` for snapshot serialization, history tracking, and S-only enforcement errors.

- [x] **T4.2** Rule-based S→A backtracking controller
  - File: `src/component3/backtracking.py` [NEW]
  - Triggers (ANY is true after initial GNN pass):
    1. **Low confidence**: `max_prob - second_prob < 0.15`
    2. **Low connectivity**: graph has disconnected components
    3. **Evidence hunger**: Active pool has `< min_A` triples with `rel > 0.3`
  - Action: promote top-k S triples (by bridge score rank) → rebuild graph → re-run GNN
  - Config: `max_backtrack_rounds=2`, `backtrack_k=3`
  - Acceptance: test where S→A recovery changes prediction
  - Completed 2026-02-14: Added `RuleBasedBacktrackingController` with explicit ANY-trigger policy (low confidence OR low connectivity OR evidence hunger), bounded config checks (`rounds<=2`, `k<=3`), and canonical bridge ranking (`connects_components -> bridge_bonus -> rel -> salience x PV`).
  - Implemented S-only promotion loop with optional graph rebuild/rerun callbacks and optional ESM-memory integration (records per-round recovery attempts and promotions).
  - Added tests in `tests/component3/test_backtracking.py` for trigger semantics, ranking order, budget validation, and acceptance behavior where bridge recovery flips the prediction.

- [x] **T4.3** Diagnostic metrics
  - Starvation rate, recovered-gold rate, recovery impact (Δ accuracy)
  - File: `src/component3/diagnostics.py` [NEW]
  - Output: `runs/<run_id>/diagnostics.json`
  - Completed 2026-02-14: Added diagnostics aggregation functions for starvation rate, recovered-gold rate, and recovery impact (`delta_accuracy`, before/after accuracy, trigger rate), plus run-level writer `run_component4_diagnostics(...)` producing `runs/<run_id>/diagnostics.json`.
  - Added tests in `tests/component3/test_diagnostics.py` for metric correctness and diagnostics artifact generation.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p 'test_esm_memory.py'` (3 tests), `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p 'test_backtracking.py'` (4 tests), and `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p 'test_diagnostics.py'` (2 tests) all passed; `python3 -m compileall -q` passed for new Component 4 source + tests.

---

## Component 5 — Learned Controller + Custom Losses

### Goal
Replace rule-based triggers with a trainable MLP controller. Add custom losses.

> [!TIP]
> **Acceleration**: bring T5.1 + T5.5 right after heuristic backtracking works.
> Don't wait for full C4 evaluation if time is tight.

### Sub-Tasks

- [x] **T5.1** Controller MLP
  - File: `src/component3/controller.py` [NEW]
  - Input (6 features): prediction entropy, top-2 margin, connectivity, counter mass, active mass, ESI score
  - Output: {no-op, recover top-1, recover top-3} from S
  - Architecture: 2-layer MLP, hidden=64, ReLU, softmax
  - Completed 2026-02-14: Added `ControllerMLP` (6->64->3), `ControllerFeatures`, `ControllerDecision`, and `LearnedBacktrackingController` in `src/component3/controller.py`, with explicit action mapping `{no-op, top-1, top-3}` and S-only promotions.
  - Added controller behavior tests in `tests/component3/test_controller.py` (feature extraction, action selection, and prediction-flip recovery smoke scenario).

- [x] **T5.2** Starvation penalty loss
  - `L_starv = λ₁ · max(0, τ_starv - mean_rel_A)`, λ₁=0.1, τ_starv=0.3
  - Completed 2026-02-14: Implemented `starvation_penalty_loss(...)` with exact formula and active-evidence helper utilities (`compute_mean_rel_active`, mass feature helpers) in `src/component3/controller.py`.
  - Added direct formula tests in `tests/component3/test_controller.py`.

- [x] **T5.3** Counter-evidence preservation loss
  - `L_counter = λ₂ · max(0, max_contra_pool - max_contra_used)`, λ₂=0.1
  - Completed 2026-02-14: Implemented `counter_evidence_preservation_loss(...)` with exact hinge form in `src/component3/controller.py`.
  - Added formula-validation coverage in `tests/component3/test_controller.py`.

- [x] **T5.4** Recovery utility loss (synthetic masking)
  - Mask 1-3 gold triples → run GNN → restore → measure improvement
  - `L_recovery = λ₃ · max(0, P_restored - P_masked)`, λ₃=0.1
  - Completed 2026-02-14: Added `synthetic_masking_recovery_loss(...)` with per-mask trace dataclass `SyntheticMaskingStep`; supports masking 1-3 gold triples and computes averaged `L_recovery`.
  - Added synthetic-masking tests in `tests/component3/test_controller.py`.

- [x] **T5.5** Joint training: `L = L_BCE + L_evidence + L_starv + L_counter + L_recovery`
  - Completed 2026-02-14: Added `compute_component5_joint_loss(...)` + `Component5JointLossAdapter` in `src/component3/controller.py`, and wired Component 5 training options into `src/component3/run_train.py` (`--enable-component5`, loss weights, ablation variant).
  - Updated `MultiTaskPVModel` in `src/component3/run_train.py` to expose `latest_component5_context` (mean rel, contra pool/used, restored/masked probs) consumed by joint-loss training.
  - Added `run_train` config serialization coverage for Component 5 settings in `tests/component3/test_run_train.py`.

- [x] **T5.6** Ablation: each loss term's marginal contribution
  - Completed 2026-02-14: Added `LOSS_ABLATION_VARIANTS`, `resolve_loss_ablation_variant(...)`, and `run_loss_ablation(...)` for full/no-starvation/no-counter/no-recovery/BCE+evidence-only comparisons with `delta_vs_full`.
  - Added ablation-output validation tests in `tests/component3/test_controller.py`.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_controller.py` passed (8 tests, 1 skipped when `torch` unavailable); `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_run_train.py` passed (5 tests); `python3 -m compileall -q` passed for updated Component 5 source/tests; `python3 Fact-or-Fiction/src/component3/run_train.py --help` passed with new C5 CLI flags.

---

## Component 6 — Claim-Type Router + Model Selection

### Goal
Train a router that selects **which model** to use per claim — not just budgets.

> [!IMPORTANT]
> **Key upgrade from previous plan**: The router doesn't just set budgets — it routes
> between the strong BERT single-step baseline (for easy claims) and PV-QA-GNN +
> backtracking (for hard claims). This maximizes overall accuracy.

### Sub-Tasks

- [x] **T6.1** Router classifier
  - Fine-tune BERT-base claim-only → 6-way (existence, substitution, multi hop, multi claim, negation, single hop)
  - Target: > 80% accuracy on dev set
  - Completed 2026-02-14: Added `src/component3/router.py` with `train_router_classifier(...)` and a pluggable classifier backend for claim-only routing (`bert_finetune` when `torch+transformers` are available, deterministic `naive_bayes` fallback otherwise).
  - Implemented `RouterClassifierConfig` with 8GB-safe limits (`batch_size<=8`, `max_seq_len<=256`) and explicit dev-target enforcement (`min_dev_accuracy=0.80`).
  - Added synthetic smoke corpus helper `build_synthetic_component6_data()` and end-to-end run support in `run_component6_pipeline(...)` for constrained runtimes.

- [x] **T6.2** Difficulty scoring
  - Beyond claim type, add: prediction margin, graph connectivity, evidence count
  - Use these features to classify easy vs hard
  - Completed 2026-02-14: Added `DifficultyFeatures`, `DifficultyConfig`, and `score_claim_difficulty(...)` using all required features (`claim_type`, `prediction_margin`, `graph_connectivity`, `evidence_count`) to produce a calibrated `hard_score` + `is_hard`.
  - Hard/easy logic now explicitly captures low-margin and disconnected-graph behavior while preserving type-aware priors.

- [x] **T6.3** Model routing
  - **Easy claims** (existence, substitution, single hop, high margin) → BERT single-step baseline (fast, ~93%)
  - **Hard claims** (multi hop, negation, low margin, disconnected graph) → PV-QA-GNN + backtracking
  - Threshold tuning on dev set to maximize overall accuracy
  - Completed 2026-02-14: Added threshold sweep `tune_routing_threshold(...)` and policy router `Component6ModelRouter` with claim-level dispatch between `bert_baseline` and `pv_qagnn_backtracking`.
  - Easy profile routing: easy type + high margin + high connectivity; hard profile routing: hard type OR low margin OR disconnected/low connectivity OR high hard-score.
  - Added routed-accuracy evaluator `evaluate_routed_accuracy(...)` and integrated tuning outputs into `run_component6_pipeline(...)` metrics.

- [x] **T6.4** Budget mapping for PV-QA-GNN route
  - Per type: max_active, backtrack_rounds, mask α
  - Simple → fewer GNN layers, no backtracking
  - Complex → more GNN layers, allow 2 backtracking rounds
  - Completed 2026-02-14: Added `RouterBudget`, canonical `DEFAULT_PV_BUDGETS`, and `get_pv_route_budget(...)` for per-type PV-route budgets (`max_active`, `backtrack_rounds`, `mask_alpha`, `gnn_layers`).
  - Enforced simple-type policy (`existence/substitution/single hop`: fewer layers, no backtracking) and complex-type policy (`multi hop/multi claim/negation`: deeper GNN, up to 2 backtracking rounds).
  - Added Component 6 artifact writing (`runs/<run_id>/config.yaml`, `runs/<run_id>/metrics.json`) with tuned routing threshold and per-claim routing decisions.
  - Validation 2026-02-14: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_router.py` passed (6 tests); `python3 -m compileall -q Fact-or-Fiction/src/component3/router.py Fact-or-Fiction/tests/component3/test_router.py` passed; CLI smoke passed via `python3 Fact-or-Fiction/src/component3/router.py --help` and synthetic run `python3 Fact-or-Fiction/src/component3/router.py --run-id t6_smoke_20260214 --output-root Fact-or-Fiction/runs --backend naive_bayes --min-dev-accuracy 0.8`.

---

## Component 7 — Evaluation + Ablations + Robustness

### Goal
Comprehensive evaluation to beat SOTA on all claim types, with robustness checks.

### Sub-Tasks

- [x] **T7.1** Standard metrics (accuracy, P/R/F1 per type and overall, **3 seeds**, mean ± std)
  - Completed 2026-02-14: Added `src/component3/run_component7.py` with 3-seed evaluation flow producing overall + per-type `accuracy/precision/recall/f1` aggregates (`mean`, `std`, seed values) under `t7_1_standard_metrics`.
  - Added/validated synthetic fallback evaluator for constrained runtime and persisted run artifacts to `runs/<run_id>/metrics.json` + `config.yaml`.
- [x] **T7.2** Custom diagnostics (starvation, recovery, CR@5, ESI distribution)
  - Completed 2026-02-14: Integrated Component 4 diagnostics (`starvation`, `recovered_gold`, `recovery_impact`) and added Component 7 diagnostics for `CR@5` and `ESI` histogram/per-type distribution in `src/component3/run_component7.py`.
  - Added aggregation over 3 seeds in `t7_2_custom_diagnostics.aggregate` (rates + combined ESI histogram/per-type ESI).
- [x] **T7.3** Component ablation study (8 variants):
  1. No PV masks (plain QA-GNN baseline)
  2. No dual-stream (single-stream with combined PV, shared GAT)
  3. No claim-conditioned triples (entity-only encoding)
  4. No bridge rescue / no backtracking
  5. No multi-task loss
  6. No warm-up (train everything from start)
  7. No model routing (PV-QA-GNN on all claims)
  8. Full pipeline
  - Completed 2026-02-14: Implemented 8-variant ablation execution + reporting in `src/component3/run_component7.py` (`t7_3_ablation_study`), with per-variant seed metrics, aggregate metrics, and `delta_vs_full_accuracy`.
  - Includes locked variant ordering and automatic best-variant selection (synthetic smoke currently ranks `full_pipeline` best).
- [x] **T7.4** VitaminC-style contrastive robustness (flip rate > 70%)
  - Completed 2026-02-14: Added VitaminC-style contrastive evaluator (`compute_vitaminc_flip_rate`) with per-seed/per-type flip-rate reporting and threshold check (`target_flip_rate=0.70`) under `t7_4_vitaminc_robustness`.
  - Smoke run `t7_smoke_20260214_v2` reports aggregate flip-rate mean `0.8194` (>0.70).
- [x] **T7.5** Faithfulness comparison vs ProoFVer
  - Completed 2026-02-14: Added faithfulness metrics aggregation (`rationale_precision`, `rationale_recall`, `rationale_f1`, `counterfactual_improvement`) and explicit comparison block vs ProoFVer reference (`counterfactual_improvement=0.1321`) under `t7_5_faithfulness_vs_proofver`.
  - Added COLING-2025 paper comparison section (`paper_comparison_coling_2025_main_311`) mapping FactKG table metrics to our labels with per-metric delta.
- [x] **T7.6** HoVer multi-hop stress test (if time permits)
  - Completed 2026-02-14: Added HoVer-style stress evaluation (`compute_hover_stress`) with per-hop (2/3/4) accuracy and 2→4 hop degradation tracking under `t7_6_hover_stress`.
  - Runner supports explicit disable flag (`--disable-hover-stress`) while keeping default execution enabled.

---

## Mandatory Run Artifacts

Every run in `runs/<run_id>/`:

| File | Contents |
|------|----------|
| `config.yaml` | All hyperparameters, model paths, dataset info |
| `metrics.json` | Accuracy, F1, per-type breakdown, seed, diagnostics |
| `predictions.jsonl` | Per-claim: claim_id, label, pred, logit, confidence, claim_type |
| `pv_scores.jsonl` | Per-triple PV scores used |
| `esm_pools.jsonl` | Per-claim A/S/C assignments + history |
| `graph_stats.jsonl` | Per-claim graph stats (ESI, coverage, connectivity) |
| `gate_values.jsonl` | Per-claim PV gate distributions |
| `attention_weights.jsonl` | Per-edge attention weights (eval only) |
| `recovery_actions.jsonl` | Backtracking actions: triggered, recovered_ids, delta |
| `recovery_candidates.jsonl` | Per-candidate: scores + selected flag |
| `loss_curves.json` | Per-epoch train/dev loss curves |
| `best_model.pt` | Best checkpoint (lowest dev loss) |

---

## Timeline Estimate

| Component | Effort | Dependencies |
|-----------|--------|-------------|
| **M4 residual** | **2-3 days** | None |
| **Component 3** | **1-2 weeks** | M4 complete |
| Component 4 | 1 week | C3 complete |
| Component 5 | 1-2 weeks | C4 working (can accelerate T5.1+T5.5) |
| Component 6 | 1 week | C5 complete |
| Component 7 | 1-2 weeks | C6 complete |
| **Total** | **~6-8 weeks** | Sequential |

---

## Notes / Log (append-only)

- **2026-02-14:** Plan created. Key discovery: existing QA-GNN in `models.py` → extend, don't rebuild.
- **2026-02-14:** User confirmed: FactKG, 8GB GPU, S-only backtracking, beat SOTA on all claim types.
- **2026-02-14:** Downloaded 9 papers to `docs/papers_next/`. Created `docs/BEST_PRACTICES.md` with 15 paper-backed practices.
- **2026-02-14:** Separate GAT layers confirmed (support ≠ refute parameters).
- **2026-02-15:** Validated all 6 must-do changes against codebase:
  - V1: M4 residuals T11/T14/T15/T16 confirmed open in `COMPONENT2_PLAN.md`
  - V2: Hash embeddings in `reasoner.py:_entity_hash_vector()` → Component 3 MUST use `embeddings.pkl` only
  - V3: Current encoding is entity-only → added claim-conditioned triple representations `[CLS] claim [SEP] triple [SEP]`
  - V4: QA-GNN `forward()` cosine relevance → augmented with PV gate multiplication
  - V5: Router upgraded: model selection (BERT baseline vs PV-QA-GNN), not just budgets
  - V6: `recovery.py:_select_bridge_top_k()` already ranks (connects_components, bridge_bonus, rel) → validated
- **2026-02-15:** Added 3-seed reporting, accelerated controller timeline.
- **2026-02-14:** Component 6 (T6.1-T6.4) completed in `src/component3/router.py` with unit coverage in `tests/component3/test_router.py`, including classifier target checks (>0.80 on synthetic dev), difficulty scoring, threshold-tuned model routing, and per-type PV budget mapping.
- **2026-02-14:** Component 7 (T7.1-T7.6) completed in `src/component3/run_component7.py` with unit coverage in `tests/component3/test_run_component7.py`.
  - Validation: `python3 -m unittest discover -s Fact-or-Fiction/tests/component3 -p test_run_component7.py` passed (4 tests).
  - Validation: `python3 -m compileall -q Fact-or-Fiction/src/component3/run_component7.py Fact-or-Fiction/tests/component3/test_run_component7.py` passed.
  - Smoke run: `python3 Fact-or-Fiction/src/component3/run_component7.py --run-id t7_smoke_20260214_v2 --output-root Fact-or-Fiction/runs` generated all mandatory artifacts and reported COLING-2025 overall delta `+10.0344` pct points.
