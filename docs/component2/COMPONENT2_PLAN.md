# Component 2 Implementation Plan (Codex-maintained)

> **How to use this file**
> - Keep this as the single source of truth.
> - When a task is done, tick the checkbox and add a 1–2 line completion note under it.
> - Codex should update this file after every implementation step.

---

## Status
- Current phase: ✅ **Spec locking (COMPLETE)** ✅ Implementation (M3 complete + stability fixes) ☐ Evaluation ☐ Writing
- Target dataset: ☑ **FactKG (KG only)** ☐ HoVer (text) ☐ Both
- Reasoner backbone: ☐ GAT ☐ R-GCN ☑ Other: **Hybrid (GAT + relation-conditioned attention)**
- Bridge rescue: ☑ **PPR diffusion (default)** ☐ Path enumeration (not recommended)
- Dev strategy: ☑ **Phased (100 → 1000 → full)**

---

## Milestones

### M0 — Repo orientation (must be done first)
- [x] **T1 Locate claim entity seeds used by retrieval**  
  Acceptance: identify the exact variable / field / file that holds claim→entity seeds; write it down in `docs/component2/COMPONENT2_DECISIONS.md`.  
  ✓ **Completed 2026-02-10:** Source is `Entity_set` field in `data/factkg/factkg_{train|dev|test}.pickle` (Option A: retrieval seeds → anchors).

- [x] **T2 Confirm evidence object shape from Component 1 logs**  
  Acceptance: one example JSON record showing `A/S/C`, PV scores, and triple IDs.  
  ✓ **Completed 2026-02-10:** Schema documented in DECISIONS.md. Per-triple: `pool`, `p_ent/p_con/p_neu`, `raw_triple` [subj, rel, obj]. Per-claim: ESI, RPI, CR@k metrics.

### M1 — Minimal end-to-end (no bridge rescue)
- [x] **T3 AnchorSelector implemented**  
  Acceptance: returns ≤K anchors; anchors exist in graph; unit test passes.
  ✓ **Completed 2026-02-10:** Added `src/component2/anchor_selector.py` with seed-first anchor selection (`Entity_set`) and two fallbacks (claim-text match, degree-based). `tests/component2/test_anchor_selector.py` validates ≤K and graph-membership behavior. Post-M3 stability update guarantees anchor-pair feasibility by augmenting single-seed anchors to at least 2 (when graph has ≥2 entities) and filtering blank entities.

- [x] **T4 GraphBuilder implemented (A∪C)**  
  Acceptance: nodes=entities, edges=triples; edge features include relation id + PV; unit test passes.
  ✓ **Completed 2026-02-10:** Added `src/component2/graph_builder.py` + typed graph records in `src/component2/types.py`; builder emits entity nodes, A∪C edges, relation IDs, pool IDs, and PV-derived `rel/pol`. Covered by `tests/component2/test_graph_builder.py`.

- [x] **T5 Hybrid masked dual-stream GNN implemented (Option C)**  
  Acceptance: relation-conditioned attention is active; fixed masks are used (`m_sup=sigmoid(4*p_ent-2)`, `m_ref=sigmoid(4*p_con-2)`); forward pass returns `probs` for a batch; attention salience can be exported; CPU run on 5 claims works.
  ✓ **Completed 2026-02-10:** Added `src/component2/reasoner.py` (`HybridMaskedDualStreamReasoner`) with fixed mask formulas, relation-conditioned attention, batched `forward_batch`, and edge salience export. CPU smoke run completed on 5 claims via `python3 -m component2.run_m1`.

- [x] **T6 Pooling + classifier head implemented**  
  Acceptance: hybrid pooling per stream (mean+max+attention); concat streams; produces 2 logits; loss computes.
  ✓ **Completed 2026-02-10:** Implemented hybrid pooling and binary classifier head in `src/component2/reasoner.py`; added CE loss path and verified via `tests/component2/test_reasoner.py`.

### M2 — Bridge rescue + recovery
- [x] **T7 PPR bridge bonus implemented (A∪S∪C)**  
  Acceptance: bridge scores computed with `ε=0.001`; neutral cap works with `γ=2`; unit test with toy graph.
  ✓ **Completed 2026-02-10:** Added `src/component2/bridge_rescue.py` (`BridgeRescuePPR`) with anchor-conditioned weighted PPR on `A∪S∪C`, `BridgeBonus=e_ppr_u*(ε+rel)*e_ppr_v`, and neutral-capped `BridgeBonus'`. Covered by `tests/component2/test_bridge_rescue.py`. Post-M3 stability update relaxed solver defaults to `ppr_max_iters=100`, `ppr_tolerance=1e-6` and added regression coverage for default convergence.

- [x] **T8 Recovery policy implemented (S→A, one-step rerun)**  
  Acceptance: triggers when `esi_geom<0.3 AND DeltaConn_bridge@k>0`; recovers top-3 S edges by BridgeBonus'; on toy "disconnected anchors" graph, recovery increases connectivity and changes output.  
  **Note:** Legacy trigger `rpi_at_k>0.15` is optional/reference only; use bridge-aware trigger from T8b.
  ✓ **Completed 2026-02-10:** Added `src/component2/recovery.py` (`BridgeRecoveryPolicy`) with one-step `S→A` recovery and trigger `esi_geom<0.3 AND DeltaConn_bridge@k>0`. Integrated rerun flow in `src/component2/run_m2.py`, validated connectivity gain + probability change in `tests/component2/test_run_m2.py`.

- [x] **T8b Add policy-aware recovery predictor (REQUIRED for correct triggering)**  
  Acceptance: compute DeltaConn_bridge@k using bridge-based Select_k on S; log RPI_rel@k and RPI_bridge@k; recovery trigger uses `(ESI<0.3 AND DeltaConn_bridge@k>0)`, not RPI_rel threshold; toy graph where rel-based top-k fails but bridge-based top-k reconnects must pass.
  ✓ **Completed 2026-02-10:** Implemented policy-aware predictor in `BridgeRecoveryPolicy.evaluate()` with `RPI_rel@k`, `RPI_bridge@k`, and `DeltaConn_bridge@k` diagnostics. Added toy test where rel top-k fails but bridge top-k reconnects in `tests/component2/test_recovery.py`. Post-M3 stability update fixed missing-anchor connectivity semantics (`missing anchor => disconnected`) and bridge ranking priority for edges that can attach missing anchor components.

### M3 — Reliability + interpretability
- [x] **T9 Selective prediction (abstain score) + risk–coverage evaluation**  
  Acceptance: abstain score computed with `α=0.5`; can plot risk–coverage; AURC computed.
  ✓ **Completed 2026-02-10:** Added `src/component2/selective.py` (`abstain_score`, risk–coverage curve builder, AURC) and integrated outputs into `src/component2/run_m3.py` with `risk_coverage.json` and summary AURC logging.

- [x] **T10 Salience extraction + top-N rationale edges**  
  Acceptance: outputs top-10 edges using attention×PV; leave-one-out removal test on small subset; faithfulness verified.
  ✓ **Completed 2026-02-10:** Added `src/component2/salience.py` for attention×PV top-N rationale extraction and leave-one-out faithfulness checks; M3 smoke outputs include per-claim `top_rationale_edges` and `faithfulness_loo`.

### M3 Patch — Design Alignment Fixes (A-D)
- [x] **A) Wire anchors from FactKG pickles by default (Option A)**  
  Acceptance: inject `Entity_set` from `data/factkg/factkg_{train|dev|test}.pickle` when incoming records are missing entity seeds; log anchor source per claim and missing-mapping counts.
  ✓ **Completed 2026-02-10:** Added `src/component2/anchor_adapter_factkg.py` and wired it into `run_m1.py`, `run_m2.py`, and `run_m3.py`. Runs now log `anchor_source`, `anchors_hash`, and aggregate `claims_missing_pickle_mapping`.

- [x] **B) Add fixed abstention operating point (`tau_abstain`) in addition to sweep**  
  Acceptance: when `tau_abstain` is set, emit `coverage`, `risk`, `accuracy_on_answered`, and `abstain_rate`; keep optional risk-coverage/AURC sweep.
  ✓ **Completed 2026-02-10:** Extended `src/component2/selective.py` + `run_m3.py` with `tau_abstain` and `compute_risk_coverage_sweep` controls; outputs now include fixed-point metrics and preserve sweep by default.

- [x] **C) Clarify relation-embedding fidelity status vs paper motivation**  
  Acceptance: clearly document current placeholder status and future trainable PyTorch milestone.
  ✓ **Completed 2026-02-10:** Updated DECISIONS/CONFIG/runtime logs to mark current relation embeddings as non-trainable numpy features and added explicit PyTorch migration TODO in docs.

- [x] **D) Add numpy performance guardrails + profiling note**  
  Acceptance: warn on oversized edge sets in numpy mode and document next optimization steps.
  ✓ **Completed 2026-02-10:** Added `numpy_edge_warn_threshold` guardrail to all runners and logged warning counts/claim IDs; documented complexity and optimization roadmap in patch dev notes.

### M4 — Ablations + experiments
- [ ] **T11 Ablation: no PV mask**  
- [x] **T12 Ablation: no bridge rescue**  
  ✓ **Completed 2026-02-11:** Added switch-driven ablation runner (`src/component2/ablation_runner.py`, `scripts/run_ablation_component2.py`) with mode `A1` (no bridge) and paper-grade outputs (`predictions_A1.jsonl`, `metrics_A1.json`, selective S1/S2 reports, plots).
- [x] **T13 Ablation: no recovery loop**  
  ✓ **Completed 2026-02-11:** Added mode `A2` (bridge scoring, no recovery) and mode `A3` (full recovery) in the same runner for direct no-recovery vs recovery comparison, including bridge correlations and recovery-trigger diagnostics in per-ablation metrics + summary table.
- [ ] **T14 Stress tests: drop bridge triples / add distractors**
- [ ] **T15 Ablation: learnable PV mask coefficients (`a_s,b_s,a_r,b_r`)**
- [ ] **T16 Recovery-threshold sensitivity (`tau_esi` sweep)**  
  Acceptance: evaluate `tau_esi ∈ {0.2, 0.3, 0.4}` on validation subset; report recovery trigger rate and downstream accuracy/risk tradeoff.

---

## Notes / Log (append-only)

- **2026-02-10 12:05 IST:** Spec-locking interview started
- **2026-02-10 12:05 IST:** Q1 answered — Anchors will use `Entity_set` from FactKG pickles (Option A from spec §4.2)
- **2026-02-10 12:10 IST:** Q2 answered — Component 1 output schema confirmed (per-triple pool/PV, per-claim ESI/RPI/CR@k)
- **2026-02-10 12:15 IST:** Q3 answered — Reasoner backbone set to Option C (Hybrid GAT + relation-conditioned attention)
- **2026-02-10 12:14 IST:** Q4 answered — Model sizing set to Option B (hidden=256, heads=4, rel_emb=64)
- **2026-02-10 12:20 IST:** Q5 answered — Bridge rescue defaults locked: `ε=0.001`, `γ=2`
- **2026-02-10 12:25 IST:** Q6 answered — PV mask set to Option A fixed coefficients (`m_sup=sigmoid(4*p_ent-2)`, `m_ref=sigmoid(4*p_con-2)`); learnable mask moved to ablation.
- **2026-02-10 12:21 IST:** Q7 answered — Pooling set to Option C (hybrid: mean+max+attention per stream, then concat)
- **2026-02-10 12:23 IST:** Q8 answered — Recovery triggers set to spec defaults (`τ_esi=0.3`, `τ_rpi=0.15`, `k=3`)
- **2026-02-10 12:26 IST:** Q9 answered — Abstention set to spec default (`α=0.5`, threshold-free risk–coverage evaluation)
- **2026-02-10 12:28 IST:** Q10 answered — Salience method set to Hybrid (attention×PV default, leave-one-out validation subset)
- **2026-02-10 12:32 IST:** Q11 answered — Dev dataset strategy set to phased (100 → 1000 → full)
- **2026-02-10 12:32 IST:** ✅ **SPEC LOCKING COMPLETE** — All 11 questions answered. Ready for implementation.
- **2026-02-10 12:45 IST:** M1 code scaffold created under `src/component2/` and tests under `tests/component2/`.
- **2026-02-10 12:51 IST:** M1 acceptance checks passed (`PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`) and smoke run generated logs under `logs/component2/m1/20260210_072118/`.
- **2026-02-10 13:19 IST:** M2 modules added (`bridge_rescue.py`, `recovery.py`, `run_m2.py`) with tests (`test_bridge_rescue.py`, `test_recovery.py`, `test_run_m2.py`).
- **2026-02-10 13:20 IST:** M2 acceptance checks passed (`PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`) and M2 smoke run generated logs under `logs/component2/m2/20260210_075247/`.
- **2026-02-10 13:27 IST:** Post-review hardening: added PPR diagnostics (convergence/disconnected-anchor indicators) to M2 outputs and tests; latest smoke logs under `logs/component2/m2/20260210_075756/`.
- **2026-02-10 13:34 IST:** M3 modules added (`selective.py`, `salience.py`, `run_m3.py`) with tests (`test_selective.py`, `test_salience.py`, `test_run_m3.py`); smoke outputs under `logs/component2/m3/20260210_080452/`.
- **2026-02-10 (post-M3 stability patch):** Fixed anchor-selection early-return issue, blank-anchor leakage, and missing-anchor recovery connectivity behavior; updated PPR defaults (`100`, `1e-6`); component2 suite now passes with 46 tests.
- **2026-02-10 (revalidation):** M3 100-claim rerun at `logs/component2/m3/20260210_094916_968777/` shows `anchor_len_counts={2:100}` and `num_ppr_not_converged=0` (was `100` in `20260210_093453_086968`).
- **2026-02-10 (A-D fixes patch):** Added FactKG anchor adapter + selective `tau_abstain` operating point + relation embedding status notes + numpy perf guardrails. Component2/full test suites pass with 52 tests.
- **2026-02-10 (A-D validation runs):** New smoke snapshots written at `logs/component2/m1/20260210_104934_026305/`, `logs/component2/m2/20260210_104939_838090/`, `logs/component2/m3/20260210_104944_757895/`, and pickle-injection verification run at `logs/component2/m3/20260210_105002_449038/`.
- **2026-02-11 (M4 ablation runner):** Added `A0` baseline + `A1/A2/A3` switch-driven ablation runner with deterministic subset logging, selective S1/S2 reporting, bridge/recovery correlation analysis, plots, and summary table generation under `logs/ablations/component2/<run_id>/`.
