# Component 2 Implementation Plan (Codex-maintained)

> **How to use this file**
> - Keep this as the single source of truth.
> - When a task is done, tick the checkbox and add a 1–2 line completion note under it.
> - Codex should update this file after every implementation step.

---

## Status
- Current phase: ✅ **Spec locking (COMPLETE)** ✅ Implementation (M1 complete) ☐ Evaluation ☐ Writing
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
  ✓ **Completed 2026-02-10:** Added `src/component2/anchor_selector.py` with seed-first anchor selection (`Entity_set`) and two fallbacks (claim-text match, degree-based). `tests/component2/test_anchor_selector.py` validates ≤K and graph-membership behavior.

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
- [ ] **T7 PPR bridge bonus implemented (A∪S∪C)**  
  Acceptance: bridge scores computed with `ε=0.001`; neutral cap works with `γ=2`; unit test with toy graph.

- [ ] **T8 Recovery policy implemented (S→A, one-step rerun)**  
  Acceptance: triggers when `esi_geom<0.3 AND DeltaConn_bridge@k>0`; recovers top-3 S edges by BridgeBonus'; on toy "disconnected anchors" graph, recovery increases connectivity and changes output.  
  **Note:** Legacy trigger `rpi_at_k>0.15` is optional/reference only; use bridge-aware trigger from T8b.

- [ ] **T8b Add policy-aware recovery predictor (REQUIRED for correct triggering)**  
  Acceptance: compute DeltaConn_bridge@k using bridge-based Select_k on S; log RPI_rel@k and RPI_bridge@k; recovery trigger uses `(ESI<0.3 AND DeltaConn_bridge@k>0)`, not RPI_rel threshold; toy graph where rel-based top-k fails but bridge-based top-k reconnects must pass.

### M3 — Reliability + interpretability
- [ ] **T9 Selective prediction (abstain score) + risk–coverage evaluation**  
  Acceptance: abstain score computed with `α=0.5`; can plot risk–coverage; AURC computed.

- [ ] **T10 Salience extraction + top-N rationale edges**  
  Acceptance: outputs top-10 edges using attention×PV; leave-one-out removal test on small subset; faithfulness verified.

### M4 — Ablations + experiments
- [ ] **T11 Ablation: no PV mask**  
- [ ] **T12 Ablation: no bridge rescue**  
- [ ] **T13 Ablation: no recovery loop**  
- [ ] **T14 Stress tests: drop bridge triples / add distractors**
- [x] **T15 Ablation: learnable PV mask coefficients (`α_s,β_s,α_r,β_r`)**
  Acceptance: mask formula `σ(α·p + β)` uses configurable alpha/beta per stream; grid-search tuning module implemented; CLI flags for manual override and auto-tuning; backward compatible (defaults 4.0/−2.0 reproduce original behavior).
  ✓ **Completed 2026-02-13:** Added `mask_tuning.py` (two-stage/independent/joint grid search), alpha/beta aliases in `config.py`, CLI args in all runners (`--mask-sup-alpha/beta`, `--mask-ref-alpha/beta`), ablation runner integration (`--tune-mask-coeffs`). Outputs: `best_mask_params.json`, `mask_tuning_report.json`. 13 files changed, +484/−64. All tests pass (test_reasoner, test_mask_tuning, test_ablation_modes_smoke, test_run_m1/m2/m3).

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
- **2026-02-12 IST:** ESI bug fix completed — ESI now computed on A+C (not A-only) for symmetric treatment of SUPPORTED/REFUTED predictions.
- **2026-02-13 01:30 IST:** T15 mask coefficient tuning implemented. Mask formula upgraded from hardcoded `σ(4p−2)` to tunable `σ(α·p+β)` with per-stream params. Added `mask_tuning.py` (grid search with two-stage/independent/joint strategies), alpha/beta config aliases, CLI injection in all runners, ablation runner integration. Default params (α=4, β=−2) preserve backward compatibility. 13 files changed, +484/−64.
