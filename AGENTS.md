# AGENTS.md — Post-Component 2 Continuation Guide

## Purpose
This file gives any new Codex/Antigravity chat the exact context and continuation rules for Component 3+ work.

---

## Current Project State (as of 2026-02-15)

### ✅ Component 1 — Evidence Scoring & Partitioning (COMPLETE)
- **Source**: `src/component1/` (7 files)
- PV scoring with DeBERTa, ESM A/S/C partitioning, ESI/CR@k/RPI@k metrics
- Status: Production-ready, 5 tests passing

### ✅ Component 2 — Graph Reasoning (M0-M3 COMPLETE, M4 PARTIAL)
- **Source**: `src/component2/` (19 files)
- Dual-stream masked GNN (numpy), PPR bridge rescue, recovery, selective prediction, salience
- **M4 open tasks**: T11 (no-PV-mask ablation), T14 (stress tests), T15 (learnable PV masks), T16 (tau_esi sweep)
- **Hash embeddings**: `reasoner.py:_entity_hash_vector()` uses blake2b → random vectors (MUST NOT use in C3)

### ✅ Original QA-GNN Pipeline (ALREADY EXISTS)
- **Source**: Root-level files — original Fact-or-Fiction implementation

| File | What it does |
|------|-------------|
| `models.py` | `QAGNN`: BERT + GATConv + cosine relevance + global_mean_pool + classifier |
| `train.py` | `run_epoch_qa_gnn()`: BCEWithLogitsLoss, training loop, early stopping=3 |
| `evaluate.py` | Per-type metrics (existence, substitution, multi hop, multi claim, negation, single hop) |
| `datasets.py` | `convert_to_pyg_format()`: precomputed BERT embeddings → PyG |
| `run_stuff.py` | CLI: GNN h=256, L=2, dropout=0.3, lr=1e-5, batch=32, AdamW, linear warmup |

### ☐ Component 3+ — Not Started
- `src/component3/` exists but is empty

---

## Architecture: PV-Enhanced QA-GNN

```
                    ┌──────────────────────────────────────────┐
                    │          PV-QA-GNN (Component 3)         │
                    │                                          │
  Claim text ──────►│  BERT encoder ──► [CLS] claim embedding  │
                    │         │                                │
  C1 evidence ─────►│  Build PyG graph (A+C triples)           │
  (A/S/C pools)     │  Node features = BERT entity embeddings  │
                    │  Edge attrs = claim-conditioned +PV      │
                    │         │                                │
                    │    ┌────┴────┐                           │
                    │    ▼         ▼                           │
                    │  Support   Refute                        │
                    │  GATConv   GATConv     ← SEPARATE params │
                    │  (cosine   (cosine                       │
                    │   × σ(α·   × σ(α·                       │
                    │   p_ent+β)) p_con+β))                    │
                    │    │         │                           │
                    │   pool      pool                        │
                    │    └────┬────┘                           │
                    │         ▼                                │
                    │  concat(sup_pool, ref_pool, claim_embed) │
                    │         ▼                                │
                    │     Classifier → SUPPORTED / REFUTED     │
                    │                                          │
                    │  If low confidence + disconnected graph: │
                    │    → S-only bridge rescue (ranked)       │
                    │    → Re-run GNN (max 2 rounds, k=3)     │
                    └──────────────────────────────────────────┘
```

---

## Locked Design Decisions (Validated 2026-02-15)

| Decision | Value | Source | Codebase Validation |
|----------|-------|--------|-------------------|
| Dual-stream GAT | **Separate** parameters | Novel contribution | N/A — new code |
| Edge masking | **Soft** `σ(α·p+β)` | CO-GAT (+0.79%) | N/A — new code |
| Joint attention | **cosine × PV gate** | QA-GNN + CO-GAT | `models.py:forward()` already has cosine |
| Node embeddings | **Precomputed BERT** only | `datasets.py` | `convert_to_pyg_format()` supports `embedding_dict` |
| No hash vectors | **Enforced** (error if missing) | User decision | `reasoner.py:_entity_hash_vector()` = C2 only |
| Claim-conditioned edges | **[CLS] claim [SEP] triple [SEP]** | GEAR, CO-GAT | `convert_to_pyg_format()` does entity-only — needs upgrade |
| Multi-task loss | BCE + evidence quality | CO-GAT (+0.18%) | N/A — new code |
| Training | 2-stage warm-up | CO-GAT | `run_stuff.py` has AdamW+warmup |
| Batch size | 8 (8GB GPU) | CO-GAT | Existing is 32 → reduce |
| Max seq length | 256 | CO-GAT | Existing is 512 → reduce |
| Backtracking | **S-only**, ranked | User + RPS | `recovery.py:_select_bridge_top_k()` already ranks correctly |
| Recovery budget | rounds≤2, k≤3 | RPS sensitivity | `config.py:recovery_top_k=3` |
| Router | **Model selection** (BERT vs PV-QA-GNN) | User feedback | `evaluate.py` has per-type breakdown |
| Reporting | **3 seeds** + mean ± std | User feedback | N/A — new code |
| PV mask init | α=4.0, β=−2.0 (trainable) | C2 defaults | `reasoner.py` uses fixed 4/−2 |

---

## Hard Constraints

1. **Finish M4 residuals first** (T11, T14, T15, T16)
2. New code under `src/component3/`, tests under `tests/component3/`
3. Do NOT modify Component 1 source code. Component 2 source may ONLY be modified for M4 residual tasks (T11, T14, T15, T16)
4. Do NOT modify original `models.py` / `train.py` etc. — **extend** them
5. **No hash embeddings** in Component 3 — use precomputed BERT only
6. **Separate GAT layers** for support and refute streams
7. **8GB GPU**: batch_size ≤ 8, max_seq_len=256, gradient accumulation if needed
8. After each task, tick checkbox in `PLAN.md` with completion note

---

## Canonical Docs to Read First

1. `AGENTS.md` ← you are here
2. `PLAN.md` — Components 3-7 task-level roadmap
3. `docs/BEST_PRACTICES.md` — **15 paper-backed design decisions** (MUST read)
4. `docs/TRAINING_SPEC.md` — **Trainable params, metrics, I/O spec** (MUST read)
5. `PAPERS_NEXT.md` — Paper references with links
6. `models.py` — **Existing QA-GNN implementation** (MUST read)
7. `train.py` — Existing training loop (reuse for C3)
8. `datasets.py` — Existing graph dataset (extend for C3)
9. `evaluate.py` — Existing per-type evaluation (reuse for all experiments)
10. `run_stuff.py` — Existing CLI and hyperparameters
11. `docs/component2/COMPONENT2_PLAN.md` — M4 residuals
12. `docs/component2/COMPONENT2_DECISIONS.md` — Locked C2 decisions

---

## Copy-Paste Prompt For New Chat

```text
Read (in this order):
1. AGENTS.md — current state + architecture + locked decisions
2. docs/BEST_PRACTICES.md — 15 paper-backed design decisions (MUST read)
3. models.py — EXISTING QA-GNN (QAGNN class, forward, training)
4. train.py — existing training loop (run_epoch_qa_gnn)
5. datasets.py — existing FactKGDatasetGraph + convert_to_pyg_format
6. evaluate.py — per-reasoning-type evaluation
7. run_stuff.py — CLI and hyperparameters
8. PLAN.md — Components 3-7 roadmap

Your job: Continue from PLAN.md. Check which tasks are unchecked.

Architecture: PV-QA-GNN = existing QAGNN + dual-stream (SEPARATE GAT layers) + PV edge masks + claim-conditioned triples + joint attention prior (cosine × σ(α·p+β))

Key changes from user review (2026-02-15):
- M4 residuals MUST be done first
- No hash embeddings — error if embeddings.pkl missing
- Claim-conditioned: [CLS] claim [SEP] triple [SEP] as edge_attr
- Joint attention = cosine relevance × PV gate (not just PV alone)
- Router selects MODEL (BERT for easy, PV-QA-GNN for hard), not just budgets
- S-only recovery ranked: bridge_connects_components (bool) → bridge_bonus (PPR) → rel (tiebreaker). C3 adds: salience×PV after GNN pass
- 3 seeds for all final results

Hard constraints: 8GB GPU, batch≤8, max_len=256, S-only, separate GAT, no hash
```
