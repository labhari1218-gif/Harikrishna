# TRAINING_SPEC.md — PV-QA-GNN: Parameters, Metrics, I/O, and Diagnostics

> **Purpose**: Exact specification of what is trained, what is measured, what is saved, and what is tracked.
> Train on **full FactKG train set**, evaluate on **full test set**, 3 seeds.

---

## 1. Trainable Parameters Inventory

### 1.1 Inherited from Existing QA-GNN (`models.py`)

| Module | Shape | Params | Trainable When |
|--------|-------|--------|---------------|
| **BERT-base encoder** (all layers except pooler) | ~85M | 84,934,656 | Stage 2 only (pooler), rest frozen |
| BERT pooler dense | 768 × 768 + 768 | 590,592 | Stage 1 (frozen), Stage 2 (unfrozen) |
| **GATConv layer 1** (input → hidden, heads=1) | W: 768×256, att: 2×256 | 197,376 | Both stages |
| **GATConv layer 2** (hidden → output, heads=1) | W: 256×256, att: 2×256 | 66,048 | Both stages |
| BatchNorm (1 layer, between GATConv) | 256 × 2 | 512 | Both stages |
| Classifier (concat → 1) | 512→1 + 1 | 513 | Both stages |

> [!NOTE]
> Existing QAGNN uses `heads=1` (PyG default). We keep `heads=1` for PV-QA-GNN to match
> the proven baseline. Multi-head attention can be explored as a hyperparameter experiment.

### 1.2 NEW in PV-QA-GNN (Component 3 additions)

| Module | Shape | Params | Purpose |
|--------|-------|--------|--------|
| **α_sup** (PV gate slope, support) | scalar | 1 | Controls steepness of entailment mask |
| **β_sup** (PV gate bias, support) | scalar | 1 | Controls midpoint of entailment mask |
| **α_ref** (PV gate slope, refute) | scalar | 1 | Controls steepness of contradiction mask |
| **β_ref** (PV gate bias, refute) | scalar | 1 | Controls midpoint of contradiction mask |
| **GATConv_sup layer 1** (support stream, heads=1) | 768→256, att: 2×256 | 197,376 | Support-specific graph reasoning |
| **GATConv_sup layer 2** (heads=1) | 256→256, att: 2×256 | 66,048 | Support-specific graph reasoning |
| **GATConv_ref layer 1** (refute stream, heads=1) | 768→256, att: 2×256 | 197,376 | Refute-specific graph reasoning |
| **GATConv_ref layer 2** (heads=1) | 256→256, att: 2×256 | 66,048 | Refute-specific graph reasoning |
| BatchNorm_sup (support stream) | 256 × 2 | 512 | Support stream normalization |
| BatchNorm_ref (refute stream) | 256 × 2 | 512 | Refute stream normalization |
| **Evidence quality head** (auxiliary loss) | 256→1 + 1 | 257 | Predicts whether edge belongs to gold subgraph |
| **Classifier** (expanded concat → 1) | (256+256+768)→1 + 1 | 1,281 | `concat(sup_pool, ref_pool, claim_embed)` |

### 1.3 Parameter Count Summary

| Group | Params | Notes |
|-------|--------|-------|
| BERT frozen (Stage 1) | 84,934,656 | Not updated |
| BERT pooler (Stage 2) | 590,592 | Unfrozen in Stage 2 |
| GNN support stream | 263,936 | GATConv_sup × 2 + BN |
| GNN refute stream | 263,936 | GATConv_ref × 2 + BN |
| PV gate params | 4 | α_sup, β_sup, α_ref, β_ref |
| Classifier + evidence head | 1,538 | Main + auxiliary |
| **Total trainable (Stage 1)** | **529,414** | GNN + PV gates + heads |
| **Total trainable (Stage 2)** | **1,120,006** | + BERT pooler |
| **Total model params** | **86,054,662** | Including frozen BERT |

### 1.4 Parameter Initialization

```python
# PV gate parameters (from Component 2 defaults)
α_sup = nn.Parameter(torch.tensor(4.0))    # sigmoid midpoint at p_ent ≈ 0.5
β_sup = nn.Parameter(torch.tensor(-2.0))
α_ref = nn.Parameter(torch.tensor(4.0))    # sigmoid midpoint at p_con ≈ 0.5
β_ref = nn.Parameter(torch.tensor(-2.0))

# GATConv layers: PyG default init (Glorot uniform)
# BatchNorm: default (γ=1, β=0)
# Classifier: default (Kaiming uniform)
# Evidence head: default (Kaiming uniform)
```

---

## 2. Training Procedure

### 2.1 Two-Stage Training

| | Stage 1 (Epochs 1-5) | Stage 2 (Epochs 6-10) |
|---|---|---|
| **BERT layers** | Frozen (all) | Frozen (except pooler) |
| **BERT pooler** | Frozen | **Unfrozen** |
| **GNN streams** | Training | Training |
| **PV gates (α, β)** | Training | Training |
| **Classifier** | Training | Training |
| **Learning rate** | 1e-5 | 2e-6 |
| **Optimizer** | AdamW (weight_decay=0.01) | AdamW (weight_decay=0.01) |
| **Scheduler** | Linear warmup (50 steps) | Linear warmup (50 steps) |
| **Early stopping** | patience=3 (on dev loss) | patience=3 (on dev loss) |

### 2.2 Loss Functions

```python
# Primary: Binary cross-entropy for claim verification
L_bce = BCEWithLogitsLoss()(claim_logit, label)   # label ∈ {0, 1}

# Auxiliary: Evidence quality prediction
# For each edge in the graph, predict if it's in the gold subgraph
L_evidence = BCEWithLogitsLoss()(edge_quality_logit, is_gold_edge)  # per-edge

# Total
L_total = L_bce + λ * L_evidence     # λ = 0.1
```

### 2.3 Data Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Train set | **Full FactKG train** (~86K claims) | User requirement |
| Dev set | **Full FactKG dev** (~11K claims) | For early stopping + threshold tuning |
| Test set | **Full FactKG test** (~11K claims) | Final evaluation, never used for tuning |
| Batch size | 8 | 8GB GPU constraint |
| Max seq length | 256 tokens | CO-GAT recommendation |
| Gradient accumulation | 4 (effective batch = 32) | Matches original QA-GNN batch size |
| Seeds | {42, 1337, 2026} | 3-seed reporting |

---

## 3. Metrics — How They Are Measured

### 3.1 Primary Metrics (from existing `evaluate.py`)

| Metric | Formula | Granularity |
|--------|---------|-------------|
| **Accuracy** | `correct / total` | Overall + per-type |
| **Precision** | `TP / (TP + FP)` | Overall + per-type |
| **Recall** | `TP / (TP + FN)` | Overall + per-type |
| **F1** | `2 * P * R / (P + R)` | Overall + per-type |
| **Loss** (BCE) | `BCEWithLogitsLoss` | Per-epoch train + dev |

**Per-type breakdown** uses the 6 FactKG reasoning types (from `evaluate.py:25`):
- `existence` (existence checks)
- `substitution` (entity substitution)
- `multi hop` (chain of relations)
- `multi claim` (conjunction/AND conditions)
- `negation` (negated claims)
- `single hop` (derived: all claims NOT tagged `multi hop`)

### 3.2 ESI Metrics (from existing `esi.py`)

These are **not loss terms** — they are diagnostic metrics computed per claim after graph construction.

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **ESI_geom** | `(mass_norm × coverage_s × connectivity_s)^(1/3)` | Overall evidence sufficiency |
| **mass_A** | `sum(rel for triple in A+C)` | Total relevance mass of active evidence |
| **mass_A_normalized** | `1 - exp(-mass_A / |A+C|)` | Normalized mass (saturates at ~1.0) |
| **coverage_A** | `|anchors_in_graph| / |anchors|` | Fraction of claim entities covered |
| **connectivity_A** | `connected_anchor_pairs / total_anchor_pairs` | Graph structural coherence |
| **starve_score** | `1 - ESI_geom` | Evidence starvation indicator |

**When computed**: After graph construction (before GNN forward pass), saved per-claim.

### 3.3 PV Metrics (from Component 1)

Computed per-triple by the DeBERTa NLI model (Component 1 output, not recomputed during training):

| Metric | Formula | Source |
|--------|---------|-------|
| **p_entail** (p_ent) | softmax(NLI)[entailment] | DeBERTa MNLI |
| **p_contra** (p_con) | softmax(NLI)[contradiction] | DeBERTa MNLI |
| **p_neutral** (p_neu) | softmax(NLI)[neutral] | DeBERTa MNLI |
| **rel** | `p_ent + p_con` | Relevance (how informative) |
| **pol** | `p_ent - p_con` | Polarity (support vs refute) |

### 3.5 PV Gate Metrics (NEW for Component 3)

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **gate_sup** distribution | `σ(α_sup · p_ent + β_sup)` per edge | Support stream edge weight |
| **gate_ref** distribution | `σ(α_ref · p_con + β_ref)` per edge | Refute stream edge weight |
| **α_sup, β_sup values** | Raw parameter values per epoch | Gate parameter convergence |
| **α_ref, β_ref values** | Raw parameter values per epoch | Gate parameter convergence |
| **effective threshold** | `-β/α` (sigmoid midpoint) | At what PV score does the gate open? |

### 3.6 Backtracking Metrics (from Component 2)

| Metric | Formula | When |
|--------|---------|------|
| **rpi_bridge_at_k** | `connectivity(A+bridge_k) - connectivity(A)` | After bridge rescue |
| **should_recover** | `ESI_geom < tau_esi AND rpi_bridge > 0` | Recovery trigger decision |
| **delta_accuracy** | `acc_after_recovery - acc_before_recovery` | Impact of backtracking |
| **recovery_rate** | `claims_recovered / claims_total` | How often backtracking fires |
| **recovered_gold_rate** | `recovered_is_gold / recovered_total` | Quality of recovery selections |

> [!IMPORTANT]
> **Canonical recovery ranking** (matching validated `recovery.py:_select_bridge_top_k`):
> 1. `connects_components` (boolean) — does this triple bridge disconnected graph components?
> 2. `bridge_bonus` (float) — PPR-derived score from `bridge_rescue.py`
> 3. `rel` (float) — relevance tiebreaker (= p_ent + p_con)
> 
> Component 3 adds a 4th factor after GNN pass: `salience × PV` (GNN attention weight × p_ent)

### 3.7 Robustness Metrics (Component 7)

| Metric | How measured | Target |
|--------|-------------|--------|
| **VitaminC flip rate** | Swap one evidence triple → does prediction flip? | > 70% |
| **Confidence calibration** | ECE (Expected Calibration Error, 10 equal-width bins) | < 0.05 |
| **Per-type SOTA delta** | Our accuracy − BERT baseline accuracy, per type | > 0 on ALL types |
| **Margin formula** | `confidence = |sigmoid(logit) − 0.5| × 2` (0 = uncertain, 1 = confident) | For backtracking trigger: margin < 0.15 |

---

## 4. Intermediate I/O — What is Saved at Every Stage

### 4.1 Before Training (Precomputation)

| File | Format | Contents | When |
|------|--------|----------|------|
| `embeddings.pkl` | pickle | Entity → 768-dim BERT CLS embeddings | **Already exists** |
| `claim_triple_embeddings.pkl` | pickle | `(claim_id, triple_id) → 768-dim` BERT embedding of `[CLS] claim [SEP] triple [SEP]` | **NEW — precompute once** |
| `pv_scores.jsonl` | JSONL | Per-triple PV scores from Component 1 (p_ent, p_con, p_neu, rel, pol) | **Already exists** (C1 output) |
| `esm_pools.jsonl` | JSONL | Per-claim A/S/C assignments from Component 1 ESM | **Already exists** (C1 output) |

### 4.2 Per-Epoch Training Outputs

| File | Format | Contents |
|------|--------|----------|
| `runs/<run_id>/epoch_{n}_train.json` | JSON | `{loss, loss_bce, loss_evidence, accuracy, lr, α_sup, β_sup, α_ref, β_ref}` |
| `runs/<run_id>/epoch_{n}_dev.json` | JSON | Same metrics on dev set |
| `runs/<run_id>/epoch_{n}_checkpoint.pt` | PyTorch | Full model state dict (best-on-dev only, to save disk) |

### 4.3 Per-Claim Diagnostic Outputs (saved during evaluation)

| File | Format | Contents per claim |
|------|--------|--------------------|
| `runs/<run_id>/predictions.jsonl` | JSONL | `{claim_id, claim_text, label, pred, logit, confidence, claim_type}` |
| `runs/<run_id>/graph_stats.jsonl` | JSONL | `{claim_id, n_nodes, n_edges_A, n_edges_C, n_triples_S, anchors, ESI_geom, coverage, connectivity, mass_A}` |
| `runs/<run_id>/gate_values.jsonl` | JSONL | `{claim_id, gate_sup_mean, gate_sup_min, gate_sup_max, gate_ref_mean, gate_ref_min, gate_ref_max}` |
| `runs/<run_id>/attention_weights.jsonl` | JSONL | `{claim_id, edge_id, source, target, relation, attn_sup, attn_ref, gate_sup, gate_ref, p_ent, p_con}` — **per-edge** |
| `runs/<run_id>/recovery_actions.jsonl` | JSONL | `{claim_id, triggered, reason, round, recovered_ids, delta_accuracy, old_logit, new_logit}` |
| `runs/<run_id>/recovery_candidates.jsonl` | JSONL | `{claim_id, round, evidence_id, connects_components, bridge_bonus, rel, p_ent, salience_x_pv, selected}` — **per-candidate** |

### 4.4 Final Aggregated Outputs

| File | Format | Contents |
|------|--------|----------|
| `runs/<run_id>/config.yaml` | YAML | All hyperparameters, paths, git hash, timestamp |
| `runs/<run_id>/metrics.json` | JSON | Overall + per-type accuracy/P/R/F1, with confidence intervals |
| `runs/<run_id>/gate_convergence.json` | JSON | Per-epoch α/β values + effective thresholds |
| `runs/<run_id>/esi_distribution.json` | JSON | ESI histogram (10 bins), per-type ESI means |
| `runs/<run_id>/backtracking_summary.json` | JSON | Recovery rate, trigger rate, recovered-gold rate, Δ accuracy |
| `runs/<run_id>/loss_curves.json` | JSON | Per-epoch train/dev loss (BCE + evidence + total) |
| `runs/<run_id>/best_model.pt` | PyTorch | Best checkpoint (lowest dev loss) |

---

## 5. Forward Pass — Step by Step

```
INPUT per batch (batch_size=8 claims):
├── claim_tokens: {input_ids, attention_mask} — shape [8, 256]
├── data_graphs: PyG Batch with:
│   ├── x: node features [total_nodes, 768] — from embeddings.pkl
│   ├── edge_index: [2, total_edges] — A+C edges
│   ├── edge_attr: [total_edges, 768+5] — claim-conditioned embed + [p_ent, p_con, p_neu, rel, pool_id]
│   ├── batch: node-to-graph mapping [total_nodes]
│   └── pv_scores: [total_edges, 3] — [p_ent, p_con, p_neu]
└── labels: [8] — 0 or 1

FORWARD PASS:
1. BERT encode claims:
   claim_emb = BERT(claim_tokens).last_hidden_state[:, 0]    # [8, 768]

2. Compute cosine relevance per node:
   claim_exp = claim_emb[batch.batch]                         # [total_nodes, 768]
   cos_rel = cosine_similarity(claim_exp, batch.x, dim=-1)    # [total_nodes]

3. Compute PV gates per edge:
   gate_sup = σ(α_sup · batch.pv_scores[:, 0] + β_sup)       # [total_edges]  (p_ent)
   gate_ref = σ(α_ref · batch.pv_scores[:, 1] + β_ref)       # [total_edges]  (p_con)

4. Compute joint attention weights per edge:
   # For each edge (src → dst), use cos_rel of source node
   cos_edge = cos_rel[batch.edge_index[0]]                    # [total_edges]
   w_sup = cos_edge * gate_sup                                # [total_edges]
   w_ref = cos_edge * gate_ref                                # [total_edges]

5. Support stream GNN:
   x_sup = batch.x * cos_rel.unsqueeze(-1)                    # weighted node features
   for layer in GATConv_sup:
       x_sup = layer(x_sup, edge_index, edge_weight=w_sup)
       x_sup = BatchNorm_sup(x_sup); x_sup = ReLU(x_sup)
   sup_pool = global_mean_pool(x_sup, batch.batch)            # [8, 256]

6. Refute stream GNN:
   x_ref = batch.x * cos_rel.unsqueeze(-1)
   for layer in GATConv_ref:
       x_ref = layer(x_ref, edge_index, edge_weight=w_ref)
       x_ref = BatchNorm_ref(x_ref); x_ref = ReLU(x_ref)
   ref_pool = global_mean_pool(x_ref, batch.batch)            # [8, 256]

7. Classify:
   combined = concat(sup_pool, ref_pool, claim_emb)           # [8, 1280]
   combined = Dropout(0.2)(combined)
   logit = Classifier(combined)                                # [8, 1]

8. Evidence quality head (auxiliary):
   edge_repr = x_sup[edge_index[0]] + x_ref[edge_index[0]]   # [total_edges, 256]
   edge_quality_logit = evidence_head(edge_repr)              # [total_edges, 1]

9. Loss:
   L_bce = BCEWithLogitsLoss(logit.squeeze(), labels)
   L_evidence = BCEWithLogitsLoss(edge_quality_logit.squeeze(), is_gold_edge)
   L_total = L_bce + 0.1 * L_evidence

SAVED per forward pass (eval mode only):
├── logit, pred, confidence per claim
├── gate_sup, gate_ref per edge
├── attention weights per edge (from GATConv)
└── edge_quality_pred per edge
```

---

## 6. Backtracking Pass (After Initial Forward)

```
For each claim where BOTH conditions hold:
  (a) confidence < 0.15 (margin between top-2 probabilities)
  (b) connectivity_A < 1.0 (graph has disconnected components)

BACKTRACKING ROUND (max 2 rounds):
1. Get S triples for this claim (from esm_pools.jsonl)
2. Rank S triples by (canonical order from recovery.py:_select_bridge_top_k):
   (i)   connects_components — does triple bridge disconnected components? (bool, primary)
   (ii)  bridge_bonus — PPR-derived score from bridge_rescue.py
   (iii) rel — relevance tiebreaker (= p_ent + p_con)
   (iv)  C3 extension: salience × PV — GNN attention × p_ent from initial pass
3. Select top-k=3 triples
4. Promote S → A: add to graph, rebuild PyG data
5. Re-run forward pass (steps 1-8 above)
6. Save recovery action to recovery_actions.jsonl

SAVED per backtracking round:
├── which S triples were promoted (evidence_ids)
├── old logit vs new logit
├── delta_confidence
├── delta_connectivity
└── whether prediction changed
```

---

## 7. Full Pipeline Execution Order

```
Phase 1: PRECOMPUTATION (run once)
├── Load C1 outputs: pv_scores.jsonl, esm_pools.jsonl
├── Load existing: embeddings.pkl (entity embeddings)
├── Precompute: claim_triple_embeddings.pkl
│   (for each claim × each A+C triple, encode [CLS] claim [SEP] triple [SEP])
├── Build PyG graphs for all claims (A+C edges only)
└── Compute ESI metrics per claim → graph_stats.jsonl

Phase 2: TRAINING (3 seeds × 10 epochs)
├── Stage 1 (epochs 1-5): BERT frozen, lr=1e-5
│   ├── Train on full train set (batch=8, grad_accum=4)
│   ├── Eval on dev set after each epoch
│   ├── Save epoch metrics + PV gate values
│   └── Early stopping (patience=3 on dev loss)
├── Stage 2 (epochs 6-10): BERT pooler unfrozen, lr=2e-6
│   ├── Same as Stage 1 but with lower lr
│   └── Save best checkpoint (by dev loss)
└── Save: config.yaml, loss_curves.json, gate_convergence.json

Phase 3: EVALUATION (on full test set)
├── Load best checkpoint
├── Forward pass on all test claims
├── Save: predictions.jsonl, graph_stats.jsonl, gate_values.jsonl
├── Compute: per-type metrics → metrics.json
├── Run backtracking on low-confidence claims
├── Save: recovery_actions.jsonl, backtracking_summary.json
└── Save: attention_weights.jsonl (for salience analysis)

Phase 4: DIAGNOSTICS (post-eval analysis)
├── ESI distribution vs accuracy correlation
├── PV gate analysis (what α/β converged to)
├── Per-type error analysis
├── Backtracking impact analysis
└── Generate summary report
```

---

## 8. Diagnostic Questions This Spec Answers

After training, you can answer these questions from saved artifacts:

| Question | Where to look |
|----------|---------------|
| Did PV gates converge to reasonable thresholds? | `gate_convergence.json` → check `-β/α` |
| Which claim types benefit most from PV gating? | `metrics.json` per-type vs QA-GNN baseline |
| Does backtracking actually help? | `backtracking_summary.json` → Δ accuracy |
| Are we over-recovering (too many S triples)? | `recovery_actions.jsonl` → count per claim |
| Which edges does the model attend to? | `attention_weights.jsonl` → sort by attn weight |
| Is evidence quality head learning? | `loss_curves.json` → L_evidence trend |
| Do support/refute streams specialize? | `gate_values.jsonl` → compare sup vs ref gate distributions |
| Where does the model fail? | `predictions.jsonl` → filter wrong + per-type |
| Is ESI predictive of accuracy? | Correlate `graph_stats.jsonl:ESI_geom` with `predictions.jsonl:correct` |
