# Best Practices Extracted from Papers — For PV-QA-GNN

> **Purpose**: Every design decision for Component 3+ backed by specific paper evidence.
> **Context**: FactKG dataset, 8GB GPU, existing QA-GNN in `models.py`, S-only backtracking.

---

## 1. Graph Architecture Best Practices

### 1.1 Use Soft Confidence Gating (Not Hard Masks)
- **Source**: CO-GAT §V-B (Ablation Study)
- **Finding**: Soft masking (CO-SCO ∈ [0,1]) beats hard 0/1 masking by **+0.79%** on FEVER
- **How we apply**: PV edge masks = `σ(α·p_ent + β)` per edge, where α and β are trainable parameters
- **Init**: α=4.0, β=−2.0 (sigmoidal midpoint at p≈0.5, matching Component 2 defaults)
- **Why it matters**: Hard masks lose gradient flow during training. Soft masks let the model learn which evidence to trust.

### 1.2 GAT Over GCN for Evidence Graphs
- **Source**: GEAR §4.1 (Comparison of aggregators), Existing QA-GNN code
- **Finding**: GAT (attention-based) > mean/max aggregator on FEVER by ~0.3%
- **How we apply**: Keep existing `GATConv` from `models.py` — proven architecture
- **Config**: hidden=256, layers=2, heads=1, dropout=0.3, batch_norm=True (all from existing code)

### 1.3 Cosine Relevance + PV Scores as Joint Attention Prior
- **Source**: Existing QA-GNN `forward()` + CO-GAT node masking
- **Finding**: QA-GNN already computes `cosine_similarity(claim_emb, node_features)` as relevance
- **How we apply**: Augment: `relevance = cosine(claim, node) × σ(α·pv_score + β)`
- **Rationale**: Cosine captures semantic similarity; PV score captures entailment/contradiction signal from DeBERTa NLI. Multiplying them gives a richer attention prior.

### 1.4 Dual-Stream Separation with SEPARATE GAT Parameters (Support vs Refute)
- **Source**: Our Component 2 design (validated by CO-GAT's single-stream limitation)
- **Finding**: GEAR and CO-GAT use single streams → conflicting evidence signals get mixed
- **How we apply**: Two **independent** GATConv layer stacks:
  - `GATConv_sup(h=256, L=2, heads=1)`: edges weighted by `cosine × σ(α_sup · p_ent + β_sup)`
  - `GATConv_ref(h=256, L=2, heads=1)`: edges weighted by `cosine × σ(α_ref · p_con + β_ref)`
  - Final: `concat(support_pool, refute_pool, claim_embed) → classifier`
- **Why separate (not shared)**: Support reasoning (entailment patterns) and refute reasoning (contradiction patterns) are fundamentally different. Separate parameters let each stream specialize. Extra ~2MB is negligible on 8GB.
- **Why novel**: No existing paper does dual-stream evidence reasoning with NLI-derived masks and separate GNN parameters. This is our contribution.

### 1.5 Claim-Conditioned Triple Representations (Key SOTA Trick)
- **Source**: GEAR §3.1, CO-GAT §IV-A, BEST_PRACTICES §3.3
- **Finding**: Most papers encode entities independently. Encoding each triple as `[CLS] claim [SEP] subject relation object [SEP]` makes the GNN "claim-aware" per edge.
- **How we apply**: For each edge in the graph:
  - Compute `BERT([CLS] claim [SEP] subj rel obj [SEP])[CLS]` → 768-dim embedding
  - Use as edge_attr in PyG graph (alongside PV scores)
  - Precompute and cache as `claim_triple_embeddings.pkl`
- **Why it matters for SOTA**: Helps multi-hop and negation types — the GNN can now see how each triple relates to the claim, not just to other triples.
- **Validated against codebase**: Current `convert_to_pyg_format()` encodes entities independently — this is the upgrade.

### 1.6 Progressive Evidence Accumulation
- **Source**: RPS (SR-MFV) §IV-B1 (Subgraphs Construction)
- **Finding**: Building progressively growing subgraphs (G₁ ⊂ G₂ ⊂ ... ⊂ Gₙ) captures "how critical information emerges" — gains increase with hop count (1.48% → 1.51% → **3.09%** improvement from 2-hop to 4-hop)
- **How we apply**: Our backtracking rounds ARE progressive subgraph construction:
  - Round 0: Graph from A+C triples only
  - Round 1: + top-k recovered triples from S pool
  - Round 2: + additional S recoveries
- **Key insight from RPS**: "Retrieving too many sentences may lead to... increasing reasoning difficulty and degrading final verification accuracy" → validates our conservative S-only recovery (max 2 rounds, top-3 per round)

---

## 2. Training Best Practices

### 2.1 Two-Stage Warm-Up Training
- **Source**: CO-GAT §IV-D (Training Strategy)
- **Finding**: Warm-up schedule improves convergence significantly
- **How we apply**:
  - **Stage 1** (5 epochs): Freeze BERT, train GNN + PV mask params only, lr=1e-5
  - **Stage 2** (5 epochs): Unfreeze BERT pooler layer, lr=2e-6
- **Rationale**: BERT is already pretrained. Training GNN first lets it learn graph structure without perturbing language representations. Then fine-tuning BERT's pooler adapts it to our specific task.

### 2.2 Multi-Task Loss
- **Source**: CO-GAT §IV-C (Multi-Task Training)
- **Finding**: Adding evidence relevance prediction as auxiliary task gives **+0.18%** overall
- **How we apply**:
  ```
  L_total = L_BCE(claim_pred, label) + λ · L_evidence(edge_pred, is_gold_evidence)
  ```
  where λ=0.1 and `is_gold_evidence` = 1 if the triple appears in the gold subgraph
- **8GB constraint**: No extra model parameters needed — just add a linear head on edge representations

### 2.3 Optimizer and Schedule
- **Source**: Existing `run_stuff.py`, CO-GAT §IV-D
- **Config**:
  - Optimizer: AdamW (existing code)
  - Learning rate: 1e-5 (Stage 1), 2e-6 (Stage 2)
  - Scheduler: Linear warmup with 50 warmup steps (existing code)
  - Early stopping: patience=3 (existing code, within CO-GAT's recommended ≤5)

### 2.4 Batch Size for 8GB GPU
- **Source**: CO-GAT §IV-D ("batch_size=8 for BERT-large")
- **Analysis**:
  - BERT-base: ~440MB model
  - GATConv layers: ~2MB
  - Batch of 8 graphs (avg ~15 nodes each): ~50MB activations
  - Total: ~2-3GB with gradients → fits in 8GB
  - If tight: use `gradient_accumulation_steps=4` with batch_size=2
- **Config**: `batch_size=8` (down from existing 32)

### 2.5 Max Sequence Length = 256
- **Source**: CO-GAT §IV-D
- **Finding**: Truncation at 256 tokens is sufficient for claim-evidence pairs
- **Existing code**: Uses `max_length=512` → reduce to 256 to save ~40% VRAM per batch
- **Truncation**: Tail truncation (keep claim, truncate evidence)

---

## 3. Evidence Graph Construction Best Practices

### 3.1 Build Graphs from A+C Triples Only
- **Source**: Our Component 1 ESM design + FactKG dataset structure
- **How we apply**:
  - Node set = entities from Active + Counter pools
  - Edge set = triples from A+C pools
  - Edge attributes = `[p_ent, p_con, p_neu, rel_score, pool_id]`
  - S pool is held in reserve for backtracking
- **Why not include S**: S triples scored below PV thresholds — they are noise unless bridge rescue identifies them as structurally necessary

### 3.2 Node Features = Precomputed BERT Embeddings
- **Source**: Existing `datasets.py` (`get_precomputed_embeddings()`)
- **How we apply**: Reuse existing `embeddings.pkl` containing BERT CLS embeddings for each entity
- **Advantage over hash vectors**: Semantic meaning (Component 2 used hash → no semantics)
- **8GB constraint**: Precomputed embeddings = no extra forward pass during training

### 3.3 Edge Attributes = Claim-Conditioned Triple Embeddings
- **Source**: GEAR §3.1, CO-GAT §IV-A, see also §1.5 above
- **Finding**: Both papers encode evidence as `[CLS] claim [SEP] evidence [SEP]`
- **How we apply**: For each **edge** (NOT node), encode the triple as `BERT([CLS] claim [SEP] subj rel obj [SEP])[CLS]` → 768-dim **edge_attr**
- **Distinction from §3.2**: Node features remain precomputed entity-level BERT embeddings. Edge attributes carry claim-conditioned triple embeddings. These are DIFFERENT things serving DIFFERENT purposes.
- **Precomputation**: Cache as `claim_triple_embeddings.pkl` to avoid recomputing during training

---

## 4. Backtracking Best Practices

### 4.1 S-Only Recovery
- **Source**: User decision + Component 1 ESM design + RPS validation
- **Rationale**:
  - A triples are already in the graph (high PV, active evidence)
  - C triples are already in the graph (contradiction evidence)
  - S triples are the only "lost information" — weakly scored but potentially useful as:
    - Bridge triples (connecting disconnected graph components)
    - Late-binding evidence (gains relevance given other evidence)

### 4.2 Conservative Recovery Budget
- **Source**: RPS §V-E (Hyperparameter Sensitivity Analysis)
- **Finding**: "Retrieving too many sentences... introduces redundant or irrelevant information, increasing reasoning difficulty" — performance degrades past optimal hop count
- **How we apply**:
  - `max_backtrack_rounds=2` (diminishing returns beyond 2)
  - `backtrack_k=3` per round (max 6 recovered triples total)
  - Trigger: only when low confidence (margin < 0.15) AND structural deficiency (disconnected graph)

### 4.3 Principled Recovery Selection (Validated)
- **Source**: Component 2's PPR bridge rescue + SaGP salience scores
- **Validated**: `recovery.py:_select_bridge_top_k()` already implements this hierarchy:
  1. **Bridge potential**: `_edge_connects_components()` — boolean, ranks first
  2. **Bridge bonus**: PPR-derived `bridge_bonus_capped` from `bridge_rescue.py`
  3. **PV/rel score**: `triple.rel` as tiebreaker
- **For Component 3**, add after GNN pass:
  4. **Salience × PV**: `GNN_attention_weight × p_ent` from initial forward pass

---

## 5. Evaluation Best Practices

### 5.1 Per-Reasoning-Type Breakdown
- **Source**: FactKG paper §4, existing `evaluate.py`
- **Baselines on FactKG**:
  - Published SOTA: ~86.82% overall (PGR, 2025); other 2025 methods: MHGCI ~84.41%, ClaimPKG ~85.22%
  - Our reproduced BERT single-step baseline: ~93% overall (strong internal baseline)
  - Original FactKG paper baselines: BERT 65.20%, GEAR 77.65%
- **Goal**: Beat published SOTA (~86.82%) and match/exceed our BERT baseline (~93%) on ALL types, especially hard types (multi hop, negation)
- **Already implemented**: `evaluate.py` does per-type accuracy, precision, recall, F1
- **Canonical type labels** (from evaluate.py:25): `existence`, `substitution`, `multi hop`, `multi claim`, `negation`, `single hop` (derived: not multi-hop)

### 5.2 Model-Routing for Maximum Overall Accuracy
- **Source**: User feedback (2026-02-15), Component 6 design
- **Key insight**: Don't force PV-QA-GNN to handle every claim. Route:
  - **Easy claims** (existence, substitution, single hop, high margin) → BERT single-step baseline (fast, ~93%)
  - **Hard claims** (multi-hop, negation, low margin, disconnected) → PV-QA-GNN + backtracking
- **Why this is critical for SOTA**: If PV-QA-GNN is even slightly worse on easy claims (due to graph noise), routing easy claims to BERT preserves those wins while gaining on hard claims.

### 5.3 Contrastive Robustness Test (VitaminC-style)
- **Source**: VitaminC (NAACL 2021)
- **Method**: For each test claim, create a minimal contrastive pair by swapping one evidence triple → prediction should flip
- **Metric**: Flip rate (target > 70%) — measures whether model actually reads the evidence
- **Why important**: Many fact-checkers learn shortcuts (entity frequency, claim style) without actually reasoning. This catches them.

### 5.4 Faithfulness Comparison (ProoFVer baseline)
- **Source**: ProoFVer (TACL 2022)
- **Finding**: ProoFVer achieves **+13.21%** improvement on counterfactual instances by using natural logic proofs
- **How we use**: Compare our salience-based rationales (attention × PV) against ProoFVer's formal logic proofs
- **Our advantage**: We produce rationales from graph attention — more interpretable than black-box classifiers, but less formal than logic proofs. Position this honestly in thesis.

### 5.5 Component Ablation Study
- **Source**: RPS §V-D, CO-GAT §V-B (both do systematic ablations)
- **Required ablations**:
  1. No PV masks (plain QA-GNN baseline)
  2. No dual-stream (single-stream with combined PV, shared GAT)
  3. No claim-conditioned triples (entity-only encoding)
  4. No bridge rescue / no backtracking
  5. No multi-task loss
  6. No warm-up (train everything from start)
  7. No model routing (PV-QA-GNN on all claims)
  8. Full pipeline

### 5.6 Three-Seed Reporting
- **Source**: User feedback (2026-02-15)
- **Why**: Single-seed results are unreliable for <1% improvements
- **How**: Run 3 seeds, report mean ± std for all metrics
- **Seeds**: {42, 1337, 2026}

---

## 6. GPU Memory Budget (8GB)

| Component | VRAM Estimate |
|-----------|-------------|
| BERT-base model | ~440 MB |
| BERT gradients (pooler only) | ~50 MB |
| GATConv layers (2 × 256-dim) | ~2 MB |
| GATConv activations + gradients | ~50 MB |
| Batch of 8 graphs (15 nodes avg) | ~20 MB |
| Tokenized sequences (256 tokens × 8) | ~10 MB |
| PyG batch overhead | ~20 MB |
| **Total (training)** | **~2-3 GB** |
| **Available headroom** | **~5-6 GB** |

> [!TIP]
> With 5-6 GB headroom, we can potentially increase batch_size to 16 or use online embeddings (compute BERT embeddings during training) without running out of memory. Start conservative (batch=8, precomputed), then scale up.

---

## 7. Design Decisions Summary (Locked)

| Decision | Value | Paper Source | Confidence |
|----------|-------|-------------|-----------|
| Edge masking type | Soft (σ-gated) | CO-GAT | ★★★★★ |
| GNN architecture | GATConv, h=256, L=2 | Existing QA-GNN | ★★★★★ |
| PV mask init | α=4.0, β=−2.0 (trainable) | Component 2 defaults | ★★★★☆ |
| Training stages | 2-stage warm-up | CO-GAT | ★★★★★ |
| Multi-task loss | BCE + evidence quality | CO-GAT | ★★★★☆ |
| Optimizer | AdamW + linear warmup | Existing code | ★★★★★ |
| Early stopping | patience=3 | Existing code | ★★★★★ |
| Batch size | 8 (8GB GPU) | CO-GAT large model | ★★★★☆ |
| Max seq length | 256 | CO-GAT | ★★★★☆ |
| Backtracking source | S-only | User + RPS validation | ★★★★★ |
| Recovery budget | max 2 rounds, k=3 | RPS sensitivity analysis | ★★★★☆ |
| Dual-stream | Yes (support + refute) | Novel (our contribution) | ★★★★☆ |
| Node features | Precomputed BERT | Existing code | ★★★★★ |
| Robustness test | VitaminC contrastive | VitaminC paper | ★★★★★ |
| Faithfulness comparison | vs ProoFVer | ProoFVer paper | ★★★★☆ |

---

## 8. Position Against Related Work

| Method | Architecture | Evidence Handling | Our Advantage |
|--------|-------------|------------------|---------------|
| GEAR | BERT + fully-connected GAT | Single stream, all evidence equal | We separate support/refute streams with NLI-derived masks |
| CO-GAT | BERT + masked GAT | Node masking with claim confidence | We mask *edges* (triples) not nodes, using PV scores from dedicated NLI model |
| RPS/SR-MFV | BERT + progressive subgraphs | Multiple retrieval hops | We do progressive *recovery* (backtracking), not progressive retrieval |
| ProgramFC | LLM + program decomposition | Sub-task handlers | We use end-to-end GNN, no LLM inference cost |
| ProoFVer | Natural logic proofs | Formal logic chains | We use GNN attention for interpretability, more scalable |
| Our PV-QA-GNN | BERT + dual-stream GAT + PV masks | A/S/C partition + S-only backtracking | NLI-informed graph reasoning with evidence management |
