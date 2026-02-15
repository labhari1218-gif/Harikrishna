# Component 2 Reference Papers — Implementation Guide

## Papers Downloaded

All PDFs in `docs/component2/references/`:

1. **FactKG** (`FactKG_2305.06590.pdf`) — arXiv 2305.06590
   - Multi-hop, conjunction, negation reasoning types
   - Repo: `docs/component2/references/FactKG/`

2. **GEAR** (`GEAR_1908.01843.pdf`) — arXiv 1908.01843  
   - Graph-based evidence aggregation framework
   - Repo: NOT cloned (cancelled)

3. **CO-GAT** (`CO-GAT_2405.10481.pdf`) — arXiv 2405.10481
   - Node confidence masking for noisy evidence
   - Soft masks vs hard pruning
   - Repo: NOT cloned (cancelled)

4. **HoVer** (`HoVer_EMNLP2020.pdf`) — ACL Findings 2020
   - Many-hop fact extraction benchmark
   - Repo: NOT cloned (cancelled)

5. **RPS** (`RPS_2506.07075.pdf`) — arXiv 2506.07075
   - Reasoning paths as structured graphs (SR-MFV framework)
   - Bridge rescue motivation: multi-hop connectivity

---

## What to Use Where (Component 2 Implementation Map)

Implementation status note (2026-02-10):
- M1-M3 implementation tasks are complete in this repo.
- Current focus is M4 ablations/evaluation; mappings below remain the citation guide for both implemented modules and upcoming ablations.

### T7–T8b: Bridge Rescue + Recovery

**Use RPS (arXiv 2506.07075)**
- **Why:** Explicitly models reasoning paths as graphs, builds subgraphs step-by-step
- **What to steal:**
  - Structure-enhanced multi-hop evidence retrieval (§3.1)
  - How they track reasoning progression through subgraph sequences
  - Connectivity metrics for bridge edges
- **Key insight:** Recovery = adding bridge edges that connect reasoning subgraphs
- **Action:** Read §3 (SR-MFV Framework), especially "Structure-enhanced multi-hop retrieval"

**Use GEAR (arXiv 1908.01843)**
- **Why:** First to use fully-connected evidence graphs with message passing
- **What to steal:**
  - Graph construction from claim-evidence pairs (§3.1)
  - Evidence aggregation strategies (mean vs attention-based, §3.2)
- **Action:** Read §3 (GEAR Model), focus on graph building patterns

### T7: PPR Bridge Scoring

**Use RPS + your FactKG spec**
- **PPR setup:** Weight edges by `rel(e) = p_ent + p_con` (your Component 1 output)
- **Anchor selection:** Use FactKG `Entity_set` (already locked in M1)
- **Neutrality safeguard:** Inspired by CO-GAT's node confidence masking

### T8b: Policy-Aware Recovery Predictor

**Use RPS (arXiv 2506.07075)**
- **Why:** Shows how to measure connectivity gain from adding edges
- **What to implement:**
  - Compute connectivity before/after adding top-k bridge edges
  - Use DeltaConn instead of static RPI threshold
- **Action:** Read §3.1 (graph progression), §4.2 (ablations showing connectivity impact)

### T9: Abstention (Selective Prediction)

**Use CO-GAT (arXiv 2405.10481)**
- **Why:** Introduces node confidence scores for filtering noisy evidence
- **What to steal:**
  - How to combine model confidence with evidence quality metrics
  - Risk–coverage evaluation patterns (if present)
- **Action:** Read §3.2 (node confidence calculation), §4 (experiments)

### T10: Salience Extraction

**Use GEAR (arXiv 1908.01843)**
- **Why:** Uses attention-based aggregation; attention weights = salience proxy
- **What to steal:**
  - How they extract and visualize evidence importance
  - Attention weight calculation in GAT layers
- **Action:** Read §3.2 (Reasoning with GEAR), §4.4 (case study/analysis)

### T5 (already done): Dual-Stream Masking

**Reference CO-GAT (arXiv 2405.10481)**
- **Why:** Validates soft masking approach (you already implemented this in M1)
- **Comparison point:** Your fixed masks vs their learned node confidence
- **Action:** Read §3.3 (node masking mechanism) to confirm your approach is sound

---

## Quick Lookup Table

| Task | Primary Paper | Section | What to Extract |
|------|---------------|---------|-----------------|
| **T7-T8** Bridge Rescue | RPS §3.1 | Structure-enhanced retrieval | Connectivity metrics, graph progression |
| **T7** PPR Scoring | RPS + spec | (your design) | Anchor-based diffusion |
| **T8b** DeltaConn | RPS §4.2 | Ablations | Connectivity gain measurement |
| **T9** Abstention | CO-GAT §3.2 | Node confidence | Confidence scoring patterns |
| **T10** Salience | GEAR §3.2, §4.4 | Attention aggregation | Attention-based importance |
| **Background** Graph Construction | GEAR §3.1 | Graph setup | Entity nodes + typed edges |
| **Validation** Soft Masks | CO-GAT §3.3 | Node masking | Mask formula validation |

---

## Next Steps

1. **Run M4 ablations (T11-T16 in `COMPONENT2_PLAN.md`):**
   - no-PV-mask, no-bridge-rescue, no-recovery-loop
   - stress tests (drop bridge triples, add distractors)
   - learnable mask coefficients and `tau_esi` sensitivity

2. **Use papers as evaluation references (not re-implementation):**
   - RPS §4.2 for connectivity-gain reporting (`DeltaConn_bridge@k`)
   - CO-GAT masking sections for learnable confidence-mask ablation framing
   - GEAR analysis sections for rationale/salience reporting style

3. **Preserve comparability in reports:**
   - keep bridge-policy trigger definition fixed (`ESI<0.3 AND DeltaConn_bridge@k>0`) while sweeping only explicit ablation knobs
   - report both selective prediction metrics (risk-coverage/AURC) and bridge/recovery diagnostics
