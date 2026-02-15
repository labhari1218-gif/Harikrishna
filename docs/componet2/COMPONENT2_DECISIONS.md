# Component 2 Decisions Log

Write short bullets here so a new chat (or a reviewer) can reconstruct the exact choices.

---

## Component 1 Output Schema
**Source:** `logs/component1/{train|val}/claims.jsonl` (joined with reconstruction of pool labels)

**Per-claim fields:**
- `claim_id`, `claim_text`, `label` (TRUE/FALSE)
- `counts_logged`: {A, S, C} counts
- `sufficiency`: {esi_geom, esi_prod, mass_A, coverage_A, connectivity_A, neutral_rate_A, ...}
- `counter_retention`: {has_counter, cr_at_5, cr_at_10, max_contra_C, ...}
- `recovery`: {bridge_count_S, rpi_at_1, rpi_at_3, rpi_at_5, ...}
- `triples`: array of evidence objects

**Per-triple fields (in `triples` array):**
- `evidence_id`: unique ID (e.g., `train_0_triple_7`)
- `raw_triple`: [subject, relation, object] (strings; relation may have `~` prefix for inverse)
- `pool`: "A" | "S" | "C"
- `p_ent`, `p_con`, `p_neu`: PV probabilities (floats summing to ~1.0)

**Derived features (to compute in Component 2):**
- `rel(e) = p_ent + p_con` (decision relevance)
- `pol(e) = p_ent - p_con` (polarity: +support, -refute)

## Anchors
- **Source of anchors:** `Entity_set` field from FactKG split pickles (`data/factkg/factkg_{train|dev|test}.pickle`)  
  ✓ Implements **Option A** from spec §4.2 (use retrieval seed entities directly)
- **Format:** List of entity strings (e.g., `["John_E._Beck"]` or `["Bibliography_of_Ulysses_S._Grant", "Henry_Wilson"]`)
- **Access path:** Loaded via `datasets.py:40`, used in `retrieve_subgraphs.py:147`
- **K (max anchors):** `5`
- **Fallback behavior (locked in M1):**
  - first fallback: graph-entity substring matches in claim text (normalized `_`→space, lowercase)
  - second fallback: top-2 entities by graph degree (tie-break by first appearance order)
- **Rationale:** Spec suggests `K=2..5`; we choose the upper bound (`5`) to avoid dropping anchor coverage in multi-entity claims while retaining a deterministic cap.

## Graph
- **Nodes:** entities (subject/object of triples)
- **Edges:** triples with typed relations
- **Edge features:** relation_id, p_ent, p_con, p_neu, pool (A/S/C as integer)
- **Relation embedding dim:** 64
- **Node init:** Learnable embedding per entity (or mean of incident edge features)

## Reasoner
- **Backbone:** Option C (Hybrid) = GAT + relation-conditioned attention
- **Rationale:** keep native attention weights for salience extraction while injecting explicit relation-type signal into attention/message scoring.
- **Model sizing (Option B):**
  - `hidden_dim = 256`
  - `num_heads = 4`
  - `rel_emb_dim = 64`
  - Balanced capacity without sacrificing training speed
- **Mask formula (Option A: fixed coefficients, v1):**
  - `m_sup(e) = sigmoid(4 * p_ent(e) - 2)`
  - `m_ref(e) = sigmoid(4 * p_con(e) - 2)`
  - Rationale: deterministic and interpretable for v1; learnable coefficients deferred to ablation.
- **Message passing depth (M1):** `1` layer per stream (support/refute).
- **Random seed (M1):** `13` for deterministic initialization in local smoke tests.
- **Pooling (Option C: hybrid):**
  - Per stream: `pool = concat(mean(nodes), max(nodes), attention_pool(nodes))`
  - Final: `concat(pool_sup, pool_ref)` → MLP classifier
  - Rationale: maximum expressiveness; lets model learn which signal matters

## Bridge rescue
- Method: PPR diffusion
- ε: **0.001** (minimum edge weight floor during PPR)
- γ (neutral cap): **2** in `BridgeBonus'(e) = BridgeBonus(e) * (1 - p_neu)^γ`

## Recovery
- **Trigger conditions (legacy spec defaults for reference):**
  - `esi_geom < 0.3` (low sufficiency) AND
  - `rpi_at_k > 0.15` (recovery potential exists) OR graph disconnected between anchors
  - **Note:** `τ_rpi = 0.15` is legacy; see RPI section below for actual implementation
- **Real trigger (Component 2, policy-aware):**
  - `esi_geom < 0.3` (low sufficiency) AND
  - `DeltaConn_bridge@k > 0` (bridge policy predicts connectivity gain)
- **Recovery amount:** `k = 3` (top 3 S edges by BridgeBonus')
- **Constraints:** prefer edges connecting different components; optionally enforce relation diversity

## Abstention
- **Abstain score formula:** `abstain_score = α * (1 - max(probs)) + (1-α) * (1 - esi_geom)`
- **Alpha (α):** `0.5` (balanced: equal weight to model confidence and ESI)
- **Threshold (τ):** None fixed in v1; evaluate via risk–coverage curves and AURC metric
- **Evaluation:** Generate risk–coverage plots; lower risk at same coverage = better insufficiency detection

## Rationale
- **Salience method (Hybrid):**
  - **Default:** Attention weights × PV relevance (fast, computed during forward pass)
  - **Validation subset:** Leave-one-out removal on small subset for faithfulness verification
- **Top-N:** 10 edges per claim
- **Faithfulness test:** On validation subset, removing top-N should significantly drop confidence

## Development Dataset Strategy
- **Phased approach (Option C):**
  - **Phase 1:** 100 train claims (sanity check, architecture debugging)
  - **Phase 2:** 1000 train + 500 val claims (hyperparameter tuning, ablations)
  - **Phase 3:** Full dataset (86K train + 13K val)
- **Rationale:** Safest for debugging; catch issues early before expensive full training

## RPI and Recovery Policy (Important)
- Keep existing Component 1 RPI@k unchanged for comparability; refer to it as RPI_rel@k if needed.
- Add a policy-aware metric for Component 2:
  - RPI_bridge@k = Conn(A ∪ Select_k(S, bridge_policy)) − Conn(A)
  - DeltaConn_bridge@k is the same quantity (explicit name in logs).
- Recovery trigger must NOT rely only on RPI_rel@k.
  Use: trigger recovery if ESI is low AND DeltaConn_bridge@k > 0 (bridge-policy predicts connectivity gain).
- **Rationale:** BridgeBonus changes selection from S; it doesn't change the graph. If RPI was computed using rel-based selection, it can be low even when a true bridge exists in S.
