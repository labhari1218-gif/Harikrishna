# Component 2 Decisions Log

Write short bullets here so a new chat (or a reviewer) can reconstruct the exact choices.

---

## Component 1 Output Schema
**Source:** `logs/component1/{train|val}/claims.jsonl` (joined with reconstruction of pool labels)

**Compatibility loader decision (M3):**
- Component 2 accepts:
  - normalized claim JSONL with inline `triples`, or
  - Component 1 claim metrics + companion `pairs.jsonl`.
- For claims+pairs mode, pair logs must include pool labels for full `A/S/C` reconstruction (recommended Component 1 setting: `--pair_log_mode all`).

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
- **Source priority order (runtime):**
  - `provided`: keep incoming `entity_set`/`Entity_set` when already present
  - `pickle`: inject from FactKG split pickles via claim mapping
  - `heuristic`: fallback to AnchorSelector text/degree heuristics when mapping is unavailable
- **FactKG source files:** `data/factkg/factkg_train.pickle`, `data/factkg/factkg_dev.pickle`, `data/factkg/factkg_test.pickle`
- **Existing retrieval path in repo (archaeology):**
  - load split pickles in `datasets.py:get_df(...)`
  - read claim seed entities in `retrieve_subgraphs.py` (`entities = df["Entity_set"][idx]`)
- **FactKG schema used for anchors:** pickle row payload key `Entity_set` (list of KG entity IDs/strings)
- **Claim mapping rule for injection (Option A):**
  - pickles are dicts keyed by claim sentence; we enumerate insertion order and map to Component 1 IDs:
    - `train_{idx}` from `factkg_train.pickle`
    - `val_{idx}` and alias `dev_{idx}` from `factkg_dev.pickle`
    - `test_{idx}` from `factkg_test.pickle`
  - fallback key when ID lookup misses: exact `claim_text` string
- **Format:** list of entity strings (e.g., `["John_E._Beck"]` or `["Bibliography_of_Ulysses_S._Grant", "Henry_Wilson"]`)
- **Normalization:** injected anchors normalize whitespace to underscore-form KG IDs so they match triple entity IDs in graphs.
- **K (max anchors):** `5`
- **Fallback behavior (locked in M1):**
  - first fallback: graph-entity substring matches in claim text (normalized `_`→space, lowercase)
  - second fallback: top-2 entities by graph degree (tie-break by first appearance order)
  - implementation guard: if seed matching yields fewer than 2 anchors and the graph has ≥2 valid entities, continue fallbacks to reach at least 2 anchors (enables pair-based recovery diagnostics)
  - implementation guard: blank entities are excluded from anchor candidates
- **Validation policy:** anchors are required to exist in the claim graph; we do **not** enforce strict substring assertions for seed anchors because FactKG seed entities can be canonical KG forms not verbatim in claim text.
- **Rationale:** Spec suggests `K=2..5`; we choose the upper bound (`5`) to avoid dropping anchor coverage in multi-entity claims while retaining a deterministic cap.
- **Run logging (M3 patch):** every run now logs `anchor_source`, `anchors`, `anchors_hash`, and aggregate `claims_missing_pickle_mapping`.

## Graph
- **Nodes:** entities (subject/object of triples)
- **Edges:** triples with typed relations
- **Edge features:** relation_id, p_ent, p_con, p_neu, pool (A/S/C as integer)
- **Relation embedding dim:** 64
- **Node init (current numpy implementation):** deterministic hash-initialized vectors + degree/anchor structural features (not trainable in current smoke/inference path).

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
- **Relation embedding status (M3 patch):**
  - current implementation is a minimal faithful placeholder: relation vectors are random-initialized and used in relation-conditioned attention, but not optimized end-to-end because Component 2 currently has no training loop.
  - locked roadmap item: move to PyTorch relation-aware GAT with trainable relation embeddings (+optional node-type features) in a future training milestone.
- **Pooling (Option C: hybrid):**
  - Per stream: `pool = concat(mean(nodes), max(nodes), attention_pool(nodes))`
  - Final: `concat(pool_sup, pool_ref)` → MLP classifier
  - Rationale: maximum expressiveness; lets model learn which signal matters

## Bridge rescue
- Method: PPR diffusion
- ε: **0.001** (minimum edge weight floor during PPR)
- γ (neutral cap): **2** in `BridgeBonus'(e) = BridgeBonus(e) * (1 - p_neu)^γ`
- PPR restart/solver defaults (chosen in M2 implementation and now locked):
  - `ppr_alpha = 0.15` (restart probability)
  - `ppr_max_iters = 100`
  - `ppr_tolerance = 1e-6`
- Graph used for bridge scoring: `A ∪ S ∪ C` as an undirected weighted walk graph with edge weight `w(e)=ε+rel(e)`.
- Runtime diagnostics (M2 hardening):
  - per-claim PPR convergence metadata (`converged`, `iterations`, `residual_l1`)
  - anchor connectivity flags (`anchor_pairs_total`, `anchor_pairs_connected`, `has_disconnected_anchors`)
- Rationale: we observed repeated near-converged plateaus at 64/1e-9 (`residual_l1≈6e-5`) on real M3 runs, so we relaxed tolerance and increased iteration budget to avoid systematic false non-convergence while preserving deterministic behavior.

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
- **Connectivity semantics (post-M3 stability fix):**
  - missing anchors in the Active graph are treated as disconnected for `Conn(A)` and `DeltaConn_bridge@k`,
  - S-edges connecting an existing Active component to a missing-anchor node are prioritized as bridge candidates.

## Abstention
- **Abstain score formula:** `abstain_score = α * (1 - max(probs)) + (1-α) * (1 - esi_geom)`
- **Alpha (α):** `0.5` (balanced: equal weight to model confidence and ESI)
- **Operating mode:** support both
  - sweep mode: risk–coverage curve + AURC (sort by ascending abstain score)
  - fixed operating point mode: optional `tau_abstain`
- **Fixed-point metrics (when `tau_abstain` is set):**
  - `coverage = answered / total`
  - `risk = errors_on_answered / answered`
  - `accuracy_on_answered = correct_on_answered / answered`
  - `abstain_rate = 1 - coverage`
- **AURC interpretation:** lower is better (lower error as coverage increases).

## Performance Guardrails
- Numpy-mode warning is emitted when per-claim edge count exceeds `numpy_edge_warn_threshold` (default `5000`).
- Run logs include warning counts and claim IDs to flag expensive slices early.
- Rationale: current numpy loops are correct but not optimized for large graphs; guardrails prevent silent slowdowns before torch/sparse migration.

## Rationale
- **Salience method (Hybrid):**
  - **Default:** `max(attention_sup, attention_ref) * (p_ent + p_con)` (attention × PV relevance)
  - **Validation subset:** Leave-one-out removal on small subset for faithfulness verification
- **Top-N:** 10 edges per claim
- **Faithfulness subset size (M3 default):** 3 claims per run
- **Leave-one-out edges per checked claim (M3 default):** up to top-10 rationale edges
- **Faithfulness criterion (M3 run logging):** mean confidence drop after removal > 0
- **Rationale for subset size:** keeps smoke/evaluation runtime bounded while still validating edge-importance behavior.

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
- Recovery selection details (M2):
  - Rel-policy baseline: top-k S edges by `rel`.
  - Bridge policy: top-k S edges by `BridgeBonus'`, with priority for edges linking different Active connected components.
  - If an edge links an Active node to a node not yet present in Active, it is treated as component-connecting for bridge ranking.
- Recovery trigger must NOT rely only on RPI_rel@k.
  Use: trigger recovery if ESI is low AND DeltaConn_bridge@k > 0 (bridge-policy predicts connectivity gain).
- **Rationale:** BridgeBonus changes selection from S; it doesn't change the graph. If RPI was computed using rel-based selection, it can be low even when a true bridge exists in S.
