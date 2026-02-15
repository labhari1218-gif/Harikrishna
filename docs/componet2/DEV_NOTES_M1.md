# DEV NOTES M1 (Minimal End-to-End, No Bridge Recovery)

## 1) What was implemented

New code files (additive only):
- `src/component2/config.py`
  - `@dataclass(frozen=True) class Component2Config`
  - `DEFAULT_CONFIG = Component2Config()`
  - `Component2Config.to_snapshot_dict(self) -> Dict[str, Any]`
- `src/component2/types.py`
  - `@dataclass(frozen=True) class EvidenceTriple`
  - `@dataclass(frozen=True) class ClaimRecord`
  - `@dataclass(frozen=True) class GraphEdge`
  - `@dataclass class EvidenceGraph`
  - `evidence_from_dict(raw: Dict[str, Any], index: int = 0) -> EvidenceTriple`
  - `claim_from_dict(raw: Dict[str, Any]) -> ClaimRecord`
  - `claim_label_to_index(label: Optional[str]) -> Optional[int]`
- `src/component2/anchor_selector.py`
  - `normalize_entity(entity: str) -> str`
  - `class AnchorSelector`
  - `AnchorSelector.select_anchors(self, claim_text: str, triples: Sequence[EvidenceTriple], seed_entities: Sequence[str] | None = None) -> List[str]`
- `src/component2/graph_builder.py`
  - `@dataclass class RelationVocab`
  - `class GraphBuilder`
  - `GraphBuilder.build(self, claim_id: str, triples: Sequence[EvidenceTriple], anchors: Sequence[str]) -> EvidenceGraph`
  - `GraphBuilder.relation_vocab_snapshot(self) -> Dict[str, int]`
- `src/component2/reasoner.py`
  - `class HybridMaskedDualStreamReasoner`
  - `HybridMaskedDualStreamReasoner.support_mask(p_ent: np.ndarray) -> np.ndarray`
  - `HybridMaskedDualStreamReasoner.refute_mask(p_con: np.ndarray) -> np.ndarray`
  - `HybridMaskedDualStreamReasoner.forward_graph(self, graph: EvidenceGraph) -> Dict[str, object]`
  - `HybridMaskedDualStreamReasoner.forward_batch(self, graphs: Sequence[EvidenceGraph], labels: Optional[Sequence[int]] = None) -> Dict[str, object]`
  - `HybridMaskedDualStreamReasoner.compute_loss(self, logits: np.ndarray, labels: Sequence[int]) -> float`
- `src/component2/io_utils.py`
  - `load_claims_jsonl(path: Path, max_claims: int | None = None) -> List[ClaimRecord]`
  - `write_json(path: Path, payload: Dict[str, object]) -> None`
  - `write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None`
- `src/component2/run_m1.py`
  - `run_smoke(input_jsonl: Path | None, output_root: Path, max_claims: int, config: Component2Config = DEFAULT_CONFIG) -> Path`
  - CLI: `python3 -m component2.run_m1 --max-claims 5 --output-root logs/component2`
  - Writes reproducibility snapshot: `run_config.json`

New tests:
- `tests/component2/test_anchor_selector.py`
- `tests/component2/test_graph_builder.py`
- `tests/component2/test_reasoner.py`
- `tests/component2/test_run_m1.py`

## 2) Exact defaults used (numeric) and where they came from

From locked decisions:
- `hidden_dim=256`, `num_heads=4`, `rel_emb_dim=64` (Model sizing Option B in `COMPONENT2_DECISIONS.md`)
- Masks: `m_sup=sigmoid(4*p_ent-2)`, `m_ref=sigmoid(4*p_con-2)` (fixed coefficients)
- Bridge/recovery constants reserved: `epsilon=0.001`, `gamma=2`, `tau_esi=0.3`, `tau_rpi_legacy=0.15`, `k=3`

Missing values resolved in M1 and immediately recorded:
- `anchor_max_k=5` (spec says 2..5; chose upper bound for coverage)
- `anchor_fallback_degree_k=2`
- `random_seed=13` (deterministic local smoke behavior)
- `message_passing_layers=1` (single-pass M1 baseline)

Canonical source for current defaults:
- `docs/component2/COMPONENT2_CONFIG.md`
- runtime dataclass in `src/component2/config.py`

## 3) Mechanism walkthrough with toy graph (intermediate numbers)

Toy graph (A∪C used in M1):
- `e1`: Obama-birthPlace-Honolulu, `p_ent=0.72`, `p_con=0.06`, `p_neu=0.22`
- `e2`: Honolulu-isPartOf-Hawaii, `p_ent=0.64`, `p_con=0.05`, `p_neu=0.31`
- `e3`: Obama-birthPlace-Kenya, `p_ent=0.05`, `p_con=0.75`, `p_neu=0.20`

Mask values (exact from formula):
- `m_sup(e1)=sigmoid(0.88)=0.706822`
- `m_sup(e2)=sigmoid(0.56)=0.636453`
- `m_sup(e3)=sigmoid(-1.80)=0.141851`
- `m_ref(e1)=sigmoid(-1.76)=0.146790`
- `m_ref(e2)=sigmoid(-1.80)=0.141851`
- `m_ref(e3)=sigmoid(1.00)=0.731059`

Observed in smoke outputs (`predictions.jsonl`):
- `attention_sup` equals mask values for this topology because each target has one incoming edge (`base_attention_sup=1.0`).
- highest refute salience is `e3` (`0.731059`), matching high contradiction mass.

Bridge score + connectivity delta preview (for continuity with upcoming M2):
- `rel(e1)=0.78`, `rel(e2)=0.69`, `rel(e3)=0.80`
- with assumed PPR masses (`p(Obama)=0.40`, `p(Honolulu)=0.20`, `p(Hawaii)=0.35`, `p(Kenya)=0.08`), `epsilon=0.001`, `gamma=2`:
  - `BridgeBonus'(e1)=0.037980`
  - `BridgeBonus'(e2)=0.022796`
  - `BridgeBonus'(e3)=0.015390`
- connectivity illustration: if anchor path lacks `e2`, then `Conn(A)=0`; adding `e2` gives `Conn(A∪{e2})=1`, so `DeltaConn_bridge@1=1`.

## 4) Why each decision was taken (tied to DECISIONS)

- Anchor source remains `Entity_set` to preserve retrieval-time entity linking consistency.
- Fixed masks were retained per locked decision to keep M1 interpretable and auditable before learnable-mask ablation.
- Hybrid pooling (mean+max+attention per stream) was implemented exactly per locked Option C to preserve both global and salient-local signals.
- Reproducibility snapshot in runner was added to satisfy the explicit run-config constraint and to bind every run to exact tunables.

## 5) Logs/outputs produced and exact paths

- Smoke run command:
  - `PYTHONPATH=src python3 -m component2.run_m1 --max-claims 5 --output-root logs/component2`
- Run directory:
  - `logs/component2/m1/20260210_072118/`
- Files:
  - `logs/component2/m1/20260210_072118/run_config.json`
  - `logs/component2/m1/20260210_072118/summary.json`
  - `logs/component2/m1/20260210_072118/predictions.jsonl`

## 6) Tests added and what they validate

- `tests/component2/test_anchor_selector.py`
  - validates seed-first anchors, text fallback, degree fallback, and max-K behavior.
- `tests/component2/test_graph_builder.py`
  - validates A∪C filtering, node/entity construction, relation ID mapping, and `rel/pol` feature values.
- `tests/component2/test_reasoner.py`
  - validates exact fixed-mask math, batch forward output shapes (`probs/logits`), salience emission, and relation-conditioned effect on outputs.
- `tests/component2/test_run_m1.py`
  - validates smoke runner output creation and reproducibility snapshot presence/content (`run_config.json`).

Executed test command:
- `PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`

## 7) Alternatives considered (>=2) and why not chosen now

- PyTorch/torch-geometric implementation for M1:
  - not chosen here because the active execution environment lacks torch packages; numpy baseline allowed immediate, deterministic delivery and tests.
- Hard-pruning low-PV edges:
  - rejected because locked design requires soft masks to avoid suppressing potentially recoverable bridge evidence.
- Single-stream classifier (no support/refute split):
  - rejected because locked design explicitly requires polarity-aware dual streams.

## 8) Failure modes / limitations discovered + next-milestone dependencies

- Current M1 attention can collapse to mask-only values on nodes with single incoming edges; richer multi-incoming topologies are needed for stronger relation-attention contrast.
- Runner currently defaults to synthetic claims when no input JSONL is supplied; full integration with reconstructed Component 1 triples still needs a dedicated loader path.
- Bridge scoring and recovery policy are not active yet; M2 depends on adding:
  - PPR bridge bonus with neutral cap,
  - policy-aware `DeltaConn_bridge@k` computation,
  - one-step `S→A` recovery and rerun trigger.
