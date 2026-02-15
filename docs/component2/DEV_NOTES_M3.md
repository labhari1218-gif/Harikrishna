# DEV NOTES M3 (Reliability + Interpretability)

## 1) Implemented files/classes/functions/signatures

New code files:
- `src/component2/selective.py`
  - `@dataclass(frozen=True) class RiskCoveragePoint`
  - `abstain_score(probs: Sequence[float], esi_geom: float, alpha: float = 0.5) -> float`
  - `risk_coverage_curve(abstain_scores: Sequence[float], is_correct: Sequence[bool]) -> List[RiskCoveragePoint]`
  - `compute_aurc(points: Sequence[RiskCoveragePoint]) -> float`
  - `evaluate_selective_prediction(abstain_scores: Sequence[float], is_correct: Sequence[bool]) -> dict`
- `src/component2/salience.py`
  - `extract_top_rationale_edges(graph: EvidenceGraph, reasoner_output: Dict[str, object], top_n: int = 10) -> List[Dict[str, object]]`
  - `leave_one_out_faithfulness(reasoner, graph: EvidenceGraph, rationale_edges: Sequence[Dict[str, object]], predicted_class_idx: Optional[int] = None, max_edges: Optional[int] = None, base_output: Optional[Dict[str, object]] = None) -> Dict[str, object]`
  - `grouped_edge_removal_faithfulness(reasoner, graph: EvidenceGraph, rationale_edges: Sequence[Dict[str, object]], predicted_class_idx: Optional[int] = None, max_edges: Optional[int] = None, base_output: Optional[Dict[str, object]] = None) -> Dict[str, object]`
- `src/component2/run_m3.py`
  - `run_smoke_m3(input_jsonl: Path | None, output_root: Path, max_claims: int, config: Component2Config = DEFAULT_CONFIG) -> Path`
  - CLI entrypoint: `python3 -m component2.run_m3`

Updated files:
- `src/component2/config.py` (added M3 tunables)
- `src/component2/__init__.py` (exports for selective + salience utilities)

New tests:
- `tests/component2/test_selective.py`
- `tests/component2/test_salience.py`
- `tests/component2/test_run_m3.py`
- `tests/component2/test_run_m3_component1_input.py`

## 2) Exact numeric defaults + source

From `docs/component2/COMPONENT2_DECISIONS.md` and mirrored in `src/component2/config.py`:
- `abstain_alpha = 0.5`
- `rationale_top_n = 10`
- `faithfulness_top_k_edges = 10`

Missing value resolved in M3 and recorded in decisions/config:
- `faithfulness_subset_size = 3` claims per run
  - Rationale: bounded runtime for leave-one-out while still validating faithfulness behavior.

Inherited M1/M2 defaults used unchanged:
- `mask_sup = sigmoid(4*p_ent - 2)`
- `mask_ref = sigmoid(4*p_con - 2)`
- `recovery_tau_esi = 0.3`, `recovery_top_k = 3`

## 3) Tiny toy walkthrough with intermediate numbers

Claim (from synthetic M3 smoke run): `synthetic_m3_0`, anchors `A, C`.

Final (post-recovery) model output:
- `prob_supported_after = 0.3444325990`
- `prob_refuted_after = 0.6555674010`
- predicted class = `REFUTED` with confidence `max_prob=0.6555674010`
- `esi_geom = 0.2`

Abstain score:
- formula: `0.5*(1-max_prob) + 0.5*(1-esi_geom)`
- `= 0.5*(1-0.6555674010) + 0.5*(1-0.2)`
- `= 0.5722162995`

Top rationale edge score (attention×PV):
- edge `0_c1`: `attention_max=0.6899744811`, `pv_relevance = p_ent+p_con = 0.75`
- salience score: `0.6899744811 * 0.75 = 0.5174808608`

Leave-one-out faithfulness (same claim, top rationale edges):
- remove `0_c1`: confidence `0.6555674010 -> 0.6379624966` (drop `0.0176049045`)
- remove `0_s_bad1`: confidence drop `0.0432243505`
- mean drop over checked edges: `0.0344492295` (`is_faithful = true`)

Risk–coverage on 5-claim smoke set:
- coverage/risk points written to `risk_coverage.json` (e.g., coverage `0.6`, risk `0.3333`)
- AURC: `0.2266666667`

## 4) Why each decision was taken

- Kept abstention threshold-free (risk–coverage + AURC) to avoid locking an arbitrary production threshold before calibration.
- Used attention×PV (`max(att_sup,att_ref) * rel`) to combine model-use signal with Component 1 relevance mass.
- Ran leave-one-out on a fixed small subset (`3` claims) to keep M3 smoke deterministic and fast, while still checking faithfulness behavior on real forward passes.

## 5) Logs/outputs produced + exact paths

Synthetic smoke run command (explicit `input_jsonl=None`):
```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from component2.run_m3 import run_smoke_m3
print(run_smoke_m3(input_jsonl=None, output_root=Path("logs/component2"), max_claims=5))
PY
```

Component1-compat revalidation command:
```bash
PYTHONPATH=src python3 -m component2.run_m3 --input-jsonl logs/component1_m3_compat/train/claims.jsonl --max-claims 100 --output-root logs/component2
```

Synthetic smoke run directory (used in walkthrough above):
- `logs/component2/m3/20260210_095628_527172/`

Component1-compat 100-claim revalidation directory (post-fix):
- `logs/component2/m3/20260210_094916_968777/`

Files (synthetic smoke):
- `logs/component2/m3/20260210_095628_527172/run_config.json`
- `logs/component2/m3/20260210_095628_527172/summary.json`
- `logs/component2/m3/20260210_095628_527172/predictions.jsonl`
- `logs/component2/m3/20260210_095628_527172/risk_coverage.json`

## 6) Tests added + what they validate

- `tests/component2/test_selective.py`
  - validates abstain score formula, risk–coverage curve construction, and AURC numeric integration.
- `tests/component2/test_salience.py`
  - validates top-N ranking by attention×PV and leave-one-out confidence-drop reporting.
- `tests/component2/test_run_m3.py`
  - validates M3 smoke outputs (`run_config.json`, `summary.json`, `predictions.jsonl`, `risk_coverage.json`) and presence of abstention/salience fields.
- `tests/component2/test_run_m3_component1_input.py`
  - validates Component 1 claims+pairs adaptation, pool alias handling, and explicit failures when pool labels are missing from pair logs.

Suite run:
- `PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`

## 7) Alternatives considered (>=2) + why not chosen

- Fixed abstain threshold (`abstain_score > tau`) in M3:
  - not chosen because threshold tuning/calibration belongs to experiment phase (M4), while M3 target is threshold-free reliability curves.
- Salience by raw attention only:
  - not chosen because it ignores PV relevance; attention×PV is better aligned with Component 1 confidence signals.
- Full-dataset leave-one-out for every claim:
  - not chosen due cost; subset-based validation provides faster iteration and keeps smoke runs lightweight.

## 8) Failure modes / limitations + next-milestone dependencies

- Faithfulness currently uses mean confidence drop > 0; stronger checks (contrastive/random-baseline removals) are deferred.
- M3 smoke still uses synthetic toy claims by default when no external JSONL is provided.
- Next milestone (M4) depends on this instrumentation to run ablations:
  - no-PV-mask, no-bridge, no-recovery, stress tests, learnable mask coefficients, and `tau_esi` sensitivity.

## 9) Post-M3 critical-fix update (2026-02-10)

A 100-claim Component 1 compatibility run originally surfaced three blocking issues:
- `anchor_len_counts={1:100}` (single-anchor collapse),
- `num_ppr_not_converged=100` with strict solver defaults (`64`, `1e-9`),
- recovery never triggering in practice under those diagnostics.

Applied fixes:
- anchor selection now augments single-seed cases to at least 2 anchors when possible and filters blank entities,
- recovery connectivity now counts missing anchors as disconnected and prioritizes S-edges that can attach missing-anchor nodes,
- PPR solver defaults relaxed to `ppr_max_iters=100`, `ppr_tolerance=1e-6`.

Before/after M3 run comparison:
- old run: `logs/component2/m3/20260210_093453_086968/`
  - `anchor_len_counts={1:100}`
  - `num_ppr_not_converged=100`
- latest run: `logs/component2/m3/20260210_094916_968777/`
  - `anchor_len_counts={2:100}`
  - `num_ppr_not_converged=0`
  - blank anchors removed (`anchors_with_blank=0`)

Remaining observed limitation on this slice:
- `num_recovered=0` and high `AURC≈0.966` persisted because very few claims satisfied both trigger conditions (`esi_geom<0.3` and `DeltaConn_bridge@k>0`), indicating a data/distribution issue rather than the fixed plumbing bugs.

## 10) Design-alignment patch (A-D) follow-up (2026-02-10)

- Added FactKG anchor adapter wiring (`provided -> pickle -> heuristic`) across M1/M2/M3 runners with per-claim source logging and missing-mapping counts.
- Added selective fixed operating-point support (`tau_abstain`) while retaining optional risk-coverage sweep/AURC.
- Clarified relation embedding status as non-trainable numpy placeholder in current pipeline and recorded PyTorch training migration target.
- Added numpy performance guardrails (`numpy_edge_warn_threshold`) plus profiling plan.
- Detailed patch notes: `docs/component2/DEV_NOTES_M3_PATCH.md` and `docs/component2/DEV_NOTES_FIXES.md`.
