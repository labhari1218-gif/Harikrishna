# DEV NOTES M3 PATCH (Design Alignment A-D)

## Scope

This patch aligns Component 2 behavior with locked design decisions and paper-motivated expectations without modifying Component 1 modules.

## A) FactKG Anchor Adapter (Option A)

Implemented:
- `src/component2/anchor_adapter_factkg.py`
  - `FactKGAnchorAdapter.load()` reads:
    - `data/factkg/factkg_train.pickle`
    - `data/factkg/factkg_dev.pickle`
    - `data/factkg/factkg_test.pickle`
  - builds `claim_id -> Entity_set` mapping via split-index convention:
    - `train_{idx}`, `val_{idx}`/`dev_{idx}`, `test_{idx}`
  - fallback lookup: exact `claim_text`
  - `inject_anchors(record)` and `inject_claim(claim)` support source priority:
    - `provided` -> `pickle` -> `heuristic`

Runner wiring:
- `src/component2/run_m1.py`
- `src/component2/run_m2.py`
- `src/component2/run_m3.py`

Runtime logs now include:
- per claim: `anchor_source`, `anchors`, `anchors_hash`
- run aggregate: `anchor_source_counts`, `claims_missing_pickle_mapping`

Validation:
- `tests/component2/test_anchor_adapter_factkg.py`
  - pickle injection by `claim_id`
  - fallback to heuristic when mapping missing
  - regression: bridge score changes when injected anchors differ from heuristic anchors

## B) Fixed Abstention Operating Point (`tau_abstain`)

Implemented:
- `src/component2/selective.py`
  - `selective_operating_point(...)`
  - `evaluate_selective_prediction(..., tau_abstain=None, compute_curve=True)`

Metrics at fixed tau:
- `coverage`
- `risk`
- `accuracy_on_answered`
- `abstain_rate`

Sweep behavior:
- preserved by default (`compute_risk_coverage_sweep=True`)
- can be disabled while retaining fixed-point metrics

Runner integration:
- `src/component2/run_m3.py`
  - config/runtime snapshot now logs `tau_abstain` and sweep toggle
  - summary logs fixed-point metrics when tau is provided

Validation:
- `tests/component2/test_selective.py`
- `tests/component2/test_run_m3.py`

## C) Relation Embeddings Status Clarification

Current implementation:
- relation vectors are random-initialized numpy parameters used in relation-conditioned attention
- not end-to-end trainable in current Component 2 smoke/inference path

Documentation/runtime updates:
- `docs/component2/COMPONENT2_DECISIONS.md`
- `docs/component2/COMPONENT2_CONFIG.md`
- `run_config.json` runtime now states `relation_embeddings.mode=fixed_features_numpy`, `trainable=false`

Planned milestone:
- PyTorch relation-aware GAT with trainable relation embeddings and optional node-type features

## D) Performance Guardrails and Profiling Note

Guardrail implementation:
- new config key: `numpy_edge_warn_threshold` (default `5000`)
- warnings emitted in `run_m1.py`, `run_m2.py`, `run_m3.py` when edge count exceeds threshold
- summary/runtime logs include warning counts and claim IDs

Current complexity hotspots:
- reasoner stream pass: roughly `O(L * H * E)` for attention/message loops (`L` layers, `H` heads, `E` edges)
- bridge PPR solver: roughly `O(I * (N + E))` per claim (`I` iterations, `N` nodes)
- bridge/recovery ranking: `O(E log E)` sorting in S-edge selection paths

Optimization plan (next step, no behavior change in this patch):
1. vectorize edge-loop message computations in numpy
2. switch dense transition operations to sparse ops for PPR
3. add torch backend for relation-aware message passing + batched graphs
4. provide claim-batch chunking presets for large validation sweeps

## Validation Runs

Smoke outputs:
- `logs/component2/m1/20260210_104934_026305/`
- `logs/component2/m2/20260210_104939_838090/`
- `logs/component2/m3/20260210_104944_757895/`
- pickle-injection verification (`entity_set` removed input): `logs/component2/m3/20260210_105002_449038/`

Tests:
- `PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- Result: 52/52 tests passing.
