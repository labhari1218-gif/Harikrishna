# DEV NOTES FIXES (Component 2 A-D Alignment)

## Why this patch

Component 2 behavior drifted from the locked design in three places:
- anchors defaulted to heuristics when Component 1 logs lacked `entity_set`
- abstention only reported sweep metrics (no fixed operating point)
- relation-embedding and performance status were under-documented for paper alignment

This patch fixes those gaps additively under `src/component2/` and `tests/component2/`.

## What changed

1. **Anchor source alignment (Option A)**
- Added `src/component2/anchor_adapter_factkg.py`
- Injects `Entity_set` from FactKG pickles by `claim_id`/`claim_text` when seeds are missing
- Wired into `run_m1.py`, `run_m2.py`, `run_m3.py`
- Logs `anchor_source`, `anchors_hash`, and `claims_missing_pickle_mapping`
- Archaeology validation: Component 1 IDs matched FactKG pickle ordering exactly on available splits (`train`: `82,083/82,083`, `val`: `12,491/12,491` by both claim text and index-based `split_{idx}` mapping).

2. **Selective fixed operating point**
- Extended `src/component2/selective.py` with `tau_abstain` metrics:
  - `coverage`, `risk`, `accuracy_on_answered`, `abstain_rate`
- Preserved risk-coverage sweep/AURC via `compute_risk_coverage_sweep`
- Integrated into `run_m3.py` outputs and config snapshots

3. **Relation embedding status clarity**
- Documented current mode as fixed numpy features (non-trainable in current pipeline)
- Added explicit roadmap note for trainable PyTorch relation-aware model

4. **Performance guardrails**
- Added `numpy_edge_warn_threshold` config
- Runners now warn and log when edge counts exceed threshold

## Tests added/updated

- New: `tests/component2/test_anchor_adapter_factkg.py`
- Updated:
  - `tests/component2/test_selective.py`
  - `tests/component2/test_run_m1.py`
  - `tests/component2/test_run_m2.py`
  - `tests/component2/test_run_m3.py`
- Full suite status: 52 tests passing.

## Runtime snapshot examples with new config fields

- `logs/component2/m1/20260210_104934_026305/run_config.json`
- `logs/component2/m2/20260210_104939_838090/run_config.json`
- `logs/component2/m3/20260210_105002_449038/run_config.json`

## Anchor default confirmation run

Using Component 1-style claims with `entity_set` removed:
- `logs/component2/m3/20260210_105002_449038/summary.json`
- observed `anchor_source_counts = {"pickle": 5, "provided": 0, "heuristic": 0}`
- observed `claims_missing_pickle_mapping = 0`
