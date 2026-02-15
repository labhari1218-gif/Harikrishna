# Component 2 Ablation Study (A0-A4)

This document describes the M4 ablation runner for Component 2 and its baseline comparison.

## Ablations

- `A0` Baseline (no Component 2)
  - Uses a deterministic pre-Component2 baseline vote over Component1 A/C evidence masses.
  - Outputs claim predictions, confidence, and standard metrics.

- `A1` Component 2 without bridge scoring and without recovery
  - Uses PV masks + dual-stream reasoner.
  - Reasons on graph built from `A ∪ C` only.
  - No PPR/BridgeBonus and no `S -> A` recovery.

- `A2` Component 2 with bridge scoring and without recovery
  - Same as `A1`, but computes BridgeBonus' over `S` using PPR on `A ∪ S ∪ C`.
  - Logs bridge statistics and bridge correlations.
  - No `S -> A` recovery.

- `A3` Component 2 full
  - Same as `A2`, plus one-step `S -> A` recovery.
  - Recovers top-`k` S edges by bridge policy only when:
    - `ESI` is low (`esi_geom < tau_esi`) and
    - `delta_conn_bridge@k > 0` (equivalently bridge RPI gain > 0).
  - Re-runs reasoner once after recovery and logs before/after diagnostics.

- `A4` Component 2 full without PV masks (T11)
  - Same as `A3`, but disables PV masks by forcing `mask_sup = mask_ref = 1.0` for all edges.
  - Isolates the value of NLI-informed PV masking while keeping bridge/rescue behavior unchanged.

## Selective Prediction Modes

Each ablation reports:

- `S1` Sweep mode (`tau_abstain=None`): risk-coverage curve and `AURC`.
- `S2` Fixed operating point (`tau_abstain=0.60` default):
  - `coverage`
  - `risk`
  - `accuracy_on_answered`
  - `abstain_rate`

## Key Metrics

- `accuracy`: fraction of correct labeled predictions.
- `macro_f1`: class-balanced F1 over supported/refuted.
- `risk-coverage`: error rate among answered predictions as coverage varies.
- `AURC`: area under risk-coverage curve (lower is better).
- `delta_conn_bridge@k`: predicted connectivity gain from bridge-selected S edges.

## Run Command

Run all A0-A4 ablations on a split:

```bash
PYTHONPATH=src python3 scripts/run_ablation_component2.py \
  --split val \
  --claims_jsonl logs/component1/val/claims.jsonl \
  --n_claims 2000 \
  --seed 13 \
  --tau_abstain 0.60
```

Run only one mode (example `A2`):

```bash
PYTHONPATH=src python3 scripts/run_ablation_component2.py \
  --split val \
  --claims_jsonl logs/component1/val/claims.jsonl \
  --n_claims 2000 \
  --seed 13 \
  --ablation_ids A2
```

## Outputs

Each run creates:

- `logs/ablations/component2/<run_id>/run_config.json`
- `logs/ablations/component2/<run_id>/selected_claim_ids.txt`
- `logs/ablations/component2/<run_id>/predictions_A{0..4}.jsonl`
- `logs/ablations/component2/<run_id>/metrics_A{0..4}.json`
- `logs/ablations/component2/<run_id>/summary_table.md`
- `logs/ablations/component2/<run_id>/plots/`
  - risk-coverage
  - accuracy-vs-coverage
  - bridgebonus histogram (`A2`,`A3`)
  - delta-conn histogram (`A3`)

If matplotlib is unavailable, CSV artifacts are saved in `plots/` along with `plot_from_csv.py`.
