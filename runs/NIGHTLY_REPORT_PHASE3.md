# NIGHTLY REPORT: Phase-3 Medium A/B

## Run Context
- Date: 2026-02-16
- Branch: `fix/component1-v4-hardening`
- Conda env: `bert_factcheck`
- Sessions: `t33_p3_medA_noBT`, `t33_p3_medB_BT` (completed)

## Preflight Gate (Updated Rule)
- Source: `runs/t33_phase3_gate_precheck.json`
- Hard-fail fields all zero across train/val/test: `claims_using_fallback`, `missing_entity_embeddings_total`, `missing_pv_fields`, `missing_pool_labels`, `malformed_probs`, `missing_rel`, `missing_schema_version`, `non_v2_schema`.
- Empty evidence sentinel allowance applied: all sentinel claims map to zero-triple subgraphs (`sentinel_nonzero_subgraph_claims=0` for train/val/test).
- train: sentinel=1126, zero-triple=1126, nonzero-triple=0
- val: sentinel=125, zero-triple=125, nonzero-triple=0
- test: sentinel=108, zero-triple=108, nonzero-triple=0

## Commands Executed
```bash
tmux new-session -d -s t33_p3_medA_noBT 'bash -lc "cd \"/home/bs_thesis/shift (Copy)/Fact-or-Fiction\" && source ~/miniconda/etc/profile.d/conda.sh && conda activate bert_factcheck && export PYTHONPATH=src:. && mkdir -p runs/t33_p3_mediumA_no_bt && python src/component3/run_train.py --run-id t33_p3_mediumA_no_bt --seed 42 --use-component1-pairs --component1-logs-root logs/component1 --factkg-require-pv-metadata --disable-backtracking --train-subset-size 10000 --val-subset-size 2000 --stage1-epochs 2 --stage2-epochs 1 2>&1 | tee runs/t33_p3_mediumA_no_bt/run_console.log"'
tmux new-session -d -s t33_p3_medB_BT 'bash -lc "cd \"/home/bs_thesis/shift (Copy)/Fact-or-Fiction\" && source ~/miniconda/etc/profile.d/conda.sh && conda activate bert_factcheck && export PYTHONPATH=src:. && mkdir -p runs/t33_p3_mediumB_bt && python src/component3/run_train.py --run-id t33_p3_mediumB_bt --seed 42 --use-component1-pairs --component1-logs-root logs/component1 --factkg-require-pv-metadata --enable-backtracking --backtracking-directional-delta 0.0 --backtracking-promotion-gamma 2.0 --backtracking-flip-conf-min 0.1 --train-subset-size 10000 --val-subset-size 2000 --stage1-epochs 2 --stage2-epochs 1 2>&1 | tee runs/t33_p3_mediumB_bt/run_console.log"'
conda run -n bert_factcheck python scripts/analyze_backtracking_effectiveness.py --run-dir runs/t33_p3_mediumB_bt --useful-delta 0.002 --out-json runs/t33_p3_mediumB_bt/backtracking_effectiveness_delta_0p002.json
conda run -n bert_factcheck python scripts/analyze_backtracking_effectiveness.py --run-dir runs/t33_p3_mediumB_bt --useful-delta 0.01 --out-json runs/t33_p3_mediumB_bt/backtracking_effectiveness_delta_0p01.json
```

## Artifact Check
- `t33_p3_mediumA_no_bt`: `metrics.json`, `predictions.jsonl`, `recovery_actions.jsonl`, `recovery_candidates.jsonl`, `run_console.log` present.
- `t33_p3_mediumB_bt`: `metrics.json`, `predictions.jsonl`, `recovery_actions.jsonl`, `recovery_candidates.jsonl`, `run_console.log` present.
- BT analysis outputs: `runs/t33_p3_mediumB_bt/backtracking_effectiveness_delta_0p002.json`, `runs/t33_p3_mediumB_bt/backtracking_effectiveness_delta_0p01.json`.

## A/B Metrics Diff
- A (no-BT) overall accuracy: 0.512001
- B (BT) overall accuracy: 0.512222
- Delta (B-A): +0.000221 (+0.0221 pp)

| Slice | A acc | B acc | Delta (B-A) |
|---|---:|---:|---:|
| existence | 0.516551 | 0.516551 | +0.000000 (+0.0000 pp) |
| substitution | 0.777688 | 0.778498 | +0.000810 (+0.0810 pp) |
| multi hop | 0.491076 | 0.490593 | -0.000482 (-0.0482 pp) |
| multi claim | 0.550562 | 0.551169 | +0.000607 (+0.0607 pp) |
| negation | 0.528158 | 0.527397 | -0.000761 (-0.0761 pp) |
| single hop | 0.518226 | 0.518657 | +0.000431 (+0.0431 pp) |

## Prediction Change Analysis
- Compared claims: 9041
- Changed predictions (A vs B): 6
- Improved (A wrong -> B correct): 4
- Worsened (A correct -> B wrong): 2
- Net improved: 2
- Negation+Substitution combined accuracy A: 0.724375
- Negation+Substitution combined accuracy B: 0.724792
- Negation+Substitution delta (B-A): +0.000417 (+0.0417 pp)

## Backtracking Effectiveness (BT Run)
- Source: `runs/t33_p3_mediumB_bt/backtracking_effectiveness_delta_0p002.json` and `..._0p01.json`
- Trigger rate: 0.204181 (1846/9041)
- Wrong-before-triggered rate: 0.388949 (718/1846)
- flip_to_correct_claims: 4
- tentative_flip_to_correct_total: 6
- rounds_reverted_rate: 0.722484 (1687/2335)
- Useful recovery rate @delta>=0.002: 0.284399
- Useful recovery rate @delta>=0.01: 0.002167

## Pass/Fail Gates
- Gate 1 overall delta >= +0.3 pp: FAIL (observed +0.0221 pp)
- Gate 2 (negation+substitution) delta >= +0.5 pp: FAIL (observed +0.0417 pp)
- Gate 3 flip_to_correct_claims >= 20: FAIL (observed 4)
- **Phase-3 Verdict: FAIL**

## Additional Notes
- Both runs raised `RuntimeError: No-collapse gate failed` at process end, but all required run artifacts were written and were used for analysis.
- no-collapse gate A: False (best_val_accuracy=50.85)
- no-collapse gate B: False (best_val_accuracy=50.85)
- Phase-4 was not attempted (blocked by Phase-3 FAIL gates).

## Next Recommended Patch List (Phase-3 FAIL)
1. Implement Phase-4 encoder tuning with VRAM-safe default: `--encoder_tuning lora` (rank=8, alpha=16, dropout=0.05) on attention projections only.
2. Add alternative `--encoder_tuning unfreeze_last2` path and compare against LoRA on medium val.
3. Add logit calibration pass (temperature scaling on val) before backtracking triggering to improve margin reliability.
4. Re-run medium A/B with chosen tuning mode and keep backtracking knobs frozen.
5. If medium gates pass, then run full-seed parity suite for paper-ready claims.
