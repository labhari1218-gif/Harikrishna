# Medium A/B Clean Report (Phase 3)

## Runs
- A (no BT): `runs/t33_p3_mediumA_no_bt_clean`
- B (with BT): `runs/t33_p3_mediumB_bt_clean`
- Env: `bert_factcheck`
- Launch sessions: `t33_medA_noBT`, `t33_medB_BT`

## Artifact Validation
- `runs/t33_p3_mediumA_no_bt_clean/metrics.json`: present
- `runs/t33_p3_mediumA_no_bt_clean/predictions.jsonl`: present
- `runs/t33_p3_mediumB_bt_clean/metrics.json`: present
- `runs/t33_p3_mediumB_bt_clean/predictions.jsonl`: present
- `runs/t33_p3_mediumB_bt_clean/recovery_actions.jsonl`: present
- `runs/t33_p3_mediumB_bt_clean/recovery_candidates.jsonl`: present
- `runs/t33_p3_mediumB_bt_clean/backtracking_effectiveness_0p002.json`: present

## Runtime Flags
- A runtime_flags: `{'used_s_pool': True, 'used_backtracking': False, 'no_collapse_gate_enforced': False, 'backtracking_enabled': False, 'backtracking_triggered': False, 'backtracking_attempted': False, 'used_router': False, 'used_component5': False}`
- B runtime_flags: `{'used_s_pool': True, 'used_backtracking': True, 'no_collapse_gate_enforced': False, 'backtracking_enabled': True, 'backtracking_triggered': True, 'backtracking_attempted': True, 'used_router': False, 'used_component5': False}`

## Accuracy Delta (B - A)
- Overall A: `0.502931`
- Overall B: `0.502820`
- Overall delta: `-0.000111` (`-0.0111 pp`)

| Slice | A acc | B acc | Delta |
|---|---:|---:|---:|
| existence | 0.508083 | 0.507313 | -0.000770 (-0.0770 pp) |
| substitution | 0.781740 | 0.781199 | -0.000540 (-0.0540 pp) |
| multi hop | 0.458755 | 0.458755 | +0.000000 (+0.0000 pp) |
| multi claim | 0.545399 | 0.545399 | +0.000000 (+0.0000 pp) |
| negation | 0.559361 | 0.559361 | +0.000000 (+0.0000 pp) |
| single hop | 0.516073 | 0.515930 | -0.000144 (-0.0144 pp) |

## Prediction Diff
- Compared claims: `9041`
- Changed predictions: `11`
- Improved (A wrong -> B correct): `5`
- Worsened (A correct -> B wrong): `6`
- Net improved: `-1`

## Backtracking Effectiveness (`delta=0.002`)
- `flip_to_correct_claims`: `5`
- `trigger_rate`: `0.204513` (1849/9041)
- `accepted_rate`: `0.263021` (606/2304)
- `rounds_reverted_rate`: `0.736979`

## Gate Readout
- BT helping minimum gate (flip_to_correct>0 AND net acc delta>0 AND accepted flips exist): `False`
- Near-chance baseline check (~0.49): `True`
- Verdict: **baseline broken / near chance**. Move to Phase 4 (LoRA or unfreeze-last-N) before any further BT tuning.
- BT does **not** pass the minimum helping gate in this run pair.

## Notes
- Analyzer flag adapted to current script API: used `--useful-delta 0.002` (the script has no `--delta-threshold` flag).
- Both medium runs exited cleanly with subset no-collapse enforcement skipped as expected.
- Current `tmux ls`: `hari` only (A/B sessions finished).
