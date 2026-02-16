# FAILURE REPORT

- stage: `stage1_overfit200`
- run_id: `autopilot_stage1_overfit200_i3`
- gate_report: `runs/autopilot_stage1_overfit200_i3/gate_stage1.json`

## Top 3 Suspected Causes
- `datasets.py:327` label extraction path can silently collapse if data format shifts
- `evaluate.py:9` binary decode boundary and threshold consistency
- `src/component3/run_train.py:227` trainable-freeze policy may block learning when encoder tuning is active

## Minimal Patch Next
- apply retry profile #2 for stage1_overfit200 and rerun same stage
