# AUTOPILOT REPORT

## Resume / Partial Run
- Resume latest state: `bash scripts/autopilot_component3.sh --env bert_factcheck --resume`
- Run one stage only: `bash scripts/autopilot_component3.sh --env bert_factcheck --only <stage>`

## Chronological Log
- Iteration 1: running `stage0_preflight`.
  - PASS: stage=`stage0_preflight` run_id=`autopilot_stage0_preflight_i1` gate=`runs/PREFLIGHT_REPORT.json`
- Iteration 2: running `stage1_overfit200`.
  - FAIL: stage=`stage1_overfit200` run_id=`autopilot_stage1_overfit200_i2` patch_retry=1
- Iteration 3: running `stage1_overfit200`.
  - FAIL: stage=`stage1_overfit200` run_id=`autopilot_stage1_overfit200_i3` patch_retry=2
- Iteration 4: running `stage1_overfit200`.
  - FAIL: stage=`stage1_overfit200` run_id=`autopilot_stage1_overfit200_i4` patch_retry=3
- Iteration 5: running `stage1_overfit200`.
  - FAIL: stage=`stage1_overfit200` exhausted max patch retries (3).

## Final State
```json
{
  "schema_version": 1,
  "args": {
    "env": "bert_factcheck",
    "max_iters": 5,
    "resume": 0,
    "only": null
  },
  "stages": {
    "stage0_preflight": {
      "status": "pass",
      "attempts": 1,
      "patch_retries": 0,
      "last_run_id": "autopilot_stage0_preflight_i1",
      "last_gate_path": "runs/PREFLIGHT_REPORT.json",
      "last_error": ""
    },
    "stage1_overfit200": {
      "status": "fail",
      "attempts": 4,
      "patch_retries": 3,
      "last_run_id": "autopilot_stage1_overfit200_i5",
      "last_gate_path": "runs/autopilot_stage1_overfit200_i5/gate_stage1.json",
      "last_error": "stage_exit=0;gate_ok=0"
    },
    "stage2_claim_only": {
      "status": "pending",
      "attempts": 0,
      "patch_retries": 0,
      "last_run_id": "",
      "last_gate_path": "",
      "last_error": ""
    },
    "stage3_pvqagnn_no_bt": {
      "status": "pending",
      "attempts": 0,
      "patch_retries": 0,
      "last_run_id": "",
      "last_gate_path": "",
      "last_error": ""
    },
    "stage4_bt_helping": {
      "status": "pending",
      "attempts": 0,
      "patch_retries": 0,
      "last_run_id": "",
      "last_gate_path": "",
      "last_error": ""
    },
    "stage5_readiness": {
      "status": "pending",
      "attempts": 0,
      "patch_retries": 0,
      "last_run_id": "",
      "last_gate_path": "",
      "last_error": ""
    }
  }
}
```
