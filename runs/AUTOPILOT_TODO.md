# AUTOPILOT TODO

- Canonical command: `bash scripts/autopilot_component3.sh --env bert_factcheck --max-iters 5`
- Environment: `bert_factcheck`
- Source of truth: this file is the authoritative checklist and stage log.

## [x] stage0_preflight
- Objective: Validate strict PV metadata, sentinel exception policy, and embedding/cache prerequisites before training.
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage0_preflight --max-iters 1
```
- Expected artifacts: runs/PREFLIGHT_REPORT.json; runs/AUTOPILOT_STATE.json; runs/AUTOPILOT_REPORT.md; runs/autopilot_stage0_preflight_i*/run_console.log
- Gate pass condition: Safety Gate: strict PV metadata clean OR sentinel-only-zero-triple exception confirmed (with embeddings/cache present).
- Notes / Fixes applied:
- None yet.

## [ ] stage1_overfit200
- Objective: Prove the baseline can memorize a 200-sample subset (sanity check for labels, grads, and optimizer flow).
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage1_overfit200 --max-iters 1
```
- Expected artifacts: runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/gate_stage1.json
- Gate pass condition: G1 Overfit200: train_acc >= 0.95
- Notes / Fixes applied:
- Blocker: retry profiles did not change overfit command knobs in practice (`--stage1-epochs` remained 25); likely env propagation gap across tmux/conda wrapper (`scripts/autopilot_component3.sh:560`, `scripts/run_tmux_job.sh:63`).
- FAIL gate=G1_FAIL_train_acc=0.530000<0.95; hypothesis=Learning wiring issue (labels/grad flow/trainable masks); inspect datasets.py:327 and src/component3/run_train.py:1876.
- Patch retry #3: apply retry profile #3 for stage1_overfit200 and rerun same stage
- FAIL gate=G1_FAIL_train_acc=0.530000<0.95; hypothesis=Learning wiring issue (labels/grad flow/trainable masks); inspect datasets.py:327 and src/component3/run_train.py:1876.
- Patch retry #2: apply retry profile #2 for stage1_overfit200 and rerun same stage
- FAIL gate=G1_FAIL_train_acc=0.530000<0.95; hypothesis=Learning wiring issue (labels/grad flow/trainable masks); inspect datasets.py:327 and src/component3/run_train.py:1876.
- Patch retry #1: apply retry profile #1 for stage1_overfit200 and rerun same stage
- FAIL gate=G1_FAIL_train_acc=0.530000<0.95; hypothesis=Learning wiring issue (labels/grad flow/trainable masks); inspect datasets.py:327 and src/component3/run_train.py:1876.

## [ ] stage2_claim_only
- Objective: Validate claim-only learning path and metric parity before graph/backtracking effects are considered.
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage2_claim_only --max-iters 1
```
- Expected artifacts: runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/gate_stage2.json
- Gate pass condition: G2 ClaimOnlyMedium: test_acc >= 0.60
- Notes / Fixes applied:
- None yet.

## [ ] stage3_pvqagnn_no_bt
- Objective: Raise no-backtracking PV-QAGNN baseline accuracy out of near-chance territory with encoder tuning.
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage3_pvqagnn_no_bt --max-iters 1
```
- Expected artifacts: runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/encoder_tune_selected.txt; runs/<run_id>/gate_stage3.json
- Gate pass condition: G3 PVQAGNN_NoBT_Medium: test_acc >= 0.65
- Notes / Fixes applied:
- None yet.

## [ ] stage4_bt_helping
- Objective: Measure BT impact only after baseline passes, with BT logic frozen and A/B comparison.
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage4_bt_helping --max-iters 1
```
- Expected artifacts: runs/<run_a>/config.yaml; runs/<run_a>/metrics.json; runs/<run_a>/predictions.jsonl; runs/<run_b>/config.yaml; runs/<run_b>/metrics.json; runs/<run_b>/predictions.jsonl; runs/<run_b>/recovery_actions.jsonl; runs/<run_b>/recovery_candidates.jsonl; runs/<run_b>/backtracking_effectiveness_0p002.json; runs/<run_b>/gate_stage4.json
- Gate pass condition: G4 BT_Helping_Medium: delta_acc >= 0.001 OR (flip_to_correct_claims >= 10 AND worsened <= improved)
- Notes / Fixes applied:
- None yet.

## [ ] stage5_readiness
- Objective: Emit full-run readiness plan (3 seeds, estimated compute) without launching full training.
- Exact command:
```bash
bash scripts/autopilot_component3.sh --env bert_factcheck --only stage5_readiness --max-iters 1
```
- Expected artifacts: runs/autopilot_stage5_readiness/stage5_readiness.json; runs/AUTOPILOT_REPORT.md; runs/AUTOPILOT_STATE.json
- Gate pass condition: Readiness artifact exists and includes 3-seed full-run proposal.
- Notes / Fixes applied:
- None yet.

## Execution Log
- iteration=5 stage=stage1_overfit200 status=FAIL retries_exhausted=3 run_id=autopilot_stage1_overfit200_i5 gate=runs/autopilot_stage1_overfit200_i5/gate_stage1.json
- iteration=5 stage=stage1_overfit200 status=running
- iteration=4 stage=stage1_overfit200 status=FAIL patch_retry=3 run_id=autopilot_stage1_overfit200_i4 gate=runs/autopilot_stage1_overfit200_i4/gate_stage1.json
- iteration=4 stage=stage1_overfit200 status=running
- iteration=3 stage=stage1_overfit200 status=FAIL patch_retry=2 run_id=autopilot_stage1_overfit200_i3 gate=runs/autopilot_stage1_overfit200_i3/gate_stage1.json
- iteration=3 stage=stage1_overfit200 status=running
- iteration=2 stage=stage1_overfit200 status=FAIL patch_retry=1 run_id=autopilot_stage1_overfit200_i2 gate=runs/autopilot_stage1_overfit200_i2/gate_stage1.json
- iteration=2 stage=stage1_overfit200 status=running
- iteration=1 stage=stage0_preflight status=PASS run_id=autopilot_stage0_preflight_i1 gate=runs/PREFLIGHT_REPORT.json
- iteration=1 stage=stage0_preflight status=running
