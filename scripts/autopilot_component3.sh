#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source scripts/run_tmux_job.sh

ENV_NAME="bert_factcheck"
MAX_ITERS=5
RESUME=0
ONLY_STAGE=""

STAGES=(
  stage0_preflight
  stage1_overfit200
  stage2_claim_only
  stage3_pvqagnn_no_bt
  stage4_bt_helping
  stage5_readiness
)

STATE_FILE="runs/AUTOPILOT_STATE.json"
REPORT_FILE="runs/AUTOPILOT_REPORT.md"
TODO_FILE="runs/AUTOPILOT_TODO.md"
mkdir -p runs
export TMPDIR="${REPO_ROOT}/runs/.tmp"
mkdir -p "${TMPDIR}"

usage() {
  cat <<USAGE
Usage: $0 [--env <conda_env>] [--max-iters <N>] [--resume] [--only <stage>]

Stages:
  stage0_preflight
  stage1_overfit200
  stage2_claim_only
  stage3_pvqagnn_no_bt
  stage4_bt_helping
  stage5_readiness
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --max-iters)
      MAX_ITERS="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --only)
      ONLY_STAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "${MAX_ITERS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ITERS}" -lt 1 ]]; then
  echo "--max-iters must be a positive integer" >&2
  exit 2
fi

if [[ -n "${ONLY_STAGE}" ]]; then
  found=0
  for s in "${STAGES[@]}"; do
    if [[ "${s}" == "${ONLY_STAGE}" ]]; then
      found=1
      break
    fi
  done
  if [[ "${found}" -ne 1 ]]; then
    echo "Invalid --only stage: ${ONLY_STAGE}" >&2
    exit 2
  fi
fi

init_state() {
  AUTOPILOT_ENV_NAME="${ENV_NAME}" \
  AUTOPILOT_MAX_ITERS="${MAX_ITERS}" \
  AUTOPILOT_RESUME="${RESUME}" \
  AUTOPILOT_ONLY_STAGE="${ONLY_STAGE}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

stages = [
    "stage0_preflight",
    "stage1_overfit200",
    "stage2_claim_only",
    "stage3_pvqagnn_no_bt",
    "stage4_bt_helping",
    "stage5_readiness",
]
payload = {
    "schema_version": 1,
    "args": {
        "env": os.environ["AUTOPILOT_ENV_NAME"],
        "max_iters": int(os.environ["AUTOPILOT_MAX_ITERS"]),
        "resume": int(os.environ["AUTOPILOT_RESUME"]),
        "only": os.environ["AUTOPILOT_ONLY_STAGE"] or None,
    },
    "stages": {},
}
for stage in stages:
    payload["stages"][stage] = {
        "status": "pending",
        "attempts": 0,
        "patch_retries": 0,
        "last_run_id": "",
        "last_gate_path": "",
        "last_error": "",
    }
Path("runs/AUTOPILOT_STATE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

init_report() {
  cat > "${REPORT_FILE}" <<'MD'
# AUTOPILOT REPORT

## Resume / Partial Run
- Resume latest state: `bash scripts/autopilot_component3.sh --env bert_factcheck --resume`
- Run one stage only: `bash scripts/autopilot_component3.sh --env bert_factcheck --only <stage>`

## Chronological Log
MD
}

append_report() {
  printf "%s\n" "$1" >> "${REPORT_FILE}"
}

stage_objective() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      echo "Validate strict PV metadata, sentinel exception policy, and embedding/cache prerequisites before training."
      ;;
    stage1_overfit200)
      echo "Prove the baseline can memorize a 200-sample subset (sanity check for labels, grads, and optimizer flow)."
      ;;
    stage2_claim_only)
      echo "Validate claim-only learning path and metric parity before graph/backtracking effects are considered."
      ;;
    stage3_pvqagnn_no_bt)
      echo "Raise no-backtracking PV-QAGNN baseline accuracy out of near-chance territory with encoder tuning."
      ;;
    stage4_bt_helping)
      echo "Measure BT impact only after baseline passes, with BT logic frozen and A/B comparison."
      ;;
    stage5_readiness)
      echo "Emit full-run readiness plan (3 seeds, estimated compute) without launching full training."
      ;;
    *)
      echo "Unknown stage objective."
      ;;
  esac
}

stage_exact_command() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage0_preflight --max-iters 1"
      ;;
    stage1_overfit200)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage1_overfit200 --max-iters 1"
      ;;
    stage2_claim_only)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage2_claim_only --max-iters 1"
      ;;
    stage3_pvqagnn_no_bt)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage3_pvqagnn_no_bt --max-iters 1"
      ;;
    stage4_bt_helping)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage4_bt_helping --max-iters 1"
      ;;
    stage5_readiness)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --only stage5_readiness --max-iters 1"
      ;;
    *)
      echo "bash scripts/autopilot_component3.sh --env bert_factcheck --max-iters 5"
      ;;
  esac
}

stage_expected_artifacts() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      echo "runs/PREFLIGHT_REPORT.json; runs/AUTOPILOT_STATE.json; runs/AUTOPILOT_REPORT.md; runs/autopilot_stage0_preflight_i*/run_console.log"
      ;;
    stage1_overfit200)
      echo "runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/gate_stage1.json"
      ;;
    stage2_claim_only)
      echo "runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/gate_stage2.json"
      ;;
    stage3_pvqagnn_no_bt)
      echo "runs/<run_id>/config.yaml; runs/<run_id>/metrics.json; runs/<run_id>/predictions.jsonl; runs/<run_id>/run_console.log; runs/<run_id>/encoder_tune_selected.txt; runs/<run_id>/gate_stage3.json"
      ;;
    stage4_bt_helping)
      echo "runs/<run_a>/config.yaml; runs/<run_a>/metrics.json; runs/<run_a>/predictions.jsonl; runs/<run_b>/config.yaml; runs/<run_b>/metrics.json; runs/<run_b>/predictions.jsonl; runs/<run_b>/recovery_actions.jsonl; runs/<run_b>/recovery_candidates.jsonl; runs/<run_b>/backtracking_effectiveness_0p002.json; runs/<run_b>/gate_stage4.json"
      ;;
    stage5_readiness)
      echo "runs/autopilot_stage5_readiness/stage5_readiness.json; runs/AUTOPILOT_REPORT.md; runs/AUTOPILOT_STATE.json"
      ;;
    *)
      echo "None"
      ;;
  esac
}

stage_gate_condition() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      echo "Safety Gate: strict PV metadata clean OR sentinel-only-zero-triple exception confirmed (with embeddings/cache present)."
      ;;
    stage1_overfit200)
      echo "G1 Overfit200: train_acc >= 0.95"
      ;;
    stage2_claim_only)
      echo "G2 ClaimOnlyMedium: test_acc >= 0.60"
      ;;
    stage3_pvqagnn_no_bt)
      echo "G3 PVQAGNN_NoBT_Medium: test_acc >= 0.65"
      ;;
    stage4_bt_helping)
      echo "G4 BT_Helping_Medium: delta_acc >= 0.001 OR (flip_to_correct_claims >= 10 AND worsened <= improved)"
      ;;
    stage5_readiness)
      echo "Readiness artifact exists and includes 3-seed full-run proposal."
      ;;
    *)
      echo "Unknown gate condition."
      ;;
  esac
}

stage_failure_hypothesis() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      echo "Coverage strictness or sentinel validation mismatch; inspect scripts/autopilot_gates.py:148 and src/component3/c1_pairs_loader.py:225."
      ;;
    stage1_overfit200)
      echo "Learning wiring issue (labels/grad flow/trainable masks); inspect datasets.py:327 and src/component3/run_train.py:1876."
      ;;
    stage2_claim_only)
      echo "Claim-only metric/label path mismatch; inspect src/component3/run_train.py:646 and evaluate.py:9."
      ;;
    stage3_pvqagnn_no_bt)
      echo "Encoder tune path not taking effect; inspect src/component3/run_train.py:686 and src/component3/run_train.py:727."
      ;;
    stage4_bt_helping)
      echo "Delta/flip accounting mismatch despite BT artifacts; inspect scripts/autopilot_gates.py:390 and scripts/analyze_backtracking_effectiveness.py:42."
      ;;
    stage5_readiness)
      echo "Readiness artifact generation failed; inspect scripts/autopilot_component3.sh:438."
      ;;
    *)
      echo "Unknown failure hypothesis."
      ;;
  esac
}

init_todo() {
  {
    cat <<'MD'
# AUTOPILOT TODO

- Canonical command: `bash scripts/autopilot_component3.sh --env bert_factcheck --max-iters 5`
- Environment: `bert_factcheck`
- Source of truth: this file is the authoritative checklist and stage log.

MD
    local stage
    for stage in "${STAGES[@]}"; do
      cat <<MD
## [ ] ${stage}
- Objective: $(stage_objective "${stage}")
- Exact command:
\`\`\`bash
$(stage_exact_command "${stage}")
\`\`\`
- Expected artifacts: $(stage_expected_artifacts "${stage}")
- Gate pass condition: $(stage_gate_condition "${stage}")
- Notes / Fixes applied:
- None yet.

MD
    done
    cat <<'MD'
## Execution Log
- None yet.
MD
  } > "${TODO_FILE}"
}

set_todo_stage_checkbox() {
  local stage="$1"
  local checked="$2"
  python3 - "${TODO_FILE}" "${stage}" "${checked}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
checked = sys.argv[3] == "1"
text = path.read_text(encoding="utf-8")
target = f"## [{'x' if checked else ' '}] {stage}"
pattern = re.compile(rf"^## \[(?: |x)\] {re.escape(stage)}$", re.MULTILINE)
text_new, count = pattern.subn(target, text, count=1)
if count == 0:
    text_new = text.rstrip() + f"\n\n{target}\n"
path.write_text(text_new, encoding="utf-8")
PY
}

append_todo_stage_note() {
  local stage="$1"
  local note="$2"
  python3 - "${TODO_FILE}" "${stage}" "${note}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
note = sys.argv[3]
text = path.read_text(encoding="utf-8")
header = re.compile(rf"^## \[(?: |x)\] {re.escape(stage)}$", re.MULTILINE)
match = header.search(text)
if match is None:
    path.write_text(text.rstrip() + f"\n- {stage}: {note}\n", encoding="utf-8")
    sys.exit(0)

section_start = match.start()
next_header = re.search(r"^## \[(?: |x)\] stage[0-9]_", text[match.end():], re.MULTILINE)
if next_header is None:
    exec_log = re.search(r"^## Execution Log$", text, re.MULTILINE)
    section_end = exec_log.start() if exec_log else len(text)
else:
    section_end = match.end() + next_header.start()

section = text[section_start:section_end]
needle = "- Notes / Fixes applied:\n"
if needle not in section:
    section = section.rstrip() + "\n" + needle

idx = section.index(needle) + len(needle)
tail = section[idx:]
if tail.startswith("- None yet.\n"):
    tail = tail[len("- None yet.\n"):]
elif tail.strip() == "- None yet.":
    tail = ""
line = f"- {note}\n"
section_new = section[:idx] + line + tail
text_new = text[:section_start] + section_new + text[section_end:]
path.write_text(text_new, encoding="utf-8")
PY
}

append_todo_execution_log() {
  local line="$1"
  python3 - "${TODO_FILE}" "${line}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
line = sys.argv[2]
text = path.read_text(encoding="utf-8")
marker = "## Execution Log\n"
if marker not in text:
    text = text.rstrip() + "\n\n## Execution Log\n"
idx = text.index(marker) + len(marker)
tail = text[idx:]
if tail.startswith("- None yet.\n"):
    tail = tail[len("- None yet.\n"):]
elif tail.strip() == "- None yet.":
    tail = ""
entry = f"- {line}\n"
text_new = text[:idx] + entry + tail
path.write_text(text_new, encoding="utf-8")
PY
}

if [[ "${RESUME}" -eq 0 || ! -f "${STATE_FILE}" ]]; then
  init_state
  init_report
  init_todo
else
  [[ -f "${REPORT_FILE}" ]] || init_report
  [[ -f "${TODO_FILE}" ]] || init_todo
fi

state_get() {
  local stage="$1"
  local key="$2"
  jq -r ".stages.\"${stage}\".${key}" "${STATE_FILE}"
}

state_set() {
  local stage="$1"
  local key="$2"
  local value="$3"
  local tmp
  tmp="$(mktemp)"
  jq ".stages.\"${stage}\".${key} = \"${value}\"" "${STATE_FILE}" > "${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

state_set_raw() {
  local stage="$1"
  local key="$2"
  local raw="$3"
  local tmp
  tmp="$(mktemp)"
  jq ".stages.\"${stage}\".${key} = ${raw}" "${STATE_FILE}" > "${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

state_inc() {
  local stage="$1"
  local key="$2"
  local tmp
  tmp="$(mktemp)"
  jq ".stages.\"${stage}\".${key} += 1" "${STATE_FILE}" > "${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

next_stage() {
  if [[ -n "${ONLY_STAGE}" ]]; then
    local st
    st="$(state_get "${ONLY_STAGE}" status)"
    if [[ "${st}" == "pass" ]]; then
      echo ""
    else
      echo "${ONLY_STAGE}"
    fi
    return
  fi
  local s
  for s in "${STAGES[@]}"; do
    local st
    st="$(state_get "${s}" status)"
    if [[ "${st}" != "pass" ]]; then
      echo "${s}"
      return
    fi
  done
  echo ""
}

# Retry profile knobs (auto-remediation without mutating label mapping).
export OVERFIT_STAGE1_EPOCHS="${OVERFIT_STAGE1_EPOCHS:-25}"
export OVERFIT_STAGE1_LR="${OVERFIT_STAGE1_LR:-2e-5}"
export CLAIM_ONLY_STAGE1_EPOCHS="${CLAIM_ONLY_STAGE1_EPOCHS:-4}"
export CLAIM_ONLY_STAGE2_EPOCHS="${CLAIM_ONLY_STAGE2_EPOCHS:-2}"
export PHASE4_STAGE1_EPOCHS="${PHASE4_STAGE1_EPOCHS:-3}"
export PHASE4_STAGE2_EPOCHS="${PHASE4_STAGE2_EPOCHS:-3}"
export MEDIUM_STAGE1_EPOCHS="${MEDIUM_STAGE1_EPOCHS:-3}"
export MEDIUM_STAGE2_EPOCHS="${MEDIUM_STAGE2_EPOCHS:-3}"

write_failure_report() {
  local stage="$1"
  local run_id="$2"
  local gate_path="$3"
  local patch_note="$4"
  local report_path="runs/${run_id}/FAILURE_REPORT.md"
  mkdir -p "runs/${run_id}"

  local suspects=""
  case "${stage}" in
    stage1_overfit200)
      suspects=$'- `datasets.py:327` label extraction path can silently collapse if data format shifts\n- `evaluate.py:9` binary decode boundary and threshold consistency\n- `src/component3/run_train.py:227` trainable-freeze policy may block learning when encoder tuning is active'
      ;;
    stage2_claim_only)
      suspects=$'- `src/component3/run_train.py:640` claim-only model path wiring\n- `evaluate.py:9` binary decode parity with training labels\n- `src/component3/run_train.py:1740` criterion selection for claim-only vs multitask'
      ;;
    stage3_pvqagnn_no_bt)
      suspects=$'- `src/component3/run_train.py:708` LoRA adapter injection path\n- `src/component3/run_train.py:760` unfreeze-last-N selector for encoder layers\n- `src/component3/run_train.py:1769` stage-level telemetry and optimizer progression'
      ;;
    stage4_bt_helping)
      suspects=$'- `scripts/analyze_backtracking_effectiveness.py:42` flip-to-correct extraction path\n- `scripts/autopilot_gates.py:390` BT gate delta/flip logic\n- `src/component3/run_train.py:1156` prediction and recovery action artifact completeness'
      ;;
    stage0_preflight)
      suspects=$'- `scripts/autopilot_gates.py:148` strict preflight coverage checks\n- `src/component3/c1_pairs_loader.py:225` fallback counters in loader coverage\n- `scripts/audit_component1_coverage.py:78` schema and sentinel health checks'
      ;;
    *)
      suspects='- Unknown stage'
      ;;
  esac

  {
    echo "# FAILURE REPORT"
    echo ""
    echo "- stage: \`${stage}\`"
    echo "- run_id: \`${run_id}\`"
    echo "- gate_report: \`${gate_path}\`"
    echo ""
    echo "## Top 3 Suspected Causes"
    echo "${suspects}"
    echo ""
    echo "## Minimal Patch Next"
    echo "- ${patch_note}"
  } > "${report_path}"
}

run_targeted_tests_for_stage() {
  local stage="$1"
  case "${stage}" in
    stage0_preflight)
      conda run --no-capture-output -n "${ENV_NAME}" python -m unittest discover -s tests/component3 -p 'test_autopilot_gates.py'
      ;;
    stage1_overfit200|stage2_claim_only|stage3_pvqagnn_no_bt)
      conda run --no-capture-output -n "${ENV_NAME}" python -m unittest discover -s tests/component3 -p 'test_run_train.py'
      ;;
    stage4_bt_helping)
      conda run --no-capture-output -n "${ENV_NAME}" python -m unittest discover -s tests/component3 -p 'test_analyze_backtracking_effectiveness.py'
      ;;
    *)
      ;;
  esac
}

apply_patch_profile() {
  local stage="$1"
  local retries="$2"
  case "${stage}" in
    stage1_overfit200)
      if [[ "${retries}" -eq 0 ]]; then
        export OVERFIT_STAGE1_EPOCHS=35
      elif [[ "${retries}" -eq 1 ]]; then
        export OVERFIT_STAGE1_EPOCHS=45
        export OVERFIT_STAGE1_LR=3e-5
      else
        export OVERFIT_STAGE1_EPOCHS=60
      fi
      ;;
    stage2_claim_only)
      if [[ "${retries}" -eq 0 ]]; then
        export CLAIM_ONLY_STAGE1_EPOCHS=6
        export CLAIM_ONLY_STAGE2_EPOCHS=3
      elif [[ "${retries}" -eq 1 ]]; then
        export CLAIM_ONLY_STAGE1_EPOCHS=8
        export CLAIM_ONLY_STAGE2_EPOCHS=4
      else
        export CLAIM_ONLY_STAGE1_EPOCHS=10
        export CLAIM_ONLY_STAGE2_EPOCHS=5
      fi
      ;;
    stage3_pvqagnn_no_bt)
      if [[ "${retries}" -eq 0 ]]; then
        export PHASE4_STAGE1_EPOCHS=4
        export PHASE4_STAGE2_EPOCHS=4
      elif [[ "${retries}" -eq 1 ]]; then
        export PHASE4_STAGE1_EPOCHS=5
        export PHASE4_STAGE2_EPOCHS=5
      else
        export PHASE4_STAGE1_EPOCHS=6
        export PHASE4_STAGE2_EPOCHS=6
      fi
      ;;
    stage4_bt_helping)
      if [[ "${retries}" -eq 0 ]]; then
        export MEDIUM_STAGE1_EPOCHS=4
        export MEDIUM_STAGE2_EPOCHS=4
      elif [[ "${retries}" -eq 1 ]]; then
        export MEDIUM_STAGE1_EPOCHS=5
        export MEDIUM_STAGE2_EPOCHS=5
      else
        export MEDIUM_STAGE1_EPOCHS=6
        export MEDIUM_STAGE2_EPOCHS=6
      fi
      ;;
    *)
      ;;
  esac
}

stage3_selected_tune="lora"
stage3_selected_unfreeze_n="2"

run_stage() {
  local stage="$1"
  local iter_idx="$2"

  local run_id=""
  local gate_path=""

  case "${stage}" in
    stage0_preflight)
      local session="autopilot_stage0_preflight_i${iter_idx}"
      run_tmux_job "${session}" "${ENV_NAME}" \
        python scripts/autopilot_gates.py preflight \
          --logs-root logs/component1 \
          --subgraph-type direct_filled \
          --embeddings-path data/embeddings.pkl \
          --claim-triple-cache-path data/claim_triple_embeddings.pkl \
          --out-json runs/PREFLIGHT_REPORT.json >/dev/null
      wait_tmux_job "${session}" >/dev/null
      run_id="${session}"
      gate_path="runs/PREFLIGHT_REPORT.json"
      ;;

    stage1_overfit200)
      run_id="autopilot_stage1_overfit200_i${iter_idx}"
      local session="autopilot_stage1_i${iter_idx}"
      run_tmux_job "${session}" "${ENV_NAME}" \
        bash scripts/overfit_200.sh --run-id "${run_id}" >/dev/null
      wait_tmux_job "${session}" >/dev/null
      gate_path="runs/${run_id}/gate_stage1.json"
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/autopilot_gates.py gate \
          --stage stage1_overfit200 \
          --run-id "${run_id}" \
          --out-json "${gate_path}" >/dev/null
      ;;

    stage2_claim_only)
      run_id="autopilot_stage2_claim_only_i${iter_idx}"
      local session="autopilot_stage2_i${iter_idx}"
      run_tmux_job "${session}" "${ENV_NAME}" \
        bash scripts/claim_only_baseline.sh --run-id "${run_id}" >/dev/null
      wait_tmux_job "${session}" >/dev/null
      gate_path="runs/${run_id}/gate_stage2.json"
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/autopilot_gates.py gate \
          --stage stage2_claim_only \
          --run-id "${run_id}" \
          --out-json "${gate_path}" >/dev/null
      ;;

    stage3_pvqagnn_no_bt)
      run_id="autopilot_stage3_pvqagnn_no_bt_i${iter_idx}"
      local session="autopilot_stage3_i${iter_idx}"
      run_tmux_job "${session}" "${ENV_NAME}" \
        bash scripts/phase4_lora_or_unfreeze.sh --run-id "${run_id}" >/dev/null
      wait_tmux_job "${session}" >/dev/null
      if [[ -f "runs/${run_id}/encoder_tune_selected.txt" ]]; then
        stage3_selected_tune="$(tr -d '[:space:]' < "runs/${run_id}/encoder_tune_selected.txt")"
      fi
      if [[ "${stage3_selected_tune}" == "unfreeze_lastN" ]]; then
        stage3_selected_unfreeze_n="${PHASE4_UNFREEZE_LAST_N:-2}"
      fi
      gate_path="runs/${run_id}/gate_stage3.json"
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/autopilot_gates.py gate \
          --stage stage3_pvqagnn_no_bt \
          --run-id "${run_id}" \
          --out-json "${gate_path}" >/dev/null
      ;;

    stage4_bt_helping)
      local run_a_id="autopilot_stage4_mediumA_i${iter_idx}"
      local run_b_id="autopilot_stage4_mediumB_i${iter_idx}"
      run_id="${run_b_id}"
      local session="autopilot_stage4_i${iter_idx}"
      run_tmux_job "${session}" "${ENV_NAME}" \
        bash scripts/medium_ab_eval.sh \
          --run-a-id "${run_a_id}" \
          --run-b-id "${run_b_id}" \
          --encoder_tune "${stage3_selected_tune}" \
          --unfreeze_last_n "${stage3_selected_unfreeze_n}" \
          --lora_r "${PHASE4_LORA_R:-8}" \
          --lora_alpha "${PHASE4_LORA_ALPHA:-16}" \
          --lora_dropout "${PHASE4_LORA_DROPOUT:-0.05}" >/dev/null
      wait_tmux_job "${session}" >/dev/null
      gate_path="runs/${run_b_id}/gate_stage4.json"
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/autopilot_gates.py gate \
          --stage stage4_bt_helping \
          --run-a-id "${run_a_id}" \
          --run-b-id "${run_b_id}" \
          --out-json "${gate_path}" >/dev/null
      ;;

    stage5_readiness)
      run_id="autopilot_stage5_readiness"
      gate_path="runs/${run_id}/stage5_readiness.json"
      mkdir -p "runs/${run_id}"
      cat > "${gate_path}" <<JSON
{
  "mode": "stage5_readiness",
  "pass": true,
  "summary": {
    "full_run_plan": {
      "seeds": [42, 1337, 2026],
      "commands": [
        "python -m component3.run_train --run-id full_seed42 --seed 42 --train-subset-size 0 --val-subset-size 0 --stage1-epochs 5 --stage2-epochs 5 --disable-backtracking",
        "python -m component3.run_train --run-id full_seed1337 --seed 1337 --train-subset-size 0 --val-subset-size 0 --stage1-epochs 5 --stage2-epochs 5 --disable-backtracking",
        "python -m component3.run_train --run-id full_seed2026 --seed 2026 --train-subset-size 0 --val-subset-size 0 --stage1-epochs 5 --stage2-epochs 5 --disable-backtracking"
      ],
      "estimated_compute": {
        "gpu": "1x 8GB",
        "estimated_total_hours": "12-24"
      }
    }
  }
}
JSON
      ;;

    *)
      echo "Unsupported stage: ${stage}" >&2
      return 2
      ;;
  esac

  echo "${run_id}|${gate_path}"
}

gate_failure_reason() {
  local gate_path="$1"
  if [[ -z "${gate_path}" || ! -f "${gate_path}" ]]; then
    echo "gate_report_missing_or_unreadable"
    return
  fi
  python3 - <<PY
import json
from pathlib import Path
p = Path(${gate_path@Q})
try:
    data = json.loads(p.read_text())
except Exception:
    print("gate_report_parse_error")
    raise SystemExit(0)
reasons = data.get("reasons", [])
if isinstance(reasons, list) and reasons:
    print(str(reasons[0]))
else:
    print("gate_failed_without_reason")
PY
}

for ((iter=1; iter<=MAX_ITERS; iter++)); do
  stage="$(next_stage)"
  if [[ -z "${stage}" ]]; then
    append_report "- Iteration ${iter}: all requested stages are PASS."
    append_todo_execution_log "iteration=${iter} status=all_pass"
    break
  fi

  state_inc "${stage}" "attempts"
  state_set "${stage}" "status" "in_progress"
  append_report "- Iteration ${iter}: running \`${stage}\`."
  append_todo_execution_log "iteration=${iter} stage=${stage} status=running"

  if stage_result="$(run_stage "${stage}" "${iter}")"; then
    stage_status=0
  else
    stage_status=$?
  fi

  run_id="${stage_result%%|*}"
  gate_path="${stage_result#*|}"
  [[ "${gate_path}" == "${stage_result}" ]] && gate_path=""
  if [[ -z "${run_id}" ]]; then
    run_id="${stage}_i${iter}"
  fi

  gate_ok=0
  if [[ "${stage_status}" -eq 0 ]]; then
    if [[ -n "${gate_path}" && -f "${gate_path}" ]]; then
      gate_ok="$(python3 - <<PY
import json
from pathlib import Path
p=Path(${gate_path@Q})
try:
 d=json.loads(p.read_text())
 print(1 if bool(d.get('pass',False)) else 0)
except Exception:
 print(0)
PY
)"
    else
      gate_ok=0
    fi
  fi

  if [[ "${stage_status}" -eq 0 && "${gate_ok}" == "1" ]]; then
    state_set "${stage}" "status" "pass"
    state_set "${stage}" "last_run_id" "${run_id}"
    state_set "${stage}" "last_gate_path" "${gate_path}"
    state_set "${stage}" "last_error" ""
    set_todo_stage_checkbox "${stage}" "1"
    append_todo_execution_log "iteration=${iter} stage=${stage} status=PASS run_id=${run_id} gate=${gate_path}"
    append_report "  - PASS: stage=\`${stage}\` run_id=\`${run_id}\` gate=\`${gate_path}\`"
    if [[ "${stage}" == "stage5_readiness" ]]; then
      append_report ""
      append_report "## Stage 5 Readiness"
      append_report "- Full-run proposal: 3 seeds (`42, 1337, 2026`), full train/val/test, BT frozen unless explicitly requested."
      append_report "- Estimated compute envelope: `12-24` GPU-hours on a single 8GB GPU."
      append_report "- Command templates written to \`${gate_path}\`."
    fi
    continue
  fi

  state_set "${stage}" "status" "fail"
  state_set "${stage}" "last_run_id" "${run_id}"
  state_set "${stage}" "last_gate_path" "${gate_path}"
  state_set "${stage}" "last_error" "stage_exit=${stage_status};gate_ok=${gate_ok}"
  set_todo_stage_checkbox "${stage}" "0"

  retries="$(state_get "${stage}" patch_retries)"
  fail_reason="$(gate_failure_reason "${gate_path}")"
  hypothesis="$(stage_failure_hypothesis "${stage}")"
  append_todo_stage_note "${stage}" "FAIL gate=${fail_reason}; hypothesis=${hypothesis}"
  if [[ "${retries}" =~ ^[0-9]+$ ]] && [[ "${retries}" -ge 3 ]]; then
    append_todo_execution_log "iteration=${iter} stage=${stage} status=FAIL retries_exhausted=3 run_id=${run_id} gate=${gate_path}"
    append_report "  - FAIL: stage=\`${stage}\` exhausted max patch retries (3)."
    break
  fi

  patch_note="apply retry profile #$((retries + 1)) for ${stage} and rerun same stage"
  write_failure_report "${stage}" "${run_id}" "${gate_path}" "${patch_note}"
  apply_patch_profile "${stage}" "${retries}"
  run_targeted_tests_for_stage "${stage}"
  state_inc "${stage}" "patch_retries"
  append_todo_stage_note "${stage}" "Patch retry #$((retries + 1)): ${patch_note}"
  append_todo_execution_log "iteration=${iter} stage=${stage} status=FAIL patch_retry=$((retries + 1)) run_id=${run_id} gate=${gate_path}"
  append_report "  - FAIL: stage=\`${stage}\` run_id=\`${run_id}\` patch_retry=$((retries + 1))"

done

append_report ""
append_report "## Final State"
append_report '```json'
cat "${STATE_FILE}" >> "${REPORT_FILE}"
append_report '```'

echo "State: ${STATE_FILE}"
echo "Report: ${REPORT_FILE}"
echo "TODO: ${TODO_FILE}"
