#!/usr/bin/env bash
set -euo pipefail

RUN_A_ID=""
RUN_B_ID=""
ENV_NAME=""
ENCODER_TUNE="none"
UNFREEZE_LAST_N="2"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-a-id)
      RUN_A_ID="$2"
      shift 2
      ;;
    --run-b-id)
      RUN_B_ID="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --encoder_tune)
      ENCODER_TUNE="$2"
      shift 2
      ;;
    --unfreeze_last_n)
      UNFREEZE_LAST_N="$2"
      shift 2
      ;;
    --lora_r)
      LORA_R="$2"
      shift 2
      ;;
    --lora_alpha)
      LORA_ALPHA="$2"
      shift 2
      ;;
    --lora_dropout)
      LORA_DROPOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_A_ID}" || -z "${RUN_B_ID}" ]]; then
  echo "Usage: $0 --run-a-id <run_a> --run-b-id <run_b> [--env <conda_env>]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "runs/${RUN_A_ID}" "runs/${RUN_B_ID}"

TRAIN_SUBSET="${MEDIUM_TRAIN_SUBSET:-10000}"
VAL_SUBSET="${MEDIUM_VAL_SUBSET:-2000}"
STAGE1_EPOCHS="${MEDIUM_STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${MEDIUM_STAGE2_EPOCHS:-3}"

run_python() {
  if [[ -n "${ENV_NAME}" ]]; then
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
  else
    "$@"
  fi
}

common_args=(
  --dataset-name factkg
  --model-mode pv_qagnn
  --encoder_tune "${ENCODER_TUNE}"
  --unfreeze_last_n "${UNFREEZE_LAST_N}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --train-subset-size "${TRAIN_SUBSET}"
  --val-subset-size "${VAL_SUBSET}"
  --subset-sampling stratified
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --batch-size 8
  --max-seq-len 256
  --use-component1-pairs
  --component1-logs-root logs/component1
  --missing-pv-policy strict
  --factkg-require-pv-metadata
  --factkg-include-s-pool
  --factkg-require-claim-triple-cache
  --no-enforce-no-collapse-gate
)

set -o pipefail
run_python python -m component3.run_train \
  --run-id "${RUN_A_ID}" \
  "${common_args[@]}" \
  --disable-backtracking \
  2>&1 | tee "runs/${RUN_A_ID}/run_console.log"

run_python python -m component3.run_train \
  --run-id "${RUN_B_ID}" \
  "${common_args[@]}" \
  --enable-backtracking \
  --backtracking-challenge-mode \
  --backtracking-promotion-gamma 2.0 \
  --backtracking-directional-delta 0.0 \
  --backtracking-hunger-mode percentile \
  --backtracking-hunger-percentile 0.9 \
  2>&1 | tee "runs/${RUN_B_ID}/run_console.log"

run_python python scripts/analyze_backtracking_effectiveness.py \
  --run-dir "runs/${RUN_B_ID}" \
  --useful-delta 0.002 \
  --default-split test \
  --out-json "runs/${RUN_B_ID}/backtracking_effectiveness_0p002.json"

run_python python scripts/autopilot_gates.py artifacts --run-id "${RUN_A_ID}" >/dev/null
run_python python scripts/autopilot_gates.py artifacts --run-id "${RUN_B_ID}" --bt-enabled >/dev/null
