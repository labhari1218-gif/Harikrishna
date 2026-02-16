#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
ENV_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 --run-id <run_id> [--env <conda_env>]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "runs/${RUN_ID}"

TRAIN_SUBSET="${PHASE4_TRAIN_SUBSET:-10000}"
VAL_SUBSET="${PHASE4_VAL_SUBSET:-2000}"
STAGE1_EPOCHS="${PHASE4_STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${PHASE4_STAGE2_EPOCHS:-3}"
UNFREEZE_LAST_N="${PHASE4_UNFREEZE_LAST_N:-2}"
LORA_R="${PHASE4_LORA_R:-8}"
LORA_ALPHA="${PHASE4_LORA_ALPHA:-16}"
LORA_DROPOUT="${PHASE4_LORA_DROPOUT:-0.05}"

run_python() {
  if [[ -n "${ENV_NAME}" ]]; then
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
  else
    "$@"
  fi
}

# LoRA dependency install step (requested behavior).
if ! run_python python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('peft') else 1)" >/dev/null 2>&1; then
  run_python python -m pip install peft
fi

lora_cmd=(
  python -m component3.run_train
  --run-id "${RUN_ID}"
  --dataset-name factkg
  --model-mode pv_qagnn
  --encoder_tune lora
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
  --disable-backtracking
  --no-enforce-no-collapse-gate
)

unfreeze_cmd=(
  python -m component3.run_train
  --run-id "${RUN_ID}"
  --dataset-name factkg
  --model-mode pv_qagnn
  --encoder_tune unfreeze_lastN
  --unfreeze_last_n "${UNFREEZE_LAST_N}"
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
  --disable-backtracking
  --no-enforce-no-collapse-gate
)

set -o pipefail
if run_python "${lora_cmd[@]}" 2>&1 | tee "runs/${RUN_ID}/run_console.log"; then
  printf "lora\n" > "runs/${RUN_ID}/encoder_tune_selected.txt"
else
  run_python "${unfreeze_cmd[@]}" 2>&1 | tee -a "runs/${RUN_ID}/run_console.log"
  printf "unfreeze_lastN\n" > "runs/${RUN_ID}/encoder_tune_selected.txt"
fi

if [[ -n "${ENV_NAME}" ]]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python scripts/autopilot_gates.py artifacts --run-id "${RUN_ID}" >/dev/null
else
  python scripts/autopilot_gates.py artifacts --run-id "${RUN_ID}" >/dev/null
fi
