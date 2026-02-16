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

STAGE1_EPOCHS="${OVERFIT_STAGE1_EPOCHS:-25}"
STAGE1_LR="${OVERFIT_STAGE1_LR:-2e-5}"

cmd=(
  python -m component3.run_train
  --run-id "${RUN_ID}"
  --dataset-name factkg
  --model-mode pv_qagnn
  --encoder_tune none
  --train-subset-size 200
  --val-subset-size 200
  --subset-sampling stratified
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs 0
  --stage1-lr "${STAGE1_LR}"
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

if [[ -n "${ENV_NAME}" ]]; then
  cmd=(conda run --no-capture-output -n "${ENV_NAME}" "${cmd[@]}")
fi

set -o pipefail
"${cmd[@]}" 2>&1 | tee "runs/${RUN_ID}/run_console.log"

gate_cmd=(python scripts/autopilot_gates.py artifacts --run-id "${RUN_ID}")
if [[ -n "${ENV_NAME}" ]]; then
  gate_cmd=(conda run --no-capture-output -n "${ENV_NAME}" "${gate_cmd[@]}")
fi
"${gate_cmd[@]}" >/dev/null
