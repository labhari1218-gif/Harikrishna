#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [run_id]"
  echo "Default run_id: t33_full_seed57"
  exit 0
fi

RUN_ID="${1:-t33_full_seed57}"
PRECOMPUTE_COMPONENT3="${PRECOMPUTE_COMPONENT3:-0}"
COMPONENT3_LOADER_NUM_WORKERS="${COMPONENT3_LOADER_NUM_WORKERS:-2}"
COMPONENT3_LOADER_PREFETCH_FACTOR="${COMPONENT3_LOADER_PREFETCH_FACTOR:-2}"
COMPONENT3_CLAIM_TRIPLE_CACHE_PATH="${COMPONENT3_CLAIM_TRIPLE_CACHE_PATH:-data/claim_triple_embeddings.pkl}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

run_component1_split() {
  local split="$1"
  local log_file="${RUN_DIR}/component1_${split}.log"
  echo "[C1] split=${split} -> ${log_file}"
  conda run --no-capture-output -n fact_check_env python scripts/run_component1_pv_esm.py \
    --split "${split}" \
    --pair_log_mode all \
    --model_name microsoft/deberta-base-mnli \
    --batch_size 8 \
    --max_length 256 \
    2>&1 | tee "${log_file}"
}

echo "[START] $(date -u +'%Y-%m-%dT%H:%M:%SZ') run_id=${RUN_ID}" | tee "${RUN_DIR}/orchestration.log"

run_component1_split train
run_component1_split val
run_component1_split test

echo "[C1 COVERAGE] validating pair-log coverage" | tee -a "${RUN_DIR}/orchestration.log"
conda run --no-capture-output -n fact_check_env python scripts/validate_component1_pairs_coverage.py \
  --splits train,val,test \
  --logs-root logs/component1 \
  --subgraph-type direct_filled \
  --missing-policy hybrid_fallback \
  --out-json "${RUN_DIR}/c1_coverage.json" \
  2>&1 | tee "${RUN_DIR}/c1_coverage.log"

if [[ "${PRECOMPUTE_COMPONENT3}" == "1" ]]; then
  echo "[C3 PRECOMPUTE] building claim-triple cache" | tee -a "${RUN_DIR}/orchestration.log"
  conda run --no-capture-output -n fact_check_env python scripts/precompute_component3_embeddings.py \
    --dataset factkg \
    --splits train,val,test \
    --encoder-model-name bert-base-uncased \
    --max-seq-len 256 \
    --batch-size 32 \
    --use-component1-pairs \
    --component1-logs-root logs/component1 \
    --missing-pv-policy hybrid_fallback \
    --subgraph-type direct_filled \
    --factkg-claim-triple-cache-path "${COMPONENT3_CLAIM_TRIPLE_CACHE_PATH}" \
    --manifest-path "${RUN_DIR}/precompute_component3_manifest.json" \
    2>&1 | tee "${RUN_DIR}/precompute_component3.log"
fi

echo "[C3] full-dataset training/eval" | tee -a "${RUN_DIR}/orchestration.log"
set +e
conda run --no-capture-output -n fact_check_env python -m component3.run_train \
  --run-id "${RUN_ID}" \
  --train-subset-size 0 \
  --val-subset-size 0 \
  --subset-sampling stratified \
  --use-component1-pairs \
  --component1-logs-root logs/component1 \
  --missing-pv-policy hybrid_fallback \
  --gradient-accumulation-steps 4 \
  --stage1-epochs 5 \
  --stage2-epochs 5 \
  --batch-size 8 \
  --loader-num-workers "${COMPONENT3_LOADER_NUM_WORKERS}" \
  --loader-pin-memory \
  --loader-persistent-workers \
  --loader-prefetch-factor "${COMPONENT3_LOADER_PREFETCH_FACTOR}" \
  --non-blocking-transfers \
  --factkg-claim-triple-cache-path "${COMPONENT3_CLAIM_TRIPLE_CACHE_PATH}" \
  --factkg-require-claim-triple-cache \
  --factkg-precompute-batch-size 32 \
  2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_STATUS=$?
set -e

if [[ -f "${RUN_DIR}/metrics.json" ]]; then
  echo "[COMPARE] SOTA/baseline comparison" | tee -a "${RUN_DIR}/orchestration.log"
  conda run --no-capture-output -n fact_check_env python scripts/compare_component3_to_sota.py \
    --run-metrics "${RUN_DIR}/metrics.json" \
    --baseline-metrics "runs/t33_tmux_20260215_095704/metrics.json" \
    --out-json "${RUN_DIR}/sota_comparison.json" \
    2>&1 | tee "${RUN_DIR}/sota_compare.log"
fi

echo "[END] $(date -u +'%Y-%m-%dT%H:%M:%SZ') status=${TRAIN_STATUS}" | tee -a "${RUN_DIR}/orchestration.log"
exit "${TRAIN_STATUS}"
