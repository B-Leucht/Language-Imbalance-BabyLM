#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="$1"
CONFIG="./${CONFIG_NAME}_config.json"

BUDGETS=(1000000 3000000 10000000 30000000 50000000)
BUDGET_NAMES=("1m" "3m" "10m" "30m" "50m")

mkdir -p outputs logs

for i in "${!BUDGETS[@]}"; do
  TOKENS="${BUDGETS[$i]}"
  BUDGET_NAME="${BUDGET_NAMES[$i]}"
  RUN_NAME="mono_eng_${BUDGET_NAME}_${CONFIG_NAME}"

  echo "=================================================="
  echo "Running ${RUN_NAME}"
  echo "Config: ${CONFIG}"
  echo "Tokens: ${TOKENS}"
  echo "=================================================="

  python train.py \
    --dataset BabyLM-community/babylm-eng \
    --config "${CONFIG}" \
    --tokenizer_dir ./shared_tokenizer \
    --output_dir "./outputs/${RUN_NAME}" \
    --model_name "${RUN_NAME}" \
    --max_length 128 \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --max_tokens "${TOKENS}" \
    --seed 42 \
    2>&1 | tee "./logs/${RUN_NAME}.log"
done
