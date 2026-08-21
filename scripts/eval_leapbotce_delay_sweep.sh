#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: scripts/eval_leapbotce_delay_sweep.sh CHECKPOINT [TASK_ID]}"
if [[ -n "${2:-}" ]]; then
  TASK_IDS=("$2")
else
  TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
fi
for DELAY in 0 5 10 20; do
  for TASK_ID in "${TASK_IDS[@]}"; do
  python experiments/libero/eval_libero_cloudedge.py \
    task=libero_spatial_leapbotce \
    ckpt="${CKPT}" \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id="${TASK_ID}" \
    EVALUATION.max_delay_steps="${DELAY}" \
    EVALUATION.action_horizon=32 \
    EVALUATION.output_dir="./evaluate_results/leapbotce/delay_${DELAY}"
  done
done
