#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: scripts/eval_leapbotce_delay_sweep.sh CHECKPOINT [TASK_ID] [HYDRA_OVERRIDE ...]}"
shift
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  TASK_IDS=("$1")
  shift
else
  TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
fi
EXTRA_OVERRIDES=("$@")

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$PWD/third_party/LIBERO:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
read -r -a DELAYS <<< "${LEAPBOTCE_DELAYS:-0 5 10 20}"
for DELAY in "${DELAYS[@]}"; do
  for TASK_ID in "${TASK_IDS[@]}"; do
  python experiments/libero/eval_libero_cloudedge.py \
    task=libero_spatial_leapbotce \
    ckpt="${CKPT}" \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id="${TASK_ID}" \
    EVALUATION.max_delay_steps="${DELAY}" \
    EVALUATION.action_horizon=32 \
    EVALUATION.output_dir="./evaluate_results/leapbotce/delay_${DELAY}" \
    "${EXTRA_OVERRIDES[@]}" \
    model.skip_dit_load_from_pretrain=false
  done
done
