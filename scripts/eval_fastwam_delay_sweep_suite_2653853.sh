#!/usr/bin/env bash
# SUITE-AWARE COPY of scripts/eval_fastwam_delay_sweep.sh.
# The original has running instances (FastWAM spatial d40) — scripts must never
# be edited in place while any instance is running (byte-offset corruption).
# Only change vs the original: FASTWAM_TASK_SUITE env (default libero_spatial)
# substitutes into EVALUATION.task_suite_name. OUTPUT_ROOT stays env-driven via
# FASTWAM_DELAY_OUTPUT_ROOT so each suite writes under its own root.
set -euo pipefail

CKPT="${1:?Usage: scripts/eval_fastwam_delay_sweep_suite_2653853.sh CHECKPOINT [TASK_ID] [HYDRA_OVERRIDE ...]}"
shift
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  TASK_IDS=("$1")
  shift
else
  TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
fi
EXTRA_OVERRIDES=("$@")

TASK_SUITE="${FASTWAM_TASK_SUITE:-libero_spatial}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$PWD/src:$PWD:/dahuafs/userdata/2653853/code/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT="${FASTWAM_DELAY_OUTPUT_ROOT:-/mnt/c2-worldmodel/2653853/baseline-evaluations/fastwam/libero_spatial}"
DATASET_STATS="${FASTWAM_DATASET_STATS:-/mnt/c2-worldmodel/Models/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
read -r -a DELAYS <<< "${FASTWAM_DELAYS:-0 5 10 20 40}"

for DELAY in "${DELAYS[@]}"; do
  for TASK_ID in "${TASK_IDS[@]}"; do
    python experiments/libero/eval_libero_delayed.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt="${CKPT}" \
      seed=1234 \
      EVALUATION.task_suite_name="${TASK_SUITE}" \
      EVALUATION.task_id="${TASK_ID}" \
      EVALUATION.max_delay_steps="${DELAY}" \
      EVALUATION.dataset_stats_path="${DATASET_STATS}" \
      EVALUATION.sigma_shift=5.0 \
      EVALUATION.output_dir="${OUTPUT_ROOT}/delay_${DELAY}" \
      model.redirect_common_files=false \
      "${EXTRA_OVERRIDES[@]}"
  done
done
