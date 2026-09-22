#!/usr/bin/env bash
# Statistical-scale delay sweep: all 10 libero_spatial tasks per delay, one
# delay per GPU. Same protocol as the smoke runs (seed 1234, replan 10).
set -euo pipefail

ROOT="/dahuafs/userdata/2653853/code/FastWAM"
CKPT="${FASTWAM_CHECKPOINT:-/mnt/c2-worldmodel/Models/fastwam_release/libero_uncond_2cam224.pt}"
OUTPUT_ROOT="${FASTWAM_DELAY_OUTPUT_ROOT:-/mnt/c2-worldmodel/2653853/baseline-evaluations/fastwam/libero_spatial/full10_20260919}"
read -r -a DELAYS <<< "${FASTWAM_DELAYS:-0 5 10 20 40}"
read -r -a GPUS <<< "${FASTWAM_GPUS:-0 1 2 3 4}"
if [[ ${#DELAYS[@]} -ne ${#GPUS[@]} ]]; then
  echo "FASTWAM_DELAYS and FASTWAM_GPUS must have the same length" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

pids=()
for i in "${!DELAYS[@]}"; do
  delay="${DELAYS[$i]}"
  gpu="${GPUS[$i]}"
  echo "[launch] FastWAM delay=${delay} physical_gpu=${gpu}"
  (
    cd "${ROOT}"
    source scripts/env_2653853.sh
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export FASTWAM_DELAYS="${delay}"
    export FASTWAM_DELAY_OUTPUT_ROOT="${OUTPUT_ROOT}"
    # no TASK_ID arg -> sweep script runs tasks 0..9 sequentially
    bash scripts/eval_fastwam_delay_sweep.sh "${CKPT}" \
      EVALUATION.num_trials="${FASTWAM_NUM_TRIALS:-1}" \
      EVALUATION.max_episode_steps=null \
      EVALUATION.num_inference_steps=10 \
      EVALUATION.replan_steps=10 \
      EVALUATION.compile_action_infer=false
  ) >"${OUTPUT_ROOT}/driver_delay${delay}.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] FastWAM delay=${DELAYS[$i]}"
  else
    echo "[failed] FastWAM delay=${DELAYS[$i]}; see ${OUTPUT_ROOT}/driver_delay${DELAYS[$i]}.log" >&2
    status=1
  fi
done

source "${ROOT}/scripts/env_2653853.sh" || true
python "${ROOT}/scripts/summarize_leapbotce_delays.py" "${OUTPUT_ROOT}" || true
exit "${status}"
