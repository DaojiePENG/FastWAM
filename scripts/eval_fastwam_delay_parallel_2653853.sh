#!/usr/bin/env bash
set -euo pipefail

ROOT="/dahuafs/userdata/2653853/code/FastWAM"
CKPT="${FASTWAM_CHECKPOINT:-/mnt/c2-worldmodel/Models/fastwam_release/libero_uncond_2cam224.pt}"
OUTPUT_ROOT="${FASTWAM_DELAY_OUTPUT_ROOT:-/mnt/c2-worldmodel/2653853/baseline-evaluations/fastwam/libero_spatial/full_$(date +%Y%m%d)}"
TASK_ID="${FASTWAM_TASK_ID:-0}"
read -r -a DELAYS <<< "${FASTWAM_DELAYS:-0 5 10 20 40}"
read -r -a GPUS <<< "${FASTWAM_GPUS:-0 1 2 3 0}"
if [[ ${#DELAYS[@]} -ne ${#GPUS[@]} ]]; then
  echo "FASTWAM_DELAYS and FASTWAM_GPUS must have the same length" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local delay="$1" gpu="$2"
  (
    cd "${ROOT}"
    source scripts/env_2653853.sh
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export FASTWAM_DELAYS="${delay}"
    export FASTWAM_DELAY_OUTPUT_ROOT="${OUTPUT_ROOT}"
    bash scripts/eval_fastwam_delay_sweep.sh "${CKPT}" "${TASK_ID}" \
      EVALUATION.num_trials="${FASTWAM_NUM_TRIALS:-1}" \
      EVALUATION.max_episode_steps=null \
      EVALUATION.num_inference_steps=10 \
      EVALUATION.replan_steps=10 \
      EVALUATION.compile_action_infer=false
  ) >"${OUTPUT_ROOT}/driver_delay${delay}.log" 2>&1
}

# One model process per H800. Repeated GPU IDs queue behind that GPU's run.
declare -A gpu_pid=() gpu_delay=()
status=0
for i in "${!DELAYS[@]}"; do
  delay="${DELAYS[$i]}" gpu="${GPUS[$i]}"
  if [[ -n "${gpu_pid[$gpu]:-}" ]]; then
    if wait "${gpu_pid[$gpu]}"; then
      echo "[done] FastWAM delay=${gpu_delay[$gpu]}"
    else
      echo "[failed] FastWAM delay=${gpu_delay[$gpu]}; inspect ${OUTPUT_ROOT}/driver_delay${gpu_delay[$gpu]}.log" >&2
      status=1
    fi
  fi
  echo "[launch] FastWAM delay=${delay} physical_gpu=${gpu}"
  run_one "${delay}" "${gpu}" &
  gpu_pid[$gpu]="$!"
  gpu_delay[$gpu]="${delay}"
done
for gpu in "${!gpu_pid[@]}"; do
  if wait "${gpu_pid[$gpu]}"; then
    echo "[done] FastWAM delay=${gpu_delay[$gpu]}"
  else
    rc=$?
    echo "[failed] FastWAM delay=${gpu_delay[$gpu]} rc=${rc}; ${OUTPUT_ROOT}/driver_delay${gpu_delay[$gpu]}.log" >&2
    status=1
  fi
done
if [[ "${status}" == 0 ]]; then
  source "${ROOT}/scripts/env_2653853.sh"
  python "${ROOT}/scripts/summarize_leapbotce_delays.py" "${OUTPUT_ROOT}"
fi
exit "${status}"
