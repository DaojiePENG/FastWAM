#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_s1000_dev10}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
NUM_TRIALS="${NUM_TRIALS:-10}"
GPU_IDS=(6 7)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_task() {
    local gpu="$1"
    local task_id="$2"
    local output_dir="$EVAL_ROOT/fastwam_release"
    local log_dir="$EVAL_ROOT/task_logs"
    local log_file="$log_dir/fastwam_release_task${task_id}_gpu${gpu}.log"
    local result_file="$output_dir/libero_10/gpu${gpu}_task${task_id}_results.json"

    if [[ -s "$result_file" ]] \
        && [[ "$(jq -r '.total_episodes // 0' "$result_file")" == "$NUM_TRIALS" ]] \
        && jq -e '[.memory_metrics[]?.replans[]?.timing.total_inference_s] | length > 0' \
            "$result_file" >/dev/null; then
        log "skip completed baseline task=$task_id gpu=$gpu"
        return 0
    fi

    mkdir -p "$output_dir" "$log_dir" "$ROOT_DIR/.cache/matplotlib"
    log "start baseline task=$task_id gpu=$gpu"
    env -u CUDA_VISIBLE_DEVICES \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID=0 \
        PYOPENGL_PLATFORM=egl \
        MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
        PYTHONPATH="/home/sheng/workspace/LIBERO:$ROOT_DIR/experiments/libero" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/libero/eval_libero_single.py" \
        --config-name sim_libero \
        task=libero_uncond_2cam224_1e-4 \
        "ckpt=$RELEASE_CHECKPOINT" \
        "gpu_id=$gpu" \
        "EVALUATION.device=cuda:$gpu" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        EVALUATION.binarize_gripper=true \
        EVALUATION.visualize_future_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.output_dir=$output_dir" \
        >"$log_file" 2>&1
    log "done baseline task=$task_id gpu=$gpu"
}

worker() {
    local worker_index="$1"
    local gpu="${GPU_IDS[$worker_index]}"
    local task_id
    for task_id in $(seq "$worker_index" "${#GPU_IDS[@]}" 9); do
        run_task "$gpu" "$task_id"
    done
}

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    worker "$worker_index" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if (( failed )); then
    log "baseline worker failure; inspect $EVAL_ROOT/task_logs"
    exit 1
fi

log "FastWAM release development baseline complete"
