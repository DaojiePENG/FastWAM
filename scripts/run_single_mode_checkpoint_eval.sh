#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:?TRAIN_ROOT is required}"
EVAL_ROOT="${EVAL_ROOT:?EVAL_ROOT is required}"
MODE="${MODE:-action_aggregator}"
FINAL_STEP="${FINAL_STEP:?FINAL_STEP is required}"
NUM_TRIALS="${NUM_TRIALS:-2}"
GPU_IDS_CSV="${GPU_IDS_CSV:-6,7}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
VIDEO_LORA_ENABLED="${VIDEO_LORA_ENABLED:-true}"
MERGE_VIDEO_LORA="${MERGE_VIDEO_LORA:-true}"
MEMORY_ENABLED="${MEMORY_ENABLED:-true}"
MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS:-70}"
FINAL_STEP_TAG="$(printf 'step_%06d' "$FINAL_STEP")"
CHECKPOINT="$TRAIN_ROOT/$MODE/checkpoints/weights/$FINAL_STEP_TAG.pt"

IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_CSV"
NUM_WORKERS="${#GPU_IDS[@]}"
if (( NUM_WORKERS == 0 )); then
    printf 'No evaluation GPUs configured.\n' >&2
    exit 2
fi
if [[ ! -s "$CHECKPOINT" ]]; then
    printf 'Checkpoint not ready: %s\n' "$CHECKPOINT" >&2
    exit 2
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_task() {
    local gpu="$1"
    local task_id="$2"
    local output_dir="$EVAL_ROOT/$MODE"
    local log_file="$EVAL_ROOT/task_logs/${MODE}_task${task_id}_gpu${gpu}.log"
    local existing_result

    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        if [[ -s "$existing_result" ]] \
            && [[ "$(jq -r '.total_episodes // 0' "$existing_result")" == "$NUM_TRIALS" ]] \
            && jq -e '[.memory_metrics[]?.replans[]?.timing.total_inference_s] | length > 0' \
                "$existing_result" >/dev/null; then
            log "skip completed mode=$MODE task=$task_id result=$existing_result"
            return 0
        fi
    done

    mkdir -p "$output_dir"
    log "start mode=$MODE step=$FINAL_STEP task=$task_id gpu=$gpu"
    if env -u CUDA_VISIBLE_DEVICES \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID=0 \
        PYOPENGL_PLATFORM=egl \
        MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
        PYTHONPATH="/home/sheng/workspace/LIBERO:$ROOT_DIR/experiments/libero" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/libero/eval_libero_single.py" \
        --config-name sim_leapbot_libero \
        task=libero_leapbot_2cam224 \
        "ckpt=$CHECKPOINT" \
        "gpu_id=$gpu" \
        "EVALUATION.device=cuda:$gpu" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        EVALUATION.save_rollout_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.output_dir=$output_dir" \
        "model.causal_mode=$MODE" \
        "model.video_lora.enabled=$VIDEO_LORA_ENABLED" \
        "EVALUATION.merge_video_lora=$MERGE_VIDEO_LORA" \
        "EVALUATION.memory.enabled=$MEMORY_ENABLED" \
        "EVALUATION.memory.causal_mode=$MODE" \
        EVALUATION.memory.exit_depth=30 \
        "EVALUATION.memory.max_history_blocks=$MAX_HISTORY_BLOCKS" \
        >"$log_file" 2>&1; then
        log "done mode=$MODE step=$FINAL_STEP task=$task_id gpu=$gpu"
    else
        log "FAILED mode=$MODE step=$FINAL_STEP task=$task_id gpu=$gpu log=$log_file"
        return 1
    fi
}

worker() {
    local worker_index="$1"
    local gpu="${GPU_IDS[$worker_index]}"
    local task_id
    for task_id in $(seq 0 9); do
        if (( task_id % NUM_WORKERS == worker_index )); then
            run_task "$gpu" "$task_id"
        fi
    done
}

mkdir -p "$EVAL_ROOT/task_logs" "$ROOT_DIR/.cache/matplotlib"
worker_pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    worker "$worker_index" &
    worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    log "one or more single-mode evaluation workers failed; inspect $EVAL_ROOT/task_logs"
    exit 1
fi

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
    "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT/pareto" \
    --expected-tasks 10 \
    --expected-trials-per-task "$NUM_TRIALS" \
    --require-profiled
log "single-mode evaluation complete: $EVAL_ROOT/pareto/results.csv"
