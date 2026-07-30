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
RETAINED_HISTORY_BLOCKS="${RETAINED_HISTORY_BLOCKS:-full}"
EXIT_DEPTH="${EXIT_DEPTH:-30}"
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
case "$EXIT_DEPTH" in
    8|16|24|30) ;;
    *)
        printf 'EXIT_DEPTH must be one of 8,16,24,30.\n' >&2
        exit 2
        ;;
esac
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$MEMORY_ENABLED" == "true" ]]; then
    FINGERPRINT_EPISODE_CAPACITY="$MAX_HISTORY_BLOCKS"
    if [[ "$RETAINED_HISTORY_BLOCKS" == "full" ]]; then
        MEMORY_RETENTION_OVERRIDE=null
        FINGERPRINT_HISTORY_CAP="$MAX_HISTORY_BLOCKS"
    elif [[ "$RETAINED_HISTORY_BLOCKS" =~ ^[0-9]+$ ]] \
        && (( RETAINED_HISTORY_BLOCKS <= MAX_HISTORY_BLOCKS )); then
        MEMORY_RETENTION_OVERRIDE="$RETAINED_HISTORY_BLOCKS"
        FINGERPRINT_HISTORY_CAP="$RETAINED_HISTORY_BLOCKS"
    else
        printf 'RETAINED_HISTORY_BLOCKS must be full or an integer in [0,%s].\n' \
            "$MAX_HISTORY_BLOCKS" >&2
        exit 2
    fi
else
    FINGERPRINT_EPISODE_CAPACITY=0
    FINGERPRINT_HISTORY_CAP=0
    MEMORY_RETENTION_OVERRIDE=null
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fingerprint_path() {
    local task_id="$1"
    printf '%s/.fingerprints/%s_d%s_h%s_task%s.json\n' \
        "$EVAL_ROOT" "$MODE" "$EXIT_DEPTH" "$FINGERPRINT_HISTORY_CAP" "$task_id"
}

build_task_fingerprint() {
    local task_id="$1"
    local fingerprint_file
    fingerprint_file="$(fingerprint_path "$task_id")"

    mkdir -p "$(dirname "$fingerprint_file")"
    PYTHONPATH="/home/sheng/workspace/LIBERO:$ROOT_DIR/experiments/libero" \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/build_eval_fingerprint.py" \
        --config-name sim_leapbot_libero \
        --output "$fingerprint_file" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        -- \
        task=libero_leapbot_2cam224 \
        "ckpt=$CHECKPOINT" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        EVALUATION.save_rollout_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "model.causal_mode=$MODE" \
        model.training_strategy=video_lora_action_full \
        "model.video_lora.enabled=$VIDEO_LORA_ENABLED" \
        "EVALUATION.merge_video_lora=$MERGE_VIDEO_LORA" \
        "EVALUATION.memory.enabled=$MEMORY_ENABLED" \
        "EVALUATION.memory.causal_mode=$MODE" \
        "EVALUATION.memory.exit_depth=$EXIT_DEPTH" \
        "EVALUATION.memory.max_history_blocks=$MAX_HISTORY_BLOCKS" \
        "EVALUATION.memory.retained_history_blocks=$MEMORY_RETENTION_OVERRIDE" \
        >/dev/null
}

result_matches_task() {
    local result="$1"
    local task_id="$2"
    [[ -s "$result" ]] && PYTHONPATH="$ROOT_DIR/src" \
        "$ROOT_DIR/.venv/bin/python" -m leapbot_va.eval_fingerprint matches \
        "$result" \
        --expected "$(fingerprint_path "$task_id")"
}

# Validate the entire result tree before any worker can start an evaluator.
# This keeps a stale result for a later task from racing with earlier tasks.
for task_id in $(seq 0 9); do
    build_task_fingerprint "$task_id"
    for existing_result in "$EVAL_ROOT/$MODE/libero_10"/gpu*_task"${task_id}"_results.json; do
        [[ -e "$existing_result" ]] || continue
        if ! result_matches_task "$existing_result" "$task_id"; then
            log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
            exit 2
        fi
    done
done

run_task() {
    local gpu="$1"
    local task_id="$2"
    local output_dir="$EVAL_ROOT/$MODE"
    local log_file="$EVAL_ROOT/task_logs/${MODE}_task${task_id}_gpu${gpu}.log"
    local fingerprint_file
    local existing_result
    fingerprint_file="$(fingerprint_path "$task_id")"

    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        [[ -e "$existing_result" ]] || continue
        if result_matches_task "$existing_result" "$task_id"; then
            log "skip exact completed mode=$MODE task=$task_id result=$existing_result"
            return 0
        fi
        log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
        return 2
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
        model.training_strategy=video_lora_action_full \
        "model.video_lora.enabled=$VIDEO_LORA_ENABLED" \
        "EVALUATION.merge_video_lora=$MERGE_VIDEO_LORA" \
        "EVALUATION.memory.enabled=$MEMORY_ENABLED" \
        "EVALUATION.memory.causal_mode=$MODE" \
        "EVALUATION.memory.exit_depth=$EXIT_DEPTH" \
        "EVALUATION.memory.max_history_blocks=$MAX_HISTORY_BLOCKS" \
        "EVALUATION.memory.retained_history_blocks=$MEMORY_RETENTION_OVERRIDE" \
        "+EVALUATION.expected_fingerprint_path=$fingerprint_file" \
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
