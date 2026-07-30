#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/phase1_h8_d30_s1000}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_s1000_dev10}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_GPUS="${NUM_GPUS:-8}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FINAL_STEP="${FINAL_STEP:-1000}"
FINAL_STEP_TAG="$(printf 'step_%06d' "$FINAL_STEP")"

MODES=(interleaved vision_causal action_aggregator)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

mode_checkpoint() {
    local mode="$1"
    printf '%s/%s/checkpoints/weights/%s.pt\n' "$TRAIN_ROOT" "$mode" "$FINAL_STEP_TAG"
}

training_complete() {
    local mode checkpoint log_file
    for mode in "${MODES[@]}"; do
        checkpoint="$(mode_checkpoint "$mode")"
        log_file="$TRAIN_ROOT/$mode/train.log"
        if [[ ! -s "$checkpoint" ]] \
            || ! grep -q "\[done\] max_steps reached step=$FINAL_STEP" "$log_file"; then
            return 1
        fi
    done
    return 0
}

mkdir -p "$EVAL_ROOT/task_logs" "$ROOT_DIR/.cache/matplotlib"

while ! training_complete; do
    progress=()
    for mode in "${MODES[@]}"; do
        step="$({ grep -o "step=[0-9]\\+/$FINAL_STEP" "$TRAIN_ROOT/$mode/train.log" 2>/dev/null || true; } | tail -1)"
        progress+=("$mode:${step:-initializing}")
    done
    log "waiting for phase-1 checkpoints (${progress[*]})"
    sleep "$POLL_SECONDS"
done

log "all phase-1 checkpoints are complete; starting LIBERO-Long dev evaluation"

run_task() {
    local gpu="$1"
    local mode="$2"
    local task_id="$3"
    local output_dir="$EVAL_ROOT/$mode"
    local log_file="$EVAL_ROOT/task_logs/${mode}_task${task_id}_gpu${gpu}.log"
    local result_file="$output_dir/libero_10/gpu${gpu}_task${task_id}_results.json"
    local existing_result
    local checkpoint config_name task_choice
    local -a mode_args

    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        if [[ -s "$existing_result" ]] \
            && [[ "$(jq -r '.total_episodes // 0' "$existing_result")" == "$NUM_TRIALS" ]] \
            && jq -e '[.memory_metrics[]?.replans[]?.timing.total_inference_s] | length > 0' \
                "$existing_result" >/dev/null; then
            log "skip completed mode=$mode task=$task_id result=$existing_result"
            return 0
        fi
    done

    mkdir -p "$output_dir"
    if [[ "$mode" == "fastwam_release" ]]; then
        checkpoint="$RELEASE_CHECKPOINT"
        config_name="sim_libero"
        task_choice="libero_uncond_2cam224_1e-4"
        mode_args=(
            "EVALUATION.visualize_future_video=false"
            "EVALUATION.binarize_gripper=true"
        )
    else
        checkpoint="$(mode_checkpoint "$mode")"
        config_name="sim_leapbot_libero"
        task_choice="libero_leapbot_2cam224"
        mode_args=(
            "model.causal_mode=$mode"
            "EVALUATION.memory.causal_mode=$mode"
            "EVALUATION.memory.exit_depth=30"
            "EVALUATION.memory.max_history_blocks=70"
        )
    fi

    log "start mode=$mode task=$task_id gpu=$gpu"
    if env -u CUDA_VISIBLE_DEVICES \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID=0 \
        PYOPENGL_PLATFORM=egl \
        MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
        PYTHONPATH="/home/sheng/workspace/LIBERO:$ROOT_DIR/experiments/libero" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/libero/eval_libero_single.py" \
        --config-name "$config_name" \
        "task=$task_choice" \
        "ckpt=$checkpoint" \
        "gpu_id=$gpu" \
        "EVALUATION.device=cuda:$gpu" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.output_dir=$output_dir" \
        "${mode_args[@]}" \
        >"$log_file" 2>&1; then
        log "done mode=$mode task=$task_id gpu=$gpu"
    else
        log "FAILED mode=$mode task=$task_id gpu=$gpu log=$log_file"
        return 1
    fi
}

worker() {
    local gpu="$1"
    local index=0
    local mode task_id
    for mode in fastwam_release "${MODES[@]}"; do
        for task_id in $(seq 0 9); do
            if (( index % NUM_GPUS == gpu )); then
                run_task "$gpu" "$mode" "$task_id"
            fi
            index=$((index + 1))
        done
    done
}

worker_pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    worker "$gpu" &
    worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if (( failed )); then
    log "one or more evaluation workers failed; inspect $EVAL_ROOT/task_logs"
    exit 1
fi

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
    "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT/pareto"
log "phase-1 comparison complete: $EVAL_ROOT/pareto/results.csv"
