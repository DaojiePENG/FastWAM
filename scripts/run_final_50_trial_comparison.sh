#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
ACTION_TRAIN_ROOT="${ACTION_TRAIN_ROOT:-$ROOT_DIR/runs/action_aggregator_full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5}"
REMAINING_TRAIN_ROOT="${REMAINING_TRAIN_ROOT:-$ROOT_DIR/runs/remaining_modes_full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/correct_full_prefix_comparison/temporal_rope_v2/final_50_trials}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FINAL_STEP="${FINAL_STEP:-1115}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-2048}"
MODES=(fastwam_release action_aggregator interleaved vision_causal)
FINAL_TAG="step_$(printf '%06d' "$FINAL_STEP")"

IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_CSV"
NUM_WORKERS="${#GPU_IDS[@]}"
if (( NUM_WORKERS == 0 )); then
    printf 'No evaluation GPUs configured.\n' >&2
    exit 2
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

selected_gpus_are_free() {
    local used
    while IFS= read -r used; do
        (( used <= MAX_GPU_USED_MIB )) || return 1
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v ids="$GPU_IDS_CSV" 'BEGIN {split(ids, a, ","); for (i in a) wanted[a[i]]=1} wanted[NR-1] {print $1}')
}

mode_checkpoint() {
    local mode="$1"
    if [[ "$mode" == "action_aggregator" ]]; then
        printf '%s/%s/checkpoints/weights/%s.pt\n' \
            "$ACTION_TRAIN_ROOT" "$mode" "$FINAL_TAG"
    else
        printf '%s/%s/checkpoints/weights/%s.pt\n' \
            "$REMAINING_TRAIN_ROOT" "$mode" "$FINAL_TAG"
    fi
}

for mode in action_aggregator interleaved vision_causal; do
    checkpoint="$(mode_checkpoint "$mode")"
    if [[ ! -s "$checkpoint" ]]; then
        printf 'Final checkpoint missing for %s: %s\n' "$mode" "$checkpoint" >&2
        exit 2
    fi
done

while ! selected_gpus_are_free; do
    log "waiting for all final-evaluation GPUs: $GPU_IDS_CSV"
    sleep "$POLL_SECONDS"
done

run_task() {
    local gpu="$1"
    local mode="$2"
    local task_id="$3"
    local output_dir="$EVAL_ROOT/$mode"
    local log_file="$EVAL_ROOT/task_logs/${mode}_task${task_id}_gpu${gpu}.log"
    local existing_result checkpoint config_name task_choice
    local -a mode_args

    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        if [[ -s "$existing_result" ]] \
            && [[ "$(jq -r '.total_episodes // 0' "$existing_result")" == "$NUM_TRIALS" ]] \
            && jq -e '[.memory_metrics[]?.replans[]?.timing.total_inference_s] | length > 0' \
                "$existing_result" >/dev/null; then
            log "skip completed final mode=$mode task=$task_id result=$existing_result"
            return 0
        fi
    done

    mkdir -p "$output_dir"
    if [[ "$mode" == "fastwam_release" ]]; then
        checkpoint="$RELEASE_CHECKPOINT"
        config_name=sim_libero
        task_choice=libero_uncond_2cam224_1e-4
        mode_args=(
            EVALUATION.visualize_future_video=false
            EVALUATION.binarize_gripper=true
        )
    else
        checkpoint="$(mode_checkpoint "$mode")"
        config_name=sim_leapbot_libero
        task_choice=libero_leapbot_2cam224
        mode_args=(
            "model.causal_mode=$mode"
            model.video_lora.enabled=true
            EVALUATION.merge_video_lora=true
            EVALUATION.memory.enabled=true
            "EVALUATION.memory.causal_mode=$mode"
            EVALUATION.memory.exit_depth=30
            EVALUATION.memory.max_history_blocks=70
        )
    fi

    log "start final mode=$mode task=$task_id trials=$NUM_TRIALS gpu=$gpu"
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
        EVALUATION.save_rollout_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.output_dir=$output_dir" \
        "${mode_args[@]}" \
        >"$log_file" 2>&1; then
        log "done final mode=$mode task=$task_id gpu=$gpu"
    else
        log "FAILED final mode=$mode task=$task_id gpu=$gpu log=$log_file"
        return 1
    fi
}

worker() {
    local worker_index="$1"
    local gpu="${GPU_IDS[$worker_index]}"
    local index=0 mode task_id
    for mode in "${MODES[@]}"; do
        for task_id in $(seq 0 9); do
            if (( index % NUM_WORKERS == worker_index )); then
                run_task "$gpu" "$mode" "$task_id"
            fi
            index=$((index + 1))
        done
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
    log "one or more final-evaluation workers failed; inspect $EVAL_ROOT/task_logs"
    exit 1
fi

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
    "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT/pareto" \
    --expected-tasks 10 \
    --expected-trials-per-task "$NUM_TRIALS" \
    --require-profiled
MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/plot_pareto.py" \
    "$EVAL_ROOT/pareto" \
    --output-dir "$EVAL_ROOT/pareto"
log "final 4-config x 10-task x $NUM_TRIALS comparison complete: $EVAL_ROOT/pareto/results.csv"
