#!/usr/bin/env bash

set -euo pipefail

# Evaluate the pinned FastWAM release baseline over all LIBERO-Long tasks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LIBERO_ROOT="${LIBERO_ROOT:-$(cd "$ROOT_DIR/.." && pwd)/LIBERO}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/fastwam_release_dev10}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
NUM_TRIALS="${NUM_TRIALS:-10}"
GPU_IDS_CSV="${GPU_IDS_CSV:-6,7}"
IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_CSV"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if (( ${#GPU_IDS[@]} == 0 )); then
    printf 'No evaluation GPUs configured.\n' >&2
    exit 2
fi
if [[ ! -s "$RELEASE_CHECKPOINT" ]]; then
    printf 'FastWAM release checkpoint missing: %s\n' "$RELEASE_CHECKPOINT" >&2
    exit 2
fi
CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')"

fingerprint_path() {
    local task_id="$1"
    printf '%s/.fingerprints/fastwam_release_task%s.json\n' "$EVAL_ROOT" "$task_id"
}

build_task_fingerprint() {
    local task_id="$1"
    local expected
    expected="$(fingerprint_path "$task_id")"
    mkdir -p "$(dirname "$expected")"
    PYTHONPATH="$LIBERO_ROOT:$ROOT_DIR/experiments/libero" \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/build_eval_fingerprint.py" \
        --config-name sim_libero \
        --output "$expected" \
        --checkpoint-sha256 "$CHECKPOINT_SHA256" \
        -- \
        task=libero_uncond_2cam224_1e-4 \
        "ckpt=$RELEASE_CHECKPOINT" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        EVALUATION.save_rollout_video=false \
        EVALUATION.binarize_gripper=true \
        EVALUATION.visualize_future_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        >/dev/null
}

for task_id in $(seq 0 9); do
    build_task_fingerprint "$task_id"
    for existing_result in "$EVAL_ROOT/fastwam_release/libero_10"/gpu*_task"${task_id}"_results.json; do
        [[ -e "$existing_result" ]] || continue
        if ! PYTHONPATH="$ROOT_DIR/src" \
            "$ROOT_DIR/.venv/bin/python" -m leapbot_va.eval_fingerprint matches \
            "$existing_result" \
            --expected "$(fingerprint_path "$task_id")"; then
            log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
            exit 2
        fi
    done
done

run_task() {
    local gpu="$1"
    local task_id="$2"
    local output_dir="$EVAL_ROOT/fastwam_release"
    local log_dir="$EVAL_ROOT/task_logs"
    local log_file="$log_dir/fastwam_release_task${task_id}_gpu${gpu}.log"
    local expected_fingerprint="$(fingerprint_path "$task_id")"
    local existing_result

    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        [[ -e "$existing_result" ]] || continue
        if PYTHONPATH="$ROOT_DIR/src" \
            "$ROOT_DIR/.venv/bin/python" -m leapbot_va.eval_fingerprint matches \
            "$existing_result" \
            --expected "$expected_fingerprint"; then
            log "skip exact completed baseline task=$task_id result=$existing_result"
            return 0
        fi
        log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
        return 2
    done

    mkdir -p "$output_dir" "$log_dir" "$ROOT_DIR/.cache/matplotlib"
    log "start baseline task=$task_id gpu=$gpu"
    env -u CUDA_VISIBLE_DEVICES \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
        PYOPENGL_PLATFORM=egl \
        MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
        PYTHONPATH="$LIBERO_ROOT:$ROOT_DIR/experiments/libero" \
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
        EVALUATION.save_rollout_video=false \
        EVALUATION.binarize_gripper=true \
        EVALUATION.visualize_future_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.output_dir=$output_dir" \
        "+EVALUATION.expected_fingerprint_path=$expected_fingerprint" \
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
