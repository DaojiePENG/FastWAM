#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
ACTION_TRAIN_ROOT="${ACTION_TRAIN_ROOT:?ACTION_TRAIN_ROOT is required}"
REMAINING_TRAIN_ROOT="${REMAINING_TRAIN_ROOT:?REMAINING_TRAIN_ROOT is required}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/correct_full_prefix_comparison/temporal_rope_v2/final_50_trials}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FINAL_STEP="${FINAL_STEP:?FINAL_STEP is required}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-2048}"
MODES=(fastwam_release action_aggregator interleaved vision_causal)
FINAL_TAG="step_$(printf '%06d' "$FINAL_STEP")"
declare -A CHECKPOINT_BY_MODE
declare -A CHECKPOINT_SHA256_BY_MODE

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
if [[ ! -s "$RELEASE_CHECKPOINT" ]]; then
    printf 'FastWAM release checkpoint missing: %s\n' "$RELEASE_CHECKPOINT" >&2
    exit 2
fi
mkdir -p "$EVAL_ROOT/.checkpoint_validation"
for mode in action_aggregator interleaved vision_causal; do
    if [[ "$mode" == "action_aggregator" ]]; then
        mode_train_root="$ACTION_TRAIN_ROOT"
    else
        mode_train_root="$REMAINING_TRAIN_ROOT"
    fi
    run_contract_file="$mode_train_root/$mode/run_contract.txt"
    if [[ ! -s "$run_contract_file" ]]; then
        printf 'Training run contract missing: %s\n' "$run_contract_file" >&2
        exit 2
    fi
    expected_run_contract_sha256="$(awk -F= '$1 == "run_contract_sha256" {print $2}' "$run_contract_file")"
    expected_code_commit="$(awk -F= '$1 == "code_commit" {print $2}' "$run_contract_file")"
    if [[ ! "$expected_run_contract_sha256" =~ ^[0-9a-f]{64}$ ]] \
        || [[ ! "$expected_code_commit" =~ ^[0-9a-f]{40}$ ]]; then
        printf 'Invalid training identity in %s\n' "$run_contract_file" >&2
        exit 2
    fi
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
        "$(mode_checkpoint "$mode")" \
        --expected-step "$FINAL_STEP" \
        --expected-mode "$mode" \
        --expected-trained-exit-depths 30 \
        --expected-history-vae-batch-chunk-size 1 \
        --expected-run-contract-sha256 "$expected_run_contract_sha256" \
        --expected-code-commit "$expected_code_commit" \
        --state-dir "$mode_train_root/$mode/checkpoints/state/$FINAL_TAG" \
        --output "$EVAL_ROOT/.checkpoint_validation/${mode}_${FINAL_TAG}.json" \
        >/dev/null
done

CHECKPOINT_BY_MODE[fastwam_release]="$RELEASE_CHECKPOINT"
for mode in action_aggregator interleaved vision_causal; do
    CHECKPOINT_BY_MODE[$mode]="$(mode_checkpoint "$mode")"
done
for mode in "${MODES[@]}"; do
    log "hashing checkpoint once for result identity: mode=$mode"
    CHECKPOINT_SHA256_BY_MODE[$mode]="$(sha256sum "${CHECKPOINT_BY_MODE[$mode]}" | awk '{print $1}')"
done

fingerprint_path() {
    local mode="$1"
    local task_id="$2"
    printf '%s/.fingerprints/%s_task%s.json\n' "$EVAL_ROOT" "$mode" "$task_id"
}

build_mode_fingerprint() {
    local mode="$1"
    local task_id="$2"
    local checkpoint="${CHECKPOINT_BY_MODE[$mode]}"
    local checkpoint_sha256="${CHECKPOINT_SHA256_BY_MODE[$mode]}"
    local expected
    local config_name task_choice
    local -a mode_args
    expected="$(fingerprint_path "$mode" "$task_id")"
    if [[ "$mode" == "fastwam_release" ]]; then
        config_name=sim_libero
        task_choice=libero_uncond_2cam224_1e-4
        mode_args=(
            EVALUATION.visualize_future_video=false
            EVALUATION.binarize_gripper=true
        )
    else
        config_name=sim_leapbot_libero
        task_choice=libero_leapbot_2cam224
        mode_args=(
            "model.causal_mode=$mode"
            model.training_strategy=video_lora_action_full
            model.video_lora.enabled=true
            EVALUATION.merge_video_lora=true
            EVALUATION.memory.enabled=true
            "EVALUATION.memory.causal_mode=$mode"
            EVALUATION.memory.exit_depth=30
            EVALUATION.memory.max_history_blocks=70
            EVALUATION.memory.retained_history_blocks=null
        )
    fi
    mkdir -p "$(dirname "$expected")"
    PYTHONPATH="/home/sheng/workspace/LIBERO:$ROOT_DIR/experiments/libero" \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/build_eval_fingerprint.py" \
        --config-name "$config_name" \
        --output "$expected" \
        --checkpoint-sha256 "$checkpoint_sha256" \
        -- \
        "task=$task_choice" \
        "ckpt=$checkpoint" \
        EVALUATION.task_suite_name=libero_10 \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$NUM_TRIALS" \
        EVALUATION.num_inference_steps=10 \
        EVALUATION.replan_steps=10 \
        EVALUATION.save_rollout_video=false \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "${mode_args[@]}" \
        >/dev/null
}

result_matches_mode() {
    local result="$1"
    local mode="$2"
    local task_id="$3"
    [[ -s "$result" ]] && PYTHONPATH="$ROOT_DIR/src" \
        "$ROOT_DIR/.venv/bin/python" -m leapbot_va.eval_fingerprint matches \
        "$result" \
        --expected "$(fingerprint_path "$mode" "$task_id")"
}

# Refuse a mixed result tree before any worker can acquire a GPU. Matching
# results remain resumable; legacy or differently configured results require a
# fresh EVAL_ROOT (or explicit operator cleanup after audit).
pending_evaluations=0
for mode in "${MODES[@]}"; do
    for task_id in $(seq 0 9); do
        matching_result_found=false
        build_mode_fingerprint "$mode" "$task_id"
        for existing_result in "$EVAL_ROOT/$mode/libero_10"/gpu*_task"${task_id}"_results.json; do
            [[ -e "$existing_result" ]] || continue
            if ! result_matches_mode "$existing_result" "$mode" "$task_id"; then
                log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
                log "use a fresh EVAL_ROOT or remove the mismatched result after auditing it"
                exit 2
            fi
            matching_result_found=true
        done
        if [[ "$matching_result_found" != "true" ]]; then
            pending_evaluations=$((pending_evaluations + 1))
        fi
    done
done

if (( pending_evaluations > 0 )); then
    while ! selected_gpus_are_free; do
        log "waiting for all final-evaluation GPUs: $GPU_IDS_CSV"
        sleep "$POLL_SECONDS"
    done
else
    log "all final-evaluation results exactly match; no GPU wait required"
fi

run_task() {
    local gpu="$1"
    local mode="$2"
    local task_id="$3"
    local output_dir="$EVAL_ROOT/$mode"
    local log_file="$EVAL_ROOT/task_logs/${mode}_task${task_id}_gpu${gpu}.log"
    local existing_result checkpoint checkpoint_sha256 config_name task_choice
    local expected_fingerprint
    local -a mode_args

    checkpoint="${CHECKPOINT_BY_MODE[$mode]}"
    checkpoint_sha256="${CHECKPOINT_SHA256_BY_MODE[$mode]}"
    expected_fingerprint="$(fingerprint_path "$mode" "$task_id")"
    if [[ "$mode" == "fastwam_release" ]]; then
        config_name=sim_libero
        task_choice=libero_uncond_2cam224_1e-4
        mode_args=(
            EVALUATION.visualize_future_video=false
            EVALUATION.binarize_gripper=true
        )
    else
        config_name=sim_leapbot_libero
        task_choice=libero_leapbot_2cam224
        mode_args=(
            "model.causal_mode=$mode"
            model.training_strategy=video_lora_action_full
            model.video_lora.enabled=true
            EVALUATION.merge_video_lora=true
            EVALUATION.memory.enabled=true
            "EVALUATION.memory.causal_mode=$mode"
            EVALUATION.memory.exit_depth=30
            EVALUATION.memory.max_history_blocks=70
            EVALUATION.memory.retained_history_blocks=null
        )
    fi
    for existing_result in "$output_dir/libero_10"/gpu*_task"${task_id}"_results.json; do
        [[ -e "$existing_result" ]] || continue
        if result_matches_mode "$existing_result" "$mode" "$task_id"; then
            log "skip exact completed final mode=$mode task=$task_id result=$existing_result"
            return 0
        fi
        log "REFUSING mixed evaluation directory: stale or mismatched result=$existing_result"
        log "use a fresh EVAL_ROOT or remove the mismatched result after auditing it"
        return 2
    done

    mkdir -p "$output_dir"

    log "start final mode=$mode task=$task_id trials=$NUM_TRIALS gpu=$gpu"
    if env -u CUDA_VISIBLE_DEVICES \
        MUJOCO_GL=egl \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
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
        "+EVALUATION.expected_fingerprint_path=$expected_fingerprint" \
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
