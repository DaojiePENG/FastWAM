#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/remaining_modes_full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5}"
EVAL_BASE="${EVAL_BASE:-$ROOT_DIR/evaluate_results/correct_full_prefix_comparison/temporal_rope_v2}"
ACTION_EVAL_ROOT="${ACTION_EVAL_ROOT:-$EVAL_BASE/action_aggregator_step1115}"
BASELINE_SOURCE="${BASELINE_SOURCE:-$ROOT_DIR/evaluate_results/phase1_h8_d30_s1000_dev10/fastwam_release}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$EVAL_BASE/all_modes_step1115}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_TRIALS="${NUM_TRIALS:-10}"
FINAL_STEP="${FINAL_STEP:-1115}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-2048}"
MODES=(interleaved vision_causal)
PORTS=(29963 29964)

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

eval_is_complete() {
    local eval_root="$1"
    [[ -s "$eval_root/pareto/results.csv" ]] || return 1
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
        "$eval_root" \
        --output-dir "$eval_root/pareto" \
        --expected-tasks 10 \
        --expected-trials-per-task "$NUM_TRIALS" \
        --require-profiled \
        >/dev/null 2>&1
}

run_smoke() {
    local mode="$1"
    local output_dir="$ROOT_DIR/runs/smoke_full_prefix_h50_temporal_rope_v2_$mode"
    local result="$output_dir/full_prefix_smoke.json"
    if [[ -s "$result" ]] \
        && [[ "$(jq -r '.causal_mode // empty' "$result")" == "$mode" ]] \
        && [[ "$(jq -r '.history_blocks // -1' "$result")" == "50" ]] \
        && [[ "$(jq -r '.finite_gradients // false' "$result")" == "true" ]]; then
        log "skip valid H50 smoke mode=$mode result=$result"
        return 0
    fi
    while ! selected_gpus_are_free; do
        log "waiting for GPUs before H50 smoke mode=$mode"
        sleep "$POLL_SECONDS"
    done
    mkdir -p "$output_dir"
    log "start real-6B H50 forward/backward smoke mode=$mode gpu=0"
    CUDA_VISIBLE_DEVICES=0 \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/full_prefix_smoke.py" \
        task=libero_leapbot_2cam224 \
        "output_dir=$output_dir" \
        "model.causal_mode=$mode" \
        model.history_training_mode=incremental_detached_prefix \
        model.training_strategy=video_lora_action_full \
        model.video_lora.enabled=true \
        model.video_lora.rank=16 \
        model.video_lora.alpha=16.0 \
        model.video_lora.dropout=0.0 \
        model.video_lora.learning_rate_multiplier=10.0 \
        data.train.full_episode_history=true \
        data.train.min_history_blocks=0 \
        data.train.max_history_blocks=70 \
        'model.training_exit_depths=[30]' \
        "+smoke.checkpoint=$RELEASE_CHECKPOINT" \
        +smoke.device=cuda:0 \
        +smoke.history_blocks=50 \
        +smoke.compare_native=false \
        >"$output_dir/smoke.log" 2>&1
    jq -e --arg mode "$mode" \
        '.causal_mode == $mode and .history_blocks == 50 and .finite_gradients == true' \
        "$result" >/dev/null
    log "H50 smoke complete mode=$mode result=$result"
}

mkdir -p "$TRAIN_ROOT" "$EVAL_BASE" "$COMPARISON_ROOT"
while ! eval_is_complete "$ACTION_EVAL_ROOT"; do
    log "waiting for completed action_aggregator 10x10 real-memory rollout"
    sleep "$POLL_SECONDS"
done

for index in "${!MODES[@]}"; do
    mode="${MODES[$index]}"
    eval_root="$EVAL_BASE/${mode}_step${FINAL_STEP}"
    run_smoke "$mode"
    MODE="$mode" \
    TRAIN_ROOT="$TRAIN_ROOT" \
    GPU_IDS_CSV="$GPU_IDS_CSV" \
    MAIN_PROCESS_PORT="${PORTS[$index]}" \
    RUN_NAME="${mode//_/-}-full-prefix-temporal-rope-v2-h70-e5-h800-bs128-lr2e5-seed42" \
    bash "$ROOT_DIR/scripts/run_full_prefix_mode_e5.sh"

    while ! selected_gpus_are_free; do
        log "training complete mode=$mode; waiting for rollout GPUs"
        sleep "$POLL_SECONDS"
    done
    log "start 10-task x $NUM_TRIALS real-memory rollout mode=$mode"
    TRAIN_ROOT="$TRAIN_ROOT" \
    EVAL_ROOT="$eval_root" \
    MODE="$mode" \
    FINAL_STEP="$FINAL_STEP" \
    NUM_TRIALS="$NUM_TRIALS" \
    GPU_IDS_CSV="$GPU_IDS_CSV" \
    VIDEO_LORA_ENABLED=true \
    MERGE_VIDEO_LORA=true \
    MEMORY_ENABLED=true \
    MAX_HISTORY_BLOCKS=70 \
    bash "$ROOT_DIR/scripts/run_single_mode_checkpoint_eval.sh"
    eval_is_complete "$eval_root"
    log "formal rollout complete mode=$mode results=$eval_root/pareto/results.csv"
done

mkdir -p \
    "$COMPARISON_ROOT/fastwam_release" \
    "$COMPARISON_ROOT/action_aggregator" \
    "$COMPARISON_ROOT/interleaved" \
    "$COMPARISON_ROOT/vision_causal"
cp -a "$BASELINE_SOURCE/." "$COMPARISON_ROOT/fastwam_release/"
cp -a "$ACTION_EVAL_ROOT/action_aggregator/." "$COMPARISON_ROOT/action_aggregator/"
for mode in "${MODES[@]}"; do
    cp -a "$EVAL_BASE/${mode}_step${FINAL_STEP}/$mode/." "$COMPARISON_ROOT/$mode/"
done

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
    "$COMPARISON_ROOT" \
    --output-dir "$COMPARISON_ROOT/pareto" \
    --expected-tasks 10 \
    --expected-trials-per-task "$NUM_TRIALS" \
    --require-profiled
MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/plot_pareto.py" \
    "$COMPARISON_ROOT/pareto" \
    --output-dir "$COMPARISON_ROOT/pareto"
log "10-trial development comparison complete: $COMPARISON_ROOT/pareto/results.csv"
GPU_IDS_CSV="$GPU_IDS_CSV" \
NUM_TRIALS=50 \
    bash "$ROOT_DIR/scripts/run_final_50_trial_comparison.sh"
log "full final comparison complete"
