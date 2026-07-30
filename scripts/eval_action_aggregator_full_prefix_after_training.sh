#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/action_aggregator_full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/correct_full_prefix_comparison/temporal_rope_v2/action_aggregator_step1115}"
MODE=action_aggregator
FINAL_STEP="${FINAL_STEP:-1115}"
NUM_TRIALS="${NUM_TRIALS:-10}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-2048}"
FINAL_TAG="step_$(printf '%06d' "$FINAL_STEP")"
CHECKPOINT="$TRAIN_ROOT/$MODE/checkpoints/weights/$FINAL_TAG.pt"
STATE_DIR="$TRAIN_ROOT/$MODE/checkpoints/state/$FINAL_TAG"
TRAIN_LOG="$TRAIN_ROOT/$MODE/train.log"

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

mkdir -p "$EVAL_ROOT"
while [[ ! -s "$CHECKPOINT" ]] \
    || ! grep -q "max_steps reached step=$FINAL_STEP" "$TRAIN_LOG" 2>/dev/null; do
    log "waiting for correct full-prefix checkpoint $FINAL_TAG"
    sleep "$POLL_SECONDS"
done

while ! selected_gpus_are_free; do
    log "checkpoint ready; waiting for all evaluation GPUs to be released"
    sleep "$POLL_SECONDS"
done

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$CHECKPOINT" \
    --expected-step "$FINAL_STEP" \
    --expected-mode "$MODE" \
    --state-dir "$STATE_DIR" \
    --output "$EVAL_ROOT/checkpoint_validation.json"
log "starting 10-task x $NUM_TRIALS real-memory rollout for $MODE/$FINAL_TAG"
TRAIN_ROOT="$TRAIN_ROOT" \
EVAL_ROOT="$EVAL_ROOT" \
MODE="$MODE" \
FINAL_STEP="$FINAL_STEP" \
NUM_TRIALS="$NUM_TRIALS" \
GPU_IDS_CSV="$GPU_IDS_CSV" \
MEMORY_ENABLED=true \
MAX_HISTORY_BLOCKS=70 \
bash "$ROOT_DIR/scripts/run_single_mode_checkpoint_eval.sh"
log "correct action_aggregator evaluation complete: $EVAL_ROOT/pareto/results.csv"
