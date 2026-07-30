#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/action_aggregator_h8_e5_bs72_lr2e5}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/action_aggregator_h8_e5_bs72_lr2e5_curve2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-397}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MODE=action_aggregator

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

checkpoint_ready() {
    local step="$1"
    local tag
    tag="$(printf 'step_%06d' "$step")"
    [[ -s "$TRAIN_ROOT/$MODE/checkpoints/weights/$tag.pt" ]] \
        && [[ -s "$TRAIN_ROOT/$MODE/checkpoints/state/$tag/trainer_state.json" ]]
}

mkdir -p "$EVAL_ROOT"
log "GPUs 6-7 reserved for action_aggregator learning-curve evaluation"

for epoch in $(seq 1 "$NUM_EPOCHS"); do
    step=$((epoch * STEPS_PER_EPOCH))
    # The final five-epoch checkpoint receives the full 10x10 evaluation after
    # recipe selection; curve screening covers epochs 1-4 at two trials/task.
    if (( epoch == NUM_EPOCHS )); then
        break
    fi
    tag="$(printf 'step_%06d' "$step")"
    while ! checkpoint_ready "$step"; do
        log "waiting for action_aggregator epoch=$epoch checkpoint=$tag"
        sleep "$POLL_SECONDS"
    done
    log "evaluating action_aggregator epoch=$epoch checkpoint=$tag (2 episodes/task)"
    TRAIN_ROOT="$TRAIN_ROOT" \
    EVAL_ROOT="$EVAL_ROOT/$tag" \
    MODE="$MODE" \
    FINAL_STEP="$step" \
    NUM_TRIALS=2 \
    GPU_IDS_CSV=6,7 \
    bash "$ROOT_DIR/scripts/run_single_mode_checkpoint_eval.sh"
done
log "action_aggregator epoch 1-4 intermediate evaluations complete: $EVAL_ROOT"
