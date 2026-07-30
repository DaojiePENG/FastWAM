#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/phase1_h8_d30_e1_bs32}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_e1_bs32_curve2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STEPS=(223 446 669)
MODES=(interleaved vision_causal action_aggregator)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

checkpoint_ready() {
    local step="$1"
    local tag mode
    tag="$(printf 'step_%06d' "$step")"
    for mode in "${MODES[@]}"; do
        [[ -s "$TRAIN_ROOT/$mode/checkpoints/weights/$tag.pt" ]] || return 1
        [[ -s "$TRAIN_ROOT/$mode/checkpoints/state/$tag/trainer_state.json" ]] || return 1
    done
}

mkdir -p "$EVAL_ROOT"
for step in "${STEPS[@]}"; do
    tag="$(printf 'step_%06d' "$step")"
    while ! checkpoint_ready "$step"; do
        log "waiting for all $tag checkpoints"
        sleep "$POLL_SECONDS"
    done
    log "evaluating intermediate checkpoint $tag (2 episodes/task, GPUs 6-7)"
    TRAIN_ROOT="$TRAIN_ROOT" \
    EVAL_ROOT="$EVAL_ROOT/$tag" \
    FINAL_STEP="$step" \
    NUM_TRIALS=2 \
    GPU_IDS_CSV=6,7 \
    INCLUDE_BASELINE=false \
    VIDEO_LORA_ENABLED=true \
    MERGE_VIDEO_LORA=true \
    REQUIRE_TRAINING_COMPLETE=false \
    bash "$ROOT_DIR/scripts/run_phase1_eval_after_training.sh"
done
log "all intermediate checkpoint evaluations complete: $EVAL_ROOT"
