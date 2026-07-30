#!/usr/bin/env bash

set -euo pipefail

# Sequential, controlled comparison. Running all three modes with the same
# eight-rank topology keeps sampler sharding, global batches, RNG streams, and
# optimizer updates comparable.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
SELECTED_LR="${SELECTED_LR:?SELECTED_LR must come from the paired LR audit}"
MAX_STEPS="${MAX_STEPS:-1115}"
SAVE_EVERY="${SAVE_EVERY:-223}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
LR_TAG="${SELECTED_LR//./p}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/causal_full_bptt_d30_e5_bs128_cosine_lr${LR_TAG}}"
MODES=(action_aggregator interleaved vision_causal)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$NUM_PROCESSES" -ne 8 ]] || [[ "$BATCH_SIZE" -ne 2 ]] || [[ "$GRAD_ACCUM" -ne 8 ]]; then
    log "formal comparison requires 8 GPUs x batch 2 x grad accumulation 8 (global batch 128)"
    exit 1
fi

mkdir -p "$TRAIN_ROOT"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
for mode in "${MODES[@]}"; do
    output_dir="$TRAIN_ROOT/$mode"
    log "start controlled full-BPTT mode=$mode output=$output_dir"
    MODE="$mode" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    GPU_IDS_CSV="$GPU_IDS_CSV" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_EVERY="$SAVE_EVERY" \
    LEARNING_RATE="$SELECTED_LR" \
    LR_SCHEDULER_TYPE=cosine \
    VIDEO_LORA_MULTIPLIER=1.0 \
    HISTORY_VAE_BATCH_CHUNK_SIZE="$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="causal-full-bptt-d30-e5-${mode//_/-}-bs128-cosine-lr${LR_TAG}-seed42" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    MAIN_PROCESS_PORT=29971 \
        bash "$ROOT_DIR/scripts/run_hierarchical_raw_v1_peft_5k.sh"
    log "complete controlled full-BPTT mode=$mode"
done

log "all causal modes complete: $TRAIN_ROOT"
