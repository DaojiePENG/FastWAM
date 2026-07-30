#!/usr/bin/env bash

set -euo pipefail

# Controlled comparison. Every selected mode uses the same eight-rank topology,
# sampler sharding, global batches, RNG streams, and optimizer updates.  The
# optional MODES_CSV subset permits an effect audit between expensive stages;
# invoking the remaining modes later with the same TRAIN_ROOT preserves the
# identical per-mode run contracts.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
SELECTED_LR="${SELECTED_LR:?SELECTED_LR must come from the paired LR audit}"
MAX_STEPS="${MAX_STEPS:-1115}"
SAVE_EVERY="${SAVE_EVERY:-223}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
INITIAL_BLOCK_OVERSAMPLE="${INITIAL_BLOCK_OVERSAMPLE:?INITIAL_BLOCK_OVERSAMPLE must come from the H0-retention audit}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
LR_TAG="${SELECTED_LR//./p}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/causal_incremental_full_bptt_v3_d30_e5_bs128_cosine_lr${LR_TAG}}"
MODES_CSV="${MODES_CSV:-action_aggregator,interleaved,vision_causal}"
IFS=',' read -r -a MODES <<<"$MODES_CSV"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$NUM_PROCESSES" -ne 8 ]] || [[ "$BATCH_SIZE" -ne 1 ]] || [[ "$GRAD_ACCUM" -ne 16 ]]; then
    log "formal comparison requires 8 GPUs x batch 1 x grad accumulation 16 (global batch 128)"
    exit 1
fi
if (( ${#MODES[@]} == 0 )); then
    log "at least one causal mode must be selected"
    exit 1
fi
declare -A seen_modes=()
for mode in "${MODES[@]}"; do
    case "$mode" in
        action_aggregator|interleaved|vision_causal) ;;
        *)
            log "invalid causal mode in MODES_CSV: $mode"
            exit 1
            ;;
    esac
    if [[ -n "${seen_modes[$mode]:-}" ]]; then
        log "duplicate causal mode in MODES_CSV: $mode"
        exit 1
    fi
    seen_modes[$mode]=1
done

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
    INITIAL_BLOCK_OVERSAMPLE="$INITIAL_BLOCK_OVERSAMPLE" \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="causal-incremental-full-bptt-v3-d30-e5-${mode//_/-}-bs128-cosine-lr${LR_TAG}-seed42" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    MAIN_PROCESS_PORT=29971 \
        bash "$ROOT_DIR/scripts/run_hierarchical_raw_v1_peft_5k.sh"
    log "complete controlled full-BPTT mode=$mode"
done

log "selected causal modes complete ($MODES_CSV): $TRAIN_ROOT"
