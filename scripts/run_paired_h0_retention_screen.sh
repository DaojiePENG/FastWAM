#!/usr/bin/env bash

set -euo pipefail

# Paired action_aggregator screen for native H0 retention. Both runs start from
# the same FastWAM release and differ only in genuine episode-start frequency.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-10}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-2}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/h0_retention_relative_v2_s${MAX_STEPS}_bs80_lr1e-4_chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}}"
OVERSAMPLE_FACTORS=(1 4)
GPU_GROUPS=(0,1,2,3 4,5,6,7)
PORTS=(29976 29977)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$BATCH_SIZE" -ne 2 ]] || [[ "$GRAD_ACCUM" -ne 10 ]]; then
    log "paired H0 screen requires 4 GPUs x batch 2 x grad accumulation 10"
    exit 1
fi
if [[ "$LEARNING_RATE" != "1.0e-4" ]]; then
    log "paired H0 screen contract requires LEARNING_RATE=1.0e-4"
    exit 1
fi

mkdir -p "$SCREEN_ROOT"
pids=()
for index in "${!OVERSAMPLE_FACTORS[@]}"; do
    factor="${OVERSAMPLE_FACTORS[$index]}"
    gpu_ids="${GPU_GROUPS[$index]}"
    port="${PORTS[$index]}"
    output_dir="$SCREEN_ROOT/h0x${factor}"
    log "launch paired H0 screen factor=$factor gpus=$gpu_ids output=$output_dir"
    MODE=action_aggregator \
    NUM_PROCESSES=4 \
    GPU_IDS_CSV="$gpu_ids" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_EVERY="$MAX_STEPS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LR_SCHEDULER_TYPE=constant \
    VIDEO_LORA_MULTIPLIER=1.0 \
    HISTORY_VAE_BATCH_CHUNK_SIZE="$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    INITIAL_BLOCK_OVERSAMPLE="$factor" \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="h0-retention-relative-v2-action-aggregator-x${factor}-s${MAX_STEPS}-bs80-lr1e-4-seed42" \
    WANDB_ENABLED=false \
    WANDB_MODE=disabled \
    MAIN_PROCESS_PORT="$port" \
        bash "$ROOT_DIR/scripts/run_hierarchical_raw_v1_peft_5k.sh" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    log "one or both paired H0 screens failed; inspect $SCREEN_ROOT/*/train.log"
    exit 1
fi

log "paired H0 retention screens complete: $SCREEN_ROOT"
