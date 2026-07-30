#!/usr/bin/env bash

set -euo pipefail

# Two strictly controlled action_aggregator screens.  Both runs use the same
# seeded initialization/data order/noise order and differ only in action/LoRA LR.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-10}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-2}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/lr_screen_seeded_v3_s${MAX_STEPS}_bs80_chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}}"
LEARNING_RATES=(1.0e-5 1.0e-4)
GPU_GROUPS=(0,1,2,3 4,5,6,7)
PORTS=(29974 29975)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$BATCH_SIZE" -ne 2 ]] || [[ "$GRAD_ACCUM" -ne 10 ]]; then
    log "paired screen contract requires 4 GPUs x batch 2 x grad accumulation 10"
    exit 1
fi

mkdir -p "$SCREEN_ROOT"
pids=()
for index in "${!LEARNING_RATES[@]}"; do
    learning_rate="${LEARNING_RATES[$index]}"
    gpu_ids="${GPU_GROUPS[$index]}"
    port="${PORTS[$index]}"
    lr_tag="${learning_rate//./p}"
    output_dir="$SCREEN_ROOT/lr${lr_tag}"
    log "launch paired LR screen lr=$learning_rate gpus=$gpu_ids output=$output_dir"
    MODE=action_aggregator \
    NUM_PROCESSES=4 \
    GPU_IDS_CSV="$gpu_ids" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_EVERY="$MAX_STEPS" \
    LEARNING_RATE="$learning_rate" \
    LR_SCHEDULER_TYPE=constant \
    VIDEO_LORA_MULTIPLIER=1.0 \
    HISTORY_VAE_BATCH_CHUNK_SIZE="$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="lr-screen-seeded-v3-action-aggregator-${lr_tag}-s${MAX_STEPS}-bs80-chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}-seed42" \
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
    log "one or both paired LR screens failed; inspect $SCREEN_ROOT/*/train.log"
    exit 1
fi

log "paired LR screens complete: $SCREEN_ROOT"
