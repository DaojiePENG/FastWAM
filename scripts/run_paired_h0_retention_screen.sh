#!/usr/bin/env bash

set -euo pipefail

# Paired action_aggregator screen for native H0 retention. Both runs start from
# the same FastWAM release and differ only in genuine episode-start frequency.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?LR_SELECTION_MANIFEST is required}"
SELECTED_LR="$("$ROOT_DIR/.venv/bin/python" \
    "$ROOT_DIR/scripts/history_audit_selection.py" validate \
    --manifest "$LR_SELECTION_MANIFEST" \
    --expected-kind learning_rate \
    --selected-value-only)"
LR_SELECTION_MANIFEST_SHA256="$(sha256sum "$LR_SELECTION_MANIFEST" | awk '{print $1}')"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
LR_TAG="${SELECTED_LR//./p}"
LR_TAG="${LR_TAG//+/_}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/h0_retention_incremental_v6_mb10_ga2_s${MAX_STEPS}_bs80_lr${LR_TAG}_chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}}"
OVERSAMPLE_FACTORS=(1 4)
GPU_GROUPS=(0,1,2,3 4,5,6,7)
PORTS=(29976 29977)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$BATCH_SIZE" -ne 10 ]] || [[ "$GRAD_ACCUM" -ne 2 ]]; then
    log "paired H0 screen requires 4 GPUs x batch 10 x grad accumulation 2"
    exit 1
fi
case "$SELECTED_LR" in
    1.0e-5|1.0e-4) ;;
    *)
        log "SELECTED_LR must be one of the paired screen candidates; got $SELECTED_LR"
        exit 1
        ;;
esac

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
    LEARNING_RATE="$SELECTED_LR" \
    LR_SCHEDULER_TYPE=constant \
    VIDEO_LORA_MULTIPLIER=1.0 \
    HISTORY_VAE_BATCH_CHUNK_SIZE="$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    INITIAL_BLOCK_OVERSAMPLE="$factor" \
    LR_SELECTION_MANIFEST_SHA256="$LR_SELECTION_MANIFEST_SHA256" \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="h0-retention-incremental-v6-mb10-ga2-action-aggregator-x${factor}-s${MAX_STEPS}-bs80-lr${LR_TAG}-seed42" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_GROUP="paired-h0-incremental-v6-mb10-ga2-s${MAX_STEPS}-bs80-lr${LR_TAG}-seed42" \
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
