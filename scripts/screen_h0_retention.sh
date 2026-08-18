#!/usr/bin/env bash

set -euo pipefail

# Canonical paired action_aggregator screen for native H0 retention. Both runs start from
# the same FastWAM release and differ only in genuine episode-start frequency.

ROOT_DIR="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${LEAPBOT_PYTHON:-$(command -v python 2>/dev/null || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Python is unavailable; activate Conda/uv or set LEAPBOT_PYTHON.\n' >&2
    exit 2
fi
MAX_STEPS="${MAX_STEPS:-100}"
# Single topology configuration point for this paired screen.
GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-0,1,2,3;4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
IFS=";" read -r -a GPU_GROUPS <<<"$GPU_GROUPS_CSV"
NUM_PROCESSES="$(awk -F, '{print NF}' <<<"${GPU_GROUPS[0]}")"
GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?LR_SELECTION_MANIFEST is required}"
SELECTED_LR="$("$PYTHON_BIN" \
    "$ROOT_DIR/scripts/history_audit_selection.py" validate \
    --manifest "$LR_SELECTION_MANIFEST" \
    --expected-kind learning_rate \
    --allowed-basis fixed_noise_audit \
    --allowed-basis user_directed \
    --selected-value-only)"
LR_SELECTION_MANIFEST_SHA256="$(sha256sum "$LR_SELECTION_MANIFEST" | awk '{print $1}')"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
LR_TAG="${SELECTED_LR//./p}"
LR_TAG="${LR_TAG//+/_}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/h0_retention_incremental_v6_w${NUM_PROCESSES}_mb${BATCH_SIZE}_ga${GRAD_ACCUM}_s${MAX_STEPS}_bs${GLOBAL_BATCH}_lr${LR_TAG}_chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}}"
OVERSAMPLE_FACTORS=(1 4)
PORTS=(29976 29977)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if (( BATCH_SIZE <= 0 || GRAD_ACCUM <= 0 )) || [[ "${#GPU_GROUPS[@]}" -ne "${#OVERSAMPLE_FACTORS[@]}" ]]; then
    log "invalid paired topology: groups=$GPU_GROUPS_CSV batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
    exit 1
fi
for gpu_ids in "${GPU_GROUPS[@]}"; do
    if [[ "$(awk -F, '{print NF}' <<<"$gpu_ids")" -ne "$NUM_PROCESSES" ]]; then
        log "all paired GPU groups must have the same size: $GPU_GROUPS_CSV"
        exit 1
    fi
done

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
    NUM_PROCESSES="$NUM_PROCESSES" \
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
    RUN_NAME="h0-retention-incremental-v6-w${NUM_PROCESSES}-mb${BATCH_SIZE}-ga${GRAD_ACCUM}-action-aggregator-x${factor}-s${MAX_STEPS}-bs${GLOBAL_BATCH}-lr${LR_TAG}-seed42" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_GROUP="paired-h0-incremental-v6-w${NUM_PROCESSES}-mb${BATCH_SIZE}-ga${GRAD_ACCUM}-s${MAX_STEPS}-bs${GLOBAL_BATCH}-lr${LR_TAG}-seed42" \
    REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true \
    MAIN_PROCESS_PORT="$port" \
        bash "$ROOT_DIR/scripts/train_leapbot.sh" &
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
