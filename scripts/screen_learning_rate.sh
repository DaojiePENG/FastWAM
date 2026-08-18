#!/usr/bin/env bash

set -euo pipefail

# Paired learning-rate-scheduler screen. Both runs use the same 1.0e-4
# action/LoRA learning rate, seeded initialization, data order, and noise order;
# they differ only in using a constant or cosine learning-rate scheduler.

ROOT_DIR="$(git rev-parse --show-toplevel)"
MAX_STEPS="${MAX_STEPS:-100}"
# Single topology configuration point for this paired screen.
GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-0,1,2,3;4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
IFS=";" read -r -a GPU_GROUPS <<<"$GPU_GROUPS_CSV"
NUM_PROCESSES="$(awk -F, '{print NF}' <<<"${GPU_GROUPS[0]}")"
GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
RELEASE_CHECKPOINT_SHA256="${RELEASE_CHECKPOINT_SHA256:-$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/lr_scheduler_screen_incremental_v6_w${NUM_PROCESSES}_mb${BATCH_SIZE}_ga${GRAD_ACCUM}_s${MAX_STEPS}_bs${GLOBAL_BATCH}_chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}}"
CANDIDATE_NAMES=(constant cosine)
LEARNING_RATES=(1.0e-4 1.0e-4)
LR_SCHEDULER_TYPES=(constant cosine)
PORTS=(29974 29975)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if (( BATCH_SIZE <= 0 || GRAD_ACCUM <= 0 )) \
    || [[ "${#GPU_GROUPS[@]}" -ne "${#CANDIDATE_NAMES[@]}" ]] \
    || [[ "${#LEARNING_RATES[@]}" -ne "${#CANDIDATE_NAMES[@]}" ]] \
    || [[ "${#LR_SCHEDULER_TYPES[@]}" -ne "${#CANDIDATE_NAMES[@]}" ]] \
    || [[ "${#PORTS[@]}" -ne "${#CANDIDATE_NAMES[@]}" ]]; then
    log "invalid paired topology: groups=$GPU_GROUPS_CSV batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM"
    exit 1
fi
for gpu_ids in "${GPU_GROUPS[@]}"; do
    if [[ "$(awk -F, '{print NF}' <<<"$gpu_ids")" -ne "$NUM_PROCESSES" ]]; then
        log "all paired GPU groups must have the same size: $GPU_GROUPS_CSV"
        exit 1
    fi
done

mkdir -p "$SCREEN_ROOT"
pids=()
for index in "${!CANDIDATE_NAMES[@]}"; do
    candidate="${CANDIDATE_NAMES[$index]}"
    learning_rate="${LEARNING_RATES[$index]}"
    lr_scheduler_type="${LR_SCHEDULER_TYPES[$index]}"
    gpu_ids="${GPU_GROUPS[$index]}"
    port="${PORTS[$index]}"
    lr_tag="${learning_rate//./p}"
    output_dir="$SCREEN_ROOT/${candidate}_lr${lr_tag}"
    log "launch paired LR-scheduler screen scheduler=$lr_scheduler_type lr=$learning_rate gpus=$gpu_ids output=$output_dir"
    MODE=action_aggregator \
    NUM_PROCESSES="$NUM_PROCESSES" \
    GPU_IDS_CSV="$gpu_ids" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_EVERY="$MAX_STEPS" \
    LEARNING_RATE="$learning_rate" \
    LR_SCHEDULER_TYPE="$lr_scheduler_type" \
    VIDEO_LORA_MULTIPLIER=1.0 \
    HISTORY_VAE_BATCH_CHUNK_SIZE="$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    INITIAL_BLOCK_OVERSAMPLE=1 \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="lr-scheduler-screen-incremental-v6-w${NUM_PROCESSES}-mb${BATCH_SIZE}-ga${GRAD_ACCUM}-action-aggregator-${candidate}-lr${lr_tag}-s${MAX_STEPS}-bs${GLOBAL_BATCH}-chunk${HISTORY_VAE_BATCH_CHUNK_SIZE}-seed42" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_GROUP="paired-lr-scheduler-incremental-v6-w${NUM_PROCESSES}-mb${BATCH_SIZE}-ga${GRAD_ACCUM}-s${MAX_STEPS}-bs${GLOBAL_BATCH}-lr${lr_tag}-seed42" \
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
    log "one or both paired LR-scheduler screens failed; inspect $SCREEN_ROOT/*/train.log"
    exit 1
fi

log "paired LR-scheduler screens complete: $SCREEN_ROOT"
