#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
SOURCE_TRAIN_ROOT="${SOURCE_TRAIN_ROOT:?SOURCE_TRAIN_ROOT is required}"
MODE="${MODE:?MODE is required}"
SOURCE_STEP="${SOURCE_STEP:?SOURCE_STEP is required}"
MULTI_EXIT_LR="${MULTI_EXIT_LR:?MULTI_EXIT_LR is required}"
MAX_STEPS="${MAX_STEPS:?MAX_STEPS is required}"
INITIAL_BLOCK_OVERSAMPLE="${INITIAL_BLOCK_OVERSAMPLE:?INITIAL_BLOCK_OVERSAMPLE is required}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/multi_exit_incremental_full_bptt}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
SAVE_EVERY="${SAVE_EVERY:-$MAX_STEPS}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
SOURCE_TAG="$(printf 'step_%06d' "$SOURCE_STEP")"
SOURCE_CHECKPOINT="$SOURCE_TRAIN_ROOT/$MODE/checkpoints/weights/$SOURCE_TAG.pt"
SOURCE_STATE="$SOURCE_TRAIN_ROOT/$MODE/checkpoints/state/$SOURCE_TAG"
SOURCE_CONTRACT="$SOURCE_TRAIN_ROOT/$MODE/run_contract.txt"
OUTPUT_DIR="$TRAIN_ROOT/$MODE"

contract_value() {
    local key="$1"
    awk -F= -v key="$key" '$1 == key {print $2}' "$SOURCE_CONTRACT"
}

if [[ ! -s "$SOURCE_CHECKPOINT" || ! -s "$SOURCE_CONTRACT" ]]; then
    printf 'Winning D30 checkpoint or run contract is missing: %s\n' \
        "$SOURCE_CHECKPOINT" >&2
    exit 2
fi
SOURCE_RUN_CONTRACT_SHA256="$(awk -F= '$1 == "run_contract_sha256" {print $2}' "$SOURCE_CONTRACT")"
SOURCE_CODE_COMMIT="$(awk -F= '$1 == "code_commit" {print $2}' "$SOURCE_CONTRACT")"
if [[ ! "$SOURCE_RUN_CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$SOURCE_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Invalid source training identity in %s\n' "$SOURCE_CONTRACT" >&2
    exit 2
fi
if [[ "$NUM_PROCESSES" -ne 8 ]] || [[ "$BATCH_SIZE" -ne 1 ]] \
    || [[ "$GRAD_ACCUM" -ne 16 ]]; then
    printf 'Multi-exit training requires 8 GPUs x batch 1 x grad accumulation 16.\n' >&2
    exit 2
fi
for expected_field in \
    num_processes=8 \
    batch_size=1 \
    gradient_accumulation_steps=16 \
    history_vae_batch_chunk_size=1 \
    history_training_mode=incremental_full_bptt \
    full_episode_history=true \
    max_history_blocks=70 \
    replan_steps=10 \
    action_horizon=32 \
    training_exit_depths=30; do
    key="${expected_field%%=*}"
    expected="${expected_field#*=}"
    actual="$(contract_value "$key")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'Source contract mismatch for %s: expected=%s actual=%s\n' \
            "$key" "$expected" "$actual" >&2
        exit 2
    fi
done
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$SOURCE_CHECKPOINT" \
    --expected-step "$SOURCE_STEP" \
    --expected-mode "$MODE" \
    --expected-trained-exit-depths 30 \
    --expected-history-vae-batch-chunk-size 1 \
    --expected-run-contract-sha256 "$SOURCE_RUN_CONTRACT_SHA256" \
    --expected-code-commit "$SOURCE_CODE_COMMIT" \
    --state-dir "$SOURCE_STATE" \
    --output "$SOURCE_TRAIN_ROOT/$MODE/checkpoint_validation.json" \
    >/dev/null

MODE="$MODE" \
NUM_PROCESSES="$NUM_PROCESSES" \
GPU_IDS_CSV="$GPU_IDS_CSV" \
BATCH_SIZE="$BATCH_SIZE" \
GRAD_ACCUM="$GRAD_ACCUM" \
MAX_STEPS="$MAX_STEPS" \
SAVE_EVERY="$SAVE_EVERY" \
LEARNING_RATE="$MULTI_EXIT_LR" \
LR_SCHEDULER_TYPE=cosine \
VIDEO_LORA_MULTIPLIER=1.0 \
HISTORY_VAE_BATCH_CHUNK_SIZE=1 \
INITIAL_BLOCK_OVERSAMPLE="$INITIAL_BLOCK_OVERSAMPLE" \
INITIAL_CHECKPOINT="$SOURCE_CHECKPOINT" \
TRAINING_EXIT_DEPTHS_CSV=8,16,24,30 \
REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_NAME="multi-exit-incremental-full-bptt-${MODE//_/-}-s${MAX_STEPS}-seed42" \
WANDB_ENABLED="$WANDB_ENABLED" \
WANDB_MODE="$WANDB_MODE" \
MAIN_PROCESS_PORT=29972 \
    bash "$ROOT_DIR/scripts/run_hierarchical_raw_v1_peft_5k.sh"
