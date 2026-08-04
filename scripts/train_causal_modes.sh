#!/usr/bin/env bash

set -euo pipefail

# Candidate canonical controlled causal-mode training. Every mode uses the same
# eight-rank B20/GA1 topology, sampler sharding, global batches, RNG streams, and
# optimizer updates. This topology must pass the real strict-window ZeRO-2 capacity
# smoke before it is frozen for formal results. The optional MODES_CSV subset
# permits an effect audit between expensive stages; invoking the remaining modes
# later with the same TRAIN_ROOT preserves identical per-mode run contracts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$ROOT_DIR/.venv/bin/accelerate}"
SELECTED_LR="${LEARNING_RATE:-1.0e-4}"
INITIAL_BLOCK_OVERSAMPLE="${INITIAL_BLOCK_OVERSAMPLE:-4}"
DATASET_STATS="${DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-179}"
BATCH_SIZE="${BATCH_SIZE:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
HISTORY_WINDOW_BLOCKS="${HISTORY_WINDOW_BLOCKS:-8}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-42}"
LR_TAG="${SELECTED_LR//./p}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/causal_strict_window_v7_w${HISTORY_WINDOW_BLOCKS}_b20_ga1_d30_s${MAX_STEPS}_bs160_cosine_lr${LR_TAG}}"
MODES_CSV="${MODES_CSV:-action_aggregator,interleaved,vision_causal}"
IFS=',' read -r -a MODES <<<"$MODES_CSV"
CANONICAL_MODES=(action_aggregator interleaved vision_causal)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "$NUM_PROCESSES" -ne 8 ]] || [[ "$BATCH_SIZE" -ne 20 ]] || [[ "$GRAD_ACCUM" -ne 1 ]]; then
    log "candidate formal comparison requires 8 GPUs x batch 20 x grad accumulation 1 (global batch 160); run the matching strict-window capacity smoke first"
    exit 1
fi
if ! [[ "$HISTORY_WINDOW_BLOCKS" =~ ^[1-9][0-9]*$ ]] \
    || (( HISTORY_WINDOW_BLOCKS > 70 )); then
    log "history window must be a positive integer no greater than 70; got $HISTORY_WINDOW_BLOCKS"
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
if [[ ! -s "$RELEASE_CHECKPOINT" ]] || [[ ! -s "$DATASET_STATS" ]]; then
    log "missing release checkpoint or dataset statistics"
    exit 1
fi
ACTUAL_RELEASE_CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')"
if [[ -n "${RELEASE_CHECKPOINT_SHA256:-}" ]] \
    && [[ "$RELEASE_CHECKPOINT_SHA256" != "$ACTUAL_RELEASE_CHECKPOINT_SHA256" ]]; then
    log "configured release checkpoint hash does not match the checkpoint bytes"
    exit 1
fi
RELEASE_CHECKPOINT_SHA256="$ACTUAL_RELEASE_CHECKPOINT_SHA256"
DATASET_STATS_SHA256="$(sha256sum "$DATASET_STATS" | awk '{print $1}')"
CODE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
if [[ "$GLOBAL_BATCH" -ne 160 ]]; then
    log "candidate formal comparison requires global batch 160; got $GLOBAL_BATCH"
    exit 1
fi

validate_existing_contract_group() {
    local mode mode_dir contract_file
    local -a contract_args=()
    for mode in "${CANONICAL_MODES[@]}"; do
        mode_dir="$TRAIN_ROOT/$mode"
        contract_file="$mode_dir/run_contract.txt"
        if [[ -e "$contract_file" && ! -s "$contract_file" ]]; then
            log "invalid empty run contract: $contract_file"
            return 2
        fi
        if [[ -d "$mode_dir" ]] \
            && [[ -n "$(find "$mode_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
            && [[ ! -s "$contract_file" ]]; then
            log "refusing uncontracted existing mode directory: $mode_dir"
            return 2
        fi
        if [[ -s "$contract_file" ]]; then
            contract_args+=(--contract "$mode=$contract_file")
        fi
    done
    if (( ${#contract_args[@]} == 0 )); then
        return 0
    fi
    "$PYTHON_BIN" \
        "$ROOT_DIR/scripts/validate_run_contract_group.py" \
        "${contract_args[@]}" \
        --expected-field "code_commit=$CODE_COMMIT" \
        --expected-field "release_checkpoint_sha256=$RELEASE_CHECKPOINT_SHA256" \
        --expected-field "dataset_stats_sha256=$DATASET_STATS_SHA256" \
        --expected-field "num_processes=$NUM_PROCESSES" \
        --expected-field "batch_size=$BATCH_SIZE" \
        --expected-field "gradient_accumulation_steps=$GRAD_ACCUM" \
        --expected-field "global_batch=$GLOBAL_BATCH" \
        --expected-field "max_steps=$MAX_STEPS" \
        --expected-field "learning_rate=$SELECTED_LR" \
        --expected-field lr_scheduler_type=cosine \
        --expected-field "history_vae_batch_chunk_size=$HISTORY_VAE_BATCH_CHUNK_SIZE" \
        --expected-field "initial_block_oversample=$INITIAL_BLOCK_OVERSAMPLE" \
        --expected-field history_training_mode=strict_replay_window_bptt \
        --expected-field history_sampling_mode=recent_window \
        --expected-field "history_window_blocks=$HISTORY_WINDOW_BLOCKS" \
        --expected-field "save_every=$SAVE_EVERY" \
        --expected-field "seed=$SEED" \
        --expected-field padding_attention_mask=true \
        --expected-field training_exit_depths=30 \
        --output "$TRAIN_ROOT/run_contract_group.validation.json" \
        >/dev/null
}

existing_asset_manifest_sha() {
    local mode contract_file value
    for mode in "${CANONICAL_MODES[@]}"; do
        contract_file="$TRAIN_ROOT/$mode/run_contract.txt"
        if [[ -s "$contract_file" ]]; then
            value="$(awk -F= '$1 == "training_asset_manifest_sha256" {print $2}' "$contract_file")"
            if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
                log "invalid training asset identity in $contract_file"
                return 2
            fi
            printf '%s\n' "$value"
            return 0
        fi
    done
}

validate_existing_contract_group
for mode in "${MODES[@]}"; do
    EXPECTED_ASSET_MANIFEST_SHA256="$(existing_asset_manifest_sha)"
    output_dir="$TRAIN_ROOT/$mode"
    log "start controlled full-BPTT mode=$mode output=$output_dir topology=8xb20xga1 global_batch=160 max_steps=$MAX_STEPS save_every=$SAVE_EVERY"
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
    HISTORY_WINDOW_BLOCKS="$HISTORY_WINDOW_BLOCKS" \
    INITIAL_BLOCK_OVERSAMPLE="$INITIAL_BLOCK_OVERSAMPLE" \
    EXPECTED_TRAINING_ASSET_MANIFEST_SHA256="$EXPECTED_ASSET_MANIFEST_SHA256" \
    REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true \
    RELEASE_CHECKPOINT="$RELEASE_CHECKPOINT" \
    RELEASE_CHECKPOINT_SHA256="$RELEASE_CHECKPOINT_SHA256" \
    DATASET_STATS="$DATASET_STATS" \
    SEED="$SEED" \
    OUTPUT_DIR="$output_dir" \
    PYTHON_BIN="$PYTHON_BIN" \
    ACCELERATE_BIN="$ACCELERATE_BIN" \
    RUN_NAME="causal-strict-window-v7-w${HISTORY_WINDOW_BLOCKS}-b20-ga1-d30-s${MAX_STEPS}-${mode//_/-}-bs160-cosine-lr${LR_TAG}-seed${SEED}" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_MODE="$WANDB_MODE" \
    MAIN_PROCESS_PORT=29971 \
        bash "$ROOT_DIR/scripts/train_leapbot.sh"
    validate_existing_contract_group
    log "complete controlled full-BPTT mode=$mode"
done

log "selected causal modes complete ($MODES_CSV): $TRAIN_ROOT"
