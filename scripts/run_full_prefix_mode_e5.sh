#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
MODE="${MODE:-action_aggregator}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5}"
DATASET_SIZE="${DATASET_SIZE:-28523}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
VIDEO_LORA_MULTIPLIER="${VIDEO_LORA_MULTIPLIER:-10.0}"
WANDB_ENTITY="${WANDB_ENTITY:-pengdaojie-the-hong-kong-university-of-science-and-techn}"
WANDB_PROJECT="${WANDB_PROJECT:-leapbot-va}"
WANDB_GROUP="${WANDB_GROUP:-leapbot-full-prefix-temporal-rope-v2-causal-modes-seed42}"
MODE_DASHED="${MODE//_/-}"
RUN_NAME="${RUN_NAME:-$MODE_DASHED-full-prefix-temporal-rope-v2-h70-e5-h800-bs128-lr2e5-seed42}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29963}"
MAX_PREFLIGHT_USED_MIB="${MAX_PREFLIGHT_USED_MIB:-2048}"
POLL_SECONDS="${POLL_SECONDS:-30}"

case "$MODE" in
    action_aggregator|interleaved|vision_causal) ;;
    *)
        printf 'Unsupported causal mode: %s\n' "$MODE" >&2
        exit 2
        ;;
esac

GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$(((DATASET_SIZE + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
FINAL_STEP=$((STEPS_PER_EPOCH * NUM_EPOCHS))
OUTPUT_DIR="$TRAIN_ROOT/$MODE"
LOG_FILE="$OUTPUT_DIR/train.log"
FINAL_TAG="step_$(printf '%06d' "$FINAL_STEP")"
FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoints/weights/$FINAL_TAG.pt"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

latest_state() {
    { find "$OUTPUT_DIR/checkpoints/state" -mindepth 1 -maxdepth 1 \
        -type d -name 'step_*' 2>/dev/null || true; } | sort | tail -1
}

selected_gpus_are_free() {
    local expected_gpu_count used
    expected_gpu_count="$(awk -F, '{print NF}' <<<"$GPU_IDS_CSV")"
    if [[ "$expected_gpu_count" -ne "$NUM_PROCESSES" ]]; then
        log "GPU list/process mismatch: ids=$GPU_IDS_CSV processes=$NUM_PROCESSES"
        return 1
    fi
    while IFS= read -r used; do
        (( used <= MAX_PREFLIGHT_USED_MIB )) || return 1
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v ids="$GPU_IDS_CSV" 'BEGIN {split(ids, a, ","); for (i in a) wanted[a[i]]=1} wanted[NR-1] {print $1}')
}

mkdir -p \
    "$OUTPUT_DIR" \
    "$ROOT_DIR/.cache/wandb/config" \
    "$ROOT_DIR/.cache/wandb/cache" \
    "$ROOT_DIR/.cache/wandb/data"

if [[ -s "$FINAL_CHECKPOINT" ]] \
    && grep -q "max_steps reached step=$FINAL_STEP" "$LOG_FILE" 2>/dev/null; then
    log "skip completed full-prefix training mode=$MODE checkpoint=$FINAL_CHECKPOINT"
    exit 0
fi

RESUME_PATH="$(latest_state)"
if [[ -z "$RESUME_PATH" ]]; then
    RESUME_PATH="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
    : >"$LOG_FILE"
else
    log "resume mode=$MODE from $RESUME_PATH" >>"$LOG_FILE"
fi

while ! selected_gpus_are_free; do
    log "waiting for selected training GPUs: $GPU_IDS_CSV"
    sleep "$POLL_SECONDS"
done

log "start full-prefix training mode=$MODE epochs=$NUM_EPOCHS gpus=$GPU_IDS_CSV micro_batch=$BATCH_SIZE global_batch=$GLOBAL_BATCH steps_per_epoch=$STEPS_PER_EPOCH final_step=$FINAL_STEP base_lr=$LEARNING_RATE video_lora_lr_multiplier=$VIDEO_LORA_MULTIPLIER resume=$RESUME_PATH"

CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_CONFIG_DIR="$ROOT_DIR/.cache/wandb/config" \
    WANDB_CACHE_DIR="$ROOT_DIR/.cache/wandb/cache" \
    WANDB_DATA_DIR="$ROOT_DIR/.cache/wandb/data" \
    WANDB_DIR="$OUTPUT_DIR" \
    WANDB_RUN_ID="$RUN_NAME" \
    WANDB_RESUME=allow \
    "$ROOT_DIR/.venv/bin/accelerate" launch \
    --config_file "$ROOT_DIR/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$ROOT_DIR/scripts/train.py" \
    task=libero_leapbot_2cam224 \
    "output_dir=$OUTPUT_DIR" \
    "model.causal_mode=$MODE" \
    model.history_training_mode=incremental_detached_prefix \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    "model.video_lora.learning_rate_multiplier=$VIDEO_LORA_MULTIPLIER" \
    data.train.full_episode_history=true \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=70 \
    'model.training_exit_depths=[30]' \
    max_steps=null \
    "num_epochs=$NUM_EPOCHS" \
    "learning_rate=$LEARNING_RATE" \
    lr_scheduler_type=cosine \
    "gradient_accumulation_steps=$GRAD_ACCUM" \
    "batch_size=$BATCH_SIZE" \
    num_workers=3 \
    log_every=1 \
    "save_every=$STEPS_PER_EPOCH" \
    eval_every=0 \
    seed=42 \
    wandb.enabled=true \
    "wandb.workspace=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "wandb.group=$WANDB_GROUP" \
    "wandb.name=$RUN_NAME" \
    wandb.mode=online \
    "resume=$RESUME_PATH" \
    >>"$LOG_FILE" 2>&1

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$FINAL_CHECKPOINT" \
    --expected-step "$FINAL_STEP" \
    --expected-mode "$MODE" \
    --state-dir "$OUTPUT_DIR/checkpoints/state/$FINAL_TAG"
log "full-prefix five-epoch training complete mode=$MODE checkpoint=$FINAL_CHECKPOINT"
