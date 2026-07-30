#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
STAGE1_EVAL_ROOT="${STAGE1_EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_e1_bs32_dev10}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/action_aggregator_h8_e5_bs72_lr2e5}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DATASET_SIZE="${DATASET_SIZE:-28523}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
VIDEO_LORA_MULTIPLIER="${VIDEO_LORA_MULTIPLIER:-10.0}"
WANDB_ENTITY="${WANDB_ENTITY:-pengdaojie-the-hong-kong-university-of-science-and-techn}"
WANDB_PROJECT="${WANDB_PROJECT:-leapbot-va}"
WANDB_GROUP="${WANDB_GROUP:-action-aggregator-e5-h800-bs72-lr2e5-seed42}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29901}"

GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$(((DATASET_SIZE + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
FINAL_STEP=$((STEPS_PER_EPOCH * NUM_EPOCHS))
MODE=action_aggregator
OUTPUT_DIR="$TRAIN_ROOT/$MODE"
LOG_FILE="$OUTPUT_DIR/train.log"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

stage1_complete() {
    [[ -s "$STAGE1_EVAL_ROOT/pareto/pareto.json" ]] || return 1
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
        "$STAGE1_EVAL_ROOT" \
        --output-dir "$STAGE1_EVAL_ROOT/pareto" \
        --expected-tasks 10 \
        --expected-trials-per-task 10 \
        --require-profiled \
        >"$STAGE1_EVAL_ROOT/pareto/action_e5_gate_validation.log" 2>&1
}

latest_state() {
    { find "$OUTPUT_DIR/checkpoints/state" -mindepth 1 -maxdepth 1 \
        -type d -name 'step_*' 2>/dev/null || true; } | sort | tail -1
}

mkdir -p \
    "$OUTPUT_DIR" \
    "$ROOT_DIR/.cache/wandb/config" \
    "$ROOT_DIR/.cache/wandb/cache" \
    "$ROOT_DIR/.cache/wandb/data"

while ! stage1_complete; do
    log "waiting for strict stage-1 10x10 comparison: $STAGE1_EVAL_ROOT/pareto/pareto.json"
    sleep "$POLL_SECONDS"
done

FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoints/weights/step_$(printf '%06d' "$FINAL_STEP").pt"
if [[ -s "$FINAL_CHECKPOINT" ]] \
    && grep -q "max_steps reached step=$FINAL_STEP" "$LOG_FILE" 2>/dev/null; then
    log "skip completed action_aggregator exploration: $FINAL_CHECKPOINT"
    exit 0
fi

RESUME_PATH="$(latest_state)"
if [[ -z "$RESUME_PATH" ]]; then
    # Restart from the known-good release instead of inheriting the conservative
    # one-epoch run whose intermediate rollouts were 0/60.
    RESUME_PATH="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
fi

log "start action_aggregator exploration epochs=$NUM_EPOCHS gpus=$GPU_IDS_CSV micro_batch=$BATCH_SIZE global_batch=$GLOBAL_BATCH steps_per_epoch=$STEPS_PER_EPOCH final_step=$FINAL_STEP action_lr=$LEARNING_RATE video_lora_lr_multiplier=$VIDEO_LORA_MULTIPLIER resume=$RESUME_PATH"
CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    WANDB_CONFIG_DIR="$ROOT_DIR/.cache/wandb/config" \
    WANDB_CACHE_DIR="$ROOT_DIR/.cache/wandb/cache" \
    WANDB_DATA_DIR="$ROOT_DIR/.cache/wandb/data" \
    WANDB_DIR="$OUTPUT_DIR" \
    WANDB_RUN_ID=action-aggregator-e5-h800-bs72-lr2e5-seed42 \
    WANDB_RESUME=allow \
    "$ROOT_DIR/.venv/bin/accelerate" launch \
    --config_file "$ROOT_DIR/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$ROOT_DIR/scripts/train.py" \
    task=libero_leapbot_2cam224 \
    "output_dir=$OUTPUT_DIR" \
    model.causal_mode=action_aggregator \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    "model.video_lora.learning_rate_multiplier=$VIDEO_LORA_MULTIPLIER" \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=8 \
    'model.training_exit_depths=[30]' \
    max_steps=null \
    "num_epochs=$NUM_EPOCHS" \
    "learning_rate=$LEARNING_RATE" \
    lr_scheduler_type=cosine \
    "gradient_accumulation_steps=$GRAD_ACCUM" \
    "batch_size=$BATCH_SIZE" \
    num_workers=4 \
    log_every=10 \
    "save_every=$STEPS_PER_EPOCH" \
    eval_every=0 \
    seed=42 \
    wandb.enabled=true \
    "wandb.workspace=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "wandb.group=$WANDB_GROUP" \
    wandb.name=action-aggregator-e5-h800-bs72-lr2e5-seed42 \
    wandb.mode=online \
    "resume=$RESUME_PATH" \
    >"$LOG_FILE" 2>&1
log "action_aggregator five-epoch exploration complete: $FINAL_CHECKPOINT"
