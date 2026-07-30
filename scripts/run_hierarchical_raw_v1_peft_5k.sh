#!/usr/bin/env bash

set -euo pipefail

# Correct-architecture phase-A v2 recipe:
#   * FastWAM release initialization
#   * block-local RoPE + zero-initialized absolute episode timing
#   * one raw causal-attention softmax (no history gate)
#   * complete packed episode prefix with full BPTT
#   * ActionDiT full fine-tuning + conservative VideoDiT LoRA

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
MODE="${MODE:-action_aggregator}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-5}"
MAX_STEPS="${MAX_STEPS:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/final_hierarchical_raw_v2_${MODE}_peft_${MAX_STEPS}steps_bs80_lr1e5}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
VIDEO_LORA_MULTIPLIER="${VIDEO_LORA_MULTIPLIER:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29971}"
MAX_PREFLIGHT_USED_MIB="${MAX_PREFLIGHT_USED_MIB:-2048}"
WANDB_ENTITY="${WANDB_ENTITY:-pengdaojie-the-hong-kong-university-of-science-and-techn}"
WANDB_PROJECT="${WANDB_PROJECT:-leapbot-va}"
WANDB_GROUP="${WANDB_GROUP:-final-hierarchical-raw-v2-seed42}"
RUN_NAME="${RUN_NAME:-final-hierarchical-raw-v2-${MODE//_/-}-peft-${MAX_STEPS}steps-bs80-lr1e5-seed42}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"

GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
LOG_FILE="$OUTPUT_DIR/train.log"
FINAL_TAG="step_$(printf '%06d' "$MAX_STEPS")"
FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoints/weights/$FINAL_TAG.pt"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

latest_state() {
    { find "$OUTPUT_DIR/checkpoints/state" -mindepth 1 -maxdepth 1 \
        -type d -name 'step_*' 2>/dev/null || true; } | sort | tail -1
}

preflight_gpus() {
    local expected_gpu_count used
    expected_gpu_count="$(awk -F, '{print NF}' <<<"$GPU_IDS_CSV")"
    if [[ "$expected_gpu_count" -ne "$NUM_PROCESSES" ]]; then
        log "GPU list/process mismatch: ids=$GPU_IDS_CSV processes=$NUM_PROCESSES"
        return 1
    fi
    while IFS= read -r used; do
        if (( used > MAX_PREFLIGHT_USED_MIB )); then
            log "GPU preflight failed: selected GPU uses ${used} MiB (limit ${MAX_PREFLIGHT_USED_MIB} MiB)"
            return 1
        fi
    done < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk -v ids="$GPU_IDS_CSV" \
                'BEGIN {split(ids, a, ","); for (i in a) wanted[a[i]]=1} wanted[NR-1] {print $1}'
    )
}

if [[ ! -s "$RELEASE_CHECKPOINT" ]]; then
    log "missing FastWAM release checkpoint: $RELEASE_CHECKPOINT"
    exit 1
fi
case "$MODE" in
    interleaved|vision_causal|action_aggregator) ;;
    *)
        log "invalid causal mode: $MODE"
        exit 1
        ;;
esac
if [[ ! -s "$DATASET_STATS" ]]; then
    log "missing release normalization statistics: $DATASET_STATS"
    exit 1
fi

mkdir -p \
    "$OUTPUT_DIR" \
    "$ROOT_DIR/.cache/wandb/config" \
    "$ROOT_DIR/.cache/wandb/cache" \
    "$ROOT_DIR/.cache/wandb/data"

if [[ -s "$FINAL_CHECKPOINT" ]] \
    && grep -q "max_steps reached step=$MAX_STEPS" "$LOG_FILE" 2>/dev/null; then
    log "skip completed run: $FINAL_CHECKPOINT"
    exit 0
fi

RESUME_PATH="$(latest_state)"
if [[ -z "$RESUME_PATH" ]]; then
    RESUME_PATH="$RELEASE_CHECKPOINT"
    : >"$LOG_FILE"
else
    log "resume from full trainer state: $RESUME_PATH" >>"$LOG_FILE"
fi

preflight_gpus
CODE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
log "start hierarchical raw-v2 PEFT: commit=$CODE_COMMIT mode=$MODE gpus=$GPU_IDS_CSV micro_batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM global_batch=$GLOBAL_BATCH max_steps=$MAX_STEPS action_lr=$LEARNING_RATE video_lora_multiplier=$VIDEO_LORA_MULTIPLIER resume=$RESUME_PATH"

CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" \
    LEAPBOT_DATASET_STATS="$DATASET_STATS" \
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
    model.history_training_mode=packed_full_bptt \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    "model.video_lora.learning_rate_multiplier=$VIDEO_LORA_MULTIPLIER" \
    model.mot_checkpoint_mixed_attn=true \
    data.train.full_episode_history=true \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=70 \
    'model.training_exit_depths=[30]' \
    "max_steps=$MAX_STEPS" \
    num_epochs=100 \
    "learning_rate=$LEARNING_RATE" \
    lr_scheduler_type=constant \
    "gradient_accumulation_steps=$GRAD_ACCUM" \
    "batch_size=$BATCH_SIZE" \
    num_workers=3 \
    max_grad_norm=1.0 \
    weight_decay=1.0e-2 \
    log_every=1 \
    "save_every=$SAVE_EVERY" \
    eval_every=0 \
    seed=42 \
    mixed_precision=bf16 \
    "wandb.enabled=$WANDB_ENABLED" \
    "wandb.workspace=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "wandb.group=$WANDB_GROUP" \
    "wandb.name=$RUN_NAME" \
    "wandb.mode=$WANDB_MODE" \
    "resume=$RESUME_PATH" \
    >>"$LOG_FILE" 2>&1

log "hierarchical raw-v2 PEFT complete: $FINAL_CHECKPOINT"
