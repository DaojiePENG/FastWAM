#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
SCREEN_EVAL_ROOT="${SCREEN_EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_s1000_dev10}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/runs/phase1_h8_d30_e1_bs32}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluate_results/phase1_h8_d30_e1_bs32_dev10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FINAL_STEP="${FINAL_STEP:-892}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_EVERY="${SAVE_EVERY:-223}"
WANDB_ENTITY="${WANDB_ENTITY:-pengdaojie-the-hong-kong-university-of-science-and-techn}"
WANDB_PROJECT="${WANDB_PROJECT:-leapbot-va}"
WANDB_GROUP="${WANDB_GROUP:-phase1-h8-d30-e1-bs32-seed42}"
FINAL_EVAL_GPU_IDS_CSV="${FINAL_EVAL_GPU_IDS_CSV:-}"

MODES=(interleaved vision_causal action_aggregator)
GPU_PAIRS=(0,1 2,3 4,5)
PORTS=(29701 29702 29703)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

screening_complete() {
    [[ -s "$SCREEN_EVAL_ROOT/pareto/pareto.json" ]] || return 1
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
        "$SCREEN_EVAL_ROOT" \
        --output-dir "$SCREEN_EVAL_ROOT/pareto" \
        --expected-tasks 10 \
        --expected-trials-per-task 10 \
        --require-profiled \
        >"$SCREEN_EVAL_ROOT/pareto/validation.log" 2>&1
}

final_checkpoint() {
    local mode="$1"
    printf '%s/%s/checkpoints/weights/step_%06d.pt\n' "$TRAIN_ROOT" "$mode" "$FINAL_STEP"
}

latest_state() {
    local mode="$1"
    { find "$TRAIN_ROOT/$mode/checkpoints/state" -mindepth 1 -maxdepth 1 \
        -type d -name 'step_*' 2>/dev/null || true; } | sort | tail -1
}

run_mode() {
    local index="$1"
    local mode="${MODES[$index]}"
    local gpu_pair="${GPU_PAIRS[$index]}"
    local port="${PORTS[$index]}"
    local output_dir="$TRAIN_ROOT/$mode"
    local log_file="$output_dir/train.log"
    local checkpoint resume_path

    checkpoint="$(final_checkpoint "$mode")"
    if [[ -s "$checkpoint" ]] \
        && grep -q "max_steps reached step=$FINAL_STEP" "$log_file" 2>/dev/null; then
        log "skip completed full training mode=$mode checkpoint=$checkpoint"
        return 0
    fi

    mkdir -p "$output_dir"
    resume_path="$(latest_state "$mode")"
    if [[ -z "$resume_path" ]]; then
        resume_path="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
    fi

    log "start full training mode=$mode gpus=$gpu_pair micro_batch=$BATCH_SIZE effective_batch=$((2 * BATCH_SIZE * GRAD_ACCUM)) resume=$resume_path"
    CUDA_VISIBLE_DEVICES="$gpu_pair" \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        WANDB_CONFIG_DIR="$ROOT_DIR/.cache/wandb/config" \
        WANDB_CACHE_DIR="$ROOT_DIR/.cache/wandb/cache" \
        WANDB_DATA_DIR="$ROOT_DIR/.cache/wandb/data" \
        WANDB_DIR="$output_dir" \
        WANDB_RUN_ID="phase1-e1-bs32-${mode}-seed42" \
        WANDB_RESUME=allow \
        "$ROOT_DIR/.venv/bin/accelerate" launch \
        --config_file "$ROOT_DIR/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
        --num_processes 2 \
        --main_process_port "$port" \
        "$ROOT_DIR/scripts/train.py" \
        task=libero_leapbot_2cam224 \
        "output_dir=$output_dir" \
        "model.causal_mode=$mode" \
        model.training_strategy=video_lora_action_full \
        model.video_lora.enabled=true \
        model.video_lora.rank=16 \
        model.video_lora.alpha=16.0 \
        model.video_lora.dropout=0.0 \
        model.video_lora.learning_rate_multiplier=10.0 \
        data.train.min_history_blocks=0 \
        data.train.max_history_blocks=8 \
        'model.training_exit_depths=[30]' \
        max_steps=null \
        num_epochs=1 \
        "gradient_accumulation_steps=$GRAD_ACCUM" \
        "batch_size=$BATCH_SIZE" \
        num_workers=4 \
        log_every=10 \
        "save_every=$SAVE_EVERY" \
        eval_every=0 \
        seed=42 \
        wandb.enabled=true \
        "wandb.workspace=$WANDB_ENTITY" \
        "wandb.project=$WANDB_PROJECT" \
        "wandb.group=$WANDB_GROUP" \
        "wandb.name=phase1-e1-bs32-${mode}-seed42" \
        wandb.mode=online \
        "resume=$resume_path" \
        >"$log_file" 2>&1
    log "done full training mode=$mode"
}

mkdir -p \
    "$TRAIN_ROOT" \
    "$EVAL_ROOT" \
    "$ROOT_DIR/.cache/wandb/config" \
    "$ROOT_DIR/.cache/wandb/cache" \
    "$ROOT_DIR/.cache/wandb/data"
while ! screening_complete; do
    log "waiting for screening comparison: $SCREEN_EVAL_ROOT/pareto/pareto.json"
    sleep "$POLL_SECONDS"
done

log "screening comparison complete; launching one-full-epoch causal comparison"
pids=()
for index in "${!MODES[@]}"; do
    run_mode "$index" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    log "one or more full-training workers failed; inspect $TRAIN_ROOT/*/train.log"
    exit 1
fi

if [[ -z "$FINAL_EVAL_GPU_IDS_CSV" ]]; then
    if grep -q "all intermediate checkpoint evaluations complete" \
        "$ROOT_DIR/runs/phase1_intermediate_eval_supervisor.log" 2>/dev/null; then
        FINAL_EVAL_GPU_IDS_CSV="0,1,2,3,4,5,6,7"
    else
        # GPUs 6-7 run the checkpoint learning-curve evaluation concurrently.
        FINAL_EVAL_GPU_IDS_CSV="0,1,2,3,4,5"
    fi
fi

log "all full-training workers complete; starting formal 10x10 evaluation"
log "formal evaluation GPUs: $FINAL_EVAL_GPU_IDS_CSV"
# The screening gate has already strictly validated this identical profiled
# release baseline (same tasks, trials, initial states, and evaluator). Reuse it
# so the formal stage spends its rollout budget on the three trained variants.
mkdir -p "$EVAL_ROOT/fastwam_release"
cp -a "$SCREEN_EVAL_ROOT/fastwam_release/." "$EVAL_ROOT/fastwam_release/"
TRAIN_ROOT="$TRAIN_ROOT" \
EVAL_ROOT="$EVAL_ROOT" \
FINAL_STEP="$FINAL_STEP" \
NUM_TRIALS=10 \
GPU_IDS_CSV="$FINAL_EVAL_GPU_IDS_CSV" \
VIDEO_LORA_ENABLED=true \
MERGE_VIDEO_LORA=true \
bash "$ROOT_DIR/scripts/run_phase1_eval_after_training.sh"
log "full one-epoch comparison complete: $EVAL_ROOT/pareto/pareto.json"
