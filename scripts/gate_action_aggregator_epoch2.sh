#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/runs/action_aggregator_full_prefix_temporal_rope_v2_h70_e5_bs128_lr2e5/action_aggregator}"
GATE_STEP="${GATE_STEP:-446}"
POLL_SECONDS="${POLL_SECONDS:-10}"
STEP_TAG="step_$(printf '%06d' "$GATE_STEP")"
CHECKPOINT="$TRAIN_DIR/checkpoints/weights/$STEP_TAG.pt"
STATE_DIR="$TRAIN_DIR/checkpoints/state/$STEP_TAG"
TRAIN_LOG="$TRAIN_DIR/train.log"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/evaluate_results/training_diagnostics/action_aggregator_epoch2_gate}"
RELEASE_CHECKPOINT="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STEP223_CHECKPOINT="$TRAIN_DIR/checkpoints/weights/step_000223.pt"
AUDIT_DIR="$OUTPUT_DIR/history_stratified"
LOG_FILE="$OUTPUT_DIR/gate.log"

mkdir -p "$OUTPUT_DIR" "$AUDIT_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

training_pids() {
    ps -eo pid=,args= | awk -v marker="output_dir=$TRAIN_DIR" \
        'index($0, marker) && $0 ~ /scripts\/train.py/ {print $1}'
}

accelerate_pids() {
    ps -eo pid=,args= | awk -v marker="output_dir=$TRAIN_DIR" \
        'index($0, marker) && $0 ~ /accelerate launch/ {print $1}'
}

selected_gpus_are_free() {
    local used
    while IFS= read -r used; do
        (( used <= 2048 )) || return 1
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
}

stop_after_safe_checkpoint() {
    local pids
    pids="$(accelerate_pids | tr '\n' ' ')"
    if [[ -n "$pids" ]]; then
        log "stopping accelerate after completed $STEP_TAG: $pids"
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
    fi
    for _ in $(seq 1 12); do
        [[ -z "$(training_pids)" ]] && return 0
        sleep 5
    done
    pids="$(training_pids | tr '\n' ' ')"
    if [[ -n "$pids" ]]; then
        log "terminating remaining train workers: $pids"
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
    fi
    for _ in $(seq 1 12); do
        [[ -z "$(training_pids)" ]] && return 0
        sleep 5
    done
    pids="$(training_pids | tr '\n' ' ')"
    if [[ -n "$pids" ]]; then
        log "force-stopping unresponsive train workers after grace period: $pids"
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
    fi
}

run_full_bptt_smoke() {
    local history="$1"
    local smoke_dir="$OUTPUT_DIR/full_bptt_h${history}"
    mkdir -p "$smoke_dir"
    log "starting packed full-BPTT 6B smoke H=$history on GPU0"
    set +e
    CUDA_VISIBLE_DEVICES=0 \
        TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/full_prefix_smoke.py" \
        task=libero_leapbot_2cam224 \
        "output_dir=$smoke_dir" \
        model.causal_mode=action_aggregator \
        model.history_training_mode=packed_full_bptt \
        model.training_strategy=video_lora_action_full \
        model.video_lora.enabled=true \
        model.video_lora.rank=16 \
        model.video_lora.alpha=16.0 \
        model.video_lora.dropout=0.0 \
        model.video_lora.learning_rate_multiplier=10.0 \
        data.train.full_episode_history=true \
        data.train.min_history_blocks=0 \
        data.train.max_history_blocks=70 \
        'model.training_exit_depths=[30]' \
        "+smoke.checkpoint=$RELEASE_CHECKPOINT" \
        +smoke.device=cuda:0 \
        "+smoke.history_blocks=$history" \
        +smoke.compare_native=false \
        >"$smoke_dir/smoke.log" 2>&1
    local status=$?
    set -e
    printf '%s\t%s\n' "$history" "$status" >>"$OUTPUT_DIR/full_bptt_status.tsv"
    if [[ "$status" -eq 0 ]]; then
        log "packed full-BPTT smoke passed H=$history"
    else
        log "packed full-BPTT smoke failed H=$history status=$status (preserved for diagnosis)"
    fi
}

log "waiting for complete checkpoint $STEP_TAG"
while [[ ! -s "$CHECKPOINT" ]] \
    || [[ ! -s "$STATE_DIR/trainer_state.json" ]] \
    || ! grep -q "step=$GATE_STEP" "$TRAIN_LOG" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

# Rich wraps long checkpoint paths in the log, so completeness is established
# from stable on-disk sizes rather than a fragile single-line path match.
previous_size=-1
stable_observations=0
while (( stable_observations < 2 )); do
    current_size=$((
        $(stat -c '%s' "$CHECKPOINT")
        + $(du -sb "$STATE_DIR" | awk '{print $1}')
    ))
    if [[ "$current_size" -eq "$previous_size" ]]; then
        stable_observations=$((stable_observations + 1))
    else
        stable_observations=0
        previous_size="$current_size"
    fi
    sleep 5
done

# The trainer logs the checkpoint only after all distributed state shards are
# closed.  Stop before spending another epoch on an unvalidated recipe.
stop_after_safe_checkpoint

while ! selected_gpus_are_free; do
    log "waiting for training GPU memory to be released"
    sleep 5
done

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$CHECKPOINT" \
    --expected-step "$GATE_STEP" \
    --expected-mode action_aggregator \
    --state-dir "$STATE_DIR" \
    >>"$LOG_FILE" 2>&1
log "checkpoint validation passed for $STEP_TAG"

CUDA_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/history_stratified_loss.py" \
    task=libero_leapbot_2cam224 \
    "output_dir=$AUDIT_DIR" \
    model.causal_mode=action_aggregator \
    model.history_training_mode=incremental_detached_prefix \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    data.train.full_episode_history=true \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=70 \
    'model.training_exit_depths=[30]' \
    "+stratified.checkpoints=[$RELEASE_CHECKPOINT,$STEP223_CHECKPOINT,$CHECKPOINT]" \
    '+stratified.history_lengths=[0,1,4,8,12,16,24,32,40,50]' \
    +stratified.samples_per_history=2 \
    +stratified.selection_seed=42 \
    +stratified.noise_seed=2000042 \
    +stratified.include_native=true \
    +stratified.device=cuda:0 \
    >>"$LOG_FILE" 2>&1
log "fixed-noise history-stratified audit complete"

: >"$OUTPUT_DIR/full_bptt_status.tsv"
for history in 0 8 16 32 50; do
    run_full_bptt_smoke "$history"
done
log "epoch-2 gate diagnostics complete"
