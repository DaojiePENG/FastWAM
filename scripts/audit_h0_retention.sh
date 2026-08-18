#!/usr/bin/env bash

set -euo pipefail

# Audit native H0 retention and long-history behavior for the two episode-
# start sampling factors using the exact fixed-noise protocol used for LR choice.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
PYTHON_BIN="${LEAPBOT_PYTHON:-$(command -v python 2>/dev/null || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Python is unavailable; activate Conda/uv or set LEAPBOT_PYTHON.\n' >&2
    exit 2
fi
LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?LR_SELECTION_MANIFEST is required}"
SELECTED_LR="$("$PYTHON_BIN" \
    "$ROOT_DIR/scripts/history_audit_selection.py" validate \
    --manifest "$LR_SELECTION_MANIFEST" \
    --expected-kind learning_rate \
    --allowed-basis fixed_noise_audit \
    --allowed-basis user_directed \
    --selected-value-only)"
FINAL_STEP="${FINAL_STEP:-100}"
ALLOW_DIRTY="${ALLOW_DIRTY:-true}"
GPU_ID="${GPU_ID:-0}"
LR_TAG="${SELECTED_LR//./p}"
LR_TAG="${LR_TAG//+/_}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/h0_retention_incremental_v6_mb10_ga2_s${FINAL_STEP}_bs80_lr${LR_TAG}_chunk1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/h0_retention_audit_incremental_v6_mb10_ga2_s${FINAL_STEP}_lr${LR_TAG}}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
FINAL_TAG="$(printf 'step_%06d' "$FINAL_STEP")"
X1_CHECKPOINT="$SCREEN_ROOT/h0x1/checkpoints/weights/$FINAL_TAG.pt"
X4_CHECKPOINT="$SCREEN_ROOT/h0x4/checkpoints/weights/$FINAL_TAG.pt"

case "$SELECTED_LR" in
    1.0e-5|1.0e-4) ;;
    *)
        printf 'SELECTED_LR must be one of the paired screen candidates; got %s\n' \
            "$SELECTED_LR" >&2
        exit 2
        ;;
esac
if [[ "$ALLOW_DIRTY" != "true" ]] \
    && [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
    printf 'Refusing paired H0 audit from a dirty worktree.\n' >&2
    exit 2
fi
for required in \
    "$RELEASE_CHECKPOINT" \
    "$DATASET_STATS" \
    "$X1_CHECKPOINT" \
    "$X4_CHECKPOINT"; do
    if [[ ! -s "$required" ]]; then
        printf 'Paired H0 audit input is missing: %s\n' "$required" >&2
        exit 2
    fi
done

for factor in 1 4; do
    candidate_root="$SCREEN_ROOT/h0x${factor}"
    contract_file="$candidate_root/run_contract.txt"
    expected_contract="$(awk -F= '$1 == "run_contract_sha256" {print $2}' "$contract_file")"
    expected_commit="$(awk -F= '$1 == "code_commit" {print $2}' "$contract_file")"
    if [[ ! "$expected_contract" =~ ^[0-9a-f]{64}$ ]] \
        || [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]] \
        || [[ "$(awk -F= '$1 == "initial_block_oversample" {print $2}' "$contract_file")" != "$factor" ]] \
        || [[ "$(awk -F= '$1 == "learning_rate" {print $2}' "$contract_file")" != "$SELECTED_LR" ]]; then
        printf 'Invalid H0 screen contract for factor %s: %s\n' \
            "$factor" "$contract_file" >&2
        exit 2
    fi
    "$PYTHON_BIN" \
        "$ROOT_DIR/scripts/validate_run_contract_group.py" \
        --contract "action_aggregator=$contract_file" \
        --expected-field "learning_rate=$SELECTED_LR" \
        --expected-field "initial_block_oversample=$factor" \
        --expected-field "max_steps=$FINAL_STEP" \
        --expected-field num_processes=4 \
        --expected-field batch_size=10 \
        --expected-field gradient_accumulation_steps=2 \
        --expected-field global_batch=80 \
        --expected-field padding_attention_mask=true \
        >/dev/null
    candidate="$candidate_root/checkpoints/weights/$FINAL_TAG.pt"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
        "$candidate" \
        --expected-step "$FINAL_STEP" \
        --expected-mode action_aggregator \
        --expected-trained-exit-depths 30 \
        --expected-video-lora-multiplier 1.0 \
        --expected-history-vae-batch-chunk-size 1 \
        --expected-run-contract-sha256 "$expected_contract" \
        --expected-code-commit "$expected_commit" \
        --state-dir "$candidate_root/checkpoints/state/$FINAL_TAG" \
        --output "$candidate_root/checkpoint_validation.json" \
        >/dev/null
done

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
LEAPBOT_DATASET_STATS="$DATASET_STATS" \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" "$ROOT_DIR/scripts/history_stratified_loss.py" \
    task=libero_leapbot_2cam224 \
    "output_dir=$OUTPUT_DIR" \
    mixed_precision=bf16 \
    model.causal_mode=action_aggregator \
    model.history_training_mode=incremental_full_bptt \
    model.history_vae_batch_chunk_size=1 \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    model.video_lora.learning_rate_multiplier=1.0 \
    "+stratified.device=cuda:0" \
    "+stratified.checkpoints=[$RELEASE_CHECKPOINT,$X1_CHECKPOINT,$X4_CHECKPOINT]" \
    '+stratified.history_lengths=[0,1,4,8,16,32,50]' \
    +stratified.samples_per_history=2 \
    +stratified.selection_seed=42 \
    '+stratified.fixed_u_values=[0.1,0.3,0.5,0.7,0.9]' \
    +stratified.noise_repeats=2 \
    +stratified.noise_seed=2000042 \
    '+stratified.history_variants=[correct,masked,shuffled]' \
    +stratified.shuffle_seed=3000042 \
    +stratified.include_native=true \
    +stratified.executed_action_steps=10 \
    +stratified.continuous_action_dims=6 \
    +stratified.gripper_action_index=6 \
    +stratified.bootstrap_iterations=2000 \
    +stratified.bootstrap_seed=4000042 \
    +stratified.output_name=paired_h0_fixed_noise.json

"$PYTHON_BIN" "$ROOT_DIR/scripts/history_audit_selection.py" \
    create \
    --audit "$OUTPUT_DIR/paired_h0_fixed_noise.json" \
    --kind initial_block_oversample \
    --output "$OUTPUT_DIR/initial_block_oversample_selection.json" \
    >/dev/null

printf 'Paired H0 fixed-noise audit complete: %s\n' \
    "$OUTPUT_DIR/paired_h0_fixed_noise.json"
printf 'Validated H0-retention selection manifest: %s\n' \
    "$OUTPUT_DIR/initial_block_oversample_selection.json"
