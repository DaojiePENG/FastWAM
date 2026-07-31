#!/usr/bin/env bash

set -euo pipefail

# Canonical single-mode LeapBot training recipe:
#   * FastWAM release initialization
#   * block-local RoPE + first-block-anchored episode timing
#   * one raw causal-attention softmax (no history gate)
#   * runtime-isomorphic chronological prefix with full BPTT
#   * ActionDiT full fine-tuning + conservative VideoDiT LoRA

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RELEASE_CHECKPOINT}"
DATASET_STATS="${DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
ASSET_DOWNLOAD_MANIFEST="${ASSET_DOWNLOAD_MANIFEST:-$ROOT_DIR/data/leapbot_asset_download_manifest.json}"
TEXT_EMBEDDING_CACHE="${TEXT_EMBEDDING_CACHE:-$ROOT_DIR/data/text_embeds_cache/libero}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-$ROOT_DIR/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
MODE="${MODE:-action_aggregator}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
VIDEO_LORA_MULTIPLIER="${VIDEO_LORA_MULTIPLIER:-1.0}"
HISTORY_VAE_BATCH_CHUNK_SIZE="${HISTORY_VAE_BATCH_CHUNK_SIZE:-1}"
INITIAL_BLOCK_OVERSAMPLE="${INITIAL_BLOCK_OVERSAMPLE:-4}"
TRAINING_EXIT_DEPTHS_CSV="${TRAINING_EXIT_DEPTHS_CSV:-30}"
SAVE_EVERY="${SAVE_EVERY:-500}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29971}"
MAX_PREFLIGHT_USED_MIB="${MAX_PREFLIGHT_USED_MIB:-2048}"
WANDB_ENTITY="${WANDB_ENTITY:-pengdaojie-the-hong-kong-university-of-science-and-techn}"
WANDB_PROJECT="${WANDB_PROJECT:-leapbot-va}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-42}"
ALLOW_DIRTY="${ALLOW_DIRTY:-false}"
ALLOW_CROSS_CONTRACT_RESUME="${ALLOW_CROSS_CONTRACT_RESUME:-false}"
ALLOW_EXISTING_UNCONTRACTED="${ALLOW_EXISTING_UNCONTRACTED:-false}"
REQUIRE_SELF_IDENTIFYING_CHECKPOINT="${REQUIRE_SELF_IDENTIFYING_CHECKPOINT:-false}"
LR_SELECTION_MANIFEST_SHA256="${LR_SELECTION_MANIFEST_SHA256:-}"
H0_SELECTION_MANIFEST_SHA256="${H0_SELECTION_MANIFEST_SHA256:-}"
EXPECTED_TRAINING_ASSET_MANIFEST_SHA256="${EXPECTED_TRAINING_ASSET_MANIFEST_SHA256:-}"

GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))
TOPOLOGY_TAG="w${NUM_PROCESSES}_b${BATCH_SIZE}_ga${GRAD_ACCUM}_bs${GLOBAL_BATCH}"
LR_TAG="${LEARNING_RATE//./p}"
LR_TAG="${LR_TAG//+/_}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/final_incremental_full_bptt_v5_${TOPOLOGY_TAG}_${MODE}_peft_${MAX_STEPS}steps_${LR_SCHEDULER_TYPE}_lr${LR_TAG}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-final-incremental-full-bptt-v5-${TOPOLOGY_TAG}-seed${SEED}}"
RUN_NAME="${RUN_NAME:-final-incremental-full-bptt-v5-${TOPOLOGY_TAG//_/-}-${MODE//_/-}-peft-${MAX_STEPS}steps-${LR_SCHEDULER_TYPE}-lr${LR_TAG}-seed${SEED}}"
LOG_FILE="$OUTPUT_DIR/train.log"
FINAL_TAG="step_$(printf '%06d' "$MAX_STEPS")"
FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoints/weights/$FINAL_TAG.pt"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

latest_state() {
    local state
    while IFS= read -r state; do
        # trainer_state.json is written only after every DeepSpeed shard and RNG
        # state has completed, so never resume a half-written checkpoint.
        if [[ -s "$state/trainer_state.json" ]]; then
            printf '%s\n' "$state"
        fi
    done < <(
        { find "$OUTPUT_DIR/checkpoints/state" -mindepth 1 -maxdepth 1 \
            -type d -name 'step_*' 2>/dev/null || true; } | sort
    ) | tail -1
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
if [[ ! -s "$INITIAL_CHECKPOINT" ]]; then
    log "missing initialization checkpoint: $INITIAL_CHECKPOINT"
    exit 1
fi
case "$TRAINING_EXIT_DEPTHS_CSV" in
    30|8,16,24,30) ;;
    *)
        log "training exits must be 30 or 8,16,24,30; got $TRAINING_EXIT_DEPTHS_CSV"
        exit 1
        ;;
esac
case "$MODE" in
    interleaved|vision_causal|action_aggregator) ;;
    *)
        log "invalid causal mode: $MODE"
        exit 1
        ;;
esac
case "$LR_SCHEDULER_TYPE" in
    cosine|constant) ;;
    *)
        log "invalid LR scheduler: $LR_SCHEDULER_TYPE"
        exit 1
        ;;
esac
if [[ ! -s "$DATASET_STATS" ]]; then
    log "missing release normalization statistics: $DATASET_STATS"
    exit 1
fi
if [[ ! -s "$ASSET_DOWNLOAD_MANIFEST" || ! -d "$TEXT_EMBEDDING_CACHE" \
    || ! -s "$VAE_CHECKPOINT" ]]; then
    log "formal training assets or pinned download manifest are missing"
    exit 1
fi
if [[ "$HISTORY_VAE_BATCH_CHUNK_SIZE" != "1" ]]; then
    log "runtime-isomorphic training requires history VAE chunk 1; got $HISTORY_VAE_BATCH_CHUNK_SIZE"
    exit 1
fi
if ! [[ "$INITIAL_BLOCK_OVERSAMPLE" =~ ^[1-9][0-9]*$ ]]; then
    log "initial block oversampling must be a positive integer; got $INITIAL_BLOCK_OVERSAMPLE"
    exit 1
fi
for selection_sha in \
    "$LR_SELECTION_MANIFEST_SHA256" \
    "$H0_SELECTION_MANIFEST_SHA256" \
    "$EXPECTED_TRAINING_ASSET_MANIFEST_SHA256"; do
    if [[ -n "$selection_sha" && ! "$selection_sha" =~ ^[0-9a-f]{64}$ ]]; then
        log "selection manifest identity must be a SHA-256: $selection_sha"
        exit 1
    fi
done
if [[ "$ALLOW_DIRTY" != "true" ]] \
    && [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
    log "refusing formal training from a dirty worktree; commit the exact code first"
    exit 1
fi

# Hold a shared lock for the complete training process. Text-cache generation
# takes the matching exclusive lock, so cached tensors cannot change after the
# manifest is verified or midway through an epoch.
TEXT_CACHE_LOCK_FILE="$TEXT_EMBEDDING_CACHE/.leapbot_text_cache.lock"
exec {TEXT_CACHE_LOCK_FD}>>"$TEXT_CACHE_LOCK_FILE"
flock -s "$TEXT_CACHE_LOCK_FD"

ASSET_MANIFEST_TMP="$(mktemp "${TMPDIR:-/tmp}/leapbot-training-assets.XXXXXX.json")"
cleanup_asset_manifest() {
    rm -f "$ASSET_MANIFEST_TMP"
}
trap cleanup_asset_manifest EXIT
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/training_asset_manifest.py" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot" \
    --text-embedding-cache "$TEXT_EMBEDDING_CACHE" \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --download-manifest "$ASSET_DOWNLOAD_MANIFEST" \
    --output "$ASSET_MANIFEST_TMP"
manifest_value() {
    "$ROOT_DIR/.venv/bin/python" - "$ASSET_MANIFEST_TMP" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}
TRAINING_ASSET_MANIFEST_SHA256="$(manifest_value manifest_sha256)"
ASSET_DOWNLOAD_MANIFEST_SHA256="$(manifest_value download_manifest.sha256)"
DATASET_CONTENT_SHA256="$(manifest_value dataset_content_sha256)"
DATASET_FILE_COUNT="$(manifest_value dataset_file_count)"
DATASET_BYTES="$(manifest_value dataset_bytes)"
TEXT_EMBEDDING_CACHE_SHA256="$(manifest_value text_embedding_cache.sha256)"
TEXT_EMBEDDING_CACHE_FILE_COUNT="$(manifest_value text_embedding_cache.file_count)"
TEXT_CACHE_PROVENANCE_SHA256="$(manifest_value text_cache_provenance.provenance_sha256)"
TEXT_CACHE_VERIFICATION_METHOD="$(manifest_value text_cache_provenance.verification.method)"
TEXT_CACHE_VERIFIED_FILE_COUNT="$(manifest_value text_cache_provenance.verification.verified_file_count)"
TEXT_ENCODER_CHECKPOINT_SHA256="$(manifest_value text_cache_provenance.source_assets.text_encoder.sha256)"
TOKENIZER_SHA256="$(manifest_value text_cache_provenance.source_assets.tokenizer.sha256)"
VAE_CHECKPOINT_SHA256="$(manifest_value vae_checkpoint.sha256)"
if [[ -n "$EXPECTED_TRAINING_ASSET_MANIFEST_SHA256" ]] \
    && [[ "$TRAINING_ASSET_MANIFEST_SHA256" != "$EXPECTED_TRAINING_ASSET_MANIFEST_SHA256" ]]; then
    log "training assets differ from the required source checkpoint contract"
    exit 1
fi

CODE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
ACTUAL_RELEASE_CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')"
if [[ -n "${RELEASE_CHECKPOINT_SHA256:-}" ]] \
    && [[ "$RELEASE_CHECKPOINT_SHA256" != "$ACTUAL_RELEASE_CHECKPOINT_SHA256" ]]; then
    log "configured release checkpoint hash does not match the checkpoint bytes"
    exit 1
fi
RELEASE_CHECKPOINT_SHA256="$ACTUAL_RELEASE_CHECKPOINT_SHA256"
INITIAL_CHECKPOINT_SHA256="$(sha256sum "$INITIAL_CHECKPOINT" | awk '{print $1}')"
DATASET_STATS_SHA256="$(sha256sum "$DATASET_STATS" | awk '{print $1}')"
contract_fields=( \
    "code_commit=$CODE_COMMIT" \
    "release_checkpoint_sha256=$RELEASE_CHECKPOINT_SHA256" \
)
if [[ "$INITIAL_CHECKPOINT_SHA256" != "$RELEASE_CHECKPOINT_SHA256" ]]; then
    contract_fields+=("initial_checkpoint_sha256=$INITIAL_CHECKPOINT_SHA256")
fi
contract_fields+=( \
    "dataset_stats_sha256=$DATASET_STATS_SHA256" \
    "training_asset_manifest_sha256=$TRAINING_ASSET_MANIFEST_SHA256" \
    "asset_download_manifest_sha256=$ASSET_DOWNLOAD_MANIFEST_SHA256" \
    "dataset_content_sha256=$DATASET_CONTENT_SHA256" \
    "dataset_file_count=$DATASET_FILE_COUNT" \
    "dataset_bytes=$DATASET_BYTES" \
    "text_embedding_cache_sha256=$TEXT_EMBEDDING_CACHE_SHA256" \
    "text_embedding_cache_file_count=$TEXT_EMBEDDING_CACHE_FILE_COUNT" \
    "text_cache_provenance_sha256=$TEXT_CACHE_PROVENANCE_SHA256" \
    "text_cache_verification_method=$TEXT_CACHE_VERIFICATION_METHOD" \
    "text_cache_verified_file_count=$TEXT_CACHE_VERIFIED_FILE_COUNT" \
    "text_encoder_checkpoint_sha256=$TEXT_ENCODER_CHECKPOINT_SHA256" \
    "tokenizer_sha256=$TOKENIZER_SHA256" \
    "vae_checkpoint_sha256=$VAE_CHECKPOINT_SHA256" \
)
if [[ -n "$LR_SELECTION_MANIFEST_SHA256" ]]; then
    contract_fields+=("lr_selection_manifest_sha256=$LR_SELECTION_MANIFEST_SHA256")
fi
if [[ -n "$H0_SELECTION_MANIFEST_SHA256" ]]; then
    contract_fields+=("h0_selection_manifest_sha256=$H0_SELECTION_MANIFEST_SHA256")
fi
contract_fields+=( \
    "mode=$MODE" \
    "num_processes=$NUM_PROCESSES" \
    "batch_size=$BATCH_SIZE" \
    "gradient_accumulation_steps=$GRAD_ACCUM" \
    "global_batch=$GLOBAL_BATCH" \
    "max_steps=$MAX_STEPS" \
    "learning_rate=$LEARNING_RATE" \
    "lr_scheduler_type=$LR_SCHEDULER_TYPE" \
    "video_lora_multiplier=$VIDEO_LORA_MULTIPLIER" \
    "history_vae_batch_chunk_size=$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    "initial_block_oversample=$INITIAL_BLOCK_OVERSAMPLE" \
    "h0_anchor_mixing=per_global_micro_batch" \
    "save_every=$SAVE_EVERY" \
    "seed=$SEED" \
    "padding_attention_mask=true" \
    "history_training_mode=incremental_full_bptt" \
    "full_episode_history=true" \
    "max_history_blocks=70" \
    "replan_steps=10" \
    "action_horizon=32" \
    "training_exit_depths=$TRAINING_EXIT_DEPTHS_CSV" \
    "mixed_precision=bf16" \
    "optimizer=adamw_beta0.9_0.95_wd0.01_clip1.0" \
)
CONTRACT_PAYLOAD="$(printf '%s\n' "${contract_fields[@]}")"
RUN_CONTRACT_SHA256="$(printf '%s' "$CONTRACT_PAYLOAD" | sha256sum | awk '{print $1}')"
RUN_CONTRACT_FILE="$OUTPUT_DIR/run_contract.txt"

mkdir -p \
    "$OUTPUT_DIR" \
    "$ROOT_DIR/.cache/wandb/config" \
    "$ROOT_DIR/.cache/wandb/cache" \
    "$ROOT_DIR/.cache/wandb/data"

if [[ -s "$RUN_CONTRACT_FILE" ]]; then
    STORED_CONTRACT_SHA256="$(awk -F= '$1 == "run_contract_sha256" {print $2}' "$RUN_CONTRACT_FILE")"
    if [[ "$STORED_CONTRACT_SHA256" != "$RUN_CONTRACT_SHA256" ]] \
        && [[ "$ALLOW_CROSS_CONTRACT_RESUME" != "true" ]]; then
        log "refusing output-dir reuse across run contracts: stored=$STORED_CONTRACT_SHA256 current=$RUN_CONTRACT_SHA256"
        exit 1
    fi
elif [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    && [[ "$ALLOW_EXISTING_UNCONTRACTED" != "true" ]]; then
    log "refusing uncontracted non-empty output directory: $OUTPUT_DIR"
    exit 1
else
    {
        printf 'run_contract_sha256=%s\n' "$RUN_CONTRACT_SHA256"
        printf '%s\n' "$CONTRACT_PAYLOAD"
    } >"$RUN_CONTRACT_FILE.tmp"
    mv "$RUN_CONTRACT_FILE.tmp" "$RUN_CONTRACT_FILE"
fi
cp "$ASSET_MANIFEST_TMP" "$OUTPUT_DIR/training_asset_manifest.json.tmp"
mv "$OUTPUT_DIR/training_asset_manifest.json.tmp" \
    "$OUTPUT_DIR/training_asset_manifest.json"

if [[ -s "$FINAL_CHECKPOINT" ]] \
    && grep -q "max_steps reached step=$MAX_STEPS" "$LOG_FILE" 2>/dev/null; then
    log "skip completed run: $FINAL_CHECKPOINT"
    exit 0
fi

RESUME_PATH="$(latest_state)"
if [[ -z "$RESUME_PATH" ]]; then
    RESUME_PATH="$INITIAL_CHECKPOINT"
    : >"$LOG_FILE"
else
    log "resume from full trainer state: $RESUME_PATH" >>"$LOG_FILE"
fi

preflight_gpus
log "start incremental full-BPTT PEFT: commit=$CODE_COMMIT contract=$RUN_CONTRACT_SHA256 mode=$MODE topology=$TOPOLOGY_TAG gpus=$GPU_IDS_CSV micro_batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM global_batch=$GLOBAL_BATCH max_steps=$MAX_STEPS action_lr=$LEARNING_RATE lr_scheduler=$LR_SCHEDULER_TYPE video_lora_multiplier=$VIDEO_LORA_MULTIPLIER history_vae_batch_chunk=$HISTORY_VAE_BATCH_CHUNK_SIZE initial_block_oversample=$INITIAL_BLOCK_OVERSAMPLE h0_anchor_mixing=per_global_micro_batch resume=$RESUME_PATH"

CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" \
    PYTHONHASHSEED="$SEED" \
    LEAPBOT_RUN_CONTRACT_SHA256="$RUN_CONTRACT_SHA256" \
    LEAPBOT_CODE_COMMIT="$CODE_COMMIT" \
    LEAPBOT_DATASET_STATS="$DATASET_STATS" \
    DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
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
    model.history_training_mode=incremental_full_bptt \
    "model.history_vae_batch_chunk_size=$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    "model.video_lora.learning_rate_multiplier=$VIDEO_LORA_MULTIPLIER" \
    model.mot_checkpoint_mixed_attn=true \
    data.train.full_episode_history=true \
    "data.train.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE" \
    "data.train.dataset_dirs=[$ROOT_DIR/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot]" \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=70 \
    "data.train.initial_block_oversample=$INITIAL_BLOCK_OVERSAMPLE" \
    "model.training_exit_depths=[$TRAINING_EXIT_DEPTHS_CSV]" \
    "max_steps=$MAX_STEPS" \
    num_epochs=100 \
    "learning_rate=$LEARNING_RATE" \
    "lr_scheduler_type=$LR_SCHEDULER_TYPE" \
    "gradient_accumulation_steps=$GRAD_ACCUM" \
    "batch_size=$BATCH_SIZE" \
    num_workers=3 \
    max_grad_norm=1.0 \
    weight_decay=1.0e-2 \
    log_every=1 \
    "save_every=$SAVE_EVERY" \
    eval_every=0 \
    "seed=$SEED" \
    mixed_precision=bf16 \
    "wandb.enabled=$WANDB_ENABLED" \
    "wandb.workspace=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "wandb.group=$WANDB_GROUP" \
    "wandb.name=$RUN_NAME" \
    "wandb.mode=$WANDB_MODE" \
    "resume=$RESUME_PATH" \
    >>"$LOG_FILE" 2>&1

checkpoint_identity_args=()
if [[ "${REQUIRE_SELF_IDENTIFYING_CHECKPOINT:-false}" == "true" ]]; then
    checkpoint_identity_args=(
        --expected-run-contract-sha256 "$RUN_CONTRACT_SHA256"
        --expected-code-commit "$CODE_COMMIT"
    )
fi
expected_training_exit_depths="${TRAINING_EXIT_DEPTHS_CSV:-30}"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$FINAL_CHECKPOINT" \
    --expected-step "$MAX_STEPS" \
    --expected-mode "$MODE" \
    --expected-trained-exit-depths "$expected_training_exit_depths" \
    --expected-video-lora-multiplier "$VIDEO_LORA_MULTIPLIER" \
    --expected-history-vae-batch-chunk-size "$HISTORY_VAE_BATCH_CHUNK_SIZE" \
    "${checkpoint_identity_args[@]}" \
    --state-dir "$OUTPUT_DIR/checkpoints/state/$FINAL_TAG" \
    --output "$OUTPUT_DIR/checkpoint_validation.json" \
    >>"$LOG_FILE" 2>&1

log "incremental full-history PEFT complete: $FINAL_CHECKPOINT"
