#!/usr/bin/env bash

set -euo pipefail

# Hardware acceptance only: two checkpoint-free ZeRO-2 optimizer updates on a
# fixed per-rank batch of distinct, genuine LIBERO H41-H50 episode prefixes.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MODE="${MODE:-action_aggregator}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29969}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
MAX_PREFLIGHT_USED_MIB="${MAX_PREFLIGHT_USED_MIB:-2048}"
PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-7200}"
RELEASE_CHECKPOINT="${RELEASE_CHECKPOINT:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
TEXT_EMBEDDING_CACHE="${TEXT_EMBEDDING_CACHE:-$ROOT_DIR/data/text_embeds_cache/libero}"
ASSET_DOWNLOAD_MANIFEST="${ASSET_DOWNLOAD_MANIFEST:-$ROOT_DIR/data/leapbot_asset_download_manifest.json}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-$ROOT_DIR/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/acceptance/zero2_real_h41_h50_${MODE}_b${BATCH_SIZE}}"
LOG_FILE="$OUTPUT_DIR/probe.log"
REPORT_FILE="$OUTPUT_DIR/capacity_probe.json"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

case "$BATCH_SIZE" in
    16|18|20|22) ;;
    *)
        log "BATCH_SIZE must be one of 16,18,20,22; use B20 first and B18 after OOM"
        exit 2
        ;;
esac
case "$MODE" in
    action_aggregator|interleaved|vision_causal) ;;
    *)
        log "MODE must be action_aggregator, interleaved or vision_causal"
        exit 2
        ;;
esac
if ! [[ "$PROBE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    log "PROBE_TIMEOUT_SECONDS must be a positive integer"
    exit 2
fi
if [[ "$NUM_PROCESSES" -ne 8 ]]; then
    log "capacity probe requires exactly eight ranks"
    exit 2
fi
if [[ "$(awk -F, '{print NF}' <<<"$GPU_IDS_CSV")" -ne 8 ]]; then
    log "GPU_IDS_CSV must contain exactly eight GPU ids"
    exit 2
fi
if [[ ! -s "$RELEASE_CHECKPOINT" || ! -s "$DATASET_STATS" ]]; then
    log "release checkpoint or release normalization statistics are missing"
    exit 2
fi
if [[ ! -d "$TEXT_EMBEDDING_CACHE" || ! -s "$ASSET_DOWNLOAD_MANIFEST" \
    || ! -s "$VAE_CHECKPOINT" ]]; then
    log "verified T5 cache, download manifest or VAE checkpoint is missing"
    exit 2
fi
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
    log "capacity acceptance must run from a clean committed worktree"
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]] \
    && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    log "refusing non-empty capacity output directory: $OUTPUT_DIR"
    exit 2
fi

while IFS= read -r used; do
    if (( used > MAX_PREFLIGHT_USED_MIB )); then
        log "GPU preflight failed: selected GPU uses ${used} MiB"
        exit 2
    fi
done < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v ids="$GPU_IDS_CSV" \
            'BEGIN {split(ids, a, ","); for (i in a) wanted[a[i]]=1} wanted[NR-1] {print $1}'
)

mkdir -p "$OUTPUT_DIR"
TEXT_CACHE_LOCK_FILE="$TEXT_EMBEDDING_CACHE/.leapbot_text_cache.lock"
exec {TEXT_CACHE_LOCK_FD}>>"$TEXT_CACHE_LOCK_FILE"
flock -s "$TEXT_CACHE_LOCK_FD"

TRAINING_ASSET_MANIFEST="$OUTPUT_DIR/training_asset_manifest.json"
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/training_asset_manifest.py" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot" \
    --dataset-dir "$ROOT_DIR/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot" \
    --text-embedding-cache "$TEXT_EMBEDDING_CACHE" \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --download-manifest "$ASSET_DOWNLOAD_MANIFEST" \
    --output "$TRAINING_ASSET_MANIFEST"
TRAINING_ASSET_MANIFEST_SHA256="$(
    "$ROOT_DIR/.venv/bin/python" - "$TRAINING_ASSET_MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])
PY
)"
if [[ ! "$TRAINING_ASSET_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    log "training asset manifest did not produce a valid identity"
    exit 2
fi
CODE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
RELEASE_CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT" | awk '{print $1}')"
DATASET_STATS_SHA256="$(sha256sum "$DATASET_STATS" | awk '{print $1}')"
GLOBAL_BATCH=$((NUM_PROCESSES * BATCH_SIZE))
{
    printf 'kind=zero2_real_high_history_capacity_probe\n'
    printf 'code_commit=%s\n' "$CODE_COMMIT"
    printf 'release_checkpoint_sha256=%s\n' "$RELEASE_CHECKPOINT_SHA256"
    printf 'dataset_stats_sha256=%s\n' "$DATASET_STATS_SHA256"
    printf 'training_asset_manifest_sha256=%s\n' "$TRAINING_ASSET_MANIFEST_SHA256"
    printf 'mode=%s\n' "$MODE"
    printf 'num_processes=8\n'
    printf 'batch_size=%s\n' "$BATCH_SIZE"
    printf 'gradient_accumulation_steps=1\n'
    printf 'global_batch=%s\n' "$GLOBAL_BATCH"
    printf 'optimizer_updates=2\n'
    printf 'history_range=41-50\n'
    printf 'history_source=distinct_real_dataset_prefixes\n'
    printf 'history_training_mode=incremental_full_bptt\n'
    printf 'history_vae_batch_chunk_size=1\n'
    printf 'world_model_conditioning=lingbot_teacher_forced_v1\n'
    printf 'num_video_frames=9\n'
    printf 'future_video_condition_noise_probability=0.5\n'
    printf 'future_video_condition_min_u=0.5\n'
    printf 'future_video_condition_max_u=1.0\n'
    printf 'training_strategy=video_lora_action_full\n'
    printf 'training_exit_depths=30\n'
    printf 'mixed_precision=bf16\n'
    printf 'zero_stage=2\n'
    printf 'wandb=false\n'
    printf 'optimizer=adamw_beta0.9_0.95_clip1.0\n'
    printf 'optimizer_group_action_and_aux_weight_decay=0.01\n'
    printf 'optimizer_group_video_lora_weight_decay=0.0\n'
    printf 'timeout_seconds=%s\n' "$PROBE_TIMEOUT_SECONDS"
    printf 'checkpoint_writes=false\n'
} >"$OUTPUT_DIR/probe_contract.txt"

log "start ZeRO-2 real H41-H50 capacity probe: mode=${MODE} B${BATCH_SIZE}/GA1/global${GLOBAL_BATCH}"
set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" \
PYTHONHASHSEED=42 \
LEAPBOT_CODE_COMMIT="$CODE_COMMIT" \
LEAPBOT_TRAINING_ASSET_MANIFEST_SHA256="$TRAINING_ASSET_MANIFEST_SHA256" \
LEAPBOT_CAPACITY_PROBE_TIMEOUT_SECONDS="$PROBE_TIMEOUT_SECONDS" \
LEAPBOT_DATASET_STATS="$DATASET_STATS" \
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    timeout --signal=TERM --kill-after=60s "$PROBE_TIMEOUT_SECONDS" \
    "$ROOT_DIR/.venv/bin/accelerate" launch \
    --config_file "$ROOT_DIR/scripts/accelerate_configs/accelerate_zero2_ds.yaml" \
    --num_processes 8 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$ROOT_DIR/scripts/probe_zero2_high_history_capacity.py" \
    task=libero_leapbot_2cam224 \
    "output_dir=$OUTPUT_DIR" \
    "model.causal_mode=$MODE" \
    model.history_training_mode=incremental_full_bptt \
    model.history_vae_batch_chunk_size=1 \
    model.future_video_conditioning=lingbot_teacher_forced_v1 \
    model.num_video_frames=9 \
    model.future_video_condition_noise_probability=0.5 \
    model.future_video_condition_min_u=0.5 \
    model.future_video_condition_max_u=1.0 \
    model.training_strategy=video_lora_action_full \
    model.video_lora.enabled=true \
    model.video_lora.rank=16 \
    model.video_lora.alpha=16.0 \
    model.video_lora.dropout=0.0 \
    model.video_lora.learning_rate_multiplier=1.0 \
    model.mot_checkpoint_mixed_attn=true \
    "model.training_exit_depths=[30]" \
    data.train.full_episode_history=true \
    data.train.min_history_blocks=0 \
    data.train.max_history_blocks=70 \
    data.train.initial_block_oversample=1 \
    "data.train.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE" \
    "data.train.dataset_dirs=[$ROOT_DIR/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,$ROOT_DIR/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot]" \
    "batch_size=$BATCH_SIZE" \
    gradient_accumulation_steps=1 \
    max_steps=2 \
    num_epochs=1 \
    "learning_rate=$LEARNING_RATE" \
    lr_scheduler_type=constant \
    num_workers=3 \
    max_grad_norm=1.0 \
    weight_decay=1.0e-2 \
    log_every=1 \
    save_every=0 \
    eval_every=0 \
    seed=42 \
    mixed_precision=bf16 \
    wandb.enabled=false \
    wandb.mode=disabled \
    "resume=$RELEASE_CHECKPOINT" \
    +capacity_probe.optimizer_updates=2 \
    +capacity_probe.history_min=41 \
    +capacity_probe.history_max=50 \
    >"$LOG_FILE" 2>&1
launcher_status=$?
set -e

checkpoint_files_present=false
if find "$OUTPUT_DIR/checkpoints" -type f -print -quit 2>/dev/null | grep -q .; then
    checkpoint_files_present=true
fi
if [[ "$launcher_status" -eq 0 && -s "$REPORT_FILE" \
    && "$checkpoint_files_present" == false ]]; then
    log "capacity probe passed: $REPORT_FILE"
    exit 0
fi

oom_detected=false
if find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'capacity_probe.rank_*.oom.json' \
    -print -quit | grep -q .; then
    oom_detected=true
elif grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$LOG_FILE"; then
    oom_detected=true
fi
timed_out=false
if [[ "$launcher_status" -eq 124 ]]; then
    timed_out=true
fi

"$ROOT_DIR/.venv/bin/python" - \
    "$REPORT_FILE" "$OUTPUT_DIR" "$launcher_status" "$oom_detected" "$timed_out" \
    "$BATCH_SIZE" "$GLOBAL_BATCH" "$CODE_COMMIT" "$MODE" \
    "$TRAINING_ASSET_MANIFEST_SHA256" "$PROBE_TIMEOUT_SECONDS" <<'PY'
import glob
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
launcher_status = int(sys.argv[3])
oom_detected = sys.argv[4].lower() == "true"
timed_out = sys.argv[5].lower() == "true"
rank_reports = []
for path in sorted(glob.glob(str(output_dir / "capacity_probe.rank_*.oom.json"))):
    with open(path, encoding="utf-8") as handle:
        rank_reports.append(json.load(handle))
checkpoint_files = sorted(
    str(path.relative_to(output_dir))
    for path in (output_dir / "checkpoints").rglob("*")
    if path.is_file()
)
if checkpoint_files:
    status = "contract_violation"
elif timed_out:
    status = "timeout"
elif oom_detected:
    status = "oom"
else:
    status = "failed"
payload = {
    "kind": "zero2_real_high_history_capacity_probe",
    "status": status,
    "batch_size_per_rank": int(sys.argv[6]),
    "world_size": 8,
    "gradient_accumulation_steps": 1,
    "global_batch_size": int(sys.argv[7]),
    "optimizer_updates_requested": 2,
    "history_range": [41, 50],
    "history_tensors_synthesized_or_extended": False,
    "launcher_exit_code": launcher_status,
    "code_commit": sys.argv[8],
    "causal_mode": sys.argv[9],
    "training_asset_manifest_sha256": sys.argv[10],
    "timeout_seconds": int(sys.argv[11]),
    "timed_out": timed_out,
    "rank_oom_reports": rank_reports,
    "checkpoint_files_written": checkpoint_files,
    "contract_violation": bool(checkpoint_files),
    "optimizer": {
        "name": "AdamW",
        "betas": [0.9, 0.95],
        "max_grad_norm": 1.0,
        "parameter_groups": {
            "action_and_aux": {"weight_decay": 0.01},
            "video_lora": {"weight_decay": 0.0},
        },
    },
    "fallback": (
        "rerun this acceptance probe with BATCH_SIZE=18 and a fresh OUTPUT_DIR"
        if oom_detected and int(sys.argv[6]) == 20 and not checkpoint_files
        else None
    ),
}
temporary = report_path.with_suffix(report_path.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
temporary.replace(report_path)
PY

if [[ "$checkpoint_files_present" == true ]]; then
    log "capacity probe checkpoint-write contract violation recorded in $REPORT_FILE"
    exit 3
fi
if [[ "$timed_out" == true ]]; then
    log "capacity probe wall-clock timeout recorded in $REPORT_FILE"
    exit 124
fi
if [[ "$oom_detected" == true ]]; then
    log "capacity probe OOM recorded in $REPORT_FILE"
    exit 42
fi
log "capacity probe failed; inspect $LOG_FILE and $REPORT_FILE"
if [[ "$launcher_status" -eq 0 ]]; then
    exit 1
fi
exit "$launcher_status"
