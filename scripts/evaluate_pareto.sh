#!/usr/bin/env bash

set -euo pipefail

# Evaluate the trained exit-depth x retained-KV grid and select the Pareto set.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
TRAIN_ROOT="${TRAIN_ROOT:?TRAIN_ROOT is required}"
MODE="${MODE:?MODE is required}"
FINAL_STEP="${FINAL_STEP:?FINAL_STEP is required}"
GRID_ROOT="${GRID_ROOT:-$ROOT_DIR/evaluate_results/leapbot_depth_history_pareto}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS:-70}"
DEPTHS_CSV="${DEPTHS_CSV:-8,16,24,30}"
HISTORY_CAPS_CSV="${HISTORY_CAPS_CSV:-0,8,16,32,full}"
KV_RETENTION_CAPS_CSV="${KV_RETENTION_CAPS_CSV:-$HISTORY_CAPS_CSV}"
BASELINE_RESULTS_ROOT="${BASELINE_RESULTS_ROOT:-}"
DATASET_STATS="${LEAPBOT_DATASET_STATS:-$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
FINAL_STEP_TAG="$(printf 'step_%06d' "$FINAL_STEP")"
CHECKPOINT="$TRAIN_ROOT/$MODE/checkpoints/weights/$FINAL_STEP_TAG.pt"
RUN_CONTRACT_FILE="$TRAIN_ROOT/$MODE/run_contract.txt"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ ! -s "$CHECKPOINT" ]]; then
    printf 'Multi-exit checkpoint not ready: %s\n' "$CHECKPOINT" >&2
    exit 2
fi
if [[ ! -s "$RUN_CONTRACT_FILE" ]]; then
    printf 'Training run contract missing: %s\n' "$RUN_CONTRACT_FILE" >&2
    exit 2
fi
EXPECTED_RUN_CONTRACT_SHA256="$(awk -F= '$1 == "run_contract_sha256" {print $2}' "$RUN_CONTRACT_FILE")"
EXPECTED_CODE_COMMIT="$(awk -F= '$1 == "code_commit" {print $2}' "$RUN_CONTRACT_FILE")"
if [[ ! "$EXPECTED_RUN_CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$EXPECTED_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Invalid training identity in %s\n' "$RUN_CONTRACT_FILE" >&2
    exit 2
fi

mkdir -p "$GRID_ROOT/.checkpoint_validation"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/validate_leapbot_checkpoint.py" \
    "$CHECKPOINT" \
    --expected-step "$FINAL_STEP" \
    --expected-mode "$MODE" \
    --expected-trained-exit-depths 8,16,24,30 \
    --expected-history-vae-batch-chunk-size 1 \
    --expected-run-contract-sha256 "$EXPECTED_RUN_CONTRACT_SHA256" \
    --expected-code-commit "$EXPECTED_CODE_COMMIT" \
    --state-dir "$TRAIN_ROOT/$MODE/checkpoints/state/$FINAL_STEP_TAG" \
    --output "$GRID_ROOT/.checkpoint_validation/${MODE}_${FINAL_STEP_TAG}.json" \
    >/dev/null

IFS=',' read -r -a DEPTHS <<<"$DEPTHS_CSV"
IFS=',' read -r -a KV_RETENTION_CAPS <<<"$KV_RETENTION_CAPS_CSV"
for depth in "${DEPTHS[@]}"; do
    case "$depth" in
        8|16|24|30) ;;
        *)
            printf 'Unsupported exit depth: %s\n' "$depth" >&2
            exit 2
            ;;
    esac
    for kv_retention_cap in "${KV_RETENTION_CAPS[@]}"; do
        if [[ "$kv_retention_cap" != "full" ]] \
            && { [[ ! "$kv_retention_cap" =~ ^[0-9]+$ ]] \
                || (( kv_retention_cap > MAX_HISTORY_BLOCKS )); }; then
            printf 'Invalid KV retention cap: %s\n' "$kv_retention_cap" >&2
            exit 2
        fi
        config_root="$GRID_ROOT/configs/d${depth}_kvret${kv_retention_cap}"
        log "evaluate mode=$MODE depth=$depth kv-retention=$kv_retention_cap trials=$NUM_TRIALS"
        ROOT_DIR="$ROOT_DIR" \
        TRAIN_ROOT="$TRAIN_ROOT" \
        EVAL_ROOT="$config_root" \
        MODE="$MODE" \
        FINAL_STEP="$FINAL_STEP" \
        NUM_TRIALS="$NUM_TRIALS" \
        GPU_IDS_CSV="$GPU_IDS_CSV" \
        LEAPBOT_DATASET_STATS="$DATASET_STATS" \
        MAX_HISTORY_BLOCKS="$MAX_HISTORY_BLOCKS" \
        RETAINED_HISTORY_BLOCKS="$kv_retention_cap" \
        EXIT_DEPTH="$depth" \
        EXPECTED_TRAINED_EXIT_DEPTHS=8,16,24,30 \
        bash "$ROOT_DIR/scripts/evaluate_checkpoint.sh"
    done
done

pareto_inputs=("$GRID_ROOT/configs")
if [[ -n "$BASELINE_RESULTS_ROOT" ]]; then
    if [[ ! -d "$BASELINE_RESULTS_ROOT" ]]; then
        printf 'BASELINE_RESULTS_ROOT does not exist: %s\n' "$BASELINE_RESULTS_ROOT" >&2
        exit 2
    fi
    pareto_inputs+=("$BASELINE_RESULTS_ROOT")
fi

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/pareto.py" \
    "${pareto_inputs[@]}" \
    --output-dir "$GRID_ROOT/pareto" \
    --expected-tasks 10 \
    --expected-trials-per-task "$NUM_TRIALS" \
    --require-profiled
MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib" \
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/experiments/leapbot/plot_pareto.py" \
    "$GRID_ROOT/pareto" \
    --output-dir "$GRID_ROOT/pareto"
log "depth/kv-retention Pareto complete: $GRID_ROOT/pareto/results.csv"
