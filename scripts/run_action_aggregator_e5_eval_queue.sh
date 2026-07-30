#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
E1_TRAIN_ROOT="${E1_TRAIN_ROOT:-$ROOT_DIR/runs/phase1_h8_d30_e1_bs32}"
E1_EVAL_ROOT="${E1_EVAL_ROOT:-$ROOT_DIR/evaluate_results/action_aggregator_h8_e1_bs32_lr1e5_curve2/step_000892}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

mkdir -p "$E1_EVAL_ROOT"
if [[ ! -s "$E1_EVAL_ROOT/pareto/pareto.json" ]]; then
    log "screening conservative one-epoch action_aggregator on GPUs 6-7"
    if ! TRAIN_ROOT="$E1_TRAIN_ROOT" \
        EVAL_ROOT="$E1_EVAL_ROOT" \
        MODE=action_aggregator \
        FINAL_STEP=892 \
        NUM_TRIALS=2 \
        GPU_IDS_CSV=6,7 \
        bash "$ROOT_DIR/scripts/run_single_mode_checkpoint_eval.sh"; then
        log "warning: conservative action_aggregator screening failed; continuing e5 queue"
    fi
else
    log "skip completed conservative one-epoch action_aggregator screening"
fi

log "switching GPUs 6-7 to five-epoch action_aggregator learning curve"
exec bash "$ROOT_DIR/scripts/run_action_aggregator_e5_intermediate_evals.sh"
