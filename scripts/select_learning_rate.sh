#!/usr/bin/env bash

set -euo pipefail

# Canonical non-statistical LR decision path. This validates only the selected
# completed run and records an explicit user decision. It never runs or claims
# the fixed-noise paired LR audit.

ROOT_DIR="${ROOT_DIR:-/home/sheng/workspace/leapbot-va}"
SCREEN_ROOT="${SCREEN_ROOT:-$ROOT_DIR/runs/lr_screen_incremental_v6_mb10_ga2_s100_bs80_chunk1}"
SELECTED_LR="${SELECTED_LR:-1.0e-4}"
FINAL_STEP="${FINAL_STEP:-100}"
SELECTION_REASON="${SELECTION_REASON:?SELECTION_REASON is required}"
USER_SELECTION_NOTE="${USER_SELECTION_NOTE:?USER_SELECTION_NOTE is required}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$ROOT_DIR/runs/lr_selection_user_directed_s100/learning_rate_selection.json}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-false}"

if [[ "$FINAL_STEP" -ne 100 ]]; then
    printf 'User-directed LR selection requires the complete step100 screen; got %s.\n' \
        "$FINAL_STEP" >&2
    exit 2
fi
case "$SELECTED_LR" in
    1.0e-5) run_name=lr1p0e-5 ;;
    1.0e-4) run_name=lr1p0e-4 ;;
    *)
        printf 'SELECTED_LR must be exactly 1.0e-5 or 1.0e-4; got %s.\n' \
            "$SELECTED_LR" >&2
        exit 2
        ;;
esac

if [[ -e "$OUTPUT_MANIFEST" && "$ALLOW_OVERWRITE" != "true" ]]; then
    printf 'Refusing to overwrite existing LR decision manifest: %s\n' \
        "$OUTPUT_MANIFEST" >&2
    exit 2
fi

run_root="$SCREEN_ROOT/$run_name"
final_tag="$(printf 'step_%06d' "$FINAL_STEP")"
checkpoint="$run_root/checkpoints/weights/$final_tag.pt"
state_dir="$run_root/checkpoints/state/$final_tag"
run_contract="$run_root/run_contract.txt"

"$ROOT_DIR/.venv/bin/python" \
    "$ROOT_DIR/scripts/history_audit_selection.py" \
    create-user-directed-learning-rate \
    --selected-learning-rate "$SELECTED_LR" \
    --checkpoint "$checkpoint" \
    --state-dir "$state_dir" \
    --run-contract "$run_contract" \
    --selection-reason "$SELECTION_REASON" \
    --user-selection-note "$USER_SELECTION_NOTE" \
    --output "$OUTPUT_MANIFEST" \
    >/dev/null

"$ROOT_DIR/.venv/bin/python" \
    "$ROOT_DIR/scripts/history_audit_selection.py" validate \
    --manifest "$OUTPUT_MANIFEST" \
    --expected-kind learning_rate \
    --allowed-basis user_directed \
    >/dev/null

printf 'User-directed LR selection recorded (no statistical LR audit): %s\n' \
    "$OUTPUT_MANIFEST"
