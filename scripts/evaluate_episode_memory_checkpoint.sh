#!/usr/bin/env bash

set -euo pipefail

# Episode-memory evaluation is controlled by sim_leapbot_libero_episode_memory.yaml.
# The checkpoint is the only experiment-specific runtime input.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${LEAPBOT_PYTHON:-${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python 2>/dev/null || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Python is unavailable; activate Conda/uv or set LEAPBOT_PYTHON.\n' >&2
    exit 2
fi

CKPT="${CKPT:?CKPT is required}"
if [[ ! -s "$CKPT" ]]; then
    printf 'Checkpoint not ready: %s\n' "$CKPT" >&2
    exit 2
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" experiments/libero/run_libero_manager.py \
    --config-name sim_leapbot_libero_episode_memory \
    task=libero_leapbot_episode_memory \
    "ckpt=$CKPT"
