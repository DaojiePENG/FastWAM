#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${LEAPBOTCE_GPU:-5}"

exec bash scripts/train_zero1.sh 1 \
  task=libero_spatial_leapbotce_smoke \
  "$@"
