#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/tmp/leapbotce_20261029/FastWAM
cd "$TASK_ROOT/deploy_cloudedge"
export PYTHONPATH=/tmp/leapbotce_20261029/site-packages:$TASK_ROOT/src
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
exec /home/aihub/miniconda3/envs/fastwam/bin/python -u cloud_server.py --port 8600 \
 --cloud-artifact /tmp/leapbotce_20261029/artifacts/performance_full_seed0.pt
