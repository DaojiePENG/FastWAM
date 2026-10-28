#!/usr/bin/env bash
# LeapBot-CE cloud-edge deployment — debug-pod launcher (LAN, H800 x2).
#
# Cloud stage on $CLOUD_GPU, edge stage on $EDGE_GPU (defaults 3 / 5 — pick
# the two cards with the most free VRAM via nvidia-smi before launching).  The
# bench client runs on CPU and drives both services over HTTP; on one host the
# legs share CLOCK_MONOTONIC.
#
# 4090D/5080 testbed: run cloud_server.py on the cloud host and
# edge_server.py on the edge host (same flags), then
#   python bench_client.py --cloud http://<cloud-host>:8600 \
#                          --edge http://<edge-host>:8601 --out results/....
#
# Env knobs: CLOUD_GPU / EDGE_GPU / CLOUD_PORT / EDGE_PORT / N_REPLANS /
# WARMUP / CKPT / STATS / EXTRA_BENCH_ARGS.
set -euo pipefail

FASTWAM_ROOT="/dahuafs/userdata/2653853/code/FastWAM"
PY="/dahuafs/userdata/2653853/code/.envs/fastwam-robotwin-2653853/bin/python"
RUN_DIR="/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/codec_a6_lora_fs_15ep_seed42_h200"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_032544.pt}"
STATS="${STATS:-${RUN_DIR}/dataset_stats.json}"
CLOUD_GPU="${CLOUD_GPU:-3}"
EDGE_GPU="${EDGE_GPU:-5}"
CLOUD_PORT="${CLOUD_PORT:-8600}"
EDGE_PORT="${EDGE_PORT:-8601}"
N_REPLANS="${N_REPLANS:-50}"
WARMUP="${WARMUP:-5}"
OUT="${OUT:-${FASTWAM_ROOT}/deploy_cloudedge/results/${1:-h800_lan}.json}"

cd "${FASTWAM_ROOT}/deploy_cloudedge"
mkdir -p results logs

# Same environment the LIBERO sweep lanes run under (local backbone weights,
# no downloads; thread caps against the cluster thread limit).
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/c2-worldmodel/2653853/leapbot-ce/models"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HOME="/mnt/c2-worldmodel/2653853/leapbot-ce/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

cleanup() {
  [[ -n "${CLOUD_PID:-}" ]] && kill "${CLOUD_PID}" 2>/dev/null || true
  [[ -n "${EDGE_PID:-}" ]] && kill "${EDGE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[launch] cloud stage -> GPU ${CLOUD_GPU}, port ${CLOUD_PORT}"
CUDA_VISIBLE_DEVICES="${CLOUD_GPU}" "${PY}" cloud_server.py \
  --port "${CLOUD_PORT}" --ckpt "${CKPT}" --stats "${STATS}" \
  > "logs/cloud_${CLOUD_GPU}.log" 2>&1 &
CLOUD_PID=$!

echo "[launch] edge stage  -> GPU ${EDGE_GPU}, port ${EDGE_PORT}"
CUDA_VISIBLE_DEVICES="${EDGE_GPU}" "${PY}" edge_server.py \
  --port "${EDGE_PORT}" --ckpt "${CKPT}" --stats "${STATS}" \
  > "logs/edge_${EDGE_GPU}.log" 2>&1 &
EDGE_PID=$!

wait_http() {
  local url="$1" name="$2" i=0
  until "${PY}" - "$url" <<'PYEOF' >/dev/null 2>&1
import sys, requests
requests.get(sys.argv[1], timeout=5).raise_for_status()
PYEOF
  do
    i=$((i + 1))
    if (( i > 240 )); then
      echo "[launch] ${name} did not become healthy in 20 min; see logs/" >&2
      exit 1
    fi
    sleep 5
  done
}

wait_http "http://127.0.0.1:${CLOUD_PORT}/health" "cloud"
echo "[launch] cloud healthy"
wait_http "http://127.0.0.1:${EDGE_PORT}/health" "edge"
echo "[launch] edge healthy"

"${PY}" bench_client.py \
  --cloud "http://127.0.0.1:${CLOUD_PORT}" \
  --edge "http://127.0.0.1:${EDGE_PORT}" \
  --n "${N_REPLANS}" --warmup "${WARMUP}" \
  --tag "${1:-h800_lan}" --out "${OUT}" \
  ${EXTRA_BENCH_ARGS:-}
