#!/usr/bin/env bash
# LeapBotCE training-ablation eval, unattended: WAIT for each training line to
# write its terminal checkpoint, then run the standard four-window LIBERO sweep
# for it, back to back on the same node.
#
# Why this exists (2026-11-05): the two rows still missing from tab:ablations
# are `w/o cloud (training)` (codec_a6_edgeonly_15ep_seed42_h200) and
# `single-branch random-age` (codec_a6_singleage_15ep_seed42_h200). Their
# terminal ckpt step_032544.pt does not exist yet, so the plain variant config
# (AIFlow_config_eval_libero_a6_edgeonly.py / ..._singleage.py) refuses to run.
# Submitting this wrapper instead ties up the node while the trainings finish,
# which is cheaper than missing the window: the user is travelling and cannot
# babysit two separate submissions.
#
# Contract with the plain variants (must stay identical):
#   TAG              a6_edgeonly_step32544 / a6_singleage_step32544
#   delays           0 10 20 40          (10-15 delay-window rule)
#   trials           50                  → 4 suites × 10 tasks × 50 = 2000
#   PACK             1                   (per-worker ~25 GB, 4 workers on 4 cards)
#   eval extra       edgeonly: model.cloudedge.use_cloud_cache=false
#                    singleage: none
# The sweep is skip-existing, so re-running this script never duplicates cells.
#
# Usage (platform CMD):  bash scripts/eval_a6_trainablations_wait.sh
# Optional env: LEAPBOTCE_WAIT_MAX_H (default 36), LEAPBOTCE_SKIP_EDGEONLY=1,
#               LEAPBOTCE_SKIP_SINGLEAGE=1, GPU_NUM (card count, default 4).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
SWEEP="${ROOT}/scripts/eval_leapbotce_libero_sweep_2653853.sh"
RUNS=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora
WAIT_MAX_H="${LEAPBOTCE_WAIT_MAX_H:-36}"
GPU_NUM="${GPU_NUM:-4}"
LOG_DIR=/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/logs
mkdir -p "${LOG_DIR}"
HEARTBEAT="${LOG_DIR}/a6_trainablations_wait.heartbeat"
STAMP() { date '+%F %T'; }
SAY()   { echo "[$(STAMP)] $*" | tee -a "${HEARTBEAT}"; }

[[ -f "${SWEEP}" ]] || { SAY "FATAL: sweep driver missing: ${SWEEP}"; exit 2; }

# wait_ckpt <ckpt path> <label>
# A 2 GB safetensors write is not atomic; treat the file as ready only once its
# size is >1 GB and unchanged across a 120 s window.
wait_ckpt() {
  local path="$1" label="$2" s1 s2 deadline
  deadline=$(( $(date +%s) + WAIT_MAX_H * 3600 ))
  while :; do
    SAY "wait:${label}: polling ${path}"
    if [[ -f "${path}" ]]; then
      s1=$(stat -c %s "${path}" 2>/dev/null || echo 0)
      sleep 120
      s2=$(stat -c %s "${path}" 2>/dev/null || echo 0)
      if [[ "${s1}" -gt 1000000000 && "${s1}" == "${s2}" ]]; then
        SAY "wait:${label}: READY (${s2} bytes, stable)"
        return 0
      fi
      SAY "wait:${label}: still writing (${s1} -> ${s2})"
    fi
    if (( $(date +%s) > deadline )); then
      SAY "wait:${label}: TIMEOUT after ${WAIT_MAX_H}h, giving up on this row"
      return 1
    fi
    sleep 300
  done
}

# run_row <tag> <run dir> <ckpt> <extra overrides or empty>
run_row() {
  local tag="$1" run_dir="$2" ckpt="$3" extra="${4:-}"
  local -a envs=(
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
    "LEAPBOTCE_CKPT=${ckpt}"
    "RUN_DIR=${run_dir}"
    "LIBERO_EVAL_TAG=${tag}"
    "LIBERO_EVAL_DELAYS=0 10 20 40"
    "NUM_TRIALS=50"
    "LIBERO_LIBERO_PACK=1"
    "GPU_NUM=${GPU_NUM}"
  )
  [[ -n "${extra}" ]] && envs+=("LIBERO_EVAL_EXTRA=${extra}")
  SAY "eval:${tag}: start (extra='${extra:-none}')"
  if [[ "${LEAPBOTCE_DRYRUN:-0}" == "1" ]]; then
    SAY "eval:${tag}: DRYRUN cmd: env ${envs[*]} bash ${SWEEP}"
    return 0
  fi
  ( cd "${ROOT}" && env "${envs[@]}" bash "${SWEEP}" ) \
    >> "${LOG_DIR}/a6_trainablations_${tag}.log" 2>&1
  local rc=$?
  SAY "eval:${tag}: exit rc=${rc}  (log: ${LOG_DIR}/a6_trainablations_${tag}.log)"
  return "${rc}"
}

EDGE_RUN="${LEAPBOTCE_EDGE_RUN:-${RUNS}/codec_a6_edgeonly_15ep_seed42_h200}"
EDGE_CKPT="${LEAPBOTCE_EDGE_CKPT:-${EDGE_RUN}/checkpoints/weights/step_032544.pt}"
SING_RUN="${LEAPBOTCE_SING_RUN:-${RUNS}/codec_a6_singleage_15ep_seed42_h200}"
SING_CKPT="${LEAPBOTCE_SING_CKPT:-${SING_RUN}/checkpoints/weights/step_032544.pt}"

SAY "start: GPU_NUM=${GPU_NUM} WAIT_MAX_H=${WAIT_MAX_H}"

if [[ "${LEAPBOTCE_SKIP_EDGEONLY:-0}" != "1" ]]; then
  if wait_ckpt "${EDGE_CKPT}" edgeonly; then
    run_row a6_edgeonly_step32544 "${EDGE_RUN}" "${EDGE_CKPT}" \
      "model.cloudedge.use_cloud_cache=false"
  fi
fi

if [[ "${LEAPBOTCE_SKIP_SINGLEAGE:-0}" != "1" ]]; then
  if wait_ckpt "${SING_CKPT}" singleage; then
    run_row a6_singleage_step32544 "${SING_RUN}" "${SING_CKPT}" ""
  fi
fi

SAY "done: both rows attempted; see per-tag logs"
