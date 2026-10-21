#!/usr/bin/env bash
# LeapBotCE (ours) RoboTwin 2.0 cloud-delay sweep — company training-cluster
# driver. Mirrors fastwam_official's run_sweep_cluster_2653853.sh so the
# LeapBotCE row is collected under the same conditions as the baseline rows.
#
# Runs on an AIFlow-allocated GPU node (also usable standalone). Discovers the
# node's GPUs, PACKs multiple lanes per GPU by VRAM, deals the
# 5 delays x 10 tasks round-robin across ALL lanes, waits, and prints a
# per-delay completion summary.
#
# Usage:
#   bash eval_leapbotce_robotwin_sweep_2653853.sh [GPU_IDS_CSV] ["0 5 10 20 40"]
#
# GPU discovery order:
#   1. $1 (physical or comma CSV)
#   2. $CUDA_VISIBLE_DEVICES (platform-assigned; lanes get logical ids)
#   3. $GPU_NUM (AIFlow) -> 0..N-1
#   4. all GPUs from nvidia-smi
#
# Overrides for smoke runs:
#   ROBOTWIN_EVAL_DELAYS="0" ROBOTWIN_EVAL_TASKS="beat_block_hammer" ...
#   FASTWAM_EVAL_PYTHON=/path/to/python   (conda env on non-shared homes)
#   LANE_DRIVER=/path/to/lane_driver.sh
#   LANES_PER_GPU=2                       (default: auto by smallest card VRAM)
#   LEAPBOTCE_CKPT=/path/to/step_xxxxxx.pt (or "latest"; REQUIRED)
#
# Results: /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/robotwin_sweep/delay_<D>/<task>/
#   shared storage — the debug node's report generator picks them up directly.
set -u

LANE_DRIVER="${LANE_DRIVER:-/dahuafs/userdata/2653853/code/baseline_delay/leapbotce_robotwin_sweep_lane_2653853.sh}"
# LeapBotCE dev/training tree (leapbotce_policy + sim_robotwin_leapbotce.yaml).
FASTWAM_ROOT="${FASTWAM_ROOT:-/dahuafs/userdata/2653853/code/FastWAM}"
SWEEP_ROOT="${LEAPBOTCE_SWEEP_ROOT:-/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/robotwin_sweep}"
LOGDIR="${SWEEP_ROOT}/_cluster_logs"
RUN_DIR="${LEAPBOTCE_RUN_DIR:-/mnt/c2-worldmodel/2653853/leapbot-ce/runs/robotwin_all_leapbotce/robotwin_all_dtrain20_seed42_h200_n2}"
WAN_ROOT="/mnt/c2-worldmodel/2653853/leapbot-ce/models"
PY="${FASTWAM_EVAL_PYTHON:-/dahuafs/userdata/2653853/code/.envs/fastwam-robotwin-2653853/bin/python}"
ALL_TASKS=(beat_block_hammer adjust_bottle click_alarmclock dump_bin_bigbin grab_roller handover_block put_object_cabinet scan_object shake_bottle stack_blocks_two)

echo "[cluster-sweep] start $(date '+%F %T') host=$(hostname)"

# ---------- checkpoint resolution ----------
CKPT="${LEAPBOTCE_CKPT:-latest}"
if [[ "$CKPT" == "latest" ]]; then
  CKPT=$(ls -1v "${RUN_DIR}/checkpoints/weights/"step_*.pt 2>/dev/null | tail -1)
fi

# ---------- preflight: fail fast BEFORE occupying any GPU ----------
fail() { echo "[cluster-sweep] PREFLIGHT FAIL: $*" >&2; exit 1; }
[[ -f "$LANE_DRIVER" ]] || fail "lane driver missing: $LANE_DRIVER"
[[ -x "$PY" ]] || fail "eval python missing: $PY (set FASTWAM_EVAL_PYTHON)"
[[ -n "$CKPT" && -f "$CKPT" ]] || fail "leapbotce checkpoint not found (set LEAPBOTCE_CKPT; looked at: '${CKPT}'; have: $(ls -1v "${RUN_DIR}/checkpoints/weights/" 2>/dev/null | tr '\n' ' '))"
[[ -f "${RUN_DIR}/dataset_stats.json" ]] || fail "dataset_stats.json missing in $RUN_DIR"
[[ -d "$WAN_ROOT" ]] || fail "Wan2.2 backbone dir missing: $WAN_ROOT"
"$PY" -c 'import timm' 2>/dev/null || fail "eval env lacks timm (LeapBotCE edge SigLIP): $PY"
[[ -d "$FASTWAM_ROOT/third_party/RoboTwin" ]] || fail "vendored RoboTwin missing under $FASTWAM_ROOT"
[[ -e "$FASTWAM_ROOT/third_party/RoboTwin/assets/embodiments" ]] || fail "RoboTwin assets link broken (assets/embodiments)"
grep -q "aloha-agilex" "$FASTWAM_ROOT/third_party/RoboTwin/task_config/demo_clean.yml" \
  || fail "task_config/demo_clean.yml does not set embodiment: [aloha-agilex]"
# ---------- render probe: fail fast BEFORE occupying any GPU ----------
# 2026-09-30 pod job 0b218e7a: SAPIEN render init failed silently on the pod,
# every cell burned its 50 episodes in ~90 s and still exited rc=0. Probe the
# exact init path now, with the same userland wiring the lane driver applies
# (nvidia_userland_wire.sh picks the package matching this node's kernel-
# module driver version), so a broken render environment dies here with a
# diagnosis instead of burning the whole allocation.
# shellcheck disable=SC1091
source /dahuafs/userdata/2653853/code/baseline_delay/nvidia_userland_wire.sh
if [[ -n "${NW_WIRE_MISSING:-}" ]]; then
  fail "no NVIDIA userland package for host driver $NW_WIRE_MISSING (have: $(ls -d /dahuafs/userdata/2653853/nvidia-graphics-*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')) — build it: nvidia-graphics-extract.sh <VER> <matching NVIDIA-Linux-x86_64-<VER>.run>"
fi
PROBE_LOG=/tmp/leapbotce_render_probe.log
# VK_LOADER_DEBUG=error: if the probe fails we get the Vulkan loader's own
# verdict (ICD rejected / rmapi handshake) in PROBE_LOG, not just SAPIEN's
# generic "failed to find a rendering device" (2026-10-10 h200 verdict chain).
if ! env VK_LOADER_DEBUG=error "$PY" -c 'import sapien.core as sapien; e = sapien.Engine(); e.set_renderer(sapien.SapienRenderer())' \
    2>"$PROBE_LOG"; then
  echo "[cluster-sweep] RENDER PROBE FAILED (host driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); wired: ${NVGL:-container-native})" >&2
  echo "[cluster-sweep] probe stderr (last 20 lines of $PROBE_LOG):" >&2
  tail -20 "$PROBE_LOG" >&2 2>/dev/null || true
  fail "SAPIEN render init failed — fix the userland/driver match before resubmitting (stderr above)"
fi
echo "[cluster-sweep] render probe OK"

# ---------- GPU discovery ----------
declare -a GPUS=()
if [[ $# -ge 1 && -n "$1" ]]; then
  IFS=',' read -ra GPUS <<< "$1"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
  PLATFORM_CVD=1
elif [[ -n "${GPU_NUM:-}" ]]; then
  GPUS=($(seq 0 $((GPU_NUM - 1))))
else
  GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '))
fi
N=${#GPUS[@]}
[[ "$N" -ge 1 ]] || fail "no GPUs discovered"
echo "[cluster-sweep] GPUs: ${GPUS[*]} (n=$N)"

# ---------- delays, tasks, lane packing ----------
read -ra DELAYS <<< "${ROBOTWIN_EVAL_DELAYS:-${2:-0 5 10 20 40}}"
read -ra TASKS <<< "${ROBOTWIN_EVAL_TASKS:-${ALL_TASKS[*]}}"
ND=${#DELAYS[@]}

# Cells are ~34 GB each (same footprint as the fastwam baseline row: Wan2.2-5B
# video expert + T5 + SigLIP edge encoder + SAPIEN rendering). Pack as many
# lanes per GPU as VRAM allows. RESERVE absorbs allocator spikes when two Wan
# loads overlap. Re-measure on first launch before raising PACK.
CELL_VRAM_MIB=34000
RESERVE_MIB=6000
MIN_TOTAL=""
if [[ -n "${LANES_PER_GPU:-}" ]]; then
  PACK="$LANES_PER_GPU"
else
  MIN_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  [[ -n "$MIN_TOTAL" ]] || MIN_TOTAL=81559
  PACK=$(( (MIN_TOTAL - RESERVE_MIB) / CELL_VRAM_MIB ))
  (( PACK >= 1 )) || PACK=1
fi
L=$(( N * PACK ))
echo "[cluster-sweep] ckpt: $CKPT"
echo "[cluster-sweep] delays: ${DELAYS[*]} | tasks: ${#TASKS[@]} | lanes/GPU=${PACK} -> ${L} lanes"

# Enumerate cells delay-major, deal round-robin over lanes. Lane L_i collects
# cells i, i+L, i+2L, ...; consecutive same-delay cells merge into one lane
# driver invocation (the driver runs one delay per call).
CELLS=()
for D in "${DELAYS[@]}"; do
  for T in "${TASKS[@]}"; do CELLS+=("${D}:${T}"); done
done
C=${#CELLS[@]}

mkdir -p "$LOGDIR"
PIDS=()
for LANE in $(seq 0 $((L - 1))); do
  SLOT=$((LANE % N))
  RAW_GPU="${GPUS[$SLOT]}"
  # Platform CUDA_VISIBLE_DEVICES: lane processes inherit it, so they address
  # devices LOGICALLY (0..n-1). Standalone: use physical ids.
  if [[ -n "${PLATFORM_CVD:-}" ]]; then
    LANE_GPU="$SLOT"
  else
    LANE_GPU="$RAW_GPU"
  fi
  # Collect this lane's cells, merging runs of equal delay.
  SEGS=()
  CUR=""
  PREV_D=""
  for ((c = LANE; c < C; c += L)); do
    CELL="${CELLS[$c]}"
    D="${CELL%%:*}"
    T="${CELL##*:}"
    if [[ "$D" == "$PREV_D" ]]; then
      CUR="$CUR $T"
    else
      [[ -n "$CUR" ]] && SEGS+=("$CUR")
      CUR="$D $T"
      PREV_D="$D"
    fi
  done
  [[ -n "$CUR" ]] && SEGS+=("$CUR")
  (( ${#SEGS[@]} >= 1 )) || continue   # more lanes than cells (small smokes)
  TAG="lane${LANE}_gpu${RAW_GPU}"
  SCRIPT="${LOGDIR}/${TAG}_$(date +%Y%m%d_%H%M%S).sh"
  {
    echo "#!/usr/bin/env bash"
    echo "export LEAPBOTCE_CKPT='$CKPT'"
    echo "# auto-generated shard: lane ${LANE} on GPU ${RAW_GPU} (logical ${LANE_GPU})"
    for SEG in "${SEGS[@]}"; do
      echo "bash '$LANE_DRIVER' '$LANE_GPU' $SEG"
    done
  } > "$SCRIPT"
  NSEG=${#SEGS[@]}
  echo "[cluster-sweep] $TAG: gpu=${LANE_GPU} (phys ${RAW_GPU}) ${NSEG} segment(s): ${SEGS[*]}"
  setsid nohup bash "$SCRIPT" > "${LOGDIR}/${TAG}.log" 2>&1 &
  PIDS+=($!)
done

echo "[cluster-sweep] ${#PIDS[@]} lanes launched; waiting..."
RC=0
for P in "${PIDS[@]}"; do
  wait "$P" || RC=1
done

# ---------- summary ----------
echo "[cluster-sweep] lanes finished $(date '+%F %T'); completion per delay:"
for D in "${DELAYS[@]}"; do
  NDONE=$(ls "${SWEEP_ROOT}/delay_${D}"/*.done 2>/dev/null | wc -l)
  echo "  delay_${D}: ${NDONE}/${#TASKS[@]} cells done"
done
if [[ "$RC" -eq 0 ]]; then
  echo "[cluster-sweep] OK"
else
  echo "[cluster-sweep] SOME LANE FAILED — check ${LOGDIR}/ and per-cell logs"
fi
exit "$RC"
