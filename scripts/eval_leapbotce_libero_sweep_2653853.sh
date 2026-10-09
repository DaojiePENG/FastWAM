#!/usr/bin/env bash
# LeapBotCE (ours) LIBERO cloud-delay sweep — company training-cluster driver.
# Counterpart of eval_leapbotce_robotwin_sweep_2653853.sh (RoboTwin) and of the
# local eval_leapbotce_full_sweep_2653853.sh / eval_leapbotce_110k_2653853.sh,
# so the cluster-trained LIBERO checkpoint is collected under the same
# protocol as every other row in docs/baseline_results_libero.md.
#
# Target checkpoint: the cluster run libero_all_dtrain20_seed42_h200
# (full-LIBERO joint training, d_train=20, seed 42, batch 16, max_steps 24,951
# = 10 epochs, finished 2026-10-01; final ckpt step_024951.pt).
#
# Protocol (identical to the local rows): delays D ∈ {0,5,10,20,40}; D>0
# samples the per-episode delay from U{1..D} on the CLOUD stream only
# (cloud mosaic + cloud proprio; edge views stay fresh — the CE observation
# contract); seed=1234; replan=10, wait=30, action_horizon=32,
# num_inference_steps=10, binarize_gripper=true; 4 suites × 10 tasks × 50
# trials = 2000 episodes per delay.
#
# ── Scheduling: PACKED cells (the pattern the earlier small-model LIBERO
# sweeps used: several delay workers share one card) ─────────────────────
# Unit = one (delay × suite) cell = 500 episodes ≈ 6 worker·h, run by ONE
# run_libero_manager.py with its own card set and suite scope; results go
# STRAIGHT to the canonical <delay>/libero_<suite>/ (manager bookkeeping —
# worker_pool/queue/tasks.txt — is namespaced per suite inside the manager),
# so co-resident managers on one card never collide and completion is
# trusted from the manager's exit code (it self-checks completed==total).
# Cells are packed onto cards greedily:
# a card hosts up to PACK co-resident workers (each ~25 GB), so total
# concurrency = N × PACK instead of N. 5×4=20 cells at 16×H200 PACK=4 →
# 60-64 workers, whole sweep in ONE wave ≈ 2.5-3 h (vs ~9 h at 1 worker/card).
# PACK auto = clamp((min_card_vram − 8 GB)/25 GB, 1, 4): H200 141G → 4,
# H800 80G → 2 (4/card does NOT fit 80G for this 25 GB/worker model).
# PACK=1 falls back to the legacy proven mode: one manager, all GPUs,
# serial delays (idempotent, 1 worker/card).
#
# Results: /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/<TAG>/
#   delay_<D>/libero_<suite>/gpu*_task<N>_results.json + delay_sweep_summary.json
# The debug-node report generator (docs/generate_baseline_report_libero.py)
# maps the default TAG to row "LeapBotCE (d_train=20, h200 25k)".
#
# Usage — AIFlow: driven by AIFlow_config_eval_libero.py. Standalone:
#   cd /dahuafs/userdata/2653853/code/FastWAM
#   setsid nohup bash scripts/eval_leapbotce_libero_sweep_2653853.sh <GPUS_CSV> \
#     > /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/driver_libero_h200.log 2>&1 &
#
# Env overrides (all defaulted):
#   LEAPBOTCE_CKPT       ckpt path, or "latest" = newest step_*.pt in RUN_DIR
#   RUN_DIR              training run dir (default: the h200 run)
#   LIBERO_EVAL_TAG      result subdir under EVAL_ROOT (default
#                        dtrain20_seed42_h200). SMOKE MUST SET ITS OWN TAG —
#                        the manager skips existing per-task results, so a
#                        smoke written into the real tag would poison it.
#   LIBERO_EVAL_DELAYS   "0 5 10 20 40" (default)
#   NUM_TRIALS           50 (default)
#   SEED                 1234 (default)
#   LIBERO_EVAL_SUITES   comma list (default all four suites)
#   LIBERO_LIBERO_PACK   co-resident workers/card; 0 (default) = auto by VRAM;
#                        1 = legacy single-manager mode
#   LIBERO_REGEN_REPORT  1 = also regenerate docs/baseline_results_libero.md
#                        (default 0 on cluster nodes: the report writes into
#                        the platform's userPath copy, not the shared tree)
#   FASTWAM_EVAL_PYTHON  override python (env_2653853.sh's leapbot-va conda env)
#
# GPU discovery order (same as the RoboTwin driver):
#   $1 CSV → $CUDA_VISIBLE_DEVICES (platform-assigned; card refs stay
#   LOGICAL 0..n-1) → $GPU_NUM (AIFlow) → all GPUs from nvidia-smi
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

RUN_DIR="${RUN_DIR:-/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/libero_all_dtrain20_seed42_h200}"
EVAL_ROOT="/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce"
LOG_DIR="${EVAL_ROOT}/logs"

CKPT="${LEAPBOTCE_CKPT:-latest}"
TAG="${LIBERO_EVAL_TAG:-dtrain20_seed42_h200}"
read -ra DELAY_ARR <<< "${LIBERO_EVAL_DELAYS:-0 5 10 20 40}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-1234}"
IFS=',' read -ra SUITES <<< "${LIBERO_EVAL_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
REGEN_REPORT="${LIBERO_REGEN_REPORT:-0}"
EXPECTED=$(( ${#SUITES[@]} * 10 ))
# hydra MULTIRUN 列表必须逗号分隔 ("[a,b]"); "${SUITES[*]}" 默认空格 join 会
# 被解析成单个字符串。
SUITES_HYDRA="$(IFS=,; echo "[${SUITES[*]}]")"

echo "[libero-sweep] start $(date '+%F %T') host=$(hostname) cores=$(nproc)"

# ---------- checkpoint resolution ----------
if [[ "$CKPT" == "latest" ]]; then
  CKPT=$(ls -1v "${RUN_DIR}/checkpoints/weights/"step_*.pt 2>/dev/null | tail -1)
fi

# ---------- preflight: fail fast BEFORE occupying any GPU ----------
fail() { echo "[libero-sweep] PREFLIGHT FAIL: $*" >&2; exit 1; }
[[ -n "$CKPT" && -f "$CKPT" ]] || fail "checkpoint not found (set LEAPBOTCE_CKPT; looked at '${CKPT}'; have: $(ls -1v "${RUN_DIR}/checkpoints/weights/" 2>/dev/null | tr '\n' ' '))"
[[ -f "${RUN_DIR}/config.yaml" ]] || fail "run config.yaml missing in $RUN_DIR (eval loads model dims from the run config)"
[[ -f "${RUN_DIR}/dataset_stats.json" ]] || fail "dataset_stats.json missing in $RUN_DIR"
[[ -d "${FASTWAM_ROOT}/.libero" ]] || fail "LIBERO config dir missing: ${FASTWAM_ROOT}/.libero"
[[ -d "/mnt/c2-worldmodel/2653853/leapbot-ce/models" ]] || fail "Wan2.2/T5 backbone dir missing (DIFFSYNTH_MODEL_BASE_PATH)"
PY="${FASTWAM_EVAL_PYTHON:-$(command -v python)}"
[[ -x "$PY" ]] || fail "eval python not found after env_2653853.sh (set FASTWAM_EVAL_PYTHON)"
"$PY" -c 'import torch' 2>/dev/null || fail "eval env lacks torch: $PY"

# Render + benchmark probe (mirrors the RoboTwin SAPIEN probe): a broken
# pod GL stack otherwise burns every episode silently and still exits rc=0.
PROBE_LOG=/tmp/leapbotce_libero_probe.log
if ! "$PY" - >"$PROBE_LOG" 2>&1 <<'PYEOF'
import contextlib
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco/>")
mujoco.Renderer(m, 64, 64)
from libero.libero import benchmark
bench = benchmark.get_benchmark_dict()
with contextlib.redirect_stdout(open("/dev/null", "w")):
    suite = bench["libero_spatial"]()
print("probe OK: libero_spatial tasks =", suite.n_tasks)
PYEOF
then
  echo "[libero-sweep] PROBE FAILED (MUJOCO_GL=${MUJOCO_GL:-unset})" >&2
  echo "[libero-sweep] probe stderr (last 20 lines of $PROBE_LOG):" >&2
  tail -20 "$PROBE_LOG" >&2 2>/dev/null || true
  fail "mujoco/libero probe failed — fix the eval env before resubmitting"
fi
echo "[libero-sweep] env probe OK ($PY)"

# ---------- GPU discovery ----------
declare -a GPUS=()
PLATFORM_CVD=""
if [[ $# -ge 1 && -n "$1" ]]; then
  IFS=',' read -ra GPUS <<< "$1"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPUS <<< "$(tr -d ' ' <<< "${CUDA_VISIBLE_DEVICES}")"
  PLATFORM_CVD=1
elif [[ -n "${GPU_NUM:-}" ]]; then
  GPUS=($(seq 0 $((GPU_NUM - 1))))
else
  GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '))
fi
N=${#GPUS[@]}
[[ "$N" -ge 1 ]] || fail "no GPUs discovered"

# ---------- threads, cells, packing ----------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${LOG_DIR}"
log() { echo "[libero-sweep $(date '+%m-%d %H:%M:%S')] $*"; }
count_results() { find "$1" -name 'gpu*_task*_results.json' 2>/dev/null | wc -l; }
suite_results() { find "$1/$2" -name 'gpu*_task*_results.json' 2>/dev/null | wc -l; }
# dpc 共享盘目录列表有秒级元数据滞后: manager 刚退完就 find 窄路径可能拿到
# 空/短列表（pack smoke 实测: 窄路径 count=0, 1s 后递归 find 已见全部 10 份）。
# 计数 <10 时退避重读（上限 ~30s）; 真缺结果也只是多等半分钟再走报错分支。
count_stable() {
  local d="$1" n t
  n="$(count_results "$d")"
  for t in 1 2 3; do
    (( n >= 10 )) && break
    sleep $(( t * 5 ))
    n="$(count_results "$d")"
  done
  echo "${n}"
}

MANAGER_ARGS=(
  task=libero_all_leapbotce
  ckpt="${CKPT}"
  seed="${SEED}"
  model.load_text_encoder=true
  model.skip_dit_load_from_pretrain=false
  EVALUATION.num_trials="${NUM_TRIALS}"
  EVALUATION.action_horizon=32
  EVALUATION.num_inference_steps=10
  EVALUATION.replan_steps=10
  EVALUATION.binarize_gripper=true
  EVALUATION.num_steps_wait=30
  EVALUATION.max_episode_steps=null
)

PACK="${LIBERO_LIBERO_PACK:-0}"
if (( PACK < 1 )); then
  MIN_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  [[ -n "$MIN_TOTAL" ]] || MIN_TOTAL=81559
  PACK=$(( (MIN_TOTAL - 8000) / 25000 ))
  (( PACK > 4 )) && PACK=4
  (( PACK < 1 )) && PACK=1
fi
log "ckpt: ${CKPT}"
log "tag: ${TAG} | delays: ${DELAY_ARR[*]} | suites: ${SUITES[*]} | trials: ${NUM_TRIALS} | seed: ${SEED}"
log "GPUs: ${GPUS[*]} (n=$N, platform_cvd=${PLATFORM_CVD:-0}) | PACK=${PACK} workers/card"

RC=0

if (( PACK <= 1 )); then
  # ── legacy mode: one manager, all GPUs (1 worker/card), serial delays ──
  if [[ -z "${PLATFORM_CVD}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
  fi
  for delay in "${DELAY_ARR[@]}"; do
    out="${EVAL_ROOT}/${TAG}/delay_${delay}"
    logf="${LOG_DIR}/${TAG}_delay${delay}.log"
    if [[ -f "${out}/.done" ]]; then log "跳过已完成: delay=${delay}"; continue; fi
    mkdir -p "${out}"
    log "开始: delay=${delay}（日志 ${logf}）"
    rc=0
    python experiments/libero/run_libero_manager.py \
      "${MANAGER_ARGS[@]}" \
      EVALUATION.max_delay_steps="${delay}" \
      EVALUATION.output_dir="${out}" \
      "MULTIRUN.task_suite_names=${SUITES_HYDRA}" \
      MULTIRUN.num_gpus="${N}" > "${logf}" 2>&1 || rc=$?
    n="$(count_results "${out}")"
    if (( rc != 0 )) && (( n < EXPECTED )); then
      log "失败（rc=${rc}, 结果 ${n}/${EXPECTED}），120s 后重试一次"
      sleep 120
      python experiments/libero/run_libero_manager.py \
        "${MANAGER_ARGS[@]}" \
        EVALUATION.max_delay_steps="${delay}" \
        EVALUATION.output_dir="${out}" \
        "MULTIRUN.task_suite_names=${SUITES_HYDRA}" \
        MULTIRUN.num_gpus="${N}" > "${logf}.retry" 2>&1 || rc=$?
      n="$(count_results "${out}")"
    fi
    if (( rc == 0 )) || (( n >= EXPECTED )); then
      touch "${out}/.done"; log "完成: delay=${delay}（结果 ${n}/${EXPECTED}）"
    else
      log "错误: delay=${delay} 重试后仍不完整（${n}/${EXPECTED}）"; RC=1
    fi
  done
else
  # ── packed mode: (delay × suite) cells, several managers share a card ──
  CELLS=()
  for D in "${DELAY_ARR[@]}"; do
    for S in "${SUITES[@]}"; do CELLS+=("${D}:${S}"); done
  done
  NC=${#CELLS[@]}
  W=$(( N * PACK ))
  K_BASE=$(( W / NC )); (( K_BASE < 1 )) && K_BASE=1
  (( K_BASE > PACK )) && K_BASE=$PACK
  REM=$(( W - K_BASE * NC )); (( REM < 0 )) && REM=0
  log "packed: ${NC} cells (delay×suite) | workers total ${W} (N×PACK) | per-cell k=${K_BASE}+1(first ${REM}, cap ${PACK})"

  declare -a USED=()
  for ((j = 0; j < N; j++)); do USED[j]=0; done
  declare -A CELL_K CELL_CARDS CELL_SLOTS CELL_TRIES CELL_PID
  PENDING=()
  for i in "${!CELLS[@]}"; do
    cell="${CELLS[$i]}"; D="${cell%%:*}"; S="${cell##*:}"
    if (( $(suite_results "${EVAL_ROOT}/${TAG}/delay_${D}" "${S}") >= 10 )); then
      log "cell ${i} (${CELLS[$i]}) 规范目录已满 10 份, 跳过"
      continue
    fi
    k=$K_BASE; (( i < REM )) && k=$(( k + 1 ))
    (( k > PACK )) && k=$PACK
    (( k > N )) && k=$N
    CELL_K[$i]=$k; CELL_TRIES[$i]=0
    PENDING+=("$i")
  done

  run_cell() {  # $1=cell idx, rest = physical GPU ids (GPUS[j], not slot index)
    local i="$1"; shift
    # ⚠️ 不能写成一行 local cell=... D="${cell%%:*}": 同一条 local 里后续
    # RHS 在任何赋值生效前展开, ${cell} 会取到全局残留值 (pack smoke 首跑
    # 两个 cell 都被解析成最后一个 suite)。必须分条赋值。
    local i_cell="${CELLS[$i]}"
    local D="${i_cell%%:*}"
    local S="${i_cell##*:}"
    local cards_csv k="$#"
    cards_csv="$(IFS=,; echo "$*")"
    # 结果直写规范布局 <delay>/libero_<suite>/: worker 本来就往
    # output_dir/<suite>/ 写结果, manager 簿记 (worker_pool/锁/tasks.txt/
    # manager_config) 已在 manager 侧按 suite 命名空间隔离 —— 同 delay 多
    # manager 共存安全, 免 scratch 免上移。完成与否以 manager rc 为准
    # (退出前自检 completed==total, 不完整抛错; dpc 对 find 深路径列表
    # 有滞后, 盘面 find 只作日志展示不作门槛)。
    local out="${EVAL_ROOT}/${TAG}/delay_${D}"
    mkdir -p "${out}"
    local logf="${LOG_DIR}/${TAG}_delay${D}_${S}.log"
    echo "[cell ${i} d${D} ${S}] k=${k} cards=${cards_csv} attempt $((${CELL_TRIES[$i]} + 1)) $(date '+%T')" >> "$logf"
    CUDA_VISIBLE_DEVICES="$cards_csv" \
    python experiments/libero/run_libero_manager.py \
      "${MANAGER_ARGS[@]}" \
      EVALUATION.max_delay_steps="${D}" \
      EVALUATION.output_dir="${out}" \
      "MULTIRUN.task_suite_names=[${S}]" \
      MULTIRUN.num_gpus="${k}" >> "$logf" 2>&1
  }

  ROUND=0
  while (( ${#PENDING[@]} > 0 )) && (( ROUND < 6 )); do
    ROUND=$(( ROUND + 1 ))
    # ---- launch phase: greedy least-used distinct-card assignment ----
    NEXT=()
    LAUNCHED=()
    for i in "${PENDING[@]}"; do
      k=${CELL_K[$i]}
      PICKS=(); SLOTS=()
      while (( ${#PICKS[@]} < k )); do
        BEST=-1; BESTU=999999
        for ((j = 0; j < N; j++)); do
          [[ " ${SLOTS[*]} " == *" $j "* ]] && continue
          (( USED[j] + 1 > PACK )) && continue
          if (( USED[j] < BESTU )); then BESTU=${USED[j]}; BEST=$j; fi
        done
        # SLOTS 记逻辑槽位（USED 占用记账用）, PICKS 记物理卡号（CVD 用）——
        # standalone 传 "7" 时逻辑 0 ≠ 物理 7, 直接把槽位当 CVD 会跑错卡
        # （首跑 smoke 实锤: 打到了 GPU 0 用户卡上）。
        (( BEST >= 0 )) && { SLOTS+=("$BEST"); PICKS+=("${GPUS[BEST]}"); } || break
      done
      if (( ${#PICKS[@]} == k )); then
        CELL_CARDS[$i]="${PICKS[*]}"
        CELL_SLOTS[$i]="${SLOTS[*]}"
        for j in "${SLOTS[@]}"; do USED[$j]=$(( USED[$j] + 1 )); done
        LAUNCHED+=("$i")
        log "wave${ROUND}: cell ${i} (${CELLS[$i]}) k=${k} -> cards [${CELL_CARDS[$i]}]"
        run_cell "$i" "${PICKS[@]}" &
        CELL_PID[$i]=$!
      else
        NEXT+=("$i")
      fi
    done
    PENDING=("${NEXT[@]}")
    if (( ${#LAUNCHED[@]} == 0 )); then
      fail "packing deadlock: no cell could be placed (N=$N PACK=$PACK) — report bug"
    fi
    # ---- wait phase: reap, retry once in place, free cards ----
    # manager rc=0 即完成铁证: 退出前自检 completed==len(tasks), 不完整抛
    # RuntimeError。盘面 find 仅作日志展示 (dpc 列表可能滞后, 不作门槛)。
    for i in "${LAUNCHED[@]}"; do
      wait "${CELL_PID[$i]}"; rc=$?
      cell="${CELLS[$i]}"; D="${cell%%:*}"; S="${cell##*:}"
      out="${EVAL_ROOT}/${TAG}/delay_${D}"
      CELL_TRIES[$i]=$(( CELL_TRIES[$i] + 1 ))
      if (( rc != 0 )) && (( CELL_TRIES[$i] < 2 )); then
        log "wave${ROUND}: cell ${i} (${CELLS[$i]}) rc=${rc} — 60s 后原地重试 (manager 只补缺失任务)"
        sleep 60
        run_cell "$i" ${CELL_CARDS[$i]} &   # 同一组卡原地重试 (幂等: 只补缺失任务)
        wait "$!"; rc=$?
        CELL_TRIES[$i]=$(( CELL_TRIES[$i] + 1 ))
      fi
      for j in ${CELL_SLOTS[$i]}; do USED[$j]=$(( USED[$j] - 1 )); done
      n="$(count_results "${out}/${S}")"
      if (( rc == 0 )); then
        log "wave${ROUND}: cell ${i} (${CELLS[$i]}) 完成 (rc=0, 盘面 ${n}/10) → ${out}/${S}"
      else
        log "错误: cell ${i} (${CELLS[$i]}) 未完整 (rc=${rc}, 盘面 ${n}/10, 尝试 ${CELL_TRIES[$i]} 次; 日志 ${LOG_DIR}/${TAG}_delay${D}_${S}.log)"
        RC=1
      fi
    done
  done
  (( ROUND < 6 )) || { log "错误: 超过 6 轮仍未排完 (剩 ${#PENDING[@]} 格)"; RC=1; }

  # delay 级 .done: 该 delay 全部 suite 都满 10 份 (count_stable 抗 dpc 列表滞后)
  for D in "${DELAY_ARR[@]}"; do
    out="${EVAL_ROOT}/${TAG}/delay_${D}"
    full=1
    for S in "${SUITES[@]}"; do
      (( $(count_stable "${out}/${S}") >= 10 )) || { full=0; break; }
    done
    (( full )) && [[ ! -f "${out}/.done" ]] && { touch "${out}/.done"; log "delay=${D} 全 suite 完成"; }
  done
fi

# ---------- summarize (writes into the shared EVAL_ROOT) ----------
python scripts/summarize_leapbotce_delays.py "${EVAL_ROOT}/${TAG}" \
  > "${LOG_DIR}/${TAG}_summary.log" 2>&1 \
  && log "汇总完成: ${EVAL_ROOT}/${TAG}/delay_sweep_summary.json" \
  || { log "错误: ${TAG} 汇总失败，见 ${LOG_DIR}/${TAG}_summary.log"; RC=1; }

if [[ "${REGEN_REPORT}" == "1" ]]; then
  python docs/generate_baseline_report_libero.py \
    > "${LOG_DIR}/report_regenerate_${TAG}.log" 2>&1 \
    && log "报告已更新: docs/baseline_results_libero.md" \
    || log "错误: 报告生成失败，见 ${LOG_DIR}/report_regenerate_${TAG}.log"
fi

log "全部结束（tag=${TAG}）— completion per delay (results / ${EXPECTED:-40}):"
for delay in "${DELAY_ARR[@]}"; do
  NDONE="$(count_results "${EVAL_ROOT}/${TAG}/delay_${delay}")"
  echo "  delay_${delay}: ${NDONE} task results"
done
exit "$RC"
