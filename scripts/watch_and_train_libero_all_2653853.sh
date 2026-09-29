#!/usr/bin/env bash
# =============================================================================
# 看守启动脚本：等待服务器出现 N 张空闲 GPU，然后自动启动
# LeapBotCE 全 LIBERO（spatial+object+goal+10 四套件）CE 训练。
#
# 配套正式启动脚本: scripts/train_leapbotce_libero_2653853.sh --task all
#
# 默认计划（2026-09-22 与作者确认）：d=20 和 d=40 各 50000 步（约 2.9 epoch），
# 串行执行。每轮流程（全部幂等）：
#   1) 轮询 nvidia-smi，直到出现 NEEDED_GPUS 张"空闲"卡
#      （空闲定义：显存占用 < FREE_MEM_MB 且无任何计算进程，避开他人任务）
#   2) 预计算全 LIBERO 40 条指令的 T5 文本嵌入（已缓存自动跳过，秒级）
#   3) 显存标定：用 3 步冒烟按当前 DELAY 窗口依次试探每卡 batch {8, 4, 2}，
#      取能装下（峰值 + 6GB 余量 < 卡总显存）的最大值；梯度累积按
#      EFFECTIVE_BATCH/(卡数*batch) 反算，有效 batch 恒为 16（与官方一致）
#   4) setsid 脱离会话启动该轮训练，轮询等待其正常结束（或失败终止整个计划）
#   一轮正常结束后自动进入下一轮（重新等卡、重新标定）。
#
# 用法（务必用 setsid/nohup 脱离终端，防止会话退出杀掉看守）：
#   cd /dahuafs/userdata/2653853/code/FastWAM
#   setsid nohup bash scripts/watch_and_train_libero_all_2653853.sh \
#     > /mnt/c2-worldmodel/2653853/leapbot-ce/runs/watch_launch.log 2>&1 &
#
# 可用环境变量（都有默认值）：
#   DELAYS="20 40"         串行训练的时延上限列表（P0: 先 20 后 40）
#   STEPS=50000            每轮 max_steps（约 2.9 epoch / 3.5 天 / 轮 @2卡）
#   NEEDED_GPUS=2          需要几张空闲卡
#   FREE_MEM_MB=2048       显卡显存占用低于该值(MiB)且无进程才算空闲
#   POLL_INTERVAL=300      轮询间隔秒
#   EFFECTIVE_BATCH=16     有效 batch（官方配方 16，勿随意改）
#   BATCH_CANDIDATES="8 4 2"  每卡 batch 试探列表（从大到小）
#   SEED=42                随机种子
#   OMP_NUM_THREADS=8      限制线程数，避免 pthread_create EAGAIN
#
# wandb：正式训练自动带 --wandb 在线上报（project=leapbotce，每次运行
# wandb.name=libero_all_leapbotce_dtrainXX_seed42 以区分），凭据读 ~/.netrc；
# 训练启动脚本内置临时代理 127.0.0.1:${PROXY_PORT:-33716}
# （代理断了只影响 wandb 上报，不影响训练）。
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${FASTWAM_RUNS_ROOT:-/mnt/c2-worldmodel/2653853/leapbot-ce/runs}"
LOG_DIR="${RUNS_ROOT}/libero_all_leapbotce"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DELAYS="${DELAYS:-20 40}"
STEPS="${STEPS:-50000}"
NEEDED_GPUS="${NEEDED_GPUS:-2}"
FREE_MEM_MB="${FREE_MEM_MB:-2048}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-16}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-8 4 2}"
SEED="${SEED:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

log() { echo "[watch $(date '+%m-%d %H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# 环境准备（conda + 路径），供预计算/冒烟等前台步骤使用
# -----------------------------------------------------------------------------
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# 等待空闲 GPU：输出选中的 GPU 索引串（逗号分隔，如 "2,5"）
# 空闲 = 显存占用 < FREE_MEM_MB 且该卡上没有任何计算进程（避开他人任务）
# -----------------------------------------------------------------------------
find_free_gpus() {
  python3 - "${FREE_MEM_MB}" <<'PY'
import subprocess, sys, collections
free_mb = int(sys.argv[1])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used",
     "--format=csv,noheader,nounits"], text=True)
used = {}
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    used[int(idx)] = int(mem)
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
    text=True).strip().splitlines()
busy_uuids = collections.Counter(a.strip() for a in apps if a.strip() and a.strip() != "[N/A]")
idx_uuid = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
has_proc = set()
for line in idx_uuid.strip().splitlines():
    idx, uuid_ = [x.strip() for x in line.split(", ")]
    if busy_uuids.get(uuid_, 0) > 0:
        has_proc.add(int(idx))
free = [i for i, m in sorted(used.items())
        if m < free_mb and i not in has_proc]
print(",".join(str(i) for i in free))
PY
}

wait_for_free_gpus() {
  log "等待 ${NEEDED_GPUS} 张空闲卡（占用<${FREE_MEM_MB}MiB 且无进程），每 ${POLL_INTERVAL}s 检查一次..."
  while :; do
    local free_list n_free
    free_list="$(find_free_gpus)"
    if [[ -n "${free_list}" ]]; then
      n_free="$(awk -F',' '{print NF}' <<<"${free_list}")"
    else
      n_free=0
    fi
    if (( n_free >= NEEDED_GPUS )); then
      WAIT_GPUS="$(cut -d, -f1-"${NEEDED_GPUS}" <<<"${free_list}")"
      log "选中空闲卡: ${WAIT_GPUS}（当前空闲: ${free_list}）"
      return 0
    fi
    log "空闲卡不足（现有: ${free_list:-无}），${POLL_INTERVAL}s 后重试..."
    sleep "${POLL_INTERVAL}"
  done
}

# -----------------------------------------------------------------------------
# 预计算全 LIBERO 文本嵌入（幂等；已缓存的 prompt 自动跳过）
# -----------------------------------------------------------------------------
precompute_embeds() {
  local gpu="$1" plog="${LOG_DIR}/precompute_${STAMP}.log"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  log "预计算文本嵌入（task=libero_all_leapbotce, GPU ${gpu}）..."
  if python scripts/precompute_text_embeds.py task=libero_all_leapbotce \
       > "${plog}" 2>&1; then
    log "文本嵌入就绪"
  else
    log "错误：文本嵌入预计算失败，见 ${plog}"
    return 1
  fi
}

# -----------------------------------------------------------------------------
# 显存标定：从大到小试探每卡 batch，成功标准 = 3 步冒烟无 OOM
# 且（峰值显存 + 6GB 余量）< 卡总显存。输出 CHOSEN_BS / CHOSEN_ACCUM。
# -----------------------------------------------------------------------------
HEADROOM_MB=6144
probe() {  # $1=batch_size  $2=gpu_index  $3=delay  $4=peak结果变量名
  local bs="$1" gpu="$2" delay="$3" peak_var="$4"
  local plog="${LOG_DIR}/probe_d${delay}_bs${bs}_${STAMP}.log"
  local base end_peak
  base="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    timeout 3600 bash scripts/train_leapbotce_libero_2653853.sh \
      --task all-smoke --gpus 1 --batch-size "${bs}" --steps 3 \
      --save-every 3 --run-name "probe_d${delay}_bs${bs}_${STAMP}" \
      --delay "${delay}" > "${plog}" 2>&1
  ) &
  local pid=$!
  end_peak=0
  while kill -0 "${pid}" 2>/dev/null; do
    local u
    u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null || echo 0)"
    (( u > end_peak )) && end_peak="${u}"
    sleep 5
  done
  wait "${pid}" || { printf -v "${peak_var}" '%s' "999999"; return 1; }
  printf -v "${peak_var}" '%s' "$(( end_peak > base ? end_peak - base : 0 ))"
  return 0
}

calibrate_batch() {  # $1=gpu  $2=delay
  local gpu="$1" delay="$2" bs accum PEAK
  local total_mem
  total_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu}" | head -1)"
  CHOSEN_BS=""
  for bs in ${BATCH_CANDIDATES}; do
    accum=$(( EFFECTIVE_BATCH / (NEEDED_GPUS * bs) ))
    if (( accum < 1 || EFFECTIVE_BATCH % (NEEDED_GPUS * bs) != 0 )); then
      log "跳过 batch=${bs}：无法凑出有效 batch ${EFFECTIVE_BATCH}（${NEEDED_GPUS}卡）"
      continue
    fi
    log "标定 batch=${bs}（3 步冒烟，delay=${delay}，GPU ${gpu}）..."
    PEAK=0
    if probe "${bs}" "${gpu}" "${delay}" PEAK && \
       (( PEAK + HEADROOM_MB <= total_mem )); then
      log "batch=${bs} 通过：峰值 ${PEAK}MiB，余量 $(( total_mem - PEAK ))MiB"
      CHOSEN_BS="${bs}"
      break
    else
      log "batch=${bs} 不通过（峰值 ${PEAK}MiB / 总量 ${total_mem}MiB 或运行失败），试下一个"
    fi
  done
  [[ -z "${CHOSEN_BS}" ]] && { log "错误：所有 batch 候选都装不下"; return 1; }
  CHOSEN_ACCUM=$(( EFFECTIVE_BATCH / (NEEDED_GPUS * CHOSEN_BS) ))
  log "标定结果：每卡 batch=${CHOSEN_BS} × 累积 ${CHOSEN_ACCUM} × ${NEEDED_GPUS} 卡 = 有效 batch ${EFFECTIVE_BATCH}"
  # 清理探针训练产物：3 步冒烟的 checkpoint 状态每份约 20-40GB，纯占盘，
  # 无保留价值；只保留探针日志（.log）用于复查。
  rm -rf "${RUNS_ROOT}"/libero_all_leapbotce_smoke/probe_d"${delay}"_bs*_${STAMP} 2>/dev/null
  return 0
}

# -----------------------------------------------------------------------------
# 启动一轮训练并等待其结束；训练正常完成返回 0，异常退出返回 1
# -----------------------------------------------------------------------------
launch_and_wait() {  # $1=gpus  $2=delay
  local gpus="$1" delay="$2"
  local run_name="dtrain${delay}_seed${SEED}"
  local tlog="${LOG_DIR}/train_${run_name}_${STAMP}.log"

  log "启动训练: GPUs=${gpus} bs=${CHOSEN_BS} accum=${CHOSEN_ACCUM} delay=${delay} seed=${SEED} steps=${STEPS}"
  log "日志: ${tlog}"
  setsid nohup env CUDA_VISIBLE_DEVICES="${gpus}" \
    bash scripts/train_leapbotce_libero_2653853.sh \
    --task all --gpus "${NEEDED_GPUS}" \
    --batch-size "${CHOSEN_BS}" --accum "${CHOSEN_ACCUM}" \
    --delay "${delay}" --seed "${SEED}" --steps "${STEPS}" --wandb \
    --run-name "${run_name}" \
    -- "wandb.name=libero_all_leapbotce_${run_name}" \
    > "${tlog}" 2>&1 &
  local tpid=$!

  # 轮询等待训练进程退出（每 10 分钟检查一次并汇报进度）
  while kill -0 "${tpid}" 2>/dev/null; do
    sleep 600
    if kill -0 "${tpid}" 2>/dev/null; then
      local last_step
      last_step="$(grep -o 'step=[0-9]*/' "${tlog}" 2>/dev/null | tail -1)"
      log "训练中 (${run_name}): ${last_step:-模型加载/数据准备中}"
    fi
  done
  wait "${tpid}" 2>/dev/null
  local rc=$?

  if grep -q "max_steps reached" "${tlog}"; then
    log "训练正常完成: ${run_name}"
    grep -A2 "max_steps reached" "${tlog}" | tail -3
    return 0
  fi
  log "错误: 训练异常退出 (rc=${rc})，见 ${tlog} 末尾："
  tail -5 "${tlog}"
  return 1
}

# =============================================================================
# 主流程：逐个 delay 串行执行
# =============================================================================
PLAN_OK=1
for DELAY in ${DELAYS}; do
  log "===== 第 ${DELAY} 轮（delay=${DELAY}, steps=${STEPS}）====="
  wait_for_free_gpus || { PLAN_OK=0; break; }
  GPUS="${WAIT_GPUS}"
  IFS=',' read -r -a GPU_ARR <<<"${GPUS}"

  # 第一轮在等卡时预计算已包含在上面的尝试里；这里再确保一次（幂等）
  precompute_embeds "${GPU_ARR[0]}" || { PLAN_OK=0; break; }

  calibrate_batch "${GPU_ARR[0]}" "${DELAY}" || { PLAN_OK=0; break; }
  launch_and_wait "${GPUS}" "${DELAY}" || { PLAN_OK=0; break; }
done

if (( PLAN_OK )); then
  log "全部计划完成（delays: ${DELAYS}），看守脚本退出。"
else
  log "计划中止（见上方日志），看守脚本退出。"
fi
