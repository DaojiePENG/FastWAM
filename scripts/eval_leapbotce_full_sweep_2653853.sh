#!/usr/bin/env bash
# =============================================================================
# LeapBotCE 全量时延鲁棒性评测驱动（结果与其它 baseline 汇总到同一份报告）
#
# 评测对象：本仓库训练的两个全 LIBERO（四套件联合）LeapBotCE 检查点
#   - dtrain20_seed42：d_train=20 训练 50k 步
#   - dtrain40_seed42：d_train=40 训练 50k 步
# 评测协议（与 docs/baseline_results_libero.md 中各 baseline 完全对齐）：
#   - 时延上限 D ∈ {0, 5, 10, 20, 40}；D>0 时每个 episode 实际时延从
#     U{1..D} 采样（时延只作用于云端流：云马赛克图 + 云本体感知；
#     边缘视图/边缘本体感知始终新鲜 —— 这是 CE 架构的观测契约）
#   - seed=1234（与 baseline 的 delay_seed=1234 约定一致）
#   - 4 套件 × 10 任务 × 50 trials = 500 episodes / (模型, D, 套件)
#   - replan=10、wait=30、action_horizon=32、num_inference_steps=10、
#     binarize_gripper=true（均为默认值，这里显式写出以便复查）
#   - 每个 episode 的时延 RNG = RandomState(seed + episode_idx)
#
# 目录布局（与报告生成器 docs/generate_baseline_report_libero.py 的约定一致）：
#   ${EVAL_ROOT}/dtrain20_seed42/delay_<D>/libero_<suite>/gpu*_task<N>_results.json
#   ${EVAL_ROOT}/dtrain20_seed42/delay_sweep_summary.json   <- 汇总脚本产出
#   （dtrain40_seed42 同理）
#
# 执行方式（2026-09-28 起 GPU 0-3 被 d20 200k 步长训练占用，评测用 4-7 卡）：
# 两个 sweep 并行跑在 GPU 4-7 上（各 MULTIRUN.num_gpus=4）→ 每卡 2 个评测
# 进程共存（实测双 worker 峰值约 40GB / 80GB 卡，eval 为 CPU/仿真瓶颈）。
# 第二个 sweep 错峰 STAGGER_S 秒启动。预计总时长 ~30 小时
# （约 240 worker·小时 / 8 workers）。
#
# 幂等/断点续跑：每个 (模型, D) 完成后写 <out>/.done 标记；重启本脚本会
# 跳过已完成的 delay。manager 进程内部自带崩溃 worker 隔离与任务重排队。
# 失败的 delay 自动重试一次。
#
# 用法（务必 setsid/nohup 脱离会话，防会话退出杀掉评测）：
#   cd /dahuafs/userdata/2653853/code/FastWAM
#   setsid nohup bash scripts/eval_leapbotce_full_sweep_2653853.sh \
#     > /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/driver.log 2>&1 &
#
# 可用环境变量（都有默认值）：
#   DELAYS="0 20 40 10 5"   时延上限列表（0 和训练匹配的 20/40 优先出数）
#   NUM_TRIALS=50           每任务 trials（统计规模与 baseline 一致）
#   SEED=1234               时延/环境种子（对齐 baseline）
#   GPUS="4,5,6,7"          评测使用的物理 GPU 列表
#   NUM_GPUS=4              每个 sweep 的 worker 数（<= GPUS 数量）
#   STAGGER_S=180           第二个 sweep 的错峰秒数
#   SKIP_D40=0              置 1 可只跑 dtrain20（调试用）
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

EVAL_ROOT="/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce"
LOG_DIR="${EVAL_ROOT}/logs"
RUN_ROOT="/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce"
CKPT_D20="${RUN_ROOT}/dtrain20_seed42/checkpoints/weights/step_050000.pt"
CKPT_D40="${RUN_ROOT}/dtrain40_seed42/checkpoints/weights/step_050000.pt"

DELAYS="${DELAYS:-0 20 40 10 5}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-1234}"
GPUS="${GPUS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
STAGGER_S="${STAGGER_S:-180}"
SKIP_D40="${SKIP_D40:-0}"

# 两个 sweep 共用同一批物理卡：manager 会从 CUDA_VISIBLE_DEVICES 取前
# NUM_GPUS 张给 worker（每卡落 2 个 worker，两 sweep 各一）
export CUDA_VISIBLE_DEVICES="${GPUS}"

# 评测是 CPU/仿真密集型；限制线程数防止 pthread_create EAGAIN（每卡会共存
# 2 个 worker，8 卡共 16 个 worker 进程）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${LOG_DIR}"
log() { echo "[sweep $(date '+%m-%d %H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# 检查一个 delay 输出目录的结果文件数（完整 = 4 套件 × 10 任务 = 40 份）
# -----------------------------------------------------------------------------
count_results() {
  find "$1" -name 'gpu*_task*_results.json' 2>/dev/null | wc -l
}

# -----------------------------------------------------------------------------
# 跑单个 (模型, delay)：run_libero_manager.py 统一调度 4 套件 × 10 任务
# 到 NUM_GPUS 个 worker（每 worker 常驻加载一次模型，循环领任务）
# -----------------------------------------------------------------------------
run_manager() {  # $1=ckpt  $2=out_dir  $3=delay  $4=log_file
  local ckpt="$1" out="$2" delay="$3" logf="$4"
  python experiments/libero/run_libero_manager.py \
    task=libero_all_leapbotce \
    ckpt="${ckpt}" \
    seed="${SEED}" \
    model.load_text_encoder=true \
    model.skip_dit_load_from_pretrain=false \
    EVALUATION.num_trials="${NUM_TRIALS}" \
    EVALUATION.max_delay_steps="${delay}" \
    EVALUATION.action_horizon=32 \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=10 \
    EVALUATION.binarize_gripper=true \
    EVALUATION.num_steps_wait=30 \
    EVALUATION.max_episode_steps=null \
    EVALUATION.output_dir="${out}" \
    "MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]" \
    MULTIRUN.num_gpus="${NUM_GPUS}" \
    > "${logf}" 2>&1
}

# -----------------------------------------------------------------------------
# 一个模型的完整 delay sweep：串行跑 DELAYS 列表，最后汇总出
# delay_sweep_summary.json（供报告生成器读取）
# -----------------------------------------------------------------------------
run_sweep() {  # $1=tag(dtrain20/dtrain40)  $2=ckpt
  local tag="$1" ckpt="$2" delay out logf rc n
  for delay in ${DELAYS}; do
    out="${EVAL_ROOT}/${tag}/delay_${delay}"
    logf="${LOG_DIR}/${tag}_delay${delay}.log"
    if [[ -f "${out}/.done" ]]; then
      log "跳过已完成: ${tag} delay=${delay}"
      continue
    fi
    mkdir -p "${out}"
    log "开始: ${tag} delay=${delay}（日志 ${logf}）"
    rc=0
    run_manager "${ckpt}" "${out}" "${delay}" "${logf}" || rc=$?
    n="$(count_results "${out}")"
    if (( rc != 0 )) && (( n < 40 )); then
      log "失败（rc=${rc}, 结果 ${n}/40），60s 后重试一次: ${tag} delay=${delay}"
      sleep 60
      run_manager "${ckpt}" "${out}" "${delay}" "${logf}.retry" || rc=$?
      n="$(count_results "${out}")"
    fi
    if (( rc == 0 )) || (( n >= 40 )); then
      touch "${out}/.done"
      log "完成: ${tag} delay=${delay}（结果 ${n}/40）"
    else
      log "错误: ${tag} delay=${delay} 重试后仍不完整（${n}/40），继续下一个 delay"
    fi
  done
  # 汇总该模型全部 delay 的结果 -> delay_sweep_summary.json + delay_sweep_tasks.csv
  python scripts/summarize_leapbotce_delays.py "${EVAL_ROOT}/${tag}" \
    > "${LOG_DIR}/${tag}_summary.log" 2>&1 \
    && log "汇总完成: ${EVAL_ROOT}/${tag}/delay_sweep_summary.json" \
    || log "错误: ${tag} 汇总失败，见 ${LOG_DIR}/${tag}_summary.log"
}

# =============================================================================
# 主流程：两个 sweep 并行（错峰启动），等全部结束后重新生成汇总报告
# =============================================================================
for c in "${CKPT_D20}" "${CKPT_D40}"; do
  [[ -f "${c}" ]] || { log "错误: 检查点不存在 ${c}"; exit 1; }
done

log "LeapBotCE 全量时延评测启动: delays=[${DELAYS}] trials=${NUM_TRIALS} seed=${SEED} gpus/sweep=${NUM_GPUS}"
log "输出根目录: ${EVAL_ROOT}"

run_sweep dtrain20 "${CKPT_D20}" &
PID_D20=$!

if [[ "${SKIP_D40}" != "1" ]]; then
  sleep "${STAGGER_S}"
  run_sweep dtrain40 "${CKPT_D40}" &
  PID_D40=$!
else
  PID_D40=""
  log "SKIP_D40=1，跳过 dtrain40 sweep"
fi

RC20=0; wait "${PID_D20}" || RC20=$?
RC40=0; [[ -n "${PID_D40}" ]] && { wait "${PID_D40}" || RC40=$?; }

# 两个 sweep 都结束后重新生成 baseline 汇总报告（LeapBotCE 行并入其中）
log "重新生成 baseline 汇总报告..."
python docs/generate_baseline_report_libero.py \
  > "${LOG_DIR}/report_regenerate.log" 2>&1 \
  && log "报告已更新: docs/baseline_results_libero.md" \
  || log "错误: 报告生成失败，见 ${LOG_DIR}/report_regenerate.log"

log "全部结束（dtrain20 rc=${RC20}, dtrain40 rc=${RC40}）"
exit $(( RC20 + RC40 ))
