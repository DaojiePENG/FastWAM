#!/usr/bin/env bash
# =============================================================================
# LeapBotCE d20 加长轮（200k 计划）中途检查点评测驱动
#
# 背景：200k 步加长训练于 2026-09-29 23:56 因 NCCL watchdog SIGABRT 崩溃
# （rank 0），崩溃前最后落盘检查点为 step_110000（09-29 23:23，约 5.5 epoch，
# stale warmup=100k 已过、stale_loss_weight≈0.33、cosine LR≈6.4e-5）。
# 本脚本按用户要求评测该检查点 D ∈ {0,10,20,40}（不做 D=5），协议与
# baseline 完全对齐（seed=1234、replan=10、wait=30、horizon=32、
# inference=10、4 套件 × 10 任务 × 50 trials），结果目录：
#   /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/dtrain20_seed42_110k/
# 结束后自动汇总并重生成 docs/baseline_results_libero.md（与 50k 模型及
# 其它 baseline 并列对比）。
#
# 用法（setsid 脱离会话）：
#   cd /dahuafs/userdata/2653853/code/FastWAM
#   nice -n 10 setsid nohup env GPUS=0,1,2,3 NUM_GPUS=4 \
#     bash scripts/eval_leapbotce_110k_2653853.sh \
#     > /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/driver_110k.log 2>&1 &
#
# 可用环境变量：
#   CKPT       评测检查点（默认 step_110000.pt）
#   TAG        结果子目录名（默认 dtrain20_seed42_110k；续训到 200k 后
#              重新评测时设 TAG=dtrain20_seed42_200k 即可换行展示）
#   DELAYS     时延上限列表（默认 "0 10 20 40"）
#   GPUS       物理卡列表（默认 0,1,2,3）
#   NUM_GPUS   worker 数（默认 4；manager 每卡 1 worker，一 worker ~25GB）
#   NUM_TRIALS 每任务 trials（默认 50）
#   SEED       时延/环境种子（默认 1234，与 baseline 对齐）
# 幂等：run_libero_manager.py 建队列时自动跳过已有结果的任务；每个 delay
# 完成后写 .done 标记，重启本脚本只补缺失部分。
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

RUN_ROOT="/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42_200k/checkpoints/weights"
EVAL_ROOT="/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce"
LOG_DIR="${EVAL_ROOT}/logs"

CKPT="${CKPT:-${RUN_ROOT}/step_110000.pt}"
TAG="${TAG:-dtrain20_seed42_110k}"
DELAYS="${DELAYS:-0 10 20 40}"
GPUS="${GPUS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-1234}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
# manager 从 CUDA_VISIBLE_DEVICES 取前 NUM_GPUS 张卡给 worker
export CUDA_VISIBLE_DEVICES="${GPUS}"

mkdir -p "${LOG_DIR}"
log() { echo "[110k $(date '+%m-%d %H:%M:%S')] $*"; }

count_results() { find "$1" -name 'gpu*_task*_results.json' 2>/dev/null | wc -l; }

run_manager() {  # $1=out_dir  $2=delay  $3=log_file
  local out="$1" delay="$2" logf="$3"
  python experiments/libero/run_libero_manager.py \
    task=libero_all_leapbotce \
    ckpt="${CKPT}" \
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

for delay in ${DELAYS}; do
  out="${EVAL_ROOT}/${TAG}/delay_${delay}"
  logf="${LOG_DIR}/${TAG}_delay${delay}.log"
  if [[ -f "${out}/.done" ]]; then
    log "跳过已完成: delay=${delay}"
    continue
  fi
  mkdir -p "${out}"
  log "开始: delay=${delay}（日志 ${logf}）"
  rc=0
  run_manager "${out}" "${delay}" "${logf}" || rc=$?
  n="$(count_results "${out}")"
  if (( rc != 0 )) && (( n < 40 )); then
    log "失败（rc=${rc}, 结果 ${n}/40），120s 后重试一次: delay=${delay}"
    sleep 120
    run_manager "${out}" "${delay}" "${logf}.retry" || rc=$?
    n="$(count_results "${out}")"
  fi
  if (( rc == 0 )) || (( n >= 40 )); then
    touch "${out}/.done"
    log "完成: delay=${delay}（结果 ${n}/40）"
  else
    log "错误: delay=${delay} 重试后仍不完整（${n}/40），继续下一个 delay"
  fi
done

python scripts/summarize_leapbotce_delays.py "${EVAL_ROOT}/${TAG}" \
  > "${LOG_DIR}/${TAG}_summary.log" 2>&1 \
  && log "汇总完成: ${EVAL_ROOT}/${TAG}/delay_sweep_summary.json" \
  || log "错误: 汇总失败，见 ${LOG_DIR}/${TAG}_summary.log"

log "重新生成 baseline 汇总报告..."
python docs/generate_baseline_report_libero.py \
  > "${LOG_DIR}/report_regenerate_110k.log" 2>&1 \
  && log "报告已更新: docs/baseline_results_libero.md" \
  || log "错误: 报告生成失败，见 ${LOG_DIR}/report_regenerate_110k.log"

log "110k 检查点评测全部结束"
