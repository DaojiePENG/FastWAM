#!/usr/bin/env bash
# =============================================================================
# LeapBotCE RoboTwin 训练启动脚本（单机多卡 / 多机多卡集群）
#
# 底层调用链: 本脚本 -> scripts/train_zero1.sh (ZeRO-1 + accelerate) -> scripts/train.py
# 环境自动 source: scripts/env_2653853.sh (conda 环境 / 模型与数据路径)
#
# 一、快速上手
#   cd /dahuafs/userdata/2653853/code/FastWAM
#
#   # 1) 10 步冒烟测试，确认环境和显存没问题（约几分钟）
#   bash scripts/train_leapbotce_robotwin_2653853.sh --task all --gpus 1 --steps 10
#
#   # 2) 只看最终执行的命令而不真正启动（检查用）
#   bash scripts/train_leapbotce_robotwin_2653853.sh --task all --gpus 8 --dry-run
#
# 二、任务
#   all / 完整 task 配置名 robotwin_all_leapbotce
#       RoboTwin 2.0 全量（27,500 条 demo / 607.5 万帧，是 LIBERO 全集的 21.9 倍）
#       + LeapBotCE CloudEdge 时延包装（默认 d_train=20）。
#       官方 RoboTwin 配方为 5 epoch；单 epoch 在有效 batch 128（8xH200）下
#       约 4.75 万步。数据来自 FastWAM 官方发布同源
#       /mnt/c2-worldmodel/training_data/robotwin2.0-fastwam。
#       首次训练前必须先预计算文本嵌入（见 scripts/precompute_text_embeds.py，
#       已预计算过则跳过）。
#
# 三、常用超参（与 scripts/train_leapbotce_libero_2653853.sh 完全同构）
#   --delay N          data.train.max_delay_steps，默认 20（d_train=40 时设 40）
#   --steps N          max_steps 总步数（覆盖 epoch 折算）
#   --epochs N         num_epochs，默认任务配置 5
#   --batch-size N     每 GPU micro-batch，默认任务配置 8
#                      注意 RoboTwin 3cam@384 分辨率比 LIBERO 2cam@224 重约
#                      1.2 倍/样本，80G 卡建议 4-8，141G 卡建议 8-12（smoke 实测后定）
#   --lr X             learning_rate，默认 1e-4（扩 batch 时按 sqrt 缩放）
#   --save-every N     save_every，默认 5000
#   --keep-ckpt N      keep_last_checkpoints，默认 1
#   --resume PATH      断点续训（DeepSpeed 状态目录=完整恢复；step_*.pt=仅权重）
#   --wandb            打开 wandb（默认关闭）
#   --stale-lambda/--stale-warmup/--kv-rank/--kv-dropout 同 LIBERO 脚本
#
# 四、多机多卡（同 LIBERO 脚本，train_zero1.sh 原生支持）
#   每台机器同一条命令，用 NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT 区分身份；
#   输出统一写共享盘 ${FASTWAM_RUNS_ROOT}/robotwin_all_leapbotce/<run_id>/。
#
# 五、输出位置
#   ${FASTWAM_RUNS_ROOT}/robotwin_all_leapbotce/<run_id>/checkpoints/
#   delta 权重 ~2GB/个；DeepSpeed 可恢复状态 ~27GB/个。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TASK="all"
NGPUS=1
DELAY=""
STEPS=""
EPOCHS=""
BATCH_SIZE=""
ACCUM=""
LR=""
SCHEDULER=""
WEIGHT_DECAY=""
GRAD_CLIP=""
SEED=""
WORKERS=""
SAVE_EVERY=""
KEEP_CKPT=""
RUN_NAME=""
RESUME=""
STALE_LAMBDA=""
STALE_WARMUP=""
KV_RANK=""
KV_DROPOUT=""
USE_WANDB=0
DRY_RUN=0

OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)          TASK="${2:?}"; shift 2 ;;
    --gpus)          NGPUS="${2:?}"; shift 2 ;;
    --delay)         DELAY="${2:?}"; shift 2 ;;
    --steps)         STEPS="${2:?}"; shift 2 ;;
    --epochs)        EPOCHS="${2:?}"; shift 2 ;;
    --batch-size)    BATCH_SIZE="${2:?}"; shift 2 ;;
    --accum)         ACCUM="${2:?}"; shift 2 ;;
    --lr)            LR="${2:?}"; shift 2 ;;
    --scheduler)     SCHEDULER="${2:?}"; shift 2 ;;
    --weight-decay)  WEIGHT_DECAY="${2:?}"; shift 2 ;;
    --grad-clip)     GRAD_CLIP="${2:?}"; shift 2 ;;
    --seed)          SEED="${2:?}"; shift 2 ;;
    --workers)       WORKERS="${2:?}"; shift 2 ;;
    --save-every)    SAVE_EVERY="${2:?}"; shift 2 ;;
    --keep-ckpt)     KEEP_CKPT="${2:?}"; shift 2 ;;
    --run-name)      RUN_NAME="${2:?}"; shift 2 ;;
    --resume)        RESUME="${2:?}"; shift 2 ;;
    --stale-lambda)  STALE_LAMBDA="${2:?}"; shift 2 ;;
    --stale-warmup)  STALE_WARMUP="${2:?}"; shift 2 ;;
    --kv-rank)       KV_RANK="${2:?}"; shift 2 ;;
    --kv-dropout)    KV_DROPOUT="${2:?}"; shift 2 ;;
    --wandb)         USE_WANDB=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --)              shift; OVERRIDES=("$@"); break ;;
    -h|--help)       sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（--help 查看用法）" >&2; exit 1 ;;
  esac
done

case "${TASK}" in
  all)        TASK_CFG="robotwin_all_leapbotce" ;;
  *)          TASK_CFG="${TASK}" ;;
esac

HYDRA_ARGS=( "task=${TASK_CFG}" )

[[ -n "${DELAY}"        ]] && HYDRA_ARGS+=("data.train.max_delay_steps=${DELAY}")
[[ -n "${STEPS}"        ]] && HYDRA_ARGS+=("max_steps=${STEPS}")
[[ -n "${EPOCHS}"       ]] && HYDRA_ARGS+=("num_epochs=${EPOCHS}")
[[ -n "${BATCH_SIZE}"   ]] && HYDRA_ARGS+=("batch_size=${BATCH_SIZE}")
[[ -n "${ACCUM}"        ]] && HYDRA_ARGS+=("gradient_accumulation_steps=${ACCUM}")
[[ -n "${LR}"           ]] && HYDRA_ARGS+=("learning_rate=${LR}")
[[ -n "${SCHEDULER}"    ]] && HYDRA_ARGS+=("lr_scheduler_type=${SCHEDULER}")
[[ -n "${WEIGHT_DECAY}" ]] && HYDRA_ARGS+=("weight_decay=${WEIGHT_DECAY}")
[[ -n "${GRAD_CLIP}"    ]] && HYDRA_ARGS+=("max_grad_norm=${GRAD_CLIP}")
[[ -n "${SEED}"         ]] && HYDRA_ARGS+=("seed=${SEED}")
[[ -n "${WORKERS}"      ]] && HYDRA_ARGS+=("num_workers=${WORKERS}")
[[ -n "${SAVE_EVERY}"   ]] && HYDRA_ARGS+=("save_every=${SAVE_EVERY}")
[[ -n "${KEEP_CKPT}"    ]] && HYDRA_ARGS+=("keep_last_checkpoints=${KEEP_CKPT}")
[[ -n "${RESUME}"       ]] && HYDRA_ARGS+=("resume=${RESUME}")

[[ -n "${STALE_LAMBDA}" ]] && HYDRA_ARGS+=("model.cloudedge.stale_loss_lambda_max=${STALE_LAMBDA}")
[[ -n "${STALE_WARMUP}" ]] && HYDRA_ARGS+=("model.cloudedge.stale_loss_warmup_steps=${STALE_WARMUP}")
if [[ -n "${KV_RANK}" || -n "${KV_DROPOUT}" ]]; then
  HYDRA_ARGS+=("model.cloudedge.kv_adapter.enabled=true")
  [[ -n "${KV_RANK}"    ]] && HYDRA_ARGS+=("model.cloudedge.kv_adapter.rank=${KV_RANK}")
  [[ -n "${KV_DROPOUT}" ]] && HYDRA_ARGS+=("model.cloudedge.kv_adapter.dropout=${KV_DROPOUT}")
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  HYDRA_ARGS+=("wandb.enabled=true")
fi

HYDRA_ARGS+=(${OVERRIDES[@]+"${OVERRIDES[@]}"})

# -----------------------------------------------------------------------------
# 环境：conda + 数据/模型路径
# -----------------------------------------------------------------------------
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# 临时代理：wandb 在线同步（与 LIBERO 脚本同一套保护逻辑）
# -----------------------------------------------------------------------------
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-33716}"
export http_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
export https_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export all_proxy="socks5://${PROXY_HOST}:${PROXY_PORT}"
export ALL_PROXY="${all_proxy}"
if [[ "${PROXY_ENABLED:-1}" == "1" ]]; then
  if wget http://www.baidu.com -O /dev/null -q --timeout=5 2>/dev/null; then
    echo "[leapbotce] 代理工作正常 (${PROXY_HOST}:${PROXY_PORT})，wandb 可在线同步"
  else
    echo "[leapbotce] 警告: 代理 ${PROXY_HOST}:${PROXY_PORT} 不可达，wandb 同步可能失败（不影响训练）"
    if [[ "${USE_WANDB:-0}" == "1" && -z "${WANDB_MODE:-}" ]]; then
      export WANDB_MODE=offline
      echo "[leapbotce] 已自动降级 WANDB_MODE=offline（训练不受影响，曲线可稍后 wandb sync 补传）"
    fi
  fi
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  echo "[leapbotce] 代理已按 PROXY_ENABLED=0 关闭"
fi

if [[ -n "${RUN_NAME}" ]]; then
  export RUN_ID="${RUN_NAME}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${NGPUS}" -gt 1 ]]; then
  echo "[提示] 未设置 CUDA_VISIBLE_DEVICES，将使用本机前 ${NGPUS} 张卡。"
  echo "        如需指定卡，例如: CUDA_VISIBLE_DEVICES=4,5,6,7 bash $0 ..."
fi

echo "[leapbotce] task=${TASK_CFG} gpus=${NGPUS} nnodes=${NNODES:-1} node_rank=${NODE_RANK:-0}"
echo "[leapbotce] overrides: ${HYDRA_ARGS[*]}"
echo "[leapbotce] 输出目录: ${FASTWAM_RUNS_ROOT:-./runs}/${TASK_CFG}/${RUN_ID:-<时间戳>}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] 将执行: bash scripts/train_zero1.sh ${NGPUS} ${HYDRA_ARGS[*]}"
  exit 0
fi

exec bash scripts/train_zero1.sh "${NGPUS}" "${HYDRA_ARGS[@]}"
