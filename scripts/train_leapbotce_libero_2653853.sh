#!/usr/bin/env bash
# =============================================================================
# LeapBotCE LIBERO 训练启动脚本（单机多卡 / 多机多卡集群）
#
# 底层调用链: 本脚本 -> scripts/train_zero1.sh (ZeRO-1 + accelerate) -> scripts/train.py
# 环境自动 source: scripts/env_2653853.sh (conda 环境 / 模型与数据路径 / LIBERO 配置)
#
# 一、快速上手
#   cd /dahuafs/userdata/2653853/code/FastWAM
#
#   # 1) 先跑 10 步冒烟测试，确认环境和显存没问题（约几分钟）
#   bash scripts/train_leapbotce_libero_2653853.sh --task smoke --gpus 1
#
#   # 2) 正式训练（推荐）：全 LIBERO 四套件联合训练，单机 2 卡
#   #    数据 = spatial + object + goal + 10（与 FastWAM 官方 libero 检查点同源）
#   bash scripts/train_leapbotce_libero_2653853.sh --task all --gpus 2
#
#   # 3) 仅 LIBERO-Spatial 的基线配方，单机 8 卡
#   bash scripts/train_leapbotce_libero_2653853.sh --task spatial --gpus 8
#
#   # 4) 训练类型改为"云端 KV 适配器"配方，时延上限改为 40
#   bash scripts/train_leapbotce_libero_2653853.sh --task kv --gpus 8 --delay 40
#
#   # 5) 只想看看最终执行的命令而不真正启动（检查用）
#   bash scripts/train_leapbotce_libero_2653853.sh --task all --gpus 2 --dry-run
#
# 二、训练类型（--task 取值）
#   all        task=libero_all_leapbotce
#              全 LIBERO 联合训练（spatial/object/goal/10 四套件，~27.8 万帧，
#              有效 batch 16 时约 1.74 万步/epoch）。数据与 FastWAM 官方
#              libero_uncond_2cam224_1e-4 检查点同源，套上 CloudEdge 时延包装。
#              这是论文 P0 主实验的配置。
#   all-smoke  task=libero_all_leapbotce_smoke
#              全 LIBERO 数据上的 10 步冒烟，验证新数据管线（首次使用建议先跑）。
#   spatial    task=libero_spatial_leapbotce
#              LeapBotCE 基线：云端 Wan 冻结，直接把 MoT 视频 KV cache 作为
#              规划表征传给边缘；可训练参数 = ActionDiT + SigLIP projector
#              + proprio projector。数据仅 LIBERO-Spatial（10 任务）。
#   kv         task=libero_spatial_leapbotce_kv
#              在 spatial 基线上额外启用云端 KV 适配器（零初始化低秩残差，
#              起点等价于基线），是"云端表征适配"的推荐第一步。
#   smoke      task=libero_spatial_leapbotce_smoke
#              10 个 optimizer step 的集成冒烟测试，用于验证环境，不产生结果。
#   也可以直接写完整 Hydra task 配置名，如 --task libero_all_leapbotce
#   兼容别名: baseline = spatial。
#
# 三、超参数分两类
#   (A) LeapBotCE(CE) 专属 —— 控制"时延场景"和"云端-边缘接口"：
#     --delay N          data.train.max_delay_steps
#                        训练时每次采样 cloud 观测的最大时延步数（1..N 均匀采样，
#                        历史不足时回退当前帧）。默认 20；做 d_train=40 的 P0 实验
#                        时设 40。数据集会自动扩窗，无需手动改 num_frames。
#     --stale-lambda X   model.cloudedge.stale_loss_lambda_max
#                        陈旧(stale)预测损失权重上限。默认 0.5。
#     --stale-warmup N   model.cloudedge.stale_loss_warmup_steps
#                        stale 损失从 0 线性升到上限的步数。
#                        默认 -1 = 自动 = max_steps/2（随 --steps 缩放，推荐）。
#     --kv-rank N        model.cloudedge.kv_adapter.rank（仅 --task kv 时有意义）
#                        KV 适配器的低秩维度。默认 64。
#     --kv-dropout X     model.cloudedge.kv_adapter.dropout。默认 0.0。
#
#   (B) FastWAM 原有通用超参 —— 控制普通训练过程：
#     --steps N          max_steps   总 optimizer 步数。默认 10000（官方配方）。
#     --epochs N         num_epochs  与 max_steps 二选一生效（先到为准）。
#     --batch-size N     batch_size  每 GPU micro-batch。默认=任务配置
#                        （spatial: 2，all: 8）。
#     --accum N          gradient_accumulation_steps 梯度累积。默认=任务配置
#                        （spatial: 4，all: 1）。
#                        有效 batch = gpus × batch_size × accum
#                        （两个配方在 2 卡时均为 16，与官方一致）。
#     --lr X             learning_rate 学习率。默认 1e-4。
#     --scheduler NAME   lr_scheduler_type，默认 cosine。
#     --weight-decay X   weight_decay，默认 1e-2。
#     --grad-clip X      max_grad_norm，默认 1.0。
#     --seed N           seed，默认 42。多 seed 实验时改这里。
#     --workers N        num_workers dataloader 进程数，默认 4。
#     --save-every N     save_every 每 N 步存一次 checkpoint，默认 500。
#     --keep-ckpt N      keep_last_checkpoints 保留最近 N 个权重，
#                        0=全部保留（注意 DeepSpeed 可恢复状态很大，磁盘紧张时
#                        保持 1）。默认 1。
#     --resume PATH      resume 断点续训：给 DeepSpeed 状态目录=完整恢复；
#                        给 step_XXXXXX.pt=只加载权重。
#     --wandb            打开 wandb（默认关闭，只写本地日志）。
#
# 四、多卡 / 多机集群
#   单机多卡:  --gpus N 即可；想指定具体哪几张卡，用环境变量，例如：
#              CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_leapbotce_2653853.sh \
#                --task baseline --gpus 4
#
#   多机多卡:  每台机器都跑同一条命令，通过环境变量区分身份
#              （scripts/train_zero1.sh 原生支持）：
#                主节点( rank 0 ):
#                  NNODES=2 NODE_RANK=0 \
#                  MASTER_ADDR=<主节点IP> MASTER_PORT=29500 \
#                  bash scripts/train_leapbotce_2653853.sh --task baseline --gpus 8
#                从节点( rank 1 ):
#                  NNODES=2 NODE_RANK=1 \
#                  MASTER_ADDR=<主节点IP> MASTER_PORT=29500 \
#                  bash scripts/train_leapbotce_2653853.sh --task baseline --gpus 8
#              注意: 脚本会用 MASTER_PORT+11 端口做各节点 run_id(输出目录名)同步，
#              主节点防火墙需放行这两个端口。所有节点的输出都写到共享存储
#              FASTWAM_RUNS_ROOT 下同一目录（env_2653853.sh 已配置）。
#
#   ZeRO 阶段:  本脚本固定走 ZeRO-1（train_zero1.sh，官方默认，显存足够）。
#              Wan 冻结、可训练参数不大，一般不需要 ZeRO-2/3。
#
# 五、输出位置
#   checkpoints / 日志 / 配置快照写入:
#     ${FASTWAM_RUNS_ROOT}/<task名>/<run_id>/
#   FASTWAM_RUNS_ROOT 默认 = /mnt/c2-worldmodel/2653853/leapbot-ce/runs
#   （LeapBotCE 的 delta checkpoint 约 2 GB；DeepSpeed 可恢复状态可达几十 GB，
#    训练确认不再续训后可手动清理其 deepspeed 状态目录。）
#
#   训练完成后用 scripts/eval_leapbotce_delay_sweep.sh 做时延评测，
#   详见仓库 LEAPBOTCE.md。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# 命令行参数解析（所有参数都有默认值，与官方任务配置一致）
# -----------------------------------------------------------------------------
TASK="all"               # 训练类型: all | all-smoke | spatial | kv | smoke | 完整 task 配置名
NGPUS=1                  # 本机 GPU 数
DELAY=""                 # data.train.max_delay_steps（空=用任务配置默认 20）
STEPS=""                 # max_steps（空=用任务配置默认 10000；smoke 任务自带 10）
EPOCHS=""                # num_epochs（一般留空，用 max_steps 控制）
BATCH_SIZE=""            # 每 GPU micro-batch（空=任务配置默认 2）
ACCUM=""                 # 梯度累积（空=任务配置默认 4）
LR=""                    # 学习率（空=默认 1e-4）
SCHEDULER=""             # lr_scheduler_type（空=默认 cosine）
WEIGHT_DECAY=""          # weight_decay（空=默认 1e-2）
GRAD_CLIP=""             # max_grad_norm（空=默认 1.0）
SEED=""                  # 随机种子（空=默认 42）
WORKERS=""               # num_workers（空=默认 4）
SAVE_EVERY=""            # save_every（空=默认 500）
KEEP_CKPT=""             # keep_last_checkpoints（空=默认 1）
RUN_NAME=""              # run_id 输出子目录名（空=自动时间戳；多机建议显式指定）
RESUME=""                # 断点续训路径
STALE_LAMBDA=""          # stale 损失权重上限（空=默认 0.5）
STALE_WARMUP=""          # stale 损失 warmup 步数（空=-1 自动=max_steps/2）
KV_RANK=""               # KV 适配器 rank（空=默认 64）
KV_DROPOUT=""            # KV 适配器 dropout（空=默认 0.0）
USE_WANDB=0              # 是否打开 wandb
DRY_RUN=0                # 1=只打印命令不执行

OVERRIDES=()             # `--` 之后透传给 Hydra 的额外覆盖，优先级最高

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
    -h|--help)       sed -n '2,110p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（--help 查看用法）" >&2; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# 训练类型 -> Hydra task 配置名 映射
# -----------------------------------------------------------------------------
case "${TASK}" in
  all)        TASK_CFG="libero_all_leapbotce" ;;
  all-smoke)  TASK_CFG="libero_all_leapbotce_smoke" ;;
  spatial)    TASK_CFG="libero_spatial_leapbotce" ;;
  baseline)   TASK_CFG="libero_spatial_leapbotce" ;;   # 兼容别名 = spatial
  kv)         TASK_CFG="libero_spatial_leapbotce_kv" ;;
  smoke)      TASK_CFG="libero_spatial_leapbotce_smoke" ;;
  *)          TASK_CFG="${TASK}" ;;
esac

# -----------------------------------------------------------------------------
# 组装 Hydra 覆盖项：只在用户显式给出时才覆盖，其余沿用任务配置默认值
# -----------------------------------------------------------------------------
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

# CE 专属：stale 损失与 KV 适配器
[[ -n "${STALE_LAMBDA}" ]] && HYDRA_ARGS+=("model.cloudedge.stale_loss_lambda_max=${STALE_LAMBDA}")
[[ -n "${STALE_WARMUP}" ]] && HYDRA_ARGS+=("model.cloudedge.stale_loss_warmup_steps=${STALE_WARMUP}")
if [[ -n "${KV_RANK}" || -n "${KV_DROPOUT}" ]]; then
  # 改 KV 适配器参数时自动确保适配器开启（baseline 任务上也可用）
  HYDRA_ARGS+=("model.cloudedge.kv_adapter.enabled=true")
  [[ -n "${KV_RANK}"    ]] && HYDRA_ARGS+=("model.cloudedge.kv_adapter.rank=${KV_RANK}")
  [[ -n "${KV_DROPOUT}" ]] && HYDRA_ARGS+=("model.cloudedge.kv_adapter.dropout=${KV_DROPOUT}")
fi

# wandb（train.yaml 默认 enabled=false）
if [[ "${USE_WANDB}" == "1" ]]; then
  HYDRA_ARGS+=("wandb.enabled=true")
fi

# 额外透传的 Hydra 覆盖（`--` 后面的内容），优先级最高，可覆盖上面一切
# 写法兼容 bash 4.2（CentOS 7）：空数组在 set -u 下直接展开会报 unbound variable
HYDRA_ARGS+=(${OVERRIDES[@]+"${OVERRIDES[@]}"})

# -----------------------------------------------------------------------------
# 环境：conda + 数据/模型路径 + LIBERO 仿真配置
# -----------------------------------------------------------------------------
# shellcheck source=env_2653853.sh
source "${SCRIPT_DIR}/env_2653853.sh"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# 临时代理：让 wandb 在线同步到账号（训练本身不依赖代理，代理断了只影响
# wandb 上报，不影响训练继续跑）。地址/端口可用 PROXY_HOST / PROXY_PORT
# 环境变量覆盖；PROXY_HOST 默认本机（调试机上跑训练的场景）。
# -----------------------------------------------------------------------------
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-33716}"
export http_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
export https_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export all_proxy="socks5://${PROXY_HOST}:${PROXY_PORT}"
export ALL_PROXY="${all_proxy}"
# 可用 PROXY_ENABLED=0 关掉代理（如代理客户端没开时）
if [[ "${PROXY_ENABLED:-1}" == "1" ]]; then
  if wget http://www.baidu.com -O /dev/null -q --timeout=5 2>/dev/null; then
    echo "[leapbotce] 代理工作正常 (${PROXY_HOST}:${PROXY_PORT})，wandb 可在线同步"
  else
    echo "[leapbotce] 警告: 代理 ${PROXY_HOST}:${PROXY_PORT} 不可达，wandb 同步可能失败（不影响训练）"
  fi
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  echo "[leapbotce] 代理已按 PROXY_ENABLED=0 关闭"
fi

# 多机训练时建议显式固定 RUN_ID，保证各节点写入同一输出目录
if [[ -n "${RUN_NAME}" ]]; then
  export RUN_ID="${RUN_NAME}"
fi

# 可选：训练前先挑一块空闲 GPU 做显存自检提示（不强制）
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

# -----------------------------------------------------------------------------
# 启动（train_zero1.sh 负责多机 run_id 同步与 accelerate launch）
# -----------------------------------------------------------------------------
exec bash scripts/train_zero1.sh "${NGPUS}" "${HYDRA_ARGS[@]}"
