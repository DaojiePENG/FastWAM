"""AIFlow 平台训练变体：a4 = codec 主线续训 + video-expert LoRA（producer 侧适配）。

═══════════════════════════════════════════════════════════════════════
★ 当前用途（10-16 定）：a4 LoRA 线 — a1 终点 resume + LoRA 续训 10ep
═══════════════════════════════════════════════════════════════════════
背景：a0≈a1（无压缩也 91.9）+ a2≈a1（kv_decoder 解耦无增益）双判决把 D0 /
保留率缺口归因于「生产-消费适配 / 训练配方」。a4 = 对冻结的 Wan video expert
注入 LoRA（q/k/v/o，r16/α32，120 wrappers ≈11.8M 参数）并从 a1 终点
（codec_a1_release_seed42_h200 step_024951.pt）weight-only 续训 10ep：
  * 零-B 注入 + 权重续训 → step0 行为 = a1 终点，严格衔接；
  * 续训收益（D20 保留率，趋势探针 +9.0/3.3ep 零递减）与 LoRA 收益
    （D0/goal/10，纯续训 D0 已饱和）单线干净归因。
配方：constant lr 1e-4（cosine 已走完，重启不砸已收敛模块）、8×H200 bs16
（对齐 a1）、epoch 主口径 5ep = 10848 步（每 1ep 存 1 个 ckpt 共 5 个，供中途
测评）、stale warmup -1 = 立即满权重 0.5（leapbotce.py 语义，
与 a1 动力学无缝衔接，无重新爬坡）。
本地冒烟已全链路通过（FastWAM 5dbc687）：训练侧 20 步 loss 正常 + lora_b
非零；评测侧按 run config 重建 + strict load 逐位一致 + 双路前向。

提交方式：任务名 leapbot_ce_train_libero_a4_lora（分发器整词匹配后缀
libero_a4_lora；训练线函数 globals 绑定定义模块，env 薄委托无效，故本文件
为全量复制版，管线部分与基座逐字一致，改配置只动常改区）。

平台对接说明（与基座一致）：工作目录选 FastWAM 根；平台 import 分发器
AIFlow_config.py 读取 __C.TRAIN.CMD 并调用 AutoTrain(**params)；训练走
scripts/train_leapbotce_libero_2653853.sh（与调试机同一入口）。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import torch
    if not hasattr(torch, "cuda"):
        torch = None  # 平台镜像 (leapbot_lite:v1, py3.12) 里有同名 torch stub/
                      # 残包: import 成功但无 .cuda —— 实锤 21835144 秒挂
                      # (AttributeError: module 'torch' has no attribute 'cuda'),
                      # 统一当 None 处理, 真正的训练跑在 shell 里激活的 conda env。
except Exception:  # noqa: BLE001 - 平台镜像 torch 残包 import 阶段也可能炸
    torch = None

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__


# ═══════════════════════════════════════════════════════════════════════
# 第 1 部分：常改区（按用途分类，改完保存即可，平台下次提交自动生效）
# ═══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# ① 训练任务：练什么
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_TASK = "libero_all_leapbotce_codec_lora_a4"  # a4 task 配置（codec + video LoRA + a1 resume）
                                       # = LIBERO 四套件（spatial/object/goal/10）联合训练
AI_FLOW_DELAY = 20    # d_train：训练时云端观测时延上限（U{1..D} 采样）。
                      #   20 = 主实验配方；改 40 可练 d=40 版本，其余不用动
AI_FLOW_SEED = 42     # 训练随机种子（数据打乱/初始化；与之前 50k 轮一致）
AI_FLOW_GPU_NUM = 8   # a4 为 weight-only resume（.pt 文件，非 ZeRO 分片目录），无 4-rank
                      # 铁律约束；8×H200 bs16 与 a1 训练动力学完全对齐。AutoTrain 的
                      # world-size 守卫只对目录型 resume 生效，.pt 文件自动跳过。

# ─────────────────────────────────────────────────────────────────────
# ② 批大小 / 学习率 / 总步数 —— 充分利用申请到的卡（H100/H200 自动适配）
# ─────────────────────────────────────────────────────────────────────
# 总数据量固定 277,713 帧（1,712 条 demo），换算关系：
#   有效 batch = 卡数 × 每卡 batch × 累积
#   总步数     = AI_FLOW_EPOCHS × 277713 ÷ 有效 batch（自动折算）
#
# 两种填法：
#   自动（推荐，默认）：AI_FLOW_BATCH_SIZE = 0 且 AI_FLOW_LR = 0
#     每卡 batch 按 0 号卡显存查 AI_FLOW_BATCH_BY_VRAM_GB 阈值表：
#       ≥128GB（H200-141G）→ 16；≥72GB（H100/A800-80G）→ 8；≥40GB → 4；更小 → 2
#     学习率按 √(有效batch/16) × 1e-4 从官方配方折算（有效 128 → 2.8e-4）。
#     申请到 H200 还是 H100 都不用改配置；H100 上结果与之前实测配方一致
#     （bs8 / 2e-4）。
#   手动固定：AI_FLOW_BATCH_SIZE=8、AI_FLOW_LR=2e-4 —— 想严格复现已验证的
#     有效 batch 64 配方时用（代价：H200 上只用到一半显存）。
#
# 参考组合（lr 按 √batch 从官方 1e-4@16 折算）：
#   ┌───────────────────────┬──────────┬───────────────┬─────────┐
#   │ 组合                  │ 有效batch │ 总步数        │ lr      │
#   ├───────────────────────┼──────────┼───────────────┼─────────┤
#   │ 官方配方（2卡×bs8）    │ 16       │ 50k/2.9ep     │ 1.0e-4  │
#   │ 8×H100 自动 ★         │ 64       │ ≈49.9k/11.5ep │ 2e-4    │
#   │ 8×H200 自动 ★         │ 128      │ ≈25.0k/11.5ep │ 2.8e-4  │
#   └───────────────────────┴──────────┴───────────────┴─────────┘
# H100-80G 上 bs8 实测峰值 64.2GB；H200-141GB 上 bs16 预计 ~115-130GB，
# 万一 OOM 把 AI_FLOW_BATCH_SIZE 钉到 12 或 8（lr 同步手动给固定值）。
AI_FLOW_BATCH_SIZE = 16    # 对齐 a1 release 轮（8卡×bs16 = 有效 batch 128）；H200-141GB
                           # 显存余量极大（冒烟 B4 自用 ~16GB，checkpointing 已省激活
                           # 大头；B16 激活增量仅 +5GB 量级）。
AI_FLOW_ACCUM = 1          # 梯度累积步数；原轮就是 1，保持
AI_FLOW_LR = 0.0001        # LoRA 标准 lr + 续训稳定性：cosine 已走完，constant 1e-4
                           # 不砸已收敛模块（action expert / codec / LoRA 同一 lr）。
AI_FLOW_BATCH_BY_VRAM_GB = [(128, 16), (72, 8), (40, 4), (0, 2)]
                           # 显存(GB)阈值 → 每卡 batch，从上往下取第一个满足的
AI_FLOW_EPOCHS = 5        # a4 续训量：5ep = 10848 步 ≈ 15h（用户指示尽快出结果、
                          # epoch 从紧；a1 趋势探针显示续训收益主要在保留率且无
                          # 过训迹象，5ep 足够出结果，不够可从终点 ckpt 再续）
AI_FLOW_MAX_STEPS = 0     # 0 = 按 AI_FLOW_EPOCHS 折算总步数（10-16 用户准则：epoch
                          # 为主口径）。8 卡 eff128 下 1ep = 277713/128 = 2170 步。
                          # stale warmup=-1 → 立即满权重 0.5（leapbotce.py 语义），
                          # 与 a1 动力学无缝衔接。

# ─────────────────────────────────────────────────────────────────────
# ③ 其余训练超参（官方/已验证默认值，一般不动）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_SCHEDULER = "constant"  # 续训重启不重入 cosine；1e-4 恒定（LoRA 惯例）
AI_FLOW_WEIGHT_DECAY = 1.0e-2    # 权重衰减
AI_FLOW_GRAD_CLIP = 1.0          # 梯度裁剪
AI_FLOW_WORKERS = 8              # 每个 rank 的 dataloader 进程数（训练节点核多，8 合适）
AI_FLOW_STALE_LAMBDA = 0.5       # stale 损失权重上限（CloudEdge 抗时延目标）
AI_FLOW_STALE_WARMUP = -1        # stale 权重线性爬升步数；-1 = 自动取总步数/2

# ─────────────────────────────────────────────────────────────────────
# ④ 保存与恢复：指定"存几个"而不是"隔多久存"
# ─────────────────────────────────────────────────────────────────────
# 推荐用法：AI_FLOW_NUM_SAVE 填想保存的 checkpoint 总个数（含最终那一个），
# 保存间隔 = 总步数 ÷ 个数（向上取整），随申请到的卡型自动缩放，不用手算：
#   8×H100  总步数 49,902 → 每  9,981 步一个 → 10k/20k/30k/40k + 最终 49,902
#   8×H200  总步数 24,951 → 每  4,991 步一个 →  5k/10k/15k/20k + 最终 24,951
# （训练到总步数时 train.py 必存最终 checkpoint，与间隔是否整除无关，
#   见 src/fastwam/trainer.py 训练循环结尾的 "[done] max_steps reached"）
AI_FLOW_NUM_SAVE = 5            # 每 epoch 存 1 个（用户指示：中间 ckpt 留测评）；
AI_FLOW_SAVE_EVERY = 0           # 0 = 自动按 NUM_SAVE 均分 → 每 2170 步（=每 1ep）
                                 # 一个：2170/4340/6510/8680/10848（终点）
AI_FLOW_KEEP_CKPT = 5            # 全保留（delta ckpt ≈2GB/个，5 个 ≈10GB）
AI_FLOW_RESUME = ("/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
                  "libero_all_leapbotce_codec/"
                  "codec_a1_release_seed42_h200/checkpoints/weights/step_024951.pt")
                                 # a1 终点 weight-only resume（.pt 文件 = 只加载权重，
                                 # 步数归零、优化器/scheduler 重开 = constant 1e-4 全新
                                 # 动力学）。LoRA wrappers 构建时注入，load 时 strict
                                 # 恢复 lora_a/lora_b（冒烟验证逐位一致）。

# ─────────────────────────────────────────────────────────────────────
# ⑤ wandb 在线同步与临时代理（训练不受影响，只影响上报）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_WANDB = 1                # 1 = 打开 wandb 在线上报（train.yaml 默认关闭，这里强制打开）
AI_FLOW_WANDB_PROJECT = "leapbotce"  # wandb 项目名（与之前几轮一致，方便对比曲线）
AI_FLOW_RUN_NAME = "codec_a4_lora_seed42_h200"
                                 # 输出 runs/libero_all_leapbotce_codec/<run名>/（沿用
                                 # codec 线目录，评测线按 run 目录 config.yaml 重建模型）

#
# 临时代理（wandb 出网用）。代理客户端目前跑在调试机上：
#   - 默认 PROXY_HOST 为空 → 启动脚本用 127.0.0.1，只在调试机上训练时可达；
#   - 在 AIFlow 训练节点上要让 wandb 真正同步，二选一：
#     (a) 把 AI_FLOW_PROXY_HOST 设为调试机内网 IP，且调试机代理客户端
#         开"允许局域网连接"（监听 0.0.0.0）；
#     (b) 若训练节点本身可直连外网，把 AI_FLOW_PROXY_ENABLED 设为 0。
#
# ★ 09-30 事故教训：首次 AIFlow 提交（8×H200）就是死在这里——训练节点代理
#   不通，wandb online 初始化重试 90s 后抛 CommError 弄死 rank0，其余 7 卡
#   在 NCCL 集合通信上等满 600s 超时被 watchdog 全杀。现已双保险修复：
#   启动脚本检测到代理不通会自动 WANDB_MODE=offline（曲线落盘到
#   run 目录 /mnt/c2-worldmodel/.../<run名>/wandb/，训练完在调试机上
#   wandb sync <run目录>/wandb/latest-run 补传即可看到曲线）；
#   trainer.py 里 wandb.init 失败也会自动降级 offline，绝不再影响训练。
AI_FLOW_PROXY_ENABLED = 1        # 1 = 走代理出网；0 = 不走代理
AI_FLOW_PROXY_HOST = ""          # 代理地址；空 = 127.0.0.1（仅调试机本机有效）
AI_FLOW_PROXY_PORT = 33716       # 代理端口（与调试机上代理客户端一致）

# ─────────────────────────────────────────────────────────────────────
# ⑥ 集群 NCCL 稳定性（同事趟过的坑，勿动；调试机 09-29 的 NCCL SIGABRT 即同类问题）
# ─────────────────────────────────────────────────────────────────────
#   NCCL_CUMEM_ENABLE=0              关闭 CUDA cumem 注册（集群上会坏）
#   TORCH_NCCL_ASYNC_ERROR_HANDLING=1  NCCL 超时后主动抛错而不是全体卡死
#   NCCL_TIMEOUT=1200                集合通信超时 20 分钟（数据加载偶发慢）
#   NCCL_NET_GDR_LEVEL=LOC           禁用跨 NIC 的 GPU Direct RDMA（集群 GDR 注册损坏），
#                                    保留 IB 传输；GDR 修复后可用 LEAPBOT_NCCL_NET_GDR_LEVEL 覆盖
AI_FLOW_NCCL_CUMEM_ENABLE = "0"
AI_FLOW_NCCL_TIMEOUT = "1200"

# ─────────────────────────────────────────────────────────────────────
# ⑦ 高级：额外 Hydra 覆盖（追加在启动命令 `--` 之后，优先级最高）
# ─────────────────────────────────────────────────────────────────────
# 例：开云端 KV 适配器实验 → ["model.cloudedge.kv_adapter.enabled=true",
#                            "model.cloudedge.kv_adapter.rank=8"]
AI_FLOW_EXTRA_OVERRIDES: list[str] = [
    # 冻结基座必须与 a1 训练时同一来源（a1 CMD 原样透传）：官方 libero uncond
    # 检查点。a1 delta ckpt 不含 video expert 权重，重建靠 mot_pretrained_path。
    "model.mot_pretrained_path=/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt",
]

# ─────────────────────────────────────────────────────────────────────
# ⑧ 环境（一般不动）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_CONDA_ENV = os.environ.get(
    "AI_FLOW_CONDA_ENV", "/dahuafs/userdata/2653853/code/.envs/leapbot-va"
)                                # 共享家目录里的 conda 环境（含 torch/accelerate/diffsynth）
AI_FLOW_DATA_FRAMES = 277713     # 四套件联合训练总帧数（算总步数用）


# ═══════════════════════════════════════════════════════════════════════
# 第 2 部分：平台管线（与同事模板同构，一般不用改）
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent
LAUNCHER_REL = Path("scripts") / "train_leapbotce_libero_2653853.sh"


def _q(value) -> str:
    """shell 单词引用，防路径里有空格。"""
    return shlex.quote(str(value))


def _value(container, key):
    if not container:
        return None
    return container.get(key) if isinstance(container, dict) else getattr(container, key, None)


def _aiflow_value(key, params=None):
    """取配置值：params(平台传入) 优先于模块常量，方便平台侧覆盖。"""
    conf = globals().get("__C")
    params_conf = params.get("conf") if isinstance(params, dict) else None
    candidates = [
        _value(_value(conf, "TRAIN"), key), _value(conf, key),
        _value(_value(params_conf, "TRAIN"), key), _value(params_conf, key),
        _value(params, key),
        globals().get(key),  # 兜底：第 1 部分定义的 AI_FLOW_* 模块常量
    ]
    return next((item for item in candidates if item not in (None, "")), None)


def _repo_workdir() -> Path:
    """训练运行目录：平台给的 userPath（须含我们的启动脚本），否则本文件所在目录。"""
    user_path = os.environ.get("userPath")
    if user_path and (Path(user_path) / LAUNCHER_REL).is_file():
        return Path(user_path)
    return REPO_ROOT


def _gpu_count() -> int:
    """本机卡数：模块常量 > 平台环境变量 GPU_NUM > torch 探测 > 1。"""
    if _aiflow_value("AI_FLOW_GPU_NUM") is not None:
        return max(int(_aiflow_value("AI_FLOW_GPU_NUM")), 1)
    if os.environ.get("GPU_NUM"):
        return max(int(os.environ["GPU_NUM"]), 1)
    if torch is not None and hasattr(torch, "cuda"):
        try:
            if torch.cuda.is_available():
                return max(torch.cuda.device_count(), 1)
        except Exception:  # noqa: BLE001 - torch 残包/驱动异常一律走兜底
            pass
    return 1


def _gpu_name() -> str:
    if torch is None or not hasattr(torch, "cuda"):
        return "unknown"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _gpu_vram_gb() -> float:
    """0 号卡显存（GB）；探测不到返回 0，由调用方走保守兜底。"""
    if torch is None or not hasattr(torch, "cuda"):
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        return 0.0


def _gpu_tag() -> str:
    """显卡短标签（run 目录名用）：h200 / h100 / a100 / gpu。"""
    name = _gpu_name().lower()
    for key in ("h200", "h100", "a100", "a800", "l40", "a40"):
        if key in name:
            return key
    return "gpu"


def _gpu_batch_size() -> int:
    """每卡 batch：显式指定(>0)优先，否则按显存阈值表自动选。

    探测不到显存时兜底 8（H100-80G 实测安全值，即之前的默认行为）。
    """
    explicit = int(_aiflow_value("AI_FLOW_BATCH_SIZE") or 0)
    if explicit > 0:
        return explicit
    vram = _gpu_vram_gb()
    if vram <= 0:
        return 8
    for min_gb, bs in _aiflow_value("AI_FLOW_BATCH_BY_VRAM_GB"):
        if vram >= float(min_gb):
            return int(bs)
    return 8


def _effective_batch(local_gpus: int, nnodes: int) -> int:
    return max(local_gpus * nnodes * _gpu_batch_size()
               * int(_aiflow_value("AI_FLOW_ACCUM")), 1)


def _lr_value(eff_batch: int) -> str:
    """学习率：显式指定(>0)优先，否则按 √batch 从官方 1e-4@有效16 折算。"""
    explicit = float(_aiflow_value("AI_FLOW_LR") or 0)
    if explicit > 0:
        return str(_aiflow_value("AI_FLOW_LR"))
    return f"{1.0e-4 * (eff_batch / 16.0) ** 0.5:.2g}"


def _parse_dist_url(value):
    if not value:
        return None, None
    parsed = urlparse(str(value))
    if parsed.scheme:
        return parsed.hostname, parsed.port
    if ":" in str(value):
        host, port = str(value).rsplit(":", 1)
        return host, int(port)
    return str(value), None


def _dist_args(params=None) -> dict:
    """解析平台的分布式信息 → 本启动脚本认识的 NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT。"""
    nodeips = params.get("nodeips") if isinstance(params, dict) else None
    nodeips = list(nodeips or _aiflow_value("nodeips", params) or [])
    my_ip = params.get("myIp") if isinstance(params, dict) else None
    host, port = _parse_dist_url(
        os.environ.get("DIST_URL") or _aiflow_value("DIST_URL", params)
    )
    nnodes = os.environ.get("NNODES") or _aiflow_value("WORLD_SIZE", params) or len(nodeips) or 1
    nnodes = max(int(nnodes), 1)
    rank = os.environ.get("NODE_RANK") or _aiflow_value("RANK", params)
    if rank in (None, "") and my_ip in nodeips:
        rank = nodeips.index(my_ip)
    rank = int(rank or 0)
    if not 0 <= rank < nnodes:
        raise ValueError(f"Invalid node rank {rank} for {nnodes} nodes")
    master_addr = os.environ.get("MASTER_ADDR") or host or (nodeips[0] if nodeips else "127.0.0.1")
    if nnodes > 1 and str(master_addr).strip().lower() in {"127.0.0.1", "localhost", "::1"}:
        if nodeips:
            master_addr = nodeips[0]
    return {
        "nnodes": nnodes,
        "rank": rank,
        "master_addr": master_addr,
        "master_port": int(os.environ.get("MASTER_PORT") or port or 29971),
    }


def _total_steps(local_gpus: int, nnodes: int) -> int:
    """总步数：显式指定优先，否则按 epoch 折算（帧数 ÷ 有效 batch × epochs）。"""
    max_steps = int(_aiflow_value("AI_FLOW_MAX_STEPS") or 0)
    if max_steps > 0:
        return max_steps
    eff_batch = _effective_batch(local_gpus, nnodes)
    epochs = float(_aiflow_value("AI_FLOW_EPOCHS"))
    return max(int(round(epochs * int(_aiflow_value("AI_FLOW_DATA_FRAMES")) / eff_batch)), 1000)


def _build_cmd(params=None) -> str:
    """拼训练 shell 命令：cd 仓库 → conda → 环境变量 → 启动脚本。

    训练本体统一走 scripts/train_leapbotce_libero_2653853.sh（与调试机同一入口，
    它自己 source env_2653853.sh 拿模型/数据路径，不需要在这里重复 export）。
    """
    workdir = _repo_workdir()
    dist = _dist_args(params)
    local_gpus = _gpu_count()
    steps = _total_steps(local_gpus, dist["nnodes"])
    conda_env = Path(str(_aiflow_value("AI_FLOW_CONDA_ENV")))
    # 镜像瘦身 (10-09): 论文 envs 迁共享盘 .envs (非 conda root), conda.sh 不能
    # 再从 env 路径上推 —— 与 scripts/env_2653853.sh 同款双 root 探测。
    conda_root = next(
        (Path(root) for root in ("/root/miniconda3", "/home/myuser/miniconda3")
         if (Path(root) / "etc" / "profile.d" / "conda.sh").is_file()),
        conda_env.parent.parent,
    )
    conda_init = conda_root / "etc" / "profile.d" / "conda.sh"

    batch_size = _gpu_batch_size()
    eff_batch = _effective_batch(local_gpus, dist["nnodes"])
    lr = _lr_value(eff_batch)

    # checkpoint 节奏：指定"存几个"（NUM_SAVE，含最终），间隔随总步数自动均分；
    # SAVE_EVERY>0 时退回旧的固定间隔用法
    num_save = max(int(_aiflow_value("AI_FLOW_NUM_SAVE") or 0), 0)
    save_every = int(_aiflow_value("AI_FLOW_SAVE_EVERY") or 0)
    if save_every <= 0:
        save_every = max(-(-steps // max(num_save, 2)), 1)  # 向上取整：中途 N-1 个 + 最终 1 个
    keep_ckpt = int(_aiflow_value("AI_FLOW_KEEP_CKPT") or 0)
    if keep_ckpt <= 0:
        keep_ckpt = max(num_save, 1)  # 默认全保留

    run_name = str(_aiflow_value("AI_FLOW_RUN_NAME") or "").strip()
    if not run_name:
        run_name = (f"libero_all_dtrain{_aiflow_value('AI_FLOW_DELAY')}"
                    f"_seed{_aiflow_value('AI_FLOW_SEED')}_{_gpu_tag()}")
    if dist["nnodes"] > 1:
        run_name = f"{run_name}_n{dist['nnodes']}"

    hydra_overrides = [
        f"wandb.name={run_name}",
        f"wandb.project={_aiflow_value('AI_FLOW_WANDB_PROJECT')}",
        *[str(item) for item in (_aiflow_value("AI_FLOW_EXTRA_OVERRIDES") or [])],
    ]

    launcher_args = [
        "--task", str(_aiflow_value("AI_FLOW_TASK")),
        "--gpus", str(local_gpus),
        "--batch-size", str(batch_size),
        "--accum", str(_aiflow_value("AI_FLOW_ACCUM")),
        "--delay", str(_aiflow_value("AI_FLOW_DELAY")),
        "--seed", str(_aiflow_value("AI_FLOW_SEED")),
        "--steps", str(steps),
        "--lr", lr,
        "--scheduler", str(_aiflow_value("AI_FLOW_SCHEDULER")),
        "--weight-decay", str(_aiflow_value("AI_FLOW_WEIGHT_DECAY")),
        "--grad-clip", str(_aiflow_value("AI_FLOW_GRAD_CLIP")),
        "--workers", str(_aiflow_value("AI_FLOW_WORKERS")),
        "--stale-lambda", str(_aiflow_value("AI_FLOW_STALE_LAMBDA")),
        "--stale-warmup", str(_aiflow_value("AI_FLOW_STALE_WARMUP")),
        "--save-every", str(save_every),
        "--keep-ckpt", str(keep_ckpt),
        "--run-name", run_name,
    ]
    resume = str(_aiflow_value("AI_FLOW_RESUME") or "").strip()
    if resume:
        launcher_args += ["--resume", resume]
    if int(_aiflow_value("AI_FLOW_WANDB") or 0):
        launcher_args += ["--wandb"]

    shell = [
        f"cd {_q(workdir)}",
        f". {_q(conda_init)} && conda activate {_q(conda_env)}",
        # 线程数：训练节点 CPU 闲，8 线程/进程足够且不挤爆
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}"
        " MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}"
        " OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-8}",
        # 集群 NCCL 稳定性修复（见常改区 ⑥，勿动）
        "export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-"
        + str(_aiflow_value("AI_FLOW_NCCL_CUMEM_ENABLE")) + "}",
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}",
        "export NCCL_TIMEOUT=${NCCL_TIMEOUT:-" + str(_aiflow_value("AI_FLOW_NCCL_TIMEOUT")) + "}",
        # 保留 IB 传输，但避开集群损坏的 CUDA GDR 注册
        "export NCCL_NET_GDR_LEVEL=${LEAPBOT_NCCL_NET_GDR_LEVEL:-LOC}",
        # wandb 临时代理透传（启动脚本内部检测可达性，不通只警告）
        "export PROXY_ENABLED=" + str(_aiflow_value("AI_FLOW_PROXY_ENABLED")),
        "export PROXY_PORT=" + str(_aiflow_value("AI_FLOW_PROXY_PORT")),
    ]
    proxy_host = str(_aiflow_value("AI_FLOW_PROXY_HOST") or "").strip()
    if proxy_host:
        shell.append(f"export PROXY_HOST={_q(proxy_host)}")
    if dist["nnodes"] > 1:
        # 多机时把平台分配信息喂给启动脚本（scripts/train_zero1.sh 消费）
        shell += [
            f"export NNODES={dist['nnodes']}",
            f"export NODE_RANK={dist['rank']}",
            f"export MASTER_ADDR={_q(dist['master_addr'])}",
            f"export MASTER_PORT={dist['master_port']}",
        ]
    shell.append(
        f"bash {_q(workdir / LAUNCHER_REL)} {shlex.join(launcher_args)}"
        + ((" -- " + " ".join(_q(o) for o in hydra_overrides)) if hydra_overrides else "")
    )
    return " && ".join(shell)


GPU_NUM = str(_gpu_count())
GPU_NAME = _gpu_name()

# 平台读取的配置对象（字段名是平台约定，保持与同事模板一致）
__C = edict()
cfg = __C
__C.FRAME = "general"
__C.TRAIN = edict()
__C.TRAIN.ENABLE = True
__C.TRAIN.NET_TYPE = "generation"
__C.TRAIN.NAMES_MAP = ""
__C.TRAIN.PRE_TRAINED_MODLE_PATH = ""
__C.TRAIN.NET_PATH = ""
__C.TRAIN.TENSORBOARD_DIR = ""
__C.TRAIN.SNAPSHOT_PREFIX = ""
__C.TRAIN.SNAPSHOT_ITERS = 0
__C.TRAIN.BEST_MODEL = ""
__C.TRAIN.IS_SAVE_MED_MODEL = False
__C.TRAIN.DATA_LIST_PATH = ""
__C.TRAIN.VAL_DATA_LIST_PATH = ""
# 注意：下面这行在 import 时按"默认卡数"折算总步数，仅作展示；真正执行时
# AutoTrain 会用平台实际分配的 GPU_NUM/多机信息重新生成一遍。
__C.TRAIN.CMD = _build_cmd()
__C.CMD = __C.TRAIN.CMD


def AutoTrain(**params) -> bool:
    """平台训练入口：重建命令（带真实卡数/多机信息）→ 写日志 → 阻塞到训练结束。"""
    # 前置检查：模型/数据/输出都在共享盘，没挂载直接失败，别浪费一次申请
    if not os.path.isdir("/mnt/c2-worldmodel"):
        print("[LeapBot AIFlow] 错误: 训练节点未挂载 /mnt/c2-worldmodel"
              "（模型与数据路径都在该共享盘，见 scripts/env_2653853.sh）")
        return False
    workdir = _repo_workdir()
    if not (workdir / LAUNCHER_REL).is_file():
        print(f"[LeapBot AIFlow] 错误: 工作目录 {workdir} 下找不到 {LAUNCHER_REL}，"
              "请确认申请资源时工作目录选到 FastWAM 仓库根目录")
        return False

    # 续训 world-size 守卫（铁律①）：DeepSpeed ZeRO-1 优化器状态按 rank 分片，
    # state/step_110000 是 4 rank（4×H800 PCIe）保存的 —— world ≠ 4 时 load
    # 要么缺 rank 文件报错、要么分片错位，宁可这里快速失败也别烧一次申请。
    resume = str(_aiflow_value("AI_FLOW_RESUME") or "").strip()
    if resume and Path(resume).is_dir():
        dist_guard = _dist_args(params)
        world = _gpu_count() * dist_guard["nnodes"]
        if world != 4:
            print(f"[LeapBot AIFlow] 错误: 续训状态 {resume} 按 4 rank 保存，"
                  f"当前 world size = {_gpu_count()}卡 × {dist_guard['nnodes']}节点 = {world}。"
                  "请申请 4 卡 1 节点（或清空 AI_FLOW_RESUME 从头训）")
            return False

    cmd = _build_cmd(params)
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    dist = _dist_args(params)
    log_dir = os.environ.get("JinnTrainResult")
    log_name = "train.log" if dist["rank"] == 0 else f"train.node{dist['rank']}.log"
    log_path = Path(log_dir) / log_name if log_dir else None
    gpus = _gpu_count()
    steps = _total_steps(gpus, dist["nnodes"])
    eff = _effective_batch(gpus, dist["nnodes"])
    save_every = int(_aiflow_value("AI_FLOW_SAVE_EVERY") or 0)
    if save_every <= 0:
        save_every = max(-(-steps // max(int(_aiflow_value("AI_FLOW_NUM_SAVE") or 0), 2)), 1)
    print(f"[LeapBot AIFlow] task={_aiflow_value('AI_FLOW_TASK')}"
          f" d_train={_aiflow_value('AI_FLOW_DELAY')}"
          f" gpu={_gpu_name()}({GPU_NUM}×{dist['nnodes']}nodes)"
          f" bs={_gpu_batch_size()}/卡 eff_batch={eff}"
          f" steps={steps}(≈{steps * eff / int(_aiflow_value('AI_FLOW_DATA_FRAMES')):.1f} epoch)"
          f" lr={_lr_value(eff)} rank={dist['rank']}/{dist['nnodes']}")
    print(f"[LeapBot AIFlow] checkpoint: 每 {save_every} 步存 1 个，共 "
          f"{(steps - 1) // save_every + 1} 个（含最终）")
    print(f"[LeapBot AIFlow] CMD={cmd}")
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(cmd, shell=True, stdout=stream, stderr=stream, text=True)
                while proc.poll() is None:
                    time.sleep(10)
        else:
            proc = subprocess.Popen(cmd, shell=True)
            while proc.poll() is None:
                time.sleep(10)
        print(f"[LeapBot AIFlow] 训练退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:
        print(f"[LeapBot AIFlow] 训练失败: {exc}")
        return False
