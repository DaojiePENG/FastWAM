"""AIFlow 平台训练变体：RoboTwin 2.0 a6 式 = LoRA + codec 冷启动联合训练。

═══════════════════════════════════════════════════════════════════════
★ 论文标准配方上 RoboTwin（2026-10-19 用户定调：标准做法 = LoRA+Codec，
  epoch 锚官方 5ep；与 LIBERO a6 终判同源）
═══════════════════════════════════════════════════════════════════════
配方 = LIBERO a6 逐项移植（configs/task/robotwin_all_leapbotce_codec_lora.yaml）：
  - video LoRA（q/k/v/o，r16/α16）从 step0 与 KV codec + action expert 共训
    （a4 式"中途注入"的构型缺陷在 LIBERO 已实锤，杜绝）；
  - KV codec 7×8→**RoboTwin 特调 [6,10]=60 planning tokens**（token grid
    12×10 vs LIBERO 7×14，见 base leapbotce.yaml 注释），载荷 ~0.36MB 不变；
  - 冷启动 init = **robotwin_uncond_3cam_384.pt**（fastwam_release，经
    AI_FLOW_EXTRA_OVERRIDES 注入 model.mot_pretrained_path，与 LIBERO a6 注入
    libero_uncond 同机制）；数据侧 norm stats / text_embeds 921k 用线 B 同款；
  - 5 epoch 锚官方 robotwin_joint_3cam_384_1e-4；lr 1e-4@eff128 √batch 折算；
  - stale λ0.5，warmup -1（自动 total/2，与 a6/a1 同构）；d_train=20；seed 42。
与 10-01 非-codec 线 B（robotwin_all_leapbotce，已完训 5 ckpt）的差异即论文
CE 主张本身：codec 压缩 + LoRA 容量 + select 边缘前缀。

checkpoint 密度（10-19 用户指示"每天 1-2 个 ckpt 做探针"）：固定
save_every=14000。吞吐是 dataloader 瓶颈（每样本 3 路 mp4×53 帧解码），
steps/day ≈ 28k 与卡数/卡型基本无关（实测锚：16×H200 eff256 0.16 step/s，
10-01 线 B 全程），所以 14000 步 ≈ 0.5 天 ≈ 2 ckpt/天在任意档位都成立。
keep 20 > 最多 17 个（16 卡 5ep），全程不删。

⚠️ 时长只随卡数变（dataloader 瓶颈，加大每卡 batch 不加速）：
  ┌─────────────────┬──────────┬──────────┬─────────┬───────────────┐
  │ 组合             │ 有效batch │ 5ep 总步数│ lr      │ 预计耗时       │
  ├─────────────────┼──────────┼──────────┼─────────┼───────────────┤
  │ 16×H100(2机) ★   │ 128      │ ≈237.3k  │ 1e-4    │ ≈8.4 天        │
  │ 16×H200(2机) ★   │ 256      │ ≈118.7k  │ 1.4e-4  │ ≈8.4 天(实测锚)│
  │ 24×H100          │ 192      │ ≈158.2k  │ 1.2e-4  │ ≈5.6 天        │
  │ 24×H200(2-3机)   │ 384      │ ≈79.1k   │ 1.7e-4  │ ≈5.6 天        │
  │ 8×H200(1机)      │ 128      │ ≈237.3k  │ 1e-4    │ ≈17 天(不推荐) │
  └─────────────────┴──────────┴──────────┴─────────┴───────────────┘
  实测锚 16×H200 eff256 = 8.5 天（线 B 10-01→10-10 全程）；样本速率 ~2.6/s/rank
  （128 dataloader workers @16 rank），外推 24 rank ≈ 5.6 天。16×H100 与
  16×H200 同时长——瓶颈在数据解码不在卡，选哪档看空闲。

⚠️ 兼容口径（10-19 用户指示）：本变体沿用 robotwin 变体全套自动适配——
  _gpu_count()（AI_FLOW_GPU_NUM > 平台 GPU_NUM > torch 探测）、
  AI_FLOW_BATCH_BY_VRAM_GB 显存阈值表（H200-141G→bs16 / H100,H800-80G→bs8，
  bs8@80G 实测峰值 61.2GB 非-codec；codec 仅增 ~0.4GB/样本激活，余量充足；
  万一 H200 bs16 OOM 就 AI_FLOW_BATCH_SIZE=12 + AI_FLOW_LR 手动固定）、
  lr √(eff/128) 自动折算、run 目录自动卡型 tag（_h100/_h200 + _nN 节点数）。

⚠️ 命名防撞：变体后缀带 train 词（train_robotwin_a6_lora）——评测任务名
  leapbot_ce_eval_robotwin_a6_lora 只含 eval+robotwin+a6+lora，不会命中本
  变体（train 非其词），仍落 eval_robotwin；训练任务名
  leapbot_ce_train_robotwin_a6_lora 五词全含、最长后缀胜出。

提交方式：任务名 leapbot_ce_train_robotwin_a6_lora，工作目录（userPath）选
FastWAM 仓库根，无需任何改名操作（dispatcher 按任务名整词路由到本文件）。
提交后先核对 jinnTrainResult/train.log 头部 [LeapBot AIFlow] 行：
task / eff_batch / steps / lr / save 间隔。

改配置：只动下面「第 1 部分：常改区」；「第 2 部分：平台管线」与
AIFlow_config_robotwin.py 逐字一致，一般不用改。
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
                      # 残包: import 成功但无 .cuda —— LIBERO 训练变体实锤
                      # 21835144 秒挂, 统一当 None 处理
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
AI_FLOW_TASK = "robotwin_all_leapbotce_codec_lora"
                                         # 任务配置 configs/task/robotwin_all_leapbotce_codec_lora.yaml
                                         # = a6 式 LoRA+codec 冷启动（RoboTwin 数据侧同线 B）
AI_FLOW_DELAY = 20    # d_train：训练时云端观测时延上限（U{1..D} 采样）。
                      #   20 = 主实验配方；改 40 可练 d=40 版本，其余不用动
AI_FLOW_SEED = 42     # 训练随机种子（数据打乱/初始化；与线 B 一致）

# ─────────────────────────────────────────────────────────────────────
# ② 批大小 / 学习率 / 总步数 —— 充分利用申请到的卡（H100/H200 自动适配）
# ─────────────────────────────────────────────────────────────────────
# 官方配方（configs/task/robotwin_joint_3cam_384_1e-4.yaml）：每卡 batch 16、
# lr 1e-4（8 卡 = 有效 batch 128）、5 epoch。总数据量 6,075,103 帧
# （27,500 条 demo，LIBERO 全集的 21.9 倍），换算关系：
#   有效 batch = 节点数 × 每节点卡数 × 每卡 batch
#   总步数     = AI_FLOW_EPOCHS × 6075103 ÷ 有效 batch（自动折算）
#
# 自动（推荐，默认）：AI_FLOW_BATCH_SIZE = 0 且 AI_FLOW_LR = 0
#   每卡 batch 按 0 号卡显存查 AI_FLOW_BATCH_BY_VRAM_GB 阈值表：
#     ≥128GB（H200-141G）→ 16（=官方原值）；≥72GB（H100/H800-80G）→ 8
#     （80G smoke 实测 bs8 峰值 61.2GB；bs12 估算 ~69GB 余量偏小，求稳取 8）
#     ≥40GB → 4；更小 → 2
#   学习率按 √(有效batch/128) × 1e-4 从官方配方折算（官方 eff128 → 1e-4 原值）。
#   申请到 H200 还是 H100/H800 都不用改配置。
#
# ⚠️ 本任务吞吐瓶颈在 dataloader（每样本要解 3 路 mp4 各 53 帧，GPU 计算
#   只占 ~10-20%），加大每卡 batch 不会更快，卡数是唯一加速杠杆；16 卡档
#   H100/H200 同时长（≈8.4 天），24 卡档 ≈5.6 天——见文件头分档表。
AI_FLOW_BATCH_SIZE = 0     # 0 = 按显存自动选；>0 = 固定每卡 batch
AI_FLOW_ACCUM = 1          # 梯度累积步数；自动档已对齐官方，保持 1
AI_FLOW_LR = 0             # 0 = 按有效 batch 自动折算；>0 = 固定（如 1e-4）
AI_FLOW_BATCH_BY_VRAM_GB = [(128, 16), (72, 8), (40, 4), (0, 2)]
                           # 显存(GB)阈值 → 每卡 batch，从上往下取第一个满足的
AI_FLOW_EPOCHS = 5.0       # 训练多少个 epoch（对齐官方 RoboTwin 配方 5 epoch，
                           # 10-19 用户定调；中途饱和可杀任务保留已存 ckpt 梯子）
AI_FLOW_MAX_STEPS = 0      # >0 时直接指定总步数（覆盖 EPOCHS 折算）；0 = 自动

# ─────────────────────────────────────────────────────────────────────
# ③ 其余训练超参（官方/已验证默认值，一般不动）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_SCHEDULER = "cosine"     # 学习率调度：cosine 全退火（与 a6/线 B 一致）
AI_FLOW_WEIGHT_DECAY = 1.0e-2    # 权重衰减
AI_FLOW_GRAD_CLIP = 1.0          # 梯度裁剪
AI_FLOW_WORKERS = 8              # 每个 rank 的 dataloader 进程数（训练节点核多，8 合适）
AI_FLOW_STALE_LAMBDA = 0.5       # stale 损失权重上限（CloudEdge 抗时延目标）
AI_FLOW_STALE_WARMUP = -1        # stale 权重线性爬升步数；-1 = 自动取总步数/2

# ─────────────────────────────────────────────────────────────────────
# ④ 保存与恢复：固定间隔 = 每天恒定 ckpt 密度
# ─────────────────────────────────────────────────────────────────────
# 10-19 用户指示：训练要一两天以上，每天 1-2 个 ckpt 做探针实验。吞吐是
# dataloader 瓶颈 → steps/day ≈ 28k 与卡数/卡型基本无关（16 卡 eff128 实测
# 外推 0.325 step/s；16×H200 eff256 0.16 step/s 实测锚，见文件头表），所以
# 固定 save_every=14000 ≈ 0.5 天/个 ≈ 2 个/天，在 16/24 卡档都成立。
# 个数随档位变：16 卡 5ep ≈ 17 个、24 卡 ≈ 12 个；keep 20 全程不删。
# （每个 checkpoint ≈ 2.1GB delta 权重，10-04 以线 B step_023731 实测修正）
AI_FLOW_NUM_SAVE = 0             # 0 = 不用均分模式；用下面 SAVE_EVERY 固定间隔
AI_FLOW_SAVE_EVERY = 14000       # 每 14k 步存 1 个（≈2 个/天）+ 最终必存 1 个
AI_FLOW_KEEP_CKPT = 20           # 最多保留最近 20 个（> 16 卡档最多 17 个，全程不删）
AI_FLOW_RESUME = ""              # 断点续训：填 DeepSpeed 状态目录，例如
                                 #   /mnt/c2-worldmodel/2653853/leapbot-ce/runs/
                                 #     robotwin_all_leapbotce_codec_lora/<run名>/
                                 #     checkpoints/state
                                 # 留空 = 从头训。注意续训需与原 run 的卡数/batch 一致

# ─────────────────────────────────────────────────────────────────────
# ⑤ wandb 在线同步与临时代理（训练不受影响，只影响上报）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_WANDB = 1                # 1 = 打开 wandb 在线上报（train.yaml 默认关闭，这里强制打开）
AI_FLOW_WANDB_PROJECT = "leapbotce"  # wandb 项目名（与之前几轮一致，方便对比曲线）
AI_FLOW_RUN_NAME = ""            # wandb run 名 / 输出子目录名；留空 = 自动取
                                 #   robotwin_all_codec_lora_dtrain{D}_seed{S}_{显卡型号}
                                 #   （如 _h100 / _h200，避免不同卡的 run 混目录）
                                 # 注意：同名会续写同一输出目录，重跑请改名
#
# 临时代理（wandb 出网用）。代理客户端目前跑在调试机上：
#   - 默认 PROXY_HOST 为空 → 启动脚本用 127.0.0.1，只在调试机上训练时可达；
#   - 在 AIFlow 训练节点上要让 wandb 真正同步，二选一：
#     (a) 把 AI_FLOW_PROXY_HOST 设为调试机内网 IP，且调试机代理客户端
#         开"允许局域网连接"（监听 0.0.0.0）；
#     (b) 若训练节点本身可直连外网，把 AI_FLOW_PROXY_ENABLED 设为 0。
#
# ★ 09-30 事故教训（线 B 首提即死于此）：训练节点代理不通，wandb online
#   初始化重试 90s 后抛 CommError 弄死 rank0，其余卡在 NCCL 上等满超时被
#   watchdog 全杀。现已双保险修复：启动脚本检测到代理不通会自动
#   WANDB_MODE=offline（曲线落盘 run 目录，训练完 wandb sync 补传）；
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
# a6 式冷启动 init：冻结 expert 从 RoboTwin uncond release 载入（与 LIBERO a6
# 注入 libero_uncond_2cam224.pt 同机制；LeapBotCE 只存 delta，评测时冻结底座
# 也从该路径重建，评测侧记得对齐同一 path）。
AI_FLOW_EXTRA_OVERRIDES: list[str] = [
    "model.mot_pretrained_path=/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/robotwin_uncond_3cam_384.pt",
]

# ─────────────────────────────────────────────────────────────────────
# ⑧ 环境（一般不动）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_CONDA_ENV = os.environ.get(
    "AI_FLOW_CONDA_ENV", "/dahuafs/userdata/2653853/code/.envs/leapbot-va"
)                                # 共享家目录里的 conda 环境（含 torch/accelerate/diffsynth）
AI_FLOW_DATA_FRAMES = 6075103    # RoboTwin 2.0 全量总帧数（算总步数用）


# ═══════════════════════════════════════════════════════════════════════
# 第 2 部分：平台管线（与 AIFlow_config_robotwin.py 逐字一致，一般不用改）
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent
LAUNCHER_REL = Path("scripts") / "train_leapbotce_robotwin_2653853.sh"


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
    for key in ("h200", "h800", "h100", "a100", "a800", "l40", "a40"):
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
    """学习率：显式指定(>0)优先，否则按 √batch 从官方 1e-4@有效128 折算。

    官方 RoboTwin 配方（robotwin_joint_3cam_384_1e-4.yaml）：每卡 batch 16
    × 8 卡 = 有效 batch 128，lr 1e-4；其它有效 batch 按 √ 缩放。
    """
    explicit = float(_aiflow_value("AI_FLOW_LR") or 0)
    if explicit > 0:
        return str(_aiflow_value("AI_FLOW_LR"))
    return f"{1.0e-4 * (eff_batch / 128.0) ** 0.5:.2g}"


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

    训练本体统一走 scripts/train_leapbotce_robotwin_2653853.sh（与调试机同一入口，
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

    # checkpoint 节奏：SAVE_EVERY>0 = 固定间隔（本变体 14000 ≈ 2 个/天，见常改区④）；
    # SAVE_EVERY=0 时退回按 NUM_SAVE 均分
    num_save = max(int(_aiflow_value("AI_FLOW_NUM_SAVE") or 0), 0)
    save_every = int(_aiflow_value("AI_FLOW_SAVE_EVERY") or 0)
    if save_every <= 0:
        save_every = max(-(-steps // max(num_save, 2)), 1)  # 向上取整：中途 N-1 个 + 最终 1 个
    keep_ckpt = int(_aiflow_value("AI_FLOW_KEEP_CKPT") or 0)
    if keep_ckpt <= 0:
        keep_ckpt = max(num_save, 1)  # 默认全保留

    run_name = str(_aiflow_value("AI_FLOW_RUN_NAME") or "").strip()
    if not run_name:
        run_name = (f"robotwin_all_codec_lora_dtrain{_aiflow_value('AI_FLOW_DELAY')}"
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
