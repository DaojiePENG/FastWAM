"""AIFlow 平台训练变体：a6 ablation —— D_train=40（训练时延上探）。

═══════════════════════════════════════════════════════════════════════
★ Table 4 training ablation 之三：除 AI_FLOW_DELAY 外与 a6_lora_fs 逐字
  一致（母本 = AIFlow_config_libero_a6_lora_fs.py）
═══════════════════════════════════════════════════════════════════════
语义：训练侧 delay 上探 40（部署评测只测 D{0,10,20,40}，40 在域内训过
之后检验四窗曲线是否整体抬升、外推担忧是否消除）。对应论文
tab:ablations 行名 "D_train=40"，4_experiments.tex L241 预言它
"probes generalization"。
链路已核（2026-10-27）：AI_FLOW_DELAY → --delay N → launcher 注入
data.train.max_delay_steps=N → CloudEdgeRobotVideoDataset 窗口 =
ceil((max_delay+horizon)/4)*4+1 = 73 帧、sample_stale_index 在
[earliest, current) 均匀采样、episode 开头 pad 帧自动跳过——无任何
cap，delay 40 原生支持，零代码改动。模型侧 stale loss 权重只依赖训练
步数，cloud_delay_steps 仅进日志，对 d>20 无假设。
对照 = deployed a6 行（91.0/83.7/79.7/70.7，d_train=20）。唯一差异三项：
AI_FLOW_DELAY 20→40、AI_FLOW_RUN_NAME、任务名路由后缀；其余（EPOCHS=15
/ STALE_LAMBDA=0.5 / SEED=42 / 8×H200 / lr 2.8e-4 / 冷启动）逐字保留。

⚠️ 评测口径：训练收工后复用 AIFlow_config_eval_libero_a4_lora 变体，
把 AI_FLOW_RUN_DIR / AI_FLOW_CKPT 指到本 run 目录
（runs/libero_all_leapbotce_codec_lora/<run名>）、改 AI_FLOW_TAG，
LIBERO 四窗 D{0,10,20,40} 各 2000 eps 回填 Table 4。

提交方式：任务名 leapbot_ce_train_libero_a6_dtrain40（分发器整词后缀
libero_a6_dtrain40；训练线函数 globals 绑定定义模块，env 薄委托无效，故本
文件为全量复制版，管线部分与基座逐字一致，改配置只动常改区）。

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
AI_FLOW_TASK = "libero_all_leapbotce_codec_lora"  # a6 task：codec + video LoRA
                                       # （model 块与 a4 yaml 逐字一致）；LIBERO
                                       # 四套件联合训练
AI_FLOW_DELAY = 40    # ablation：d_train=40（上探训练时延；数据集窗口自适应，已核
                      # 无 cap，见文件头链路核对）
AI_FLOW_SEED = 42     # 与 a1 一致
AI_FLOW_GPU_NUM = 8   # 8×H200 bs16 = eff 128（a1 是 4卡×accum2，同 eff，动力学
                      # 不变）；冷启动无 resume，无 4-rank 铁律约束

# ─────────────────────────────────────────────────────────────────────
# ② 批大小 / 学习率 / 总步数
# ─────────────────────────────────────────────────────────────────────
# 总数据量固定 277,713 帧（1,712 条 demo）：
#   有效 batch = 卡数 × 每卡 batch × 累积 = 8×16×1 = 128（= a1）
#   总步数     = AI_FLOW_EPOCHS × 277713 ÷ 128（自动折算）
AI_FLOW_BATCH_SIZE = 16    # 对齐 a1/a4 release 轮（8×H200 实测余量大）
AI_FLOW_ACCUM = 1          # eff 128 = a1 动力学
AI_FLOW_LR = 0.00028       # = a1/a0（eff 128 的 √batch 折算值，勿改）——从头
                           # 训练用全配方 lr；LoRA 从 step0 co-adapt，无谷底问题
AI_FLOW_BATCH_BY_VRAM_GB = [(128, 16), (72, 8), (40, 4), (0, 2)]
                           # 显存(GB)阈值 → 每卡 batch（显式值优先，本表仅兜底）
AI_FLOW_EPOCHS = 15        # a1 是 11.5ep 且 D20 终点仍在爬 → 多给 3.5ep；每 1ep
                           # 1 ckpt 组成梯子，中途饱和可直接杀任务保留 ckpt
AI_FLOW_MAX_STEPS = 0      # 0 = 按 AI_FLOW_EPOCHS 折算：15 × 277713 / 128 = 32544

# ─────────────────────────────────────────────────────────────────────
# ③ 其余训练超参（全部对齐 a1）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_SCHEDULER = "cosine"     # 全程退火，同 a1（从头训练不适用 constant）
AI_FLOW_WEIGHT_DECAY = 1.0e-2
AI_FLOW_GRAD_CLIP = 1.0
AI_FLOW_WORKERS = 8
AI_FLOW_STALE_LAMBDA = 0.5
AI_FLOW_STALE_WARMUP = -1        # 自动 = total/2 = 16272，与 a1（12475/24951）
                                 # 同为半程 ramp 构型，动力学同构

# ─────────────────────────────────────────────────────────────────────
# ④ 保存与恢复
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_NUM_SAVE = 15            # 每 1ep 存 1 个（save_every 自动 = 2170 整除）
AI_FLOW_SAVE_EVERY = 0
AI_FLOW_KEEP_CKPT = 15           # 全保留（~2GB/个 ≈ 31GB）
AI_FLOW_RESUME = ""              # 冷启动：官方 ActionDiT ckpt 由任务配置加载，
                                 # 不续训（同 a1）

# ─────────────────────────────────────────────────────────────────────
# ⑤ wandb 在线同步与临时代理（训练不受影响，只影响上报）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_WANDB = 1
AI_FLOW_WANDB_PROJECT = "leapbotce"
AI_FLOW_RUN_NAME = "codec_a6_dtrain40_15ep_seed42_h200"
                                 # 输出 runs/libero_all_leapbotce_codec_lora/<run名>/
AI_FLOW_PROXY_ENABLED = 1        # 节点代理不通自动 WANDB_MODE=offline（双保险）
AI_FLOW_PROXY_HOST = ""
AI_FLOW_PROXY_PORT = 33716

# ─────────────────────────────────────────────────────────────────────
# ⑥ 集群 NCCL 稳定性（同事趟过的坑，勿动）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_NCCL_CUMEM_ENABLE = "0"
AI_FLOW_NCCL_TIMEOUT = "1200"

# ─────────────────────────────────────────────────────────────────────
# ⑦ 高级：额外 Hydra 覆盖（追加在启动命令 `--` 之后，优先级最高）
# ─────────────────────────────────────────────────────────────────────
AI_FLOW_EXTRA_OVERRIDES: list[str] = [
    # 冻结基座与 a1 训练同一来源（官方 libero uncond 检查点）——从头训练的
    # LoRA 挂在该基座的 video expert 上，来源必须可追溯。
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
