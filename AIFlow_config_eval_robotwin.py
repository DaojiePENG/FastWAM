"""AIFlow 平台评测入口：LeapBotCE (ours) RoboTwin 2.0 云延迟 sweep。

═══════════════════════════════════════════════════════════════════════
使用方法（公司自动申请显卡评测）—— ⚠️ 本文件默认**未激活**
═══════════════════════════════════════════════════════════════════════
本树根的 `AIFlow_config.py` 是**活跃训练配置**（LIBERO 训练线）。本评测文件
以 `AIFlow_config_eval_robotwin.py` 备用名存放，申请 LeapBotCE RoboTwin 评测
前先做换名 dance（训练已结束、平台快照不受影响）：

    cd /dahuafs/userdata/2653853/code/FastWAM
    mv AIFlow_config.py AIFlow_config_libero.py
    cp AIFlow_config_eval_robotwin.py AIFlow_config.py

1. 申请评测资源时，工作目录（userPath）选到本文件所在目录（FastWAM 仓库根）。
   平台约定：工作目录下必须有名为 AIFlow_config.py 的脚本，平台 import 它、
   以 `cfg` 读取 __C.TRAIN.CMD 作为命令（非空 CMD → CUSTOM_CMD 分支直接 shell 执行）。
2. 评测命令统一走 scripts/eval_leapbotce_robotwin_sweep_2653853.sh：
   自动探测平台分配的卡、按显存 PACK 泳道（80G→2/卡，141G→3/卡）、
   5 延迟 × 10 任务共 50 格轮转分片、幂等续跑；结果落公共盘
   /mnt/c2-worldmodel/.../baseline-evaluations/leapbotce/robotwin_sweep/。
3. smoke：把下面 AI_FLOW_DELAYS / AI_FLOW_TASKS 改成只跑 1 格（如 "10" /
   "beat_block_hammer"），验证环境后恢复空串再全量提交。
4. checkpoint：默认 LEAPBOTCE_CKPT=latest（训练 run 下最新的 step_*.pt，
   5 epoch 共 5 个 delta ckpt）。也可经 AI_FLOW_CKPT 指定具体某个。

与 fastwam_official/AIFlow_config.py（Fast-WAM baseline 行）是两个不同根上的
两个文件，互不相干。

平台对接面（cfg / __C.TRAIN.* 字段集 / AutoTrain 签名）与同事模板及本树的
训练配置同构。已知提交坑（2026-09-30 实锤，两处都已规避）：
  ① 配置必须有 `cfg` 名（平台 `from AIFlow_config import cfg as conf`）；
  ② 平台把 userPath 拷到任务目录后再 import，__file__ 指向副本，
     工作目录必须用 userPath 解析。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__


# ─────────────────────────────────────────────────────────────────────
# 常改区
# ─────────────────────────────────────────────────────────────────────
# smoke 开关：空串 = 全量 5 延迟 × 10 任务；填值 = 只跑子集（1 卡即可验证）
AI_FLOW_DELAYS = os.environ.get("AI_FLOW_DELAYS", "").strip()
AI_FLOW_TASKS = os.environ.get("AI_FLOW_TASKS", "").strip()
# checkpoint："latest" = 训练 run 下最新 step_*.pt；或具体 ckpt 绝对路径
AI_FLOW_CKPT = os.environ.get("AI_FLOW_CKPT", "latest")

SWEEP_REL = Path("scripts") / "eval_leapbotce_robotwin_sweep_2653853.sh"

THREAD_CAPS = {
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
}


def _q(value) -> str:
    return shlex.quote(str(value))


def _repo_workdir() -> Path:
    """评测运行目录：平台给的 userPath（须含集群驱动脚本），否则本文件所在目录。

    平台会把 userPath 拷到任务目录再 import 本文件，__file__ 指向副本，
    所以必须以 userPath 为准（与训练配置同款逻辑）。
    """
    user_path = os.environ.get("userPath")
    if user_path and (Path(user_path) / SWEEP_REL).is_file():
        return Path(user_path)
    return Path(__file__).resolve().parent


def _build_cmd() -> str:
    exports = " ".join(f"{k}={v}" for k, v in THREAD_CAPS.items())
    smoke = ""
    if AI_FLOW_DELAYS:
        smoke += f" ROBOTWIN_EVAL_DELAYS={_q(AI_FLOW_DELAYS)}"
    if AI_FLOW_TASKS:
        smoke += f" ROBOTWIN_EVAL_TASKS={_q(AI_FLOW_TASKS)}"
    return (
        f"cd {_q(_repo_workdir())} && {exports} "
        f"LEAPBOTCE_CKPT={_q(AI_FLOW_CKPT)}{smoke} "
        f"bash {_q(SWEEP_REL)}"
    )


GPU_NUM = os.environ.get("GPU_NUM", "1")
GPU_NAME = "unknown"

# 平台读取的配置对象（字段名是平台约定，保持与同事模板一致）。
# 必须有 cfg 名：平台 `from AIFlow_config import cfg as conf`。
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
__C.TRAIN.CMD = _build_cmd()
__C.CMD = __C.TRAIN.CMD


def AutoTrain(**params) -> bool:
    """平台评测入口：重建命令（userPath 解析）→ 写日志 → 阻塞到评测结束。"""
    if not os.path.isdir("/mnt/c2-worldmodel"):
        print("[LeapBotCE eval AIFlow] 错误: 评测节点未挂载 /mnt/c2-worldmodel"
              "（checkpoint 与结果路径都在该共享盘）")
        return False
    workdir = _repo_workdir()
    if not (workdir / SWEEP_REL).is_file():
        print(f"[LeapBotCE eval AIFlow] 错误: 工作目录 {workdir} 下找不到 {SWEEP_REL}，"
              "请确认申请资源时工作目录选到 FastWAM 仓库根目录")
        return False

    cmd = _build_cmd()
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    log_dir = os.environ.get("JinnTrainResult")
    print(f"[LeapBotCE eval AIFlow] GPU_NUM={GPU_NUM} CMD={cmd}", flush=True)
    try:
        if log_dir:
            log_path = Path(log_dir) / "train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(cmd, shell=True, stdout=stream,
                                        stderr=stream, text=True)
                while proc.poll() is None:
                    time.sleep(10)
        else:
            proc = subprocess.Popen(cmd, shell=True)
            while proc.poll() is None:
                time.sleep(10)
        print(f"[LeapBotCE eval AIFlow] 评测退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001 - 平台入口不许抛
        print(f"[LeapBotCE eval AIFlow] 评测失败: {exc}")
        return False
