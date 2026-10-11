"""AIFlow 平台评测入口：LeapBotCE (ours) LIBERO 云延迟 sweep（h200 集群训练轮）。

═══════════════════════════════════════════════════════════════════════
使用方法（公司自动申请显卡评测）—— dispatcher 时代：本文件常驻不改名
═══════════════════════════════════════════════════════════════════════
本树根 `AIFlow_config.py` 已是常驻 dispatcher（10-10 改造）：任务名含
eval+libero（或 bench+libero 同义归一）即自动路由到本文件，**无需任何改名/
覆盖操作**。提交 = AIFlow 建任务，任务名如 `leapbot_ce_eval_libero`，
工作目录（userPath）选 FastWAM 仓库根。

1. 申请评测资源时，工作目录（userPath）选到本文件所在目录（FastWAM 仓库根）。
   平台约定：工作目录下必须有名为 AIFlow_config.py 的脚本，平台 import 它、
   以 `cfg` 读取 __C.TRAIN.CMD 作为命令（非空 CMD → CUSTOM_CMD 分支直接 shell 执行）。
2. 评测命令统一走 scripts/eval_leapbotce_libero_sweep_2653853.sh：
   自动探测平台分配的卡、**打包调度**（(delay×suite) 格 × 每格 1 个 manager，
   一卡共存 PACK 个 worker——H200 141G 自动 4、H800 80G 自动 2，各 ~25GB；
   `LIBERO_LIBERO_PACK=1` 回退单 manager 全卡串行模式）、
   D ∈ {0,5,10,20,40} × 4 套件 × 10 任务 × 50 trials、崩溃重试、幂等续跑；
   结果落公共盘
   /mnt/c2-worldmodel/.../baseline-evaluations/leapbotce/dtrain20_seed42_110k/。
   预计时长：16×H200 ≈ 2.5-3h；8×H200 ≈ 6h；8×H800 ≈ 12h（详见
   daojie_leap_configs/leapbot_ce_eval_libero.md §3.0 卡数表）。
3. smoke（强烈建议先跑一次验证 pod 环境，GL/mujoco 问题在 preflight 即报）：
   把下面 AI_FLOW_TAG 设成 smoke 专用 tag、AI_FLOW_TRIALS=2、
   AI_FLOW_SUITES=libero_spatial、AI_FLOW_DELAYS="0"，
   即 10 任务 × 2 trials = 20 episodes，几分钟内出结果。⚠️ smoke 必须用独立
   TAG —— manager 会跳过已有结果的任务，写进正式 tag 会污染正式 sweep。
   验证通过后清空这些开关再全量提交（smoke tag 留着不影响正式结果）。
4. checkpoint：默认 LEAPBOTCE_CKPT=latest（200k 线 run dtrain20_seed42_200k
   下最新 step_*.pt = step_110000.pt；该线 09-29 中止于 111,470 步，若日后
   续训到 200k，latest 自动升级到新 ckpt —— 届时**必须换新 TAG**，否则新旧
   ckpt 结果混在同一 tag 里）。也可经 AI_FLOW_CKPT 指定具体某个。

与 AIFlow_config_eval_robotwin.py（RoboTwin 评测行）同构；与
fastwam_official/AIFlow_config.py（Fast-WAM baseline 行）互不相干。

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
# smoke 开关：空串 = 全量（5 延迟 × 4 套件 × 10 任务 × 50 trials）。
# smoke 示例：AI_FLOW_TAG="smoke_h200" AI_FLOW_TRIALS="2"
#             AI_FLOW_SUITES="libero_spatial" AI_FLOW_DELAYS="0"
# ⚠️ smoke 必须用独立 TAG（manager 跳过已有结果的任务，会污染正式 sweep）。
AI_FLOW_DELAYS = os.environ.get("AI_FLOW_DELAYS", "").strip()
AI_FLOW_TAG = os.environ.get("AI_FLOW_TAG", "").strip()
AI_FLOW_SUITES = os.environ.get("AI_FLOW_SUITES", "").strip()
AI_FLOW_TRIALS = os.environ.get("AI_FLOW_TRIALS", "").strip()
# checkpoint："latest" = RUN_DIR 下最新 step_*.pt；或具体 ckpt 绝对路径
AI_FLOW_CKPT = os.environ.get("AI_FLOW_CKPT", "latest")
# 训练 run 目录：评测从这读 config.yaml / dataset_stats.json / latest ckpt。
# 默认 = 200k 原目录（与 sweep 脚本默认一致）。评 200k 续训结果或 codec
# 消融线时必须改这里，例如：
#   .../runs/libero_all_leapbotce/dtrain20_seed42_200k_resume   （200k 续训结果）
#   .../runs/libero_all_leapbotce_codec/codec_a1_seed42_h200    （A1 消融线）
AI_FLOW_RUN_DIR = os.environ.get("AI_FLOW_RUN_DIR", "").strip() or (
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/"
    "dtrain20_seed42_200k")
# eval-time hydra overrides 透传（A4 消融 =
# model.cloudedge.edge_prefix_enabled=false，必须换独立 TAG）。
AI_FLOW_EVAL_EXTRA = os.environ.get("AI_FLOW_EVAL_EXTRA", "").strip()

SWEEP_REL = Path("scripts") / "eval_leapbotce_libero_sweep_2653853.sh"

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
        smoke += f" LIBERO_EVAL_DELAYS={_q(AI_FLOW_DELAYS)}"
    if AI_FLOW_TAG:
        smoke += f" LIBERO_EVAL_TAG={_q(AI_FLOW_TAG)}"
    if AI_FLOW_SUITES:
        smoke += f" LIBERO_EVAL_SUITES={_q(AI_FLOW_SUITES)}"
    if AI_FLOW_TRIALS:
        smoke += f" NUM_TRIALS={_q(AI_FLOW_TRIALS)}"
    if AI_FLOW_EVAL_EXTRA:
        smoke += f" LIBERO_EVAL_EXTRA={_q(AI_FLOW_EVAL_EXTRA)}"
    return (
        f"cd {_q(_repo_workdir())} && {exports} "
        f"LEAPBOTCE_CKPT={_q(AI_FLOW_CKPT)} RUN_DIR={_q(AI_FLOW_RUN_DIR)}"
        f"{smoke} "
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
        print("[LeapBotCE LIBERO eval AIFlow] 错误: 评测节点未挂载 /mnt/c2-worldmodel"
              "（checkpoint 与结果路径都在该共享盘）")
        return False
    workdir = _repo_workdir()
    if not (workdir / SWEEP_REL).is_file():
        print(f"[LeapBotCE LIBERO eval AIFlow] 错误: 工作目录 {workdir} 下找不到 {SWEEP_REL}，"
              "请确认申请资源时工作目录选到 FastWAM 仓库根目录")
        return False

    cmd = _build_cmd()
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    log_dir = os.environ.get("JinnTrainResult")
    print(f"[LeapBotCE LIBERO eval AIFlow] GPU_NUM={GPU_NUM} CMD={cmd}", flush=True)
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
        print(f"[LeapBotCE LIBERO eval AIFlow] 评测退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001 - 平台入口不许抛
        print(f"[LeapBotCE LIBERO eval AIFlow] 评测失败: {exc}")
        return False
