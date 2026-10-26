"""AIFlow 平台评测入口：LeapBotCE (ours) RoboTwin 2.0 云延迟 sweep。

═══════════════════════════════════════════════════════════════════════
使用方法（公司自动申请显卡评测）—— 经树根 AIFlow_config.py dispatcher 路由
═══════════════════════════════════════════════════════════════════════
本文件以备用名 `AIFlow_config_eval_robotwin.py` 存放；树根的
`AIFlow_config.py` 是常驻 dispatcher（10-10 改造），任务名含 `eval`+`robotwin`
即自动路由到本文件（如 `leapbot_ce_eval_robotwin`），**无需任何换名 dance**：

    提交评测：任务名 leapbot_ce_eval_robotwin（或 …_smoke），userPath 选
    /dahuafs/userdata/2653853/code/FastWAM，8 卡起。
    B6 梯子 run 全量：任务名 leapbot_ce_eval_robotwin_b6 —— 自动切 B6
    codec_lora run + 独立结果根 robotwin_sweep_b6（默认值指向废 mot 基座的
    B 线 run + 共享 sweep 根，B6 严禁走默认），时延窗按 10-15 准则只跑
    {0,10,20,40}。

1. 平台约定：userPath 下必须有名为 AIFlow_config.py 的脚本，平台 import 它、
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
# 任务名拆单快捷键（优先于 AI_FLOW_DELAYS）：任务名含标记即锁定 delay 子集，
# 如 leapbot_ce_eval_robotwin_d0510 只跑 D∈{0,5,10}，…_d2040 只跑 {20,40}。
# 先剥掉平台追加的 "-<8hex>" 哈希尾巴再匹配，避免哈希里的 d5/d0 假命中。
_raw_name = os.path.basename(os.environ.get("JinnTrainResult", "")).lower()
_tail = _raw_name.rsplit("-", 1)[-1]
_TASK_NAME = _raw_name[:-len(_tail) - 1] if len(_tail) == 8 and all(c in "0123456789abcdef" for c in _tail) else _raw_name
for _tok, _sub in (("d0510", "0 5 10"), ("d2040", "20 40"),
                   ("d1020", "10 20"), ("d05", "0 5"),
                   ("d0", "0"), ("d5", "5"), ("d10", "10"),
                   ("d20", "20"), ("d40", "40")):
    if _tok in _TASK_NAME:
        AI_FLOW_DELAYS = _sub
        break
# B6（codec_lora 梯子 run）快捷键（10-22）：任务名含 "b6" 即切 run + 结果根。
# 默认值指向废 mot 基座的 B 线 run + 共享 sweep 根，B6 走默认 = 评错产物 +
# 与 B 线全量 .done 撞车（整根全 skip）。时延窗按 10-15 长期准则只跑
# {0,10,20,40}（B 线旧协议含 D5，不再跑）；名字里的 d40/d2040 等时延令牌
# 优先于本默认（上面的 loop 先行赋值，此处只补空）。
_B6_RUN_DIR = ("/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
               "robotwin_all_leapbotce_codec_lora/"
               "robotwin_all_codec_lora_dtrain20_seed42_gpu_n3")
_B6_SWEEP_ROOT = ("/mnt/c2-worldmodel/2653853/baseline-evaluations/"
                  "leapbotce/robotwin_sweep_b6")
_B6_EVAL_TASK = "robotwin_all_leapbotce_codec_lora"
AI_FLOW_RUN_DIR = os.environ.get("AI_FLOW_RUN_DIR", "").strip()
AI_FLOW_SWEEP_ROOT = os.environ.get("AI_FLOW_SWEEP_ROOT", "").strip()
AI_FLOW_EVAL_TASK = os.environ.get("AI_FLOW_EVAL_TASK", "").strip()
# 评测配方（lane 驱动的 LEAPBOTCE_EVAL_TASK，决定 hydra task= 即模型构建配方）。
# 10-26 事故：b6 全量首提 40 格全灭（71 min 快败）—— lane 缺省回 B 线
# recipe `robotwin_all_leapbotce`（edge_vision projector 为 plain Sequential），
# 加载 codec_lora ckpt（scale_k/scale_v/proj.*）必键不匹配。b6 分支强制
# 注入 codec_lora 配方；裸 B 线提交不受影响（默认即 B 线配方）。
if "b6" in _TASK_NAME:
    if not AI_FLOW_RUN_DIR:
        AI_FLOW_RUN_DIR = _B6_RUN_DIR
    if not AI_FLOW_SWEEP_ROOT:
        AI_FLOW_SWEEP_ROOT = _B6_SWEEP_ROOT
    if not AI_FLOW_DELAYS:
        AI_FLOW_DELAYS = "0 10 20 40"
    if not AI_FLOW_EVAL_TASK:
        AI_FLOW_EVAL_TASK = _B6_EVAL_TASK
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
    extra = ""
    if AI_FLOW_RUN_DIR:
        extra += f" LEAPBOTCE_RUN_DIR={_q(AI_FLOW_RUN_DIR)}"
    if AI_FLOW_SWEEP_ROOT:
        extra += f" LEAPBOTCE_SWEEP_ROOT={_q(AI_FLOW_SWEEP_ROOT)}"
    if AI_FLOW_EVAL_TASK:
        extra += f" LEAPBOTCE_EVAL_TASK={_q(AI_FLOW_EVAL_TASK)}"
    return (
        f"cd {_q(_repo_workdir())} && {exports} "
        f"LEAPBOTCE_CKPT={_q(AI_FLOW_CKPT)}{smoke}{extra} "
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
