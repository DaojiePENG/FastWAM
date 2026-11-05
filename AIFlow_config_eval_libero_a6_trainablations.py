"""AIFlow 平台评测入口：a6 训练消融两行「等到终点 ckpt 再评」的合并单。

用途（2026-11-05 建，用户出差期间无人值守）：
`tab:ablations` 训练块还差两行 —— `w/o cloud (training)`（codec_a6_edgeonly_15ep_
seed42_h200）与 `single-branch random-age`（codec_a6_singleage_15ep_seed42_h200）。
两条训练线分别约 11-06 08:00 / 13:25 才写 step_032544.pt，而普通变体配置
（AIFlow_config_eval_libero_a6_edgeonly.py / ..._singleage.py）在 import 期就会
拒绝不存在的 ckpt。本变体把「等 ckpt + 连跑两次四窗评测」封装成一个 CMD，
提交一次即可无人值守跑完全部：

    bash scripts/eval_a6_trainablations_wait.sh

契约（与两个普通变体逐项一致，改动须同步）：
    TAG            a6_edgeonly_step32544 / a6_singleage_step32544
    delays         0 10 20 40      （10-15 准则）
    trials         50              （4 套件 × 10 任务 × 50 = 2000 集/窗）
    PACK           1               （每 worker ~25GB，4 卡 4 worker）
    edgeonly extra model.cloudedge.use_cloud_cache=false
    singleage      （无；训练目标形态不影响部署接口）

资源与时长：4×H100 足够（PACK=1）；等待时间取决于训练进度（最长 36h，
脚本内 LEAPBOTCE_WAIT_MAX_H 可调），两次四窗评测各 ~12–13h 实测。
等待期间 GPU 空闲（训练线在别的 pod），这是无值守窗口的代价，可接受。

⚠️ 只有提交卡数 = 期望 worker 数。GPU_NUM 由平台注入，脚本沿用。
⚠️ 结果根 TAG 与普通变体相同：若之后又单独提交普通变体，manager 会跳过
已有结果，不会重复跑，也不会混来源。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# PACK=1 钉死，防节点漂移；等待脚本自己也显式传值，两处一致即可。
os.environ.setdefault("AI_FLOW_PACK", "1")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402

WAIT_REL = os.path.join("scripts", "eval_a6_trainablations_wait.sh")


def _repo_workdir():
    """运行目录：平台给的 userPath（须含本树 scripts/），否则本文件所在目录。"""
    from pathlib import Path

    user_path = os.environ.get("userPath")
    if user_path and (Path(user_path) / WAIT_REL).is_file():
        return Path(user_path)
    return Path(__file__).resolve().parent


def _build_wait_cmd() -> str:
    import shlex

    workdir = _repo_workdir()
    return f"cd {shlex.quote(str(workdir))} && bash {shlex.quote(WAIT_REL)}"


# 覆盖基座在 import 期按 AI_FLOW_* 默认值拼出的 CMD（那份指向 200k 线的 latest）。
__C.TRAIN.CMD = _build_wait_cmd()
__C.CMD = __C.TRAIN.CMD


def AutoTrain(**params) -> bool:  # noqa: ARG001 - 平台签名固定
    """平台入口：不校验 ckpt 是否存在（本变体的全部意义就是等它出现）。"""
    import subprocess
    import time
    from pathlib import Path

    if not os.path.isdir("/mnt/c2-worldmodel"):
        print("[LeapBotCE a6-trainablations eval AIFlow] 错误: 评测节点未挂载 "
              "/mnt/c2-worldmodel（ckpt 与结果路径都在该共享盘）")
        return False
    workdir = _repo_workdir()
    if not (workdir / WAIT_REL).is_file():
        print(f"[LeapBotCE a6-trainablations eval AIFlow] 错误: 工作目录 {workdir} 下"
              f"找不到 {WAIT_REL}，请确认申请资源时工作目录选到 FastWAM 仓库根目录")
        return False

    cmd = _build_wait_cmd()
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    log_dir = os.environ.get("JinnTrainResult")
    print(f"[LeapBotCE a6-trainablations eval AIFlow] GPU_NUM={os.environ.get('GPU_NUM', '?')} "
          f"CMD={cmd}", flush=True)
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
        print(f"[LeapBotCE a6-trainablations eval AIFlow] 退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001 - 平台入口不许抛
        print(f"[LeapBotCE a6-trainablations eval AIFlow] 失败: {exc}")
        return False
