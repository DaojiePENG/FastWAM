"""AIFlow 平台评测入口：a6 训练消融 —— singleage 单行「等到终点 ckpt 再评」。

与 `..._trainablations_edgeonly` 成对：只跑 singleage 一行
（`LEAPBOTCE_SKIP_EDGEONLY=1`），两张单各占 4 卡并行，避免串行合并单把
singleage 的评测拖到 edgeonly 跑完之后。

时间线（11-05 18:46 实测外推）：
    singleage 训练 ≈11-06 13:25 写 step_032544.pt → 评测 12–13 h → 11-07 ≈02:00
本单会在 pod 起来后一直占 4 卡等到那一刻（等待期 GPU 空转，这是无人值守的代价）。

契约与普通变体一致：TAG `a6_singleage_step32544`、delays `0 10 20 40`、
trials 50、PACK=1、无额外 override（训练目标形态不影响部署接口）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("AI_FLOW_PACK", "1")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402

WAIT_REL = os.path.join("scripts", "eval_a6_trainablations_wait.sh")


def _repo_workdir():
    from pathlib import Path

    user_path = os.environ.get("userPath")
    if user_path and (Path(user_path) / WAIT_REL).is_file():
        return Path(user_path)
    return Path(__file__).resolve().parent


def _build_wait_cmd() -> str:
    import shlex

    return (f"cd {shlex.quote(str(_repo_workdir()))} && "
            f"LEAPBOTCE_SKIP_EDGEONLY=1 bash {shlex.quote(WAIT_REL)}")


__C.TRAIN.CMD = _build_wait_cmd()
__C.CMD = __C.TRAIN.CMD


def AutoTrain(**params) -> bool:  # noqa: ARG001 - 平台签名固定
    """平台入口：不校验 ckpt 是否存在（本变体的意义就是等它出现）。"""
    import subprocess
    import time
    from pathlib import Path

    if not os.path.isdir("/mnt/c2-worldmodel"):
        print("[LeapBotCE a6-singleage-wait eval AIFlow] 错误: 未挂载 /mnt/c2-worldmodel")
        return False
    workdir = _repo_workdir()
    if not (workdir / WAIT_REL).is_file():
        print(f"[LeapBotCE a6-singleage-wait eval AIFlow] 错误: {workdir} 下找不到 {WAIT_REL}")
        return False

    cmd = _build_wait_cmd()
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    log_dir = os.environ.get("JinnTrainResult")
    print(f"[LeapBotCE a6-singleage-wait eval AIFlow] GPU_NUM={os.environ.get('GPU_NUM', '?')} "
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
        print(f"[LeapBotCE a6-singleage-wait eval AIFlow] 退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001 - 平台入口不许抛
        print(f"[LeapBotCE a6-singleage-wait eval AIFlow] 失败: {exc}")
        return False
