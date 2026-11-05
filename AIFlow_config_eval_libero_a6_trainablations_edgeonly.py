"""AIFlow 平台评测入口：a6 训练消融 —— edgeonly 单行「等到终点 ckpt 再评」。

与 `AIFlow_config_eval_libero_a6_trainablations.py`（两行串行的合并单）只差
一点：本变体只跑 edgeonly 一行（`LEAPBOTCE_SKIP_SINGLEAGE=1`），可以和
`..._trainablations_singleage` 同时提交、各占 4 卡并行，省掉串行等待。

为什么拆两张单（2026-11-05 用户拍板）：
    串行合并单 4 卡：edgeonly 08:00→20:30，singleage 20:30→次日 09:00
    两张单各 4 卡：  edgeonly 08:00→20:30，singleage 13:25→次日 02:00（并行）
单张 8 卡更快（11-06 ≈21:00）但 PACK=1 时 8 worker × 25–30 GB ≈ 200–240 GB
会在部分 H100 节点顶到容器内存墙静默 SIGABRT（10-19 事故），无人值守不赌。

契约与普通变体一致：TAG `a6_edgeonly_step32544`、delays `0 10 20 40`、
trials 50、PACK=1、`model.cloudedge.use_cloud_cache=false`。
结果根与普通单行变体重合，后补提交会被 manager 跳过已有结果，不会重复跑。
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
            f"LEAPBOTCE_SKIP_SINGLEAGE=1 bash {shlex.quote(WAIT_REL)}")


__C.TRAIN.CMD = _build_wait_cmd()
__C.CMD = __C.TRAIN.CMD


def AutoTrain(**params) -> bool:  # noqa: ARG001 - 平台签名固定
    """平台入口：不校验 ckpt 是否存在（本变体的意义就是等它出现）。"""
    import subprocess
    import time
    from pathlib import Path

    if not os.path.isdir("/mnt/c2-worldmodel"):
        print("[LeapBotCE a6-edgeonly-wait eval AIFlow] 错误: 未挂载 /mnt/c2-worldmodel")
        return False
    workdir = _repo_workdir()
    if not (workdir / WAIT_REL).is_file():
        print(f"[LeapBotCE a6-edgeonly-wait eval AIFlow] 错误: {workdir} 下找不到 {WAIT_REL}")
        return False

    cmd = _build_wait_cmd()
    __C.TRAIN.CMD = cmd
    __C.CMD = cmd
    log_dir = os.environ.get("JinnTrainResult")
    print(f"[LeapBotCE a6-edgeonly-wait eval AIFlow] GPU_NUM={os.environ.get('GPU_NUM', '?')} "
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
        print(f"[LeapBotCE a6-edgeonly-wait eval AIFlow] 退出，returncode={proc.returncode}")
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001 - 平台入口不许抛
        print(f"[LeapBotCE a6-edgeonly-wait eval AIFlow] 失败: {exc}")
        return False
