"""AIFlow 评测变体：codec A3（grid 挑选，tied 底座）终点 ckpt 标准规模消融评测。

任务名含 eval + libero + a3 → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a3

内容 = AIFlow_config_eval_libero 的标准规模消融参数钉死版：
libero_spatial × D{0,10,20} × 50 trials，ckpt = codec A3 线 24951 终点。
⚠️ A3 训练预计 10-13 ~16:00 收尾——提交前确认
checkpoints/weights/step_024951.pt 已落盘（preflight 会 fast-fail）。
薄委托：只设 AI_FLOW_* env 默认值，其余接口（cfg / AutoTrain / 命令拼装）
全部转发基座，基座改动自动生效。
"""

import os
import sys

# 变体文件与基座同目录；保险起见把自身目录补进 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─── 常改区（钉死本变体的评测对象；平台注入同名 env 时以 env 为准） ───
os.environ.setdefault(
    "AI_FLOW_RUN_DIR",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/"
    "codec_a3_grid_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/"
    "codec_a3_grid_seed42_h200/checkpoints/weights/step_024951.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a3_grid_step24951")
os.environ.setdefault("AI_FLOW_SUITES", "libero_spatial")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20")
os.environ.setdefault("AI_FLOW_TRIALS", "50")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
