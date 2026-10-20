"""AIFlow 评测变体：A7 fresh100 臂梯子 spatial 探针（step_002170）。

任务名 leapbot_ce_eval_libero_a1_ft_fresh100_p002170 → 本文件（分发器最长后缀匹配）。

用途 = 终点全量（_full 变体）判据不合（retention 崩 / D0 不达标）时定位
sweet spot 的备用梯子，可与全量并行提交（~1h/件）。
口径 = 仅 libero_spatial × D{0,20,40} × 25 trials（250 eps/档，a6 梯子同款；
全量口径才测 D10——10-15 准则 D{0,10,20,40} 适用于正式收割，探针不跑 D10）。
TAG 独立（a1_ft_fresh100_p002170）防与全量结果混 tag。CKPT 钉死具体步数
（探针必须可复现可比，不用 latest）。
AI_FLOW_PACK=1 + GPU_NUM=4（a6 实锤内存墙对策链：4 worker ≈100-120GB 稳，
8 worker 节点容器内存顶格集体 SIGABRT）。推荐单节点 4×H200——双节点会竞速双评
（a6 教训 10/160 需手动去重）。模型侧 model.* 全部由 RUN_DIR 的 config.yaml 推导
（a7 非 LoRA → hydra task 自动 libero_all_leapbotce_codec，与 a1 全量同路径；
delta ckpt 冻结基座从同一 mot release 重建）。薄委托：只设 AI_FLOW_* env 默认值，
其余接口全部转发基座，基座改动自动生效。
"""


import os
import sys

# 变体文件与基座同目录；保险起见把自身目录补进 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault(
    "AI_FLOW_RUN_DIR",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/codec_a1_ft_fresh100_seed42",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/codec_a1_ft_fresh100_seed42/checkpoints/weights/step_002170.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a1_ft_fresh100_p002170")
os.environ.setdefault("AI_FLOW_SUITES", "libero_spatial")
os.environ.setdefault("AI_FLOW_DELAYS", "0 20 40")
os.environ.setdefault("AI_FLOW_TRIALS", "25")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
