"""AIFlow 评测变体：deployed a6 行 checkpoint 筛选 stage-1（ep14)。

任务名含 eval + libero + a6 + ep14 → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a6_ep14

背景（2026-10-30 用户指示）：deployed a6 行（final step_032544, D0=91.0）的
D0 偏低，且 stale0 消融证明管线 D0 天花板 ~95（λmax=0 → 95.3）——中期
checkpoint 可能给出更好的 D0/retention 均衡。训练期无周期性 val success
（train.log 零 eval 行），只能真跑 eval 筛。

本变体 = a6_lora_fs run（codec_a6_lora_fs_15ep_seed42_h200）的 ep14 终点
step_030380，**只跑 D{0,40} 两窗**（stage-1 筛选：D40/D0 直接读 retention；
入选者由 stage-2 补 D{10,20} 成完整四窗，同 TAG 共用结果根，skip-existing
不冲突）。选择标准（先定死）：候选中取 D0 最高且 retention ≥ 0.76 者，
改善 <2 点不换（50 trials/格 macro 噪声 ±1-2 点）。

协议与 deployed a6 行完全一致（4 套件 × 10 任务 × 50 trials；本变体仅窗
子集不同）。AI_FLOW_PACK=1 + GPU_NUM=4：内存墙规避配置（10-14/10-19 实锤）。
薄委托：只设 AI_FLOW_* env 默认值，其余接口全部转发基座。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200/checkpoints/weights/step_030380.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step030380")
os.environ.setdefault("AI_FLOW_DELAYS", "0 40")  # stage-1 筛选两窗；stage-2 另单补 "10 20"
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见基座 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 10-19 内存墙规避, 见基座 docstring

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
