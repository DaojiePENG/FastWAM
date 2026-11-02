"""AIFlow 评测变体：Table 4「D_train=40」行 ep9 中点 checkpoint D0 快评。

任务名含 eval + libero + a6 + dtrain40 + ep9 → 本文件（分发器最长后缀匹配，
5 词后缀压过 dtrain40 与 ep9 各 4 词），例如：
    leapbot_ce_eval_libero_a6_dtrain40_ep9

背景（2026-11-02）：dtrain40 终点（step_032544）四窗全塌
（D0 22.95 / D10 18.00 / D20 15.20 / D40 13.15），异常排查已定性为
真实训练效应（eval 侧零事故：同 step 对照 a6_lora_fs D0=90.95、ckpt
完好、训练 cmd diff 仅 max_delay_steps=40、loss 收敛正常 0.0065）。
本快评用 ep9 中点 step_019530 只跑 D0 一窗，定位塌方时点：
  - ep9 D0 也 ~22 量级 → 训练早期即坏（数据/机制早期问题）；
  - ep9 D0 正常（母本 ep9 参照 ~85+）→ 训练后期塌（stale 模式过拟合），
    中段 ckpt 筛选可能有救（a6_lora_fs ep9-13 筛选先例同款流程）。
对照锚（同 run 同协议）：终点 D0=22.95；母本同 step a6_lora_fs
step_019530 D0 见 a6_lora_fs_step019530 根。判读阈值：ep9 与终点差
<2 点 = 训练早期即坏；差 >20 点 = 后期塌。

协议与 dtrain40 变体完全一致（4 套件 × 10 任务 × 50 trials），仅窗子集
与 ckpt 不同。AI_FLOW_PACK=1 + GPU_NUM=4：内存墙规避配置（10-14/10-19
实锤，见基座 docstring）。
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
    "codec_a6_dtrain40_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_dtrain40_15ep_seed42_h200/checkpoints/weights/step_019530.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_dtrain40_step019530")
os.environ.setdefault("AI_FLOW_DELAYS", "0")  # 快评一窗：定位塌方时点
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见基座 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 10-19 内存墙规避, 见基座 docstring

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
