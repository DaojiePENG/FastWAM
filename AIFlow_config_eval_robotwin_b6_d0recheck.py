"""AIFlow 评测变体：RoboTwin v2 native D0 波次 —— 串号污染格重跑（ours B6 侧）。

任务名含 eval + robotwin + b6 + d0recheck → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_robotwin_b6_d0recheck

为什么重跑（2026-11-08 定位）：
    v2 主根 robotwin_v2_sweep_b6 的 delay_0 有 **20/50 格**的 `_result_clean.txt`
    与本格自己 log 指向的 run 目录不符（cell 全部 = 0.98，本格 log/run 目录则是
    0.62–1.00 各不相同）。根因是旧 lane 用 lane 时钟 ${RUN_TS} 去 glob
    evaluate_results，多 lane 同时起跑时时间戳相撞，head -1 抄到了**兄弟 lane
    那一格的第一份结果**（同一个 0.98 抄了 20 格）。D10/D20/D40 未受影响（起跑
    更分散），200 格里剩下的 180 格自洽。
    受影响格按 (cell, 本格 log run 目录) 逐格列出，见本文件 AI_FLOW_TASKS。

影响面：Table 3 的 ours native 行 D0 曾按 92.92% 记账，去污染后为 86.96%
（D10/D20/D40 = 85.84/85.16/80.88 不变）→ retention 87.0% 会变 93.0%。
本次重跑目的就是把这 20 格用干净管线重出一遍，冻结 D0 与 retention。

修复管线（本变体专用，其余一切转发基座）：
    - lane: baseline_delay/leapbotce_robotwin_sweep_lane_v2_2653853.sh
      → 结果路径取自本格 log 的 "Data has been saved to ..."，不 glob；
        取不到即判失败重试（绝不回退旧 glob）。旧脚本归档 .bak_20261108。
    - 入口: scripts/eval_leapbotce_robotwin_sweep_v2_2653853.sh（薄 wrapper）。
    - 结果根独立：robotwin_v2_sweep_b6_d0recheck（不复用主根 .done）。

口径：D0 单窗 × 20 任务 × 50 trials，replan 24（chunk 24），ckpt=latest
（与 v2 主根同一 run，10-27 实跑为 step_158206.pt 系列），EVAL_TASK=codec_lora
配方。判读：把 20 格新值与「本格 log/run 目录复算值」逐格对照，一致即入表。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─── 常改区（钉死本变体的评测对象；平台注入同名 env 时以 env 为准） ───
os.environ.setdefault(
    "AI_FLOW_RUN_DIR",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
    "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
)
os.environ.setdefault("AI_FLOW_CKPT", "latest")
os.environ.setdefault(
    "AI_FLOW_SWEEP_ROOT",
    "/mnt/c2-worldmodel/2653853/baseline-evaluations/"
    "leapbotce/robotwin_v2_sweep_b6_d0recheck",
)
os.environ.setdefault("AI_FLOW_DELAYS", "0")  # 只重跑被污染的 D0 格
os.environ.setdefault("AI_FLOW_EVAL_TASK", "robotwin_all_leapbotce_codec_lora")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # 2026-11-08 逐格审计出的 20 个污染格（cell 值 vs 本格 log run 目录值）
    "blocks_ranking_rgb blocks_ranking_size dump_bin_bigbin handover_block "
    "hanging_mug lift_pot move_can_pot move_playingcard_away move_stapler_pad "
    "open_laptop open_microwave pick_diverse_bottles place_a2b_right "
    "place_bread_basket place_burger_fries place_can_basket "
    "place_cans_plasticbox place_empty_cup place_fan place_mouse_pad",
)
# 结果拷贝修复版入口（v2 lane，见模块 docstring）
os.environ.setdefault(
    "AI_FLOW_SWEEP_REL", "eval_leapbotce_robotwin_sweep_v2_2653853.sh"
)

from AIFlow_config_eval_robotwin import *  # noqa: F401,F403,E402
from AIFlow_config_eval_robotwin import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_robotwin import AutoTrain, cfg  # noqa: E402
