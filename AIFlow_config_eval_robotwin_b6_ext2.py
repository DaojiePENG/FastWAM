"""AIFlow 评测变体：RoboTwin D60/D80 时延探针 —— 串号修复后重跑（ours B6 侧）。

任务名含 eval + robotwin + b6 + ext2 → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_robotwin_b6_ext2

为什么重跑（2026-11-08 定位）：
    2026-10-31 的 ext 轮探针本身跑得没问题（delay_log 实证 D60/D80 真采样、
    ckpt=step_158206 与主根同一文件），坏的是 lane 结果拷贝：旧 lane 用
    lane 自己的时间戳 ${RUN_TS} 去 glob evaluate_results，多 lane 同时起跑时
    时间戳相撞，head -1 取到了**兄弟 lane 的 run 目录**，把别的任务结果抄进
    本格 `_result_clean.txt`。ext 轮 20 格中 13 格被污染（cell 0.98 vs 本格
    log 0.10 这类），所以 98.0/99.0 是假数，不是模型行为异常。

修复（v2 管线，本变体专用）：
    - lane: baseline_delay/leapbotce_robotwin_sweep_lane_v2_2653853.sh
      → 从本格 log 的 "Data has been saved to <run_dir>/_result_clean.txt"
        取路径，不再 glob；取不到就判失败重试，绝不回退 glob。
      → 旧脚本已归档 .bak_20261108（未删）。
    - 入口: scripts/eval_leapbotce_robotwin_sweep_v2_2653853.sh（薄 wrapper，
      其余一切转发原驱动）。
    - 结果根独立：robotwin_v2_sweep_b6_ext2 —— 旧 ext 根的 .done/_result_clean
      一律不复用，保证每格都是本轮新产物。

冻结口径（与 v2 主根严格可比）：
  - 评测对象 = B6 v2 同一 run/ckpt（dtrain20 seed42 final，LEAPBOTCE_CKPT=latest）；
    10-31 实跑落在 step_158206.pt
  - 任务 = v1 既有 10 任务子集（dtrain20 评测口径，v2 全量表的前 10）
  - 时延窗 D{60,80}，50 trials/格，replan 24（chunk 24），其余与 b6v2 逐项一致
  - 对照窗 D{0,10,20,40} 不重跑，取主根同 10 任务子集
薄委托：只设 AI_FLOW_* env 默认值，其余接口全部转发基座。
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
    "leapbotce/robotwin_v2_sweep_b6_ext2",
)
os.environ.setdefault("AI_FLOW_DELAYS", "60 80")  # 探针两窗；对照窗取既有数据
os.environ.setdefault("AI_FLOW_EVAL_TASK", "robotwin_all_leapbotce_codec_lora")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size "
    "click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block "
    "handover_mic",
)
# 结果拷贝修复版入口（v2 lane，见模块 docstring）
os.environ.setdefault(
    "AI_FLOW_SWEEP_REL", "eval_leapbotce_robotwin_sweep_v2_2653853.sh"
)

from AIFlow_config_eval_robotwin import *  # noqa: F401,F403,E402
from AIFlow_config_eval_robotwin import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_robotwin import AutoTrain, cfg  # noqa: E402
