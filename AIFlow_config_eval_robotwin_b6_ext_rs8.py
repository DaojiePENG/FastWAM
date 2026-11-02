"""AIFlow 评测变体：RoboTwin chunk 剂量-响应探针（ours B6 侧，replan_steps=8）。

任务名含 eval + robotwin + b6 + ext + rs8 → 本文件（分发器最长后缀匹配，
5 词后缀压过 b6_ext 4 词），例如：
    leapbot_ce_eval_robotwin_b6_ext_rs8

背景（2026-10-31 用户拍板）：D60/D80 探针是把时延幅度推出 chunk 保护段；
本变体是另一半 —— 固定 D、把 chunk 执行步数 24→8，delay 采样频率 ×3
（~19 → ~56 replans/集），与 StarWAM 侧 starwam_ext_rs8 成对，画出
retention vs replan 频率的剂量曲线。判读：
  - ours 曲线平（预期，LIBERO 每 10 步高频先例 77.7%）+ StarWAM 曲线陡
    → "训练机制决定时延鲁棒性"最强证据；
  - 两条曲线都平 → chunk 保护确实只延后（§8.4 预测）；
  - StarWAM 反而更稳 → 混杂效应主导（短 chunk 观测更新增益 > 曝光惩罚），
    现有叙事需修正，诚实报告。
混杂必须披露：短 chunk 同时带来观测新鲜度增益（D0 可能升）与 delay 采样
×3 惩罚；retention 为比值可部分归一。模型与 delay 协议零改动 —— 只改
执行策略（EVALUATION.replan_steps，lane 脚本 LEAPBOTCE_REPLAN_STEPS 透传）。

冻结口径：
  - 评测对象 = B6 v2 同一 run/ckpt（LEAPBOTCE_CKPT=latest）
  - 任务 = v1 既有 10 任务子集（与 D60/D80 探针同表）
  - 窗 = D{0,40} 受限版（D40 是 retention 主读数；中段窗后补不迟）
  - 结果根独立：robotwin_v2_sweep_b6_ext_rs8（防污染 ext 根与 v2 主根）
  - chunk=24 的对照 = 既有 v2 per-task 数据，不重跑。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
    "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
)
os.environ.setdefault("AI_FLOW_CKPT", "latest")
os.environ.setdefault(
    "AI_FLOW_SWEEP_ROOT",
    "/mnt/c2-worldmodel/2653853/baseline-evaluations/"
    "leapbotce/robotwin_v2_sweep_b6_ext_rs8",
)
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40 60 80")  # 11-02 用户指示扩全六窗; 既有 D{0,40} .done 幂等跳过
os.environ.setdefault("AI_FLOW_REPLAN_STEPS", "8")  # 剂量变体本体
os.environ.setdefault("AI_FLOW_EVAL_TASK", "robotwin_all_leapbotce_codec_lora")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # v1 既有 10 任务子集（与 D60/D80 探针同表）
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size "
    "click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block "
    "handover_mic",
)

from AIFlow_config_eval_robotwin import *  # noqa: F401,F403,E402
from AIFlow_config_eval_robotwin import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_robotwin import AutoTrain, cfg  # noqa: E402
