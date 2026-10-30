"""AIFlow 评测变体：RoboTwin D60/D80 时延探针（ours B6 侧，10-31 用户拍板）。

任务名含 eval + robotwin + b6 + ext → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_robotwin_b6_ext

背景（2026-10-31）：RoboTwin v2 四窗基准（§5/论文 Table 3）里 StarWAM-MoT
保留率 80.4% 与 ours 87.0% 接近——机制上 StarWAM 的 chunk 保护（~16.6
replans/ep vs 逐步法的 ~450，27× 曝光差）在 D≤40 段内尚未被击穿。本探针把
时延推出 chunk 保护段（D60/D80 ≫ ours chunk 24 步），检验两法分离是否
重现（LIBERO 上 D40 已现 67.8pp 分离）。预测：StarWAM 加速塌陷、ours 平缓
衰减。探针结论若显著分离 → 再谈全量 50 任务补窗；否则论文叙事不动。

冻结口径（与既有 v2 数据严格可比）：
  - 评测对象 = B6 v2 同一 run/ckpt（dtrain20 seed42 final，LEAPBOTCE_CKPT=latest）
  - 任务 = v1 既有 10 任务子集（dtrain20 评测口径，v2 全量表的前 10）
  - 结果根独立：robotwin_v2_sweep_b6_ext（防污染 v2 主根的 .done 幂等/聚合）
  - 对照组不重跑：D{0,10,20,40} 直接从既有 v2 per-task 结果取同 10 任务
    子集算受限 macro（聚合/对比脚本另出）。
协议其余与 b6v2 逐项一致（50 trials/格、codec_lora 配方、PACK 泳道）。
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
    "leapbotce/robotwin_v2_sweep_b6_ext",
)
os.environ.setdefault("AI_FLOW_DELAYS", "60 80")  # 探针两窗；对照窗取既有数据
os.environ.setdefault("AI_FLOW_EVAL_TASK", "robotwin_all_leapbotce_codec_lora")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # v1 既有 10 任务子集（dtrain20 评测口径；与 StarWAM 探针侧同表）
    "adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size "
    "click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block "
    "handover_mic",
)

from AIFlow_config_eval_robotwin import *  # noqa: F401,F403,E402
from AIFlow_config_eval_robotwin import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_robotwin import AutoTrain, cfg  # noqa: E402
