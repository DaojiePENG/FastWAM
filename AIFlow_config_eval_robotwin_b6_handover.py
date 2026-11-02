"""AIFlow 评测变体：handover_block 跨批次矛盾复跑（当前代码态定性）。

任务名含 eval + robotwin + b6 + handover → 本文件（分发器最长后缀匹配），
例如：
    leapbot_ce_eval_robotwin_b6_handover

背景（2026-11-02）：D60/D80 探针收数后发现 handover_block 跨批次矛盾 ——
v1 根（10-22 批）D20=68/D40=22 与 v2 主根（10-26 批）D20=58/D40=24 两批
独立复现塌方，而 ext 轮（10-31 批，curobo 四架构重编译后）D60/D80=98/98
是孤例。裸 b6 快捷键指 v1 旧根（40 格 .done 全在 → 全 skip 秒退，单
22807730 未复跑任何东西），故立本变体在**当前代码态**复跑定性。

冻结口径：
  - 评测对象 = B6 v2 同一 ckpt（显式 step_158206.pt，防 latest 漂移）
  - 任务 = handover_block（矛盾本体）+ beat_block_hammer（锚任务：三方
    读数连续 80-98，校准"当前代码态 vs 历史批次"的可比性）
  - 窗 = D{20,40}（塌方谷区）
  - 结果根独立：robotwin_sweep_b6_handover_recheck（不碰 v1/v2/ext 根）
  - 2 任务 × 2 窗 = 4 格，4 卡 ~2-2.5h

判读矩阵（三方对照，结果落 docs §7 + §8.4 更新）：
  - handover@D40 ≈ 22-24 且锚 ≈ 86-90 → 主根可信，ext 轮 handover 格当轮
    异常；§8.4 按 v1+v2 口径更新，ext 探针 ours 侧 macro 剔除该格。
  - handover@D40 ≈ 98 且锚 ≈ 86-90 → handover 对 delay 响应真实非单调
    （V 形，D40 谷底），机制待解并 §8.4 披露。
  - handover ≈ 98 且锚也漂（>95 或 <70）→ 当前代码态整体漂移（curobo
    重编译嫌疑升级），ext 探针 ours 侧 40 格需重测。
协议与 b6v2 逐项一致（50 trials/格、codec_lora 配方、PACK 泳道）。
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
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
    "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3"
    "/checkpoints/weights/step_158206.pt",
)
os.environ.setdefault(
    "AI_FLOW_SWEEP_ROOT",
    "/mnt/c2-worldmodel/2653853/baseline-evaluations/"
    "leapbotce/robotwin_sweep_b6_handover_recheck",
)
os.environ.setdefault("AI_FLOW_DELAYS", "20 40")  # 塌方谷区两窗
os.environ.setdefault("AI_FLOW_EVAL_TASK", "robotwin_all_leapbotce_codec_lora")
os.environ.setdefault(
    "AI_FLOW_TASKS",
    # 矛盾本体 + 三方连续锚任务（校准当前代码态与历史批次可比性）
    "handover_block beat_block_hammer",
)

from AIFlow_config_eval_robotwin import *  # noqa: F401,F403,E402
from AIFlow_config_eval_robotwin import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_robotwin import AutoTrain, cfg  # noqa: E402
