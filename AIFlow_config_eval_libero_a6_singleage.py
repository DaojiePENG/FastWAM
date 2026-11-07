"""AIFlow 评测变体：Table 4「single-branch random-age」行的全量 delay 扫描。

任务名含 eval + libero + a6 + singleage → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_singleage

评测对象 = 训练变体 `AIFlow_config_libero_a6_singleage.py` 的终点 ckpt
（runs/libero_all_leapbotce_codec_lora/codec_a6_singleage_15ep_seed42_h200，
step_032544.pt）。该 run 与 a6 主行唯一差别是训练目标形态：
paired（fresh+stale 双支、λ 混合）→ single_random（单支随机龄、单损失）。

评测不需要额外 override：推理路径与 a6 完全相同（正常 cloud cache，edge 看
当前帧）；paired_age_mode 只影响 training_loss，ckpt meta 里已记录供溯源。

⚠️ 口径：与「w/o stale loss（fresh-only）」不是一回事——那一行只有 d0 监督，
本行是只有随机龄监督（龄 U{0..20}，含低龄样本但没有第二条 always-fresh 监督）。
两行 + a6 主行三点连线，才能回答审稿人 Concern 4 的"Paired-Age 独特价值"。

协议 = a6 主行完全一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials。
TAG 独立（a6_singleage_step32544）。PACK=1 钉死；GPU_NUM 不钉。

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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_singleage_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_singleage_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_singleage_step32544")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")
# 无额外 override：推理路径与 a6 相同（训练目标形态不影响部署接口）。
# AI_FLOW_GPU_NUM 不设：提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
