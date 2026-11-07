"""AIFlow 评测变体：Table 4「constant cloud tokens (eval-time)」行全量 delay 扫描。

任务名含 eval + libero + a6 + consttoken → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_consttoken

回答问题（CVPR2027 模拟审稿 Concern 2 / P0「Constant Cloud Tokens：模型是不是
学会忽略云端」）：同一张 deployed a6 ckpt，eval 期把云端 planning tokens 换成
**同长度全零**张量——注意力槽位、mask、每步算力与真实载荷完全一致，只是槽位
里没有任何信息。若本行 ≈ a6 主行 → 模型本来就不看云端；若明显掉 → 云端 token
内容是 load-bearing。

与 nocloudcache 行的区别（写论文时不可混）：nocloudcache 是**零长度** cache
（云端槽位整体消失，连算力都变了）；本行保留槽位只清内容，因此能把
"有没有云端信息"与"多了几个 token 的容量/算力"分开。

协议 = a6 主行完全一致的 4 套件 × D{0,10,20,40} × 10 任务 × 50 trials
（10-15 准则）。TAG 独立（a6_lora_fs_step32544_consttoken），不与任何行混结果根。

卡型兼容：PACK=1 钉死（每 worker ~25GB）；GPU_NUM 不钉（提交卡数即 worker 数）。
云端前向照常执行（只改 token 内容），时长应与 a6 主行同量级（约 8h / 4 worker）。

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
    "codec_a6_lora_fs_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_consttoken")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")
# Table 4 行合同：同 ckpt、唯一差异 = 云 token 内容置零（长度不变）。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.cloud_cache_mode=constant")
# AI_FLOW_GPU_NUM 不设：提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
