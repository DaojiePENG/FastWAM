"""AIFlow 评测变体：Table 4「shuffled cloud tokens (eval-time)」行全量 delay 扫描。

任务名含 eval + libero + a6 + shuffledtoken → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_shuffledtoken

回答问题（CVPR2027 模拟审稿 Concern 2 / P0「Shuffled Cloud Tokens：云端内容
是否真的语义相关」）：同一张 deployed a6 ckpt，eval 期把云端 planning tokens
的**序列位置打乱**（K/V 共用同一置换，键值配对不破坏）。token 的多重集、通道
统计、长度与算力全部与真实载荷相同，只有时序/空间对应关系被破坏。若本行与
a6 主行持平 → 云端只贡献"通用上下文能量"；若明显掉 → 云端内容的结构是
被真正消费的。

与 consttoken 的分工：consttoken 清空内容（信息=0），shuffledtoken 保留内容
统计但破坏结构——两条一起才能排除"模型只是需要一点非零信号"的解释。

协议 = a6 主行完全一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials。
TAG 独立（a6_lora_fs_step32544_shuffledtoken）。PACK=1 钉死；GPU_NUM 不钉。

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
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_shuffledtoken")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")
# Table 4 行合同：同 ckpt、唯一差异 = 云 token 位置置换（K/V 同序）。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.cloud_cache_mode=shuffled")
# AI_FLOW_GPU_NUM 不设：提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
