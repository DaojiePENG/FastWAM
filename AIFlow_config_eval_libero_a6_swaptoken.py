"""AIFlow 评测变体：Table 4「cross-episode cloud cache (swap)」行全量 delay 扫描。

任务名含 eval + libero + a6 + swaptoken → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_swaptoken

回答问题（CVPR2027 模拟审稿 Concern 2 / P0「Shuffled Cloud Tokens」的修正版）：
原来那三条 token 消融没有一条干净（见 docs/leapbotce_ops_log.md 2026-11-04 P0）：

  * `off`（零长度）：布局漂移与"无信息"混在一起；
  * `constant`（K/V 置零）：零键拿 exp(0)=1，数百槽位抢走 softmax 质量，
    崩塌是注意力被污染，不是信息被移除（实测 d0 0.25%）；
  * `shuffled`（K/V 同步置换）：**数学空操作** —— prefill 已把 RoPE 烘进 k，
    且 cache K/V 直接 cat 进 flash_attention，输出对 (K,V) 联合置换不变
    （实测 d0 90.40% / d10 84.35%，对主行 91.0% / 83.7%）。

`swap` 才是干净的那个：每个 episode 用**上一个 episode 预先算好的 cloud cache**
（`cloud_cache_mode=swap`，模型实例内单槽 FIFO）。保留：槽位数、video_seq_len、
注意力掩码、每个槽位的 K/V 范数、k 里烘好的 RoPE 位置、每步算力——载荷是真的
cloud cache，不是合成替身。破坏：槽位与**当前场景**的对应关系（以及云侧 prompt，
它被 prefill 折进 K/V 里）。

⚠️ 口径（写论文必须写清，否则会被追问）：
  1. 同一次运行内，worker 连续处理同一 suite 的任务，因此**约 98% 的 swap 是
     "同任务、不同初始状态"**（LIBERO 的 initial state 差异很大：物体位置/朝向
     不同），约每 50 集才有一次跨任务边界。所以本行严格回答的是"**当前场景的
     对应关系**是否被真正消费"，而不是"任务语义是否被消费"。
  2. 每个 worker 进程的第 1 个 episode 没有前驱，跑真实 cache（warm-up）。
     4 worker ⇒ 每 delay 最多 4/2000 = 0.2% 的 episode 未置换；worker 日志里
     会打印 `[leapbotce][swap] first cross-episode cache swap (warm-up ...)`，
     写论文时引用该行即可给出确切数字。
  3. 本行与主行唯一差异就是这个开关；ckpt、delay 协议、trials 全部一致。

协议 = a6 主行完全一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials。
TAG 独立（a6_lora_fs_step32544_swaptoken）。PACK=1 钉死；GPU_NUM 不钉。

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
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_swaptoken")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")
# Table 4 行合同：同 ckpt、唯一差异 = 用上一 episode 的 cloud cache。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.cloud_cache_mode=swap")
# AI_FLOW_GPU_NUM 不设：提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
