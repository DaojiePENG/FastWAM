"""AIFlow 评测变体：Table 4「info-matched delayed local observation」对照行。

任务名含 eval + libero + a6 + edgedelay → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_edgedelay

回答问题（CVPR2027 模拟审稿 Concern 1 / Critical「baseline 比较存在信息条件
不对等」）：审稿人指出我们的方法只延迟云端、保留端侧新鲜感知，而发布版
baseline 延迟整条观测流，因此成功率差里混着"信息条件"与"方法"两个因素。
本行用**我们自己的 ckpt**把端侧观测也按同一个 D 加龄（`EVALUATION.
edge_uses_delayed_obs=true`：edge prefix / 当前 proprio 都取 cloud 用的那条
历史帧），把观测条件对齐到 baseline，从而直接量出"新鲜本地感知"这一项值多少。

⚠️ 口径（写论文时必须写清）：这不是重实现某个 baseline，而是**在同一权重上
做信息条件对齐的受控实验**。它回答"差距中有多少来自信息条件"，不能替代
"baseline 在我们的接口下重训"（后者工程量大、不在本轮）。行名建议
`w/ delayed local obs (info-matched)`，不能写成任何 baseline 名字。

协议 = a6 主行完全一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials。
D=0 时 history.sample 返回当前帧，本行与 a6 主行应当逐位相同（自然的一致性
自检：若 D0 数字对不上，说明开关串了）。TAG 独立
（a6_lora_fs_step32544_edgedelay）。PACK=1 钉死；GPU_NUM 不钉。

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
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_edgedelay")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")
# 关键 override：端侧消费与云端同龄的历史观测（信息条件对齐 baseline）。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "EVALUATION.edge_uses_delayed_obs=true")
# AI_FLOW_GPU_NUM 不设：提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
