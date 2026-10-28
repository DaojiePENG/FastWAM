"""AIFlow 评测变体：Table 4「w/o cloud proprioception」训练行终点全量 delay 扫描。

任务名含 eval + libero + a6 + nocloudproprio → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_nocloudproprio

评测对象 = leapbot_ce_train_libero_a6_nocloudproprio 的 15ep 终点（runs/
libero_all_leapbotce_codec_lora/codec_a6_nocloudproprio_15ep_seed42_h200，
32544 步；训练期 model.cloudedge.use_cloud_proprio=false——cloud 侧 context
融合全部 vision-only，edge 侧自融不动；其余配方与母本 a6_lora_fs 全同）。
协议 = 与 deployed a6 行（eval_libero_a6_lora_fs）完全一致：4 套件 ×
D{0,10,20,40} × 10 任务 × 50 trials（10-15 准则只测 D{0,10,20,40}）。
⚠️ 提单时点：必须等训练单跑完 step_032544.pt 落盘（AutoTrain preflight 对
显式 ckpt 缺失会立刻拒绝——按设计，提前提单秒失败不占卡）。
⚠️⚠️ 本变体与其它三行唯一的评测侧差异：AI_FLOW_EVAL_EXTRA 注入
model.cloudedge.use_cloud_proprio=false——开关在模型构造期生效，sweep 的
RUN_DIR 推导白名单不含此键，漏注入 = strict load 照样通过但行为回默认 True，
整行结果作废且无声（10-27 训练 config docstring 备忘的同款教训）。
AI_FLOW_PACK=1 + GPU_NUM=4：沿 a6_lora_fs eval 的内存墙规避配置（10-14 /
10-19 两次实锤，见基座 docstring）。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_nocloudproprio_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_nocloudproprio_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_nocloudproprio_step32544")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见基座 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 10-19 内存墙规避, 见基座 docstring
# Table 4 行合同：eval 侧必须与训练侧同关 cloud proprioception（见 docstring 双叹号注）。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.use_cloud_proprio=false")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
