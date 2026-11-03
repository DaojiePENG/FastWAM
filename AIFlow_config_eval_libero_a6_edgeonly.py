"""AIFlow 评测变体：Table 4「w/o cloud (training)」edge-only 行全量 delay 扫描。

任务名含 eval + libero + a6 + edgeonly → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_edgeonly

评测对象 = edge-only 训练线终点 ckpt（runs/libero_all_leapbotce_codec_lora/
codec_a6_edgeonly_15ep_seed42_h200，step_032544.pt，与 a6 同 15ep/32544 步/
seed42/同数据）。该 run 训练时 use_cloud_cache=false（云端前向完全不做，
single-loss，无 stale 分支），评测必须同样注入
model.cloudedge.use_cloud_cache=false —— 漏注入会被 load_checkpoint 的
单向护栏直接拒绝（ckpt meta 记 use_cloud_cache=false），不会静默出错。

⚠️ 口径：本行曲线应接近**平**（协议只加龄云端，本地流永远看当前帧），
retention 100% 是构造性的，不可与 a6 的 retention 直接比；叙述价值在
"每个窗的绝对值都更低"。行名写 "w/o cloud (training, edge-only)"。

协议与 deployed a6 行一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials
（10-15 准则只测 D{0,10,20,40}）。TAG 独立（a6_edgeonly_step32544），
不与 a6/其他消融行混结果根。

卡型兼容：PACK=1 钉死（每 worker ~25GB）；GPU_NUM 不钉（提交卡数即 worker
数）。本地流推理便宜（跳过云端 encode），估 4×H200 6-9h，以实测为准。

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
    "codec_a6_edgeonly_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_edgeonly_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_edgeonly_step32544")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # 每 worker ~25GB；钉死防节点漂移
# 必须与训练口径一致：ckpt 是 edge-only 训练的，云分支从未收到梯度。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.use_cloud_cache=false")
# AI_FLOW_GPU_NUM 不设：基座默认用满 pod 全部卡，提交卡数即 worker 数。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
