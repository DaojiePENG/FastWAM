"""AIFlow 评测变体：Table 4「w/o edge prefix (eval-time)」行全量 delay 扫描。

任务名含 eval + libero + a6 + noedgeprefix → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_noedgeprefix

评测对象 = deployed a6 行同一张终点 ckpt（runs/libero_all_leapbotce_codec_lora/
codec_a6_lora_fs_15ep_seed42_h200，step_032544.pt——该 run 训练期
edge_vision.mode=select + edge_prefix_enabled=true，即边缘前缀 K/V 在训练中
存在且参与），eval 期经 AI_FLOW_EVAL_EXTRA 注入
model.cloudedge.edge_prefix_enabled=false 丢弃 edge 前缀 K/V、退回 cloud-only
attention——与 A4 消融同一条已验证路径（基座 docstring「eval-time hydra
overrides 透传」即为此设）。协议 = 与 deployed a6 行完全一致：4 套件 ×
D{0,10,20,40} × 10 任务 × 50 trials（10-15 准则只测 D{0,10,20,40}）。
⚠️ TAG 必须与 deployed 行（a6_lora_fs_step32544）不同——manager 按 TAG 跳过
已有结果，复用旧 TAG 会把 no-prefix 结果混进 deployed 行（本变体独立 TAG
a6_lora_fs_step32544_noedgeprefix，两者可并行提交互不干扰）。
卡型兼容：PACK=1 钉死（L20 48GB 每 worker ~25GB 单卡放得下；sweep 的 PACK
自动公式为纯显存式 clamp((VRAM-8G)/25G,1,4)，L20 本来就算出 1，钉死只为
不随节点漂移）。GPU_NUM 不钉（基座默认 = 用满 pod 全部卡）——**提交卡数即
worker 数旋钮**：4 卡=4 worker（host RSS ~100-120GB，a6_lora_fs eval 同款
已验证安全），8 卡=8 worker（RSS ~200-240GB，10-19 xsyd-ai-h100-7 实锤
顶格 SIGABRT 的签名；L20 节点若空闲内存 ≥~250GB 可用）。
时长实测锚点：同协议 4 worker 于 H200 节点 13.8h（10-19 15:59 → 10-20 05:47）；
L20 按 25-32 s/集折算 4 卡 ~14-18h、8 卡 ~7-9h。
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
    "codec_a6_lora_fs_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_noedgeprefix")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # L20 48G 单 worker/卡；钉死防节点漂移（见 docstring）
# Table 4 行合同：与 deployed a6 行同 ckpt、唯一差异 = eval 期关 edge prefix。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.edge_prefix_enabled=false")
# AI_FLOW_GPU_NUM 不设：基座默认用满 pod 全部卡，提交卡数即 worker 数（4 或 8）。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
