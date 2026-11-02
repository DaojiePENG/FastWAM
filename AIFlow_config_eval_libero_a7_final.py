"""AIFlow 评测变体：a7 加强版（r32/α32, 20ep）final 四窗全量评测。

任务名含 eval + libero + a7 + final → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a7_final

背景（2026-11-02）：a7 = a6 的加强版（video LoRA rank 16→32 / alpha 32，
20ep，--delay 20，λ0.5/warmup-1，eff batch 128 不变），训练 run
codec_a7_lora_r32_20ep_seed42_h100_n2（8卡×bs8×accum2 自动补偿，动态
与 16 卡计划等价）。本单测 final ckpt 四窗 D{0,10,20,40}，与 a6 主行
同表同协议（4 套件 × 10 任务 × 50 trials）。

提交时序约束（重要）：AI_FLOW_CKPT=latest 在**提交时**解析为 run 目录
最新 step_*.pt —— 必须等 a7 训练收工（预计 2026-11-02 ~20:00）之后再
提交本单；提前提交会把 TAG 钉在训练中段的 ckpt 上（基座 docstring
"latest 自动升级必须换新 TAG"陷阱）。收工判据：训练单状态=完成，或
run 目录出现 ep20 终点 step_*.pt（20ep × ~2169.65 步 ≈ step_043393）。

LoRA rank 适配：a7 为 r32 训练，eval 侧 rank 必须匹配 ckpt（strict
load；11-02 本地探针首次实证 r16 默认会炸）。sweep 脚本已改为从 run
config.yaml 派生 model.video_lora.rank/alpha（commit badbfff），本变体
无需额外覆盖。

对照锚：a6 final D0=91.0%（同协议）；a7 本地 D0 探针（step_016275，
~ep15，a7_step016275_probe 根）出数后可做中期对照。
AI_FLOW_PACK=1 + GPU_NUM=4：内存墙规避配置（10-14/10-19 实锤，见基座
docstring）。
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
    "codec_a7_lora_r32_20ep_seed42_h100_n2",
)
os.environ.setdefault("AI_FLOW_CKPT", "latest")  # 必须训练收工后再提交，见 docstring 时序约束
os.environ.setdefault("AI_FLOW_TAG", "a7_lora_r32_final")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见基座 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 10-19 内存墙规避, 见基座 docstring

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
