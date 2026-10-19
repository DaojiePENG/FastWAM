"""AIFlow 评测变体：A5（a1 终点纯续训线，无 LoRA）+8ep 峰值 ckpt 全量 delay 扫描。

任务名含 eval + libero + a5 + cont → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a5_cont

评测对象 = a5 线 +8ep 峰值（runs/libero_all_leapbotce_codec_cont/
codec_a5_cont_lr4e5_10ep_seed42_h200，step_017360 = a1 终点 24951 步
weight-only 续训 +8ep，constant lr 4e-5，stale warmup 0 —— a1 欠训练检验线）。
选 ckpt 依据（spatial 25-trial 探针梯子，10-19 终判）：+8ep **88.8（a5 梯子
峰值，vs a1 终点 83.6/84.4 = +4.4~+5.2）**，D0 94.8；+10ep 终点 85.6/92.4
双降 = 过峰，故评 +8ep 而非终点。与 a6 终点（LoRA+codec 从头 15ep）同分
88.8 —— 本线提供「小 lr 续训已够」的对照读数。
协议 = 4 套件 × D{0,10,20,40} × 50 trials（10-15 用户准则：只测
D{0,10,20,40}，不跑 D5；与 a0/a1 收割口径一致）。
AI_FLOW_PACK=1（10-14 20:xx 内存墙事故，同 a1 变体 docstring：32 workers 在
部分节点遍地 -6/-9，全量协议钉 legacy 单 manager 串行模式；H200×8 ~8.5h）。
模型侧 model.* 全部由 RUN_DIR 的 config.yaml 推导（sweep 白名单自动转发
mot_pretrained_path —— delta ckpt 只存微调部分，冻结基座必须从同一 release
ckpt 重建，见 LeapBotCE.save_checkpoint；a5 的 config.yaml 无 video_lora 块
→ sweep 推导 hydra task libero_all_leapbotce_codec，与 a1 收割同路径）。
薄委托：只设 AI_FLOW_* env 默认值，其余接口（cfg / AutoTrain / 命令拼装）
全部转发基座，基座改动自动生效。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_cont/"
    "codec_a5_cont_lr4e5_10ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_cont/"
    "codec_a5_cont_lr4e5_10ep_seed42_h200/checkpoints/weights/step_017360.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a5_cont_step17360")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见 docstring 内存墙事故

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
