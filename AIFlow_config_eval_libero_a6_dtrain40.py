"""AIFlow 评测变体：Table 4「D_train=40」训练行终点全量 delay 扫描。

任务名含 eval + libero + a6 + dtrain40 → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a6_dtrain40

评测对象 = leapbot_ce_train_libero_a6_dtrain40 的 15ep 终点（runs/
libero_all_leapbotce_codec_lora/codec_a6_dtrain40_15ep_seed42_h200，32544 步；
训练期 max_delay_steps=40（stale 采样窗 [earliest, current) 均匀，期望 20.5），
其余配方与母本 a6_lora_fs 全同（λ=0.5 保留）。评测协议与 deployed a6 行
（eval_libero_a6_lora_fs）完全一致：4 套件 × D{0,10,20,40} × 10 任务 ×
50 trials（10-15 准则只测 D{0,10,20,40}）——训练窗 40 与评测时延档位正交，
评测档位由 LIBERO_EVAL_DELAYS 外部注入，不随训练窗变。
⚠️ 提单时点：必须等训练单跑完 step_032544.pt 落盘（AutoTrain preflight 对
显式 ckpt 缺失会立刻拒绝——按设计，提前提单秒失败不占卡；本单最慢，约
10-30 深夜收工）。
模型侧由 RUN_DIR 的 config.yaml 推导（sweep 白名单转发 task/mot_pretrained_path
等；data.train.max_delay_steps=40 只影响训练数据窗，eval 不取数、无需注入）。
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
    "codec_a6_dtrain40_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_dtrain40_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_dtrain40_step32544")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见基座 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 10-19 内存墙规避, 见基座 docstring

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
