"""AIFlow 评测变体：deployed a6 行协议方差重复第 3 轮（seed 1237）。

任务名含 eval + libero + a6 + rept3 → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a6_rept3

目的（2026-11-03 用户拍板）：deployed 行（a6_lora_fs step_032544，
91.0 / 83.7 / 79.7 / 70.7）加跑 3 轮协议重复（seed 1235/1236/1237，加原轮
seed 1234 共 n=4），报均值±标准差 —— 把「a1 vs a6 差异在噪声内」升级成
有统计支撑的表述。决策规则（提交前钉死，防 cherry-pick）：
  - 报告口径 = 4 seed 均值±std；主表行是否改写为 mean±std 由论文侧定；
  - 绝不取 max / 择优替换（对同一 ckpt 择优 = 选择性报告）；
  - 任一窗偏差 ≥2~3σ → 按异常即停查协议/基建，先定性再谈表格。

seed 变更依据：delay 采样按 RandomState(seed + episode_idx) 逐集钉死
（experiments/libero/eval_libero_delayed.py:58）——沿原 seed 重复只会测到
GPU 非确定性、低估协议方差；换 seed 才采样完整协议随机性（delay 抽样 ×
动作采样非确定）。init states 为 LIBERO 固定 50/task，不引入额外方差源。

协议与原轮逐项一致：同 RUN_DIR / ckpt step_032544 / D{0,10,20,40} /
四套件 / 50 trials / PACK=1（内存墙钉 legacy 串行）；仅 TAG 与 SEED 不同
（TAG 新 → 独立结果根，manager 不会跳过已有格）。
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
os.environ.setdefault("AI_FLOW_TAG", "a6_recheck_seed1237")
os.environ.setdefault("AI_FLOW_SEED", "1237")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见 a6 变体 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")  # 平台 GPU_NUM env 优先; 4 卡 ~17h

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
