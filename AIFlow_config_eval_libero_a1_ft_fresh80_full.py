"""AIFlow 评测变体：A7 两段式后训 fresh80 臂终点正式全量 delay 扫描。

任务名 leapbot_ce_eval_libero_a1_ft_fresh80_full → 本文件（分发器最长后缀匹配）。

评测对象 = A7 fresh80 臂终点（runs/libero_all_leapbotce_codec/
codec_a1_ft_fresh80_seed42，step_003254 = 3ep 收工 ckpt；a1 终点仅权重续训 3ep，
lr 2e-5，stale λ 0.2（fresh:stale 0.8:0.2），10-20 训练 ⑤）。
协议 = 4 套件 × D{0,10,20,40} × 50 trials（与 a0/a1/a6 收割口径逐字对齐，
数字可直接进论文表；10-20 用户拍板跳过 spatial 门控一步到位全探）。
采纳判据 = D0 ≥ 94（a1 91.5 → 目标 94-95）且 D40 retention（vs D0）相对 a1
（76.6）掉幅 <2pt；终点不合 → 梯子探针变体 _p1085/_p2170 已备好可提。
AI_FLOW_PACK=1 + GPU_NUM=4（a6 实锤内存墙对策链：4 worker ≈100-120GB 稳，
8 worker 节点容器内存顶格集体 SIGABRT）。推荐单节点 4×H200——双节点会竞速双评
（a6 教训 10/160 需手动去重）。模型侧 model.* 全部由 RUN_DIR 的 config.yaml 推导
（a7 非 LoRA → hydra task 自动 libero_all_leapbotce_codec，与 a1 全量同路径；
delta ckpt 冻结基座从同一 mot release 重建）。薄委托：只设 AI_FLOW_* env 默认值，
其余接口全部转发基座，基座改动自动生效。
"""


import os
import sys

# 变体文件与基座同目录；保险起见把自身目录补进 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault(
    "AI_FLOW_RUN_DIR",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/codec_a1_ft_fresh80_seed42",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/codec_a1_ft_fresh80_seed42/checkpoints/weights/step_003254.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a1_ft_fresh80_step3254")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见 docstring 内存墙事故
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
