"""AIFlow 评测变体：Table 4「w/o cloud cache (eval-time)」行全量 delay 扫描。

任务名含 eval + libero + a6 + nocloudcache → 本文件（分发器最长后缀匹配），
例如：leapbot_ce_eval_libero_a6_nocloudcache

评测对象 = deployed a6 行同一张终点 ckpt（runs/libero_all_leapbotce_codec_lora/
codec_a6_lora_fs_15ep_seed42_h200，step_032544.pt），eval 期经 AI_FLOW_EVAL_EXTRA
注入 model.cloudedge.use_cloud_cache=false：encode_cloud 直接返回**零长度**
planning cache（跳过云端大模型前向），edge denoiser 只对「当前 edge prefix +
action tokens」做注意力——即部署模型在**完全没有云端载荷**时的表现。
与「w/o edge prefix (eval-time)」镜像：那行关本地前缀、这行关云端载荷，
两行合起来证明两条流各自都是 load-bearing。

⚠️ 口径（写论文时不可混）：这是 **eval-time** 消融，ckpt 训练时云 cache 从未
缺席，掉幅衡量的是「部署模型对云端载荷的依赖」，不能表述成「不用云也能训练
出什么」——后者是训练消融，需要从头重训（未做；外部基线 FastWAM 97.1→2.7
已在 LIBERO 表里给出同类证据）。行名必须保留 "(eval-time)" 后缀。

协议 = 与 deployed a6 行完全一致：4 套件 × D{0,10,20,40} × 10 任务 × 50 trials
（10-15 准则只测 D{0,10,20,40}）。⚠️ TAG 必须与 deployed 行不同——manager 按
TAG 跳过已有结果，复用旧 TAG 会把本行结果混进 deployed 行（独立 TAG
a6_lora_fs_step32544_nocloudcache）。

卡型兼容：PACK=1 钉死（每 worker ~25GB，L20 48GB 单卡放得下）；GPU_NUM 不钉
（提交卡数即 worker 数）。时长锚点：同协议 4 worker 于 H200 13.8h（noedgeprefix
实测）；本行跳过云端 encode，应按比例更快，估 8-11h（H200 4 worker），以 smoke
实测为准。

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
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544_nocloudcache")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # 每 worker ~25GB；钉死防节点漂移（见 docstring）
# Table 4 行合同：与 deployed a6 行同 ckpt、唯一差异 = eval 期关云端载荷。
os.environ.setdefault("AI_FLOW_EVAL_EXTRA", "model.cloudedge.use_cloud_cache=false")
# AI_FLOW_GPU_NUM 不设：基座默认用满 pod 全部卡，提交卡数即 worker 数（4 或 8）。

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
