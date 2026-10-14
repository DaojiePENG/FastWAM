"""AIFlow 评测变体：A2-release（codec + 边缘 kv_decoder，release init）终点 ckpt 全量 delay 扫描。

任务名含 eval + libero + a2 + release → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a2_release

评测对象 = release-init 矩阵 a2 线 24951 终点
（runs/libero_all_leapbotce_codec/codec_a2_release_seed42_h200，
use_kv_decoder=true 由 RUN_DIR 的 config.yaml 带出）。
协议 = libero_spatial 整套 × D=0 × 50 trials（10-14 快速门控探针，同 a1/a3：通过后翻回
4 套件 × D{0,5,10,20,40}，skip-existing 自动续跑不浪费）。
模型侧 model.* 全部由 RUN_DIR 的 config.yaml 推导（sweep 白名单自动转发
mot_pretrained_path —— delta ckpt 只存微调部分，冻结基座必须从同一 release
ckpt 重建，见 LeapBotCE.save_checkpoint）。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/"
    "codec_a2_release_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/"
    "codec_a2_release_seed42_h200/checkpoints/weights/step_024951.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a2_release_step24951")
os.environ.setdefault("AI_FLOW_DELAYS", "0")
os.environ.setdefault("AI_FLOW_SUITES", "libero_spatial")
os.environ.setdefault("AI_FLOW_TRIALS", "50")
# 探针协议 1 格子不受影响；翻回全量协议后 AI_FLOW_PACK=1 自动兜底
# （10-14 20:xx 内存墙事故，见 a1 变体 docstring：32 workers 部分节点遍地 -6/-9）。
os.environ.setdefault("AI_FLOW_PACK", "1")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
