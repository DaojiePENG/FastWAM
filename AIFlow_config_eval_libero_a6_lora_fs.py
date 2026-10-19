"""AIFlow 评测变体：A6（LoRA + codec 冷启动共训线）15ep 终点全量 delay 扫描。

任务名含 eval + libero + a6 + lora + fs → 本文件（分发器最长后缀匹配），例如：
    leapbot_ce_eval_libero_a6_lora_fs

评测对象 = a6 线 15ep 终点（runs/libero_all_leapbotce_codec_lora/
codec_a6_lora_fs_15ep_seed42_h200，32544 步 = cosine 退火到 0 的自然终点；
LoRA q/k/v/o r16 与 codec、action expert 从 step 0 共训，冷启动官方
ActionDiT ckpt，无 resume —— a4 式「收敛谷底中途注入」的干净重做）。
选终点 ckpt 依据（spatial 25-trial 探针梯子，10-19 终判）：+7ep 82.0 /
+9ep 76.8（高 lr 中途回吐谷底）/ **15ep 终点 88.8（lr→0 完全恢复，全梯子
最高，vs a1 终点 83.6/84.4 = +4.4~+5.2）**；D0 7/10 集 95.4 健康。
协议 = 4 套件 × D{0,10,20,40} × 50 trials（10-15 用户准则：只测
D{0,10,20,40}，不跑 D5；与 a0/a1 收割口径一致）。
AI_FLOW_PACK=1（10-14 20:xx 内存墙事故，同 a1 变体 docstring：32 workers 在
部分节点遍地 -6/-9，全量协议钉 legacy 单 manager 串行模式；H200×8 ~8.5h）。
模型侧 model.* 全部由 RUN_DIR 的 config.yaml 推导（sweep 白名单自动转发
mot_pretrained_path —— delta ckpt 只存微调部分，冻结基座必须从同一 release
ckpt 重建，见 LeapBotCE.save_checkpoint；a6 的 config.yaml 带 video_lora 块
→ sweep 推导 hydra task libero_all_leapbotce_codec_lora_a4，eval 按 run
config instantiate 自动注入 120 wrappers，strict load 恢复 LoRA 权重，a4
冒烟已验证 lora_b 逐位一致）。
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
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200",
)
os.environ.setdefault(
    "AI_FLOW_CKPT",
    "/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec_lora/"
    "codec_a6_lora_fs_15ep_seed42_h200/checkpoints/weights/step_032544.pt",
)
os.environ.setdefault("AI_FLOW_TAG", "a6_lora_fs_step32544")
os.environ.setdefault("AI_FLOW_DELAYS", "0 10 20 40")  # 10-15 准则：时延评测只测 D{0,10,20,40}
os.environ.setdefault("AI_FLOW_TRIALS", "50")
os.environ.setdefault("AI_FLOW_PACK", "1")  # legacy 串行模式, 见 docstring 内存墙事故
# 10-19 首提实锤: 8 worker 在 xsyd-ai-h100-7 上 ~12-13 分钟集体 SIGABRT(-6)
# （容器内存顶格 ~200-240GB; 三轮同签名; fork 子进程幸存续写 delay_0）。
# 减到 4 worker（~100-120GB）远离内存墙, 4 格一晚收完; skip-existing 补格。
os.environ.setdefault("AI_FLOW_GPU_NUM", "4")

from AIFlow_config_eval_libero import *  # noqa: F401,F403,E402
from AIFlow_config_eval_libero import __C  # noqa: F401,E402 显式导入（star 不带下划线名），平台经分发器读 __C.TRAIN.CMD
from AIFlow_config_eval_libero import AutoTrain, cfg  # noqa: E402
