#!/usr/bin/env python3
"""FastWAM 树 AIFlow 分发自测（10-19 四步铁律之①；JENA scripts/check_aiflow_dispatch.py 同款）。

模拟平台 import：对每个任务名设 JinnTrainResult / userPath / GPU_NUM 后
真实 import 树根 AIFlow_config（dispatcher），断言落点变体与 CMD 关键串。
含 b6 快捷键的双向子串包含检查（铁律之②在 CASES 里体现：b6_d40 先于裸
d40 命中、train a6_lora 名不得被 b6 误伤）。全绿 exit 0。

用法：
    <python> scripts/check_aiflow_dispatch.py          # 用 sys.executable
    CHECK_PY=<env python> python scripts/check_aiflow_dispatch.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (任务名, 期望 dispatcher 变体后缀, CMD 必含子串, CMD 必不含子串)
CASES = [
    # B 线默认提交：不得带 RUN_DIR/SWEEP_ROOT/EVAL_TASK（保持 B 线默认路由）
    ("leapbot_ce_eval_robotwin", "eval_robotwin",
     ["eval_leapbotce_robotwin_sweep_2653853.sh", "LEAPBOTCE_CKPT=latest"],
     ["LEAPBOTCE_RUN_DIR", "LEAPBOTCE_SWEEP_ROOT", "LEAPBOTCE_EVAL_TASK"]),
    # B6 v2 全量 50 任务:独立 v2 根 + 50 任务表 + 四旋钮;先于裸 b6 命中
    ("leapbot_ce_eval_robotwin_b6v2", "eval_robotwin",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40'",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora",
      "ROBOTWIN_EVAL_TASKS='adjust_bottle beat_block_hammer"],
     ["leapbotce/robotwin_sweep_b6'"]),
    # B6 全量：四旋钮齐 + 四窗默认（EVAL_TASK=codec_lora 配方，10-26 事故修复）
    ("leapbot_ce_eval_robotwin_b6", "eval_robotwin",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_sweep_b6",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40'",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora"],
     []),
    # b6 + 时延令牌：令牌优先（d40 ⊃ "d4" 无关，但 b6 仍须命中 run 切换）
    ("leapbot_ce_eval_robotwin_b6_d40", "eval_robotwin",
     ["ROBOTWIN_EVAL_DELAYS=40",
      "LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora"],
     []),
    # 裸 d40（无 b6）：B 线默认，绝不得带 B6 run
    ("leapbot_ce_eval_robotwin_d40", "eval_robotwin",
     ["ROBOTWIN_EVAL_DELAYS=40"],
     ["LEAPBOTCE_RUN_DIR"]),
    # 训练名（含 a6_lora —— 与 b6 无子串包含，不得互相污染）
    ("leapbot_ce_train_robotwin_a6_lora", "train_robotwin_a6_lora", [], []),
    # ── a7 加容量重训轮（10-30 建，10-31 改 16 卡重提；a6_lora_fs 全量复制版）──
    # 与 a6 差异仅两处：r32/α32 Hydra 覆盖、20ep（43393 步 = round(20×277713/128)）。
    # 16 卡 eff128（bs8/卡，lr 与 a6 同 2.8e-4；24 卡单 22724896 已取消）。
    # mot_pretrained_path 必须保留（10-21 事故铁律）；不得带 --resume（冷启动）。
    ("leapbot_ce_train_libero_a7_lora", "libero_a7_lora",
     ["--gpus 16", "--batch-size 8", "--lr 0.00028", "--steps 43393",
      "--delay 20", "--stale-lambda 0.5 --stale-warmup -1",
      "--run-name codec_a7_lora_r32_20ep_seed42_h100",
      "--save-every 2170", "--keep-ckpt 20",
      "model.video_lora.rank=32", "model.video_lora.alpha=32.0",
      "model.mot_pretrained_path=/mnt/c2-worldmodel/2638088/fastwam/"
      "fastwam_release/libero_uncond_2cam224.pt",
      "wandb.name=codec_a7_lora_r32_20ep_seed42_h100"],
     ["--delay 40", "--resume", "rank=16", "alpha=16", "codec_a6_lora_fs"]),
    # ── RoboTwin D60/D80 时延探针（10-31；两单并行 8×H100）──
    # ours B6 侧：B6 v2 同 run/ckpt、v1 既有 10 任务子集、独立 ext 根
    # （防污染 v2 主根 .done/聚合）、只跑 '60 80' 两窗。对照窗不重跑。
    ("leapbot_ce_eval_robotwin_b6_ext", "eval_robotwin_b6_ext",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6_ext",
      "ROBOTWIN_EVAL_DELAYS='60 80'",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora",
      "ROBOTWIN_EVAL_TASKS='adjust_bottle beat_block_hammer",
      "LEAPBOTCE_CKPT=latest"],
     ["robotwin_sweep_b6'", "0 10 20 40", "REPLAN_STEPS"]),
    # ── 串号修复重跑（11-08；旧 lane 结果拷贝抄到兄弟 lane 的 run 目录）──
    # ours 探针 20 格：独立 ext2 根 + D{60,80} + 10 任务 + **v2 入口**
    # （CMD 必须出现 v2 wrapper，否则又跑回旧 lane）。
    ("leapbot_ce_eval_robotwin_b6_ext2", "eval_robotwin_b6_ext2",
     ["LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6_ext2",
      "ROBOTWIN_EVAL_DELAYS='60 80'",
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora",
      "ROBOTWIN_EVAL_TASKS='adjust_bottle beat_block_hammer",
      "eval_leapbotce_robotwin_sweep_v2_2653853.sh",
      "LEAPBOTCE_CKPT=latest"],
     ["robotwin_v2_sweep_b6_ext'", "robotwin_v2_sweep_b6'", "0 10 20 40"]),
    # ours v2 D0 污染 20 格重跑：独立 d0recheck 根 + 单窗 D0 + 20 任务表 +
    # v2 入口。任务名含 d0 令牌，基座快捷键也会给 0（双保险，断言单窗）。
    ("leapbot_ce_eval_robotwin_b6_d0recheck", "eval_robotwin_b6_d0recheck",
     ["LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6_d0recheck",
      "ROBOTWIN_EVAL_DELAYS=0",
      "ROBOTWIN_EVAL_TASKS='blocks_ranking_rgb blocks_ranking_size",
      "eval_leapbotce_robotwin_sweep_v2_2653853.sh"],
     ["robotwin_v2_sweep_b6'", "10 20 40", "60 80"]),
    # StarWAM 侧：4 词后缀压过既有 3 词 starwam 委托；env 预设探针口径 +
    # V2_SWEEP_ROOT patch 到独立 ext 根（该常量非 env 可覆盖，缺省会写进
    # v2 主根）。
    ("leapbot_ce_eval_robotwin_starwam_ext", "eval_robotwin_starwam_ext",
     ["STARWAM_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "starwam/robotwin_v2_sweep_ext",
      "STARWAM_V2_DELAYS='60 80'",
      "STARWAM_V2_TASKS='adjust_bottle beat_block_hammer"],
     ["robotwin_v2_sweep'", "0 10 20 40", "REPLAN_STEPS"]),
    # ── chunk 剂量-响应变体（10-31；replan 24→8，与 D60/80 探针互补）──
    # ours 侧：ext 同 run/ckpt/任务表，独立 rs8 根，D{0,40} 两窗 +
    # LEAPBOTCE_REPLAN_STEPS=8（lane 脚本转 EVALUATION.replan_steps）。
    ("leapbot_ce_eval_robotwin_b6_ext_rs8", "eval_robotwin_b6_ext_rs8",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_v2_sweep_b6_ext_rs8",
      "LEAPBOTCE_REPLAN_STEPS=8",
      "ROBOTWIN_EVAL_DELAYS='0 10 20 40 60 80'",  # 11-02 六窗; 既有 D{0,40} .done 幂等跳过
      "LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora",
      "LEAPBOTCE_CKPT=latest",
      # 11-02 升级 v2 全量:不再预设 10 任务,落基座缺省 50 任务全表
      "ROBOTWIN_EVAL_TASKS='adjust_bottle beat_block_hammer"],
     ["robotwin_sweep_b6'", "REPLAN_STEPS=24"]),
    # StarWAM 侧：5 词后缀压过 starwam_ext；rs8 独立根 + STARWAM_REPLAN_STEPS=8
    # （config→chains→lane→adapter 链透传）。
    ("leapbot_ce_eval_robotwin_starwam_ext_rs8", "eval_robotwin_starwam_ext_rs8",
     ["STARWAM_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "starwam/robotwin_v2_sweep_ext_rs8",
      "STARWAM_REPLAN_STEPS=8",
      "STARWAM_V2_DELAYS='0 10 20 40 60 80'",  # 11-02 六窗; 既有 D{0,40} .done 幂等跳过
      # 11-02 升级 v2 全量:不再预设 10 任务,sibling 缺省 50 任务全表
      # (beat_block_hammer 打头) —— 断言默认表而非 10 任务子集
      "STARWAM_V2_TASKS='beat_block_hammer adjust_bottle"],
     ["robotwin_v2_sweep'"]),
    # handover_block 跨批次矛盾复跑（11-02）：显式 step_158206 + 矛盾本体+
    # 锚任务两格表 + D{20,40} 谷区 + 独立 recheck 根。绝不指向 v1/ext 根。
    ("leapbot_ce_eval_robotwin_b6_handover", "eval_robotwin_b6_handover",
     ["LEAPBOTCE_RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "robotwin_all_leapbotce_codec_lora/robotwin_all_codec_lora_dtrain20_seed42_gpu_n3",
      "step_158206.pt",
      "LEAPBOTCE_SWEEP_ROOT=/mnt/c2-worldmodel/2653853/baseline-evaluations/"
      "leapbotce/robotwin_sweep_b6_handover_recheck",
      "ROBOTWIN_EVAL_DELAYS='20 40'",
      "ROBOTWIN_EVAL_TASKS='handover_block beat_block_hammer'"],
     ["robotwin_sweep_b6'", "robotwin_v2_sweep_b6_ext", "0 10", "60 80"]),
    # AHA-WAM v2 主表 D60/80 补窗（11-02）：转发 RoboTwin 基座；平台提单
    # 无 env 栏（只能选卡+命名）→ 变体内建 setdefault AI_FLOW_DELAYS='60 80'，
    # 转发→sibling 消费→CMD 显式拼 ROBOTWIN_EVAL_DELAYS='60 80'。四窗 200
    # 格已收齐（主表 D0 93.5/D10 78.7/D20 54.3/D40 29.6），不得出现四窗串。
    ("leapbot_ce_eval_robotwin_ahawam_v2", "eval_robotwin_ahawam",
     ["ROBOTWIN_EVAL_DELAYS='60 80'",
      "XPOLICY_SWEEP_SUFFIX=robotwin_v2_sweep",
      "bash run_sweep_cluster_xpolicy_2653853.sh AHA_WAM"],
     ["0 10 20 40"]),
    # LIBERO 薄委托变体：最长后缀胜出
    ("leapbot_ce_eval_libero_a1_ft_fresh100_full",
     "eval_libero_a1_ft_fresh100_full", ["libero"], []),
    # ── Table 4 training ablations（10-27 三变体，母本 a6_lora_fs）──
    # stale0：λmax=0。断言用 "--stale-lambda 0 --stale-warmup" 尾随词锚定——
    # 若漏改仍为 0.5，子串 "--stale-lambda 0" 会被 "0.5" 藏住造成假阳
    ("leapbot_ce_train_libero_a6_stale0", "libero_a6_stale0",
     ["--delay 20", "--stale-lambda 0 --stale-warmup",
      "--run-name codec_a6_stale0_15ep_seed42_h200",
      "wandb.name=codec_a6_stale0_15ep_seed42_h200"],
     ["--stale-lambda 0.5"]),
    ("leapbot_ce_train_libero_a6_stale1", "libero_a6_stale1",
     ["--delay 20", "--stale-lambda 1 --stale-warmup",
      "--run-name codec_a6_stale1_15ep_seed42_h200"],
     []),
    ("leapbot_ce_train_libero_a6_dtrain40", "libero_a6_dtrain40",
     ["--delay 40", "--stale-lambda 0.5",
      "--run-name codec_a6_dtrain40_15ep_seed42_h200"],
     ["--delay 20"]),
    # 母本回归锁：三个新变体文件不得劫持 a6_lora_fs 本名
    ("leapbot_ce_train_libero_a6_lora_fs", "libero_a6_lora_fs",
     ["--delay 20", "--stale-lambda 0.5",
      "--run-name codec_a6_lora_fs_15ep_seed42_h200"],
     ["--delay 40"]),
    # nocloudproprio：λ/delay 全部母本值，唯一配方差异在 EXTRA_OVERRIDES 的
    # use_cloud_proprio=false；run_name 区分
    ("leapbot_ce_train_libero_a6_nocloudproprio", "libero_a6_nocloudproprio",
     ["--delay 20", "--stale-lambda 0.5",
      "--run-name codec_a6_nocloudproprio_15ep_seed42_h200",
      "model.cloudedge.use_cloud_proprio=false",
      "model.mot_pretrained_path="],
     ["use_cloud_proprio=true"]),
    # edgeonly（11-03）：Table 4「w/o cloud (training)」——训练完全不做云端
    # 前向；LoRA/codec 挂云端、云缺席时无梯度，必须同关
    ("leapbot_ce_train_libero_a6_edgeonly", "libero_a6_edgeonly",
     ["--delay 20", "--stale-lambda 0.5",
      "--run-name codec_a6_edgeonly_15ep_seed42_h200",
      "model.cloudedge.use_cloud_cache=false",
      "model.video_lora.enabled=false",
      "model.cloudedge.kv_codec.enabled=false",
      "model.mot_pretrained_path="],
     ["use_cloud_cache=true", "video_lora.enabled=true", "kv_codec.enabled=true"]),
    # ── Table 4 四行终点 eval 变体（10-28；薄委托 a6_lora_fs eval 同款）──
    # 协议四旋钮 DELAYS/TAG/TRIALS/PACK+GPU_NUM 与 deployed a6 行一致；
    # stale0 额外断言不得串行（RUN_DIR/Tag 锚定）且不带 use_cloud_proprio 注入
    ("leapbot_ce_eval_libero_a6_stale0", "eval_libero_a6_stale0",
     ["RUN_DIR=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "libero_all_leapbotce_codec_lora/codec_a6_stale0_15ep_seed42_h200",
      "LEAPBOTCE_CKPT=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/"
      "libero_all_leapbotce_codec_lora/codec_a6_stale0_15ep_seed42_h200/"
      "checkpoints/weights/step_032544.pt",
      "LIBERO_EVAL_TAG=a6_stale0_step32544",
      "LIBERO_EVAL_DELAYS='0 10 20 40'", "NUM_TRIALS=50",
      "LIBERO_LIBERO_PACK=1", "GPU_NUM=4"],
     ["use_cloud_proprio", "codec_a6_stale1", "a6_lora_fs"]),
    ("leapbot_ce_eval_libero_a6_stale1", "eval_libero_a6_stale1",
     ["codec_a6_stale1_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_stale1_step32544",
      "LIBERO_EVAL_DELAYS='0 10 20 40'"],
     ["codec_a6_stale0"]),
    ("leapbot_ce_eval_libero_a6_dtrain40", "eval_libero_a6_dtrain40",
     ["codec_a6_dtrain40_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_dtrain40_step32544",
      "LIBERO_EVAL_DELAYS='0 10 20 40'"],
     ["codec_a6_lora_fs"]),
    # dtrain40 ep9 中点快评（11-02）：只跑 D0 一窗定位塌方时点；5 词后缀
    # 压过 dtrain40(4 词)与 ep9(4 词)。ep9=step_019530 与终点 ckpt 必须区分。
    ("leapbot_ce_eval_libero_a6_dtrain40_ep9", "eval_libero_a6_dtrain40_ep9",
     ["codec_a6_dtrain40_15ep_seed42_h200", "step_019530.pt",
      "LIBERO_EVAL_TAG=a6_dtrain40_step019530",
      "LIBERO_EVAL_DELAYS=0"],  # 单窗无空格, _q() 不加引号
     ["step_032544", "a6_dtrain40_step32544", "0 10 20 40"]),
    # a7 final 四窗（11-02）：r32/alpha32 加强版 20ep；latest 解析=训练收工后
    # 提交（提前提交会把 TAG 钉在中段 ckpt，docstring 有时序约束）；rank 适配
    # 由 sweep 脚本从 run config.yaml 派生（badbfff），CMD 里无需显式覆盖。
    ("leapbot_ce_eval_libero_a7_final", "eval_libero_a7_final",
     ["codec_a7_lora_r32_20ep_seed42_h100_n2", "LEAPBOTCE_CKPT=latest",
      "LIBERO_EVAL_TAG=a7_lora_r32_final",
      "LIBERO_EVAL_DELAYS='0 10 20 40'", "GPU_NUM=4"],
     ["step_032544", "codec_a6"]),
    # nocloudproprio eval：EVAL_EXTRA 注入 use_cloud_proprio=false 是必需项
    # （sweep 推导白名单不含此键，漏注入=strict load 过但行为回 True，整行作废）
    ("leapbot_ce_eval_libero_a6_nocloudproprio", "eval_libero_a6_nocloudproprio",
     ["codec_a6_nocloudproprio_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_nocloudproprio_step32544",
      "LIBERO_EVAL_EXTRA=model.cloudedge.use_cloud_proprio=false"],
     ["use_cloud_proprio=true"]),
    # noedgeprefix：与 deployed a6 行同 ckpt，唯一差异 = EVAL_EXTRA 关 edge
    # prefix；TAG 必须区别于 deployed 行（复用旧 TAG 会混结果）
    ("leapbot_ce_eval_libero_a6_noedgeprefix", "eval_libero_a6_noedgeprefix",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_noedgeprefix",
      "LIBERO_EVAL_EXTRA=model.cloudedge.edge_prefix_enabled=false"],
     ["TAG=a6_lora_fs_step32544 ", "use_cloud_proprio"]),
    # nocloudcache：与 deployed a6 行同 ckpt，唯一差异 = EVAL_EXTRA 关云端载荷
    # （encode_cloud 返回零长度 planning cache）；TAG 必须区别于 deployed 行
    ("leapbot_ce_eval_libero_a6_nocloudcache", "eval_libero_a6_nocloudcache",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_nocloudcache",
      "LIBERO_EVAL_EXTRA=model.cloudedge.use_cloud_cache=false"],
     ["TAG=a6_lora_fs_step32544 ", "model.cloudedge.edge_prefix_enabled=false"]),
    # edgeonly eval：edge-only 训练线终点；EVAL_EXTRA 必须带 use_cloud_cache
    # =false（ckpt meta 护栏会拒绝漏注入的评测）
    ("leapbot_ce_eval_libero_a6_edgeonly", "eval_libero_a6_edgeonly",
     ["codec_a6_edgeonly_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_edgeonly_step32544",
      "LIBERO_EVAL_EXTRA=model.cloudedge.use_cloud_cache=false"],
     ["TAG=a6_lora_fs_step32544 ", "codec_a6_lora_fs_15ep_seed42_h200"]),
    # ── 11-03 审稿补实验（CVPR2027 模拟审稿 P0 三条 + Paired-Age 行为对照）──
    # consttoken：同 ckpt，云 token 内容置零但长度不变（排除"槽位容量"解释）
    ("leapbot_ce_eval_libero_a6_consttoken", "eval_libero_a6_consttoken",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_consttoken",
      "LIBERO_EVAL_EXTRA=model.cloudedge.cloud_cache_mode=constant"],
     ["TAG=a6_lora_fs_step32544 ", "use_cloud_cache=false", "cloud_cache_mode=shuffled"]),
    # shuffledtoken：K/V 同序置换（内容统计保留、结构破坏）；不得串到 constant
    ("leapbot_ce_eval_libero_a6_shuffledtoken", "eval_libero_a6_shuffledtoken",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_shuffledtoken",
      "LIBERO_EVAL_EXTRA=model.cloudedge.cloud_cache_mode=shuffled"],
     ["TAG=a6_lora_fs_step32544 ", "use_cloud_cache=false", "cloud_cache_mode=constant"]),
    # swaptoken：跨 episode 换 cache（唯一干净的 token 对照；shuffled 已证为空操作）
    ("leapbot_ce_eval_libero_a6_swaptoken", "eval_libero_a6_swaptoken",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_swaptoken",
      "LIBERO_EVAL_EXTRA=model.cloudedge.cloud_cache_mode=swap"],
     ["TAG=a6_lora_fs_step32544 ", "use_cloud_cache=false",
      "cloud_cache_mode=shuffled", "cloud_cache_mode=constant"]),
    # edgedelay：信息条件对齐对照——端侧也吃同龄历史观测（Concern 1）
    ("leapbot_ce_eval_libero_a6_edgedelay", "eval_libero_a6_edgedelay",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step32544_edgedelay",
      "LIBERO_EVAL_EXTRA=EVALUATION.edge_uses_delayed_obs=true"],
     ["TAG=a6_lora_fs_step32544 ", "use_cloud_cache=false"]),
    # singleage 训练：与 a6 逐项相同，唯一任务差异 = 单分支随机龄
    ("leapbot_ce_train_libero_a6_singleage", "libero_a6_singleage",
     ["--delay 20", "--stale-lambda 0.5",
      "--run-name codec_a6_singleage_15ep_seed42_h200",
      "model.cloudedge.paired_age_mode=single_random",
      "model.mot_pretrained_path="],
     ["paired_age_mode=paired", "use_cloud_cache=false", "video_lora.enabled=false"]),
    # singleage eval：推理路径与 a6 相同，不得带任何行为 override
    ("leapbot_ce_eval_libero_a6_singleage", "eval_libero_a6_singleage",
     ["codec_a6_singleage_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_singleage_step32544",
      "LIBERO_EVAL_DELAYS='0 10 20 40'"],
     ["codec_a6_lora_fs_15ep_seed42_h200", "LIBERO_EVAL_EXTRA"]),
    # 11-05 出差无人值守：等训练线终点 ckpt 再跑四窗评测。三张单共用同一
    # wait 脚本，靠 SKIP_* 环境变量分工；长后缀裁决保证不串到单行变体。
    # 合并单（串行两行）
    ("leapbot_ce_eval_libero_a6_trainablations", "eval_libero_a6_trainablations",
     ["bash scripts/eval_a6_trainablations_wait.sh"],
     ["eval_leapbotce_libero_sweep_2653853.sh",
      "LIBERO_EVAL_TAG", "LEAPBOTCE_CKPT", "LEAPBOTCE_SKIP_"]),
    # 并行双单（用户 11-05 拍板：各 4 卡，比串行合并单早 ~6.5 h 收工）
    ("leapbot_ce_eval_libero_a6_trainablations_edgeonly",
     "eval_libero_a6_trainablations_edgeonly",
     ["LEAPBOTCE_SKIP_SINGLEAGE=1 bash scripts/eval_a6_trainablations_wait.sh"],
     ["eval_leapbotce_libero_sweep_2653853.sh", "LIBERO_EVAL_TAG",
      "LEAPBOTCE_SKIP_EDGEONLY"]),
    ("leapbot_ce_eval_libero_a6_trainablations_singleage",
     "eval_libero_a6_trainablations_singleage",
     ["LEAPBOTCE_SKIP_EDGEONLY=1 bash scripts/eval_a6_trainablations_wait.sh"],
     ["eval_leapbotce_libero_sweep_2653853.sh", "LIBERO_EVAL_TAG",
      "LEAPBOTCE_SKIP_SINGLEAGE"]),
    # ── deployed a6 行 checkpoint 筛选 stage-1（10-30；D{0,40} 两窗）──
    # 评测对象 = a6_lora_fs run 的 ep9/11/13/14 终点；只跑两窗（stage-2 另单
    # 补 "10 20"，同 TAG 共用结果根）。选择标准：<2 点改善不换（见变体 docstring）
    ("leapbot_ce_eval_libero_a6_ep9", "eval_libero_a6_ep9",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_019530.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step019530",
      "LIBERO_EVAL_DELAYS='0 40'", "GPU_NUM=4"],
     ["codec_a6_stale0", "codec_a6_stale1", "codec_a6_dtrain40", "use_cloud_proprio"]),
    ("leapbot_ce_eval_libero_a6_ep11", "eval_libero_a6_ep11",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_023870.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step023870",
      "LIBERO_EVAL_DELAYS='0 40'", "GPU_NUM=4"],
     ["codec_a6_stale0", "codec_a6_stale1", "codec_a6_dtrain40", "use_cloud_proprio"]),
    ("leapbot_ce_eval_libero_a6_ep13", "eval_libero_a6_ep13",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_028210.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step028210",
      "LIBERO_EVAL_DELAYS='0 40'", "GPU_NUM=4"],
     ["codec_a6_stale0", "codec_a6_stale1", "codec_a6_dtrain40", "use_cloud_proprio"]),
    ("leapbot_ce_eval_libero_a6_ep14", "eval_libero_a6_ep14",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_030380.pt",
      "LIBERO_EVAL_TAG=a6_lora_fs_step030380",
      "LIBERO_EVAL_DELAYS='0 40'", "GPU_NUM=4"],
     ["codec_a6_stale0", "codec_a6_stale1", "codec_a6_dtrain40", "use_cloud_proprio"]),
    # ── deployed a6 行协议方差重复（11-03；3 seeds + 原轮 = n=4 报均值±std）──
    # 同 ckpt step_032544 / D{0,10,20,40} / 50 trials / PACK=1；仅 TAG+SEED
    # 不同（delay 采样按 seed+episode_idx 钉死，换 seed 才采样完整协议随机
    # 性）；SEED 透传旋钮（AI_FLOW_SEED → SEED=）为本批新增。独立 TAG 根，
    # manager 不跳格。决策规则防 cherry-pick：报均值±std，绝不取 max。
    ("leapbot_ce_eval_libero_a6_rept1", "eval_libero_a6_rept1",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_recheck_seed1235", "SEED=1235",
      "LIBERO_EVAL_DELAYS='0 10 20 40'", "NUM_TRIALS=50",
      "LIBERO_LIBERO_PACK=1"],
     ["use_cloud_proprio", "step_019530", "step_023870"]),
    ("leapbot_ce_eval_libero_a6_rept2", "eval_libero_a6_rept2",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_recheck_seed1236", "SEED=1236",
      "LIBERO_EVAL_DELAYS='0 10 20 40'", "NUM_TRIALS=50",
      "LIBERO_LIBERO_PACK=1"],
     ["use_cloud_proprio", "SEED=1235", "SEED=1237"]),
    ("leapbot_ce_eval_libero_a6_rept3", "eval_libero_a6_rept3",
     ["codec_a6_lora_fs_15ep_seed42_h200", "step_032544.pt",
      "LIBERO_EVAL_TAG=a6_recheck_seed1237", "SEED=1237",
      "LIBERO_EVAL_DELAYS='0 10 20 40'", "NUM_TRIALS=50",
      "LIBERO_LIBERO_PACK=1"],
     ["use_cloud_proprio", "SEED=1235", "SEED=1236"]),
]

SNIP = r"""
import json, os, sys
sys.path.insert(0, os.environ["CHECK_USERPATH"])
import AIFlow_config as d
print(json.dumps({"variant": str(getattr(d, "_VARIANT", "?")),
                  "cmd": str(d.cfg.TRAIN.CMD)}))
"""


def run_case(name: str) -> dict:
    env = dict(os.environ)
    env["JinnTrainResult"] = f"/tmp/fake_jtr/{name}"
    env["CHECK_USERPATH"] = ROOT
    env["GPU_NUM"] = "8"
    env.pop("AI_FLOW_DELAYS", None)
    env.pop("AI_FLOW_TASKS", None)
    env.pop("AI_FLOW_RUN_DIR", None)
    env.pop("AI_FLOW_SWEEP_ROOT", None)
    proc = subprocess.run([sys.executable, "-c", SNIP], env=env,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"import 失败 rc={proc.returncode}\n{proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    py = os.environ.get("CHECK_PY", sys.executable)
    print(f"[check] root={ROOT}\n[check] python={py}")
    failed = 0
    for name, want_variant, must, must_not in CASES:
        env = dict(os.environ)
        env["CHECK_USERPATH"] = ROOT
        try:
            info = run_case(name)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            failed += 1
            continue
        problems = []
        if want_variant and info["variant"] != want_variant:
            problems.append(f"变体 {info['variant']} != 期望 {want_variant}")
        for s in must:
            if s not in info["cmd"]:
                problems.append(f"CMD 缺 {s!r}")
        for s in must_not:
            if s in info["cmd"]:
                problems.append(f"CMD 不应含 {s!r}")
        if problems:
            failed += 1
            print(f"FAIL {name}: " + "; ".join(problems))
        else:
            print(f"PASS {name} -> {info['variant']}")
        print(f"       CMD: {info['cmd'][:220]}{'…' if len(info['cmd']) > 220 else ''}")
    print(f"[check] {len(CASES) - failed}/{len(CASES)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
