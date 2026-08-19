# Episode Memory 本地运行手册

所有命令均从本工作树根目录执行：

```bash
cd /dahuafs/userdata/2638087/code/leapbot-va-episode-memory
LEAPBOT_PYTHON=/home/myuser/miniconda3/envs/leapbot-va/bin/python
```

## 配置检查与单元测试

只解析完整配置并打印最终启动命令，不加载模型或启动训练：

```bash
"$LEAPBOT_PYTHON" scripts/train_leapbot_local.py \
  --task libero_leapbot_episode_memory \
  --gpu-ids 0 \
  --dry-run
```

运行 episode-memory、PCH 和 scan-BPTT 回归：

```bash
PYTHONPATH=src "$LEAPBOT_PYTHON" -m pytest -q \
  tests/test_episode_memory.py \
  tests/test_memory.py \
  tests/test_pch.py \
  tests/test_incremental_full_bptt_training.py
```

## 单卡短程调试

以下命令强制采样 12-block prefix，使两步 smoke run 能覆盖首个 H chunk 和首帧记忆启用边界。输出目录需要为空或不存在。

```bash
"$LEAPBOT_PYTHON" scripts/train_leapbot_local.py \
  --task libero_leapbot_episode_memory \
  --gpu-ids 0 \
  output_dir=./runs/debug_episode_memory \
  max_steps=2 save_every=1 num_workers=0 \
  wandb.enabled=false \
  data.train.min_history_blocks=12 \
  data.train.max_history_blocks=12 \
  model.packed_history_attention_backend=dense \
  launch.max_preflight_used_mib=999999
```

## 正式两阶段训练

阶段一只训练 episode-memory updater、reader、门控和 learned H0：

```bash
"$LEAPBOT_PYTHON" scripts/train_leapbot_local.py \
  --task libero_leapbot_episode_memory \
  --gpu-ids 0,1,2,3 --num-processes 4 \
  output_dir=./runs/episode_memory_stage1
```

阶段二从阶段一权重初始化，联合训练 VideoDiT LoRA、ActionDiT 和 episode memory：

```bash
"$LEAPBOT_PYTHON" scripts/train_leapbot_local.py \
  --task libero_leapbot_episode_memory_joint \
  --gpu-ids 0,1,2,3 --num-processes 4 \
  output_dir=./runs/episode_memory_stage2 \
  resume=./runs/episode_memory_stage1/checkpoints/weights/step_100000.pt
```

同一输出目录的中断恢复由 launcher 自动读取 `checkpoints/state/step_*`，不要把 trainer-state 目录手工传给第二阶段。

## 评估

```bash
CKPT=./runs/episode_memory_stage2/checkpoints/weights/step_100000.pt \
LEAPBOT_PYTHON="$LEAPBOT_PYTHON" \
bash scripts/evaluate_episode_memory_checkpoint.sh
```

## 配置入口

训练实验由 `configs/task/libero_leapbot_episode_memory.yaml` 和 `libero_leapbot_episode_memory_joint.yaml` 控制；在线评估由 `configs/sim_leapbot_libero_episode_memory.yaml` 控制。首帧第三记忆的唯一开关是 `model.episode_memory.first_frame_memory`，数据侧 anchor 自动引用它。PCH 窗口只需修改 `model.history_window_blocks`，episode memory 的 `window_blocks` 自动引用该值；`chunk_blocks`、H 的 slot/维度、三种 causal mode 以及 VideoDiT/ActionDiT 的 H 读取范围也都由上述 YAML 控制。命令行中的 Hydra override 只用于临时调试，正式实验应固化为独立配置。

