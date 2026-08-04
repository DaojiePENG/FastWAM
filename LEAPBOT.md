# LeapBot-VA

LeapBot-VA 是基于 FastWAM 的 causal world-action model。跨轮记忆只来自真实观测和实际下发动作；当前未来视频只作为 ActionDiT 的临时逐层 KV 条件，不 VAE decode，也不进入下一轮记忆。

当前正式架构是 `strict-window-v7`：每轮用一个可选真实 V0 anchor、最近 W 个真实 `observation -> executed action` 块和当前真实观测重新计算 causal KV。W 默认取 8、可按实验调整。它取代了旧版无限 episode-prefix KV 和“只删除最老 KV”的近似窗口。

完整训练/集群操作见 [docs/TRAINING_AND_REPRODUCTION.md](docs/TRAINING_AND_REPRODUCTION.md)，信息流与三种 mask 见 [docs/CAUSAL_MEMORY_MODES.md](docs/CAUSAL_MEMORY_MODES.md)。

## 运行时接口

```python
memory = model.create_memory(
    exit_depth=30,
    causal_mode="action_aggregator",
    max_history_blocks=70,
    history_storage_mode="strict_replay",
    history_window_blocks=8,
)

prediction = model.infer_action(
    prompt=prompt,
    input_image=current_real_image,
    proprio=current_proprio,
    action_horizon=32,
    num_video_frames=9,
    memory=memory,
)

# 环境后处理、裁剪、gripper 二值化后，只提交真正执行的 10 步；
# 先转换回模型归一化空间，再写入 memory。
model.commit_executed_actions(memory, executed_actions_model_space)
model.reset_memory(memory)
```

状态机强制 `真实观测 -> 预测 -> 提交恰好 10 个已执行动作 -> 下一真实观测`。以下情况会 fail-fast：漏提交、重复观测、episode 内切换 prompt/depth/mode/W、超过 70 blocks、checkpoint 与 evaluation 窗口不一致，或出现非有限动作。

`memory=None` 保留原 FastWAM 无记忆 inference。

## 严格真实窗口

默认信息范围：

```text
current block t <= 8:
  [masked left padding] + all available recent real (V,A)

current block t > 8:
  one real V0 anchor + blocks [t-8, ..., t-1] real (V,A)
```

在线 memory 保存最近真实单帧 VAE latent、对应 context/proprio 和实际执行动作。下一轮先清空上轮 KV，再按同一时间顺序 replay。这样被淘汰块不会继续通过较新 KV 隐式泄漏进窗口。

V0 只出现一次且为 video-only anchor；不会把第一帧复制成 W 个有效 token，也不会在 V0 离开窗口后重复 A0。早期 padding 永远不是 attention key。

## 当前世界模型条件

LIBERO 默认 9 个物理视频帧经 Wan VAE 变为一个当前真实 latent frame 和两个未来 latent frames。训练中未来条件采用课程：

- step 0–199：100% clean GT future latent；
- step 200–999：noised 条件概率线性升至 0.5；
- 之后：50% clean、50% noised，`u in [0.5, 1.0]`。

Action loss 通过 future-condition KV 回传到 VideoDiT。独立的 video flow target 噪声不进入动作 prefix。在线端从噪声经 VideoDiT 去噪未来 latent，做一次最终前推取得临时 KV，再进行动作去噪；该 KV 在调用结束后丢弃。

## 三种模式

- `action_aggregator`：每张视觉独立编码，ActionDiT 汇聚 V0、最近真实 V/A、当前真实 V、未来视频和当前 action；首个 v7 实验优先使用它。
- `interleaved`：新视觉读取历史真实视觉和动作。
- `vision_causal`：新视觉只读取历史视觉；ActionDiT 仍读取全部真实 V/A。

三种模式下当前 action chunk 内部双向，历史不能读取当前/未来 token。

## 获取代码

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
git clone -b leapbot-va --single-branch \
  https://github.com/DaojiePENG/FastWAM.git FastWAM
cd FastWAM
uv venv --python 3.10 .venv
uv sync --dev
uv run pytest -q
```

原 FastWAM 可作为只读 upstream：

```bash
git remote add upstream https://github.com/yuantianyuan01/FastWAM.git
```

## 训练

单模式正式入口：

```bash
MODE=action_aggregator \
NUM_PROCESSES=8 GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
BATCH_SIZE=20 GRAD_ACCUM=1 MAX_STEPS=5000 \
LEARNING_RATE=1.0e-4 WANDB_ENABLED=true WANDB_MODE=online \
bash scripts/train_leapbot.sh
```

三模式对比：

```bash
MODES_CSV=action_aggregator,interleaved,vision_causal \
MAX_STEPS=5000 LEARNING_RATE=1.0e-4 \
bash scripts/train_causal_modes.sh
```

启动 5000 steps 前必须在目标 GPU 上先做两步真实 6B smoke，并据 peak memory 调整 B/GA。默认使用 Accelerate + DeepSpeed ZeRO-2。训练器拒绝 dirty worktree，并把代码、资产、拓扑、W、课程和 checkpoint 绑定到 run contract。

学校 HPC 单卡调试申请：

```bash
srun -p i64m1tga800u -n 4 --mem=128G --gres=gpu:1 \
  --time=24:00:00 --pty bash
```

## 关键验证

仓库测试覆盖：

- 三种 causal mask 和 future leakage；
- 可配置 W 的 recent suffix、左 padding 和 V0 anchor；
- strict replay 后 KV 中不存在窗口外块；
- 未执行预测动作不进入 replay；
- action 后处理对应的模型空间提交；
- reset、容量、事务回滚、绝对位置和 checkpoint 合同；
- D8/D16/D24/D30 保存加载。

```bash
PYTHONPATH=src python -m pytest -q
```

GPU 验收还需要真实 6B 单步训练、默认 W8+V0 inference、profiler 和 OOM 防护。旧 `incremental_full_bptt` checkpoint 与 v7 不兼容，不能 resume 或混入正式曲线。
