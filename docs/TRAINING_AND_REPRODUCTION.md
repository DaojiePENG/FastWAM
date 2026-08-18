# LeapBot-VA 严格窗口训练、评测与集群复现

本文是交给合作者的主操作文档。当前正式实现是 `strict-window-v7`，不再使用旧版“整段 episode 前缀持续累积 KV”的训练方式，也不再使用 H41–H50 容量探针。模型原理和三种 causal mask 见 [CAUSAL_MEMORY_MODES.md](./CAUSAL_MEMORY_MODES.md)。

## 1. 当前冻结的 v7 契约

每 10 个控制步重规划一次，预测 32 步动作，只执行并提交前 10 步。当前块的信息流为：

```text
[可选的真实 V0 anchor]
  -> [最近 W=8 个完整真实块: Vt -> At(executed)]
  -> 当前真实 V
  -> 当前未来视频条件 C
  -> 当前 noisy action chunk
```

核心不变量如下：

- `max_history_blocks=70` 只是 episode 的硬容量，即最多 700 个已执行控制步；
- `history_window_blocks=8` 是默认训练和在线推理共同的信息窗口，即最近 80 个已执行动作；W 可通过 `HISTORY_WINDOW_BLOCKS` 调整，但每个 W 必须单独训练，评测值必须与 checkpoint 合同一致；
- episode 开始历史不足 W 时使用左侧全 mask padding，不复制第一帧作为有效历史；
- V0 离开最近窗口后只额外出现一次真实视觉 anchor，不重复 V0，不附带已经过期的 A0；
- 历史视觉来自同轨迹的真实重规划观测，历史动作只来自 demonstration 或环境实际下发的命令；
- 在线端保存真实单帧 VAE latent、对应上下文/proprio 和已执行动作，每轮重新计算窗口内 KV；
- 只删除最老 KV、保留吸收过旧信息的新 KV 不叫严格窗口，v7 禁止这样做；
- 当前未来视频 KV 是本轮 ActionDiT 的临时条件，绝不写入跨轮 memory；
- 未执行的后 22 个预测动作绝不进入 memory；
- episode 若在 chunk 中途结束，只提交实际执行的 1–9 步并立即 reset；这种 partial block 不能接下一观测；
- 窗口内部不 detach、不加门控，Action loss 可以沿完整 W 和当前未来视频条件反传。

正式配置为：

```text
model.history_training_mode=strict_replay_window_bptt
model.history_window_blocks=8
data.train.history_sampling_mode=recent_window
data.train.history_window_blocks=8
data.train.use_episode_anchor=true
data.train.max_history_blocks=70
```

`recent_window` 在同一 episode 中确定性选择所有可用的最近窗口，不会随机丢弃窗口内历史。日志中的 `history_blocks_mean` 因而位于 0–8：它是该 batch 中实际有效的最近历史块均值，不是随机截断开关。

## 2. Future-video 条件和 loss 诊断

训练保留 FastWAM 的 video/action flow-matching。ActionDiT 可以读取当前未来视频条件的逐层 KV，但 video flow target 使用独立噪声分支，不能泄漏到 action prefix。

条件课程是：

```text
step 0..199:       只用 clean GT future condition
step 200..999:     noised-condition 概率从 0 线性升到 0.5
step >=1000:       50% clean GT + 50% noised GT
noised GT 的 u:    [0.5, 1.0]
```

这是输入分布课程，不是 attention gate。在线推理仍从噪声经 VideoDiT 去噪得到当前未来条件，然后生成动作；不调用 VAE decode。

W&B 中重点看：

- `train/loss_action_d30`：30 层出口的当前 32 步 action FM loss；`d30` 是深度，不是 30 步预测；
- `train/loss_action_d30_ema`：上述指标的指数滑动平均；
- `train/loss_action_d30_condition_clean` 与 `_condition_noised`：区分 clean/noised future condition；
- `train/loss_action_d30_h0`、`h1_4`、`h5_8`：区分有效历史长度；
- `train/loss_action_d30_h5_8_condition_noised` 等：历史长度与条件类型的联合诊断；
- `train/future_video_condition_noise_probability`：课程当前目标概率；
- `train/future_video_condition_noised_fraction`：本次实际抽样比例；
- `train/history_blocks_mean`：batch 内真实有效 recent blocks 的均值；
- `train/episode_anchor_fraction`：V0 已离开窗口、因此启用一次 anchor 的样本比例。

不要只看聚合 action loss 判断实现错误。若 clean 低、noised 高，是条件分布难度；若 H0 低而 H5–8 高，才优先检查历史信息流；若 H0 本身明显偏离 FastWAM，则先检查基础 factorization、时间位置或 checkpoint 初始化。

## 3. 获取代码和环境

正式开发源是 DaojiePENG/FastWAM 的 `leapbot-va` 分支：

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
git clone -b leapbot-va --single-branch \
  https://github.com/DaojiePENG/FastWAM.git FastWAM
cd FastWAM
git branch --show-current
git remote -v
export ROOT_DIR="$(git rev-parse --show-toplevel)"
```

分支必须是 `leapbot-va`。原始 FastWAM 只作为参考 upstream：

```bash
git remote add upstream https://github.com/yuantianyuan01/FastWAM.git
```

推荐 Python 3.10 和锁定环境：

```bash
uv venv --python 3.10 .venv
uv sync --dev
uv run pytest -q
```

学校 HPC 已存在环境时可以直接指定解释器，不要求仓库内必须有 `.venv`：

```bash
export PYTHON_BIN=/path/to/python
export ACCELERATE_BIN=/path/to/accelerate
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q
```

`scripts/train_leapbot.sh` 和 `scripts/train_causal_modes.sh` 会使用这两个变量。正式训练从干净 commit 启动；launcher 默认拒绝 dirty worktree和已被明显占用的 GPU。

## 4. 学校 HPC 的单卡调试

申请一张 A800 调试资源：

```bash
srun -p i64m1tga800u -n 4 --mem=128G --gres=gpu:1 \
  --time=24:00:00 --pty bash
```

进入 allocation 后再执行：

```bash
cd /path/to/FastWAM
export ROOT_DIR="$(pwd)"
nvidia-smi
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q \
  tests/test_packed_training_masks.py \
  tests/test_incremental_full_bptt_training.py \
  tests/test_runtime_temporal_positions.py
```

单卡只用于 tiny-model、数据样本、真实 6B 单步和显存 smoke，不在交互 allocation 中直接开始长时间正式训练。Slurm 集群上的正式多卡启动应由站点的 `sbatch` wrapper 设置节点数、GPU 数、master address/port，再调用同一个 canonical launcher。

## 5. 资产与预计算

下载 release checkpoint、dataset stats 和官方 LeRobot LIBERO 数据：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"
"$PYTHON_BIN" scripts/download_leapbot_assets.py --root "$ROOT_DIR"
```

预计算文本编码：

```bash
"$PYTHON_BIN" scripts/precompute_text_embeds.py \
  --root "$ROOT_DIR" \
  --output "$ROOT_DIR/data/text_embeds_cache/libero"
```

正式 launcher 会校验数据、VAE、文本 cache、下载 manifest 和 release checkpoint 的 SHA-256。当前预计算内容是语言 T5 embedding 和固定资产；真实观测 VAE latent 没有离线固化，因为数据增强和当前真实帧仍需在线一致处理。训练数据加载只读取最近 W 的稀疏重规划观测与对应 `10W` 步动作，不再为一个样本解码 70 块完整前缀。

## 6. 正式启动入口

先只跑一个 mode 验证曲线，推荐 `action_aggregator`：

```bash
cd "$ROOT_DIR"
MODE=action_aggregator \
NUM_PROCESSES=8 GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
BATCH_SIZE=20 GRAD_ACCUM=1 MAX_STEPS=5000 \
LEARNING_RATE=1.0e-4 WANDB_ENABLED=true WANDB_MODE=online \
bash scripts/train_leapbot.sh
```

`B20/GA1` 是 8×80GB 的候选值，不是对任何集群都保证不 OOM。先把 `MAX_STEPS=2`、`SAVE_EVERY=2` 在目标硬件做真实模型 smoke，记录每 rank peak allocated/reserved，再逐步增加 B。出现 OOM 时优先下降 B 并增加 GA 保持 global batch，不要静默改变其他实验因素。

三种 causal mode 的同合同顺序对比：

```bash
MODES_CSV=action_aggregator,interleaved,vision_causal \
MAX_STEPS=5000 LEARNING_RATE=1.0e-4 \
bash scripts/train_causal_modes.sh
```

三个 mode 必须保持相同 release 初始化、数据、global batch、训练步数、随机种子、W、V0 anchor 和 condition curriculum。正式 checkpoint 同时保存权重、训练窗口、condition curriculum、代码 commit 和 run-contract hash。

多出口训练从获胜 D30 checkpoint 开始：

```bash
SOURCE_TRAIN_ROOT=/path/to/causal_run_root \
MODE=action_aggregator SOURCE_STEP=5000 MAX_STEPS=3000 \
bash scripts/train_multi_exit.sh
```

## 7. DeepSpeed、显存和吞吐

默认多卡训练通过 Accelerate + DeepSpeed ZeRO-2 启动：

```text
scripts/accelerate_configs/accelerate_zero2_ds.yaml
scripts/ds_configs/ds_zero2_config.json
```

ZeRO-2 分片 optimizer state 与 gradient，参数仍在每张卡。v7 的主要额外开销来自 W 内逐 segment attached KV 和反向重计算；在线推理每次重规划也会重新前推最多 `V0 + W×(V,A)` 的真实 prefix。因此：

- memory 是严格有界的，不随 episode 无限增加；
- 延迟不再是旧 incremental-KV 的 O(1)，而是有界 O(W)；
- `replay_bytes` 统计真实 latent/context/action replay buffer；
- `cache_bytes` 统计本轮重建后的逐层 KV；
- profiler 必须单独记录 `history_replay_s`、`observation_prefill_s`、`future_video_denoise_s`、`future_video_cache_s`、`action_denoise_s` 和 commit。

实际 B、吞吐和峰值显存只能由目标 commit、目标 mode 和目标 GPU 的 smoke 冻结。旧版 H41–H50 或 10GB 全局 KV 估计不适用于 v7。

## 8. 评测

评测配置必须与 checkpoint 的严格窗口合同一致：

```text
EVALUATION.memory.enabled=true
EVALUATION.memory.history_storage_mode=strict_replay
EVALUATION.memory.history_window_blocks=8
EVALUATION.memory.max_history_blocks=70
EVALUATION.replan_steps=10
EVALUATION.action_horizon=32
```

运行单 checkpoint：

```bash
CKPT=/path/to/step_005000.pt \
MODE=action_aggregator EXIT_DEPTH=30 TRIALS=10 \
bash scripts/evaluate_checkpoint.sh
```

PCH checkpoint 的 config-driven 入口为：

```bash
CKPT=/path/to/step_005000.pt bash scripts/evaluate_pch_checkpoint.sh
```

PCH 的 causal mode、trial 数、严格 replay 合同、视频保存和每卡并发数均在 `sim_leapbot_libero_pch.yaml` 中配置。

开发筛选每任务 10 次；最终每任务 50 次。记录逐任务/平均成功率、完成步数、P50/P95 重规划延迟、peak GPU memory、`cache_bytes`、`replay_bytes` 和上述各阶段耗时。Strict checkpoint 若被要求使用 `incremental_kv` 或不同 W，模型会 fail-fast，不能把不一致运行混入 Pareto 表。

## 9. 断点恢复与复现证据

一次可复现实验至少保留：

- Git commit 和干净工作树证明；
- `run_contract.txt`；
- `training_asset_manifest.json`；
- `checkpoints/weights/step_*.pt`；
- 对应的完整 DeepSpeed/Accelerate state 目录；
- W&B run ID、完整日志和 checkpoint validation JSON；
- 评测 fingerprint、逐 episode JSON 和 Pareto 汇总。

只保留 `.pt` 不能精确恢复 optimizer、scheduler、sampler 和 RNG。旧 `incremental_full_bptt` checkpoint 与 v7 的 `strict_replay_window_bptt` 合同不同，禁止跨合同 resume。

## 10. 必做验收

提交给大集群前至少通过：

```bash
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q
bash -n scripts/train_leapbot.sh scripts/train_causal_modes.sh \
  scripts/train_multi_exit.sh
```

GPU smoke 还必须确认：

1. H0、早期 padding、H8 和 `V0+H8` 都能前向/反向；
2. clean/noised condition 分项指标都出现，课程概率按 step 变化；
3. strict replay 的在线 segment 恰为 `V0 + recent W(V,A) + current V`；
4. 只执行的 10 步动作被缓存，预测后 22 步不存在于 replay；
5. reset 清空 KV、replay、anchor、prompt 和 episode clock；
6. profiler 中没有 VAE decode，future-video KV 只在当前调用存在；
7. 6B 的两步 optimizer smoke 无 NaN、无假更新、无 OOM。

若这些条件没有全部满足，不启动 5000-step 或三模式正式比较。
