# LeapBot-VA（中文翻译）

> 本文是 [`LEAPBOT.md`](./LEAPBOT.md) 的中文伴读译本。若译文与原文或实际命令存在差异，以原文为准；真正执行训练与复现实验时，只以 [`docs/TRAINING_AND_REPRODUCTION.md`](./docs/TRAINING_AND_REPRODUCTION.md) 为唯一主手册。

LeapBot-VA 是一个派生自 FastWAM 的世界—动作模型。它的持久记忆完全由真实观测和实际发送给控制器的命令构成。它保留 FastWAM 的视频/动作联合流匹配目标，并加入 LingBot-VA 风格的逆动力学条件：ActionDiT 读取想象未来视频 latent 块逐层产生的 K/V。在线推理会运行视频 DiT 和输出投影来对这些 latent 去噪，但绝不会用 VAE 将其解码，也绝不会把它们的 K/V 写入持久化 episode 记忆。

## 运行时契约

每个 episode 拥有一个显式的 `LeapMemoryState`：

```python
memory = model.create_memory(
    exit_depth=30,
    causal_mode="interleaved",
    max_history_blocks=70,
)
prediction = model.infer_action(
    prompt=prompt,
    input_image=current_real_image,
    proprio=current_proprio,
    action_horizon=32,
    num_video_frames=9,
    # -1 表示：视频去噪使用 scheduler 配置的全部步骤。
    future_video_denoise_steps=-1,
    memory=memory,
)

# 环境后处理后，最多执行 10 条命令。把这些确切执行过的命令转换回
# 归一化模型空间，然后只提交这一段。
model.commit_executed_actions(memory, executed_actions_model_space)
model.reset_memory(memory)  # episode 结束时调用
```

该状态机会拒绝：动作提交前出现第二次观测、episode 中途更改 prompt、更改深度，以及超出已配置容量的历史。checkpoint 只能使用元数据中记录为实际训练过的出口深度；只训练 D30 的 checkpoint 不能悄悄以 D8/16/24 运行。`memory=None` 会保留原始 FastWAM 的推理行为。

LIBERO 桥接层会在所有后处理之后规范化每条最终命令：先裁剪到 `env.action_spec`，再确定性地二值化夹爪，并把同一个数组同时传给 `env.step` 和记忆提交转换。启用记忆时，系统拒绝跨重规划动作集成，因此一个 32 步动作块中未执行的后 22 个预测不会进入之后的历史。

三种因果模式是：

- `interleaved`：新观测读取历史观测和历史动作的 KV。
- `vision_causal`：新观测只读取历史观测的 KV。
- `action_aggregator`：各观测独立编码，ActionDiT 聚合完整历史。

在所有模式中，当前动作 query 都能读取历史观测/动作、当前真实观测、同一块的未来视频条件，以及双向的当前动作块。独立加噪的视频流匹配目标始终对 ActionDiT 不可见。

标准的 9 帧 LIBERO clip 经 VAE 后变成 3 个 latent 帧：第一个是已经提交的真实观测，其余两个构成未来条件。训练时，这个条件采用干净或加噪 GT teacher forcing；rollout 时则从噪声生成。这种训练/运行差异是有意设计的，并通过 50% 高噪声条件进行正则化。生成块只是临时的逆动力学条件，不是 episode 记忆。

## 仓库、环境与资产

LeapBot-VA 的开发位于 [`DaojiePENG/FastWAM`](https://github.com/DaojiePENG/FastWAM) 的 `leapbot-va` 分支。新机器必须显式 clone 该分支；clone FastWAM 默认分支不会得到 LeapBot-VA 工作区。

```bash
mkdir -p /home/sheng/workspace
cd /home/sheng/workspace
git clone -b leapbot-va --single-branch \
  https://github.com/DaojiePENG/FastWAM.git leapbot-va
cd leapbot-va
git branch --show-current  # 必须输出：leapbot-va
git remote -v              # origin 必须是 DaojiePENG/FastWAM.git
```

可把原始 FastWAM 仓库单独注册为只读比较和上游同步来源：

```bash
git remote add upstream https://github.com/yuantianyuan01/FastWAM.git
```

不要用上游仓库替换 `origin`，也不要从 `main` 训练。每个正式 run 都要在 run contract 中记录准确的 `leapbot-va` commit。

此工作区内的命令预期由用户 `sheng` 执行：

```bash
cd /home/sheng/workspace/leapbot-va
uv venv --python /usr/bin/python3.10 .venv
uv sync --dev
source .venv/bin/activate
python scripts/download_leapbot_assets.py
export LEAPBOT_DATASET_STATS="$PWD/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
python scripts/precompute_text_embeds.py task=libero_leapbot_2cam224
```

下载器使用 FastWAM 官方 Hugging Face 仓库 `yuanty/LIBERO-fastwam` 和 `yuanty/fastwam`，不会读取 RLDS 数据。

## 训练阶段

生产训练路径采用与运行时同构的因果注意力和 `incremental_full_bptt`：当前重规划之前的每个真实观测/动作块都按时间顺序执行并保留在计算图中。不存在 history gate，也不存在 detached prefix。对当前块，独立的未来条件以 0.5 概率采用干净 GT 视频，以 0.5 概率采用高噪声 GT 视频（scheduler shift 前 `u in [0.5,1.0]`），对应 LingBot-VA 的 noisy-condition teacher forcing。动作 loss 会通过该条件的 K/V 反传，而不会通过视频流目标的噪声反传。

模型保留块内原生 RoPE，同时用相对于这些局部坐标的已学习 episode clock 表示进度；因此即使 clock 参数训练后，block 0 仍是严格的位置 no-op。这不会冻结共享 DiT 权重，所以 release 行为的保持程度要用配对 H0 样本单独测量。所有比较 run 都使用相同的 FastWAM release 初始化、数据 split、更新次数、global batch、scheduler 和 seed。launcher 会拒绝 dirty worktree，并把每份可恢复状态绑定到准确 commit、release 权重、数据统计、拓扑、optimizer、时间与历史配置的哈希。

正式训练把 history-VAE chunk 固定为 1。每个真实观测都使用与 rollout 相同的 batch-one、T=1 VAE 调用；更早的 chunk-2 近似和所有 packed-attention run 都已作废，不能作为结果使用。

```bash
# 配对筛选之后，用固定观测、固定 timestep、固定噪声进行审计；
# 历史分别使用正确、masked 和跨 episode 打乱版本。
bash scripts/screen_learning_rate.sh
bash scripts/audit_learning_rate.sh
LR_SELECTION_MANIFEST=<lr-selection.json> \
  bash scripts/screen_h0_retention.sh
LR_SELECTION_MANIFEST=<lr-selection.json> \
  bash scripts/audit_h0_retention.sh

# 阶段 1：配对 LR 审计选定学习率后，顺序训练全部三种模式；
# 使用完整 episode prefix、D30、BF16、相同的 8 GPU 拓扑/global batch，
# 以及 5% warmup + cosine schedule。
LR_SELECTION_MANIFEST=<lr-selection.json> \
  H0_SELECTION_MANIFEST=<h0-selection.json> \
  bash scripts/train_causal_modes.sh

# 阶段 2：从获胜的 D30 历史 checkpoint 初始化，再训练各出口。
SOURCE_TRAIN_ROOT=/path/to/d30_root MODE=<winner> SOURCE_STEP=<step> \
  MAX_STEPS=<steps> \
  bash scripts/train_multi_exit.sh
```

短的 0–8 或 0–16 窗口只是在代码层支持的受控消融，不是主配方，也不能代替完整 prefix 训练。标准生产流水线没有短窗口 launcher，当前正式结果也不运行或报告这些消融。

发布的 LIBERO 训练 episode 提供到 H=50 的真实 prefix。70-block 设置只是容量与推理外推上限；H=51..69 并没有作为观测到的训练历史出现，报告时必须明确这一点。

标为 `kvret0/8/16/32/full` 的推理消融限制的是物理保留的 KV 块数，而不是严格的信息窗口：较新的因果 KV 在计算时曾关注更老 prefix，因此仍可能编码后来已被淘汰块的信息。若要实现严格的最后 N 块信息消融，需要在每次淘汰后重放保留的原始观测/动作，这超出了在线 cache 设计。默认的完整 episode 配置不做淘汰。

阶段 3 的目标严格为 `L30 + (L8 + L16 + L24) / 3`；每个 `Ld` 都包含视频和动作流匹配 loss。

## 评测

```bash
python experiments/libero/eval_libero_single.py \
  --config-name sim_leapbot_libero \
  ckpt=/path/to/leapbot.pt \
  model.training_strategy=video_lora_action_full \
  model.video_lora.enabled=true \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  EVALUATION.memory.exit_depth=16

python experiments/leapbot/pareto.py evaluate_results/leapbot

# 训练完获胜的四出口 checkpoint 后，在隔离的结果目录中运行完整网格：
# D={8,16,24,30} × H={0,8,16,32,full}。
TRAIN_ROOT=/path/to/multi_exit_run MODE=<winner> FINAL_STEP=<step> \
  bash scripts/evaluate_pareto.sh
```

开发阶段对全部 10 个 `libero_10` 任务各跑 10 次；最终表则每任务跑 50 次。重规划延迟是从原始观测到命令的闭环测量，分别保留输入预处理、上下文条件、真实观测 prefill、想象视频设置/去噪/最终 KV prefill、动作历史物化/设置、动作去噪、命令后处理和已执行动作 KV 提交的耗时。

持久 cache 峰值包含观测后、动作提交前的临时状态；持久 KV、临时未来视频 KV 和 CUDA 总峰值分别报告。结果 fingerprint 对 LeapBot worktree、LIBERO revision、模拟器包版本、任务 BDDL、初始状态、运行时配置和 checkpoint 进行哈希。

Pareto 工具保留成功率/延迟/内存的整体非支配前沿，并将 FastWAM 作为比较项。默认 LeapBot 配置只能从启用了记忆的 LeapBot 行中，依据“成功率相差不超过 1 个百分点且置信区间重叠”的规则选出；FastWAM 永远不能被错误标记为 LeapBot 默认配置。

## 验证

```bash
PYTHONPATH=src python -m pytest tests -q
```

可选的真实 6B 验收工具包括 `scripts/validate_real_6b_runtime_training_equivalence.py`（验证训练/运行时 KV 与固定噪声 loss 等价）和 `scripts/full_prefix_smoke.py`（验证真实 prefix 的 optimizer 拓扑和容量保护）。它们是集群 preflight 工具，不是替代训练入口；准确命令和资源范围见[训练与复现手册](./docs/TRAINING_AND_REPRODUCTION.md)。

单元测试覆盖：未来视频 target 与动作 condition 的分离、动作梯度通过 condition 传播、临时 KV 生命周期、动作/观测状态转换、rollback/reset/capacity、准确的 context fingerprint、层级位置、三种模式 H=8 下 FP32/BF16 incremental 与 one-shot KV 等价、已执行夹爪动作重归一化、实例化前确定性 seed、resume-run contract、已训练出口强制检查、多深度输出和 Pareto 选择。

完整 6B H800 训练和 500-episode benchmark 需要 release 资产及训练完成的 LeapBot checkpoint；单元测试不会伪造这些结果。

未来视频条件引入之前的 6B 测量仅作为已被取代的历史报告保留在 [`reports/SMOKE_H800.md`](./reports/SMOKE_H800.md)。它们不能作为当前架构的容量、延迟或 loss 证据；接收集群必须在已提交代码上重新运行文档规定的 probe。
