# PCH 前向优化与 Task Config 评测

本文记录 Packed Causal History（PCH）训练的前向优化、Flex mask 缓存键修改、checkpoint 兼容规则，以及基于 Hydra task config 的单 checkpoint 评测入口。

## 1. 背景

原 causal-history loss 会把一个 micro-batch 拆成逐样本、逐历史块的串行 DiT/VAE 调用。PCH 已将历史 DiT 合并为一次 fixed-padding packed-history forward，但初版仍存在两个前向热点：

- `history_vae_batch_chunk_size=1` 使每个有效历史帧、anchor 和 current-real 都单独执行 VAE；
- Flex mask LRU 使用 CUDA `key_valid_mask` 生成 CPU bytes，cache lookup 会触发 GPU→CPU 同步。

本次修改只处理这两个前向问题。activation checkpoint、ZeRO-2、反向重计算和训练 metric gather 均未修改。当前 trainer 的 metric gather 仍在每个 optimizer step 执行，`log_every` 只控制最终日志和 W&B 写入。

## 2. VAE batch chunk

PCH 训练默认配置改为：

```yaml
model:
  history_vae_batch_chunk_size: 4
```

history、episode anchor、current-real 三条编码路径统一读取该配置。编码函数只在 batch 维合并独立样本，每个输入仍严格为 `T=1`，因此不会在 VAE temporal axis 上混入不同历史 block。

对于每卡 `B=4, W=8` 的满历史 batch：

```text
旧版：32 history + 4 anchor + 4 current = 40 次 VAE forward
新版： 8 history + 1 anchor + 1 current = 10 次 VAE forward
```

chunk 必须为正整数。基础模型、旧 iterative/strict-window 配置仍默认使用 1；只有 PCH task 和对应 PCH eval config 默认使用 4。

chunk 是执行优化参数而不是学习语义。checkpoint 继续记录它作为 provenance，但加载时不要求 checkpoint chunk 与当前运行配置相等，也不会用 checkpoint 值覆盖当前配置。因此旧 chunk=1 PCH checkpoint 可由 chunk=4 配置加载。

## 3. Flex validity signature

Flex attention 的 causal/padding mask 语义没有变化。三种 causal mode、invalid-query 自对角保护、Dense/Flex 对照和 LRU 容量均保持原样。

新的数据流为：

```text
Dataset CPU history_valid/anchor_valid
→ pch_slot_validity bytes
→ DataLoader/Accelerate 原样携带 bytes
→ 按实际 video/action token geometry 在 CPU 展开
→ PCHLayout.validity_signature
→ Flex BlockMask LRU key
```

展开后的 signature 与旧 `layout.key_valid_mask.contiguous().numpy().tobytes()` 逐字节一致。Flex cache lookup 和 H=0 guard 不再读取 CUDA tensor。CUDA Flex layout 如果缺少 CPU signature 会直接报错，避免以后静默重新引入 D2H 同步。

packed-replay 推理端根据 Python replay history、left padding 和 anchor 状态生成同样的 signature，不需要把 CUDA validity tensor 拷回 CPU。

主要实现位置：

- `src/leapbot_va/pch.py`：CPU metadata/signature、PCHLayout 和 Flex LRU key；
- `src/leapbot_va/data.py`：在 Dataset CPU 阶段产生 `pch_slot_validity`；
- `src/leapbot_va/training.py`：chunk=4 三路 VAE 与训练端 signature；
- `src/leapbot_va/models/leapbot.py`：packed-replay 推理 signature 和 checkpoint chunk 兼容。

## 4. 配置

PCH 训练 task：

```text
configs/task/libero_leapbot_pch.yaml
```

PCH 评测基配置：

```text
configs/sim_leapbot_libero_pch.yaml
```

`sim_leapbot_libero_pch` 默认对应 PCH 的 `packed_replay` 在线路径。下面的新启动器为了与指定的严格窗口评测合同一致，会显式覆盖为 `strict_replay`，用于与旧版评测口径进行严格对照。

## 5. Task Config 单 checkpoint 评测

入口：

```text
scripts/evaluate_pch_checkpoint.sh
```

启动器只接受 checkpoint；其余评测行为完全由 `sim_leapbot_libero_pch.yaml` 控制：

```bash
CKPT=/path/to/step_005000.pt bash scripts/evaluate_pch_checkpoint.sh
```

它通过 Hydra 启动：

```text
--config-name sim_leapbot_libero_pch
task=libero_leapbot_pch
```

PCH 评测配置固定声明严格窗口合同：

```text
EVALUATION.memory.enabled=true
EVALUATION.memory.history_storage_mode=strict_replay
EVALUATION.memory.history_window_blocks=8
EVALUATION.memory.max_history_blocks=70
EVALUATION.replan_steps=10
EVALUATION.action_horizon=32
```

`save_rollout_video: true`、trial 数、causal mode、LoRA、dataset stats、memory 参数和输出目录都由 YAML 决定，启动器不再传入这些 Hydra override。

并发复用 FastWAM 的 `run_libero_manager.py` / `run_libero_parallel_test.sh` 动态调度逻辑：

```yaml
MULTIRUN:
  task_suite_names: [libero_10]
  num_gpus: 4
  max_tasks_per_gpu: 3
```

也就是说最多可同时运行 12 个 task；LIBERO-10 的 10 个 task 会按空闲 GPU 动态分配，而不是每张卡一次只跑一个串行 worker。每个 worker 保持 `sim_leapbot_libero_pch` 作为 config-name，因此不会回退到默认 `sim_libero`。

输出目录由 YAML 中的时间戳表达式生成，例如：

```text
evaluate_results/pch_task_config/interleaved/20260811_120000/
```

`history_vae_batch_chunk_size` 仅作为 checkpoint provenance 保留，不属于评测数值合同。旧 chunk=1 PCH checkpoint 可以在当前 chunk=4 配置下加载。

## 6. 验证

实现阶段已覆盖：

- chunk=1 与 chunk=4 的输出等价、H=0、tail chunk 和 `T=1`；
- `B=4/W=8/chunk=4` 的 VAE 调用数为 10；
- CPU bytes 与旧 key-validity bytes 精确一致；
- 三种 causal mode 的 Dense/Flex KV 与梯度一致；
- chunk 不一致 checkpoint 可加载且不覆盖当前配置；
- config-driven PCH 启动器、Hydra task config、每卡多任务调度和 worker config-name 透传。

完整训练级吞吐和 GPU 利用率 A/B 仍需在目标训练集群上测量。建议固定相同 checkpoint、batch、W 和 GPU，以 Nsight/torch profiler 分别记录 VAE、packed-history、current-real、future-condition、action 和 supervision 前向耗时。
