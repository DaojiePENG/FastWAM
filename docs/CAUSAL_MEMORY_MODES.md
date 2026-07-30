# LeapBot-VA 因果 KV 记忆训练与推理契约

本文档描述当前唯一正式实现。它不包含门控历史分支、离线 KV、
detached prefix 或预测视频回灌。任何实验结论都必须来自对应 checkpoint
和真实 LIBERO rollout，不能由本文档推断。

## 1. 闭环定义

将 episode 按固定的 10 个控制步划分为重规划块
`b = 0, 1, ...`。每块包含：

- 重规划边界的一张真实观测 `o_b`；
- 当时的 proprio `p_b`；
- 长度 32 的预测动作块；
- 最多 10 个真正下发给环境的动作 `a_b`。

持久化内存严格按以下顺序增长：

```text
[real_obs_0 KV, executed_action_0 KV,
 real_obs_1 KV, executed_action_1 KV, ...]
```

动作扩散中的 noisy token、预测但未执行的后 22 个动作、未来视频
latent、视频输出头结果和 VAE decode 结果都不会进入内存。

## 2. 在线事务状态机

每个 episode 拥有独立的 `LeapMemoryState`，模型不保存全局 cache：

1. `infer_action(..., memory)` 检查 prompt、clock、深度和容量；
2. 当前真实图像只做一次 VAE encode 和 VideoDiT prefill；
3. 逐层提交当前真实图像的 K/V；
4. ActionDiT 对不可变历史做动作去噪，返回 32 步预测；
5. 环境后处理并执行最多 10 步；
6. 将实际下发命令反归一化/再归一化到模型空间；
7. `commit_executed_actions` 只编码并提交这批真实执行动作；
8. episode 结束时 `reset_memory` 释放所有 K/V。

在动作提交前重复观测、漏提交后进入下一块、episode 内切换 prompt、
切换 causal mode、切换时间契约或超出容量都会失败。手工构造的 memory
也必须通过相同兼容性检查。

`max_history_blocks=70` 表示一个 episode 总共允许 70 个重规划块：
block 0 到 69，共覆盖 700 个控制步；第 71 次观测被拒绝。

## 3. 三种可控信息流

三种模式只改变视频 query 对更早 segment 的可见性。当前动作始终读取
全部已经提交的真实历史和当前真实观测。

| query | `interleaved` | `vision_causal` | `action_aggregator` |
|---|---|---|---|
| 新视频读取更早视频 | 是 | 是 | 否 |
| 新视频读取更早动作 | 是 | 否 | 否 |
| 当前动作读取全部历史 | 是 | 是 | 是 |
| 当前动作读取当前真实帧 | 是 | 是 | 是 |
| 当前动作读取未来视频监督 | 否 | 否 | 否 |

同一 action block 内部为双向 attention。旧历史不能读取当前或未来 token。
`action_aggregator` 不是无历史模型；它是“独立观测编码、历史条件动作聚合”。

## 4. 正式训练数据

`LeapRobotVideoDataset(full_episode_history=true)` 只在真实的 10 步重规划
边界产生样本。对当前 block `b`，样本包含：

- `o_0 ... o_{b-1}`：每块一张真实边界图像；
- 每个历史块随后记录的 10 个 demonstration 动作；
- `p_0 ... p_b`：各块自身的 proprio；
- 当前真实图像、原 FastWAM 未来视频监督和 32 步目标动作；
- episode step、绝对 block 位置和 padding mask。

历史图像稀疏读取，历史动作逐控制步密集读取。底层 loader 如果在 I/O
失败后返回了另一个随机 index，数据集立即失败，绝不把原 index 的因果
元数据附到另一条轨迹。历史窗口不能跨 episode。

训练历史动作是 demonstration 中记录的已执行动作；在线历史动作是当前
策略真正下发的动作。二者存在 teacher-forcing/on-policy 分布差异，必须
通过真实 rollout 评估，不能只用训练 loss 代替。

官方 LIBERO 数据中最长轨迹只提供到约 H=50；70 是运行时容量和 OOM
保护目标，不代表训练见过 H=51...69。

## 5. Packed full-BPTT

主训练模式固定为 `packed_full_bptt`。每次 forward 将一个 batch 内所有
有效历史观测、全部历史动作、当前视频监督和当前 noisy action 放入同一
MoT 前向：

- 所有真实历史块都保留在计算图中；
- 不 detach 历史、不淘汰历史、不读取离线 cache；
- causal mask 实现第 3 节的三种信息流；
- 当前 action 对当前未来视频 token 的 mask 恒为 false；
- 只对当前未来视频和当前 32 步动作计算 flow-matching loss；
- 当前 loss 可以经 attention 反传到全部有效历史 K/V 表征。

固定容量张量仅用于 collation。前向会裁掉 batch 中对所有样本均无效的
右侧 padding，但不会裁掉任何样本的真实 prefix。分布式 sampler 按精确
历史长度分桶以减少各 rank 的 padding/同步等待；它只改变 permutation，
不改变样本内容。

## 6. Proprio 与语言隔离

packed context 为语言 token 加上每块一个 proprio token。逐 query 的
cross-attention mask 强制：

- block `i` 的视频和动作只读取语言与 `p_i`；
- padding block 不成为有效 key；
- 当前 block 只读取 `p_b`。

在线每个 segment 在产生 K/V 时使用完全相同的语言和对应 proprio。
context-only API 对完整 context 字节和 mask 做 SHA-256；proprio 不参与
episode prompt fingerprint，因为它应随重规划变化。

## 7. 时间位置

为保持 FastWAM release 的零历史能力，原生 RoPE 坐标在每个块内重置：

- 每张历史真实观测的局部视频位置为 0；
- 当前 33 帧监督经 VAE 后的局部 latent-frame 位置为 `0...8`；
- 每个历史 10 步动作的局部位置为 `0...9`；
- 当前预测动作的局部位置为 `0...31`。

episode 进度由单独的解析正弦特征和零初始化投影注入：

- 视频使用绝对重规划 block id；
- 动作同时使用绝对 block id 和真实控制步 id；
- 当前预测动作位置从 `b*10` 开始；只有实际执行的前缀会成为下一轮历史。

零初始化时，D30/H0 packed 路径与 FastWAM 原路径在 BF16 容差内一致。

## 8. 训练目标与出口

视频和动作继续使用 FastWAM release 代码的 scheduler、FM target、timestep
weight、padding normalization 及 `lambda_video=lambda_action=1`。未来视频
只提供训练监督；在线 memory 推理不创建未来视频 latent。

正式三模式比较先只训练 D30。选定模式后再训练
`D={8,16,24,30}`，目标为：

```text
L30 + (L8 + L16 + L24) / 3
```

每个 `Ld` 都含原视频和动作 FM loss。浅层视频 head 仅在训练时计算视频
监督，在线动作推理不会调用视频 head。

## 9. 优化与公平比较

当前微调阶段从同一 FastWAM release 初始化：ActionDiT 全量训练，
VideoDiT self-attention 使用 rank-16 LoRA，proprio encoder 与层级位置投影
训练；没有 gate。三种 causal mode 必须共享：

- 数据 split、sampler seed、初始化和 normalization stats；
- 模型构造前使用同一随机种子，并以 commit/config/checkpoint 哈希绑定
  每个可恢复训练状态；
- global batch、optimizer updates、AdamW 参数、梯度裁剪；
- 5% linear warmup + cosine scheduler；
- D30、32 步 horizon、10 步 replan 和 BF16。

历史图像的 VAE 输入始终是独立 `T=1`。正式 BF16 配置只允许 chunk 1
或经真实 Wan/H800 固定噪声 loss 验证通过的 chunk 2；chunk 4 的 action
loss 差超过 `1e-3`，因此不会进入正式比较。

FastWAM 官方 release 实际完成 21,700 updates、global batch 128、LR 1e-4。
LeapBot 是从该权重做历史适配，不能把很短的 LR screen 当作最终收敛或
成功率结论。

## 10. 必须通过的验证

代码级验收包括：

- 三种 mask 的允许关系和 future-video 零梯度；
- H0 packed 与原 FastWAM 数值一致；
- 三模式 H=8 的 packed 一次前向与逐块 runtime KV 前向等价（FP32/BF16）；
- 预测但未 commit 的动作不增加 action cache；
- 70 块/700 动作边界与第 71 次拒绝；
- prompt、clock、reset、rollback、checkpoint 和 resume 契约；
- LIBERO gripper 后处理的精确逆映射和重新归一化；
- 6B H50 真实 prefix、H70 合成容量的 forward/backward/OOM smoke；
- profiler 中 memory inference 无未来视频去噪、视频 head 或 VAE decode。

效果验收必须使用固定初始状态的 LIBERO-Long rollout，并分别报告成功率、
完成步数、P50/P95 observation/action/commit 延迟、峰值显存与 cache 大小。
冻结 loss 诊断只用于定位问题，不能替代 rollout 成功率。

## 11. 实现索引

| 契约 | 文件 |
|---|---|
| memory/state machine | `src/leapbot_va/memory.py` |
| packed mask、full-BPTT loss | `src/leapbot_va/training.py` |
| full-prefix LeRobot 数据 | `src/leapbot_va/data.py` |
| 在线 infer/commit/checkpoint | `src/leapbot_va/models/leapbot.py` |
| 层级时间位置 | `src/leapbot_va/positions.py` |
| LIBERO 实际动作回写 | `src/leapbot_va/libero.py` |
| LIBERO rollout | `experiments/libero/eval_libero_single.py` |
| 单模式训练入口 | `scripts/run_hierarchical_raw_v1_peft_5k.sh` |
| 三模式受控比较 | `scripts/run_causal_full_bptt_comparison.sh` |
| 固定噪声诊断 | `scripts/history_stratified_loss.py` |
| checkpoint 验证 | `scripts/validate_leapbot_checkpoint.py` |
