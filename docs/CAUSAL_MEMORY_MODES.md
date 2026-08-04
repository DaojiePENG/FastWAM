# LeapBot-VA：严格真实历史窗口与三种 Causal 模式

## 1. 目标

LeapBot-VA 在 FastWAM 的 VideoDiT/ActionDiT 联合建模上加入跨轮真实历史，同时避免把预测视频或未执行动作写入长期记忆。v7 的正式记忆不是无限前缀 KV，而是：

```text
M_t = optional(V0) + latest_W_completed[(V_i, A_i^executed)]
```

默认 `W=8`。`V_i` 是重规划边界的真实观测 latent，`A_i^executed` 是归一化模型空间中与实际下发命令一一对应的 10 步动作。

每轮计算：

```text
M_t -> current real V_t -> transient future-video C_t -> noisy A_t
```

`C_t` 帮助动作预测，但只活在当前调用；动作提交后，memory 只增加 `V_t` 和真正执行的动作。

## 2. 为什么必须 replay，而不是只删 KV

第 k 层某个较新块的 KV 已经通过 attention 吸收更早块的信息。若在线端只删除最老 KV tensor，保留的较新 KV 仍携带窗口外信息。这是“有界物理缓存”，不是严格信息窗口，无法与训练的最近 W 块对齐。

v7 保存最近真实块的单帧 VAE latent、对应时刻的 context/proprio 和实际动作。下一轮重规划时清空旧 KV，从这些真实记录按时间顺序重新计算：

```text
V_{t-W} -> A_{t-W} -> ... -> V_{t-1} -> A_{t-1} -> V_t
```

若 `t>W`，最前面再加入一次 video-only `V0` anchor。这样模型训练和真实 evaluation 拥有相同的信息边界，代价是每轮有界 O(W) prefix replay。

## 3. 早期窗口和 V0 anchor

训练张量固定为 W 个历史槽。episode 初期只有 H<W 个真实块时：

```text
[PAD ... PAD, V0/A0, ..., V(H-1)/A(H-1)]
```

PAD 槽的 observation、action 和 proprio 都为零，`history_valid_blocks=false`，不会生成 KV，也不能成为 attention key。禁止把第一帧复制 W 次并标成有效，因为那会人为放大 V0 的 softmax 权重并改变训练分布。

当当前 block 满足 `t>W` 时，最近窗口已经不含 V0。此时增加一次 `V0`：

```text
[V0 anchor] + [V(t-W)/A(t-W), ..., V(t-1)/A(t-1)]
```

anchor 只含真实视觉和它自己的 proprio 条件，不重复 A0。它保存任务初始状态身份，同时避免伪造一整窗重复历史。

## 4. 三种 causal 信息流

三种模式从同一 FastWAM release 初始化分别训练。

### 4.1 `interleaved`

- 新历史/当前视觉读取此前真实视觉和已执行动作；
- 历史动作读取此前全部真实 V/A 和本块真实 V；
- 当前未来视频读取 prefix 和当前真实 V；
- 当前动作读取全部真实 prefix、当前真实 V、当前未来视频 KV 和当前动作块。

这是视觉与动作最强耦合的跨轮世界模型。

### 4.2 `vision_causal`

- 新视觉只读取此前真实视觉，不读取历史动作；
- ActionDiT 仍读取全部历史视觉、历史动作、当前真实视觉和当前未来视频；
- 当前动作块内部双向。

它隔离了“历史动作是否应该改变视觉表征”这一因素。

### 4.3 `action_aggregator`

- 每张真实历史视觉独立编码；
- 当前未来视频只读取当前真实视觉；
- ActionDiT 汇聚 V0 anchor、最近全部真实 V/A、当前真实 V、未来视频 KV 和当前动作块。

它最接近“稳定的独立视觉 encoder + 历史 ActionDiT aggregator”，也是 v7 首先建议 smoke/训练的模式。

## 5. 允许关系

令 `i<t` 是已完成历史块，`C_t` 是当前未来视频，`A_t` 是当前 noisy action chunk：

| Query | interleaved keys | vision_causal keys | action_aggregator keys |
|---|---|---|---|
| `V_i` | 更早 V/A + 自己 V | 更早 V + 自己 V | 自己 V |
| `A_i^executed` | 截止本块的 V/A | 截止本块的 V/A | 截止本块的 V/A |
| `V_t` | 历史 V/A + 自己 | 历史 V + 自己 | 自己 |
| `C_t` | 历史 V/A + `V_t` + `C_t` | 历史 V + `V_t` + `C_t` | `V_t` + `C_t` |
| `A_t` | 历史 V/A + `V_t` + `C_t` + `A_t` | 同左 | 同左 |

硬约束：

- 历史 query 不能读当前或未来块；
- 当前真实 V 不能读 `C_t` 或 `A_t`；
- video flow target 的独立噪声 token 不能进入 action prefix；
- 当前 action chunk 内部双向；
- future condition KV 不跨轮；
- padding 不能作为有效 key。

## 6. Proprio、语言与位置

每个真实 V/A block 使用它自己的 proprio token。语言 prompt 在 episode 内不变；prompt/context 变化会要求 reset。Cross-attention mask 只允许 query 读取语言和对应时刻 proprio，禁止历史块读取当前 proprio。

FastWAM 的块内 RoPE 原点保持不变。LeapBot 另外注入：

- 视觉的 episode 绝对重规划 block index；
- 动作的真实控制步绝对 index；
- 动作所属重规划 block index。

窗口 replay 不重写这些绝对时钟，因此训练样本即使只含最近 W（默认 W8），也保留它们在完整 episode 中的位置。窗口长度在一个 checkpoint 内固定，不能中途切换。

## 7. 训练目标

历史块以 clean timestep 0 编码并保留 attached KV。当前目标分成两条 video branch：

1. `C_t`：clean/noised GT future latent，逐层 KV 供 ActionDiT 读取；
2. video flow target：独立 timestep/noise，只用于 video FM loss。

Action loss 会通过 `C_t` 的 KV 回传到 VideoDiT。Condition curriculum 先用 clean，再把 noised 概率升至 0.5；没有门控层。

多出口一次 30 层前向取得 D8/D16/D24/D30：

```text
L = L30 + (L8 + L16 + L24) / 3
Ld = video_flow_loss_d + action_flow_loss_d
```

D30 使用原 FastWAM heads；浅层使用独立轻量 exit heads。每个 episode 的推理深度固定。

## 8. 在线事务状态机

```text
EXPECT_OBSERVATION
  -> rebuild strict real prefix
  -> append current real observation KV
  -> predict 32 actions
EXPECT_ACTION_COMMIT
  -> postprocess environment commands
  -> convert actually executed 10 commands back to model space
  -> append executed-action KV and replay record
  -> EXPECT_OBSERVATION
```

以下情况 fail-fast 或要求 reset：

- 上一轮未提交动作就送入新观测；
- 提交多于 10 步；少于 10 步只允许作为 episode 的 terminal block，之后必须 reset；
- prompt/context、causal mode、depth 或 W 在 episode 中改变；
- episode 超过 70 blocks；
- action/postprocess 后出现非有限值；
- checkpoint 的训练窗口与 evaluation 窗口不一致。

任意推理或提交异常都会回滚 KV segments、replay records、anchor、pending observation、episode clock 和 phase。

## 9. 实现索引

- `src/leapbot_va/data.py`：最近窗口、左 padding、V0 anchor 和同 episode 数据检查；
- `src/leapbot_va/training.py`：逐 segment attached-BPTT、condition branch、分项指标和严格窗口验证；
- `src/leapbot_va/memory.py`：事务状态、`ReplayBlock`、KV/replay 容量与 reset；
- `src/leapbot_va/models/leapbot.py`：在线 strict replay、未来视频条件、动作预测/提交和 checkpoint 合同；
- `src/leapbot_va/positions.py`：块内与 episode 绝对时钟；
- `configs/model/leapbot.yaml`：默认 W8 与 condition curriculum；
- `configs/task/libero_leapbot_2cam224.yaml`：LIBERO 数据窗口；
- `tests/test_packed_training_masks.py`：padding/anchor 数据测试；
- `tests/test_incremental_full_bptt_training.py`：严格窗口训练与梯度测试；
- `tests/test_runtime_temporal_positions.py`：在线 replay、V0 和未执行动作隔离测试。
