# LeapBot-VA 固定容量 Episode Memory：当前代码规范

> 本文只描述 episode-memory 分支当前已经实现的行为，不把尚未实现的设想写成既成事实。核心实现基线为 transition 级 updater 提交 8208c47。配置示例以当前正式设置 W=8、C=4、H=32×1024 为准；底层实现仍允许在约束内修改这些数值。

## 目标与基本边界

Episode memory 的目标是在不保存无限增长原始历史或完整 KV 的前提下，让每一段真实 observation–executed action–next observation 闭环交互都进入一个固定容量的远期状态。当前实现不使用 VLM、语言摘要、关键帧标签、显式任务阶段、skill boundary、任务进度监督、top-k admission 或人工重要性规则。

系统维护四种作用不同的状态：

| 状态 | 当前代码中的形式 | 容量 | 是否直接用于模型推理 |
|---|---|---:|---|
| PCH | 最近 W 个 ReplayBlock，按需重建逐层精确 KV | 最多 W blocks | 是 |
| Q | 已退出 PCH、等待交接的 ClosedReplayBlock | 最多 C-1 blocks；内部事务中可短暂达到 C | 没有独立 reader，但会和 PCH 一起重建为精确历史 |
| H | episode_state: [B,N,D]，默认 N=32,D=1024 | 固定 | 是，经过独立逐层 memory reader |
| F / H0 | 首帧的固定逐层视觉 KVSegment | 一份 | block 0 离开 Q+PCH 后作为只读视觉前缀 |

F 是 episode 初始条件，不计入 transition 历史分区。H、Q、PCH 对已经完成的 block 历史互不重叠。

## 时间定义与严格历史分区

一个 block 记录当前真实 observation 和最终实际执行的动作：

\[
b_i=(o_i,a_i^{exec}).
\]

只有下一张真实 observation 到达后，上一条 transition 才闭合：

\[
\tau_i=(o_i,a_i^{exec},o_{i+1}).
\]

ActionDiT 的预测结果不会自动进入历史。调用方必须把真正发送给控制器的动作重新传给 commit_executed_actions；直到该提交完成，并且下一张真实 observation 到达，\(\tau_i\) 才有资格进入长期更新。

设当前决策前已经提交了 t 个 action blocks，当前真实 observation 是 \(o_t\)。定义：

\[
e(t)=\max(0,t-W),
\]

\[
m(t)=C\left\lfloor\frac{e(t)}{C}\right\rfloor.
\]

当前代码中的分区是：

\[
H_t\equiv [0,m(t)),\qquad
Q_t\equiv [m(t),e(t)),\qquad
P_t\equiv [e(t),t).
\]

因此：

\[
|Q_t|<C,\qquad Q_t+P_t=[m(t),t),
\]

而模型可读取的精确历史长度满足：

\[
|Q_t+P_t|\le W+C-1.
\]

代码里的 episode_partition 使用变量名 q 表示这里的 \(m(t)\)，不是交接缓冲区对象 Q。

以 W=8、C=4、t=13 为例：

~~~text
H   = blocks [0,4)
Q   = block  [4,5)
PCH = blocks [5,13)
精确 Q+PCH = blocks [4,13)，共 9 个
当前真实 observation = o13
~~~

当第四个退出 block 进入 Q 时，Q 在事务内部短暂达到 C，随即生成 chunk 算子、更新 H 并清空 Q。模型进行下一次正式推理时看到的 Q 长度仍然小于 C。

## 运行时保存的具体对象

ReplayBlock 保存：

~~~text
block_index
observation_latents
context / context_mask
executed_actions
~~~

这里的 context 是推理时的语言 context 加该 observation 对应的 proprio token。所有张量在写入 replay 状态时都会 detach。

ClosedReplayBlock 在 ReplayBlock 基础上增加：

~~~text
next_observation_latents
next_context / next_context_mask
~~~

因此 updater 得到的是完整真实 transition，而不是预测 future video。

LeapMemoryState 还保存：

~~~text
episode_state                 # 当前 H
initial_episode_state         # reset 时恢复的 learned-empty H
pch_closed_blocks             # 仍在 PCH 中但已经闭合的 transitions
handoff_blocks                # Q
episode_anchor                # 首帧 raw ReplayBlock
episode_anchor_segment        # 固定首帧逐层视觉 KV
pending_observation_latents   # 已观察但动作尚未提交的 observation
phase                         # EXPECT_OBSERVATION / EXPECT_ACTION_COMMIT
~~~

MemorySnapshot 覆盖上述所有可变字段。infer_action 和 commit_executed_actions 都在事务失败时 rollback；Q 满、H 更新、Q 清空也是一个原子事务。

## 在线推理的实际状态机

### 当前真实 observation 到达

infer_action(o_t) 首先 VAE 编码当前真实图像，然后调用 close_previous_transition。若此前已经提交 \(a_{t-1}^{exec}\)，此时用 \(o_t\) 闭合 \(\tau_{t-1}\)。

闭合 transition 先进入 pch_closed_blocks。若当前 replay PCH 已超过 W，最老的 ReplayBlock 与对应闭合 transition 一起退出 PCH，其中 transition 被移入 Q。

### Q 满 C 后先更新 H

若 Q 达到 C，_update_episode_memory：

1. 从 Q 取出连续的 C 条闭环 transition；
2. 分别重建 C+1 个 observation 的 VideoDiT clean pre-DiT token；
3. 分别重建 C 个 executed-action 段的 ActionDiT clean pre-DiT token；
4. 调用 EpisodeChunkUpdater 得到一个 chunk 仿射算子；
5. 用 FP32 将该算子应用到旧 H；
6. commit_handoff 替换 H 并清空 Q。

更新发生在当前 observation 的 PCH 重建和正式 DiT 推理之前。

### 重建精确 Q+PCH

episode-memory 模式强制使用 packed_replay。每次 observation 到达后，代码从 raw replay 输入重新构造 Q+PCH，固定物理容量为：

\[
W+C-1.
\]

有效 blocks 右对齐，左侧补零并由有效位 mask 屏蔽。Video 和 action token 在一次 packed MoT forward 中按 causal mode 形成逐层 KV。

这次 packed rebuild 不读取 H。这样写入 H 的 clean transition 证据和精确历史 KV 都不会先吸收旧 H，再循环写回 H。

### 编码当前真实 observation

当前 \(o_t\) 不属于 PCH=[e,t)，而是作为当前 block 单独经过 prefill_expert_segment。它读取 causal mode 允许的精确历史，并在允许的模式下读取 H。得到的当前视觉 KV 暂存到 memory.segments，同时 raw latent/context 被 stage，等待真实动作提交。

### 生成 future video 与 action

future VideoDiT 使用当前真实 observation 和 causal mode 允许的历史产生当前 block 的瞬时 future-video condition。它的 KV 只在本次 infer_action 内提供给 ActionDiT，不写入 PCH、Q、H 或 F。

ActionDiT 读取精确历史、当前真实视觉、瞬时 future-video condition，并在所有 causal modes 下读取 H，输出候选动作。

### 只提交实际执行动作

候选动作不会自动成为历史。调用：

~~~python
model.commit_executed_actions(memory, actions_model_space)
~~~

后，packed-replay 模式只把实际执行动作写入 ReplayBlock，增加绝对 block/action 时钟并清空当前重建 KV。下一张真实 observation 到达时，这条 transition 才闭合。

若只提交少于 replan_steps 的动作，当前契约把它视为 episode 终止情形；继续接收下一 observation 会被状态时钟校验拒绝。

## Transition 级 prediction–correction updater

### 输入 token 的真实来源

在线推理中，Q 中的 raw observation/action 会重新经过现有 VideoDiT/ActionDiT 的 clean pre_dit，并应用绝对 block/action temporal position。这里不会运行 VideoDiT 或 ActionDiT transformer blocks，也不会读取 PCH KV 或 H。

EpisodeChunkUpdater._pool_tokens 随后立即对每个 observation 的全部空间 token、每个 action 段的全部 action token 做平均：

~~~text
video:  [B,C+1,Sv,Dv] -> [B,C+1,Dv]
action: [B,C,Sa,Da]   -> [B,C,Da]
~~~

平均结果分别经过 LayerNorm → Linear(updater_dim) → SiLU，默认 updater_dim=256。因此当前 updater 的写入证据是 clean pre-DiT 的全局平均表示，不保留原始空间 token 布局；这是当前实现事实。

### 每条 transition 独立生成算子

一个 chunk 被拆成：

~~~text
(o0,a0_exec,o1)
(o1,a1_exec,o2)
...
(oC-1,aC-1_exec,oC)
~~~

所有 transition 共用 updater 参数，但彼此不做跨 transition attention。代码把 batch 与 transition 维合并为 B*C 并行处理。

预测分支的每个 transition 只有两个 token：start-observation token 和 executed-action token。两者分别加入 start/action role embedding。默认 32 个 slot_queries 通过 MultiheadAttention 读取这两个 token，产生每个 H slot 的预测参数。

校正分支只读取真实 successor-observation token，并加入 successor role embedding。它同样用 32 个 slot queries 产生真实观测证据和校正增益。

updater 本身不再使用 chunk 内位置 embedding；但是输入 clean token 已经包含绝对 episode block/action temporal position。

### 固定 H 坐标与分组更新

默认：

~~~text
H slots N = 32
state_dim D = 1024
group_dim d = 16
groups G = 64
~~~

每个 slot 的 1024 维状态被拆成 64 个 16 维 group。对 transition \(i\) 的每个 slot/group，网络产生：

\[
a_i=1+0.01\tanh(\operatorname{head}_a),\qquad
b_i=0.01\tanh(\operatorname{head}_b),
\]

\[
c_i=\operatorname{normalize}(\operatorname{head}_c),\qquad
y_i=\operatorname{head}_y,\qquad
k_i=0.25\,\sigma(\operatorname{head}_k).
\]

其中 \(a_i,b_i,c_i\) 是 16 维，\(y_i,k_i\) 是标量。\(a_i,b_i,c_i\) 来自起点 observation 与 executed action 的预测分支；\(y_i,k_i\) 来自真实 successor observation 分支。

对输入 group state \(h_i\)，实际语义为：

\[
\widehat h_{i+1}=a_i\odot h_i+b_i,
\]

\[
\widehat y_i=c_i^\top\widehat h_{i+1},
\]

\[
h_{i+1}=\widehat h_{i+1}+k_i c_i(y_i-\widehat y_i).
\]

这是真实的 prediction–correction：算子参数不读取 H，但校正残差在算子应用时依赖被预测后的 H。

展开为仿射算子：

\[
T_i=(G_i,B_i),\qquad h_{i+1}=G_i h_i+B_i,
\]

\[
G_i=(I-k_i c_ic_i^\top)\operatorname{diag}(a_i),
\]

\[
B_i=(I-k_i c_ic_i^\top)b_i+k_i c_i y_i.
\]

单条 transition 的预测矩阵是对角的，校正是每个 16 维 group 内的 rank-1 erase/write；多个 transition 组合后，每个 group 的 G 可以成为完整的 \(16\times16\) 矩阵。不同 slots/groups 之间不发生矩阵混合。

初始化为：

~~~text
a_head = 0        -> a = 1
b_head = 0        -> b = 0
k_head.bias = -2  -> 小幅校正
c_head/y_head = 小随机值
~~~

### C 条 transition 如何得到 chunk 算子

仿射组合定义为：

\[
(G_b,B_b)\circ(G_a,B_a)
=
(G_bG_a,\;G_bB_a+B_b).
\]

它满足结合律但不满足交换律。updater 对 transition 维执行 FP32 Hillis–Steele inclusive scan，并取最后一个 prefix：

\[
T_j^{chunk}=T_{s+C-1}\circ\cdots\circ T_s.
\]

在线推理只将这个最终 chunk 算子应用到 H 一次。它与按顺序物化 C 个中间 H 并逐条更新数值等价。

## Episode 级 scan-BPTT 训练

### 数据契约

episode_memory_scan_bptt 要求：

~~~yaml
history_sampling_mode: full_prefix
history_window_blocks: null
full_episode_history: true
~~~

每个训练样本对应一个目标决策时刻 t，并携带从 block 0 到 t-1 的完整真实 observation/action 前缀。当前 dataset 最大前缀由 max_history_blocks 限制，正式配置为 70；超过该容量的 episode 会在数据构建时直接报错。

当前实现不是“一次加载完整 episode，然后对多个目标共享一次 scan”。每个 dataset item 只有一个目标时刻；如果同一 episode 的多个时刻分别成为样本，它们会分别重建各自前缀和 scan。

### 训练时选择正确的 H/Q/PCH

对 batch 中每个目标 t，训练代码计算：

\[
e=\max(0,t-W),\qquad m=C\lfloor e/C\rfloor.
\]

只取 blocks [0,m) 生成 H；[m,t) 被重新打包成精确 Q+PCH。物理 exact slots 固定为 W+C-1 并右对齐。

因此 Q 未满时不会提前进入 H。训练 target 使用的状态与在线推理相同：

~~~text
H       = scan(blocks [0,m))
Q + PCH = exact blocks [m,t)
current = real observation o_t
~~~

### 并行生成 transition/chunk 算子

训练先对 [0,m] 的 observation 和 [0,m) 的 executed actions 生成 clean pre-DiT token。该步骤位于 torch.no_grad() 中，结果再次 detach；历史 VAE/pre-DiT 激活不会被 updater 梯度保留。

输入被切成不重叠的 C-transition chunks。相邻 chunks 只共享边界 observation：

~~~text
chunk 0: o0,a0,...,aC-1,oC
chunk 1: oC,aC,...,a2C-1,o2C
~~~

共享 observation 分别承担上一 transition 的真实结果和下一 transition 的起点，不代表 transition 重复。

每条 transition 算子并行生成，先在 chunk 内扫描成 chunk 算子。不同 batch rows 的有效 chunk 数可以不同；无效 padded chunk 在进入 episode scan 前被替换成恒等算子，辅助损失也通过 chunk_valid_mask 屏蔽。

### Episode 外层 prefix scan

得到 J 个 chunk 算子后，associative_affine_scan 在 chunk 维执行 FP32 Hillis–Steele inclusive scan。每个 stage 可用 non-reentrant activation checkpoint。它支持非 2 次幂 chunk 数，并返回：

\[
H^{(1)},H^{(2)},\ldots,H^{(J)}.
\]

训练按每行实际 chunk_count 选择最终合法 prefix；零 chunk 时使用 learned empty state。

~~~text
内层 scan：C 条 transition -> 1 个 chunk 算子
外层 scan：J 个 chunk 算子 -> 所有 chunk 边界的 H prefix
~~~

二者都只改变结合括号，不改变时间顺序。

### Prediction–correction 辅助损失

diagnostics 保存每条 transition 的 a、b、c、y、k 以及 transition_matrix/transition_bias。

为了给每条 transition 使用正确的输入状态，辅助损失从每个 chunk 的起点 H 出发，对 transition 算子再做一次内部 prefix scan，构造每条 transition 更新前的 \(h_i\)。

当前硬编码权重为：

~~~text
prediction loss weight = 0.1
correction loss weight = 0.1
~~~

损失分别约束：

\[
c_i^\top(a_i\odot h_i+b_i)\approx y_i,
\]

以及校正后同一观测方向与 \(y_i\) 一致。它不要求完整 H 等于视觉 embedding。

### 主 WAM 损失与梯度

扫描得到的 H 转为模型 dtype 后进入当前目标的 VideoDiT/ActionDiT memory reader。主损失仍是 VideoDiT flow loss 与 ActionDiT flow loss，总损失再加 prediction–correction auxiliary loss。

clean updater 输入停止梯度。用于主 WAM target 的 exact Q+PCH packed forward 不在 no_grad 中：memory-only 阶段主干被冻结，联合阶段则允许梯度经过这段有界精确历史流向开放的 ActionDiT 与 Video LoRA。updater、transition 内层 scan、episode 外层 scan、learned empty H、memory reader 和主 WAM 中按训练策略开放的参数保持梯度。

当前有两套任务配置：

- libero_leapbot_episode_memory.yaml：training_strategy=episode_memory_only。训练器先冻结全模型，再开启整个 episode_memory 模块；通用 trainer 还会额外开启 proprio_encoder。
- libero_leapbot_episode_memory_joint.yaml：training_strategy=video_lora_action_full。训练 episode memory、temporal positions、完整 ActionDiT、VideoDiT self-attention LoRA，以及通用 trainer 开启的 proprio encoder；VideoDiT 基础权重保持冻结。

memory reader 的 gate 初始化为零，因此刚开始时主 WAM 行为不受 H 影响；gate 本身能收到主损失梯度，而 updater 同时由辅助损失启动。

## H 如何被 VideoDiT / ActionDiT 读取

EpisodeMemoryReader 是独立的逐层 cross-attention-like 分支，不是把 H token 拼进 PCH self-attention KV。

每层读取流程是：

1. 对 H 做共享 LayerNorm；
2. 经过该层专属 low-rank adapter：
   \[
   \widetilde H_l=\operatorname{LN}(H)+U_l\operatorname{SiLU}(D_l\operatorname{LN}(H));
   \]
3. 用跨层共享线性层把 \(\widetilde H_l\) 投影为 memory K/V；
4. 用 modality-specific query projection 把当前 VideoDiT 或 ActionDiT token 投影为 Q；
5. 执行 scaled dot-product attention；
6. 经过 modality-specific output projection；
7. 乘该 modality、该 layer 的 tanh(gate)；
8. 作为 residual 加到该 transformer block 的输出 token。

默认 low-rank adapter rank 为 64。adapter_up 和所有 layer gates 初始化为零。

memory residual 在每层 transformer block 之后加入，因此会影响下一层 token 和最终输出；它不改变当前层已经生成的 PCH/self-attention K/V，也不会被写回 PCH rebuild。

## causal mode 的当前真实路由

video_reads / action_reads 普通开关已经不存在。H reader 和 PCH 路由只由 causal_mode 决定。

| causal_mode | 历史 Video token 在 packed PCH 内读取 | 当前/未来 VideoDiT 读取的精确历史 | VideoDiT 读取 H | ActionDiT |
|---|---|---|---|---|
| interleaved | 更早 video + 更早 action；同 block video | Q+PCH 的 video + action | 是 | 读取全部精确 video/action、当前真实、瞬时 future video 和 H |
| vision_causal | 更早 video；不读历史 action | Q+PCH 的 video | 否 | 读取全部精确 video/action、当前真实、瞬时 future video 和 H |
| action_aggregator | 只做本 observation 的同-frame video 编码 | 不读取更早精确历史；future video 仍读取当前真实 observation | 否 | 读取全部精确 video/action、当前真实、瞬时 future video 和 H |

在三种模式下，ActionDiT 都读取 H。只有 interleaved 的 VideoDiT 读取 H。这是当前实验定义，不是从 H 中剥离纯视觉子状态。

PCH packed prefill 本身从不读取 H；H 只进入当前真实 video、future video、video supervision 和 action 的独立 memory-reader 分支。

## 首帧记忆 F / H0

当 first_frame_memory=true 时，block 0 的 raw ReplayBlock 被保存为 episode_anchor，不保存为一个额外 action memory。

推理端第一次 packed replay 包含 block 0 时，代码从其 causal video KV 中截取一个 episode_anchor_segment 并永久保存。这个 segment 是逐层视觉 KV；其生成时的视觉 token 已经通过当时的语言和初始 proprio context 条件化，但 segment 本身不含 action KV。

在 block 0 仍位于 Q 或 PCH 时：

- 固定 F 不作为额外前缀读取，避免同一 observation 被模型读两次；
- block 0 仍作为普通精确历史参与每次 packed PCH 重建；
- 已缓存的 episode_anchor_segment 不会被覆盖。

当 block 0 离开 Q+PCH，后续 packed rebuild 不再为 F 创建 query，也不重新生成 F KV；已缓存的逐层 KV 被直接拼成 fixed prefix。mask 规则为：

~~~text
interleaved / vision_causal video queries：可读 F
action_aggregator video queries：不可读 F
所有 action queries：可读 F
~~~

因此“F 只编码并保存一次”指固定 episode_anchor_segment 只构造一次；并不表示 block 0 在仍属于精确 PCH 时不会随整个 PCH 一起重建。

训练端没有跨 dataset samples 的持久 KV cache。目标满足 \(m(t)>0\) 时，从 full prefix 找出真实 \(o_0\)，把它作为 packed layout 的 anchor slot 在该 training forward 中编码；它不会同时出现在 exact [m,t) 中。其 causal visibility 与推理端固定前缀一致，但计算缓存策略不同。

## 配置与运行约束

正式配置值是：

~~~yaml
history_training_mode: episode_memory_scan_bptt
history_window_blocks: 8
episode_memory:
  enabled: true
  window_blocks: 8
  chunk_blocks: 4
  num_slots: 32
  state_dim: 1024
  group_dim: 16
  updater_dim: 256
  updater_heads: 8
  reader_rank: 64
  first_frame_memory: true
~~~

构造时强制：

- chunk_blocks < window_blocks；
- state_dim 能被 group_dim 整除；
- updater_dim 能被 updater_heads 整除；
- enabled episode memory 必须搭配 episode_memory_scan_bptt；
- 模型 history_window_blocks 必须等于 episode-memory window_blocks；
- runtime memory 必须使用 packed_replay；
- dataset 必须使用 full_prefix，不能同时设置 dataset history_window_blocks；
- data anchor 开关必须等于 first_frame_memory；
- memory、model 和 checkpoint 的 causal mode 必须一致；
- replan_steps、action_horizon、video-frame contract 必须与训练 checkpoint 一致。

推理时 episode memory 不再受 max_history_blocks 的增长限制。主体 raw replay 只保留 W 个 PCH blocks 加不足 C 个 Q blocks；启用首帧记忆时还固定保留一份 episode_anchor raw block 和一份 episode_anchor_segment。训练数据当前仍受 max_history_blocks=70 限制。

## Checkpoint 与 reset

模型 checkpoint 保存 episode_memory.state_dict、episode_memory_config、causal_mode、history_training_mode 和 temporal/training contract。episode_memory.state_dict 包含 learned empty H、updater 和 reader 参数。

加载 native LeapBot episode-memory checkpoint 时 config 和 module keys/shapes 使用严格校验；旧的 video_reads/action_reads metadata 会被忽略。episode_memory_only checkpoint 被显式允许加载到 video_lora_action_full 第二阶段。

每个在线 episode 的当前 H、Q、PCH、pending transition 和 F 不属于模型 checkpoint。它们存在于调用方持有的 LeapMemoryState 中。memory.reset 会清空 PCH/Q/closed transitions/KV/F、将时钟归零，并把 H 恢复为 initial_episode_state 的 clone。

transition updater 改为 role_embedding 后，没有为更早实验性 episode-memory checkpoint 的旧 modality_embedding/position_embedding 提供迁移；这类 checkpoint 不能 strict load。原始 Fast-WAM release checkpoint 不包含 episode-memory state，可以作为新 memory 训练的初始化权重。

## 当前代码明确存在的边界与差距

### Updater 已经丢失空间布局

虽然输入来自全部 clean pre-DiT token，_pool_tokens 会立刻对空间/action token 维取平均。当前 H writer 没有 VideoDiT 空间交互，也没有保留物体位置、局部变化或 token 间关系。

### Updater 的训练/推理 proprio context 不完全一致

在线 _update_episode_memory 使用每个 replay block 保存的 context；该 context 在推理时包含对应 proprio token。

scan 训练生成 updater clean pre-DiT token 时调用 build_inputs(..., append_proprio=False)，随后对所有历史 observation/action 使用同一份 text base_context，没有把 history_proprio 加入 updater 输入。历史 proprio 仍用于后面的 exact PCH/WAM target branch，但不进入训练时的 H updater。这是当前可验证的 train–inference input mismatch。

### 训练没有跨目标复用 episode scan

一个 sample 对应一个目标时刻和一个完整前缀。同一 episode 的不同目标样本会重复编码共同前缀、重复生成算子和 scan；当前实现只在单个 batch forward 内并行 chunk，没有 episode-level cache 或多目标共享。

### 终止时没有显式 flush

transition 只有下一张真实 observation 到达才闭合。若环境在执行最后动作后直接 reset，没有 terminal observation，最后一条 transition 不会进入 ClosedReplayBlock。不足 C 的 Q 也不会在 reset 前强行写入 H，而是随 episode state 一起清空。在线控制期间历史覆盖完整，但最终 H 不保证覆盖没有后继 observation 的尾部动作。

### F 的训练和推理是语义对齐而非缓存执行对齐

推理跨 decision 复用固定 episode_anchor_segment；训练每个 sample 都从 raw \(o_0\) 重新计算 anchor KV。二者 causal visibility 对齐，但不是相同的缓存执行过程。

### 数值稳定措施有限

当前已有近单位 \(a\)、小 \(b\)、有界 \(k\)、16 维分组和 FP32 scan；没有额外的长 episode 谱约束、矩阵重正交化或周期性 state normalization。训练前缀最长 70 blocks，超出该范围的稳定性尚未由当前训练配置覆盖。

## 代码位置

| 文件 | 职责 |
|---|---|
| src/leapbot_va/episode_memory.py | H 配置、分组仿射算子、两级 scan、transition updater、辅助损失、memory reader |
| src/leapbot_va/memory.py | LeapMemoryState、executed-action commit、transition closure、H/Q/PCH 分区、snapshot/rollback/reset |
| src/leapbot_va/pch.py | 固定 padding PCH layout、三种 causal mask、H0 fixed-prefix mask、下游 KV 选择 |
| src/leapbot_va/models/leapbot.py | 在线 H 更新、PCH rebuild、H0 KV 缓存、VideoDiT/ActionDiT 读路由、checkpoint |
| src/fastwam/models/wan22/mot.py | packed history forward、逐层 episode-memory residual reader 注入 |
| src/leapbot_va/training.py | episode_memory_scan_bptt、H/Q/PCH target 对齐、主/辅助损失 |
| src/leapbot_va/data.py | full-prefix dataset sample、绝对 block/action metadata、anchor raw 输入 |
| src/leapbot_va/runtime.py | Hydra 配置到 EpisodeMemoryConfig 的解析 |
| configs/task/libero_leapbot_episode_memory.yaml | memory-only 第一阶段 |
| configs/task/libero_leapbot_episode_memory_joint.yaml | Video LoRA + ActionDiT + memory 联合阶段 |
| configs/sim_leapbot_libero_episode_memory.yaml | episode-memory rollout 配置 |
| tests/test_episode_memory.py | 算子展开、scan 等价/梯度、transition 组合、分区与 reader gate |
| tests/test_incremental_full_bptt_training.py | scan training、混合 chunk 数、causal routing、checkpoint |
| tests/test_runtime_temporal_positions.py | 在线 commit、PCH rebuild 和 H0 KV 对象复用 |

## 当前测试覆盖

现有测试已经直接验证：

- prediction–correction 与展开后的仿射算子数值等价；
- 非 2 次幂 affine scan 与顺序递归的前向和梯度一致；
- transition 算子按时间组合后与逐条应用一致；
- 修改一条 transition 的 action 不影响其他 transition 的算子；
- padded chunks 被替换为恒等算子且梯度被 mask；
- 训练 scan 与在线 left-fold 使用相同 chunk 算子；
- H/Q/PCH 分区完整、互斥并支持 rollback；
- 只有 committed executed action 进入 replay；
- 三种 causal mode 的 H reader 路由符合当前定义；
- reader 零 gate 恢复无 H 行为；
- mixed chunk-count batch 可以反向传播；
- episode-memory checkpoint round-trip 和 memory-only → joint 加载；
- 推理端固定 H0 KV segment 在 block 0 离开 PCH 后按对象复用并在 reset 时清空。

最近一次 updater/scan-BPTT 定向回归结果为 56 passed。
