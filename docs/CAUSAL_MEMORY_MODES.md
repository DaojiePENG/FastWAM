# LeapBot-VA 三种因果 KV 记忆方案

> 面向 ICLR / RSS 论文撰写的方法设计参考
>
> 实现基线：`leapbot-va` 当前工作树（HEAD `aab59cd`，并包含当前 full-prefix 训练改动；2026-07-30）
>
> 文档性质：实现导出的技术说明，不包含未经实验验证的性能结论或新颖性声明

## 1. 文档目的

LeapBot-VA 在共享的 Wan VideoDiT、ActionDiT、训练目标和在线记忆协议之上，实现了三种跨重规划块（replanning block）的因果信息流：

1. `interleaved`：新观测读取全部历史观测与历史动作；
2. `vision_causal`：新观测只读取历史观测；
3. `action_aggregator`：每个新观测独立编码，跨块聚合完全由 ActionDiT 完成。

三种方案不应表述为三套独立网络。更准确的论文表述是：

> **Three observation-prefill causal policies under a shared action-history aggregator.**

三者仅改变 VideoDiT 编码当前真实观测时允许读取的历史 KV；ActionDiT 预测当前动作时始终读取完整、已提交的真实历史。该控制变量设计使实验能够回答一个明确问题：**跨时间融合应该递归写入视觉表征，还是推迟到动作预测阶段统一完成？**

当前主训练方案已从“随机短历史的 packed full-BPTT”切换为 `incremental_detached_prefix`：对每个训练样本使用该时刻之前的完整 episode 前缀，按在线 rollout 相同的增量顺序重建历史 KV；历史 KV 对当前预测完全可见，但从反向传播图中分离。旧的 `packed_full_bptt` 路径仍保留，用作短历史和跨块全梯度消融，不再代表主训练配方。

## 2. 问题设定与符号

### 2.1 在线重规划块

将一个 episode 划分为重规划块 $b=0,1,\ldots$。在第 $b$ 个块中：

- $o_b$：重规划时刻获得的真实视觉观测；
- $p_b$：当前 proprioceptive state；
- $g$：整个 episode 内保持不变的语言指令；
- $\hat{a}_b^{1:P}$：模型预测的长度为 $P$ 的动作序列；
- $a_b^{1:r_b}$：实际发送给控制器并由环境执行的动作，其中 $1\le r_b\le R\le P$；
- $d$：启用的 Transformer 深度，即 early-exit depth；
- $B_{\max}$：一个 episode 允许保留的最大历史块数。

当前默认实验配置使用 $P=32$、$R=10$、$d=30$，在线容量为 $B_{\max}=70$。这些值是配置而非方法本身的固定假设。

### 2.2 分层 KV segment

真实观测 $o_b$ 经 VAE 和 VideoDiT prefill 后形成视频 segment：

$$
\mathcal{V}_b = \left\{\left(K_{b}^{v,\ell}, U_{b}^{v,\ell}\right)\right\}_{\ell=0}^{d-1},
$$

其中 $K$ 表示 key，$U$ 表示 value。使用 $U$ 而不是 $V$ 表示 value，以避免和视频 segment $\mathcal{V}$ 混淆。

实际执行动作 $a_b^{1:r_b}$ 被重新映射到模型归一化空间，并以零扩散时刻编码为动作 segment：

$$
\mathcal{A}_b = \left\{\left(K_{b}^{a,\ell}, U_{b}^{a,\ell}\right)\right\}_{\ell=0}^{d-1}.
$$

第 $b$ 次观测到达前，持久化内存按时间顺序排列为

$$
\mathcal{M}_b =
[\mathcal{V}_0,\mathcal{A}_0,\mathcal{V}_1,\mathcal{A}_1,\ldots,
\mathcal{V}_{b-1},\mathcal{A}_{b-1}].
$$

该顺序由 `LeapMemoryState.segments` 显式维护。实现不会把预测但未执行的动作写入 $\mathcal{M}_b$。

## 3. 共享在线记忆协议

### 3.1 两阶段事务

每个重规划块遵循严格的 observation-action commit 状态机：

1. **Observation prefill**：编码并提交当前真实观测 $\mathcal{V}_b$；
2. **Action inference**：基于 $\mathcal{M}_b\oplus\mathcal{V}_b$ 生成临时动作序列 $\hat{a}_b^{1:P}$；
3. **Environment execution**：执行经过环境后处理的前 $r_b$ 个命令；
4. **Action commit**：只将实际执行的 $a_b^{1:r_b}$ 编码并提交为 $\mathcal{A}_b$；
5. 进入下一次真实观测。

在动作提交完成前，状态机拒绝接收下一观测。这一约束避免了观测、动作和绝对位置之间的错位。

### 3.2 真实数据与临时预测的边界

持久化内存只包含：

- 从环境获得的真实观测；
- 最终发送给环境的动作；
- 各 segment 在前 $d$ 层产生的 KV。

以下信息不会进入持久化内存：

- 当前动作扩散过程中的 noisy action tokens；
- 预测序列中未执行的尾部动作；
- 在线预测的未来视频 latent；
- 视频输出头或 VAE 解码得到的未来画面。

因此，在线 ActionDiT 的历史来自真实闭环轨迹，而不是模型对未来状态的自回灌。

### 3.3 当前状态上下文

每个块的观测和动作还通过 cross-attention 接收语言指令 $g$ 与当前 proprio $p_b$。语言指令在 episode 内必须保持一致，proprio 则随块更新。历史 segment 在生成时已经接收其对应块的 proprio；当前 ActionDiT 另外接收当前块的 proprio。

## 4. 增量 prefill 的逐层语义

理解三种方案的关键是区分“本层写入缓存的 KV”和“本层 attention 之后传给下一层的 hidden state”。对当前 segment 的第 $\ell$ 层，模型先计算

$$
(Q_b^\ell,K_b^\ell,U_b^\ell)
=\operatorname{Proj}^\ell(X_b^\ell).
$$

给定该模式选择出的历史 $\mathcal{H}_b$，当前 query 执行

$$
Y_b^\ell = \operatorname{Attn}\!\left(
Q_b^\ell,
[K^\ell(\mathcal{H}_b);K_b^\ell],
[U^\ell(\mathcal{H}_b);U_b^\ell]
\right),
$$

随后产生下一层输入

$$
X_b^{\ell+1}
=\operatorname{BlockPost}^\ell(X_b^\ell,Y_b^\ell,g,p_b).
$$

持久化缓存只保存当前 segment 的 $(K_b^\ell,U_b^\ell)$，而不重复保存拼接后的历史。

这一执行顺序带来两个容易误解的性质：

1. 第 0 层的当前 KV 由当前输入直接投影得到，在三种模式下都尚未吸收历史；
2. 如果当前 query 在第 $\ell$ 层读取了历史，那么历史影响会进入 $X_b^{\ell+1}$，并从下一层开始反映到当前 segment 的 KV 中。

因此，“`interleaved` 的视频 KV 包含历史”是逐层成立的：较深层 KV 包含此前 attention 的历史影响，但第 0 层 KV 不包含。`action_aggregator` 在所有视频层都不提供历史前缀；其第 0 层 KV 来自当前观测与位置编码，较深层 KV 可以进一步吸收当前 prompt/proprio，但不会吸收跨块历史。

## 5. 三种 observation-prefill 策略

### 5.1 统一定义

编码当前观测 $o_b$ 时，VideoDiT 的历史选择函数为

$$
\mathcal{H}_b^v =
\begin{cases}
\mathcal{M}_b,
& \texttt{interleaved},\\
\operatorname{VideoOnly}(\mathcal{M}_b),
& \texttt{vision\_causal},\\
\varnothing,
& \texttt{action\_aggregator}.
\end{cases}
$$

三种模式均允许当前观测内部的 tokens 双向注意。差异只存在于跨块历史的可见性。

### 5.2 `interleaved`

`interleaved` 允许新观测读取所有历史视频和历史动作：

$$
\mathcal{V}_b = f_v(o_b,g,p_b,\operatorname{KV}(\mathcal{M}_b)).
$$

信息流可写为

$$
[\mathcal{V}_0,\mathcal{A}_0]
\rightarrow \mathcal{V}_1,
\qquad
[\mathcal{V}_0,\mathcal{A}_0,\mathcal{V}_1,\mathcal{A}_1]
\rightarrow \mathcal{V}_2.
$$

每个后续视频 segment 都可能成为此前视觉与动作历史的递归摘要。该方案把状态更新部分前移到 VideoDiT：ActionDiT 读取的不只是独立观测，还包括已经经过历史条件化的视觉表示。

潜在优势需要通过实验验证：

- 视觉表示能够显式结合“看到了什么”和“执行了什么”；
- 深层视频 KV 可能形成更紧凑的闭环状态表示；
- ActionDiT 可直接利用已经融合的时序上下文。

潜在风险同样需要验证：

- 历史在连续观测之间被反复混合，可能造成表示过度平滑或早期误差传播；
- observation prefill 的注意力长度随完整视频与动作历史增长；
- 当前视觉编码依赖动作历史，降低了视觉表征的独立可分析性。

### 5.3 `vision_causal`

`vision_causal` 允许新观测只读取历史视频：

$$
\mathcal{V}_b = f_v(o_b,g,p_b,
\operatorname{KV}(\mathcal{V}_0,\ldots,\mathcal{V}_{b-1})).
$$

历史动作仍保存在内存中，但不会作为当前视频 query 的 key/value。它们只在 ActionDiT 预测动作时参与聚合。

该方案将视觉状态演化与动作条件解耦：VideoDiT 建模跨观测的视觉变化，ActionDiT 再联合使用视觉历史和真实动作历史。与 `interleaved` 相比，它可以用于检验视频表示是否需要直接吸收动作 token，或者只依靠连续真实观测已经足够。

潜在优势：

- 保留跨时间视觉建模，同时避免动作 KV 直接改变后续视觉表示；
- observation prefill 的历史长度小于 `interleaved`；
- 视频表征的语义更接近纯视觉状态轨迹。

潜在风险：

- 同样的视觉变化可能由不同动作造成，而 VideoDiT 无法直接区分这些动作；
- 动作与状态转移的对齐压力被推迟到 ActionDiT；
- 视频 segment 仍然递归吸收历史视觉，仍可能产生长期表示压缩。

### 5.4 `action_aggregator`

`action_aggregator` 对每个真实观测执行无历史 prefill：

$$
\mathcal{V}_b = f_v(o_b,g,p_b,\operatorname{position}_b).
$$

“无历史”只指 VideoDiT 不读取 episode KV，并不表示整个策略没有历史。每个视频 segment 仍然：

- 使用共享的 Wan VideoDiT 参数；
- 接收当前真实图像、语言指令和当前 proprio；
- 使用当前 block/frame 的绝对位置编码；
- 在当前观测内部执行双向 self-attention。

不同块的 KV 不会相同，因为图像、proprio 和位置均可能不同。共享 Wan 权重包含从训练数据中学到的统计先验，但固定权重不是当前 episode 的动态记忆。当前 episode 的真实历史只有在显式 KV 被送入 attention 时才会被直接读取。

当前观测本身也会间接体现历史动作的物理后果。例如，机械臂执行抓取后，下一帧图像和 proprio 已经反映物体与关节状态的变化。这属于环境状态中的隐式历史，不等价于模型直接读取 $\mathcal{V}_{b-1}$ 或 $\mathcal{A}_{b-1}$。

该模式把所有跨块融合推迟到 ActionDiT。它保留一组相对独立、带时间位置标签的观测表示，要求 ActionDiT 从完整序列中恢复状态变化与动作效果。

潜在优势：

- observation prefill 的 self-attention 长度不随 episode 历史增长；
- 不会在 VideoDiT 中递归改写或压缩历史；
- 每个观测表示更容易单独分析、复用或做 representation probing。

潜在风险：

- ActionDiT 独自承担全部跨时间关系建模；
- 独立视频 KV 缺少预先融合的状态摘要，长上下文中的检索难度可能更高；
- 如果浅层 ActionDiT 表达能力不足，延迟聚合可能损失性能。

### 5.5 关键对照

| 属性 | `interleaved` | `vision_causal` | `action_aggregator` |
|---|---|---|---|
| 新视频读取历史视频 | 是，全部 | 是，全部 | 否 |
| 新视频读取历史动作 | 是，全部 | 否 | 否 |
| 视频 KV 是否递归融合跨块历史 | 视频与动作历史 | 仅视频历史 | 否 |
| ActionDiT 读取历史视频 | 是 | 是 | 是 |
| ActionDiT 读取真实执行动作 | 是 | 是 | 是 |
| ActionDiT 读取当前真实观测 | 是 | 是 | 是 |
| 当前 noisy action block 内部注意 | 双向 | 双向 | 双向 |
| 持久化 segment 类型 | 视频 + 真实动作 | 视频 + 真实动作 | 视频 + 真实动作 |
| 在线未来视频是否进入动作上下文 | 否 | 否 | 否 |

## 6. 共享的 ActionDiT 历史聚合

当前观测 $\mathcal{V}_b$ 提交后，三种模式使用相同的动作历史：

$$
\mathcal{H}_b^a = \mathcal{M}_b\oplus\mathcal{V}_b.
$$

在动作扩散第 $s$ 个去噪步，长度为 $P$ 的 noisy action block 记作 $Z_b^{(s)}$。ActionDiT 每一层将持久化历史 KV 与当前 action KV 拼接：

$$
X_{b,s}^{a,\ell+1}
=\operatorname{ActionBlock}^{\ell}\!\left(
X_{b,s}^{a,\ell},
\operatorname{Attn}\left(
Q_{b,s}^{a,\ell},
[K^\ell(\mathcal{H}_b^a);K_{b,s}^{a,\ell}],
[U^\ell(\mathcal{H}_b^a);U_{b,s}^{a,\ell}]
\right)
\right).
$$

因此，动作预测的完整可见上下文是

$$
[\mathcal{V}_0,\mathcal{A}_0,\ldots,
\mathcal{V}_{b-1},\mathcal{A}_{b-1},
\mathcal{V}_b,Z_b^{(s)}].
$$

当前 action block 内部使用全可见掩码，即 $P$ 个动作 token 在同一个去噪步内双向交互。这是 chunk-level diffusion policy，不是逐动作 token 的自回归生成。

去噪结束后，$\hat{a}_b^{1:P}$ 仍然是临时结果。只有实际执行的 $a_b^{1:r_b}$ 会以 clean action、零扩散时刻重新 prefill，并写入 $\mathcal{A}_b$。这一步使下一个块看到的是闭环控制事实，而不是上一轮完整预测。

## 7. 训练时的因果结构

### 7.1 完整 episode 前缀数据

训练样本只取自真实的 10-step 重规划边界。对当前块 $b$，主训练路径不再随机截取最近 $h$ 个块，而是令

$$
h=b,\qquad 0\le h\le B_{\max},
$$

即使用同一 demonstration 中当前时刻之前的全部真实块。当前配置设 $B_{\max}=70$。有效历史在固定容量张量中左对齐，剩余后缀补零；`history_valid_blocks` 必须形如 $[1,\ldots,1,0,\ldots,0]$，不允许内部空洞。

数据读取将视觉与动作时间表分开：历史图像和 proprio 只在每 10 个控制步的重规划边界稀疏读取，历史动作则逐控制步密集读取。这样能够保留完整的闭环动作轨迹，同时避免解码不会成为 KV 的中间视频帧。当前监督仍沿用 FastWAM 的 33-frame 视频窗口与 32-step 动作 horizon：第一帧是真实当前观测，其余 32 帧只用于视频 flow-matching 监督。

绝对位置与在线路径对齐：历史视频块使用 $0,\ldots,b-1$，第 $i$ 个历史动作块使用 $iR,\ldots,iR+R-1$，当前动作从真实 episode step $bR$ 开始。当前视频的真实首帧锚定在位置 $b$，未来监督帧使用 $b+1,\ldots,b+32$，以保留原生 FastWAM 视频分支的相对 temporal RoPE 几何；这些未来帧的 KV 不会被提交或传给 ActionDiT。

训练历史与在线历史仍有重要分布差异：

- **训练**：历史动作来自数据集中记录的 demonstration，是 teacher-forced clean prefix；
- **在线评估**：历史动作来自当前策略经环境后处理后实际执行的命令，是 on-policy executed prefix。

### 7.2 与 rollout 同构的增量前缀构建

主路径 `incremental_detached_prefix` 对每个样本按时间顺序执行：

1. 在零扩散时刻编码历史真实观测 $o_i$，VideoDiT 按当前因果模式选择其可见前缀，并生成 $\mathcal{V}_i$；
2. 将 demonstration 中随后真实执行的 $R=10$ 个 clean actions 在零扩散时刻编码，ActionDiT 读取截至 $\mathcal{V}_i$ 的全部前缀并生成 $\mathcal{A}_i$；
3. 对 $i=0,\ldots,b-1$ 重复上述过程，得到与在线内存顺序一致的 $[\mathcal{V}_0,\mathcal{A}_0,\ldots,\mathcal{V}_{b-1},\mathcal{A}_{b-1}]$；
4. 最后仅对当前块的视频与动作目标计算监督损失。

历史构建直接复用在线的 layer-wise `prefill_expert_segment` 语义，而不是依靠一个大 packed mask 近似执行顺序。实现会把具有相同 prefix depth 的样本向量化；进入当前块时，再按真实历史长度分组，使同组样本可以堆叠等长 KV。

三种模式在历史构建和当前块上都使用相同的选择规则：

| 当前视频 query | `interleaved` | `vision_causal` | `action_aggregator` |
|---|---|---|---|
| 可见的更早 segment | 全部视频与动作 | 仅视频 | 无 |
| 当前/历史动作 query | 全部已构建视频与动作 | 全部已构建视频与动作 | 全部已构建视频与动作 |

因此，`action_aggregator` 的每个视频块仍是独立编码，但历史动作 segment 和当前 noisy action 都由 ActionDiT 基于完整真实前缀计算。`detached` 不改变这项可见性。

### 7.3 Detached prefix 的准确含义

对训练样本的历史轨迹记为 $\tau_{<b}$。当前实现可写成

$$
\widetilde{\mathcal{M}}_b
=\operatorname{stopgrad}\!\left(
\operatorname{BuildPrefix}(\tau_{<b};\theta)
\right),
$$

其中 `BuildPrefix` 在每个 batch 都用**当前参数** $\theta$ 重新执行，并非读取离线缓存或旧 checkpoint；`stopgrad` 只表示不保留跨历史块的反向传播图。随后当前块损失为

$$
\mathcal{L}_{30}
=\lambda_v\mathcal{L}_{v,30}
+\lambda_a\mathcal{L}_{a,30},
\qquad \lambda_v=\lambda_a=1.
$$

梯度保留在当前视频块、当前 ActionDiT 计算和当前输出头上。ActionDiT 的动作损失还可以通过当前真实首帧的 KV 回传到当前 VideoDiT；它不能穿过 $\widetilde{\mathcal{M}}_b$ 回传到更早块。参数更新后，下一个 batch 会用新参数重新计算其历史 KV。

这一方案避免为完整历史构建过程保留跨块反向传播图，从而控制 activation memory；当前 attention 和 KV 本身仍随历史增长。它是有意的优化近似：模型训练了“如何读取完整历史”和“如何生成当前 segment”，没有对一个样本的整个 episode 做跨块 full BPTT。论文必须把这种 detachment 明确写出，不能笼统声称“对完整 episode 端到端反向传播”。

### 7.4 当前块的无泄漏视频/动作接口

当前视频张量同时含真实首帧和带噪未来监督，因此训练额外施加 segment 内掩码：

- 真实首帧 query 不能读取任何带噪未来帧；
- 未来视频 tokens 在当前视频块内部双向交互，并可按模式读取允许的历史；
- VideoDiT 生成视频损失后，只截取真实首帧对应的逐层 KV 交给 ActionDiT；
- 当前 ActionDiT 读取完整历史、当前真实首帧以及双向的 noisy action chunk，但永远不读取当前未来视频 KV。

由此，训练动作路径与在线路径在“可持久化/可用于动作的信息”上保持一致。未来帧仍为 VideoDiT 提供原始 flow-matching 监督，却不会构成 action leakage。

### 7.5 旧 packed full-BPTT 路径的定位

`packed_full_bptt` 仍保留为显式消融。该路径把历史视频、历史 clean actions、当前视频目标和当前 noisy actions 放入统一序列，通过因果 mask 控制三种模式，并允许梯度跨所有有效历史块传播。当前配置建议将其用于 `full_episode_history=false, max_history_blocks=8` 的短历史对照；实现也会裁掉 batch 内对所有样本都无效的 padding 后缀。

packed 路径还支持联合训练 $\{8,16,24,30\}$ 层出口：

$$
\mathcal{L}
=\mathcal{L}_{30}
+\frac{\mathcal{L}_8+\mathcal{L}_{16}+\mathcal{L}_{24}}{3}.
$$

当前 full-prefix 主训练只允许 `training_exit_depths=[30]`；浅层出口应在确定主因果配方后单独训练和消融。因此，现阶段不能把浅层出口结果与 full-prefix d30 主实验混写为同一个训练设置。

### 7.6 当前正式训练配方

三种因果模式分别训练 checkpoint，并共享以下配方：

| 项目 | 当前设置 |
|---|---|
| 初始化 | FastWAM release checkpoint |
| 历史训练 | `incremental_detached_prefix`，完整 episode prefix，$B_{\max}=70$ |
| 可训练参数 | ActionDiT 全参数、VideoDiT self-attention LoRA、proprio encoder |
| Video LoRA | rank 16，alpha 16，dropout 0，LR multiplier 10 |
| 优化 | AdamW，base LR $2\times10^{-5}$，video-LoRA 初始 LR $2\times10^{-4}$，cosine，5% warmup，BF16 |
| 训练规模 | 8×H800，micro-batch 16/GPU，global batch 128，gradient accumulation 1 |
| 时长 | 28,523 个训练样本，223 optimizer steps/epoch，5 epochs，共 1,115 steps |
| 监督出口 | d30，$P=32$，$R=10$，seed 42 |
| checkpoint | 每个 epoch 保存权重与完整 trainer state；训练内 rollout eval 关闭 |

`video_lora_action_full` 冻结 VideoDiT 的非 LoRA 主干，完整训练 ActionDiT，并由 trainer 额外启用 proprio encoder。LoRA 参数组使用 10 倍学习率且 weight decay 为 0；action/auxiliary 参数组使用 base LR 和任务配置的 weight decay。评估前将 Video LoRA 合并到 VideoDiT 权重。

checkpoint 同时记录 `causal_mode`、`history_training_mode`、`training_strategy`、出口和 LoRA 配置；加载时检查模式不匹配。完整 trainer state 还保存 optimizer、scheduler、随机状态以及 `epoch/batch_in_epoch`，供中断后恢复数据进度。

## 8. 在线算法

```text
Input:
    instruction g
    action horizon P
    maximum executed steps R
    causal mode m
    exit depth d
Initialize:
    M <- empty episode memory

for replanning block b = 0, 1, ...:
    o_b, p_b <- current real observation and proprio

    if m == interleaved:
        H_video <- all segments in M
    else if m == vision_causal:
        H_video <- video segments in M
    else:
        H_video <- empty

    V_b <- VideoDiT.prefill(o_b, g, p_b, H_video, depth=d)
    append V_b to M

    H_action <- all segments in M
    predicted_chunk <- ActionDiT.diffusion(H_action, g, p_b, horizon=P, depth=d)

    executed_env_actions <- execute at most R postprocessed commands
    executed_model_actions <- map the executed commands back to model space

    A_b <- ActionDiT.prefill_clean(
        executed_model_actions, g, p_b, H_action, depth=d
    )
    append A_b to M

    if episode terminates:
        reset and release M
```

## 9. 因果正确性与系统不变量

当前实现通过状态机、掩码和测试共同维护以下性质：

1. **无 future-video action leakage**：当前动作不能读取未来视频监督；
2. **真实观测优先**：每次动作预测前必须先提交当前真实观测；
3. **执行后提交**：动作预测不会自动写入内存，必须显式提交真实执行 slice；
4. **连续动作位置**：动作 RoPE position 随实际执行数量连续增长；
5. **块顺序一致**：内存严格保持 $[\mathcal{V}_0,\mathcal{A}_0,\ldots]$；
6. **episode 隔离**：episode 结束后释放全部 segment；
7. **指令稳定**：同一 episode 内修改 prompt/context 会触发错误；
8. **容量有界**：超过 $B_{\max}$ 后报错，不执行静默淘汰或滑动窗口；
9. **完整训练前缀**：full-prefix 样本包含当前块之前的全部真实块，有效块必须左对齐且无内部空洞；
10. **在线同构历史构建**：训练历史按 observation/action 交替顺序逐块 prefill，并按三种模式选择视频历史；
11. **当前首帧隔离**：当前真实视频 query 不读取未来视频监督，ActionDiT 只接收当前真实首帧的 KV；
12. **Detachment 不删历史**：历史 KV 虽不保留梯度图，但全部真实 prefix 仍供当前 attention 读取；
13. **checkpoint 配方一致**：causal mode、history training mode、LoRA 和训练出口在保存、验证与加载时显式检查。

第 8 点是当前系统限制。论文若报告超过容量的长时任务，需要先定义并实现明确的 eviction/compression 策略。

## 10. 计算与内存分析

设每个真实观测产生 $N_v$ 个视频 token，已完成 $H$ 个历史块，第 $i$ 块提交 $r_i$ 个动作 token，动作预测 horizon 为 $P$。当前动作预测前的持久化 token 数约为

$$
N_{\mathrm{hist}}
=(H+1)N_v+\sum_{i=0}^{H-1}r_i.
$$

忽略 batch 维和实现常数，深度 $d$、KV 宽度 $D_{kv}$、每元素 $s$ bytes 时，缓存大小近似为

$$
\operatorname{Memory}
\approx 2dN_{\mathrm{hist}}D_{kv}s.
$$

系数 2 对应 key 和 value。由于三种模式最终都保存视频与真实动作 segment，在相同轨迹、深度和 tokenization 下，它们的持久化 KV 容量基本相同。模式差异主要体现在 observation prefill 读取多少历史，而不是保存多少历史。

### 10.1 Observation prefill

只考虑 attention 主项：

- `interleaved`：$O\!\left(dN_v(HN_v+\sum r_i+N_v)\right)$；
- `vision_causal`：$O\!\left(dN_v(HN_v+N_v)\right)$；
- `action_aggregator`：$O(dN_v^2)$，不随历史长度增长。

### 10.2 Action denoising

三种模式共享同一动作路径。每个去噪步的 attention 主项约为

$$
O\!\left(dP(N_{\mathrm{hist}}+P)\right).
$$

如果使用 $S$ 个去噪步，总主项再乘以 $S$。因此 `action_aggregator` 可以消除 observation prefill 的历史增长，但不会消除 ActionDiT 对完整历史的读取成本。

### 10.3 Full-prefix 训练成本

`incremental_detached_prefix` 降低的是**反向传播 activation memory**，不是历史计算量或 KV 容量。每个样本仍需用当前权重从块 0 开始重建完整 teacher-forced prefix：

- `interleaved` 的每个历史视频 prefill 读取此前全部视频与动作；
- `vision_causal` 的历史视频 prefill 读取此前视频；
- `action_aggregator` 的历史视频 prefill 长度恒定，但每个历史 clean-action prefill 仍读取此前完整 prefix；
- 三种模式的当前 ActionDiT 都读取完整历史。

因此，三种模式的训练计算都会随 $H$ 增长；`action_aggregator` 只消除了视频 prefill 的跨块项，并没有把整个训练复杂度降为线性。实现通过两种方式控制实际开销：稀疏解码历史边界图像，以及在相同 prefix depth/相同历史长度的样本之间批处理 KV。论文应同时报告按 $H$ 分层的 forward/backward 时间与 peak GPU memory，避免仅用 global batch size 描述训练效率。

论文中的延迟报告应至少分解为：

- observation prefill time；
- action denoising time；
- executed-action commit time；
- end-to-end replanning time。

只报告总延迟会掩盖三种模式真正的计算差异。

## 11. 论文中的研究问题与可检验假设

以下内容是实验设计假设，不是当前文档已经证明的结论。

### 11.1 核心研究问题

**RQ1：** 对长时视觉运动控制，跨块历史应该在 VideoDiT 中递归融合，还是由 ActionDiT 在决策时统一聚合？

**RQ2：** VideoDiT 是否需要直接读取历史动作，还是只建模视觉状态演化即可？

**RQ3：** 三种信息流在成功率、历史长度扩展、延迟和 KV 内存之间形成怎样的权衡？

### 11.2 可检验假设

- **H1，动作条件视觉状态假设**：如果动作对视觉状态转移的解释不可替代，`interleaved` 应优于 `vision_causal`；
- **H2，视觉充分性假设**：如果真实观测序列已充分表示状态演化，`vision_causal` 可接近 `interleaved`，同时降低 observation prefill 成本；
- **H3，延迟聚合假设**：如果 ActionDiT 足以恢复时序关系，`action_aggregator` 可在保持控制性能的同时降低 observation prefill 延迟；
- **H4，深度依赖假设**：`action_aggregator` 对 ActionDiT 深度更敏感，因为全部跨块融合都发生在动作分支。

### 11.3 归因边界

由于三种模式的 ActionDiT 历史选择完全相同，模式间差异应主要归因于视频 segment 的生成方式。论文不应写成“只有 `action_aggregator` 使用完整动作历史”，因为完整动作历史对三种模式都可见。

同样，`action_aggregator` 不应被称为“memory-free”或“history-free”。准确说法是：

> **history-independent observation encoding with history-conditioned action generation**。

## 12. 推荐实验与报告规范

### 12.1 受控主消融

三种模式应共享：

- FastWAM 初始化 checkpoint；
- 完整 episode-prefix 数据与 action normalization；
- 优化器、学习率、batch size、训练步数与随机种子；
- 动作 horizon、replan steps 和 diffusion steps；
- d30 出口、Video LoRA 配置与可训练参数集合；
- 在线评估任务、初始状态与 trial 数。

`action_aggregator` 先训练并经过 epoch-2 gate，只是当前流水线的资源调度顺序，不构成它优于另两种模式的证据。任何模式选择必须等三个独立 checkpoint 在相同协议下完成后再进行。

正式受控比较使用 LIBERO-10 的 10 个任务、每任务 50 trials。三个 LeapBot-VA 模式均使用 step 1,115、d30、$P=32$、$R=10$、10 个动作扩散步、$B_{\max}=70$ 和合并后的 Video LoRA；FastWAM release 作为额外的无 LeapBot memory 外部基线。因果模式之间是直接控制变量比较，FastWAM baseline 与它们的训练和记忆路径不同，应单独标注。

主表和附录至少报告：

- task success rate 与逐任务结果；
- 成功次数、总 episode 数和 Wilson 置信区间；
- observation prefill、action denoising、action commit 和端到端 replanning 的 p50/p95 延迟；
- peak GPU memory、persistent KV bytes 和 completion steps；
- 实际执行 control steps 和 replanning 次数。

正式脚本要求每个结果包含 profiling 信息，并在最终聚合前校验任务数和每任务 trial 数。最终 4-config 比较共有 $4\times10\times50=2000$ 个 episodes；开发阶段可先使用每任务 10 trials，但不能把该小样本结果当作最终统计。

### 12.2 训练 gate 与诊断

当前训练流水线在扩大计算预算前执行以下检查：

1. 在真实 6B 模型上运行指定历史长度的 forward/backward smoke，要求 loss 与梯度有限，并记录时间、显存和有梯度参数量；
2. `action_aggregator` 在 step 223 与 step 446 保存 checkpoint，epoch 2 后暂停训练并验证权重、LoRA、模式元数据及完整 trainer state；
3. 用相同 dataset samples、flow-matching noise 和 timestep，对 release、epoch-1、epoch-2 checkpoint 做 fixed-noise history-stratified loss audit；
4. audit 使用 $H\in\{0,1,4,8,12,16,24,32,40,50\}$，分别记录 action/video loss，并分解 full history、absolute-position-only 和 native FastWAM 的差异；
5. packed full-BPTT 在 $H\in\{0,8,16,32,50\}$ 上仅作为可行性/显存诊断；通过 gate 后再从完整 trainer state 恢复至 5 epochs；
6. `interleaved` 与 `vision_causal` 启动正式训练前分别执行真实 H=50 forward/backward smoke。

在 $H=0$ 时，`full_prefix_smoke.py` 对 incremental 路径与原生 FastWAM loss 使用 $5\times10^{-3}$ 的 BF16 绝对差阈值。对 `action_aggregator`，视频路径不消费历史 KV，因此启用 native comparison 时还检查其 video loss 一致性。上述检查是实现/训练正确性证据，不是 task-success 结果。

### 12.3 历史长度

当前训练和在线容量都设为 70 块，因此旧版“训练 0–8、在线最多 70”的描述已经失效。不过，容量匹配不等于历史分布均匀：训练样本的 $H$ 由 demonstration 中的实际 block index 决定，较长前缀可能明显少于短前缀；在线 episode 也未必到达容量上限。

因此应至少区分并报告：

- $H=0$ 的 FastWAM-compatible 路径；
- 按精确 $H$ 分层的 offline action/video loss；
- 按 `completed_blocks` 分层的在线 cache、observation、denoise、commit 和总延迟；
- 训练集中各 $H$ 的样本频数与在线 rollout 的 $H$ 分布；
- 若未来评估超过 70 块或超出训练 demonstration 的实际范围，单独标为 extrapolation。

聚合器当前生成 `history_profile.csv`，记录每种配置在各历史长度下的 cache 与延迟 p50/p95，可直接用于历史扩展曲线。成功率仍应以 episode/task 为统计单位，不能把多个 replan 当作独立成功试验。

### 12.4 机制诊断实验

为解释三种方案的机制差异，可增加：

- 按历史距离统计 action attention mass；
- 对历史动作 KV 或历史视频 KV 做 modality dropout；
- 比较浅层与深层出口的模式排序；
- 测量 observation prefill 随历史块数的增长曲线；
- 对独立视频 KV 和递归视频 KV 做 state/progress probing；
- 检查失败轨迹中模型是否过度关注早期历史。

这些实验属于推荐项，当前实现和已有训练日志不能替代正式结果。

本节描述的是当前已编码的实验协议。只有在 checkpoint validation、完整 trial 校验和 Pareto 聚合全部通过后，相应数字才可写入论文；正在运行或仅完成 smoke/gate 的状态不能表述为最终实验结果。

## 13. 论文表述建议

### 13.1 方法章节的组织

建议将方法部分组织为：

1. **Real-Observation and Executed-Action Memory**；
2. **Incremental Layer-wise KV Prefill**；
3. **Observation Conditioning Policies**；
4. **History-Conditioned Action Diffusion**；
5. **Inference-Aligned Full-Prefix Training**；
6. **Online Commit Protocol and Complexity**。

不要把三种模式分散成三套完整算法。先写共享记忆和动作路径，再用一个公式定义 $\mathcal{H}_b^v$ 的三种选择，可以减少重复并突出受控消融。

### 13.2 可直接改写的英文方法段落

> We maintain an explicit episode-level KV memory consisting only of real observations and action commands actually executed by the controller. At replanning block $b$, the memory contains an ordered sequence of video and executed-action segments, $[\mathcal{V}_0,\mathcal{A}_0,\ldots,\mathcal{V}_{b-1},\mathcal{A}_{b-1}]$. The current observation is first encoded and committed as $\mathcal{V}_b$. ActionDiT then denoises a transient action chunk against the complete persistent prefix and the current observation. The predicted chunk is not cached. After environment execution, only the executed action slice is re-encoded at zero diffusion time and committed as $\mathcal{A}_b$.

> We compare three observation-prefill policies while keeping the action-conditioning path fixed. Interleaved prefill exposes each new video query to all earlier video and executed-action KV segments. Vision-causal prefill exposes it only to earlier video segments. Action-aggregator prefill encodes every observation independently and defers all cross-block integration to ActionDiT. In all variants, current action queries attend to every committed video and action segment, the current real observation, and the bidirectional transient action block. The training mask excludes future-video supervision from the action context.

> During training, each replanning sample is paired with its complete demonstration prefix rather than a randomly truncated short history. We reconstruct the prefix sequentially using the same observation/action prefill order and mode-specific video-history selection as online rollout. Historical KV segments are recomputed with the current parameters for every batch but detached from the backward graph to bound activation memory. Gradients are retained for the current video/action block. Although the current video target contains one real frame and noisy future-frame supervision, the real-frame query is masked from future tokens, and only its KV is exposed to ActionDiT.

上述段落是实现摘要，不包含与 prior work 的对比。正式投稿前仍需结合经核验的相关工作添加引用，并避免在没有系统文献检索的情况下使用 “first” 或 “only” 等绝对新颖性表述。

### 13.3 ICLR 与 RSS 的不同强调方向

如果面向 ICLR，可优先强调：

- 跨模态、跨时间信息在视觉专家与动作专家之间的因果分工；
- 递归表示融合与延迟决策聚合的可控比较；
- history length、exit depth 与 representation quality 的交互。

如果面向 RSS，可优先强调：

- 只提交真实执行动作的闭环一致性；
- 不生成未来视频的在线控制路径；
- 成功率、控制频率、延迟、GPU/KV 内存和长 episode 稳定性。

具体篇幅、格式和匿名要求应以投稿当年的官方 call for papers 为准。

## 14. 局限性与可能的审稿问题

1. **Teacher forcing gap**：训练历史是数据集行为轨迹，在线历史是当前策略实际执行轨迹；
2. **Detached-prefix approximation**：历史 KV 使用当前权重重算，但当前损失不会跨历史块反向传播；这不等价于 episode-level full BPTT；
3. **Long-prefix training cost**：detachment 降低 activation memory，却没有消除从块 0 重建前缀的计算和 KV 成本；
4. **Non-uniform history coverage**：训练容量与在线容量同为 70，但 demonstration 中长 $H$ 样本可能稀少，容量匹配不代表分布匹配；
5. **No eviction**：达到容量时直接失败，尚无压缩或淘汰机制；
6. **Implicit history in observations**：即使独立编码，当前图像和 proprio 也包含环境动力学造成的历史结果；
7. **Shared action bottleneck**：三种模式都依赖同一 ActionDiT 聚合完整历史，弱 ActionDiT 可能限制所有模式；
8. **Recursive representation confound**：`interleaved` 和 `vision_causal` 的深层视频 KV 已经压缩历史，不能简单解释为原始历史 token；
9. **Action postprocessing fidelity**：必须保证提交动作与环境实际执行命令严格一致；
10. **Single final depth**：当前 full-prefix 主训练仅监督 d30，尚不能支持关于 early exit 的完整结论；
11. **Task coverage**：单一 benchmark 或单一机器人形态的结果不足以支持普遍性结论。

这些问题应在实验设计或 Limitations 中显式处理，而不是通过强结论绕过。

## 15. 实现映射

| 技术组件 | 主要实现 |
|---|---|
| episode memory、segment、状态机 | [`src/leapbot_va/memory.py`](../src/leapbot_va/memory.py) |
| 三种视频历史选择 | `LeapMemoryState.selected_segments_for_video` |
| 统一动作历史选择 | `LeapMemoryState.selected_segments_for_action` |
| 在线 observation prefill 与动作去噪 | [`src/leapbot_va/models/leapbot.py`](../src/leapbot_va/models/leapbot.py) 中的 `infer_action` |
| 真实执行动作提交 | `LeapBotVA.commit_executed_actions` |
| 增量 segment prefill 与当前视频 segment mask | [`src/fastwam/models/wan22/mot.py`](../src/fastwam/models/wan22/mot.py) 中的 `prefill_expert_segment` |
| ActionDiT 完整历史聚合 | `MoT.forward_action_with_history` |
| full-prefix 增量损失、detachment 与泄漏隔离 | [`src/leapbot_va/training.py`](../src/leapbot_va/training.py) 中的 `incremental_detached_prefix_training_loss` |
| packed full-BPTT 消融 | 同文件中的 `causal_history_training_loss` 与 `build_packed_history_attention_mask` |
| 稀疏历史观测、密集动作与完整前缀数据 | [`src/leapbot_va/data.py`](../src/leapbot_va/data.py)、[`src/fastwam/datasets/lerobot/base_lerobot_dataset.py`](../src/fastwam/datasets/lerobot/base_lerobot_dataset.py) |
| 当前主训练配置 | [`configs/task/libero_leapbot_2cam224.yaml`](../configs/task/libero_leapbot_2cam224.yaml)、[`configs/model/leapbot.yaml`](../configs/model/leapbot.yaml) |
| 三模式 5-epoch 启动与恢复 | [`scripts/run_full_prefix_mode_e5.sh`](../scripts/run_full_prefix_mode_e5.sh) |
| action-aggregator epoch-2 gate | [`scripts/gate_action_aggregator_epoch2.sh`](../scripts/gate_action_aggregator_epoch2.sh) |
| 真实 6B full-prefix smoke | [`scripts/full_prefix_smoke.py`](../scripts/full_prefix_smoke.py) |
| fixed-noise 历史分层 loss audit | [`scripts/history_stratified_loss.py`](../scripts/history_stratified_loss.py) |
| checkpoint/trainer-state 验证 | [`scripts/validate_leapbot_checkpoint.py`](../scripts/validate_leapbot_checkpoint.py) |
| 最终 4-config × 10-task × 50-trial 比较 | [`scripts/run_final_50_trial_comparison.sh`](../scripts/run_final_50_trial_comparison.sh) |
| success、Pareto 与 history profile 聚合 | [`experiments/leapbot/pareto.py`](../experiments/leapbot/pareto.py)、[`experiments/leapbot/plot_pareto.py`](../experiments/leapbot/plot_pareto.py) |
| 默认 LIBERO 在线内存配置 | [`configs/sim_leapbot_libero.yaml`](../configs/sim_leapbot_libero.yaml) |
| 因果掩码与 full-prefix 训练测试 | [`tests/test_causal_masks.py`](../tests/test_causal_masks.py)、[`tests/test_packed_training_masks.py`](../tests/test_packed_training_masks.py) |
| 增量/一次性 attention 等价测试 | [`tests/test_incremental_mot.py`](../tests/test_incremental_mot.py) |
| 真实动作提交测试 | [`tests/test_inference_contract.py`](../tests/test_inference_contract.py)、[`tests/test_libero_action_commit.py`](../tests/test_libero_action_commit.py) |
| 可恢复 sampler 与统计聚合测试 | [`tests/test_resumable_sampler.py`](../tests/test_resumable_sampler.py)、[`tests/test_history_stratified_loss.py`](../tests/test_history_stratified_loss.py)、[`tests/test_pareto.py`](../tests/test_pareto.py) |

## 16. 投稿前技术核对清单

- [ ] 三种模式只改变 VideoDiT 的历史选择，其他训练条件保持一致；
- [ ] 论文明确区分 teacher-forced training history 与 on-policy executed history；
- [ ] 主实验明确写为 complete episode prefix，而不是旧的随机 0–8 history；
- [ ] 明确说明历史 KV 每 batch 用当前权重重算，但不参与跨块反向传播；
- [ ] 不把 detached prefix 描述成 history truncation，也不把它描述成 episode-level full BPTT；
- [ ] 不把 `action_aggregator` 描述为无历史模型；
- [ ] 不声称只有 `action_aggregator` 的 ActionDiT 使用完整历史；
- [ ] 解释第 0 层 KV 与较深层 KV 的历史依赖差异；
- [ ] 说明当前 action chunk 内部是双向注意；
- [ ] 说明真实首帧 query 不读取未来视频，且 ActionDiT 只接收首帧 KV；
- [ ] 说明未来视频监督使用连续 temporal RoPE，但不会进入持久化 memory；
- [ ] 主实验标明只训练 d30，浅层出口属于后续独立消融；
- [ ] 报告训练/测试的精确历史长度分布，而不只报告容量 70；
- [ ] 延迟按 observation、action、commit 分解，并按 history length 统计；
- [ ] FastWAM release 外部基线与三种因果模式的受控比较分开解释；
- [ ] 只有 checkpoint、trial 完整性和 profiling 校验全部通过的结果进入论文；
- [ ] 所有实验结论均来自可复现日志与统计分析；
- [ ] 新颖性与相关工作声明经过独立文献检索和引用核验。
