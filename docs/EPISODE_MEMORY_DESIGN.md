# LeapBot-VA 固定容量 Episode Memory 设计

> 本文是下一阶段实现规格，不描述当前主干已经具备的功能。现有 PCH、三种 causal mode、真实动作提交和多出口训练保持为基础；本文定义在其上增加的完整 episode memory。

## 目标与核心结构

新记忆系统同时保留局部精度和完整 episode 覆盖：不可变的首帧记忆 $F$ 保存真实初始观测，最近的真实闭环轨迹由 PCH 精确保存，更早的轨迹被持续吸收到固定容量的隐式世界状态 $H$。它不保存无限增长的 KV，不使用 VLM、语言摘要、任务阶段、技能边界、关键帧或进度标签，也不判断某段历史“值不值得记”。所有实际发生的闭环交互都会进入记忆，只是以不同形态存在。

第一版固定：

- PCH 窗口 $W=8$ 个真实 block；
- $H$ 每次吸收 $C=4$ 个 transition，且 $C<W$；
- $H$ 为 32 个、每个 1024 维的世界状态 token；
- $H$ 的更新以 16 维通道组为基本线性变换单元；
- 在线总容量为一个固定首帧 $F$、固定的 $H$、最多 $C-1$ 个待交接 block，以及 $W$ 个 PCH block，与 episode 长度无关。

这里一个 block 对应一个已经执行的控制闭环：

\[
b_i=(o_i,a_i^{\mathrm{exec}},o_{i+1}^{\mathrm{real}}).
\]

动作提交时只能确定 $o_i,a_i^{\mathrm{exec}}$；直到下一次真实观测 $o_{i+1}$ 到达，$b_i$ 才成为可写入 episode state 的闭合 transition。预测动作、未执行动作和预测视频永远不能进入持久记忆。episode 终止时若没有后继真实观测，最后一个未闭合 transition 不写入 $H$。

## 无重叠的历史划分

令 $t$ 表示当前真实观测 $o_t$ 的 block index，此时已经闭合的 transitions 是 $[0,t)$。定义

\[
e=\max(0,t-W),\qquad q=C\left\lfloor\frac{e}{C}\right\rfloor.
\]

在当前决策发生前，完整历史严格划分为：

\[
H_t=[0,q),\qquad Q_t=[q,e),\qquad P_t=[e,t).
\]

- $H_t$ 是已经按完整 chunk 交接的远期 episode state；
- $Q_t$ 是固定交接缓冲区，保存已经退出最近 $W$ 窗口、但尚未凑满 $C$ 个的真实 block，长度始终小于 $C$；
- $P_t$ 是 PCH，始终精确保存最近至多 $W$ 个真实 observation/action block。

三段连续、互斥且覆盖全部已闭合历史。模型决策时读取 $H_t+Q_t+P_t$，因此 chunk 更新之间也不会丢失已经退出 PCH 的交互。$Q$ 不是第三种长期记忆，只是有界、精确、不可学习的交接暂存区；实现上可以把 $Q+P$ 打包成一个最多 $W+C-1=11$ 个 slot 的 exact-history forward，但必须保留二者的角色和边界，不能让 Q 中的 block 再出现在 PCH slot 中。

当 $Q$ 临时收齐 $C$ 个 block 时，这 $C$ 个 transition 原子地生成一次 $H$ 更新，更新成功后从 $Q$ 删除。以 $C=4$ 为例，相邻两个更新 chunk 是：

```text
o0, a0_exec, o1, a1_exec, o2, a2_exec, o3, a3_exec, o4
o4, a4_exec, o5, a5_exec, o6, a6_exec, o7, a7_exec, o8
```

两段只共享边界观测 $o_4$，没有重复 transition。共享边界是闭环定义所必需的：它既是前一 transition 的真实结果，也是后一 transition 的起点，不被计作重复 block。

例如 $W=8,C=4,t=13$ 时，$e=5,q=4$：$H$ 覆盖 blocks $[0,4)$，$Q$ 保存 block 4，PCH 保存 $[5,13)$。在 $t=16$ 时，blocks $[4,8)$ 完成交接，$H$ 覆盖 $[0,8)$，$Q$ 清空，PCH 保存 $[8,16)$。

首个真实观测额外建立一份不可变首帧记忆 $F$：在 block 0 尚位于精确历史时，从其一次无历史、无 $H$ 的 causal VideoDiT prefill 中截取并永久保存逐层视觉 KV，不保存 $a_0$。block 0 仍在 Q 或 PCH 中时，模型只读取精确历史，不额外读取 $F$；第一个完整 chunk 已交接、block 0 离开 Q+PCH 后，固定的 $F$ KV 作为只读前缀参与后续推理，不再重新编码。此后 $F$ 提供不变的初始场景参照，$H$ 表示交互持续修正后的远期世界状态。$F$ 是 episode 初始条件，不属于 transition 历史划分，因此 H/Q/PCH 对闭环 blocks 的互斥覆盖保持不变。

## PCH 与交接缓冲区

PCH 沿用当前 LeapBot-VA 的真实历史语义：保存 replay 所需的真实 observation latent、对应 context/proprio 和实际执行动作，按现有三种 causal mode 重建逐层 KV。PCH 和 $Q$ 都是精确轨迹，不保存预测 future-video KV，也不保存未执行的 ActionDiT 输出。

当前实现的 `ReplayBlock` 在 action commit 后记录起点 observation 和 executed action。新协议需要额外维护 transition 的闭合状态：新观测到达时，用它关闭上一 block；窗口滚动时，最老的闭合 block 从 PCH 移入 $Q$，而不是直接删除。$Q$ 中的 block 继续通过 exact-history 路径参与注意力，顺序位于 PCH 之前。

在线事务顺序为：

1. 接收真实 $o_t$，关闭上一 transition；
2. 按上述区间滚动 PCH，将退出 block 移入 $Q$；
3. 若 $Q$ 收齐四个 block，计算 chunk 更新算子并原子更新 $H$，成功后清空这四个 block；
4. 从 $H+Q+PCH$ 构建当前 VideoDiT/ActionDiT 所需记忆，预测动作；
5. 环境实际执行后，通过现有 `commit_executed_actions` 提交真实动作；该动作只形成一个 open block，等待下一个真实 observation 关闭。

$H$、$Q$、PCH、pending observation/action 和 episode clock 必须进入同一个 snapshot/rollback 事务。任一步骤异常都恢复到本轮 observation 到达前的状态，不能只回滚 layer-wise KV。

## H 的形态与读取

$H\in\mathbb{R}^{32\times1024}$ 是一个独立于 VideoDiT 3072 维 hidden space 和 ActionDiT 1024 维 hidden space 的第三状态空间。维度相同不代表复用 ActionDiT 特征；$H$ 有独立参数、归一化、更新器和读出器。episode reset 时，$H$ 回到一组共享的 learned-empty tokens，它们不含任何当前 episode 信息。

持久状态只保存这一份 $H$，不保存它的逐层 KV。每次模型前向时，用共享基础投影和逐层低秩适配器临时产生每层 memory K/V，再通过独立的 memory-attention 分支供 VideoDiT 或 ActionDiT 读取。该分支与语言/proprio cross-attention 分开，并使用零初始化输出门，使加入 memory 的初始行为接近现有 checkpoint。这样计算形式类似逐层读取 context token，但 $H$ 不进入现有静态 `context_payload`，也不与 PCH KV 混成无法区分的持久 cache。

三种 causal mode 继续作为既有实验轴，$H$ 的读取范围只由 causal mode 决定，不再设置额外的普通开关：

- `interleaved`：VideoDiT 和 ActionDiT 都读取 $H$；
- `vision_causal`：只有 ActionDiT 读取 $H$；
- `action_aggregator`：只有 ActionDiT 读取 $H$。

每个实验只维护一个 $H$，不并行维护 $H^V/H^{VA}$。

## Chunk 独立的预测—观测校正

### 输入表示

一个更新 chunk 固定包含四个闭合 transitions。视觉和动作先分别经过现有 VideoDiT、ActionDiT 的 clean `pre_dit` 路径，保留在各自特征空间，再由独立的小型视觉适配器和动作适配器送入 chunk updater。不能使用已经吸收 PCH 的 contextual layer KV 或最终 hidden state，否则会把近期历史再次写入 $H$。当前代码没有永久保存 clean pre-DiT token；实现时应在 replay 构建中复用或按需重算这一级表示，而不是重新运行完整 VideoDiT/ActionDiT。

chunk updater 使用带固定时序位置的交错轨迹：

```text
V(o_s), A(a_s_exec), V(o_s+1), ..., A(a_s+3_exec), V(o_s+4)
```

视觉和动作直到 updater 内部才融合。预测分支对每个 transition 只能读取其起点 observation 和 executed action；真实结果分支读取对应的后继 observation。这样真实结果用于校正，而不能提前泄漏给状态预测。四个 transition 在 updater 内可以用固定长度 causal attention 并行处理，最终共同产生一个 chunk 算子。

### 直观更新

正文只需把更新理解为三步：

\[
\widehat H=\operatorname{Predict}_j(H),
\]

\[
r_j=y_j-\operatorname{ReadObservation}_j(\widehat H),
\]

\[
H'=\widehat H+\operatorname{WriteCorrection}_j(r_j).
\]

`Predict` 根据 chunk 起点和实际执行动作推进旧世界状态；`ReadObservation` 从推进后的状态预测这四次交互应产生的真实观测证据；`WriteCorrection` 只沿本 chunk 可观测的状态方向擦除错误预测并写入真实结果。未被当前视角观测到的状态方向保留，因此 $H$ 不是最新 observation 的压缩副本。

所有 chunk 都执行更新。校正门只表示某个隐状态方向在本次交互中的可观测性和校正强度，不是 chunk admission、关键帧选择或重要性筛选；不允许通过 hard top-k 或整段零门跳过历史。

### 可扫描展开

实现时将 $H$ 向量化为 $h$。chunk updater 独立产生结构化的临时参数，使三步写成：

\[
\widehat h=A_jh+b_j,
\]

\[
\widehat y_j=C_j\widehat h,
\]

\[
h'=\widehat h+C_j^\top K_j(y_j-\widehat y_j).
\]

其中 $y_j$ 是从真实后继 observations 得到的 clean 视觉证据；$C_j$ 把世界状态投影到本次观测证据空间；$C_j^\top$ 把残差写回对应状态方向；$K_j$ 是有界的校正强度。真实视觉证据不需要与 $H$ 处在同一坐标系，二者由 $C_j/C_j^\top$ 显式连接。

展开后：

\[
h'=G_jh+B_j,
\]

\[
G_j=(I-C_j^\top K_jC_j)A_j,
\qquad
B_j=(I-C_j^\top K_jC_j)b_j+C_j^\top K_jy_j.
\]

因此每个 chunk 可以在不知道输入 $H$ 的情况下，仅根据本段真实闭环交互生成一个更新算子 $(G_j,B_j)$；把它应用到任意旧状态时，仍严格等价于一次依赖旧状态预测值的 residual correction。

为使算子组合可承受，$A_j$ 和 $I-C_j^\top K_jC_j$ 在每个状态 token 内按 16 维分组构成块对角矩阵；$C_j$ 在相同分组内归一化，$K_j$ 通过有界门产生。块对角结构在任意次数组合后保持不变，避免完整稠密矩阵 scan。$A_j$ 以单位变换初始化，校正以小幅度初始化，以防训练初期快速破坏长期状态。

这一路径借用了 delta-rule 的代数思想，但 $H$ 仍是供 WAM 使用的固定世界状态 token，而不是普通 token-level Mamba/RNN hidden state；更新单位是完整闭环 chunk，不是语言 token，也不是无限视觉历史的重新汇总。

## Associative scan 与完整 episode 训练

定义算子 $(G,B)$ 对状态的作用为 $F(h)=Gh+B$。按时间先应用算子 1、再应用算子 2 时：

\[
(G_2,B_2)\circ(G_1,B_1)
=
(G_2G_1,\;G_2B_1+B_2).
\]

该组合满足结合律但不满足交换律，因此 associative/prefix scan 可以并行改变计算括号，却不会改变 chunk 的因果顺序。所有 chunk 的算子可以并行生成，再用树形 scan 在 $O(\log J)$ 的依赖深度内得到 episode 各交接边界的 $H$。这与顺序执行完全是同一个状态转移，不是 truncated BPTT 近似。

训练样本以完整 episode 或包含 episode 起点的连续长段组织。对一个 episode：

1. 批量生成所有真实 observations 和 executed actions 的 clean pre-DiT 表征；
2. 从 block 0 开始按不重叠的四-transition chunk 切分，所有完整 chunk 并行生成 $(G_j,B_j)$；
3. 对算子做 exact prefix scan，得到每个完整交接前缀的 $H$；
4. 对任一训练目标 $t$，计算 $e=\max(0,t-W)$、$q=C\lfloor e/C\rfloor$，直接索引 scan 的第 $q/C$ 个前缀状态作为 $H_t$，精确构造 $Q_t=[q,e)$ 和 PCH $P_t=[e,t)$；
5. 用 $H_t+Q_t+P_t$ 运行现有 current-real、future-condition、video-flow target 和 ActionDiT 训练路径。

同一 episode 的 chunk 算子和 scan 结果只计算一次，可以服务多个不同的目标 $t$。为控制 6B WAM 训练成本，不要求每个 block 都运行完整 VideoDiT/ActionDiT loss：memory updater 的轻量预测—校正损失覆盖所有完整 chunk，主 WAM video/action loss在每个 episode 采样若干目标 block；目标采样需覆盖早期、chunk 边界前后和长历史位置。

训练损失由三部分组成：

- 现有多出口 VideoDiT flow loss 和 ActionDiT flow loss，负责让读取到的 $H$ 对世界预测和控制有用；
- 校正前预测损失，使 $C_j(A_jh_j+b_j)$ 预测真实 $y_j$，保证 `Predict` 真正学习动作条件下的状态推进；
- 校正后观测一致性损失，只约束本次可观测方向，不强迫整个 $H_{j+1}$ 等于 $y_j$，避免抹除不可见的远期状态。

训练初期冻结 Fast-WAM 主干，只训练 chunk updater、$H$ 读出适配器、memory-attention 门和相关 loss head。历史 clean pre-DiT 表征与主干断开梯度，避免保存完整 episode 的大模型激活。稳定后再联合训练现有 LoRA、浅层 exit heads 和新 memory 模块；主 WAM 梯度可以穿过所选目标的 $H$ 和 prefix scan 回到所有先前 chunk 算子，但仍不要求对历史 block 展开完整 VideoDiT/ActionDiT。

scan 消除了原先 $H_0\rightarrow H_1\rightarrow\cdots$ 的顺序前向和顺序反向依赖；它不会消除每个 episode 必须产生 $O(J)$ 个 chunk 算子的总工作量。实现应使用 blockwise affine scan，并对 scan tree 做 activation checkpoint。训练可用 FP32 累计算子组合，再将得到的 $H$ 转回模型 dtype，降低长 episode 下门乘积的数值误差。

在线推理不需要启动并行 scan：每收齐一个交接 chunk，直接对当前 $H$ 应用一次相同的 $(G,B)$。训练是并行 prefix scan，推理是在线左折叠，但二者的算子、顺序和结果定义完全一致。

## 与现有 Fast-WAM/LeapBot 流程的结合

- `memory.py` 的 episode state 需要扩展 $H$、闭合 transition、固定 $Q$ 和事务快照；PCH 淘汰由“删除”改为“移入 Q，凑满后原子交接”。
- `leapbot.py` 在当前 observation prefill 前完成 transition closure、Q/H 交接和 memory KV 生成；`commit_executed_actions` 仍是唯一允许真实动作进入历史的入口。
- `mot.py` 保留现有 packed PCH 和 incremental segment 接口，新增独立 episode-memory K/V 与零门控读取分支；不能把 $H$ 塞入语言/proprio context。
- `training.py` 新增 episode chunk operator、prefix scan 和按目标 $t$ 组装 $H/Q/PCH$ 的训练路径；现有 PCH mask、future-video 隔离、多出口损失和 action-through-video 梯度语义保持不变。
- 数据层必须提供从 episode 起点开始的真实 observation 和实际动作序列。随机目标不能只加载最近 $W$ 个 block，否则无法生成目标所需的 scan prefix；VAE latent 和 clean pre-DiT 输入可以缓存，但不能缓存无限逐层 KV。

## 必须保持的约束与验收条件

实现至少需要验证：

- 任意 $t$ 下 $H=[0,q)$、$Q=[q,e)$、PCH=$[e,t)$，无缺口、无重复 block；
- $Q<C$，PCH 长度不超过 W，在线持久容量不随 episode 增长；
- chunk 只包含实际动作和真实后继 observation，相邻 chunk 只共享边界 observation；
- 未提交动作、预测动作和 future-video KV 无法进入 $H/Q/PCH$；
- $Q$ 收齐四个 block 后，更新 H 与删除 Q 是一个可回滚事务；
- 顺序折叠和 associative scan 在 FP32 容差内得到相同的所有 prefix $H$；
- scan 组合保持 16 维块结构，训练图中不存在跨 chunk 的 Python 顺序递归；
- episode-memory 模式只在 block 0 离开 Q+PCH 后读取单份 V0 anchor，且不保存对应动作；
- Q+PCH 的 exact-history mask 与现有三种 causal mode 一致，padding 和无效 slot 不可见；
- H 的 memory-attention 门初始化为零时，加载现有 checkpoint 的输出保持在规定容差内；
- partial terminal action、prompt/context 变化、推理异常和 episode reset 正确清理 H/Q/PCH 与 pending transition；
- 长 episode 的在线显存保持常数级，训练耗时随 chunk 数线性增长但关键依赖深度为对数级。

## 设计结论

完整方案不是“PCH 加递归摘要”。PCH 和 Q 保存有界的精确闭环轨迹，负责局部控制和 chunk 交接；$H$ 保存固定容量的 episode world state。每个四-transition chunk 独立产生一个可组合的“动作推进—观测预测—真实残差校正”算子，训练用 associative prefix scan 并行恢复所有长期状态，推理用同一算子在线更新。复杂推理留在 VideoDiT/ActionDiT 的注意力读取路径，长期写入被刻意约束成稳定、可组合且具有明确世界状态校正含义的状态滤波器。

## 当前实现

实现位于 src/leapbot_va/episode_memory.py，并已接入 LeapMemoryState、LeapBot 推理事务、MoT 逐层 forward、checkpoint 和 causal-history training dispatch。训练配置为：

    python scripts/train_leapbot_local.py --task libero_leapbot_episode_memory
    python scripts/train_leapbot_local.py --task libero_leapbot_episode_memory_joint

第一条只训练 episode memory；第二条联合训练 VideoDiT LoRA、ActionDiT 与 memory。评估入口为：

    CKPT=/path/to/checkpoint.pt LEAPBOT_PYTHON=/home/myuser/miniconda3/envs/leapbot-va/bin/python bash scripts/evaluate_episode_memory_checkpoint.sh

训练样本仍从 episode 起点提供完整真实前缀，但进入主 WAM forward 的精确历史只包含互斥的 $Q+PCH$，固定为最多 11 个 block；更早前缀只通过扫描得到的 32×1024 状态 $H$ 进入模型。历史 clean pre-DiT 特征停止梯度，updater、FP32 scan、learned-empty state 和逐层 reader 保持梯度。
