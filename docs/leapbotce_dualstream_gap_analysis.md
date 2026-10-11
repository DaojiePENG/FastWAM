# LeapBotCE 双流训练效果差距分析

> 2026-10-11 · 响应问题："libero 的双流训练并不成功？参数安排不合适、世界模型跟 VLA 的不同、还是 wam 时延训练方案不合理？CloudEdgeVLA 双流效果很好。"
> 结论先行：**双流（时延）课程本身没有失效，失效的是同步上限。** 我们的延迟鲁棒性（retention 85–97%）已经超过 CloudEdgeVLA（76.5%）；真正的差距是 D=0 绝对成功率（65–73 vs CEVLA 96 / 官方 FastWAM 97+）。主因按证据强度排序：① 云端冻结 + λ_video=0，双流梯度压力无处落地（CEVLA 骨干 LoRA 参训，这是其机制的核心一环）；② 边缘新鲜视觉通道只有 1 个 SigLIP token，带宽比 CEVLA 低一到两个量级；③ 优化配方（h200 大 batch + 全退火 25k 步反超 110k 中途 +8 点）。`loss_fresh ≈ loss_stale` 全程贴合提示**缓存利用率可能很低**——"平坦 retention"有相当一部分可能是平凡鲁棒（靠新鲜 SigLIP token + 当前本体撑着），需要一次缓存消融定性。

---

## 1. 现状数字

### 1.1 训练 run 全景（数据同源：四套件 no_noops，277,713 帧 / 1,712 demos，与官方 ckpt 同 norm stats）

| Run | 步数 | 有效 batch | LR | 退火状态 | 终点 loss_fresh / loss_stale | 备注 |
|---|---:|---:|---:|---|---:|---|
| dtrain20_seed42 (5t78mito) | 50k | 16 | 1e-4 | 完整退火 | 0.0433 / 0.0518 | stale λ 按计划爬到 0.5 |
| dtrain40_seed42 (v5je6j5q) | 50k | 16 | 1e-4 | 完整退火 | 0.0337 / 0.0364 | **训练完全正常**（见 §4.1） |
| 200k 线 @111k (dshch9sp→b8esqywj) | 111k | 16 | 1e-4 | **中途**（cosine 200k） | 0.0198 / 0.0225 | NCCL 死后续训中，ETA ~10-11 中午 |
| h200 线 25k (kemjr8t4) | 25k | 128 | 2.8e-4 | 完整退火 | **0.0083 / 0.0088** | ≈3.2M 样本，与 200k 线同量级 |

### 1.2 评测结果（四套件总体；110k 为 200/200 全齐，h200 约 104/200 部分完成）

| 检查点 | D=0 | D=5 | D=10 | D=20 | D=40 | Retention |
|---|---:|---:|---:|---:|---:|---:|
| 50k d20 | 58.1 | 55.7 | 56.7 | 56.6 | 52.8 | 90.9% |
| 110k d20（论文当前引用） | 65.0 | 62.0 | 63.2 | 59.4 | 55.2 | **85.1%** |
| h200 25k d20 | **73.2** | 74.9 | 73.0 | 74.8 | — | **≈100%（平坦）** |
| dtrain40 50k | ≈5.7 全灭 | — | — | — | — | — |
| **CloudEdgeVLA（对照）** | **96.0** | — | 93.5 | 90.2 | 73.5 avg | **76.5%** |
| FastWAM 官方（Spatial 单套件） | 97.4 | 74.6 | 36.4 | 11.0 | 2.2 | 2.3% |
| OpenWAM dual（Spatial 单套件） | 99.6 | 87.6 | 62.2 | 31.2 | 10.4 | 10.4% |

分套件（110k）：Object 81.8 → 69.2（84.6%）、Goal **46.8** → 38.6（82.5%）、LIBERO-10 74.0 → 62.4、Spatial ≈57.1（由三套件 macro 61.9 反推）。**Goal 套件最弱**——恰是精确定位/放置类任务，与"新鲜视觉带宽不足"的假说一致。

### 1.3 训练 loss 曲线关键信号（wandb 离线扫描）

- **`loss_fresh ≈ loss_stale` 全程几乎重合**：50k 线终点 0.0433 vs 0.0518（stale 仅 +20%）；h200 终点 0.0083 vs 0.0088（+6%）。如果模型真在依赖缓存内容，平均 10.5 步龄的陈旧缓存应当显著更难——两条曲线几乎一样，说明 **action expert 对缓存年龄基本不敏感**。两种解释：(a) 模型只从缓存提取延迟不变的内容（场景布局/任务语义），细节从新鲜 SigLIP token + 本体拿；(b) 缓存内容几乎被忽略，鲁棒性是平凡的。当前证据无法区分，**这是下一步实验的第一优先级**（§6-P0b）。
- **h200 终点 loss 比 50k 线低 5 倍**（0.0083 vs 0.0433）：大 batch + 2.8× LR + 完整退火带来的优化增益直接兑现成 +8 点评测。200k 线 @111k 的 0.0198 已好于 50k 终点，退火完成 后预期进一步下降。

### 1.4 协议契约说明（避免不公平比较）

baseline 表里 FastWAM/StarWAM/OpenWAM 是**单流全观测老化**（图像+本体+全部模态一起延迟），我们是**云端流老化、边缘当前双视角+当前本体始终新鲜**——契约对我们有利。即便如此我们 D=0 仍落后 30 点，进一步支持"瓶颈不在延迟处理，在同步能力"的判断。

---

## 2. 当前实现机制（代码级，`leapbot-ce` 分支）

架构：云端 = 冻结 Wan2.2 video expert（官方 `libero_uncond_2cam224_1e-4` ckpt）对 t−d 帧的云相机拼接图 + 云本体做 VAE encode + 30 层 prefill → MoT per-layer KV cache；边缘 = 1.02B action expert 以 [语言 context + 边缘当前本体 + **1 个 SigLIP token**（双视图 pooled → 可训练 projector）+ 冻结云 KV] 去噪 32 步动作块。

与训练效果直接相关的五个实现事实：

1. **可训练参数只有** action expert、`edge_vision.projector`、proprio encoder（`leapbotce.py` 的 `save_checkpoint` delta 约 2 GB）。云端 Wan 完全冻结。
2. **`loss: {lambda_video: 0.0, lambda_action: 1.0}`**（`configs/model/leapbotce.yaml:66`，`training_loss` 硬编码同值）。官方 FastWAM 配方默认 **`loss_lambda_video=1.0`**（`fastwam.py:39`），即官方 ckpt 是世界模型损失与动作损失**联合**训练出来的；我们的微调把视频损失关掉且不回传。
3. **双路损失**：同一 episode 同一时刻 t，fresh 用当前云帧缓存、stale 用 k 步前缓存（k ~ U{1..min(20, 有效历史)}，`cloudedge_robot_video_dataset.py: sample_stale_index`），共享同一动作目标：`loss = (1−λ)·fresh + λ·stale`，λ 线性 0→0.5，warmup = max_steps/2。**与 CEVLA 完全一致。**
4. **边缘无延迟年龄条件**：`_edge_context` 注入的上下文不包含 `cloud_delay_steps`；expert 无法区分 fresh cache 与 stale cache（评测也不给延迟元数据，与 CEVLA 卖点一致）。
5. **KV adapter 已实现但从未启用**：零初始化共享低秩 K/V 残差 + per-layer scales（`model.cloudedge.kv_adapter`，recipe `libero_spatial_leapbotce_kv`），接口保持 fresh/stale 两路和两段式 API 兼容。LEAPBOTCE.md 把它标为"云表征适配的推荐第一步"。**已核实（2026-10-11）：全部 5 个 run（dtrain20 / dtrain40 / 200k / 200k_resume / h200）落盘 config 均 `kv_adapter.enabled: false`，50k checkpoint 可训练权重仅 ActionDiT + `edge_vision_projector`（4 张量）+ `proprio_encoder`（2 张量），无任何 adapter 键——即云 KV 原样直通 ActionDiT，边缘只能迁就一个不可变换的缓存。** adapter 为零初始化残差，从现有 ckpt 热启动插入等价于原地开始，是验证"缓存可适配性"最便宜的路径。

### 2.1 ActionDiT 三个信息源的维度账本（2026-10-11 核对）

消费机制（`mot.py: forward_action_with_video_cache_tensor`）：30 层每层 self-attn 的 K/V = [云缓存 ⊕ action 自身 K/V] 拼接（全部 24 头×128=3072 宽），另有 cross-attn 到 context 序列（语言+本体+边缘视觉，均投到 4096 → `text_embedding` → 1024 → k/v 线性 → 3072）。

| 信息源 | token 数/层 | 进入 attention 宽度 | 原始信息 | 每层 K/V 量 |
|---|---:|---:|---|---:|
| Wan 云缓存（逐层对齐） | 98（LIBERO）/ 120（RoboTwin） | 3072 直通，无投影 | 448×224（RoboTwin 384×320）→ VAE 16× → patch 2×2 | 98×3072×2 |
| 语言 | ~10–20（cap 128） | 3072（4096→1024→3072） | umT5 嵌入 | ~10×3072×2 |
| 边缘当前本体 | 1 | 3072（同上通道） | 7D/14D → Linear → 4096 | 3072×2 |
| 边缘当前视觉 | 1 | 3072（同上通道） | 2 视图 SigLIP pooled 768×2=1536 → projector → 4096 | 3072×2 |

比例：每层 KV 预算 = 云缓存 98 : action 32 : context ~12（新鲜视觉仅 1）。**云:边缘视觉 ≈ 98:1**（RoboTwin 120:1）。

关键判断：KV adapter（rank 64）是**加性残差 + up 零初始化**（`cache + scale·up(gelu(down(LN(cache))))`），启用即恒等，输出保持全宽 3072，cache 合同不变——**Wan 世界模型信息零压缩**，参数仅 ~0.80M。真正被压缩的是边缘新鲜视觉管道：SigLIP pooled 丢掉全部空间结构后压成 1 个 token，还与语言共用为语言预训的 text_embedding MLP。故 adapter 调"缓存怎么被读"，边缘 token 数决定"现在发生什么"进得来多少——两者正交，adapter 版如果绝对值仍低于 CEVLA，边缘 token 加宽（每视图 1 token）是下一刀。

---

## 3. 与 CloudEdgeVLA 逐项对照

| 维度 | CloudEdgeVLA | LeapBotCE | 影响 |
|---|---|---|---|
| 云端骨干 | OpenVLA-OFT 7B，**LoRA r=32 参训**，双路损失都回传骨干 | Wan2.2 5B，**冻结**，λ_video=0 | **关键差异 #1**。CEVLA 原文机制："creates pressure for the cloud path to retain information shared across the pair"——这个压力必须有梯度落点才存在；我们这里它只落到边缘三个小模块上 |
| 新鲜视觉通道 | 每 action token 位置 concat（LLM hidden states + 双视图 pooled SigLIP 投影）→ MLPResNet 残差融合（`action_heads.py: VisionActionHead`） | **1 个** SigLIP token 进 ActionDiT 的 attention | **关键差异 #2**。CEVLA 的每个动作 token 都拿到一份新鲜的 pooled 双视图特征；我们全部新鲜视觉信息压进 1 个 token。pooled 特征天然丢空间精度 → 与 Goal 套件最弱一致 |
| 延迟元数据 | 推理不用 | 不用 | 平价（卖点一致） |
| 课程 | λ 0→0.5，warmup = max_steps/2 | 同 | 平价 |
| stale 采样 | 配对同动作 chunk 双路 | 同（U{1..min(D,H_t)}，k=0 仅开局回退） | 平价 |
| 骨干预训练 | VLA 骨干见过的分布与微调数据同域 | 官方 ckpt 以 WM 损失+动作联合目标训练，但 edge consumer 从未被适配 | 与 #1 叠加 |
| 云端表征性质 | LLM hidden states：语义级、随时间慢变，陈旧损伤温和 | MoT KV cache：编码精确时空细节（物体/臂位姿），陈旧部分含**错误**细节且冻结无法学会忽略 | 表征差异真实存在，但 LIBERO 场景慢变，实测 retention 反而更高——说明在当前延迟范围内该差异**不是主要矛盾** |
| 结果 | D=0 96.0，D=40 retention 76.5% | D=0 65–73，D=40 retention 85–97% | **差距在锚点，不在斜率** |

一句话：**我们忠实移植了 CEVLA 的时延训练课程，但没有移植让它成立的两个前提——云端可适配、新鲜通道够宽。**

---

## 4. 三轴归因

### 4.1 "参数安排不合适？" —— 次要原因，但是最便宜、最确定的一项

- 200k 线（被引为论文结果的 110k）用 bs16 / lr 1e-4，且评测点在 cosine **中途**；h200 线只差 batch 16→128、lr 1e-4→2.8e-4（√batch 折算）、退火跑完，**同数据同 seed 下 25k 步反超 110k 步 8 个点**（73.2 vs 65.0），且终点 loss 低 5 倍。这不是巧合量级。
- 但即使按 h200 终点算，D=0 仍是 73.2——配方只解释 ~8 点，剩下 ~23 点要靠 §4.2/§4.3。
- dtrain40 50k 的 5.7% 全灭**不是训练问题**：训练曲线全程正常（终 loss 0.0337，与 d20 的 0.0433 同水平甚至更低），基本可断定是评测/权重侧损坏（对应既有"损坏调查"待办），需重新提评测而不是重新训练。

### 4.2 "wam 时延训练方案设计不合理？" —— 课程本身按设计工作，但它设计的前提没被满足

- retention 85–97%（甚至 D=40 > D_train=20 的 h200 线平坦）说明"对陈旧云缓存鲁棒"这个训练目标达成了，超过了 CEVLA 的 76.5%。
- 50/50 双路加权把有效新鲜样本减半——CEVLA 同样如此，非差异项，不是主因。
- 真正的问题：**CEVLA 的双流损失有一条通向骨干的梯度路径**，陈旧帧上的损失会迫使骨干学会"保留跨时间共享的信息、压制时变细节"；我们冻结云端，同样的损失只能推着边缘三个小模块去"适配一个不可改的缓存"。而 expert 又没有年龄条件，唯一可行的解就是学成延迟不变映射——若缓存里时变细节占大头，延迟不变映射的最优解就是**少用缓存**。`loss_fresh≈loss_stale` 正是这个解的特征。
- 推论：继续在冻结云端下调课程（改 λ、改采样分布）大概率无效；要改的是让压力有落点（§5-3/4）或让新鲜通道足以独立支撑（§5-2）。

### 4.3 "世界模型跟 VLA 的不同导致的？" —— 表征差异真实但当前不是主要矛盾；它决定的是天花板

- CEVLA 的云端 latent 是 LLM 语义状态，时间上天然慢变；MoT KV cache 是精确时空编码，陈旧 = 含错误位姿细节。理论上 WAM 的陈旧损伤更重。
- 但实测我们的 retention 高于 CEVLA——LIBERO 场景/物体慢变 + 演示轨迹准静态，缓存 10 步龄的内容大部分仍然正确，损伤没有想象的大。
- 真正被 WAM 特性卡住的是**上限**：官方 ckpt 的缓存是为世界模型损失+动作联合目标适配的，从没为"被一个 1B expert 加 1 个 SigLIP token 消费"适配过；冻结意味着这个适配永远缺位。这正是 KV adapter 存在的理由（LEAPBOTCE.md 原文："recommended first step for cloud representation adaptation"），也是 CEVLA LoRA 的低秩等价物。
- RoboTwin 侧 ours D=0 22.8%（baseline 90+）是另一类问题（三视图/14D 适配与协议），不属于本次 LIBERO 分析范围，但同样提示"边缘适配不充分"。

### 4.4 D=0 差距 ~30 点的分解（当前证据下的最佳估计）

| 来源 | 点数（估计） | 依据 |
|---|---:|---|
| 优化配方（bs/lr/退火） | ~+8 | h200 25k vs 110k 实测 |
| 200k 级训练量 + 完整退火 | ~+2–5 | 50k→110k 的 +7 趋势外推；200k 续训完成后可直接验证 |
| 缓存适配（KV adapter / 云端低秩解冻） | 未知，P0 实验 | LEAPBOTCE.md 预留路径；CEVLA 机制移植 |
| 新鲜通道加宽（1 token → 多 token / 逐 token 融合） | 未知，P1 实验 | Goal 套件 46.8 的空间精度瓶颈 |
| 冻结 5B 云骨干的残余信息瓶颈 | 可能残留 ~5–10 | 若上述全修仍不到 90，就是"冻结云端"这个卖点的真实代价，论文里应如实报告 |

---

## 5. 备选延时鲁棒技术（若决定 beyond 当前配方）

按"改动量 × 预期收益 × 风险"排序：

1. **缓存利用率消融（评测侧，零训练）**：D=0 与 D=20 各跑一次，三种破坏——KV 全零、跨 episode 打乱 KV、反向消融（零化边缘 SigLIP token）。若打乱缓存成功率不掉，"双流训练失败"的定性就从"课程无效"改为"模型早已绕开缓存"，后续所有训练侧努力方向随之改变。**一次 AIFlow 评测的成本，决定整条技术路线。**
2. **边缘新鲜通道加宽**（`_edge_context` 改动，训练重跑）：1 token → 每视图 1 token（2 token），或 pooled + 空间网格 token（如 2×2×2 视图 = 8 token），或 CEVLA 式逐动作 token MLP 融合。直接攻 Goal 套件的空间精度瓶颈。保持"推理不用延迟元数据"卖点不变。
3. **KV adapter 启用**（recipe 已有 `libero_spatial_leapbotce_kv`，rank 64）：零初始化 K/V 低秩残差 + per-layer scales，只加 ~小参数量，云端 Wan 仍冻结。**这是"云端可适配"假说的最小成本检验**，且接口与 fresh/stale 两路完全兼容。
4. **云端 LoRA 直接解冻**（若 3 不够）：对 Wan attention 加 8–16 rank LoRA，λ_video 保持 0，动作损失回传——CEVLA 机制的低秩移植。与"Cloud training: none"卖点冲突，作为消融/上界实验而非主配方。
5. **年龄条件化**（谨慎）：给 expert 一个 age embedding 让它学会"缓存该信几分"。技术上合理，但与论文"推理不需要延迟元数据"卖点冲突；若做，可考虑训练时给 age、推理时用边缘侧自估年龄（如缓存帧与当前帧的低维统计差），改动大。
6. **潜空间前推重锚定**（世界模型特色，大改）：用 Wan 从 t−d 帧沿动作块把缓存"推演"到当前时刻再消费——把陈旧缓存的位姿细节重新对齐。这是 WAM 独有的选项（VLA 做不了），若走通是论文层面最有新意的技术，但工程量最大，建议在 1–4 的结论出来后再评估。
7. **fresh→stale 缓存一致性辅助损失**：强制边缘对同一时刻两种缓存的中间表征对齐（而非只对齐动作输出），给"忽略缓存"的平凡解加阻力。中等改动。

---

## 6. 建议行动清单

**P0（本周，均走 AIFlow，不占本地 GPU）**
- a. h200 配方复跑：bs128 / lr 2.8e-4 / 等样本量（~200k 步对应样本）完整退火 → 预期 D=0 75–80，作为新主配方基线。
- b. §5-1 缓存利用率消融（110k ckpt 即可，一次评测定方向）。
- c. KV adapter（rank 64）四套件训练一条，对照同配方 frozen-cloud。
- d. 200k 续训完成后提 200k 评测（新 tag，勿复用 110k）；dtrain40 重提评测（损坏调查）。

**P1（P0 结论出来后）**
- 边缘新鲜通道加宽（§5-2）；若 adapter 有效但不足，加云端 LoRA 消融（§5-4）。

**明确不建议**：继续在冻结云端下调 λ/warmup/采样分布——现有证据表明课程已把边缘模块推到它能到的位置，剩余差距不在课程超参上。

---

## 7. 设计提案：云端传输压缩 + 边缘并列注入（2026-10-11 定稿，**已实现**）

动机（用户定调）：① 云边协同下传输特征不能过大，当前 30 层逐层 cache ≈ 36 MB/replan（72 MB/s @2Hz）不可部署；② 信息来源要均衡，边缘实时观测应占 5–10%；③ 边缘特征"挑一挑"直接**并列**进 ActionDiT 的 attention 序列，不做 context 融合（本地无带宽压力）。

核心叙事：**云端传"慢变的 planning 摘要"（不怕延迟），边缘本地供"快变的几何细节"（没有延迟）——延迟鲁棒性从训练出来的性质变成架构内生的性质。**

### 7.1 数据流

```
云端 CloudPlanningCodec（重设计自 CloudKVAdapter）
  Wan 30层逐层cache [30×2×98×3072]
    → adaptive grid 空间池化 (7×14 → M 网格)
    → 跨层 softmax 加权聚合 α[l] (30维可学)
    → bottleneck MLP (3072→1024→3072)
    → 传输载荷 z [M, 3072] bf16        ← 唯一过网的东西

边缘 EdgeVisionEncoder（重设计）
  SigLIP 冻结 → patch tokens (196+CLS/视图, 768)
    → 每视图 CLS + CLS相似度 top-k 挑选（冻结侧打分，选择无需梯度）
    → 共享投影 Linear(768→3072) + per-layer scalar
    → z_edge [N, 3072]

ActionDiT 消费（MoT 每层 self-attn K/V 拼接）
  [z_cloud (M) ⊕ z_edge (N) ⊕ action 自身 (32)]   ← 并列注入
  cross-attn context 只留语言 + 边缘本体（低带宽、语义级）
```

第一版简化：z 同时作为 K 与 V（加 30 维 per-layer scalar），边缘侧不做 K/V 解码；z 无 RoPE（纯前缀 K/V，与现状 post-RoPE 混合方式一致）。v2 备选：边缘侧低秩 K/V 解码（传输量不变）。

### 7.2 档位与比例核算（KV token 预算）

| 档 | 云 M | 边 N | action | 边缘占比 | 传输/replan | 2Hz 带宽 | 云空间分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| A 轻压 | 98（不池化） | 14（CLS+top6 ×2视图） | 32 | 9.7% | 1.2MB | 2.4MB/s | 7×14 |
| **B 中压（推荐）** | **56（8×7）** | **10（CLS+top4 ×2视图）** | 32 | **10.2%** | **0.69MB** | **1.4MB/s** | 8×7 |
| C 深压 | 28（4×7） | 6（CLS+top2 ×2视图） | 32 | 9.1% | 0.34MB | 0.7MB/s | 4×7 |

现状对照：边缘 1/130 ≈ 0.7%，传输 36MB/replan。B 档传输压缩 ~52×，边缘信息占比 ~15×。

RoboTwin 同构：120 tokens（10×12）→ pool 5×12=60 / 边缘 3 视图×(CLS+top3)=12 → 占比 ≈11.5%；传输 0.74MB。

### 7.3 参数与改动清单

新增可学参数 ≈ 8.7M（codec：α 30 + scale 60 + bottleneck 2×3.1M + LN；边缘投影 768×3072 ≈ 2.4M）。

| 文件 | 改动 |
|---|---|
| `cloud_kv_adapter.py` | 新增 `CloudPlanningCodec`（池化+层加权+MLP→z）；保留原 adapter 类做 A 档 |
| `leapbotce.py` | `EdgeVisionEncoder` 改 patch-token 输出+top-k 挑选；`CloudPlanningCache` per-layer list → 单 z（接口字段名不变）；`_predict_from_cache` mask 改 [horizon, M+N+horizon] |
| `mot.py` | `forward_action_with_video_cache_tensor` K/V 拼接顺序 [z_cloud, z_edge, action] |
| `configs/model/leapbotce.yaml` | `cloudedge.kv_codec: {cloud_tokens: 56, edge_tokens: 10, edge_select: cls_topk}` |

### 7.4 风险与对照

- 池化后 K 的 RoPE 混合：接受（可学 scale 补偿）；若 Goal 套件掉点，改 pre-RoPE 池化 + 可学习位置编码。
- top-k 挑选偏语义、可能漏几何小物体：grid 均匀池化作对照消融。
- 云 98→56 的信息损失：A 档即"不池化只修正"的退化对照，可加跑一档回答"压缩本身损不损性能"。
- 双路损失/两段式 API/eval 脚本全部兼容，无需改评测协议。

### 7.5 实现状态（2026-10-11 落地，25/25 测试通过 + 真 SigLIP 冒烟通过）

与 7.1–7.3 讨论稿的偏差（实现为准）：

1. **cache 接口不变**：codec 输出仍是 30 层 per-layer list（z 广播 + per-layer scale），未改 `CloudPlanningCache` 为单 z——`mot.py`/`fastwam.py`/eval 脚本零改动即可兼容，K=V=z 语义相同。
2. **真 SigLIP 无 CLS**：timm SigLIP `num_prefix_tokens=0`，`forward_features` 返回 (196, 768) 纯 patch。参考向量改为 mean-patch（已冒烟验证无 NaN）；每视图 token 数恒为 `tokens_per_view`（无 CLS 占位时全为 top-k patch）。
3. **A3 对照已内置**：`edge_vision.selection: topk|grid`（grid = 均匀间隔取 patch），无需后续改代码。
4. **A4 消融开关已内置**：`cloudedge.edge_prefix_enabled: false`（eval-time override），关掉后回落云端-only 两路 attention，无需重训。
5. **K/V 解码器已实现可选**：`kv_codec.use_kv_decoder: true` 时边缘侧低秩解耦 K/V（rank 256），传输载荷仍为 z。
6. 实际配置键：`cloudedge.kv_codec: {enabled, pool_grid, bottleneck_dim, use_kv_decoder, decoder_rank, dropout}` + `edge_vision: {mode, tokens_per_view, selection}`；训练配方 `configs/task/libero_all_leapbotce_codec.yaml`（h200 配方，B 档：云 56 / 边 10 / 占比 10.2% / 传输 0.69MB）。

消融矩阵（正式跑 RoboTwin 前先在 LIBERO 小规模做完，再定最终方案）：

| 项 | 配置 | 状态 |
|---|---|---|
| A0 frozen-cloud | 现役 libero+robotwin 两线跑完即对照 | 训练中 |
| A1 codec 主配方 | `libero_all_leapbotce_codec`（无解码器） | 待跑 |
| A2 有解码器 | A1 + `use_kv_decoder=true` | 待跑 |
| A3 挑选方式 | A1 + `edge_vision.selection=grid` | 待跑 |
| A4 边缘贡献 | A1 训练，eval 时 `edge_prefix_enabled=false` | 开关已实现 |
