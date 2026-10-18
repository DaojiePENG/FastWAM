# LeapBotCE 实验监测与 AIFlow 提交记录

> 用途：跟进实验状态 + 提交 AIFlow 申请的速查。新条目在最上。
> 技术论证见 `leapbotce_dualstream_gap_analysis.md`，配方细节见根目录 `LEAPBOTCE.md`。

## 2026-10-19 02:10 第二轮探针齐:a5 梯子定案(+8ep 最优);a6 高 lr 中途回吐全景确认

**第二轮(10-18 16:45 起跑,卡 4/6)**:a5 终点 step_021696(+10ep)、
a6 step_019530(+9ep),D20→D0 各 10/10(实测 ~3.2h/delay,收尾跨零点)。

| spatial | a1 终点 | a5+8ep | a5+10ep | a6+7ep | a6+9ep |
|---|---|---|---|---|---|
| D20 | 83.6~84.4 | **88.8** | 85.6(↓3.2) | 82.0 | 76.8(↓5.2) |
| D0 | 94.4 | **94.8** | 92.4(↓2.4) | 96.0 | 93.2(↓2.8) |

- **a5 梯子定案:最优 = +8ep(step_017360)**。+8→+10ep D20/D0 同步
  退化(7/10 任务走低)——constant 4e-5 下过训临界在 +8ep 附近;a1 确实
  欠训(+8ep D20 +4.4~+5.2、D0 +0.4),但 +10 就过了。a5 线交付物 =
  step_017360,全量评测候选。
- **a6 高 lr 中途回吐(7→9ep,D20 -5.2 / D0 -2.8)**:9/15ep 处 cosine
  lr ~1e-4,churn 期双指标波动属预期(a1 同期是爬的,其 75.4→84.4 段
  lr 均值更低)。中途探针不下结论,**判据 = 15ep 终点(lr→0)**,预计
  10-19 ~05:50 落盘,05:43 自动起终点探针,12:13 终判。
- 细节:a6 task9 D20 48→88(+40)大恢复;其余中段任务 -12~-24 波动。

## 2026-10-18 17:05 a5/a6 首轮探针(双 10/10):两线双双验证用户判断

**提交实况(10-17)**:22117077 a5_cont 12:35 拉起(160.11);22117210
a6_lora_fs 18:36 拉起(160.21,排队 ~6h)。节奏 ~2.2-2.6h/ep。
**a5 已收尾**:10 ckpt 全落盘(2170×1..9 + 终点 21696);a6 到 +9ep
(19530),15ep 预计 10-19 凌晨。

**首轮探针(10-18 10:54 起跑,卡 4/6 各 1 worker,spatial D{20,0}×25,实测
~2.9h/delay,比 a4 经验的 5.5h 快近一半;卡 5 宿主涨到 53GB 弃用)**:

| | a1 终点 | a5@+8ep(17360) | a6@+7ep(15190) |
|---|---|---|---|
| spatial **D20** | 83.6%(recheck 25t)/84.4(release 50t) | **88.8%**(+4.4~+5.2) | **82.0%** |
| spatial **D0** | 94.4%(release 50t) | **94.8%**(+0.4) | **96.0%**(+1.6) |

**判读**:
- **a5 = a1 欠训练实锤**:纯续训 8ep,D20 +4.4~+5.2 且 D0 不掉反微升
  (+0.4)——a1 的 11.5ep 确实没训满,且不存在"续训砸 D0"的代价。逐任务
  D20 大面积上抬(task3/7 +20,task4/2 +12;仅 task0 -20、task9 -4)。
- **a6 = 从头 co-adapt 反超同期**:a6@7ep D20 82.0 vs a1@7.7ep 75.4
  (50-trial,Δ+6.6 且 a6 还少 0.7ep);D0 96.0 已超 a1 终点。LoRA+codec
  从 step0 共同训练不仅没有 a4 式谷底,学习速度还快于无 LoRA 的 a1——
  用户"两条干净线"的判断首轮全中。
- a6 注意项:D20 task9 48%(a1 76%)是唯一大坑,待 +9ep 看是否恢复。

**第二轮在途(16:45 起跑,同卡位)**:a5 终点 step_021696(判"是否还在
爬",平/降则 +10ep 定为最优 epoch 配方)、a6 step_019530(+9ep,出 a6 自身
斜率),D20→D0 串行,预计 D20 ~19:40 / D0 ~22:40。23:23 自动汇总。

## 2026-10-17 12:35 用户定两线:a4b 停线 → a5(纯续训)+ a6(LoRA 从头训)

**用户判决**:a4b"续训+中途注入 LoRA"构型**不伦不类**,改走两条干净线。
落地(FastWAM 新文件 ×4,验证全过;daojie 文档同步):

- **a5 = a1 欠训练检验**(任务名 `leapbot_ce_train_libero_a5_cont`,变体
  `AIFlow_config_libero_a5_cont.py` + task yaml `..._codec_cont.yaml`):
  a1 终点 weight-only 续训,**无 LoRA**,constant lr **4e-5**(a1 峰值 2.8e-4
  的 1/7,落在 a1 尾段仍产出的 ~3-11e-5 区间中位偏下;2e-5 有假阴性风险
  ——平曲线分不清"饱和"还是"lr 太小没推动"),10ep = 21696 步,每 1ep 存
  1 ckpt(+1..+10ep 梯子,keep 10 ≈20GB),**stale warmup 显式 0**(step0 即
  满权重 λ0.5 = 续 a1 终点状态;a4 的 -1 自动半程 ramp 会让前半程 λ 重爬,
  污染"最优 epoch"判读),mot_pretrained_path 钉死同源。判读:逐 ep 本地
  探针(spatial D{20,0}×25)——D20 持续爬 = 欠训练实锤,取爬平点为最优
  epoch;+3ep 平 = a1 已收敛。8×H200 ~19h。
- **a6 = LoRA+codec 从头联合训练**(任务名 `leapbot_ce_train_libero_a6_lora_fs`,
  变体 `AIFlow_config_libero_a6_lora_fs.py` + task yaml `..._codec_lora.yaml`):
  a1 原配方逐项对齐(冷启动官方 ActionDiT ckpt、lr 2.8e-4 cosine 全退火、
  λ0.5 warmup -1 自动 total/2 = 16272 与 a1 同构)+ video LoRA q/k/v/o
  r16/α16 从 step0 co-adapt——a4 谷底教训的直接解药(中途注入打散
  video×codec×action 联合适配,D20 未爬回而 D0 93.6 完好 → 破坏是 delay
  特异的,co-adaptation 必须从头),15ep = 32544 步(超 a1 的 11.5ep 共
  3.5ep;每 ep 存 ckpt,中途饱和可杀任务保留已存 ckpt)。8×H200 ~29h。
- **评测口径(两线零改动)**:a5 的 sweep 推导落 task
  `libero_all_leapbotce_codec`(模型侧逐字一致);a6 的推导落
  `libero_all_leapbotce_codec_lora_a4`(a6 yaml model 块与 a4 逐字一致,
  strict load 兼容)——全量评测届时只改 eval 变体的 RUN_DIR/CKPT/TAG。
- **验证**:yaml 校验 ×2 PASS;分发器路由(a5_cont / a6_lora_fs 整词后缀
  命中)PASS;CMD a5 12 项 / a6 11 项 ALL PASS(含 stale-warmup 0/-1、
  resume 有无、冷启动无 --resume 反向断言)。a4b 变体文件加停线横幅留档
  (误提交 = 复现 a4 式中途注入)。

## 2026-10-17 11:55 a4 终点 D0 定案(93.6%,D0 完好)→ 破坏是 delay 特异的 + A4b lr 终定 2e-5

**D0 终表(a4_probe_step010848_loc25/delay_0,10/10,25 trials)**:宏 a1
终点 94.4%(50-trial release 口径)→ a4@5ep **234/250 = 93.6%(Δ-0.8,
噪声内)**,逐任务 -4~+14 无系统性下坠 → **零时延行为 5ep 完全重构**。
⚠️ 基线源注意:D0 对照必须用 `a1_release_step24951_probe_local/delay_0`
(宏 94.4);`a1_codec_step24951_diag` 的 D0(59.0%)是早期坏口径残留
(task4 30%/task7 0%),不可用。

**机理细化**:结合 D20 76.8%(-6.8~-7.6,两口径),naive 注入的破坏是
**delay 特异的**——D0 完好说明"训练破坏原特征"应修正为"破坏 stale-KV
场景下的联合适配"(video 特征 × KV codec × action expert 的 delay 补偿
模式),纯零时延技能没被砸掉且重构容易;delay 鲁棒性是三者联合训练
24951 步攒出来的,LoRA 扰动后重构慢且 5ep 未完。(a4@2ep D0 缺测:004340
探针 v1 的 D0 是坏推导残留空壳,只有 D20 可用。)

**A4b lr 终定 2e-5**(用户 10-17 定;02:50 条目的 5e-5 系我当时的推荐,
作废;1e-5 过保守否决——LoRA B 从 0 起步学习信号弱,2ep B 范数可能
<0.15,退化成纯续训)。run name 已同步
`codec_a4b_lora_lr2e5_alpha16_seed42_h200`,双仓库已 commit。
**2ep 门控判据(step_002170 探针,spatial D{20,0} × 25 trials)**:
理想 D20 ≥ 80 且 D0 ≥ 92 → 续到 5ep;B 范数 < 0.15 → lr 升 5e-5 重提;
D20 < 76(还在砸)→ 降 1e-5 或拉长 stale warmup;5ep 不够 → 从终点续跑。

## 2026-10-17 02:50 a4 终点 D20 定案(76.8%,未爬回)→ A4b 配方就绪可提交

**终点探针终表(10/10)**:宏 2ep 52.0% → 5ep **76.8%**(基线 84.4%,还差
-7.6,<80 判据)。爬回非单调:task7 倒退(60→52)、task0 顽固(60 vs 94)、
task9 反超 +14(任务噪声)。a1 对照 10/10 全出:宏 84.4%±噪声、逐任务
±10 内 → **链路清白终局钉死**。**A4 结论:谷底路线判死**——恢复慢
(~9分/ep)、不完全、非单调。

**A4b(B 计划)配方已就绪并验证**(FastWAM commit):lr 1e-4→**5e-5** +
α 32→**16**(scale 2→1,扰动速率 ~1/4),resume 仍 a1 step_024951,5ep =
10848 步不变,每 1ep 存 ckpt(同口径可对比),run name
`codec_a4b_lora_lr5e5_alpha16_seed42_h200`。CMD 8 项验证全过。
**提交方式不变:任务名 `leapbot_ce_train_libero_a4_lora`**(变体文件原地
更新),8×H200 单机 ~15h(参考 16 卡 = 20ep 等效)。

**在途收尾**:a4 终点探针 D0 段(sweep 自动转段,~05:30 出)——补"当前
配方 D0 是否被拉升"证据,不阻塞 A4b 提交。
## 2026-10-16 23:05 a4 训练完成(10848/10848)+ D20 终表(10/10 全负,宏 52.0%)+ 终点探针启动

**训练**:23:59 前后 10848/10848,终点 ckpt step_010848.pt 22:43 落盘
(全 5 ckpt:2170/4340/6510/8680/10848,各 2.08GB)。wandb offline 数据在
run 目录,待 sync。

**D20 终表(004340 探针,10/10,25 trials)**:a1 终点 84.4% → a4@2ep
**52.0%(Δ-32.4),10/10 任务全负**(-8~-60;task5 88→28 最惨)。叠加
a1 对照 task0 92%(Δ-2,链路清白)+ action expert 偏移 median 49%/head
134% + LoRA B 范数 1.06→1.55 → **归因链闭环:naive LoRA 注入
(lr 1e-4 + scale 2)把已收敛技能砸出谷底,2ep 未恢复**。

**在途**:① 终点探针 a4_probe_step010848_loc25(卡 5/6,22:51 启动,
D20→D0,~01:20/04:00 出)—— 判"5ep 爬回了没有";② a1 对照全 10 任务
(卡 4,~00:30-01:00 出)—— 钉死链路清白。004340 实例 D20 齐后已
kill-by-pgid(pgid 62156,防自动进 D0 占卡)。

**B 计划(终点探针未爬回即触发)**:lr 1e-4→5e-5 + α 32→16(scale 2→1),
扰动速率 ~1/4;LoRA lr warmup 留下一档。
## 2026-10-16 21:40 a4 探针判读:逐任务全面大跌 + action expert 大幅重写 + B 计划定案(等终点数据触发)

**分任务对照(a1 终点 50trials 基线 vs a4@2ep 25trials,同任务同难度)**:
task0 94→48、task1 96→48、task2 90→72、task3 72→60、task4 80→28 ——
全面 -12~-52 分,非难度分布假象。

**ckpt 正确性已排除**(step=4340/120 wrappers/正确 run 目录/strict load 过)。
**数值实锤破坏**:action expert 相对 a1 终点偏移 median 49%、head 134%
(正常续训 1-10% 量级);LoRA B 范数 1.06(1ep)→1.55(2ep)加速爬升,
ΔW=2·B·A 直接加在冻结 base 上 → video 特征漂移 → action expert 追移动
目标整体重写 → 已收敛技能暂时崩。loss 平滑降(0.046)= 重构中,但 a1 历史
轨迹(66.4→84.4 爬 8ep)表明 5ep 大概率爬不回。

**用户方案讨论定案**:zero-B 初始化已实现(video_lora.py reset_parameters
= kaiming A + zeros B,step0 严格=a1 终点);问题在漂移速率不在初始化。
**B 计划(等终点探针触发)= ① lr 1e-4→5e-5 + ② α 32→16(scale 2→1)**,
正交零风险,扰动速率 ~1/4;③ LoRA lr warmup 留下一档;stale warmup -1
不动(与本案无关)。验证线:a1 终点同链路重评(卡 4,21:00 起,排除
eval 链路错;~22:00 前几任务出方向)→ a4 终点探针(22:26 训练完,
卡 5/6 D20+D0,~01:15 出)。
## 2026-10-16 18:30 a4 探针本地开跑(用户指示)+ eval task 推导修复(三坑实录)

**用户定案**:检查点探针本地调试卡跑,及时出结果。执行 = step_004340 先探,
卡 5/6 各挤 1 worker(~25GB,与宿主进程共存,缓冲 ~17GB),spatial ×
D{20,0} × 25 trials(250 eps/档)。18:27 worker 全部过模型加载 +
**video LoRA 120 wrappers 注入 + step_004340 加载通过**。时间线:D20
~20:15 出,D0 ~22:15 出;002170 随后(视 004340 结果)。对照基线 a1:
2170 步 D0 90.0/D20 66.4(a1 探针 50 trials 口径,本次 25 trials,判读
留 ±3% 噪声余量)。

**三坑实录(30 分钟排障)**:
1. **eval task 推导缺 LoRA 分支**:sweep 脚本 auto 推导只看 kv_codec →
   task=libero_all_leapbotce_codec → LeapBotCE.load_checkpoint raise
   "Checkpoint contains video_lora state ... rebuild with LoRA injected"
   (3 worker 全灭)。**修复 = 推导加 video_lora.enabled → lora_a4 task
   优先级最高**(a4_lora 全量变体明早同踩此雷,已一并解除),推导单测
   过(a4 config → lora_a4),probe 显式传 LIBERO_EVAL_TASK 双保险。
2. **双实例互踩**:v1(坏推导)delay_20 失败后转 delay_0 继续跑,与 v2
   delay_20 同时挤三卡 → 79GB 满 OOM(v1 workers 42GB 残留未清)。
   kill-by-pgid 两个进程组全停(vlash 17 天老进程 502767 不在组内,未动)。
3. **挤卡 OOM**:宿主 3 进程 37GB + worker 25GB 本可共存,叠加残留后爆。
   清理后单实例双卡,expandable_segments:True 防碎片,稳态 62.9/61.1GB。

教训:**sweep 重跑前必须 ps 确认旧实例全灭**(manager 失败会自动串行
转入下一 delay,不会自己退)。
## 2026-10-16 16:55 a4 中程进展 + run 目录推导实证 + probe 决策(用户定:本地跑,不占集群)

- **进度** step 5130/10848(47%,2.4ep),0.29 step/s,ETA 5:37 → **终点 ~22:26**。
  loss 0.4853(起步)→0.0461 平滑下降(5080 处 fresh 0.0289/stale 0.0403);
  λ=0.4682,~300 步后(5424)满 0.5 进入纯 1:1 双流。ckpt step_002170(13:51)/
  step_004340(16:03)已落盘(2.08GB/个),6510 预计 ~18:15。
- **wandb 降级 offline**(启动脚本双保险生效:代理 127.0.0.1:33716 不可达),
  曲线在 run 目录 wandb/offline-run-...-26930wht,终点后 wandb sync 补传。
- **run 目录推导实证**:输出落在
  `runs/libero_all_leapbotce_codec_lora_a4/codec_a4_lora_seed42_h200/`
  —— **runs/<hydra task 名>/<run name>**,不是 a1 的
  `runs/libero_all_leapbotce_codec/`(a1 task 名恰好没有后缀段)。两个 a4
  eval 变体的 RUN_DIR/CKPT 按此修正(否则集群提交即路径错),已复验+提交。
- **probe 决策(用户 16:49)**:spatial 探针属门控信号,单卡足够
  (PACK=1 串行 ~2-3h),不提交 8×H200;本地 H800 现状 8 卡全满(宿主进程
  795xxx 连号 8 分片,非本 pod 任务,不可管理)→ 等 4-6 空闲本地探
  step_004340(可选增益,不阻塞主线);**主线 = 终点后直接全量评
  leapbot_ce_eval_libero_a4_lora**(中间 ckpt 全在盘上,随时可补探)。
  a1 同步数对照基线不变:2170 步 D0 90.0/D20 66.4。
## 2026-10-16 13:45 a4 拉起核实（22062112，三标志全中）+ stale warmup 语义勘误

**任务 22062112 `leapbot_ce_train_libero_a4_lora`** 11:17 提交 / 11:48 拉起
（10.194.160.21，8×H200），日志 `jinnTrainResult/leapbot_ce_train_libero_a4_lora-1ce47224/`。
三标志全中：① `task=libero_all_leapbotce_codec_lora_a4`；② `video LoRA
injected: 120 wrappers, 11.80M trainable (rank=16, q/k/v/o)`（8 rank 各打印
一次）；③ resume 加载 a1 `step_024951.pt`。13:32 进度 step≈1890（0.30
step/s，39.0 samples/s），ETA 8h10m → **~21:45 终点**（10848 步总时长约
10h，快于 15h 预估）。loss 0.0561（fresh 0.0526 / stale 0.0729）正常。
首个 ckpt step_002170 约 13:50 落盘。

**stale warmup 语义勘误（10-16 实跑核实）**：`stale_loss_weight()` 的
"warmup<=0 → 满权重" 分支只有**显式传 0** 才触发；传 **-1** 时
`set_training_step()`（leapbotce.py:283-286）在第一步自动改写为
`total_steps//2`，线性 ramp 至半程。基座注释"-1=自动总步数/2"是**正确的**，
本文件与变体中此前"立即满权重"的注解系误判，已修正。实测轨迹吻合：
a4（10848 步 → warmup 5424）step1610→0.1483、step1870→0.1732 =
0.5×step/5424 精确吻合。**a4 与 a1 配方同构**（a1：24951 步 → warmup
12475，同为半程 ramp），动力学可比性保持，任务无需任何干预。对探针判读的
含义：同 step 处 a4 的 λ 比 a1 大（2170 步处 0.200 vs 0.087），早期 ckpt 的
D20 探针预期略偏向受益侧。
## 2026-10-16 08:20 a0 收官（22041674，4.75h 全过）+ a3 D0 确认补齐 → 四线完整矩阵落表；唯一待提交 = a4 训练线

**两任务判读**：
- **a0（22041674 / b83b4400，节点 10.194.160.30）**：23:21 起 4.75h，D{0,10,20,40}
  全部 40/40（total 2000/档）✓ —— 10-15 准则首战生效（无 D5）。汇总
  `baseline-evaluations/leapbotce/a0_release_step24951/delay_sweep_summary.json`。
- **a3（22041663 / a673b9e6，146s）**：skip-existing 全跳过 + 汇总再生成 ——
  D0 实际已于 10-15 16:50 前落盘（fill-up 白天完成），五档全齐 40/40 ✓。

**四线完整矩阵（D{0,10,20,40}，官方 expert D0=97.4）**：

| 线 | D0 | D10 | D20 | D40 | 保留率 D20/D40 | 接口税 (97.4−D0) |
|---|---|---|---|---|---|---|
| a0 无压缩 | **91.9** | 85.1 | 79.8 | 69.0 | 86.8 / 75.0% | 5.5 |
| a1 codec | 91.5 | 84.3 | 81.2 | 70.1 | 88.7 / 76.6% | 5.9 |
| a2 codec+kv_dec | 89.9 | 85.1 | **81.6** | **70.5** | **90.8 / 78.4%** | 7.5 |
| a3 edge-kv | 88.3 | 81.2 | 77.4 | 68.6 | 87.7 / 77.7% | 9.1 |

**判读**：① a0 D0 最高但跌得最快（D20 被 a1 反超）—— 无压缩的 D0 优势
不传导到时延档，与「生产-消费适配是瓶颈」结论一致；② 保留率冠军 = a2，
绝对值 D20/D40 冠军 = a2；③ a4（续训+LoRA）目标 = 把 a1 的 D0 拉向 95+
同时保住保留率优势，完成后四线表加一行直接对比。

**当前状态**：四线 × D{0,10,20,40} 全量数据齐 ✓，平台无排队/运行任务。

**a4 提交方式修正（10:xx 实录）**：用户平台提交 leapbot_ce_train_libero
（22057922，10:21 拉起）——平台界面无 CMD 编辑框，沿用基座 AIFlow_config_libero.py
常改区 = 旧 200k 续训命令（--task libero_all_leapbotce --gpus 4 --steps 200000
resume state/step_110000），不是 a4。用户已取消该任务。**修正 = 新建 dispatcher
变体 `AIFlow_config_libero_a4_lora.py`**（训练线函数 globals 绑定定义模块，env
薄委托无效，故全量复制基座改常改区；管线逐字一致）：任务名
`leapbot_ce_train_libero_a4_lora` → 整词匹配后缀 libero_a4_lora。本地已验证：
py_compile ✓ / 分发器选中 ✓ / 旧任务名 leapbot_ce_train_libero 仍落基座（200k
线不受影响）✓ / 拼出 CMD 与 21:15 手工版逐参数一致（--task ..._a4 --gpus 8
--bs 16 --steps 24951 --lr 0.0001 --scheduler constant --save-every 12475
--keep-ckpt 2 --resume <a1 .pt> -- model.mot_pretrained_path=...）✓ / resume
为 .pt 文件 → world-size 守卫自动跳过 ✓。

**待提交清单（10-16 定案：8 卡 + epoch 主口径 5ep + 每 epoch 存 ckpt 测评）**：

| 优先级 | 任务名 | 卡型×卡数 | 总步数/等效 | 时长 | 前置 | config |
|---|---|---|---|---|---|---|
| 1 | leapbot_ce_train_libero_a4_lora | H200×8 单机 | **5ep = 10848 步**（1ep = 277713/128 = 2170 步） | **~15h** | 无（冒烟已过，a1 终点 ckpt 已在存储） | FastWAM AIFlow_config_libero_a4_lora.py（task=libero_all_leapbotce_codec_lora_a4） |

- **epoch 主口径（10-16 用户准则）**：步数由 epoch 折算（MAX_STEPS=0 +
  EPOCHS=5），不按 step 直设；1ep = 2170 步 @ eff128。a1 的 24951 步按此口径
  = 11.5ep（管理文件旧注"10 epochs"系笔误，趋势探针 8317/16634/24951 步 =
  3.8/7.7/11.5ep）。
- **每 epoch 存 1 个 ckpt**（用户指示留测评）：save_every=2170 →
  2170/4340/6510/8680/10848 共 5 个，keep 5（delta ≈2GB/个 ≈10GB）。
  2170 步 ckpt 落盘即可做第一轮 spatial 探针（对比 a1 @3.8ep：D0 90.0 / D20 66.4）。
- 多卡数参考（用户准则：预期给多档）：16 卡双机 = 24951 步（20ep）~36-40h；
  变体天然兼容双机（NNODES/MASTER_ADDR 自动注入，run 名加 _n2），本轮用户
  已定 8 卡。

**判读补充**：a0 D0 最高但跌得最快（D20 被 a1 反超）—— 无压缩的 D0 优势
不传导到时延档，与「生产-消费适配是瓶颈」结论一致；保留率冠军 = a2，
绝对值 D20/D40 冠军 = a2。a4（续训+LoRA）目标 = 把 a1 的 D0 拉向 95+
同时保住保留率优势，完成后四线表加一行直接对比。

**提交后节奏**：首个 ckpt step_002170（~3h）落盘 → spatial 探针门控（对比
a1 同步数基线 D0 90.0 / D20 66.4）→ 每 epoch 滚动探针 → 终点 step_010848 →
全量 D{0,10,20,40} 评测（8×H200 ~5.5h/线，届时给提交表格）。若 5ep 探针显示
保留率仍陡峭上升，可从终点 ckpt 续练（变体 resume 换行即可）。非迫切暂缓
不变：λ 消融（2:1/3:1/4:1）、P0 200k 终点评测、RoboTwin 侧两 fork。

## 2026-10-15 21:15 a4 LoRA 全链路冒烟通过：训练侧 20 步 + 评测侧 load+双路前向，可以提交

**a4 = a1 终点 resume + LoRA(video expert q/k/v/o, r16/α32) + 10ep constant lr 1e-4**，
双重收益单线归因：续训 → 保留率继续爬（趋势探针 D20 +9.0/3.3ep 零递减）；
LoRA → D0/goal/10 缺口（纯续训 D0 6.7ep 已饱和，D0 上涨只能归因 LoRA）。

**实现（全部已 commit）**：
- `video_lora.py`：LoRALinear（base 存普通 list 不进 module 树 → trainer
  freeze-then-enable 只启 lora 因子，base 恒冻结）+ inject_video_lora（120 wrappers
  = 30 层 × qkvo，11.80M 参数，Kaiming-A/零-B → 注入瞬间行为=a1 终点）。
- factory（leapbotce_edge_train_factory.py）：pop video_lora 配置 → 注入 → 注册
  `model.dit["video_lora"]` → **post-init 开 video expert checkpointing**。
- save/load（leapbotce.py）：checkpoint 增 `video_lora` 段（120×(lora_a,lora_b) +
  meta）；load 侧 states/wrappers 数量不匹配即 raise，无 states 有 wrappers 仅警告。

**关键坑（MoT 守卫）**：MoT.__init__ 硬禁任何 expert 开 gradient checkpointing
（"compiled tensor core" 守卫，mot.py:53-62），但其运行时 checkpoint 路径
`_apply_post_with_optional_checkpoint` 是完整实现且 `compile_training_layers=False`
硬编码（:31）→ compile 冲突实际不存在。**解法 = 构造期保持 false 过守卫，
factory 注入 LoRA 后（MoT.__init__ 已过）直接设
`model.video_expert.use_gradient_checkpointing = True`**（MoT.mixtures 持同一对象，
运行时读取生效）。a1/a2/a3 路径零改动。

**训练侧冒烟（B4, 20 步, GPU4/H800）**：权重加载 74s 无 modelscope 问题（17:55
前夜的 ConnectTimeout = env 未齐/瞬时网络，非代码）→ MoT 构建过守卫 → LoRA 注入
打印 ✓ → resume 加载 a1 step_024951（weight-only）✓ → 20/20 步 loss 0.52-1.16 ✓
→ ckpt 含 video_lora 120 项，**lora_b abs-sum=22.0 ≠ 0（梯度确实流动）** ✓。

**评测侧冒烟（关键新链路）**：eval 与训练共用 `_target_=
leapbotce_edge_train_factory.create_leapbotce`（configs/model/leapbotce.yaml:1）→
eval 按 run config instantiate 自动注入 → load_checkpoint strict 恢复
**lora_b=22.0000 与训练端逐位一致** ✓ → encode_cloud 前向 ✓（cache 56 token/30 层，
LoRA 在 cloud KV 生产路径生效）→ infer_action_edge 前向 ✓（edge 消费路径 2 步去噪
完整返回）。eval 里 self.training=False → checkpointing 不触发，零推理开销。
sweep 脚本固定 `model.load_text_encoder=true`（:315），评测载入 T5 在线编码。

**显存外推**：B4/H800 冒烟自用 ~16GB（权重 6GB + ZeRO-1 优化器 + B4 激活，
checkpointing 已省掉 _post_fn 大头）。B16/H200 激活增量估算：30 层 ×
(attn_input+q+k+v+mixed) × [16,~300,3072] bf16 ≈ +5GB 量级，141GB 富余极大。

**a4 训练提交命令（8×H200 单机，模板=a1_release bb2ed8e6 实测改参）**：

```
bash scripts/train_leapbotce_libero_2653853.sh \
  --task libero_all_leapbotce_codec_lora_a4 \
  --gpus 8 --batch-size 16 --accum 1 --delay 20 --seed 42 \
  --steps 24951 --lr 0.0001 --scheduler constant \
  --weight-decay 0.01 --grad-clip 1.0 --workers 8 \
  --stale-lambda 0.5 --stale-warmup -1 \
  --save-every 12475 --keep-ckpt 1 \
  --run-name codec_a4_lora_seed42_h200 --wandb \
  -- resume=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce_codec/codec_a1_release_seed42_h200/checkpoints/weights/step_024951.pt \
     model.mot_pretrained_path=/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt \
     wandb.name=codec_a4_lora_seed42_h200 wandb.project=leapbotce
```

时长预估：a1_release 实测 ~24h（10-13 13:14 → 10-14 13:23）；a4 video expert 建图
+ checkpoint 重算 → **~30-36h**。中点 ckpt step_012475（save_every 12475）落盘后
可先做 spatial 探针门控，不必等终点。

**待提交清单（更新，表格制式）**：

| 优先级 | 任务名 | 卡型×卡数 | 时长 | 前置 | config |
|---|---|---|---|---|---|
| 1 | leapbot_ce_eval_libero_a3_release（同名重提，只补 D0 三套件）【已提交排队】 | H200×8 | ~1.5h | 无 | daojie_leap_configs/AIFlow_config_eval_libero_a3_release.py |
| 2 | leapbot_ce_eval_libero_a0_release（重提；坏节点 ~2min 快速失败再重提）【已提交排队】 | H200×8 | ~6.8h | 无 | daojie_leap_configs/AIFlow_config_eval_libero_a0_release.py |
| 3 | leapbot_ce_train_libero（a4 LoRA 线，命令见上） | H200×8 单机 | ~30-36h | 无（a1 终点 ckpt 已在存储；冒烟已过） | FastWAM task=libero_all_leapbotce_codec_lora_a4 |

a4 与 1/2 不冲突（训练 vs 评测不同资源池），可同时提交（24 卡 < 空闲常态 ~32）。
暂缓不变：λ 消融线（非迫切）、P0 200k 终点评测、RoboTwin 侧（两个 fork 在跑，不动）。

## 2026-10-15 17:55 a0 失败定性：又抽中小内存节点（8 workers 集体 SIGABRT）；脚本加内存 preflight；全盘提交清单

**a0（22012929 / df18020a）错误定性**：17:00 起 pod，每个 delay 双尝试全灭
（rc=1, 0/40）。worker log 显示 episode 正常跑（多个 success=True MP4 落盘）
→ 无声 SIGABRT —— **10-14 内存墙死亡模式，但这次 PACK=1（8 workers ~240GB）
也撑不住** = 平台节点池里存在 MemAvailable <200GB 的更小内存节点（比 10-14
的 350-700GB 墙更低）。四 delay 全灭 → 此节点必死，非任务问题。

**防复发（73a63f1）**：sweep 脚本加内存 preflight —— `MemAvailable < 260GB`
直接 fail-fast（exit 3，日志明示"同名重提换节点"，旁路开关
LEAPBOTCE_SKIP_MEM_PREFLIGHT）。坏节点从浪费 40 min 变 ~2 min 快速失败。
本调试节点 1435GB，正常节点（≥700GB）不误伤。

**全盘盘点（17:55，平台无排队任务）**：
- 已完成：a1 全量 ✓ / a2 全量 ✓ / a3 全量 D5-D40 ✓ / 本地趋势探针 ✓
  （读数见 14:35 条目）；a2/a3 的 stale handle 均展示问题无需重提。
- 失败待重提：a0 全量（0/40，坏节点，重提即可）。
- 缺口待补：a3 D0 的 object/goal/10（fill-up 重提）。
- 进行中（本地）：a4 = LoRA(Wan qkvo r16)+codec 续训线实现（未完，冒烟后给提交表格）。

**待提交清单（表格制式）**：

| 优先级 | 任务名 | 卡型×卡数 | 时长 | 前置 | config |
|---|---|---|---|---|---|
| 1 | leapbot_ce_eval_libero_a3_release（同名重提，只补 D0 三套件，守卫已验证） | H200×8 | ~1.5h | 无 | daojie_leap_configs/AIFlow_config_eval_libero_a3_release.py |
| 2 | leapbot_ce_eval_libero_a0_release（重提；若再抽坏节点 ~2min 快速失败，再重提即可） | H200×8 | ~6.8h | 无 | daojie_leap_configs/AIFlow_config_eval_libero_a0_release.py |
| 3 | a4 LoRA 训练线（任务名待定） | H200×8 | ~1.5 天 | 本会话完成 LoRA 注入实现+本地冒烟 | 待产出 |

1/2 可同时提交（合计 16 卡，空闲常态 ~32）。暂缓：λ 消融线（用户指示非迫切）、
P0 200k 终点评测（旧线，新矩阵优先）、RoboTwin 侧（两个 fork 在跑，不动）。

## 2026-10-15 14:35 a2 全量收官（stale handle 同剧本，数据完好）+ tied K=V 判决 + a1 趋势探针三行表

**a2 平台任务（21998406）错误定性**：14:16:50 完成 D40（40/40）后 14:17 死于
`error reading input file: Stale file handle` —— 与 a3 完全同剧本（a2 pod 08:11
拉起，早于 09:50 re-exec 加固，全程持 NFS 句柄）。**全部 5 delay 完成 40/40，
数据完好，纯展示问题，无需重提**。加固只对之后拉起的 pod 生效。

**a2 矩阵（codec + 低秩 kv_decoder r256，spatial/object/goal/10）**：

| | D0 | D5 | D10 | D20 | D40 |
|---|---|---|---|---|---|
| a2 | 89.9 | 87.8 | 85.1 | 81.6 | 70.5 |
| a1 | 91.5 | 87.2 | 84.3 | 81.2 | 70.1 |

**tied K=V 判决：a2 ≈ a1**（D0 −1.6 全来自 goal 83.8 vs 89.6，D10-D40 +0.4~0.8，
均噪声带内）→ **kv_decoder 解耦非主要瓶颈**，与 a0≈a1 一起把归因钉在
「生产-消费适配 / 训练配方」而非 codec 结构。a2 的 D20/D0 保留率 90.8% 略优于
a1 88.7%。

**a1 检查点趋势（spatial 500 eps/格，10:41 发车 13:0x 收官）**：

| ckpt | D0 | D20 | D20/D0 保留率 |
|---|---|---|---|
| step_008317 (3.3ep) | 90.0 | 66.4 | 73.8% |
| step_016634 (6.7ep) | **94.8** | 75.4 | 79.5% |
| step_024951 (10ep) | 94.4 | **84.4** | 89.4% |

**判读**：① D0 在 ~6.7ep 即饱和（94.8），之后持平（−0.4 < 噪声 ±1.3）——
**"训太多步把 D0 训坏"在 8317→24951 范围内不成立**；② 后期 3.3 个 epoch 的
收益全在 delay 鲁棒性（D20 +9.0 / 保留率 +9.9 点）→ 训练预算有效，无需减步；
③ D0 天花板 ~94-95 早饱和指向**结构/接口限制**而非配方——fresh:stale 再平衡
（2:1/3:1/4:1，λ_max 0.33/0.25/0.20）改变 tradeoff 位置但难推高已饱和的 D0；
**spatial 缺口已基本补齐（94.8 vs 期望 95），全量 macro 缺口在 goal (89.6) 与
10 (87.2)**。要过 95+/解 goal-10，主力手段仍是 LoRA 生产端适配（a4 待批）。

**待提交**（表格制式，10-15 用户指示）：

| 优先级 | 任务名 | 卡型×卡数 | 时长 | 前置 | config |
|---|---|---|---|---|---|
| 1 | leapbot_ce_eval_libero_a3_release（同名重提，只补 D0 三套件） | H200×8 | ~1.5h | 无；加固脚本不会跳 D0 | daojie_leap_configs/AIFlow_config_eval_libero_a3_release.py |
| 2 | leapbot_ce_eval_libero_a0_release | H200×8 | ~6.8h | 无；D{0,10,20,40} 新准则 | daojie_leap_configs/AIFlow_config_eval_libero_a0_release.py |

a1/a2/a3 主任务全部完成，均无需重提。

## 2026-10-15 11:05 用户补充：若过度训练实为 fresh:stale 比例失调——λ_max 消融立项（非迫切）

用户定性修正：如果趋势测试显示"训太多步 D0↓"，本质不是步数问题而是**双流比例
失调**——当前终态 fresh:stale = 1:1（`leapbotce.py:477`
`loss = (1−λ)·fresh + λ·stale`，λ→`stale_loss_lambda_max=0.5`，
`stale_loss_warmup_steps=-1` 即无预热从第一步满值）应该给 **fresh 线加权**：
（**10-16 勘误**：-1 实为自动取 total//2 半程 ramp，并非无预热满值——见
10-16 13:45 条目。λ 消融若要"前期 λ=0"应显式传 `--stale-warmup 0`。）
候选 fresh:stale = 2:1 / 3:1 / 4:1，对应 `stale_loss_lambda_max` =
0.33 / 0.25 / 0.20（旋钮现成，config 一处改动）。

**后续 ablation（非迫切，时间紧，慢慢来）**：趋势测试结果出来后，若 D0 随步数
下降，优先排一条 λ_max≈0.25~0.33 的训练线（可叠加 λ warmup：前期 λ=0 专注
fresh/D0，后期升到目标值保保留率），稳态后再考虑全扫 2:1/3:1/4:1。
另一个免费手段：中期 eval 挑 ckpt（与 λ 无关，任何配方都受益）。

## 2026-10-15 10:45 用户双指示：时延档位准则 D{0,10,20,40} + a1 检查点趋势测试发车

**准则 1（长期）**：时延评测一律只测 D{0,10,20,40}，不跑 D5（memory 已存
`delay-eval-delays-rule`）。落地：a0 全量变体 `AI_FLOW_DELAYS` 改 "0 10 20 40"
（FastWAM 65308c8 + daojie acdccc6 双树），a0 提交预估 8.5h→~6.8h；a3 fill-up
只缺 D0 不受影响。a1/a2 的 D5 数据仅作参考列，论文矩阵列 = D0/D10/D20/D40。

**准则 2 → 趋势测试**：用户假说 = 双流 0.5:0.5 加权下训太多步「保留率↑但 D0↓」
（过度训练），参照 CloudEdgeVLA 自家 ~95+。测试布局（卡 4/5/7，卡 6 有 VLASH
长跑绕开；spatial × 50 trials = 500 eps/格，seed 1234）：

| ckpt | D0 | D20 | tag |
|---|---|---|---|
| step_008317 (3.3ep) | 新跑 | 新跑 | a1_release_step008317_probe_local |
| step_016634 (6.7ep) | 新跑 | 新跑 | a1_release_step016634_probe_local |
| step_024951 (10ep) | 已有 94.4 | 新跑 | a1_release_step24951_probe_local（复用，skip D0 ✓） |

10:41/10:42 三实例起跑，driver log = `$LR/driver_a1_step{008317,016634}_trend.log`
+ `driver_a1_step024951_trend_d20.log`；24951 的 skip 走新守卫（"规范目录已满
10 份"）验证通过。ETA ~2-3h。判读：D0 随步数单调降 + D20 保留率升 = 过度训练
实锤（药方：减步数/λ warmup/中期挑 ckpt）；D0 单调升 = 欠训练（LoRA/续训方向）；
16634 峰值 = 最优 ckpt 在中间 → 后续配中期 eval 挑选。

## 2026-10-15 10:30 codec 架构 + D0 差距归因 + 训练预算对比（系统分析）

**Codec 结构**（`src/fastwam/models/wan22/cloud_kv_adapter.py`）：不是「先适配再 codec」——
`cloud_kv_adapter`（低秩残差适配器）在全部 release 线 enabled:false，**KV 直接进 codec**。
数据流 = 30 层 video-expert **V cache（K 整体丢弃）** → 固定不可学 `adaptive_avg_pool2d`
[7,8]=56 tokens → 可学层间 softmax 标量混合（30 层→1）→ 逐点 MLP 瓶颈（C→1024→C，
无 token 间交互）→ z。a1/a3 tied K=V（z 同时当 K 和 V，逐层标量 scale 调制）；a2
低秩 decoder（rank 256）从同一 z 解耦。五个表达力天花板：①K 丢弃 ②固定池化 ③层轴
单混合 ④无 token 间 attention ⑤tied K=V。

**D0 差距归因（关键对照 = a0）**：a0（无压缩、全 KV 直传、同超参）spatial D0 探针
91.0 ≈ a1 的 94.4/93.2 —— **压缩本身在 D0 几乎零成本**。a1 全量 D0 macro 91.5
（object 96.0 / spatial 93.2 / goal 89.6 / 10 87.2）vs 官方 expert 97.4 的 ~6 点差
不在 codec 表达力，在「冻结 5B expert + 只训 1.05B（action expert + projector +
codec）+ 边缘接口要重新学会消费 KV」（raw-Wan 不微调直接走接口 = 59%）。
**预算对比**：官方 libero_uncond_2cam224_1e-4.yaml = 同 4 套件 / batch 16 / 10
epochs / lr 1e-4 / eval_every 200 挑 best；我们 = 同 epoch 数（24951 步恰好 10
epochs）/ lr 2.8e-4 / 只评终点 ckpt。我们的任务更难（新模块从零学 + delay≤20
随机化 + 压缩在环），diag 同协议 step8317 42.0% → step24951 61.0%（+19，均系
mot_pretrained_path 修复前破损基座，绝对值不可比、delta 有效）= 训练末期仍陡升，
**欠训练有实证**。

**结论与建议**：等 a2 全量（ETA 13:00）——a2≈a1 则 tied K=V 也非 D0 主要瓶颈，
架构天花板暂不绑定；最直接检验 = a1 续训 2× 预算（warm restart 或恒定小 lr）+
打开中途 eval 挑 best ckpt（对齐官方做法）；架构升级（传 K / per-group 层混合 /
bottleneck 后加 token 间轻量 self-attn / edge tokens 5→8）只在预算拉平仍卡 91-92
时再动。D40 大跌（70.1）另有原因：训练 max_delay_steps=20，D40 属外推，协议问题
非容量问题。

## 2026-10-15 09:50 a3「训练程序错误」定性：NFS stale handle 收尾打断，数据完好；脚本自拷贝加固

a3 主任务（21985458 / fb73e4f6）平台状态「训练程序错误」，实查：**全部工作已完成**
——D5/D10/D20/D40 各 40/40、delay_sweep_summary.json + delay_sweep_tasks.csv 落盘、
零 worker 死亡；train.log 最后一行 `error reading input file: Stale file handle` =
跑了 5h 后 bash 增量读 NFS 上的脚本文件遇 NFS 偶发 stale handle，死在**收尾阶段**
（数据写盘之后），平台按退出码标错误。纯展示问题。

**加固**（ac4a400）：脚本开场自拷贝（脚本 + env_2653853.sh）到 /tmp/leapbotce_sweep_local
再 `exec` 本地副本，`LEAPBOTCE_PROJECT_ROOT` env 转发保持指向真仓库——5h+ 长跑不再
持有 NFS 文件句柄。假 GPU 自测：re-exec → env 探测 → preflight 响亮失败，链路全通。
后续任务（a0 及以后的 fill-up）都走此路径。

**平台任务现状**：a1（21998396 / 2a64d68e）08:11 干净收官（全部结束 5×40/40）；
a2（21998406 / 4167f848）08:11 起 pod，全量协议 + PACK=1 ✓，D0 已完（67 min/2000 eps）
09:20 起 D5，ETA ~13:00。待提交：a3 D0 fill-up（同名重提，~1.5h）、a0 全量（~8.5h）。

## 2026-10-15 07:37 收割隔夜全量：a3 主任务完（D5-D40 零死亡），a1 重提 03:14 健康（D40 收尾），初步矩阵读数

**任务状态**：
- a3 主任务 fb73e4f6 **完成**：D5/D10/D20/D40 全 40/40（02:22 收官），legacy PACK=1
  全程零 worker 死亡，节奏 ~75 min/delay。缺 D0 的 object/goal/10（探针 .done 坑）→
  **同名重提一次填 D0（~1.5h，脚本加固后不会跳）**；
- a1 重提 2a64d68e（用户 ~03:14 提交）：PACK=1 ✓ 零死亡，D0-D20 已完（D0 91.5% /
  D5 87.2% / D10 84.3% / D20 81.2%），06:49 起 D40，ETA ~08:05；
- a2：**无 pod 目录** —— 排队中或未提交，待用户核实；
- a0：未提交（等用户上午提，变体已就位）。

**初步矩阵读数**（macro / 4 套件，2000 eps/delay，episodes 齐）：

| 线 | D0 | D5 | D10 | D20 | D40 |
|---|---|---|---|---|---|
| a1 codec+topk | 91.5 | 87.2 | 84.3 | 81.2 | **70.1**（08:1x 收官 40/40） |
| a3 codec+grid | — | 85.9 | 81.2 | 77.4 | 68.6 |
| a1−a3 | † | +1.3 | **+3.1** | **+3.7** | +1.5 |

† a3 D0 不可比（只有 spatial 92.6，缺 3 套件）。a1−a3 全可比 delay 为正但 D5/D40
（+1.3/+1.5）在 2000 eps 的 ±1.75 噪声带内，**优势最清晰在 D10-D20（+3~+4）**：
topk ≥ grid 成立，不是压倒性。a1 D0→D40 掉 21.4 点。

套件模式：libero_10 恒最弱（D10 a1 78.8 / a3 74.4），object 恒最强——与 raw-Wan
诊断期一致。**a1 > a3 全 delay 成立（D40 差 ~6 点）→ 挑选因子消融答案：topk 优于
grid**。a2（kv_decoder）与 a0（接口基线）全量待出。

## 2026-10-15 00:03 a0 门控通过：spatial D0 = 91.0%，四线全部过线

a0 本地探针（GPU 6，21:52 起跑）00:03 前 10/10 任务齐：**455/500 = 91.0%**
（±2.5 CI）。探针终值总表（spatial 整套 × D=0 × 500 eps）：

| 线 | 平台探针 | 本地探针 | 备注 |
|---|---|---|---|
| a0 无压缩接口 | — | **91.0%** | 接口税基线；训练在慢节点但正常跑满 24951 步 |
| a1 codec 主线 | 93.2% | 94.4% | ≥ a0：压缩不掉点 |
| a2 kv_decoder | — | **94.6%** | 四线最高 |
| a3 grid 选择 | 92.6% | 92.8% | |

a0 可提交 `leapbot_ce_eval_libero_a0_release`（全量协议 + PACK=1 已就位，H200×8 ~8.5h）。
隔夜平台任务巡检：a3（fb73e4f6, legacy PACK=1）健康——D5/D10 完成 40/40，23:40 起
D20，**零 worker 死亡**，节奏 ~75 min/delay 预计 ~02:30 全完；a1 旧任务 e6318fee 仍在
（PACK=4 旧配置用户未停未重提），遗留明早处置。

## 2026-10-14 21:55 二号坑：探针 .done 标记污染 delay 级跳过 → a3 全量任务漏跑 D0 三套件

**实锤**：a3 全量任务（fb73e4f6，21:17 pod 起跑——**晚于 20:30 的 PACK=1 提交，已带上
修复，跑的 legacy 串行模式，不用停**）日志 21:19:36 `跳过已完成: delay=0`，但盘面
delay_0 只有 spatial 10 份。根因：packed 尾部写 delay 级 `.done`（脚本 L475-482）按
**任务自己的 suite 范围**判断——15:43/15:45 的 spatial 探针任务把自己的范围写成了
全量语义，晚到的 legacy 全量任务整段跳过 D0。a1 的 delay_0/.done 同病（15:43 写入），
不删则 a1 重提同样漏 goal/10@D0。

**处置**：
- 脚本 legacy 跳过条件加固：`.done` **且** `count_results ≥ EXPECTED` 才跳过，否则
  打日志重跑补缺（manager 只补缺失 + 重写 .done）——对任何残留标记都稳，读侧防御；
- 手动删除 a1/a3 两个说谎的 `.done`（双保险）；
- **a3 处置 = 让它跑完 D5-D40（~明 05:00-06:00），明早重提同名任务一次填 D0 的
  object/goal/10 三套件（~1.5h，加固后脚本不会跳过）**；
- a1 处置 = 停掉 e6318fee 重提（3.65h 仅 1 格完成，错误遍地，无保留价值）。

**a0 收官**：训练 21:4x step=24950/24951 + step_024951.pt 落盘（21:48 确认），21:52
本地探针已发（GPU 6 与未知容器租户 8GB 进程共存，余 ~70GB 足够 2 workers），
`a0_release_step24951_probe_local`，~23:45 出门控读数。a0 变体本就是全量协议 + PACK=1，
过线即可提交。GPU 4-7 出现连号 1023895-1023909 进程（各 ~8GB，另一容器命名空间，
ps 不可见）——不碰，只共存放量。

## 2026-10-14 20:30 a1 重提任务撞节点内存墙：32 workers 遍地 -6/-9 → 全量协议钉 PACK=1

e6318fee（a1 重提，8×H200，PACK=4 → 全场 32 workers）18:10 起跑，**10 分钟后开始
worker 静默死亡**（rc=-6 SIGABRT 无 traceback，个别 -9 SIGKILL=OOM-kill 特征），
至 19:58 共 27 次、遍布全部 20 格；goal@D0 / 10@D0 双 worker 双死 → cell 报废
（重试 2 次盘面 0/10），19:58 时仅 object@D0 完成 10/10。worker 日志里 episode
全部 success、MP4 正常保存，进程中途无声死掉 = 典型 C++ bad_alloc abort。

**根因**：32 workers × ~22-30GB 主机 RSS（Wan5B + 文本编码器 + 缓冲）≈ 700-960GB，
当天节点主机内存撑不住。旁证三点：10-10 同布局（8 卡 PACK=4）零死亡跑完 4000 eps
= 那个节点内存更大；当天下午 16 workers 探针任务（151350/151401）无恙 → 当天节点
可用内存夹在 350-700GB 之间；首个死亡在满载后 10 分钟（非慢爬坡，是硬顶）。

**处置**（AI_FLOW_PACK 旋钮 + 全量协议钉 PACK=1）：
- 基座加 `AI_FLOW_PACK` env → `LIBERO_LIBERO_PACK` 透传（镜像其它 AI_FLOW_* 旋钮）；
- 四个 release 变体 `setdefault("AI_FLOW_PACK", "1")` = legacy 单 manager 全卡串行
  （8 workers ~200GB，任何节点都稳）；a2/a0 探针协议不受影响，翻全量后自动兜底；
- a1 任务 e6318fee 请用户停掉；a3 重提同样带 PACK=1。已跑成果不浪费：spatial@D0 +
  object@D0 各 500 eps 在 TAG 下，skip-existing / manager 只补缺失自动计入。

**PACK=1 全量时长**：D0 补 goal/10（~1h）+ D5-D40（2000 eps/delay，8 workers
~20-30s/ep → ~1.5-2h/delay）≈ **8-9h 过夜**，明早出 a1/a3 全量读数。a3 待同样
配置提交；a2 本地探针 17:05 起跑、18:41 收官 **473/500 = 94.6% 四线最高**（a1
94.4 / a3 92.8），门控通过 → 20:40 a2 变体已翻回全量协议（PACK=1 已带），可随
a1/a3 一并提交；a0 训练 ~21:20 落盘。
冒烟注意：同进程连续 import 多个变体会因基座模块缓存 + env setdefault 串味
（四个全显示第一个的 TAG）——平台每任务独立进程无此问题，冒烟须逐个独立进程跑。

## 2026-10-14 17:05 探针终值双源互证 + a2 探针提前链上；用户重提 a1/a3 全量

平台探针任务（07855134/f973b5fe，pod 在 16:10 翻回全量前启动 → 跑的是探针协议）与本地
探针（`*_probe_local`）双双收官，spatial 整套 × D=0 × 500 episodes：

- **a1：平台 466/500 = 93.2%，本地 472/500 = 94.4%**
- **a3：平台 463/500 = 92.6%，本地 464/500 = 92.8%**

两源差 ≤1.2 点，均远超 90% 门控线与 89.6% 复现必要条件，门控结论稳固。平台已跑的
D0-spatial 格子写在全量 TAG 下，重提全量任务经 skip-existing 直接计入（19 格 ≈ 9500 eps）。

a2 本地探针提前链上：GPU 6 于 17:05 起跑（原 19:07 定时链作废，本地探针提前收官空出卡），
`a2_release_step24951_probe_local`，预计 ~19:00 出门控读数；过线即翻全量协议通知用户提交。
a0 训练 17:03 时 step=21700/24951（~87%），预计 ~21:20 落盘 → 本地探针 → 过线翻全量。
用户于 ~17:05 重提 a1/a3 全量任务（卡数与首提一致）。

## 2026-10-14 15:52 探针初步读数：门控通过——a1 spatial D0 95.2% / a3 94.0%（raw-Wan 同位 59%）

本地探针（GPU 6=a1、GPU 7=a3，各 2 workers，15:06/15:08 起，`*_probe_local` TAG）前 5 任务
（250 episodes，CI ±2.6 点）：

- **a1（codec 主线）：238/250 = 95.2%** —— task0-2 全对 50/50，task3 47/50，task4 41/50
- **a3（grid 选择）：235/250 = 94.0%** —— 50/48/47/49/41

对比 raw-Wan a1 同位（libero_spatial D=0）= 59.0% → **+36 点**。init 根因假设闭环确认：
24 点 gap 的头号因素就是冻结专家停在原始 Wan。≥90% 门控线双双通过（该线为复现 macro
97.4 的必要条件），全量评测（4 套件 × D{0,5,10,20,40}）值得提交。整套 500 episodes 预计
~18:00 出终值；a2 探针待本地卡空出（19:07 定时链）或平台排队。

注意：平台排队的 a1/a3 任务在 pod 启动时读变体文件——翻回全量协议后无需重提，排队任务
自动变全量；探针已跑的 D0-spatial 格子经 skip-existing 计入全量，不浪费。

**16:10 用户拍板**：a1/a3 变体已翻回全量协议（4 套件 × D{0,5,10,20,40} × 50，docstring 留
探针读数存档）；a2/a0 维持探针协议，各自过线再翻。排队的平台任务无需重提——pod 启动时
才读变体文件；卡数提示：4 卡任务全量约 ~10h（过夜），8 卡约 ~5h。

## 2026-10-14 评测降级为快速门控探针（用户决策）：spatial 整套 × D=0，≥90% 再谈全量

用户判断 ablation 阶段不必先跑全量评测，先用两个已落盘 ckpt（a1/a3）快速验证「方案有没有问题」。
落地协议（比单任务略宽，理由见下）：**libero_spatial 整套（10 任务 × 50 trials = 500 episodes）
× D=0**，a1/a3 两变体已改（e61b8fe），任务名不变 `leapbot_ce_eval_libero_a{1,3}_release`。

- 为何整套而非单任务：50 trials 单任务 CI ±14 点分不清 85/95；整套 CI ±4 点，成本只多 ~25 分钟。
- 为何 spatial：raw-Wan 时 a1 spatial 仅 59（object 96 已饱和、goal 65），init 修复红利集中在 spatial。
- 90% 门线的依据：官方 macro 97.4 = 四套件均值 ⇒ 每套件 ≥ 89.6% 是必要条件，spatial D=0 ≥ 90%
  即「能复现 97.4」的必要条件判据。
- 读数分档：**≥90** → init 根因闭环，翻回全量协议（4 套件 × D{0,5,10,20,40}，skip-existing
  自动续跑探针已完成的格子）；**56-65**（raw-Wan 原地）→ init 不是唯一根因，省下 4×8 卡 × 5h
  预算先排查（下一梯嫌疑：冻结本身 / 接口）；**a3 ≪ a1** → grid 选择与好特征不兼容，砍 a3 线。
- 卡数：每任务 4×H200（16 workers），~30-40 分钟；两任务并行共 8 卡。a2/a0 落盘后同样先探针。

## 2026-10-14 release 矩阵收官进度：a1/a3 完成，4 条评测变体就绪（含分发器 `__C` 兜底修复）

**训练线状态（14:12 定时检查 + 14:30 复核）**：a1（bb2ed8e6）、a3（edb75c02）已完成 24951 步，
terminal ckpt `step_024951.pt` 分别于 13:23 / 13:17 落盘（`checkpoints/weights/`，另存 8317/16634，
keep_last=3）；a2（c76ca2d0）step≈24000，预计 ~15:15 落盘；a0（cea5a012）step≈19500，预计 ~21:00 ——
a0 落在慢节点（全程持续 ~0.21 step/s，兄弟线 0.29，逐分钟轨迹无中断无重启，非故障），只是晚到。

**评测变体（新增 4 个，`AIFlow_config_eval_libero_a{0,1,2,3}_release.py`）**：任务名
`leapbot_ce_eval_libero_a{0..3}_release`；协议 = 4 套件 × D{0,5,10,20,40} × 50 trials（与原
A0 dtrain20 扫描同款默认）；ckpt 钉死各线 `step_024951.pt`；model.* 全由 RUN_DIR 的 config.yaml
推导（sweep 白名单自动转发 `mot_pretrained_path`，评测基座不会拿原始 Wan 重建基座）。

**提交前修复两个坑（本地冒烟暴露）**：

1. **分发器 `__C` 兜底**：薄委托变体经 `from 基座 import *` 转发接口，star-import 不带
   下划线名 → 变体无 `__C` → 分发器尾部 `_mod.__C` 在平台 import 期抛 AttributeError，
   被折成「aiflowconfig.py有错误」秒挂。老的 a1/a2/a3 诊断评测都走本地调试卡、从没经过
   分发器，所以此坑从未暴露。修复 = 分发器 `getattr(_mod, "__C", None)` 落 `cfg` 兜底
   （基座 cfg 与 __C 同一对象）+ 4 个新变体各自显式 `from 基座 import __C`。老变体
   `eval_libero_a2` 冒烟从 AttributeError 变为正常导入，证明兜底生效。
2. **评测基座 ckpt 存在守卫**（`AIFlow_config_eval_libero.py` AutoTrain）：显式指定的
   `AI_FLOW_CKPT` 不存在时立刻报错返回，不拉起 pod 跑到评测期才炸 —— a0/a2 的评测须等
   terminal ckpt 落盘后再提交。

回归：4 新变体（对抗哈希 `a0a1a2a3` 下正确路由 + 参数正确）+ 昨日 12 用例路由全过；
分发器同步 `daojie_leap_configs/`。a1/a3 评测可立即提交，a2 ~15:30 后、a0 ~21:30 后。

## 2026-10-13 a2 重提任务被分发器劫持到 a0 配置（jobid 哈希子串撞关键字）

**现象（13:30 发现）**：a2 重提任务（21932579，目录 `leapbot_ce_libero_a2_release-26fa02d4`）
跑的根本不是 a2 —— train.log 的 overrides 是**a0 的完整配方**（`task=libero_all_leapbotce`、
`wandb.name=a0_release_seed42_h200`、无 `use_kv_decoder`），输出目录也指到
`runs/libero_all_leapbotce/a0_release_seed42_h200` = **真 a0 任务（cea5a012）的同一 run 目录**。
两任务都在 0.28 step/s 正常 stepping，run 目录里已经有两个 wandb offline run
（13:18 真 a0 + 13:27 冒名顶替者），step 8317 起还会互写同名 checkpoint。首提 a2（e20fd618，
4 卡事故死）的 run_id 是正确的 `codec_a2_release_seed42_h200` —— 劫持只发生在重提上。

**根因（分发器实锤）**：`AIFlow_config.py` 按任务目录名匹配变体，而 basename 尾部拼着
平台 jobid（`<任务名>-<jobid 8 位十六进制>`）；旧逻辑是**子串匹配**
（`tok in _TASK`）。`26fa02d4` 恰好含子串 `"a0"`（f-**a-0**-2d4），于是
`libero_a0_release` 混进候选、与 `libero_a2_release` 同为 17 字符并列，同长取字母序
→ a0 胜出，静默跑错配置。a0/a1/a3 三个任务的哈希不含别人家的 `aN` 子串，未受影响。

**修复（`AIFlow_config.py`，FastWAM 与 `daojie_leap_configs/` 两份同步）**：
① 剥掉尾部 `-[0-9a-f]{8}$` 哈希再匹配；② 后缀分词改为**任务名整词集合等值匹配**
（`_/-` 切词，子串一律不算）—— 哈希再怎么撞也只是一个整词。12 用例回归：4 条 release
线任务名（含劫持原案 `…-26fa02d4` 和对抗哈希 `…-0a0a0a0a`）全部正确路由，bench/eval/
robotwin 老任务名不变，无 env 落缺省不变。`eval_libero_a2` 一例 AttributeError 是该旧变体
自身缺 `__C` 的预存在问题（dispatch 选择正确，新旧逻辑同选），与本次改动无关。

**处置（14:55 收官）**：用户 14:38 停掉 21932579；14:39 同名重提（21935761，目录
`...a2_release-c76ca2d0`），新分发器路由正确（`task=libero_all_leapbotce_codec`、
`run_id=codec_a2_release_seed42_h200`、`use_kv_decoder=true`、release init、独立输出目录），
14:55 已 stepping（0.30 step/s）。a0 run 目录污染清理完毕：冒名 wandb run 删除、
`latest-run`/`debug*.log` 符号链接指回真 a0 的 `offline-run-...131812-qy1ua4hp`；
checkpoints 目录全程未被写过（首次落盘在 step 8317），config.yaml/dataset_stats.json
被冒名者重写过但内容与真 a0 配置一致，无需恢复。真 a0 本体（cea5a012）训练全程健康。

## 2026-10-13 头号根因定位：冻结专家一直是【原始 Wan】而非 FastWAM LIBERO release + `mot_pretrained_path` 开关

**用户疑点成立（代码实锤）**：官方 FastWAM 的 97.4% 来自视频专家在 LIBERO 上联训的
release ckpt（`/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt`，
`payload = {mot: 两专家, proprio_encoder, step, torch_dtype}`）；而我们 A0/a1-a3 的模型配置
`model_id: Wan-AI/Wan2.2-TI2V-5B` 直出原始 Wan + `ActionDiT_linear_interp` 初始化，
**冻结的视频专家从未见过 LIBERO 分布**。24 点 gap（97.4→73.2）的头号嫌疑从"冻结/解冻"
升级为"冻结在错误的起点"。冻结点本身在
`src/fastwam/models/wan22/leapbotce_edge_train_factory.py:20`（`model.dit = nn.ModuleDict(可训模块)`，
trainer 的 dit-only 模式因此只解冻 action_expert/projector/adapter/codec，5B 视频专家留在
`requires_grad_(False)`）；上游对照：官方 trainer `_apply_dit_only_train_mode` 训的是
`model.dit = self.mot` = 两专家一起 —— 官方 97.4 本来就是联训产物。

**修复（已落地，`mot_pretrained_path` 开关）**：

- `leapbotce_factory.py`：新增 `mot_pretrained_path` 参数，构建后调 `load_checkpoint` 走
  v1 full-MoT 分支（`payload["mot"]` 严格载入两专家 + proprio_encoder；edge 侧
  projector/codec 不碰，保持自身初始化）。
- `configs/model/leapbotce.yaml`：`mot_pretrained_path: null` 默认（null = 原行为）。
- `scripts/eval_leapbotce_libero_sweep_2653853.sh`：任务推导白名单**补上**
  `model.mot_pretrained_path` 转发 —— 不补的话 a0_release 评测会拿原始 Wan 重建冻结基座
  + release 训出的 delta，静默错配（delta ckpt 只存微调部分，见 `save_checkpoint`）。
- 验证：release payload 键结构与我们的 MoT 完全匹配（1649 键全在 `mixtures.{video,action}`，
  proprio 8→4096）；GPU 6 全模型构建冒烟 6 张量（video block 0/29、action、proprio w/b）
  与 release 精确相等，edge projector 原样。

**新训练变体（待提交）**：`leapbot_ce_libero_a0_release`（AIFlow_config_libero_a0_release.py）
= A0 底座逐项对齐（task=libero_all_leapbotce、d=20/seed42、eff128、24951 步、stale -1）
**唯一差异** `model.mot_pretrained_path=<release>`；4×H200 ~2.1 天。读数：
`a0_release − A0(73.2)` = 冻结专家域内预训练的净贡献；接近 97.4 则确认 init 是主因，
解冻（第二梯）降级为可选。RoboTwin 同理换 `robotwin_uncond_3cam_384.pt`（指南 §11）。

**同起点（原始 Wan）codec vs A0 修正读数（10-13 下午补）**：之前"codec 占 ~10-15 点"
是错位对比（a1 spatial 59 对 A0 四套件均值 73.2 —— object 套件 95-97% 拉高了均值）。
同套件 spatial 对比：**A0 56.0/57.0/52.5（n=400/格，D=0/10/20）vs a1 codec
59.0/61.0/63.0（n=100/格）vs a2 decoder 51/54/58 —— codec(tied) 与 A0 无统计差甚至
略优，a2 略差但 CI(±10) 重叠**。含义：105× 压缩在同起点下近乎零成本，24 点 gap
与压缩无关、全部来自 init/训练配方 → codec 作为 claim 安全；主 claim 线补
`leapbot_ce_libero_a1_release`（AIFlow_config_libero_a1_release.py = a1 消融线
+ release init，其余逐项同 a1），a0_release 角色改为对照（分解 压缩成本 vs
训练配方成本）。decoder 路线（a2）是否加码（如 rank 256→512、pool_grid 加 token）
等两个信号再定：① 标准规模 a1/a2 评测（判决 tied vs decoder），② a1_release
落点 —— 在原始 Wan 的 garbage-in 区调 codec 设计没有意义。

**a2 首提 4 卡分配事故与守卫（10-13 13:18）**：a2_release 首提按 8 卡提交但平台只分了
4 卡（任务元数据 gpu:4），配置按 nproc=8 拉起 → rank 4-7 摸 cuda:4-7 → data.train
初始化 `CUDA error: invalid device ordinal`，4 分钟即炸（train.log 定位：
`jinnTrainResult/leapbot_ce_libero_a2_release-e20fd618/`）。同批 a0/a1/a3 均为真 8 卡
节点、13:15 起 stepping 正常（0.23-0.28 step/s，eta 25-30h）。处置：用户 13:23 重提
8 卡版成功。**守卫**（防复发，已提交）：`train_leapbotce_libero_2653853.sh` 参数解析后
新增"实际可见卡数守卫"——`nvidia-smi -L` 数出的可见卡压过申请值，ACCUM 自动补齐保
有效 batch（8×bs16×accum1 → 4×bs16×accum2）；nvidia-smi 缺失时跳过。曾试过配置侧
"GPU_NUM 环境变量优先"方案后撤销——调试机 shell 自带垃圾 `GPU_NUM=1`，证明 env 不可
作为实分配信号，唯一可信信号是 launcher 在真环境里数卡。

## 一、现在该提交什么（更新于 2026-10-13 下午）

| 任务名（整段照抄） | 配置文件 | 卡数 | 预计耗时 | 状态 |
|---|---|---|---|---|
| `leapbot_ce_libero_a1_release` | AIFlow_config_libero_a1_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（**主 claim 线：codec(tied) + release**） |
| `leapbot_ce_libero_a0_release` | AIFlow_config_libero_a0_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（对照：无 codec + release） |
| `leapbot_ce_libero_a2_release` | AIFlow_config_libero_a2_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（decoder 因子 + release） |
| `leapbot_ce_libero_a3_release` | AIFlow_config_libero_a3_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（grid 挑选因子 + release） |

- **决策（10-13 下午，用户拍板）**：原始 Wan 线的标准评测全部取消——同起点下
  codec≈A0 已由 diag 数字确认，tied-vs-decoder 改在 release 域内判决。
  32 卡空闲 → 四条 release 线 4×8 卡并行，~1.1 天（A0 实测 25.7h）全部落地，
  一次拿全 {无压缩, codec, +decoder, +grid} × release init 的完整消融矩阵。
- 四线 eff batch 128 不变（8×bs16×accum1）；只有 4 卡时改回 GPU_NUM=4/ACCUM=2。
- 训练完评测：各线 terminal ckpt 用 AIFlow_config_eval_libero 路线换 RUN_DIR/TAG
  （a{0,1,2,3}_release_step24951），sweep 脚本会从 config.yaml 自动转发
  mot_pretrained_path 重建冻结基座。

## 二、提交前核对（30 秒）

1. **任务名别改字**：分发器按名字关键字最长匹配。名字只有 `libero` 没有变体关键字
   （a1_codec / a2_decoder / a3_grid）的会落回基础配置 = 200k 续训线（带 `--resume`），
   误提交 = 重复续训。
2. **节点选 H200**：H100-80G 跑不了 bs16（bs8 实测峰值 64.2GB）。只有 H100 时先说，
   我出 bs8/accum4 等价版。
3. 想快：申请 8 卡 + 该配置里 `AI_FLOW_ACCUM = 1` → ~1.1 天（eff batch 128 不变，
   动力学与 4 卡版一致；`AI_FLOW_GPU_NUM` 同步改 8）。
4. 平台 pod 镜像 torch 是残包：训练配置已全部显式钉死 bs/lr/steps/卡数，别放开自动表。

## 三、消融矩阵与结果回收（LIBERO only）

| 项 | 是什么 | 结果回收 |
|---|---|---|
| A0 | frozen-cloud 基线（h200 定标那版，D=0 65-73） | 已有（tag dtrain20_seed42_h200） |
| A1 | codec 主配方（云 56 token 无解码器） | run `codec_a1_seed42_h200` |
| A2 | A1 + 边缘侧 K/V 低秩解码器 | run `codec_a2_decoder_seed42_h200` |
| A3 | A1 + 边缘挑选改 grid（均匀间隔） | run `codec_a3_grid_seed42_h200` |
| A4 | 边缘贡献（**不训练**，A1 ckpt 评测时关掉边缘） | TAG `codec_a1_seed42_h200_noedge` |

评测 codec 线：`AIFlow_config_eval_libero.py` 常改区 `AI_FLOW_RUN_DIR` →
`.../runs/libero_all_leapbotce_codec/<run_name>`，`AI_FLOW_TAG` → 对应 run 名；
A4 额外 `AI_FLOW_EVAL_EXTRA = "model.cloudedge.edge_prefix_enabled=false"`。

- 消融定夺规则：A2 若明显优于 A1 → RoboTwin 最终方案带解码器；A3 若优于 A1 → 挑选换 grid。
- RoboTwin 正式跑时 pool_grid 须 override `[6,10]`（token 网格 12×10），配置文件注释已写。

## 四、记录（新在上）

### 2026-10-13 终点 ckpt 诊断（a1/a2）+ 预算勘误 + 标准评测变体
- 训练收尾确认：a1/a2 正常目录 0 traceback，最后动作 = 写 step_024951
  weights+state（a2 10-13 02:00、a1 06:59）；a3 ETA ~16:00。
- 终点诊断（GPU 7 双 worker，libero_spatial × D{0,10,20} × 10 trials）：

  | 线 | D=0 | D=10 | D=20 | ret@20 |
  |---|---|---|---|---|
  | a1 tied | 59.0 | 61.0 | 63.0 | 1.068 |
  | a2 解码器 | 51.0 | 54.0 | 58.0 | 1.137 |

- **反转**：33% ckpt 时 a2 领先 2-3 点，终点上 a1 点估计反超 5-8 点
  （CI 重叠，n=100 ±10）。解码器在此预算下未兑现（嫌疑：随机初始化
  瓶颈欠收敛 vs 纯噪声）；判决 = 标准规模评测（500 eps/格，CI ±4.4），
  主线选择（tied vs decoder）等该数字。两条 retention 都 >1（平/升）。
- **预算勘误（重要，防止再犯）**：eff batch 128 不是科学常数 = √LR 折算
  自官方 eff16 配方（`LEAPBOTCE.md:88,101,181,208`）。P0 的"200k 步"
  是 @eff16 = **3.2M 样本**；消融线 24951 步 @eff128 = 3.19M 样本 =
  **同一预算**。即 codec 三线已是 P0 全预算，"正式线"无需再加跑
  200k@eff128（那是 8× 预算）。若学习曲线未饱和要加预算：2× = 6.4M 样本
  ≈ 8 卡 eff128 49902 步 ~2.2 天，或 16 卡 eff256（lr 4e-4）~1.1 天
  （dynamics 未验证）。
- 标准规模消融评测改走分发器变体：新增 `AIFlow_config_eval_libero_a{1,2,3}.py`
  （薄委托钉死 RUN_DIR/CKPT/TAG/SUITES/DELAYS/TRIALS；任务名
  `leapbot_ce_eval_libero_a{1,2,3}`，4×H200，可并行提交零串味；
  本地 import 冒烟全过）。a1/a2 即提；a3 等 ~16:00 ckpt 落盘后提。
  今晚标准数字出 → 定主线 → 胜者跑四套件 × 5 窗全量 sweep（填 Table 2）。
- RoboTwin codec 前置不变：pool_grid=[6,10] 适配 + 单格冒烟，今天备明天提。

### 2026-10-12 三线 step_008317（33%）调试机诊断评测 + summarize 脚本修复
- 用户要求对 a1/a2/a3 的 step_008317.pt（33% 进度）做初步测评。调试机 GPU 7
  （4/5/6 被 RoboTwin fork 占用；7 有 7.8GB 常驻可共存）。配置：libero_spatial、
  D∈{0,10,20}、10 trials（=100 eps/格）、PACK=1、单格实测 ~35-38 min。
- 结果（success rate / retention vs D=0，Wilson95 ±10 点）：

  | 线 | D=0 | D=10 | D=20 | ret@20 |
  |---|---|---|---|---|
  | a1 tied | 44.0 | 45.0 | 37.0 | 0.841 |
  | a2 解码器主线 | 46.0 | 48.0 | 42.0 | 0.913 |
  | a3 grid | 50.0 | 46.0 | 54.0 | 1.080 |

- 读法：三线 retention 都平（84–108%），codec 接口的延迟鲁棒性在诊断规模立住；
  a2 全窗口 ≥ a1 且差距随 D 单调拉大（+2/+3/+5），方向符合解码器机制预期；
  a3 nominal 最高但三线 CI 大量重叠，33% 训练 + 100 eps/格下消融不可分，
  消融表数字等 24951 终点 ckpt + 标准规模。
- **教训**：`LIBERO_EVAL_DELAYS` 必须空格分隔（脚本 `read -ra` 按 IFS 切分）；
  逗号版会先创建 `delay_0,10,20` 残留目录再炸 hydra，且该残留会让
  `summarize_leapbotce_delays.py` 在评测全部完成后 ValueError 挡汇总。
  已修脚本（过滤非整数 delay 目录，commit `95680ee`）+ 删 a2 根目录残留。
- 结果：`/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/
  {a1_codec,a2_decoder,a3_grid}_step8317_diag/delay_sweep_summary.json`。
  TAG 绑 ckpt：后续 step_16634 / 24951 复评须换新 TAG。

### 2026-10-11（晚 2）主线翻转为"开 K/V 解码器版"（LeapBotCEPaper f1987c4）
- 用户拍板：**最终版本讲开解码器的（A2 recipe = a2_decoder run）**，tied K=V
  （A1 = a1_codec run）降为消融行，专门量化解码器在零传输成本下的贡献。
  在训三线不用动：a2_decoder = full 主线，a1_codec = tied 消融，a3_grid = grid 消融。
- 论述修正两处代码事实（读 `cloud_kv_adapter.py` 核实）：
  ① 解码器是**纯线性**低秩瓶颈 `dec_up(dec_down(z))`，无 GELU（旧 KV adapter
  才带 GELU + 零初始化；新解码器默认初始化、非零初始化）；② 逐层 scale
  初始化为 1。eq:edgekv 已按线性形式重写。
- 接口参数账 9M → **12M**（codec 6.3M 云侧 + 解码器 3.1M 边缘 + 前缀投影
  2.4M）；摘要/引言/图注/实验/补充材料/结论全部同步，真机节边缘阶段行
  加了 K/V decode。main.pdf 11 页 / suppl 4 页，0 未定义引用。

### 2026-10-11（晚）论文重构至 codec 主线 + 真机时延节（LeapBotCEPaper 9f4a75a）
- **题目**改为 "LeapBot-CE: Planning-Token Codec for Delay-Robust Cloud-Edge
  Manipulation with Frozen World Models"。主线 = CloudPlanningCodec（z 载荷
  56×3072 ≈ **0.34MB bf16**，对 36MB 全 cache ~105×；`~52×` 是 LEAPBOTCE.md 里
  fp32 z 对 bf16 cache 的混精度口径，论文里统一按同精度算）+ edge select 前缀
  （2 views × 5 token = 10 token，EdgePrefixProjector 768→3072 带逐层 scale）。
- **方法章重写**：新增 "Cloud Planning Codec" 小节（eq:codec 空间池化→深度
  softmax 混合→瓶颈 MLP；eq:edgekv 边缘广播 + 逐层 scale，可选 rank-256 低秩
  K/V 解码器）；原 "Optional Cloud KV Adapter" 小节删除并入 codec（eq:adapter
  已删，全仓无悬空引用）；Edge Grounding 小节从 pooled 单 4096-d token 改为
  select 前缀（eq:edgeseq 序列 = [plan 56 | edge 10 | action 32]）。
- **实验章**：110k sweep 数字显式标注为 uncompressed-interface（A0）配置；
  消融表重构为两块（接口块 A0→A4 + 训练块）；带宽段重写（codec 载荷数学 +
  对比 CEVLA 1.8MB）；**新增 Sec 4.6 真机部署节**：Franka + 云 4090D-48GB +
  边缘 RTX 5080（暂定），加 RTT {0,400,1000}ms 沿 CEVLA 协议；表 5（成功率）
  表 6（时延分解：云 VAE+prefill / codec / 上行 0.34MB / 边缘 SigLIP+prefix /
  10 步去噪，mean/p50/p95）全占位 + 预期论述（0.34MB ≈ 2.7ms@1Gbps /
  28ms@100Mbps vs 36MB 的 288ms/2.9s → codec 是 WLAN 级链路可部署前提；云侧
  预期 10^2 ms 量级被 10 步 chunk 摊销；边缘 10 ms 量级永不等云）。
- 摘要/引言/相关工作/结论同步改口径；补充材料 codec 节替换原 adapter 节，
  两处跨文档 \cref 悬空引用改为字面节号（独立编译本来就解析不到主文 label）。
- 编译验证：main.pdf 11 页、X_suppl.pdf 4 页、0 未定义引用（8 页限制是终稿
  事，todo 红字清除后再压）。PAPER_TODO.md 已同步（表 3 新计划、表 5/6 数据
  准备指引、fig2 草图标过时）。

### 2026-10-11（下午）codec 消融评测链路调通（本地卡 4,5 冒烟 a1/a2/a3 全过）
- 背景：三条消融训练在跑（~0.2 step/s，首个 ckpt save_every=8317 预计当晚
  落盘），趁等待把评测代码和 AIFlow 启动脚本先调通。三个 run 都还没有
  ckpt → 用**合成 ckpt** 冒烟：200k 真实 action_expert/proprio_encoder +
  随机初始化 codec/select 投影（load_checkpoint 守卫允许，冒烟只验链路
  不验质量，成功率 0% 是预期正确）。
- **sweep 脚本改动**（`42a04c9`）：原写死 `task=libero_all_leapbotce`
  （frozen-cloud），codec ckpt 会在加载期 ValueError。现从
  RUN_DIR/config.yaml **自动推导** task（kv_codec.enabled → codec task）
  并把 kv_codec/edge_vision 字段钉成 eval 覆盖（pool_grid/use_kv_decoder/
  selection/mode/tokens_per_view/num_views + edge_prefix_enabled=false 时
  透传）；LIBERO_EVAL_TASK 可强制覆盖。AIFlow eval 配置加
  AI_FLOW_EVAL_TASK 透传 + 三条消融线 RUN_DIR 注释（只需换 RUN_DIR +
  新 TAG，task 不用手设）。
- **冒烟结果**（GPU 4,5 空闲卡，D=0 × libero_spatial × 2 trials，
  PACK=1 legacy 模式，独立 TAG smoke_codec_*_1011）：
  a1 codec 10/10 ✓ / a2 解码器 10/10 ✓ / a3 grid 10/10 ✓（各 ~7-13 min，
  manager config 证实 select/grid/use_kv_decoder 覆盖全部生效；
  replans=800/20ep 符合 chunk=10 节奏 schema）。fake run dir 留在
  `runs/libero_all_leapbotce_codec/_smoke_codec_a{1,2,3}/`（各 2GB）可复用。
- **正式评测提交口令**（真实 ckpt 落盘后）：AIFlow_config_eval_libero.py
  常改区 RUN_DIR→对应 codec run、TAG→新名（如 codec_a1_seed42_h200），
  其余不动。

### 2026-10-11（14:38）a3 重提后秒挂（节点抽奖）：.envs 死 shebang 已全量修复
- **现象**：a3_grid 重提后落在节点 10.194.160.30，2 分钟内死：
  `train_zero1.sh: line 111: .envs/leapbot-va/bin/accelerate: cannot execute:
  required file not found`。a1/a2 落在的节点正常（已在步进，ETA ~31-38h，
  **不用动它们**）。
- **根因**：envs 从 miniconda3 迁到 code/.envs 时**没有重写 console 脚本的
  shebang**——accelerate 等 1245 个脚本仍指向
  `#!/home/myuser/miniconda3/envs/<env>/bin/python3.10`，而该路径已随迁移
  清除。有 pod 本地残留旧 miniconda3 的节点能侥幸解析 → 节点抽奖；
  a3 抽到的节点没有 → 秒挂。18 个 env 全有雷（leapbot-va 内 85/144 脚本）。
- **修复**：`.envs/*/bin/` 全量重写 shebang →
  `#!/dahuafs/userdata/2653853/code/.envs/<env>/bin/python`（绝对共享盘路径，
  与节点无关）；18 个 env 清零，`accelerate` 本机实测可执行。对已运行任务
  无影响（进程已起，解释器在 exec 时已解析）。
- **下一步**：**只重提 a3_grid**，a1/a2 继续跑。

### 2026-10-11（下午）消融三线首提全挂 → edge 前缀 K/V 维度 bug 当日修复
- **现象**：a1_codec / a2_decoder / a3_grid 三个任务全部"训练程序错误"，
  各自启动后 1-2 分钟死于第一个训练 step。
- **排查路径**：平台日志在 `FastWAM/jinnTrainResult/<任务名>-<hash>/train.log`
  （**用户指定：以后排错优先看这里**）；三份 log 同一错误
  `RuntimeError: Expected size 3072 but got size 4096` @
  `mot.py:563 forward_action_with_video_cache_tensor` 的
  `torch.cat([video_cache_k, edge_prefix_k, k_action], dim=1)`。
  （run 目录内 wandb 佐证：a1/a2 .wandb 仅 7 字节=硬杀未刷盘；a3 优雅退出
  留下完整 traceback，与本诊断一致。）
- **根因**：`EdgeVisionEncoder` select 模式的 `edge_dim` 静默回退
  `text_dim`（umt5-xxl = 4096），而边缘前缀 K/V 要与 video cache K/V
  （`num_heads*head_dim` = 3072，kv_adapter/kv_codec 都正确用了这个）在
  注意力里拼接；TI2V-5B 两者不等。**pooled 模式（A0/frozen-cloud）输出
  的是 context token，本来就活在 text_dim，所以老配方从未触发**。
  三个消融全走 select 模式 → 全挂，A3 的 grid/topk 差异无关。
- **修复**（`0abdfde`）：`LeapBotCE.__init__` 构造 EdgeVisionEncoder 时
  显式传 `edge_dim = num_heads * head_dim`；select 模式缺 `edge_dim` 改为
  构造期 ValueError（不再静默回退）；补回归测试。26/26 过；另用真 SigLIP
  在 CPU 验证形状链：修后前缀 [B,10,3072] 与 cache 对齐，旧默认正是
  [B,10,4096]。
- **下一步**：三线**原配置重提即可**（代码修复对 pod 生效取决于仓库挂载
  /dahuafs 路径，无需改配置）；a2/a3 的 EXTRA_OVERRIDES 核对无误。

### 2026-10-11（傍晚）RoboTwin 我们泳道停测决策落实（用户确认）
- 用户决策：**LeapBot-ce RoboTwin 评测泳道停止，不再投入**（"意义不大"）。
- 核查：该 sweep 其实**已自行跑完**——50/50 格 `.done` 齐全，最后一格
  delay_40/stack_blocks_two 今日 12:44 收尾（lane 日志 `_cluster_logs/`）；
  当前无进程、无 tmux/screen，无需杀任何东西，也不烧卡了。若曾有经
  AIFlow 平台的提交，平台侧任务也已自然结束。
- 处置：**数据保留不删**（`leapbotce/robotwin_sweep/`）；生成器 FOOTNOTES
  加「已停止」披露 bullet，报告已重生成；行保留备查，**不入结论/论文**。
- 判定依据（D=0 逐任务实测 `_result_clean.txt`）：总 7.4% = shake_bottle
  一项 74% 贡献全部 37 次成功，**其余 9 任务全 0.0**（同格 Fast-WAM 99.6）；
  42.1 replans/ep 剂量正常 → 失败是模型侧（RoboTwin CE 配方/ckpt 未就位，
  与 LEAPBOTCE.md「RoboTwin CE 未实现」一致），非协议/审计问题。
- 连带影响：RoboTwin 最终方案（A2 解码器 / A3 grid 的取舍原依赖消融回收）
  的落地目标需重新定夺——LIBERO 消融三线照常等回收，但结果先只用于
  LIBERO 结论；RoboTwin 侧除非配方就位并重训，否则不重开泳道。
- **后续计划（用户 2026-10-11 确认）**：中间过程不再折腾（不为 RoboTwin
  单独排查/重训），等 LIBERO 消融（A1/A2/A3）回收、最终配方定夺后，
  **直接用最终最优方案重开 RoboTwin 泳道做正式测评一次**。前提注意：
  若最终方案是 CE codec 系（A1/A2/A3），RoboTwin CE 推理路径必须先真正
  跑通（pool_grid [6,10] / token 网格 12×10；本次 D=0=7.4% 已证实当前
  路径不可用）；若最终方案是 A0 frozen-cloud，评测路径与 Fast-WAM 泳道
  同构，成本最低。

### 2026-10-11（午后）两份 baseline 报告 + 论文 Table 1 加「replans/ep (D=40)」列（延迟剂量）
- 用户要求：本地总结表和论文表各加一列各模型预估 replan 次数并纳入分析。已落地
  （列口径 = audit 实测的每回合延迟抽样次数，即策略消费一次延迟观测并提交动作记一次）。
- **LIBERO 值**（D=40）：FastWAM 39.9 / StarWAM-MoT 39.9 / SharedDiT 39.8
  （10 步一抽、400 步上限、几乎全失败 → 跑满）；OpenWAM **581**（`policy.act`
  **每控制步**抽一次 —— 协议披露修正：它不是 10 步一 replan）；LingBot 46.3
  （原生 chunk ~16 步）；LeapBotCE 25.9/26.1（同节奏，成功率高 → 成功回合提前
  收尾；**剂量差异是成功率的结果不是原因，解读别反**）。OpenWAM 剂量最高却最
  鲁棒 → 快环架构补偿，剂量是必要披露维度但非唯一变量。
- **RoboTwin 值**：Fast-WAM 16.5（5.4-30.9/任务）vs AHA-WAM 450.7（147-795）
  = 27×；X-WAM 12.9（2 任务）；我们泳道 42.1。
- 生成器改动（两份都改，直接改 md 会被下次刷新覆盖）：LIBERO 加
  `draws_per_ep`（replans → 直方图 → OpenWAM videos delay_samples 三级回退，
  按 max_delay_steps 过滤）+ 机制表「抽样节奏」列 + 「延迟剂量披露」段；
  修 `parse_leapbotce_sweep` suite/全量 replans 口径 bug（虚高 4×，127→26）；
  RoboTwin 加三类 audit 解析（flat/samples/aha）。归因小节已并入生成器
  FOOTNOTES（重生成不再丢）。
- 论文 `sec/4_experiments.tex`：tab:audit 加 draws/ep 列 + caption 定义；
  Mechanism 段加剂量分析（chunk 系统同剂量 → 差异在模型侧；OpenWAM 581 次/ep
  最鲁棒 → 快环效应）；Confounds 段加跨基准节奏披露（RoboTwin Fast-WAM 24 步
  replan = LIBERO 泳道 2.4× 低暴露；AHA 逐步）。pdflatex 编译通过。
- ⚠️ 顺带发现：重生成后**我们自己的 RoboTwin 泳道行浮出**（之前 10-05 生成时无
  数据），D=0 仅 7.4%（shake_bottle 74，其余任务全 0），42 replans/ep ——
  该泳道质量有问题，RoboTwin 正式方案前需排查（疑似 ckpt/配方未就位，与
  LEAPBOTCE.md「RoboTwin CE 未实现」一致）。

### 2026-10-11（下午）RoboTwin Fast-WAM 曲线反常性归因完成（延迟审计复核）
- 用户问：Fast-WAM RoboTwin 只掉 18 点，LIBERO 同款官方 ckpt 掉 95 点，是否反常。
  **结论：两组数字都真，差异来自延迟暴露剂量不可比**（详见
  `baseline_results_robotwin.md` 新增归因小节）。
- 核心证据链：① RoboTwin 官方默认 replan=24 步 vs LIBERO 泳道 replan=10，
  实测 stale 规划 16.5 vs ~40 次/ep，暴露剂量差 2.4×；② 同环境 AHA 几乎逐步
  replan（147-795 次/ep）同 mean≈19 帧延迟掉 -49 → 环境本身延迟敏感，
  Fast-WAM 的 -18 是暴露被 chunk 压低的结果；③ per-task replans/ep 与掉点
  排序两方法都几乎一致；④ 3/10 准静态任务（click/shake/grab）任意暴露都
  不掉，托住均值，LIBERO 无此类任务。
- 待用户定夺：是否补 RoboTwin Fast-WAM `replan_steps=10` 对照格（D=40 一格
  ~500 eps）直接量化协议敏感性；论文表述按归因小节口径写。

### 2026-10-11（中午）codec 消融三线配置就绪；200k 线今天收尾
- **在跑**：200k 续训 11:52 实测 step 198,150/200,000（eta ~23min，speed 1.33 step/s，
  lr 已退火到 1.0e-06），预计 ~12:15 落盘 step_200000。10-10 17:07 那次提交
  （torch 残包秒挂）之后 17:22 重提成功，跑了 ~18.5h。
- **已备好**：消融三线 AIFlow 变体配置（a1/a2/a3，各 4×H200 ≈2.1 天，eff128 与 A0 逐项
  对齐）；已验证 dispatcher 分发 / 命令生成 / hydra compose / launcher dry-run。
- **耗时校准**：A0 h200 基线 8×H200 全程 25.7h（≈3.7 s/step @eff128，wandb 时间戳）。
- 评测脚本加 `LIBERO_EVAL_EXTRA` 透传（A4 用）；eval 配置加 `RUN_DIR`/`EVAL_EXTRA` 透传。
- codec 实现（CloudPlanningCodec + 边缘 select 前缀）今日已落地：25/25 测试 +
  真 SigLIP 冒烟（timm SigLIP 无 CLS → mean-patch 参考，NaN 隐患已修）。

### 2026-10-10（晚）110k 评测全量成功；200k 续训带病上线后恢复
- 110k tag 全量评测（d0/5/10/20/40 × 4 套件 × 50 trials，seed 1234）success（19:49）。
- 17:07 提交的 200k 续训秒挂（pod torch 残包，`torch.cuda` AttributeError）；
  配置加防御后 17:22 重提成功。
- 同日平台首提 8×H200 训练也死于代理不通弄死 rank0（历史教训，已双保险：
  启动脚本自动降 offline + trainer 降级，wandb 曲线事后本地 sync）。
