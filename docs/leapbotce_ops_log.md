# LeapBotCE 实验监测与 AIFlow 提交记录

> 用途：跟进实验状态 + 提交 AIFlow 申请的速查。新条目在最上。
> 技术论证见 `leapbotce_dualstream_gap_analysis.md`，配方细节见根目录 `LEAPBOTCE.md`。

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
