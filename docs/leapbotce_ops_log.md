# LeapBotCE 实验监测与 AIFlow 提交记录

> 用途：跟进实验状态 + 提交 AIFlow 申请的速查。新条目在最上。
> 技术论证见 `leapbotce_dualstream_gap_analysis.md`，配方细节见根目录 `LEAPBOTCE.md`。

## 一、现在该提交什么（更新于 2026-10-11 中午）

| 任务名（整段照抄） | 配置文件 | 卡数 | 预计耗时 | 状态 |
|---|---|---|---|---|
| `leapbot_ce_libero_a1_codec` | AIFlow_config_libero_a1_codec.py | 4×H200 | ~2.1 天 | ⬜ 待提交 |
| `leapbot_ce_libero_a2_decoder` | AIFlow_config_libero_a2_decoder.py | 4×H200 | ~2.1 天 | ⬜ 待提交 |
| `leapbot_ce_libero_a3_grid` | AIFlow_config_libero_a3_grid.py | 4×H200 | ~2.1 天 | ⬜ 待提交 |
| 200k 评测（提交名带 eval+libero） | AIFlow_config_eval_libero.py | 4×H200 | ~2h | ⬜ 待提交（见下） |
| `leapbot_ce_train_libero`（200k 续训） | AIFlow_config_libero.py | 4 | — | 🔄 今天 ~12:15 收尾 |

- 三条消融线并行 = 12 卡；空闲卡常态 ~32，够。
- **200k 评测**：训练收尾后提交。改 `AIFlow_config_eval_libero.py` 常改区两行：
  `AI_FLOW_RUN_DIR` → `.../runs/libero_all_leapbotce/dtrain20_seed42_200k_resume`
  `AI_FLOW_TAG` → `dtrain20_seed42_200k`（**勿复用 110k tag**，新旧 ckpt 结果会混）。
- 评测 GPU 数：现有配置默认平台分配；评测 4 条线都走同一配置文件，**改常改区再提交**。

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
