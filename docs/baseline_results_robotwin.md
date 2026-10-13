# RoboTwin 2.0 帧延迟基线 (CloudEdgeVLA 协议)

*更新: 2026-10-13 12:53 · 与 LIBERO 报告独立: 见 `baseline_results_libero.md`*

## 主表 (10 任务 × 50 trials, 成功率 %)

| 方法 | D=0 | D=5 | D=10 | D=20 | D=40 | replans/ep (D=40) |
|---|---|---|---|---|---|---|
| LeapBot-ce (ours, dual-stream) | **7.4** | **11.4** | **8.8** | **11.6** | **12.2** | 42.3 |
| Motus (stage-2 ckpt) | 0.0 (n=50) | 0.0 (n=50) | — | 0.0 (n=50) | — | 50.0 |
| X-WAM (robotwin_sft) | — | — | — | — | 2.0 (n=100) | 12.9 |
| Fast-WAM (official release) | **99.6** | **96.8** | **96.6** | **94.6** | **81.6** | 16.5 |
| DreamZero (AgiBot) | — | — | 0.0 (n=50) | — | — | — |
| AHA-WAM (robotwin) | **97.6** | **92.8** | **84.2** | **67.8** | **48.6** | 450.7 |
| StarWAM (mot, official) | **95.4** | **93.8** | **93.2** | **91.2** | 78.7 (n=450) | 16.6 |
| Kairos (4B, official ckpt) | — | — | — | — | — | — |
| LingBot-VA (robotwin posttrain) | — | — | — | — | — | — |

`replans/ep (D=40)` = 各泳道 audit jsonl 实测的每回合延迟抽样次数（均值；不同方法抽样节奏不同——Fast-WAM 24 步一 replan、Motus chunk=8、X-WAM 全块执行、AHA-WAM 逐步 replan——该列把节奏差异变成显式数字，见下方披露）。

### 每任务明细

**LeapBot-ce (ours, dual-stream)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| beat_block_hammer | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| click_alarmclock | **0.0** | **4.0** | **0.0** | **4.0** | **2.0** |
| dump_bin_bigbin | **0.0** | **0.0** | **0.0** | **22.0** | **28.0** |
| grab_roller | **0.0** | **0.0** | **0.0** | **0.0** | **4.0** |
| handover_block | **0.0** | **34.0** | **0.0** | **0.0** | **0.0** |
| put_object_cabinet | **0.0** | **0.0** | **0.0** | **10.0** | **10.0** |
| scan_object | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| shake_bottle | **74.0** | **76.0** | **86.0** | **80.0** | **78.0** |
| stack_blocks_two | **0.0** | **0.0** | **2.0** | **0.0** | **0.0** |

**Motus (stage-2 ckpt)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | **0.0** | **0.0** | — | **0.0** | — |
| beat_block_hammer | — | — | — | — | — |
| click_alarmclock | — | — | — | — | — |
| dump_bin_bigbin | — | — | — | — | — |
| grab_roller | — | — | — | — | — |
| handover_block | — | — | — | — | — |
| put_object_cabinet | — | — | — | — | — |
| scan_object | — | — | — | — | — |
| shake_bottle | — | — | — | — | — |
| stack_blocks_two | — | — | — | — | — |

**X-WAM (robotwin_sft)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | — | — | — | — | **4.0** |
| beat_block_hammer | — | — | — | — | **0.0** |
| click_alarmclock | — | — | — | — | — |
| dump_bin_bigbin | — | — | — | — | — |
| grab_roller | — | — | — | — | — |
| handover_block | — | — | — | — | — |
| put_object_cabinet | — | — | — | — | — |
| scan_object | — | — | — | — | — |
| shake_bottle | — | — | — | — | — |
| stack_blocks_two | — | — | — | — | — |

**Fast-WAM (official release)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | **100.0** | **100.0** | **100.0** | **100.0** | **90.0** |
| beat_block_hammer | **100.0** | **94.0** | **100.0** | **96.0** | **80.0** |
| click_alarmclock | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| dump_bin_bigbin | **100.0** | **98.0** | **98.0** | **92.0** | **64.0** |
| grab_roller | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| handover_block | **96.0** | **100.0** | **94.0** | **86.0** | **44.0** |
| put_object_cabinet | **100.0** | **84.0** | **82.0** | **90.0** | **74.0** |
| scan_object | **100.0** | **92.0** | **92.0** | **86.0** | **88.0** |
| shake_bottle | **100.0** | **100.0** | **100.0** | **96.0** | **96.0** |
| stack_blocks_two | **100.0** | **100.0** | **100.0** | **100.0** | **80.0** |

**DreamZero (AgiBot)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | — | — | — | — | — |
| beat_block_hammer | — | — | **0.0** | — | — |
| click_alarmclock | — | — | — | — | — |
| dump_bin_bigbin | — | — | — | — | — |
| grab_roller | — | — | — | — | — |
| handover_block | — | — | — | — | — |
| put_object_cabinet | — | — | — | — | — |
| scan_object | — | — | — | — | — |
| shake_bottle | — | — | — | — | — |
| stack_blocks_two | — | — | — | — | — |

**AHA-WAM (robotwin)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | **100.0** | **100.0** | **98.0** | **90.0** | **60.0** |
| beat_block_hammer | **100.0** | **84.0** | **46.0** | **20.0** | **10.0** |
| click_alarmclock | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| dump_bin_bigbin | **96.0** | **88.0** | **84.0** | **54.0** | **30.0** |
| grab_roller | **100.0** | **100.0** | **100.0** | **100.0** | **94.0** |
| handover_block | **92.0** | **62.0** | **52.0** | **36.0** | **4.0** |
| put_object_cabinet | **96.0** | **94.0** | **88.0** | **56.0** | **28.0** |
| scan_object | **92.0** | **100.0** | **80.0** | **60.0** | **52.0** |
| shake_bottle | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| stack_blocks_two | **100.0** | **100.0** | **94.0** | **62.0** | **8.0** |

**StarWAM (mot, official)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | **100.0** | **100.0** | **100.0** | **96.0** | **90.0** |
| beat_block_hammer | **96.0** | **96.0** | **96.0** | **88.0** | **66.0** |
| click_alarmclock | **100.0** | **98.0** | **100.0** | **100.0** | **100.0** |
| dump_bin_bigbin | **94.0** | **98.0** | **90.0** | **88.0** | — |
| grab_roller | **100.0** | **100.0** | **100.0** | **100.0** | **98.0** |
| handover_block | **80.0** | **68.0** | **70.0** | **64.0** | **44.0** |
| put_object_cabinet | **90.0** | **92.0** | **90.0** | **92.0** | **78.0** |
| scan_object | **94.0** | **86.0** | **86.0** | **86.0** | **60.0** |
| shake_bottle | **100.0** | **100.0** | **100.0** | **98.0** | **94.0** |
| stack_blocks_two | **100.0** | **100.0** | **100.0** | **100.0** | **78.0** |

**Kairos (4B, official ckpt)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | — | — | — | — | — |
| beat_block_hammer | — | — | — | — | — |
| click_alarmclock | — | — | — | — | — |
| dump_bin_bigbin | — | — | — | — | — |
| grab_roller | — | — | — | — | — |
| handover_block | — | — | — | — | — |
| put_object_cabinet | — | — | — | — | — |
| scan_object | — | — | — | — | — |
| shake_bottle | — | — | — | — | — |
| stack_blocks_two | — | — | — | — | — |

**LingBot-VA (robotwin posttrain)**

| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| adjust_bottle | — | — | — | — | — |
| beat_block_hammer | — | — | — | — | — |
| click_alarmclock | — | — | — | — | — |
| dump_bin_bigbin | — | — | — | — | — |
| grab_roller | — | — | — | — | — |
| handover_block | — | — | — | — | — |
| put_object_cabinet | — | — | — | — | — |
| scan_object | — | — | — | — | — |
| shake_bottle | — | — | — | — | — |
| stack_blocks_two | — | — | — | — | — |


---

### 协议与披露

- **基准**: RoboTwin 2.0 (SAPIEN, aloha-agilex 双臂), `demo_clean` 任务配置,
  seen 指令, 50 trials/任务 × 10 任务, 每延迟 n=500。所有方法行的仿真本体均由
  task_config 的 `embodiment: [aloha-agilex]` 决定 (env_cfg_type=arx_x5 仅影响
  policy 侧动作维度, 且 aloha-agilex 与 dual_x5 维度一致: arm [6,6] / ee [1,1],
  状态/动作均 14 维)。
- **帧延迟协议** (CloudEdgeVLA 风格): 单流全观测老化; 每次 replan 从
  seed=1234 的 per-episode RNG 采样, replan 时观测取 hist[-1-d]。延迟分布
  存在实现差异 (LIBERO 侧同样如此, 见 LIBERO 报告): X-WAM/Motus 为
  d ~ U{0..D} (允许 0 = 新鲜观测), Fast-WAM 官方式实现为
  d ~ U{1..min(D,len-1)} (至少 1 步老化; 首个 replan 前仅 1 帧历史时退化
  为 0)。Fast-WAM 每 24 控制步 replan (官方 RoboTwin 默认 replan_steps=24),
  Motus 每 8 步 (chunk=8), X-WAM 全 chunk (Ta=32) 执行, AHA-WAM 几乎逐步
  replan (audit 实测 147-795 次/ep)。Fast-WAM 泳道
  `skip_get_obs_within_replan=false`, 即老化窗口需要逐步渲染观测流。
- **延迟剂量披露 (replans/ep 列)**: 延迟观测的毒害剂量 ≈ 抽样次数 × 实际老化
  深度; 各泳道抽样节奏不同 (见上), `replans/ep (D=40)` 列把节奏差异变成
  audit 实测数字 (每回合延迟抽样次数的均值)。同基准内: Fast-WAM 16.5 vs
  AHA-WAM ~450 (27×); 跨基准: LIBERO 泳道 chunk=10 系统同为 10 步一抽
  (上限 40 次/ep, 实测 26-40 随成功率变化), Fast-WAM 在 RoboTwin 的官方
  24 步 replan 只有其节奏的四成 —— 跨环境对比延迟鲁棒性时必须把该列
  一并披露 (见下方归因小节)。
- **expert_check**: 采用 RoboTwin 评审标准做法 —— 脚本专家都失败的种子被跳过
  (种子消耗与策略无关, 跨延迟一致)。
- **Motus 权重披露**: 官方仅公开 Stage-2 预训练底座 (HuggingFace,
  global_steps 420k), 论文的 RoboTwin finetune 权重未发布; 因此该行评估的是
  预训练底座, 与论文报告的 87% (finetune) **不可直接对比**。为加载该 ckpt 对
  推理几何做了适配: action chunk 16→8 (匹配训练时 action-only 8-token 布局),
  registers 4→0, 丢弃几何不匹配的确定性 sin-cos pos buffer, state encoder
  权重借自 ckpt 的 action encoder。代码适配见
  `code/Motus/inference/robotwin/Motus/models/motus.py` (load_checkpoint /
  MOTUS_NUM_REGISTERS)。
- **Motus sweep 已暂停 (2026-09-23, 用户确认)**: 离线探针证实 released 底座
  在 RoboTwin 上输出**退化动作** — 动作块恒 ≈0.01±0.01, 对 state (±0.3 rad)
  与视觉 (真实 vs 灰图) 扰动几乎无响应 (max|Δ|≈0.05, 与噪声同量级); episode
  视频中机器人几乎不动, 150+ episodes 全 0%。根因: 发布物自相矛盾 — ckpt
  几何为 8 action-only token (无 state encoder/registers), 而官方推理代码
  与 README 按 48-token+state+registers 构建 (官方示例 strict 加载失败)。
  README 的 87.02% 依赖未公开 finetune 权重。表中 Motus 行仅含已完成
  adjust_bottle 三格 (0.0), 其余为 —; 详见评估根目录下探针与审计日志。
- **X-WAM 权重与协议适配**: 使用官方已发布的 `robotwin_sft` finetune 权重
  (EE-space, action_type=ee, 3 视角 head/left/right, 16-d proprio, delta 动作
  在 anchor 状态上积分还原绝对 EE 位姿, replan_steps=None 执行整块预测)。
  坐标系: 训练帧 = 环境帧绕 z 轴旋转 -90° (`pos_train=(y_env,-x_env,z_env)`,
  `R_train=Rz(-90°)@R_env`, delta 按 `R'=dR@R` 世界系左乘积分), 已用
  config.yaml 分位角 8 位有效数字逐项验证; 专家状态对照探针证实 grip 通道
  (pre-grasp 观测下 Σdgrip=-0.978 满闭, anchor 扫描极性正确)。
  帧延迟协议移植自 Motus adapter 并保持语义一致 (离线探针
  `code/baseline_delay/probe_xwam_delay_2653853.py` 逐项验证: RNG 流回放一致、
  全观测老化、逐 episode 审计 jsonl 落盘)。每 episode 审计:
  `<run_root>/delay_<D>/<task>_delay_metrics.jsonl`。
- **X-WAM sweep 已暂停 (2026-09-29, 用户确认跳过)**: 帧修复后闭环 0/140+
  (beat_block_hammer D=0/10/20/40 全 0, adjust_bottle D=40 完整格 2/50=4%,
  官方声称 20 任务均值 89.8%), 差距 ~86 个百分点。逐项排查已关闭适配器侧
  嫌疑: 坐标系变换 8 位有效数字逐项核对、grip 头在专家 pre-grasp 观测下
  正确满闭 (Σdgrip=-0.978)、anchor 极性扫描正常、执行保真 0.0001、条件
  图像正常。残余失败为**轨迹级**: 模型首块转移高度偏低 (~0.92 vs 专家
  上绕 0.99), 开指掠过瓶子将其撞走, 此后无法重建 pre-grasp。官方数字
  出自作者内部 stack (未随仓库发布; 仓库附带的 XPolicyLab 适配有帧 bug,
  未被作者端到端跑通过), 复现其声称水平所需成本超出本研究预算 → 按用户
  决策跳过。保留数据: `x-wam/robotwin_sweep/delay_40/` 两完整格 +
  `_archive_framebug_20260928/` 帧修复前全部作废产物。
- **LeapBot-ce (ours) RoboTwin sweep 已停止 (2026-10-11, 用户确认不再投入)**:
  sweep 事实上已于当日 12:44 跑完 (50/50 格全有 .done 标记), 无进程残留;
  但结果**不达结论门槛, 行保留仅备查, 不入任何结论/论文**。判定依据:
  D=0 仅 7.4%, 其中 shake_bottle 一项 (74%) 贡献了全部 37 次成功,
  其余 9 任务全 0.0 (同格 Fast-WAM 99.6); 且 42.1 replans/ep 剂量正常
  —— 失败不是协议问题而是模型侧:
  RoboTwin 的 CloudEdge 配方/ckpt 未就位 (与 LEAPBOTCE.md「RoboTwin CE
  未实现」一致, pool_grid [6,10] override 也从未生效过)。继续烧卡无意义,
  数据保留于 `leapbotce/robotwin_sweep/` 供配方就位后对照。后续计划
  (用户 2026-10-11 确认): 等 LIBERO 消融定夺最终配方后, 用最终最优方案
  重开 RoboTwin 泳道正式测评一次, 中间不再单独排查/重训。
- **StarWAM (mot) 评估披露 (2026-10-13)**: 官方已发布 RoboTwin mot 配方权重
  (本机在盘, 免下载)。官方推理 server (`examples/robotwin/policy_server.py`,
  MoT 4/4 去噪步, `action_expert_init_from=null` 按官方 recipe override,
  qpos 14-d 动作) 经 XPolicyLab 代理 adapter 接入, 双臂 aloha-agilex 仿真
  本体与各泳道一致。帧延迟协议移植自 X-WAM adapter, 逐项语义一致
  (d ~ U{0..D} 逐 replan 抽样, per-episode RNG, 单流全观测老化, 同款
  X-WAM schema 审计 jsonl 落盘于 `<run_root>/delay_<D>/<task>_delay_metrics.jsonl`)。
  **replan 节奏与 Fast-WAM 官方 24 步泳道几乎重合** (audit 实测 D=40
  16.7 vs 16.5 replans/ep; D=0 仅 8.4 —— 成功回合更短、replan 更少), 两行
  延迟暴露剂量直接可比。任务敏感度排序与 AHA/Fast-WAM 一致 (handover
  80→44 / scan_object 94→60 / bbh 96→66 最敏感, click/shake/grab_roller
  准静态几乎不掉)。冒烟: bbh 5/5 (官方 97%), 10 任务 × 2 ep 19/20。
- **Kairos (4B) 评估披露 (2026-10-13)**: 官方已发布 RoboTwin 2.0 权重
  `kairos-4B-robot-RoboTwin2.0.safetensors` (本机在盘, 免下载), 官方 WAM
  推理服务 (`uvicorn wam_service.server_multi_gpu:app`) + chunked 评测
  client 接入。超参全取官方 RoboTwin 默认: action-horizon 32 /
  replan-steps 24 / num-inference-steps 30 / cfg-scale 8.0, clean/seen
  50 trials (`--phases clean` 为本仓新增, 默认 clean,random 保持官方行为)。
  帧延迟协议移植同 X-WAM (d ~ U{0..D} 逐 replan, per-episode RNG,
  单流全观测老化), 审计同款 schema 落 `<run_root>/delay_<D>/<task>_delay_metrics.jsonl`。
  chunked client 的 `_result_*.txt` 落相对 cwd, lane 脚本以 shim 从
  chunk_result.json 汇总写 `_result.txt` 到格目录。
- **LingBot-VA 评估披露 (2026-10-13)**: 官方 posttrain-for-RoboTwin 权重
  (video-diffusion 世界模型 + 动作专家, 本机在盘), 官方 VA server
  (`wan_va_server.py`, robotwin 配置) + 评测 client fork 接入。
  **协议语义披露 (重要)**: 官方 client 每次 replan 恒用**初始帧**
  (first_obs) 查询, KV-cache 更新携带新鲜关键帧 —— 即官方设计本身
  "replan 查询永不用新鲜观测"。本 campaign 归一化协议改为 replan 查询
  消费 fresh-obs history 的 aged obs (d ~ U{1..min(D,len-1)}, 与
  LIBERO 侧 client_delay.py 及 Fast-WAM 官方同款 1-based 抽样), KV-cache
  更新保持 fresh —— 与其它泳道可比; fork 为
  `evaluation/robotwin/eval_polict_client_delay.py` (官方原文件保持
  pristine, 抽样函数与 LIBERO 参考实现代码级一致)。history 按关键帧节奏
  append (每 action_per_frame 控制步), audit 同款 X-WAM schema。

### Fast-WAM 曲线为何远好于 LIBERO 泳道 (2026-10-11 延迟审计归因)

Fast-WAM RoboTwin 只掉 18 点而 LIBERO 同款官方 ckpt 掉 95 点, 复核结论:
**两组数字都真 (实现核查 + audit jsonl 复核), 差异来自"延迟暴露剂量"不可比**,
不是 RoboTwin 实现偏松。按证据强度排列:

1. **replan 间隔差 2.4× (协议事实)**。延迟只在 replan 那一刻注入, 两次 replan
   之间开环执行。RoboTwin 用官方默认 `replan_steps=24` → 实测 5.4-30.9 次
   stale 规划/ep (10 任务均值 16.5); LIBERO 泳道 `replan=10` → D=40 实测
   ~40 次/ep。单位时间暴露剂量 RoboTwin 是 LIBERO 的 ~40%。
2. **同环境内部对照 (最硬的证据)**。AHA-WAM 同一批任务、同协议、实际延迟均值
   同为 ~19 帧, 但几乎逐步 replan (audit: 147-795 次/ep, 均值 ~450 ≈ Fast-WAM
   的 27×) → 掉 97.6→48.6 (-49)。环境本身延迟敏感, Fast-WAM 的 -18 是
   暴露次数被 24 步 chunk 压低后的结果。
3. **per-task replans/ep 与掉点排序两方法都几乎一致** (延迟按"每次 replan
   抽一次签"计费, 抽签次数决定剂量):
   - Fast-WAM: click_alarmclock 5.4 次/ep → 掉 0, grab_roller 8.3 → 0,
     shake_bottle 7.9 → -4; handover_block 30.9 → -52, stack_blocks 29.9 → -20,
     put_object_cabinet 26.6 → -26。
   - AHA: click_alarmclock 147 → 0, shake_bottle 186 → 0, grab_roller 258 → -6;
     handover 795 → -88, stack_blocks 791 → -92, put_object_cabinet 674 → -68。
   - StarWAM (2026-10-13): 16.7 replans/ep → 95.4→78.7 (-16.7), 与 Fast-WAM
     同节奏 (16.5 → -18.0) 掉幅几乎相同 —— 同剂量同掉幅, 进一步支持
     剂量决定论 (排除"模型差异主导"的解释)。
4. **任务构成**: click_alarmclock / shake_bottle / grab_roller 目标准静态,
   任意暴露量级都几乎不掉 (AHA 147-258 次暴露仍 100/100/94), 3/10 任务托住
   均值; LIBERO 四套件全是精密短回合操作, 没有这类"免死"任务。
5. **ckpt 变体差异 (次要, 未核查)**: LIBERO 行是 `libero_uncond_2cam224`
   无条件变体, RoboTwin 行是官方 release 原生配方, 两者本身稳健性可能不同。

**论文处理**: 跨环境对比延迟鲁棒性必须披露 replan interval (10 vs 24) 或把
曲线按每回合 replan 次数归一; 表述避免"Fast-WAM 对延迟鲁棒", 改为"低 replan
频次 + 准静态任务占比高使 RoboTwin 泳道暴露剂量低"。可选补强: RoboTwin
Fast-WAM `replan_steps=10` 对照 (D=40 一格 ~500 eps) 直接量化协议敏感性,
需要时按现有 sweep 脚本加一个 override 即可。

