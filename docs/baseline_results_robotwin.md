# RoboTwin 2.0 帧延迟基线 (CloudEdgeVLA 协议)

*更新: 2026-09-30 12:13 · 与 LIBERO 报告独立: 见 `baseline_results_libero.md`*

## 主表 (10 任务 × 50 trials, 成功率 %)

| 方法 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| Motus (stage-2 ckpt) | 0.0 (n=50) | 0.0 (n=50) | — | 0.0 (n=50) | — |
| X-WAM (robotwin_sft) | — | — | — | — | 2.0 (n=100) |
| Fast-WAM (official release) | — | — | — | — | — |

### 每任务明细

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
  Motus 每 8 步 (chunk=8), X-WAM 全 chunk (Ta=32) 执行。Fast-WAM 泳道
  `skip_get_obs_within_replan=false`, 即老化窗口需要逐步渲染观测流。
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

