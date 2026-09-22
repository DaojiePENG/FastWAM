# RoboTwin 2.0 帧延迟基线 (CloudEdgeVLA 协议)

*更新: 2026-09-22 17:42 · 与 LIBERO 报告独立: 见 `baseline_results_libero.md`*

## 主表 (10 任务 × 50 trials, 成功率 %)

| 方法 | D=0 | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|
| Motus (stage-2 ckpt) | — | — | — | — | — |

### 每任务明细

**Motus (stage-2 ckpt)**

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

- **基准**: RoboTwin 2.0 (SAPIEN, aloha/agilex 双臂), `demo_clean` 任务配置,
  seen 指令, 50 trials/任务 × 10 任务, 每延迟 n=500。
- **帧延迟协议** (CloudEdgeVLA 风格): 单流全观测老化; 每次 replan 从
  seed=1234 的 RNG 采样 d ~ U{0..D}; Motus replan 每 8 个控制步 (chunk=8)。
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

