# FastWAM 评测指南 (Evaluation Guide)

## 概述 (Overview)

本指南介绍如何评测 FastWAM 模型（预训练或微调后）在 LIBERO 和 RoboTwin 环境中的性能。

**支持的评测环境：**
- **LIBERO**: 4个任务套件，共40个操作任务
- **RoboTwin**: 双臂机器人协作任务

---

## 目录 (Table of Contents)

- [快速开始](#快速开始-quick-start)
- [Checkpoint 说明](#checkpoint-说明)
- [LIBERO 评测](#libero-评测)
- [RoboTwin 评测](#robotwin-评测)
- [结果分析](#结果分析)
- [高级用法](#高级用法)
- [常见问题](#常见问题-faq)

---

## 快速开始 (Quick Start)

### 1. 准备 Checkpoint

**预训练模型：**
```bash
# LIBERO
./checkpoints/fastwam_release/libero_uncond_2cam224.pt

# RoboTwin
./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

**微调后的模型：**
```bash
# LIBERO 微调模型
./runs/finetune/libero_action_head/<timestamp>/checkpoints/weights/step_XXXXXX.pt

# RoboTwin 微调模型
./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_XXXXXX.pt
```

### 2. 快速验证（1个任务，10次试验）

**LIBERO:**
```bash
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/libero_action_head/2026-04-16_12-54-06/checkpoints/weights/step_002000.pt \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  gpu_id=0
```

**RoboTwin:**
```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.task_name=pick_and_place \
  EVALUATION.num_trials=10 \
  gpu_id=0
```

---

## Checkpoint 说明

### 预训练模型 vs 微调模型

| 类型 | 路径格式 | 大小 | 用途 |
|------|---------|------|------|
| **预训练模型** | `checkpoints/fastwam_release/*.pt` | ~12GB | 直接评测预训练性能 |
| **微调模型 (weights)** | `runs/finetune/*/checkpoints/weights/step_*.pt` | ~12GB | 评测微调后的性能 |
| **微调模型 (state)** | `runs/finetune/*/checkpoints/state/step_*/` | ~24GB+ | 恢复训练（包含优化器状态） |

**重要：**
- 评测时使用 `weights/step_*.pt` 文件（纯模型权重）
- 恢复训练时使用 `state/step_*/` 目录（完整训练状态）
- 两者的模型权重完全相同，区别在于是否包含优化器状态

### 如何选择最佳 Checkpoint

**方法 1: 查看 WandB**
```bash
# 登录 WandB dashboard
# 查看训练曲线选择验证损失最低的 step
```

**方法 2: 查看训练日志**
```bash
# 查看所有保存的 checkpoint
ls -lh runs/finetune/*/checkpoints/weights/

# 列出所有 step
find runs/finetune/*/checkpoints/weights/ -name "*.pt" | sort
```

**方法 3: 快速评测对比**
```bash
# 评测多个 checkpoint，选择成功率最高的
for ckpt in runs/finetune/*/checkpoints/weights/*.pt; do
  echo "Evaluating $ckpt"
  python experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt=$ckpt \
    EVALUATION.num_trials=10
done
```

---

## LIBERO 评测

LIBERO 包含 4 个任务套件：
- **libero_10**: 10个基础任务
- **libero_goal**: 10个目标条件任务
- **libero_spatial**: 10个空间推理任务
- **libero_object**: 10个物体操作任务

### 完整评测（所有任务，50次试验）

**使用评测管理器（推荐，多GPU并行）：**

```bash
# 评测所有4个任务套件（共40个任务 × 50次 = 2000次试验）
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/libero_action_head/2026-04-16_12-54-06/checkpoints/weights/step_002000.pt
```

**默认配置（configs/sim_libero.yaml）：**
```yaml
MULTIRUN:
  task_suite_names:
    - libero_10
    - libero_goal
    - libero_spatial
    - libero_object
  num_gpus: 8              # 使用8个GPU
  max_tasks_per_gpu: 2     # 每个GPU同时运行2个任务

EVALUATION:
  num_trials: 50           # 每个任务运行50次
  output_dir: ./evaluate_results/libero/...
```

**自定义GPU数量：**
```bash
# 使用4个GPU
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=2
```

### 评测特定任务套件

**仅评测 libero_10：**
```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.task_suite_names=[libero_10] \
  MULTIRUN.num_gpus=2
```

**评测多个特定套件：**
```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.task_suite_names=[libero_10,libero_spatial] \
  MULTIRUN.num_gpus=4
```

### 评测单个任务

**适用场景：**
- 调试特定任务
- 快速验证模型
- 单GPU环境

```bash
# 评测 libero_10 的第0个任务，运行50次
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50 \
  gpu_id=0
```

**任务ID范围：**
- libero_10: task_id 0-9
- libero_goal: task_id 0-9
- libero_spatial: task_id 0-9
- libero_object: task_id 0-9

### 快速评测（减少试验次数）

```bash
# 每个任务仅运行10次（快速验证）
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.task_suite_names=[libero_10] \
  MULTIRUN.num_gpus=1 \
  EVALUATION.num_trials=10
```

### 评测配置参数详解

```yaml
EVALUATION:
  # 任务配置
  task_suite_name: libero_10    # 任务套件名称
  task_id: 0                    # 任务ID（0-9）
  num_trials: 50                # 每个任务的试验次数

  # 环境配置
  env_num: 1                    # 并行环境数量（通常为1）
  num_steps_wait: 30            # 初始等待步数
  replan_steps: 10              # 重新规划间隔（步数）
  binarize_gripper: true        # 是否二值化gripper动作

  # 推理配置
  action_horizon: null          # 动作预测窗口（null=使用模型默认值）
  num_inference_steps: 50       # 扩散模型推理步数
  text_cfg_scale: 1.0          # 文本条件引导强度
  action_cfg_scale: 1.0        # 动作条件引导强度（仅FastWAM）
  sigma_shift: null            # 噪声调度偏移（null=使用模型默认值）
  
  # 可视化
  visualize_future_video: false # 是否保存预测的视频帧
  use_action_ensembler: false   # 是否使用动作集成（时序平滑）
```

### LIBERO 评测输出

**输出目录结构：**
```
evaluate_results/libero/libero_uncond_2cam224_1e-4/<timestamp>/
├── manager_config.yaml          # 评测配置
├── tasks.txt                    # 任务列表（suite_name,task_id）
├── results/                     # 详细结果
│   ├── libero_10_task_0/
│   │   ├── trial_00.json       # 每次试验的详细信息
│   │   ├── trial_01.json
│   │   └── ...
│   ├── libero_10_task_1/
│   └── ...
├── summary.json                 # 汇总统计（成功率）
└── failed_tasks.txt             # 失败任务列表（如果有）
```

**查看结果：**
```bash
# 自动汇总成功率
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/libero_uncond_2cam224_1e-4/<timestamp>

# 输出示例：
# libero_10:     80.0% (8/10 tasks, avg 45.2/50 trials)
# libero_goal:   75.0% (7.5/10 tasks, avg 42.1/50 trials)
# libero_spatial: 70.0% (7/10 tasks, avg 38.5/50 trials)
# libero_object:  65.0% (6.5/10 tasks, avg 35.0/50 trials)
# Overall:       72.5% (avg success rate across all tasks)
```

---

## RoboTwin 评测

RoboTwin 是一个双臂机器人协作任务环境。

### 完整评测

**使用评测管理器：**

```bash
# 评测所有RoboTwin任务
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_XXXXXX.pt
```

**默认配置（configs/sim_robotwin.yaml）：**
```yaml
MULTIRUN:
  task_names:
    - pick_and_place
    - bimanual_assembly
    - handover
    - coordination_task
  num_gpus: 8
  max_tasks_per_gpu: 1     # RoboTwin任务较复杂，建议每GPU运行1个

EVALUATION:
  num_trials: 50
  output_dir: ./evaluate_results/robotwin/...
```

### 评测单个任务

```bash
# 评测单个RoboTwin任务
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.task_name=pick_and_place \
  EVALUATION.num_trials=50 \
  gpu_id=0
```

**可用任务名称：**
- `pick_and_place`: 拾取与放置
- `bimanual_assembly`: 双臂组装
- `handover`: 物体交接
- `coordination_task`: 协调任务

### RoboTwin 评测配置

```yaml
EVALUATION:
  # 任务配置
  task_name: pick_and_place     # 任务名称
  num_trials: 50                # 试验次数

  # 环境配置
  env_num: 1
  num_steps_wait: 30
  replan_steps: 15              # RoboTwin推荐15步重新规划
  binarize_gripper: false       # RoboTwin使用连续gripper

  # 推理配置（与LIBERO类似）
  num_inference_steps: 50
  text_cfg_scale: 1.0
  action_cfg_scale: 1.0
  
  # RoboTwin特定配置
  bimanual_coordination: true   # 启用双臂协调
  camera_setup: "3cam"          # 相机配置（3个相机）
```

### RoboTwin 评测输出

输出结构与LIBERO类似：
```
evaluate_results/robotwin/robotwin_uncond_3cam_384_1e-4/<timestamp>/
├── manager_config.yaml
├── tasks.txt
├── results/
│   ├── pick_and_place/
│   ├── bimanual_assembly/
│   └── ...
└── summary.json
```

---

## 结果分析

### 对比预训练模型和微调模型

**1. 分别评测：**
```bash
# 评测预训练模型
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.output_dir=./evaluate_results/libero/pretrained_baseline

# 评测微调模型
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/libero_action_head/2026-04-16_12-54-06/checkpoints/weights/step_002000.pt \
  EVALUATION.output_dir=./evaluate_results/libero/finetuned_action_head
```

**2. 汇总对比：**
```bash
# 汇总预训练结果
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/pretrained_baseline

# 汇总微调结果
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/finetuned_action_head

# 手动对比两个输出的成功率
```

### 分析特定任务的失败原因

**查看单个试验的详细日志：**
```bash
# 查看某个任务的所有试验结果
ls evaluate_results/libero/.../results/libero_10_task_0/

# 查看失败试验的详细信息
cat evaluate_results/libero/.../results/libero_10_task_0/trial_03.json
```

**试验结果JSON格式：**
```json
{
  "success": false,
  "num_steps": 300,
  "final_reward": 0.45,
  "error_type": "timeout",  // 或 "collision", "failure"
  "trajectory_length": 300
}
```

### 可视化评测结果

**保存预测的视频帧：**
```bash
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5 \
  EVALUATION.visualize_future_video=true \
  gpu_id=0
```

输出会包含预测的视频帧（用于调试模型的视频预测能力）。

---

## 高级用法

### 1. 批量评测多个Checkpoint

**创建评测脚本：**
```bash
#!/bin/bash
# eval_all_checkpoints.sh

TASK="libero_uncond_2cam224_1e-4"
CKPT_DIR="runs/finetune/libero_action_head/2026-04-16_12-54-06/checkpoints/weights"

for ckpt in $CKPT_DIR/*.pt; do
  step=$(basename $ckpt .pt)
  echo "Evaluating $step..."
  
  python experiments/libero/run_libero_manager.py \
    task=$TASK \
    ckpt=$ckpt \
    MULTIRUN.task_suite_names=[libero_10] \
    MULTIRUN.num_gpus=2 \
    EVALUATION.num_trials=20 \
    EVALUATION.output_dir=./evaluate_results/libero/comparison/$step
done

# 汇总所有结果
for step_dir in evaluate_results/libero/comparison/*/; do
  echo "=== $(basename $step_dir) ==="
  python experiments/libero/summarize_results.py --result_dir $step_dir
done
```

### 2. 调整推理超参数

**增加推理步数（提高质量但降低速度）：**
```bash
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.num_inference_steps=100 \
  EVALUATION.task_id=0
```

**调整CFG scale（条件引导强度）：**
```bash
# 增强文本条件引导
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.text_cfg_scale=2.0 \
  EVALUATION.action_cfg_scale=1.5
```

### 3. 使用动作集成（时序平滑）

```bash
# 启用动作集成器（减少抖动）
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.use_action_ensembler=true \
  EVALUATION.task_id=0
```

### 4. 调整重新规划频率

```bash
# 更频繁的重新规划（可能提高适应性但增加计算开销）
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.replan_steps=5 \
  EVALUATION.task_id=0
```

### 5. 使用外部数据集统计信息

```bash
# 使用预计算的数据集统计信息（用于动作归一化）
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_id=0
```

---

## 常见问题 (FAQ)

### Q1: 评测需要多长时间？

**A**: 取决于配置：

**LIBERO (单个任务套件，10个任务 × 50次):**
- 单GPU: ~4-6小时
- 8 GPUs (2 tasks/GPU): ~30-45分钟

**LIBERO (所有4个套件，40个任务 × 50次):**
- 单GPU: ~16-24小时
- 8 GPUs: ~2-3小时

**RoboTwin (单个任务 × 50次):**
- 单GPU: ~1-2小时
- 8 GPUs: ~15-20分钟

**快速验证 (10次试验):**
- 任何配置: 5-15分钟

### Q2: 需要多少GPU内存？

**A**: 
- **推理/评测**: 单个任务约20-24GB (A100/H100)
- **并行评测**: 每个GPU可同时运行1-2个任务
- **建议**: 对于LIBERO使用24GB+ GPU, RoboTwin使用40GB+ GPU

### Q3: 如何选择合适的试验次数？

**A**:
- **快速验证**: 10次（检查模型是否工作）
- **标准评测**: 50次（论文报告）
- **高精度评测**: 100次（关键结果）

**统计显著性**:
- 10次: 标准误差 ~10%
- 50次: 标准误差 ~4.5%
- 100次: 标准误差 ~3%

### Q4: 微调后成功率下降怎么办？

**可能原因**:
1. 过拟合（训练时间过长）
2. 学习率过大
3. 冻结策略不合适
4. 数据分布不匹配

**解决方案**:
1. 尝试早期的checkpoint
2. 降低学习率重新微调
3. 尝试不同的冻结策略（如 `action_only` 而非 `action_head_only`）
4. 检查数据集统计信息是否正确

### Q5: 评测脚本卡住不动怎么办？

**常见原因**:
1. GPU内存不足（OOM）
2. 环境初始化失败
3. 任务ID不存在
4. Checkpoint路径错误

**调试方法**:
```bash
# 1. 检查GPU内存
nvidia-smi

# 2. 减少并行任务数
MULTIRUN.max_tasks_per_gpu=1

# 3. 单任务调试
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  gpu_id=0

# 4. 查看详细日志
HYDRA_FULL_ERROR=1 python experiments/libero/eval_libero_single.py ...
```

### Q6: 如何比较不同微调策略的效果？

**A**: 使用相同的评测配置评测所有模型：

```bash
# 预训练baseline
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.output_dir=./compare/pretrained

# 微调策略1: action_head_only
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/action_head_only/.../step_002000.pt \
  EVALUATION.output_dir=./compare/action_head_only

# 微调策略2: action_only
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/action_only/.../step_002000.pt \
  EVALUATION.output_dir=./compare/action_only

# 汇总对比
for dir in compare/*/; do
  echo "=== $(basename $dir) ==="
  python experiments/libero/summarize_results.py --result_dir $dir
done
```

### Q7: 评测结果与论文不一致怎么办？

**检查清单**:
1. ✅ 使用正确的checkpoint
2. ✅ 数据集统计信息正确（dataset_stats.json）
3. ✅ 试验次数足够（至少50次）
4. ✅ 推理步数正确（默认50）
5. ✅ CFG scale正确（text_cfg_scale=1.0, action_cfg_scale=1.0）
6. ✅ 环境配置正确（LIBERO版本、相机数量等）

### Q8: 如何保存和分享评测结果？

**汇总报告：**
```bash
# 生成Markdown报告
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/.../... \
  --output_format markdown \
  --output_file evaluation_report.md

# 生成JSON报告（机器可读）
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/.../... \
  --output_format json \
  --output_file evaluation_results.json
```

**打包结果：**
```bash
# 打包评测结果（便于分享）
tar -czf evaluation_results.tar.gz \
  evaluate_results/libero/.../summary.json \
  evaluate_results/libero/.../manager_config.yaml \
  evaluation_report.md
```

---

## 最佳实践 (Best Practices)

### 1. 完整的评测流程

```bash
# 步骤1: 快速验证（确保模型能运行）
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5

# 步骤2: 小规模评测（1个任务套件，20次试验）
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.task_suite_names=[libero_10] \
  MULTIRUN.num_gpus=2 \
  EVALUATION.num_trials=20

# 步骤3: 完整评测（所有套件，50次试验）
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.num_gpus=8 \
  EVALUATION.num_trials=50
```

### 2. 记录评测配置

```bash
# 保存评测命令
echo "python experiments/libero/run_libero_manager.py ..." > eval_command.sh

# 保存git commit hash
git rev-parse HEAD > eval_commit.txt

# 保存checkpoint信息
ls -lh ./runs/finetune/.../step_002000.pt > checkpoint_info.txt
```

### 3. 并行评测多个模型

```bash
# 使用GNU parallel同时评测多个checkpoint
parallel -j 4 python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt={} \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50 \
  ::: runs/finetune/*/checkpoints/weights/*.pt
```

---

## 参考资料 (References)

- **微调指南**: [FINETUNING.md](FINETUNING.md)
- **主仓库 README**: [README.md](README.md)
- **论文**: [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- **项目主页**: https://yuantianyuan01.github.io/FastWAM/
- **LIBERO Benchmark**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **RoboTwin Dataset**: https://robotwin.github.io/

---

## 更新日志 (Changelog)

- **2026-04-16**: 创建评测指南
  - 添加LIBERO和RoboTwin完整评测流程
  - 添加checkpoint说明和选择方法
  - 添加结果分析和故障排查指南
  - 添加高级用法和最佳实践

---

## 贡献 (Contributing)

如果您在评测过程中遇到问题或有改进建议，欢迎：
1. 提交 Issue: https://github.com/yuantianyuan01/FastWAM/issues
2. 提交 Pull Request 改进本文档
3. 分享您的评测结果和经验

---

**祝评测顺利！Good luck with your evaluation!** 🚀