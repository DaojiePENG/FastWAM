# FastWAM 微调指南 (Fine-tuning Guide)

## 概述 (Overview)

本指南介绍如何使用 FastWAM 的微调功能，特别是如何仅微调动作头（action head）而保持模型主体（backbone）不变。

---

## 微调策略 (Fine-tuning Strategies)

FastWAM 支持以下四种微调策略：

### 1. `action_head_only` (推荐用于快速适应)
- **训练**: 仅 action expert 的输出层 (head)
- **冻结**: action expert 的 backbone, video expert
- **用途**: 快速适应新任务，参数量最小，训练最快
- **适用场景**: 新任务与预训练任务相似，只需微调输出映射

### 2. `action_only`
- **训练**: 整个 action expert (包括 backbone 和 head)
- **冻结**: video expert
- **用途**: 深度适应新的动作空间
- **适用场景**: 机器人动作空间发生变化

### 3. `video_only`
- **训练**: 整个 video expert
- **冻结**: action expert
- **用途**: 适应新的视觉环境
- **适用场景**: 视觉输入发生显著变化，但动作空间不变

### 4. `none` (默认，完整训练)
- **训练**: 整个 MoT (包括 video expert 和 action expert)
- **冻结**: 无
- **用途**: 从头训练或完整微调

---

## 快速开始 (Quick Start)

### 步骤 1: 下载预训练模型

```bash
# 下载 LIBERO 预训练模型
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release

# 或下载 RoboTwin 预训练模型
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

### 步骤 2: 准备数据集

确保你的数据集已准备好。参见主 README.md 中的数据集下载说明。

### 步骤 3: 运行微调

使用提供的微调脚本：

```bash
# 单 GPU 微调 LIBERO action head
bash scripts/finetune.sh 1 libero_finetune_action_head

# 多 GPU 微调 LIBERO action head (例如 4 个 GPU)
bash scripts/finetune.sh 4 libero_finetune_action_head

# 微调 RoboTwin action head
bash scripts/finetune.sh 1 robotwin_finetune_action_head
```

或直接使用训练脚本：

```bash
# 单 GPU
python scripts/train.py task=libero_finetune_action_head

# 多 GPU with DeepSpeed
bash scripts/train_zero1.sh 4 task=libero_finetune_action_head
```

---

## 自定义微调配置 (Custom Fine-tuning Configuration)

### 方法 1: 创建新的任务配置文件

创建 `configs/task/my_finetune_task.yaml`:

```yaml
# @package _global_

defaults:
  - finetune
  - data: libero_2cam  # 或 robotwin
  - model: fastwam
  - _self_

# 微调策略
finetune:
  freeze_strategy: "action_head_only"  # 选择策略
  freeze_video_expert: true
  freeze_action_backbone: true

# 训练配置
learning_rate: 1.0e-5  # 微调通常使用更小的学习率
batch_size: 4
num_epochs: 3

# 必须指定预训练检查点
resume: "./checkpoints/fastwam_release/libero_uncond_2cam224.pt"

# 输出目录
output_dir: ./runs/finetune/my_task/${now:%Y-%m-%d}_${now:%H-%M-%S}
```

然后运行：

```bash
python scripts/train.py task=my_finetune_task
```

### 方法 2: 通过命令行覆盖配置

```bash
python scripts/train.py \
  task=libero_uncond_2cam224_1e-4 \
  finetune.freeze_strategy=action_head_only \
  finetune.freeze_video_expert=true \
  finetune.freeze_action_backbone=true \
  learning_rate=1.0e-5 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  output_dir=./runs/finetune/my_experiment
```

---

## 不同微调策略的使用示例

### 示例 1: 仅微调 Action Head (最快)

**场景**: 在相似任务上快速适应

```bash
python scripts/train.py \
  task=libero_uncond_2cam224_1e-4 \
  finetune.freeze_strategy=action_head_only \
  learning_rate=1.0e-5 \
  num_epochs=3 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

**预期**:
- 训练速度: 最快
- GPU 内存: 最少
- 可训练参数: ~7K (action_dim × hidden_dim)
- 适应能力: 有限但快速

### 示例 2: 微调整个 Action Expert

**场景**: 动作空间发生变化，需要深度适应

```bash
python scripts/train.py \
  task=libero_uncond_2cam224_1e-4 \
  finetune.freeze_strategy=action_only \
  learning_rate=5.0e-6 \
  num_epochs=5 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

**预期**:
- 训练速度: 中等
- GPU 内存: 中等
- 可训练参数: ~100M (action expert 全部参数)
- 适应能力: 强

### 示例 3: 微调 Video Expert

**场景**: 视觉环境变化（新相机、新场景）

```bash
python scripts/train.py \
  task=libero_uncond_2cam224_1e-4 \
  finetune.freeze_strategy=video_only \
  learning_rate=5.0e-6 \
  num_epochs=5 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

---

## 参数说明 (Parameter Descriptions)

### 微调配置参数

```yaml
finetune:
  # 冻结策略
  freeze_strategy: "action_head_only"  # 必需
    # 可选值: "none", "action_head_only", "action_only", "video_only"
  
  # 是否冻结 video expert (仅在非 video_only 时有效)
  freeze_video_expert: true
  
  # 是否冻结 action backbone (仅在非 action_only 时有效)
  freeze_action_backbone: true
```

### 推荐的超参数

| 策略 | Learning Rate | Batch Size | Epochs | Warmup |
|------|---------------|------------|--------|--------|
| `action_head_only` | 1e-5 ~ 5e-5 | 4-8 | 2-5 | 5% |
| `action_only` | 5e-6 ~ 1e-5 | 2-4 | 5-10 | 5% |
| `video_only` | 5e-6 ~ 1e-5 | 2-4 | 5-10 | 5% |
| `none` (全量) | 1e-4 ~ 5e-4 | 2-4 | 10-20 | 5% |

**注意**: 
- 微调时通常使用比预训练小 10-100 倍的学习率
- Batch size 取决于 GPU 内存
- 使用 cosine 学习率调度器

---

## 检查训练的参数 (Check Trainable Parameters)

训练开始时，日志会显示可训练参数的数量：

```
INFO - Total trainable parameters: 7168 (0.01M)
INFO - Action expert head is trainable. All other components are frozen.
```

不同策略的参数量参考：

- **action_head_only**: ~7K-50K (取决于 action_dim 和 hidden_dim)
- **action_only**: ~100M (action expert 全部)
- **video_only**: ~3B (video expert 全部)
- **none**: ~3.1B (整个 MoT)

---

## 常见问题 (FAQ)

### Q1: 微调需要多少 GPU 内存？

**A**: 取决于策略：
- `action_head_only`: 单卡 24GB (与推理相当)
- `action_only`: 单卡 40GB 或多卡
- `video_only`/`none`: 需要 80GB+ 或多卡训练

### Q2: 微调需要多长时间？

**A**: 以 LIBERO 为例：
- `action_head_only`: ~1-2 小时 (单卡 A100)
- `action_only`: ~4-8 小时
- `video_only`: ~12-24 小时

### Q3: 如何选择微调策略？

**A**: 
1. **任务相似，数据少** → `action_head_only`
2. **新的动作空间** → `action_only`
3. **新的视觉环境** → `video_only`
4. **完全新任务，数据多** → `none`

### Q4: 可以冻结部分 backbone 层吗？

**A**: 当前版本支持 all-or-nothing 冻结。如需更细粒度的控制，可以修改 `trainer.py` 中的 `_apply_freeze_strategy` 方法。

### Q5: 微调后如何评估？

**A**: 使用与预训练模型相同的评估脚本：

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/my_task/.../checkpoints/weights/step_001000.pt

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/my_task/.../checkpoints/weights/step_001000.pt
```

---

## 高级用法 (Advanced Usage)

### 从中间检查点恢复微调

```bash
python scripts/train.py \
  task=libero_finetune_action_head \
  resume=./runs/finetune/previous_run/checkpoints/state/step_000500
```

**注意**: 使用 `state` 目录（包含优化器状态）而非 `weights` 目录。

### 混合精度训练

默认使用 `bf16` 混合精度：

```yaml
mixed_precision: "bf16"  # 推荐
# 或
mixed_precision: "fp16"  # 如果硬件不支持 bf16
# 或
mixed_precision: "no"    # 全精度（慢但更稳定）
```

### 使用 WandB 跟踪训练

```bash
python scripts/train.py \
  task=libero_finetune_action_head \
  wandb.enabled=true \
  wandb.workspace=your_workspace \
  wandb.project=fastwam-finetune \
  wandb.name=action_head_only_exp1
```

---

## 代码修改说明 (Code Modifications)

### 修改的文件

1. **`configs/finetune.yaml`** (新建)
   - 定义微调的默认配置

2. **`configs/task/libero_finetune_action_head.yaml`** (新建)
   - LIBERO 微调任务示例配置

3. **`configs/task/robotwin_finetune_action_head.yaml`** (新建)
   - RoboTwin 微调任务示例配置

4. **`src/fastwam/trainer.py`** (修改)
   - 添加 `_apply_freeze_strategy()` 方法
   - 添加 `_get_trainable_parameters()` 方法
   - 添加 `_train_action_head_only()`, `_train_action_only()`, `_train_video_only()` 方法
   - 修改优化器初始化逻辑

5. **`scripts/finetune.sh`** (新建)
   - 微调的便捷启动脚本

### 向后兼容性

所有修改都保持向后兼容：
- 不指定 `finetune` 配置时，默认行为与原始代码相同（训练整个 MoT）
- 现有的训练脚本和配置无需修改即可继续使用

---

## 故障排查 (Troubleshooting)

### 错误: "No trainable parameters found"

**原因**: 冻结策略配置错误或模型结构不匹配

**解决**:
1. 检查 `freeze_strategy` 拼写
2. 确认模型有 `action_expert` 或 `video_expert` 属性
3. 查看日志中的详细错误信息

### 错误: "Checkpoint missing both mot and dit keys"

**原因**: 加载的检查点格式不正确

**解决**:
1. 确保使用 FastWAM 的检查点（包含 `mot` 键）
2. 检查 `resume` 路径是否正确

### 训练损失不下降

**可能原因**:
1. 学习率过大或过小
2. 冻结了不应该冻结的参数

**解决**:
1. 尝试调整学习率（增大或减小 10 倍）
2. 检查日志中的 "Total trainable parameters" 是否合理
3. 尝试更简单的冻结策略（如 `action_only`）

---

## 最佳实践 (Best Practices)

1. **始终从预训练模型开始**: 指定 `resume` 参数
2. **使用较小的学习率**: 比预训练小 10-100 倍
3. **监控验证损失**: 避免过拟合
4. **保存多个检查点**: 使用较小的 `save_every` 值
5. **记录实验**: 使用 WandB 或其他工具跟踪实验

---

## 参考资料 (References)

- 主仓库 README: [README.md](../README.md)
- 论文: [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- 项目主页: https://yuantianyuan01.github.io/FastWAM/

---

## 更新日志 (Changelog)

- **2026-04-15**: 添加微调功能
  - 支持 4 种冻结策略
  - 添加配置文件和示例脚本
  - 添加使用指南
