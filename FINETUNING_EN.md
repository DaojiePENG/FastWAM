# FastWAM Fine-tuning Guide

This guide shows how to fine-tune FastWAM, especially how to fine-tune only the action head while keeping the model backbone frozen.

## Supported Fine-tuning Strategies

FastWAM supports 4 fine-tuning strategies:

### 1. `action_head_only` (Recommended for quick adaptation)
- **Train**: Only the action expert's output layer (head)
- **Freeze**: Action expert backbone, video expert
- **Use case**: Quick adaptation with minimal parameters

### 2. `action_only`
- **Train**: Entire action expert (backbone + head)
- **Freeze**: Video expert
- **Use case**: Adapting to new action spaces

### 3. `video_only`
- **Train**: Entire video expert
- **Freeze**: Action expert
- **Use case**: Adapting to new visual environments

### 4. `none` (Default, full training)
- **Train**: Entire MoT (both video and action experts)
- **Freeze**: None
- **Use case**: Training from scratch or full fine-tuning

## Quick Start

### Step 1: Download Pretrained Model

```bash
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  --local-dir ./checkpoints/fastwam_release
```

### Step 2: Run Fine-tuning

```bash
# Single GPU
bash scripts/finetune.sh 1 libero_finetune_action_head

# Multi-GPU (e.g., 4 GPUs)
bash scripts/finetune.sh 4 libero_finetune_action_head
```

Or directly:

```bash
python scripts/train.py task=libero_finetune_action_head
```

## Custom Fine-tuning Configuration

### Method 1: Create Task Config

Create `configs/task/my_finetune.yaml`:

```yaml
# @package _global_

defaults:
  - finetune
  - data: libero_2cam
  - model: fastwam
  - _self_

finetune:
  freeze_strategy: "action_head_only"
  freeze_video_expert: true
  freeze_action_backbone: true

learning_rate: 1.0e-5
batch_size: 4
num_epochs: 3

resume: "./checkpoints/fastwam_release/libero_uncond_2cam224.pt"
output_dir: ./runs/finetune/my_task/${now:%Y-%m-%d}_${now:%H-%M-%S}
```

Then run:

```bash
python scripts/train.py task=my_finetune
```

### Method 2: Command-line Override

```bash
python scripts/train.py \
  task=libero_uncond_2cam224_1e-4 \
  finetune.freeze_strategy=action_head_only \
  learning_rate=1.0e-5 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

## Recommended Hyperparameters

| Strategy | Learning Rate | Batch Size | Epochs |
|----------|---------------|------------|--------|
| `action_head_only` | 1e-5 ~ 5e-5 | 4-8 | 2-5 |
| `action_only` | 5e-6 ~ 1e-5 | 2-4 | 5-10 |
| `video_only` | 5e-6 ~ 1e-5 | 2-4 | 5-10 |
| `none` | 1e-4 ~ 5e-4 | 2-4 | 10-20 |

## Verification

Check the log for trainable parameters:

```
INFO - Total trainable parameters: 7168 (0.01M)
INFO - Action expert head is trainable. All other components are frozen.
```

Expected parameter counts:
- **action_head_only**: ~7K-50K
- **action_only**: ~100M
- **video_only**: ~3B
- **none**: ~3.1B

## Evaluation

Use the same evaluation scripts:

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

## Best Practices

1. **Always start from pretrained**: Specify `resume` parameter
2. **Use smaller learning rate**: 10-100x smaller than pretraining
3. **Monitor validation loss**: Avoid overfitting
4. **Save multiple checkpoints**: Use smaller `save_every` value
5. **Track experiments**: Use WandB or other tools

## Code Modifications

Modified files:
1. `configs/finetune.yaml` (new)
2. `configs/task/libero_finetune_action_head.yaml` (new)
3. `configs/task/robotwin_finetune_action_head.yaml` (new)
4. `src/fastwam/trainer.py` (modified)
5. `scripts/finetune.sh` (new)

All modifications are backward compatible.

## References

- Main README: [README.md](README.md)
- Paper: [Fast-WAM](https://arxiv.org/abs/2603.16666)
- Project Page: https://yuantianyuan01.github.io/FastWAM/
