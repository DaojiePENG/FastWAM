# FastWAM Evaluation Guide (EN)

Quick reference guide for evaluating FastWAM models on LIBERO and RoboTwin benchmarks.

> **中文版本**: 详细的中文评测指南请参考 [EVALUATION.md](EVALUATION.md)

---

## Quick Start

### Checkpoint Locations

**Pretrained models:**
```bash
checkpoints/fastwam_release/libero_uncond_2cam224.pt         # LIBERO
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt      # RoboTwin
```

**Finetuned models:**
```bash
runs/finetune/<task>/<timestamp>/checkpoints/weights/step_*.pt
```

### Fast Validation (Single Task, 10 Trials)

**LIBERO:**
```bash
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/libero_action_head/<timestamp>/checkpoints/weights/step_002000.pt \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  gpu_id=0
```

**RoboTwin:**
```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_*.pt \
  EVALUATION.task_name=pick_and_place \
  EVALUATION.num_trials=10 \
  gpu_id=0
```

---

## LIBERO Evaluation

### Full Benchmark (40 Tasks × 50 Trials)

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/libero_action_head/<timestamp>/checkpoints/weights/step_002000.pt
```

**Default config:**
- 4 task suites: libero_10, libero_goal, libero_spatial, libero_object
- 8 GPUs, 2 tasks per GPU
- 50 trials per task

### Evaluate Specific Suite

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  MULTIRUN.task_suite_names=[libero_10] \
  MULTIRUN.num_gpus=2
```

### View Results

```bash
python experiments/libero/summarize_results.py \
  --result_dir ./evaluate_results/libero/libero_uncond_2cam224_1e-4/<timestamp>
```

---

## RoboTwin Evaluation

### Full Benchmark

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/robotwin_action_head/<timestamp>/checkpoints/weights/step_*.pt
```

### Single Task

```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/finetune/.../step_*.pt \
  EVALUATION.task_name=pick_and_place \
  EVALUATION.num_trials=50 \
  gpu_id=0
```

**Available tasks:**
- `pick_and_place`
- `bimanual_assembly`
- `handover`
- `coordination_task`

---

## Compare Pretrained vs Finetuned

```bash
# Evaluate pretrained
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.output_dir=./evaluate_results/libero/pretrained

# Evaluate finetuned
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.output_dir=./evaluate_results/libero/finetuned

# Compare
python experiments/libero/summarize_results.py --result_dir ./evaluate_results/libero/pretrained
python experiments/libero/summarize_results.py --result_dir ./evaluate_results/libero/finetuned
```

---

## Key Parameters

```yaml
EVALUATION:
  num_trials: 50              # Trials per task
  num_inference_steps: 50     # Diffusion inference steps
  text_cfg_scale: 1.0        # Text CFG strength
  action_cfg_scale: 1.0      # Action CFG strength
  replan_steps: 10           # Replan interval (LIBERO)
                             # 15 for RoboTwin
  
MULTIRUN:
  num_gpus: 8                 # Number of GPUs
  max_tasks_per_gpu: 2        # Tasks per GPU (1 for RoboTwin)
```

---

## Estimated Time

**LIBERO (Full benchmark: 40 tasks × 50 trials)**
- 8 GPUs: ~2-3 hours
- 1 GPU: ~16-24 hours

**LIBERO (Single suite: 10 tasks × 50 trials)**
- 2 GPUs: ~30-45 minutes
- 1 GPU: ~4-6 hours

**Quick validation (10 trials)**
- Any config: 5-15 minutes

---

## Troubleshooting

**Evaluation stuck?**
```bash
# Debug with single task
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/finetune/.../step_002000.pt \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  gpu_id=0

# Check GPU memory
nvidia-smi

# Reduce parallel tasks
MULTIRUN.max_tasks_per_gpu=1
```

**Lower success rate after finetuning?**
- Try earlier checkpoints
- Check learning rate (may be too high)
- Try different freeze strategy
- Verify dataset statistics are correct

---

## References

- **Detailed Guide**: [EVALUATION.md](EVALUATION.md) (Chinese)
- **Finetuning Guide**: [FINETUNING.md](FINETUNING.md)
- **Main README**: [README.md](README.md)
- **Paper**: [Fast-WAM on arXiv](https://arxiv.org/abs/2603.16666)
- **Project Page**: https://yuantianyuan01.github.io/FastWAM/

---

**Good luck with your evaluation!** 🚀
