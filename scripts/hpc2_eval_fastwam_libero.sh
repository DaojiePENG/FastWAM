#!/bin/bash

# ── SBATCH directives (ignored when run with `bash`, used by `sbatch`) ──
#SBATCH -J fastwam_libero_eval
#SBATCH -p i64m1tga800u
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/FastWAM/evaluate_results/fastwam_libero_eval_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/FastWAM/evaluate_results/fastwam_libero_eval_%j.err
#SBATCH -D ./

# =============================================================================
# HPC2 sbatch submission — FastWAM LIBERO evaluation
#
# Usage:
#   sbatch scripts/hpc2_eval_fastwam_libero.sh <CKPT> [NUM_GPUS] [MAX_TASKS_PER_GPU]
#
#   CKPT:              checkpoint .pt path (required)
#   NUM_GPUS:          number of GPUs (default: 1)
#   MAX_TASKS_PER_GPU: parallel tasks per GPU (default: 2; on 80GB A800,
#                      FastWAM uses ~22GB per process during init.
#                      2 tasks = ~44GB, leaving headroom for KV cache & inference.)
#
# Examples:
#   # 1 GPU, 2 tasks per GPU (safe default for 80GB)
#   sbatch scripts/hpc2_eval_fastwam_libero.sh \
#     ./checkpoints/fastwam_release/libero_uncond_2cam224.pt
#
#   # 2 GPUs, 2 tasks per GPU
#   sbatch --gres=gpu:2 scripts/hpc2_eval_fastwam_libero.sh \
#     runs/.../step_000200.pt 2 2
#
#   # Interactive debug (no sbatch):
#   bash scripts/hpc2_eval_fastwam_libero.sh \
#     ./checkpoints/fastwam_release/libero_uncond_2cam224.pt 1 2
# =============================================================================

# ── Parse CLI args (also used when running directly without sbatch) ──
CKPT="${1:?Error: checkpoint path required}"
NUM_GPUS="${2:-1}"
MAX_TASKS_PER_GPU="${3:-2}"

echo "============================================"
echo " FastWAM (LIBERO) HPC2 Evaluation Job"
echo "============================================"
echo "Checkpoint         : $CKPT"
echo "Num GPUs           : $NUM_GPUS"
echo "Max tasks per GPU  : $MAX_TASKS_PER_GPU"
echo "Job started at     : $(date)"
echo "============================================"

# ── Environment ──
module load cuda/12.8
module load ffmpeg
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam-libero

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run evaluation ──
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus="$NUM_GPUS" \
  MULTIRUN.max_tasks_per_gpu="$MAX_TASKS_PER_GPU"

echo "============================================"
echo " FastWAM (LIBERO) Evaluation finished at $(date)"
echo "============================================"
