#!/bin/bash

#SBATCH -J precompute_text_embeds
#SBATCH -p i64m1tga800u
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/FastWAM/temp/precompute_text_embeds%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/FastWAM/temp/precompute_text_embeds%j.err
#SBATCH -D ./

# ==================== 环境配置 ====================
module load cuda/12.8
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fastwam

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Precompute Text Embeds"
echo "Started at: $(date)"
echo "=============================================="

# ==================== 执行训练命令 ====================
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4

echo "=============================================="
echo "Precompute Text Embeds finished at $(date)"
echo "=============================================="
