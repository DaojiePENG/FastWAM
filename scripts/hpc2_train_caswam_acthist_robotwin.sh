#!/bin/bash

#SBATCH -J caswam_acthist_robotwin
#SBATCH -p i64m1tga800u
#SBATCH -c 32
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=168:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/FastWAM/runs/caswam_acthist_robotwin_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/FastWAM/runs/caswam_acthist_robotwin_%j.err
#SBATCH -D ./

# ==================== 环境配置 ====================
module load cuda/12.8
module load ffmpeg
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fastwam

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "CasWAM-ActHist (RoboTwin) Training Job"
echo "Task: robotwin_caswam_acthist_uncond_3cam_384_1e-4"
echo "Joint video+action full-history KV cache + temporal PE + history cross-attention"
echo "Started at: $(date)"
echo "=============================================="

# ==================== 执行训练命令 ====================
bash scripts/train_ddp.sh 4 task=robotwin_caswam_acthist_uncond_3cam_384_2e-4

echo "=============================================="
echo "CasWAM-ActHist (RoboTwin) Training Job finished at $(date)"
echo "=============================================="
