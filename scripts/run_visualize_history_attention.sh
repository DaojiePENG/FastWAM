#!/bin/bash
#SBATCH -J attn_analysis
#SBATCH -p i64m1tga800u
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/FastWAM/temp/attn_analysis_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/FastWAM/temp/attn_analysis_%j.err
#SBATCH -D ./

module load cuda/12.8
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam-libero

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

TASK_SUITE=libero_spatial
TASK_IDX=0
# ── Step 1: Capture rollout with attention + video predictions (GPU) ──
# Runs full episodes (until done or max_steps), repeated --num_trials times.
python scripts/capture_history_attention_rollout.py \
  --ckpt runs/libero_caswam_acthist_uncond_2cam224_1e-4/2026-06-08_12-57-00/checkpoints/weights/step_006600.pt \
  --task_suite ${TASK_SUITE} \
  --task_idx ${TASK_IDX} \
  --num_trials 4

# ── Step 2: Run analysis (CPU) on each trial ──
LATEST_BASE=$(ls -dt evaluate_results/attention_analysis/libero_caswam_acthist_uncond_2cam224_1e-4/${TASK_SUITE}/${TASK_IDX}/ | head -1)
echo "Analyzing trials in: ${LATEST_BASE}"

for TRIAL_DIR in "${LATEST_BASE}"trial_*/; do
  echo ""
  echo "Analyzing: ${TRIAL_DIR}"
  python scripts/analyze_history_attention.py \
    --data_dir "${TRIAL_DIR}" \
    --top_n 5
done
