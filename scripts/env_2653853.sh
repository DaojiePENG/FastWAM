#!/usr/bin/env bash

FASTWAM_ROOT="/dahuafs/userdata/2653853/code/FastWAM"
if [[ -d "/root/miniconda3" ]]; then
  CONDA_ROOT_2653853="/root/miniconda3"
else
  CONDA_ROOT_2653853="/home/myuser/miniconda3"
fi

source "${CONDA_ROOT_2653853}/etc/profile.d/conda.sh"
conda activate leapbot-va

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/c2-worldmodel/2653853/leapbot-ce/models"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HOME="/mnt/c2-worldmodel/2653853/leapbot-ce/hf_cache"
export FASTWAM_RUNS_ROOT="/mnt/c2-worldmodel/2653853/leapbot-ce/runs"
export LIBERO_CONFIG_PATH="${FASTWAM_ROOT}/.libero"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}:/dahuafs/userdata/2653853/code/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
