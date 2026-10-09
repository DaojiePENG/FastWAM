#!/usr/bin/env bash

FASTWAM_ROOT="/dahuafs/userdata/2653853/code/FastWAM"
if [[ -d "/root/miniconda3" ]]; then
  CONDA_ROOT_2653853="/root/miniconda3"
else
  CONDA_ROOT_2653853="/home/myuser/miniconda3"
fi

source "${CONDA_ROOT_2653853}/etc/profile.d/conda.sh"
# 镜像瘦身 (10-09): 论文相关 envs 迁共享盘 .envs, 镜像不再内嵌副本;
# CONDA_ENVS_PATH 让按名激活能找到 .envs 下的环境 (本地/pod 通用, 不依赖 ~/.condarc)。
export CONDA_ENVS_PATH="/dahuafs/userdata/2653853/code/.envs${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
conda activate leapbot-va

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/c2-worldmodel/2653853/leapbot-ce/models"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
export HF_HOME="/mnt/c2-worldmodel/2653853/leapbot-ce/hf_cache"
export FASTWAM_RUNS_ROOT="/mnt/c2-worldmodel/2653853/leapbot-ce/runs"
export LIBERO_CONFIG_PATH="${FASTWAM_ROOT}/.libero"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}:/dahuafs/userdata/2653853/code/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
