#!/bin/bash

# Fine-tuning script for FastWAM
# This script demonstrates how to fine-tune only the action head

set -e

# Configuration
NUM_GPUS=${1:-1}
TASK=${2:-"libero_finetune_action_head"}

echo "==================================="
echo "FastWAM Fine-tuning Script"
echo "==================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Task: $TASK"
echo "==================================="

# Check if pretrained checkpoint exists
PRETRAINED_CKPT="./checkpoints/fastwam_release/libero_uncond_2cam224.pt"
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAINED_CKPT"
    echo "Please download the pretrained checkpoint first."
    echo "See README.md for instructions."
    exit 1
fi

# Run fine-tuning
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running on single GPU..."
    python scripts/train.py task=$TASK
else
    echo "Running on $NUM_GPUS GPUs with DeepSpeed ZeRO-1..."
    accelerate launch \
        --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
        --num_processes=$NUM_GPUS \
        scripts/train.py \
        task=$TASK
fi

echo "==================================="
echo "Fine-tuning completed!"
echo "==================================="
