#!/bin/bash

# FastWAM Fine-tuning Script (NOHUP + PID FILE)
# Training survives SSH disconnect | PID saved for easy killing

set -e

# Configuration
NUM_GPUS=${1:-1}
TASK=${2:-"libero_finetune_action_head"}
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="training_pid.txt"  # PID will be saved here

echo "==================================="
echo "FastWAM Fine-tuning (NOHUP + PID MODE)"
echo "==================================="
echo "GPUs:          $NUM_GPUS"
echo "Task:          $TASK"
echo "Log file:      $LOG_FILE"
echo "PID file:      $PID_FILE"
echo "==================================="

# Check if pretrained checkpoint exists
PRETRAINED_CKPT="./checkpoints/fastwam_release/libero_uncond_2cam224.pt"
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found at $PRETRAINED_CKPT"
    echo "Please download the pretrained checkpoint first."
    echo "See README.md for instructions."
    exit 1
fi

# Start training in background with nohup
echo "Starting training..."
echo "You can close SSH safely. Training will continue."

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running on single GPU..."
    nohup python scripts/train.py task=$TASK > "$LOG_FILE" 2>&1 &
else
    echo "Running on $NUM_GPUS GPUs with DeepSpeed ZeRO-1..."
    nohup accelerate launch \
        --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
        --num_processes=$NUM_GPUS \
        scripts/train.py \
        task=$TASK > "$LOG_FILE" 2>&1 &
fi

# SAVE PID TO FILE
echo $! > "$PID_FILE"

echo "==================================="
echo "TRAINING STARTED!"
echo "PID saved to: $PID_FILE"
echo "PID: $(cat $PID_FILE)"
echo ""
echo "TO MONITOR LOGS:  tail -f $LOG_FILE"
echo "TO STOP TRAINING: kill \$(cat $PID_FILE)"
echo "==================================="