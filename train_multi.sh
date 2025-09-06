#!/bin/bash
# Multi-GPU DDP training script for SignStream-RVQ

set -e

echo "Starting SignStream-RVQ multi-GPU training..."

# Configuration
CONFIG_PATH="signstream/configs/default.yaml"
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Set CUDA_VISIBLE_DEVICES if not set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate $CONDA_ENV
fi

# Run DDP training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    signstream/training/train_rvq_multiple_gpu.py \
    --config $CONFIG_PATH

echo "Multi-GPU training completed!"