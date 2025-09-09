#!/bin/bash
# Single GPU training script for SignStream-RVQ

set -e

echo "Starting SignStream-RVQ training..."

# Configuration
CONFIG_PATH="signstream/configs/default.yaml"
EXPERIMENT_NAME="rvq_single_gpu"
DEVICE="cuda:1"

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "CUDA not available, falling back to CPU"
    DEVICE="cpu"
fi

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate $CONDA_ENV
fi

# Run training
python -m signstream.training.train_rvq \
    --config $CONFIG_PATH \
    --device $DEVICE

echo "Training completed!"