#!/bin/bash
# Export tokens script for SignStream-RVQ

set -e

echo "Starting SignStream-RVQ token export..."

# Configuration
CONFIG_PATH="signstream/configs/default.yaml"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/data/data_lc/slr_rvq/signstream/checkpoints/Baseline-Test002_20250909_084115/checkpoint_epoch_165.pt"}
OUTPUT_PATH=${OUTPUT_PATH:-"exports/tokens.jsonl"}
NUM_SAMPLES=${NUM_SAMPLES:-50}
SPLIT=${SPLIT:-"val"}

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please provide a valid checkpoint path via CHECKPOINT_PATH environment variable"
    exit 1
fi

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate $CONDA_ENV
fi

# Export tokens
python -m signstream.inference.export_tokens \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --split $SPLIT \
    --num-samples $NUM_SAMPLES \
    --out $OUTPUT_PATH \
    --body-parts face left_hand right_hand body full_body

echo "Token export completed!"
echo "Output saved to: $OUTPUT_PATH"