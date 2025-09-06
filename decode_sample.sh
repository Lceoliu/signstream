#!/bin/bash
# Decode and visualize a sample for SignStream-RVQ

set -e

echo "Starting SignStream-RVQ sample decoding and visualization..."

# Configuration
CONFIG_PATH="signstream/configs/default.yaml"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"checkpoints/best_model.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"visualization_output"}
SAMPLE_ID=${SAMPLE_ID:-0}

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please provide a valid checkpoint path via CHECKPOINT_PATH environment variable"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate $CONDA_ENV
fi

echo "Decoding sample $SAMPLE_ID..."

# First export tokens for the sample
python -m signstream.inference.export_tokens \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --split val \
    --num-samples 1 \
    --out $OUTPUT_DIR/sample_tokens.jsonl

# Create visualization report  
python << EOF
import yaml
import sys
sys.path.append('.')

from signstream.inference.visualizer import create_visualization_report

try:
    create_visualization_report(
        config_path="$CONFIG_PATH",
        checkpoint_path="$CHECKPOINT_PATH", 
        token_file="$OUTPUT_DIR/sample_tokens.jsonl",
        output_dir="$OUTPUT_DIR"
    )
    print("Visualization report created successfully!")
except Exception as e:
    print(f"Error creating visualization: {e}")
    print("This may be due to missing optional dependencies (matplotlib, seaborn, sklearn)")
EOF

echo "Sample decoding completed!"
echo "Check the output directory: $OUTPUT_DIR"
echo "Files created:"
find $OUTPUT_DIR -type f -name "*.png" -o -name "*.json*" | head -10