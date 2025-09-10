# RVQ Pose Reconstruction Visualization

This directory contains comprehensive tools for visualizing RVQ model reconstruction quality, comparing original poses with decoded poses, and analyzing token export.

## Overview

The visualization system provides:
- **Token Export**: Extract RVQ tokens from video samples
- **Pose Reconstruction**: Decode tokens back to pose sequences
- **Visual Comparison**: Side-by-side comparison of original vs decoded poses
- **Error Analysis**: Quantitative reconstruction quality metrics
- **Animation Generation**: Temporal comparison videos

## Files

### Core Scripts
- `signstream/inference/visualize_reconstruction.py` - Main visualization script
- `run_visualization.py` - Simple runner script
- `example_visualization.py` - Example usage and demonstrations

### Usage Examples

#### 1. Quick Start with Runner Script
```bash
# Basic usage with a checkpoint
python run_visualization.py --checkpoint path/to/your/checkpoint.pt

# With custom options
python run_visualization.py \
    --checkpoint path/to/checkpoint.pt \
    --config signstream/configs/default.yaml \
    --output my_results \
    --samples 10 \
    --animations
```

#### 2. Direct Script Usage
```bash
python signstream/inference/visualize_reconstruction.py \
    --config signstream/configs/default.yaml \
    --checkpoint path/to/checkpoint.pt \
    --split val \
    --num-samples 5 \
    --body-parts face left_hand right_hand body \
    --output-dir visualization_output \
    --save-animations \
    --max-frames 100
```

#### 3. Example and Demo
```bash
# Show token format demonstration
python example_visualization.py --demo-tokens

# Run example visualization (auto-finds checkpoints)
python example_visualization.py --run-viz

# Both
python example_visualization.py
```

## Command Line Arguments

### Main Visualization Script

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | required | Path to model configuration file |
| `--checkpoint` | str | required | Path to trained model checkpoint |
| `--split` | str | "val" | Dataset split (train/val/test) |
| `--num-samples` | int | 5 | Number of videos to sample |
| `--body-parts` | list | ["face", "left_hand", "right_hand", "body"] | Body parts to process |
| `--output-dir` | str | "visualization_output" | Output directory |
| `--device` | str | auto | Device (cuda/cpu) |
| `--save-animations` | flag | False | Save comparison animations |
| `--max-frames` | int | 100 | Max frames per video |

### Runner Script

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | required | Path to checkpoint |
| `--config` | str | "signstream/configs/default.yaml" | Config file |
| `--output` | str | "visualization_results" | Output directory |
| `--samples` | int | 5 | Number of samples |
| `--animations` | flag | False | Save animations |

## Output Files

The visualization generates several types of output:

### Per-Video Files
- `{video_name}_tokens.json` - Exported token data
- `{video_name}_error_analysis.png` - Reconstruction error analysis plots
- `{video_name}_sample_frames.png` - Sample frame comparisons
- `{video_name}_comparison.gif` - Animation (if enabled)

### Summary Files
- `reconstruction_summary.json` - Overall metrics and results

## Token Format

The system exports tokens in a structured JSON format:

```json
{
  "video_name": "sample_video_001",
  "fps": 25,
  "chunk_len": 16,
  "text": "Hello world",
  "gloss": "HELLO WORLD",
  "tokens": {
    "face": [
      {"t": 0, "F": [245, 156, 89]},
      {"t": 1, "F": [201, 134, 67]}
    ],
    "left_hand": [
      {"t": 0, "LH": [12, 234, 156]},
      {"t": 1, "LH": [45, 198, 123]}
    ]
  },
  "templates": [
    "<T0><F:c245><LH:a12><RH:b89><B:d301>",
    "<T1><F:c201><LH:a45><RH:b123><B:d278>"
  ]
}
```

### Token Structure
- **t**: Temporal chunk index
- **Body part codes**: 3-level RVQ codes [level0, level1, level2]
- **Templates**: Human-readable token sequences
- **Code spaces**: Different prefixes for each body part
  - `c`: Face tokens
  - `a`: Left hand tokens
  - `b`: Right hand tokens
  - `d`: Body tokens
  - `e`: Full body tokens

## Visualization Types

### 1. Error Analysis Plots
Four-panel analysis showing:
- Reconstruction error over time
- Error distribution histogram
- Per-body-part error comparison
- Sample frame error progression

### 2. Sample Frame Comparisons
Grid of sample frames showing original (faded) and decoded (solid) poses overlaid.

### 3. Comparison Animations
Side-by-side animated comparison of original vs decoded pose sequences.

## Requirements

### Dependencies
```bash
# Core dependencies
torch
numpy
matplotlib
seaborn
pyyaml

# Animation support (optional)
pillow  # For GIF animations
```

### Model Requirements
- Trained RVQ model checkpoint
- Compatible configuration file
- Dataset (or dummy data will be generated)

## Example Workflow

1. **Train your RVQ model** (or use existing checkpoint)
2. **Run visualization**:
   ```bash
   python run_visualization.py --checkpoint path/to/checkpoint.pt --samples 5
   ```
3. **Analyze results**:
   - Check error analysis plots for reconstruction quality
   - Review token export data
   - Examine sample frame comparisons
4. **Iterate**: Adjust model architecture or training based on findings

## Troubleshooting

### Common Issues

#### "Dataset not found"
The script will automatically create dummy data if the real dataset is unavailable. This is normal for testing.

#### CUDA memory errors
- Reduce `--num-samples` and `--max-frames`
- Use `--device cpu` to run on CPU
- The script includes automatic memory management

#### Missing checkpoint
Ensure the checkpoint file exists and is a valid PyTorch checkpoint with either `model_state_dict` key or direct state dict.

### Performance Tips

1. **Reduce samples** for quick testing: `--num-samples 3`
2. **Limit frames** for faster processing: `--max-frames 50`
3. **Skip animations** for faster runs (large files)
4. **Use CPU** if GPU memory is limited

## Advanced Usage

### Custom Body Part Analysis
```bash
# Focus on hands only
python run_visualization.py \
    --checkpoint checkpoint.pt \
    --body-parts left_hand right_hand
```

### Batch Processing
```bash
# Process multiple checkpoints
for ckpt in checkpoints/*.pt; do
    python run_visualization.py --checkpoint "$ckpt" --output "results_$(basename $ckpt .pt)"
done
```

### Integration with Training
The visualization can be integrated into training scripts to monitor reconstruction quality during training.

## Citation

If you use this visualization tool in your research, please cite the SignStream-RVQ project.