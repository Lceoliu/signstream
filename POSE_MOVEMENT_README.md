# Pose Movement Visualization System

This system provides comprehensive visualization of pose movements through the preprocessing pipeline, showing how normalization, bounding box adjustments, and other transformations affect pose data.

## Overview

The pose movement visualizer helps you understand:
- **Preprocessing Impact**: How each step transforms pose coordinates
- **Movement Patterns**: Trajectory analysis for different body parts
- **Temporal Dynamics**: Movement speed and direction over time
- **Data Quality**: Effects of interpolation and smoothing

## Features

### 🎬 Video Generation
- High-quality MP4 videos using `av` library
- Frame-by-frame preprocessing comparisons
- Configurable frame rates and quality
- Fallback to GIF animations if needed

### 📊 Movement Analysis
- 2D trajectory plots with start/end markers
- Temporal coordinate analysis (X/Y over time)
- Movement speed calculations
- Body part-specific analysis

### 🔧 Preprocessing Visualization
- Step-by-step transformation display
- Original vs processed comparisons
- Bounding box normalization effects
- Smoothing and interpolation results

### 📐 Axes and Labels
- Proper coordinate system labeling
- Grid lines and reference points
- Normalized coordinate ranges
- Movement speed units

## Files Structure

```
signstream/visualization/
├── pose_movement_visualizer.py  # Core visualization class
run_pose_visualization.py        # Simple runner script
example_pose_movement.py         # Examples and demonstrations
POSE_MOVEMENT_README.md          # This documentation
```

## Quick Start

### Basic Usage
```bash
# Visualize a random sample
python run_pose_visualization.py

# Visualize specific sample
python run_pose_visualization.py --sample-idx 42

# Create video output
python run_pose_visualization.py --video --fps 20
```

### Full Options
```bash
python run_pose_visualization.py \
    --sample-idx 10 \
    --video \
    --fps 25 \
    --parts body left_hand right_hand face \
    --max-frames 200 \
    --output my_visualization_results
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `signstream/configs/default.yaml` | Configuration file |
| `--sample-idx` | int | Random | Specific sample to visualize |
| `--output` | str | `pose_movement_results` | Output directory |
| `--video` | flag | False | Create video output |
| `--fps` | int | 15 | Video frame rate |
| `--max-frames` | int | 150 | Maximum frames to process |
| `--parts` | list | `body left_hand right_hand` | Body parts to analyze |

## Dependencies

### Required
```bash
pip install numpy matplotlib seaborn pyyaml
```

### For Video Generation
```bash
pip install av opencv-python
```

### Check Dependencies
```bash
python example_pose_movement.py --check-deps
```

## Preprocessing Pipeline

The system visualizes these preprocessing steps:

### 1. Interpolation
- Fills missing keypoints using temporal interpolation
- Improves data continuity
- Reduces confidence-based gaps

### 2. Normalization
- Centers poses using reference points
- Scales based on body part dimensions
- Converts to standardized coordinate system

### 3. Bounding Box Normalization
- Calculates frame-wise bounding boxes
- Applies padding for context
- Normalizes to [-1, 1] range

### 4. Smoothing
- Applies temporal smoothing filter
- Reduces noise and jitter
- Preserves gesture dynamics

## Output Files

### Frame Comparisons
- `{video}_frame_{N}_comparison.png` - Side-by-side original vs processed
- Shows preprocessing effects at key frames
- Multiple sample frames per video

### Movement Analysis
- `{video}_{part}_original_movement.png` - Original trajectories
- `{video}_{part}_processed_movement.png` - Processed trajectories
- 4-panel analysis: trajectory, X/Y over time, speed

### Video Output
- `{video}_preprocessing_steps.mp4` - Video showing all steps
- `{video}_preprocessing_steps.gif` - Fallback animation
- Configurable frame rate and quality

### Summary Data
- `{video}_processing_summary.json` - Metadata and statistics
- Processing parameters and results
- Temporal and spatial metrics

## Visualization Components

### Pose Plotting
```python
class PoseMovementVisualizer:
    def _plot_pose_frame(self, pose, ax, title):
        # Color-coded body parts
        # Skeleton connections
        # Confidence-based filtering
        # Axis labels and grids
```

### Movement Analysis
```python
def create_movement_trajectory(self, poses, part_name):
    # 2D trajectory with start/end markers
    # X/Y coordinates over time
    # Movement speed calculation
    # Statistical analysis
```

### Video Generation
```python
def create_preprocessing_steps_video(self, ...):
    # Multi-step comparison layout
    # High-quality MP4 encoding
    # Frame-by-frame processing
    # Progress tracking
```

## Color Scheme

Body parts are color-coded for easy identification:
- **Face**: Red (`#FF6B6B`)
- **Left Hand**: Teal (`#4ECDC4`)
- **Right Hand**: Blue (`#45B7D1`)
- **Body**: Green (`#96CEB4`)
- **Full Body**: Yellow (`#FECA57`)

## Skeleton Connections

The visualizer draws anatomically correct connections:
- **Body**: COCO-style skeleton (torso, arms, legs)
- **Face**: Facial landmarks with contours
- **Hands**: Finger bone structures
- **Confidence-based**: Only high-confidence connections shown

## Usage Examples

### Research Analysis
```bash
# Analyze specific gesture types
python run_pose_visualization.py --sample-idx 123 --parts left_hand right_hand

# Create presentation videos
python run_pose_visualization.py --video --fps 30 --max-frames 100
```

### Development Debugging
```bash
# Check preprocessing effects
python run_pose_visualization.py --sample-idx 0 --parts body

# Quick data quality check
python run_pose_visualization.py --max-frames 50
```

### Batch Processing
```bash
# Process multiple samples
for i in {0..9}; do
    python run_pose_visualization.py --sample-idx $i --output "sample_$i"
done
```

## Advanced Features

### Custom Body Part Analysis
Focus on specific body parts for detailed movement analysis:
```bash
python run_pose_visualization.py --parts face  # Face-only analysis
python run_pose_visualization.py --parts left_hand right_hand  # Hands-only
```

### Video Quality Control
Adjust video parameters for different use cases:
```bash
# High quality for presentations
python run_pose_visualization.py --video --fps 30

# Quick preview
python run_pose_visualization.py --video --fps 10 --max-frames 50
```

### Movement Metrics
The system calculates:
- **Trajectory Length**: Total movement distance
- **Average Speed**: Mean movement velocity
- **Speed Variance**: Movement consistency
- **Displacement**: Start-to-end distance

## Troubleshooting

### Common Issues

#### "av library not found"
```bash
pip install av
# or
conda install av -c conda-forge
```

#### "Dataset not found"
The system automatically creates dummy data for testing. This is normal when the real dataset is unavailable.

#### Low video quality
Increase FPS and reduce max-frames:
```bash
python run_pose_visualization.py --video --fps 30 --max-frames 100
```

#### Memory issues
Reduce the number of frames:
```bash
python run_pose_visualization.py --max-frames 50
```

### Performance Tips

1. **Limit frames** for faster processing: `--max-frames 100`
2. **Reduce body parts** for simpler analysis: `--parts body`
3. **Skip video** for quick plots only
4. **Lower FPS** for smaller files: `--fps 10`

## Integration

### With Training Pipeline
Monitor pose quality during training:
```python
# In training script
if epoch % 10 == 0:
    subprocess.run([
        "python", "run_pose_visualization.py", 
        "--sample-idx", "0", 
        "--output", f"training_epoch_{epoch}"
    ])
```

### With Data Processing
Validate preprocessing steps:
```python
# After data preprocessing
for sample_idx in [0, 10, 20]:
    run_pose_visualization(sample_idx, output_dir=f"validation_{sample_idx}")
```

## Examples and Demonstrations

### View Examples
```bash
# Show all features
python example_pose_movement.py

# Show specific examples
python example_pose_movement.py --demo-pipeline
python example_pose_movement.py --show-usage
python example_pose_movement.py --sample-plot
```

### Create Sample Analysis
```bash
python example_pose_movement.py --sample-plot
```
This creates `sample_movement_analysis.png` showing the type of analysis the tool provides.

## Citation

If you use this visualization system in your research, please cite the SignStream-RVQ project and acknowledge the pose movement analysis capabilities.