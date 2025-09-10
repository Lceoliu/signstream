# ACTUAL Pose Movement Visualization

This is the **CORRECTED** pose movement visualization system that matches the actual data pipeline in `signstream/data/datasets.py` and `signstream/data/transforms.py`.

## What This Shows

This visualization shows poses **exactly as they are processed** by the real data pipeline:

1. **Load pose sequence** from `.npy` files (shape: `[T, 134, 3]` or `[T, 133, 3]`)
2. **Global bbox normalization** using `normalize_by_global_bbox()` 
3. **Interpolate low confidence** points using `interpolate_low_confidence_linear()`
4. **Compute velocity** using `compute_velocity()`
5. **Split body parts** using `split_body_parts()` with part-wise bbox normalization
6. **Chunk sequences** into fixed-length segments with pose+velocity

## Key Differences from Previous Version

- ✅ Uses actual `process_all()` function from `transforms.py`
- ✅ Shows real body part intervals from `BODY_PARTS_INTERVALS`
- ✅ Uses actual COCO edges from `COCO_EDGES` 
- ✅ Visualizes chunked data structure: `[N, L, K, 5]` (pose+velocity)
- ✅ Matches actual bbox normalization ranges (0-8)
- ✅ Shows data exactly as seen by the RVQ model

## Files

- `signstream/visualization/pose_movement_visualizer_fixed.py` - Corrected visualizer
- `run_actual_pose_visualization.py` - Simple runner script
- `ACTUAL_POSE_VISUALIZATION_README.md` - This documentation

## Quick Start

```bash
# Basic usage - shows one random sample
python run_actual_pose_visualization.py

# Specific sample with detailed analysis
python run_actual_pose_visualization.py --sample-idx 42 --analyze-chunks

# Create video showing the actual pipeline
python run_actual_pose_visualization.py --video --fps 20

# Full analysis
python run_actual_pose_visualization.py \
    --sample-idx 10 \
    --video \
    --analyze-chunks \
    --output detailed_actual_analysis
```

## Data Pipeline Visualization

The system shows the actual processing steps:

### 1. Original Pose Loading
- Loads `.npy` files with shape `[T, 134, 3]` (frame size + 133 keypoints)
- Or `[T, 133, 3]` (just keypoints)
- Each keypoint: `[x, y, confidence]`

### 2. Global Processing (`process_all()`)
```python
# From transforms.py - this is what actually happens:
pose_norm, _ = normalize_by_global_bbox(
    pose, bbox_range=(8.0, 8.0), conf_threshold=0.25, keep_aspect=False
)
pose_interp, fill_mask = interpolate_low_confidence_linear(
    pose_norm, confidence_threshold=0.25, max_search=10
)
velocity = compute_velocity(pose_interp, dt=1.0/fps)
parts = split_body_parts(pose_interp, partial_bbox=True)
```

### 3. Body Part Processing
Each body part gets:
- Extracted using `BODY_PARTS_INTERVALS`
- Individual bbox normalization 
- Combined with velocity: `[T, K, 5]` (x, y, conf, vx, vy)

### 4. Chunking
```python
# Final format for RVQ model:
chunked = [N_chunks, chunk_len, keypoints, 5]
# Where 5 = [x, y, confidence, velocity_x, velocity_y]
```

## Body Parts Structure

Uses actual intervals from `transforms.py`:
```python
BODY_PARTS_INTERVALS = {
    'face': (23, 91),        # 68 face keypoints  
    'left_hand': (91, 112),  # 21 left hand keypoints
    'right_hand': (112, 133), # 21 right hand keypoints
    'body': (0, 17),         # 17 body keypoints
    'full_body': (0, 133),   # All 133 keypoints
}
```

## Skeleton Connections

Uses actual `COCO_EDGES` from `transforms.py` for anatomically correct connections.

## Output Files

### Chunk Analysis
- `{video}_chunk_{N}_visualization.png` - Shows chunked data structure

### Movement Analysis  
- `{video}_{part}_movement_analysis.png` - Trajectory analysis per body part

### Video Output
- `{video}_actual_pipeline.mp4` - Video showing actual processing pipeline

### Summary
- `{video}_actual_pipeline_summary.json` - Processing metadata and chunk structure

## Sample Output Structure

```json
{
  "chunk_structure": {
    "body": [8, 16, 17, 5],      // [chunks, length, keypoints, features]
    "face": [8, 16, 68, 5], 
    "left_hand": [8, 16, 21, 5],
    "right_hand": [8, 16, 21, 5],
    "full_body": [8, 16, 133, 5]
  },
  "processing_notes": [
    "Data processed through actual pipeline in datasets.py",
    "Uses process_all() from transforms.py",
    "Final format: [N_chunks, chunk_len, keypoints, 5] with pose+velocity"
  ]
}
```

## Command Line Options

| Argument | Description |
|----------|-------------|
| `--sample-idx N` | Specific sample index (random if not specified) |
| `--analyze-chunks` | Show detailed chunk structure analysis |
| `--video` | Create video showing pipeline (requires `av` library) |
| `--fps N` | Video frame rate (default: 15) |
| `--output DIR` | Output directory (default: `actual_pose_results`) |

## Dependencies

```bash
# Required
pip install torch numpy matplotlib seaborn pyyaml

# For video output
pip install av opencv-python
```

## Verification

To verify this matches your actual data:

1. Run a sample and check the chunk structure
2. Compare with your RVQ model input expectations
3. Verify bbox ranges match (0-8 for normalized coordinates)
4. Check that body part splits match `BODY_PARTS_INTERVALS`

## Example Usage

```bash
# Quick verification
python run_actual_pose_visualization.py --sample-idx 0 --analyze-chunks

# Create presentation video
python run_actual_pose_visualization.py --video --fps 30

# Debug specific sample
python run_actual_pose_visualization.py --sample-idx 123 --analyze-chunks --video
```

This corrected version shows the **actual** data your RVQ model sees, not a hypothetical preprocessing pipeline!