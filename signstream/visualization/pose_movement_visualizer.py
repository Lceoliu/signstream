"""
Pose movement visualization script with preprocessing steps.
Shows pose transformations through normalization, bounding box adjustments, and other preprocessing.
Uses av library for high-quality video generation with axes and labels.
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import av
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import json
from datetime import datetime
import gc

from signstream.data.datasets import CSLDailyDataset
from signstream.data.transforms import normalize_pose_sequence, interpolate_missing_keypoints


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')


class PoseMovementVisualizer:
    """Visualize pose movements with preprocessing transformations."""
    
    def __init__(self, body_part_indices: Dict[str, Tuple[int, int]], 
                 center_indices: Dict[str, int]):
        self.body_part_indices = body_part_indices
        self.center_indices = center_indices
        
        # Define colors for different body parts
        self.colors = {
            'face': '#FF6B6B',      # Red
            'left_hand': '#4ECDC4',  # Teal  
            'right_hand': '#45B7D1', # Blue
            'body': '#96CEB4',       # Green
            'full_body': '#FECA57'   # Yellow
        }
        
        # Define connection patterns for skeleton visualization
        self.connections = {
            'body': [
                # Torso connections (COCO format)
                (5, 6), (5, 7), (7, 9),    # Left arm
                (6, 8), (8, 10),           # Right arm
                (5, 11), (6, 12),          # Shoulders to hips
                (11, 12),                  # Hip connection
                (11, 13), (13, 15),        # Left leg
                (12, 14), (14, 16),        # Right leg
            ],
            'face': [
                # Face outline connections (simplified)
                # Jaw line
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
                # Eyebrows
                (17, 18), (18, 19), (19, 20), (20, 21),  # Left eyebrow
                (22, 23), (23, 24), (24, 25), (25, 26),  # Right eyebrow
                # Eyes
                (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # Left eye
                (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # Right eye
                # Nose
                (27, 28), (28, 29), (29, 30),  # Nose bridge
                (31, 32), (32, 33), (33, 34), (34, 35),  # Nose base
                # Mouth
                (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
                (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
            ],
            'left_hand': [
                # Hand skeleton connections
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            ],
            'right_hand': [
                # Same as left hand
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            ]
        }
    
    def create_preprocessing_comparison(self, original_poses: np.ndarray, 
                                     processed_poses: np.ndarray,
                                     frame_idx: int,
                                     title: str = "") -> plt.Figure:
        """Create side-by-side comparison of original vs processed poses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{title} - Frame {frame_idx}', fontsize=16, fontweight='bold')
        
        # Plot original poses
        self._plot_pose_frame(original_poses[frame_idx], ax1, "Original Pose")
        
        # Plot processed poses  
        self._plot_pose_frame(processed_poses[frame_idx], ax2, "Processed Pose")
        
        plt.tight_layout()
        return fig
    
    def _plot_pose_frame(self, pose: np.ndarray, ax: plt.Axes, title: str) -> None:
        """Plot a single pose frame with skeleton connections."""
        ax.clear()
        
        # Extract coordinates
        if pose.shape[1] == 3:
            x, y = pose[:, 0], pose[:, 1]
            confidence = pose[:, 2]
        else:
            x, y = pose[:, 0], pose[:, 1]
            confidence = np.ones(len(pose))
        
        # Plot each body part
        for part_name, (start_idx, end_idx) in self.body_part_indices.items():
            if part_name == 'full_body':
                continue
                
            # Adjust indices (datasets might be 1-indexed)
            start_idx = max(0, start_idx - 1) if start_idx > 0 else start_idx
            end_idx = min(len(pose), end_idx)
            
            if start_idx < len(pose) and end_idx <= len(pose):
                part_x = x[start_idx:end_idx]
                part_y = y[start_idx:end_idx]
                part_conf = confidence[start_idx:end_idx]
                
                # Plot skeleton connections first
                if part_name in self.connections:
                    connections = self.connections[part_name]
                    for conn_start, conn_end in connections:
                        if (conn_start < len(part_x) and conn_end < len(part_x) and 
                            part_conf[conn_start] > 0.3 and part_conf[conn_end] > 0.3):
                            ax.plot([part_x[conn_start], part_x[conn_end]], 
                                   [part_y[conn_start], part_y[conn_end]], 
                                   color=self.colors.get(part_name, '#888888'), 
                                   alpha=0.6, linewidth=1.5)
                
                # Plot keypoints
                valid_points = part_conf > 0.3
                if np.any(valid_points):
                    ax.scatter(part_x[valid_points], part_y[valid_points], 
                             c=self.colors.get(part_name, '#888888'),
                             label=part_name.replace('_', ' ').title(),
                             s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Add center point if available
        if 'body' in self.center_indices:
            center_idx = self.center_indices['body'] - 1  # Adjust for 0-indexing
            if center_idx < len(pose):
                ax.scatter(x[center_idx], y[center_idx], 
                          c='red', s=100, marker='*', 
                          label='Center Point', edgecolors='white', linewidth=1)
        
        # Set axis properties
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add axis labels with units
        ax.set_xlabel('X Coordinate (normalized)', fontsize=12)
        ax.set_ylabel('Y Coordinate (normalized)', fontsize=12)
        
        # Add coordinate annotations
        ax.text(0.02, 0.98, f'Range: [{ax.get_xlim()[0]:.1f}, {ax.get_xlim()[1]:.1f}]', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_movement_trajectory(self, poses: np.ndarray, part_name: str,
                                 title: str = "") -> plt.Figure:
        """Create trajectory plot showing movement over time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} - {part_name.replace("_", " ").title()} Movement Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Get part indices
        start_idx, end_idx = self.body_part_indices[part_name]
        start_idx = max(0, start_idx - 1) if start_idx > 0 else start_idx
        end_idx = min(poses.shape[1], end_idx)
        
        if start_idx >= poses.shape[1] or end_idx > poses.shape[1]:
            return fig
        
        part_poses = poses[:, start_idx:end_idx, :]
        
        # Calculate center of mass for the part
        if part_poses.shape[2] >= 3:
            confidence = part_poses[:, :, 2]
            valid_mask = confidence > 0.3
            
            # Center of mass trajectory
            com_x = np.array([np.mean(frame[:, 0][valid_mask[i]]) if np.any(valid_mask[i]) else 0 
                             for i, frame in enumerate(part_poses)])
            com_y = np.array([np.mean(frame[:, 1][valid_mask[i]]) if np.any(valid_mask[i]) else 0 
                             for i, frame in enumerate(part_poses)])
        else:
            com_x = np.mean(part_poses[:, :, 0], axis=1)
            com_y = np.mean(part_poses[:, :, 1], axis=1)
        
        # 1. Trajectory in 2D space
        ax1.plot(com_x, com_y, color=self.colors.get(part_name, '#888888'), 
                linewidth=2, alpha=0.7, label='Trajectory')
        ax1.scatter(com_x[0], com_y[0], color='green', s=100, marker='o', 
                   label='Start', zorder=5)
        ax1.scatter(com_x[-1], com_y[-1], color='red', s=100, marker='s', 
                   label='End', zorder=5)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('2D Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. X coordinate over time
        frames = np.arange(len(com_x))
        ax2.plot(frames, com_x, color=self.colors.get(part_name, '#888888'), 
                linewidth=2, label='X Position')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('X Coordinate')
        ax2.set_title('X Movement Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Y coordinate over time
        ax3.plot(frames, com_y, color=self.colors.get(part_name, '#888888'), 
                linewidth=2, label='Y Position')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Y Movement Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Movement speed
        dx = np.diff(com_x)
        dy = np.diff(com_y)
        speed = np.sqrt(dx**2 + dy**2)
        ax4.plot(frames[1:], speed, color=self.colors.get(part_name, '#888888'), 
                linewidth=2, label='Movement Speed')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Speed (units/frame)')
        ax4.set_title('Movement Speed Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def create_preprocessing_steps_video(self, original_poses: np.ndarray,
                                       intermediate_steps: List[Tuple[str, np.ndarray]],
                                       video_name: str,
                                       output_path: Path,
                                       fps: int = 15) -> None:
        """Create video showing preprocessing steps using av library."""
        logger.info(f"Creating preprocessing steps video: {output_path}")
        
        # Prepare video writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary frames
        temp_frames = []
        num_frames = min(len(original_poses), 100)  # Limit to first 100 frames
        
        for frame_idx in range(0, num_frames, 2):  # Every 2nd frame for speed
            # Create comparison figure
            fig = plt.figure(figsize=(20, 12))
            
            # Calculate grid layout
            num_steps = len(intermediate_steps) + 1  # +1 for original
            cols = min(3, num_steps)
            rows = (num_steps + cols - 1) // cols
            
            # Plot original
            ax = plt.subplot(rows, cols, 1)
            self._plot_pose_frame(original_poses[frame_idx], ax, "Original")
            
            # Plot each preprocessing step
            for i, (step_name, step_poses) in enumerate(intermediate_steps):
                if frame_idx < len(step_poses):
                    ax = plt.subplot(rows, cols, i + 2)
                    self._plot_pose_frame(step_poses[frame_idx], ax, step_name)
            
            plt.suptitle(f'{video_name} - Frame {frame_idx} - Preprocessing Pipeline', 
                        fontsize=20, fontweight='bold')
            plt.tight_layout()
            
            # Convert to image array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            temp_frames.append(img_array)
            plt.close(fig)
            
            if (frame_idx // 2) % 10 == 0:
                logger.info(f"Processed frame {frame_idx}/{num_frames}")
        
        # Write video using av
        try:
            container = av.open(str(output_path), mode='w')
            stream = container.add_stream('libx264', rate=fps)
            stream.height = temp_frames[0].shape[0]
            stream.width = temp_frames[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            
            for i, frame_array in enumerate(temp_frames):
                # Convert RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                # Convert back to RGB for av
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)
            
            # Flush remaining packets
            packet = stream.encode()
            container.mux(packet)
            container.close()
            
            logger.info(f"Video saved: {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating video with av: {e}")
            logger.info("Falling back to matplotlib animation...")
            
            # Fallback to matplotlib animation
            fig, ax = plt.subplots(figsize=(12, 8))
            
            def animate(frame):
                ax.clear()
                if frame < len(original_poses):
                    self._plot_pose_frame(original_poses[frame], ax, f"Frame {frame}")
                return ax,
            
            anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=False)
            fallback_path = output_path.with_suffix('.gif')
            anim.save(str(fallback_path), writer='pillow', fps=fps)
            plt.close(fig)
            logger.info(f"Fallback animation saved: {fallback_path}")
        
        # Clean up temporary frames
        del temp_frames
        gc.collect()


def apply_preprocessing_steps(pose_sequence: np.ndarray, 
                            config: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """Apply preprocessing steps and return intermediate results."""
    logger.info("Applying preprocessing steps...")
    
    steps = []
    current_poses = pose_sequence.copy()
    
    # Step 1: Interpolate missing keypoints
    logger.info("Step 1: Interpolating missing keypoints...")
    interpolated = interpolate_missing_keypoints(current_poses)
    steps.append(("1. Interpolated", interpolated.copy()))
    current_poses = interpolated
    
    # Step 2: Normalize poses
    logger.info("Step 2: Normalizing poses...")
    normalized = normalize_pose_sequence(
        current_poses, 
        center_parts=config["data"]["normalize"]["center_parts"],
        scale_parts=config["data"]["normalize"]["scale_parts"],
        reference_bone=config["data"]["normalize"].get("reference_bone", "torso")
    )
    steps.append(("2. Normalized", normalized.copy()))
    current_poses = normalized
    
    # Step 3: Apply bounding box normalization (simulate)
    logger.info("Step 3: Applying bounding box normalization...")
    bbox_normalized = current_poses.copy()
    
    # Calculate bounding box for each frame
    for i in range(len(bbox_normalized)):
        frame = bbox_normalized[i]
        valid_points = frame[:, 2] > 0.3 if frame.shape[1] > 2 else np.ones(len(frame), dtype=bool)
        
        if np.any(valid_points):
            valid_coords = frame[valid_points, :2]
            min_x, min_y = np.min(valid_coords, axis=0)
            max_x, max_y = np.max(valid_coords, axis=0)
            
            # Add padding
            padding = 0.1
            width = max_x - min_x
            height = max_y - min_y
            min_x -= width * padding
            max_x += width * padding
            min_y -= height * padding
            max_y += height * padding
            
            # Normalize to [-1, 1]
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            scale = max(max_x - min_x, max_y - min_y) / 2
            
            if scale > 0:
                bbox_normalized[i, :, 0] = (frame[:, 0] - center_x) / scale
                bbox_normalized[i, :, 1] = (frame[:, 1] - center_y) / scale
    
    steps.append(("3. BBox Normalized", bbox_normalized.copy()))
    current_poses = bbox_normalized
    
    # Step 4: Smooth poses (simple moving average)
    logger.info("Step 4: Smoothing poses...")
    smoothed = current_poses.copy()
    window_size = 3
    
    for i in range(window_size, len(smoothed) - window_size):
        smoothed[i] = np.mean(current_poses[i-window_size:i+window_size+1], axis=0)
    
    steps.append(("4. Smoothed", smoothed.copy()))
    
    logger.info(f"Completed {len(steps)} preprocessing steps")
    return steps


def create_dummy_dataset_for_movement(config: Dict[str, Any]) -> CSLDailyDataset:
    """Create dummy dataset for movement visualization."""
    logger.warning("Creating dummy dataset for movement visualization")
    
    from signstream.training.train_rvq import create_dummy_dataset
    return create_dummy_dataset(config)


def main():
    """Main function for pose movement visualization."""
    parser = argparse.ArgumentParser(description="Visualize pose movements with preprocessing")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to sample from")
    parser.add_argument("--sample-idx", type=int, default=None,
                       help="Specific sample index (random if not specified)")
    parser.add_argument("--output-dir", type=str, default="movement_visualization",
                       help="Output directory")
    parser.add_argument("--fps", type=int, default=15,
                       help="Video frame rate")
    parser.add_argument("--max-frames", type=int, default=200,
                       help="Maximum frames to process")
    parser.add_argument("--create-video", action="store_true",
                       help="Create video output (requires av library)")
    parser.add_argument("--analyze-parts", type=str, nargs="+",
                       default=["body", "left_hand", "right_hand"],
                       help="Body parts to analyze for movement")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create dataset
    try:
        dataset = CSLDailyDataset(
            root_dir=config["data"]["root"],
            split=args.split,
            chunk_len=config["data"]["chunk_len"],
            fps=config["data"]["fps"],
            augment=False,
            body_part_indices=config["data"]["body_parts"],
            center_indices=config["data"]["center_indices"],
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load real dataset: {e}")
        dataset = create_dummy_dataset_for_movement(config)

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Sample a video
    if args.sample_idx is not None:
        sample_idx = args.sample_idx
    else:
        sample_idx = np.random.randint(0, len(dataset))
    
    logger.info(f"Processing sample {sample_idx}")
    sample = dataset[sample_idx]
    video_name = sample["name"]
    
    logger.info(f"Video: {video_name}")
    logger.info(f"Text: {sample.get('text', 'N/A')}")
    logger.info(f"Gloss: {sample.get('gloss', 'N/A')}")
    logger.info(f"Frames: {sample.get('num_frames', 'N/A')}")

    # Initialize visualizer
    visualizer = PoseMovementVisualizer(
        config["data"]["body_parts"], 
        config["data"]["center_indices"]
    )

    # Get original poses (reconstruct from chunks)
    logger.info("Reconstructing original poses from chunks...")
    
    # Use body part for main analysis
    if "body" in sample["chunks"]:
        body_chunks = sample["chunks"]["body"]  # [N, L, K, C]
        N, L, K, C = body_chunks.shape
        
        # Reconstruct full sequence
        original_poses = body_chunks.view(-1, K, C).numpy()  # [N*L, K, C]
        original_poses = original_poses[:args.max_frames]
        
        logger.info(f"Original poses shape: {original_poses.shape}")
        
        # Apply preprocessing steps
        preprocessing_steps = apply_preprocessing_steps(original_poses, config)
        final_processed = preprocessing_steps[-1][1]
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # 1. Preprocessing comparison for sample frames
        sample_frames = [0, len(original_poses)//4, len(original_poses)//2, 
                        3*len(original_poses)//4, len(original_poses)-1]
        
        for frame_idx in sample_frames:
            if frame_idx < len(original_poses):
                fig = visualizer.create_preprocessing_comparison(
                    original_poses, final_processed, frame_idx,
                    f"{video_name} - Preprocessing Comparison"
                )
                plt.savefig(output_dir / f"{video_name}_frame_{frame_idx}_comparison.png", 
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # 2. Movement trajectory analysis for specified body parts
        for part_name in args.analyze_parts:
            if part_name in config["data"]["body_parts"]:
                logger.info(f"Analyzing movement for {part_name}...")
                
                # Original movement
                fig = visualizer.create_movement_trajectory(
                    original_poses, part_name, f"{video_name} - Original"
                )
                plt.savefig(output_dir / f"{video_name}_{part_name}_original_movement.png",
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Processed movement
                fig = visualizer.create_movement_trajectory(
                    final_processed, part_name, f"{video_name} - Processed"
                )
                plt.savefig(output_dir / f"{video_name}_{part_name}_processed_movement.png",
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # 3. Create video if requested
        if args.create_video:
            logger.info("Creating preprocessing steps video...")
            try:
                video_path = output_dir / f"{video_name}_preprocessing_steps.mp4"
                visualizer.create_preprocessing_steps_video(
                    original_poses, preprocessing_steps, video_name, video_path, args.fps
                )
            except Exception as e:
                logger.error(f"Error creating video: {e}")
        
        # 4. Save processing summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "sample_index": sample_idx,
            "original_frames": len(original_poses),
            "processed_frames": len(final_processed),
            "preprocessing_steps": [step_name for step_name, _ in preprocessing_steps],
            "text": sample.get("text", ""),
            "gloss": sample.get("gloss", ""),
            "analysis_parts": args.analyze_parts,
            "config_file": args.config
        }
        
        with open(output_dir / f"{video_name}_processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Visualization complete! Results saved to: {output_dir}")
        logger.info(f"Processed {len(original_poses)} frames with {len(preprocessing_steps)} steps")
    
    else:
        logger.error("No body chunk data found in sample")
    
    # Clean up
    del sample, original_poses, preprocessing_steps
    gc.collect()


if __name__ == "__main__":
    main()