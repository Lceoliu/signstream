"""
Fixed pose movement visualization script that matches the actual data pipeline.
Shows pose transformations exactly as they appear in the dataset getitem method.
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
import torch

from signstream.data.datasets import CSLDailyDataset
from signstream.data.transforms import (
    process_all, normalize_by_global_bbox, interpolate_low_confidence_linear,
    compute_velocity, split_body_parts, BODY_PARTS_INTERVALS, COCO_EDGES
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')


class ActualPoseVisualizer:
    """Visualize poses exactly as processed in the actual data pipeline."""
    
    def __init__(self):
        # Use the actual body part indices from transforms.py
        self.body_part_intervals = BODY_PARTS_INTERVALS
        self.coco_edges = COCO_EDGES
        
        # Define colors for different body parts
        self.colors = {
            'face': '#FF6B6B',      # Red
            'left_hand': '#4ECDC4',  # Teal  
            'right_hand': '#45B7D1', # Blue
            'body': '#96CEB4',       # Green
            'full_body': '#FECA57'   # Yellow
        }
        
    def plot_pose_frame(self, pose: torch.Tensor, title: str = "", ax=None, alpha: float = 1.0) -> plt.Axes:
        """Plot a single pose frame exactly as processed by the data pipeline."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert to numpy for plotting
        if isinstance(pose, torch.Tensor):
            pose_np = pose.detach().cpu().numpy()
        else:
            pose_np = pose
        
        # Extract coordinates (x, y, confidence)
        if pose_np.shape[1] == 3:
            x, y = pose_np[:, 0], pose_np[:, 1]
            confidence = pose_np[:, 2]
        else:
            x, y = pose_np[:, 0], pose_np[:, 1] 
            confidence = np.ones(len(pose_np))
        
        # Plot each body part with skeleton connections
        for part_name, (start_idx, end_idx) in self.body_part_intervals.items():
            if part_name == 'full_body':
                continue  # Skip full_body as it overlaps with others
                
            if start_idx < len(pose_np) and end_idx <= len(pose_np):
                part_x = x[start_idx:end_idx]
                part_y = y[start_idx:end_idx]
                part_conf = confidence[start_idx:end_idx]
                
                # Plot skeleton connections first
                if part_name in self.coco_edges:
                    edges = self.coco_edges[part_name]
                    for edge_start, edge_end in edges:
                        # Adjust indices to be relative to the part
                        rel_start = edge_start - start_idx
                        rel_end = edge_end - start_idx
                        
                        if (0 <= rel_start < len(part_x) and 0 <= rel_end < len(part_x) and 
                            part_conf[rel_start] > 0.1 and part_conf[rel_end] > 0.1):
                            ax.plot([part_x[rel_start], part_x[rel_end]], 
                                   [part_y[rel_start], part_y[rel_end]], 
                                   color=self.colors.get(part_name, '#888888'), 
                                   alpha=alpha*0.6, linewidth=1.5)
                
                # Plot keypoints
                valid_points = part_conf > 0.1
                if np.any(valid_points):
                    ax.scatter(part_x[valid_points], part_y[valid_points], 
                             c=self.colors.get(part_name, '#888888'),
                             label=part_name.replace('_', ' ').title(),
                             s=30, alpha=alpha, edgecolors='white', linewidth=0.5)
        
        # Set axis properties to match the actual data range
        ax.set_xlim(0, 8)  # bbox_range from process_all
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinates
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add axis labels
        ax.set_xlabel('X Coordinate (normalized)', fontsize=12)
        ax.set_ylabel('Y Coordinate (normalized)', fontsize=12)
        
        return ax
    
    def create_preprocessing_comparison(self, original_poses: torch.Tensor, 
                                     processed_parts: Dict[str, Dict],
                                     frame_idx: int,
                                     title: str = "") -> plt.Figure:
        """Create comparison of original vs processed poses using actual pipeline."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{title} - Frame {frame_idx} - Actual Data Pipeline', fontsize=16, fontweight='bold')
        
        # Original pose
        ax = axes[0, 0]
        if frame_idx < len(original_poses):
            self.plot_pose_frame(original_poses[frame_idx], "Original Pose", ax)
        
        # Show each processed body part
        part_names = ['body', 'face', 'left_hand', 'right_hand', 'full_body']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, part_name in enumerate(part_names):
            if i >= len(positions):
                break
            row, col = positions[i]
            ax = axes[row, col]
            
            if part_name in processed_parts and frame_idx < processed_parts[part_name]['pose'].shape[0]:
                part_pose = processed_parts[part_name]['pose'][frame_idx]
                self.plot_pose_frame(part_pose, f"Processed {part_name.replace('_', ' ').title()}", ax)
        
        plt.tight_layout()
        return fig
    
    def create_movement_trajectory(self, poses: torch.Tensor, part_name: str,
                                 title: str = "") -> plt.Figure:
        """Create trajectory plot for a specific body part."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} - {part_name.replace("_", " ").title()} Movement Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Convert to numpy if needed
        if isinstance(poses, torch.Tensor):
            poses_np = poses.detach().cpu().numpy()
        else:
            poses_np = poses
        
        # Calculate center of mass for the part
        if poses_np.shape[2] >= 3:
            confidence = poses_np[:, :, 2]
            valid_mask = confidence > 0.1
            
            # Center of mass trajectory
            com_x = np.array([np.mean(frame[:, 0][valid_mask[i]]) if np.any(valid_mask[i]) else 0 
                             for i, frame in enumerate(poses_np)])
            com_y = np.array([np.mean(frame[:, 1][valid_mask[i]]) if np.any(valid_mask[i]) else 0 
                             for i, frame in enumerate(poses_np)])
        else:
            com_x = np.mean(poses_np[:, :, 0], axis=1)
            com_y = np.mean(poses_np[:, :, 1], axis=1)
        
        # 1. Trajectory in 2D space
        ax1.plot(com_x, com_y, color=self.colors.get(part_name, '#888888'), 
                linewidth=2, alpha=0.7, label='Trajectory')
        ax1.scatter(com_x[0], com_y[0], color='green', s=100, marker='o', 
                   label='Start', zorder=5)
        ax1.scatter(com_x[-1], com_y[-1], color='red', s=100, marker='s', 
                   label='End', zorder=5)
        ax1.set_xlabel('X Coordinate (normalized)')
        ax1.set_ylabel('Y Coordinate (normalized)')
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
    
    def create_chunked_visualization(self, chunked_parts: Dict[str, torch.Tensor],
                                   video_name: str, chunk_idx: int = 0) -> plt.Figure:
        """Visualize chunked data as processed by the dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{video_name} - Chunk {chunk_idx} - Chunked Data Visualization', 
                     fontsize=16, fontweight='bold')
        
        part_names = ['body', 'face', 'left_hand', 'right_hand', 'full_body']
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, part_name in enumerate(part_names):
            if i >= len(positions) or part_name not in chunked_parts:
                continue
                
            row, col = positions[i]
            ax = axes[row, col]
            
            chunk_data = chunked_parts[part_name]  # [N, L, K, 5] - pose+velocity
            if chunk_idx < chunk_data.shape[0]:
                # Take middle frame of the chunk
                mid_frame = chunk_data.shape[1] // 2
                frame_data = chunk_data[chunk_idx, mid_frame, :, :3]  # Only pose data (x,y,conf)
                
                self.plot_pose_frame(frame_data, f"{part_name.replace('_', ' ').title()} (Chunk {chunk_idx})", ax)
        
        # Summary statistics in the last subplot
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"Video: {video_name}\n"
        summary_text += f"Chunk Index: {chunk_idx}\n\n"
        
        for part_name in part_names:
            if part_name in chunked_parts:
                chunk_data = chunked_parts[part_name]
                summary_text += f"{part_name}: {chunk_data.shape}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_preprocessing_steps_video(self, sample_data: Dict,
                                       video_name: str,
                                       output_path: Path,
                                       fps: int = 15) -> None:
        """Create video showing the actual preprocessing pipeline."""
        logger.info(f"Creating preprocessing pipeline video: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get original poses and processed parts
        chunks = sample_data['chunks']
        
        # Reconstruct sequences from chunks for visualization
        max_frames = min(100, sample_data.get('num_frames', 50))
        
        temp_frames = []
        for frame_idx in range(0, max_frames, 2):  # Every 2nd frame
            fig = plt.figure(figsize=(20, 12))
            
            # Create a grid showing different body parts
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            part_names = ['body', 'face', 'left_hand', 'right_hand', 'full_body']
            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
            
            for i, part_name in enumerate(part_names):
                if i >= len(positions) or part_name not in chunks:
                    continue
                    
                row, col = positions[i]
                ax = fig.add_subplot(gs[row, col])
                
                chunk_data = chunks[part_name]  # [N, L, K, 5]
                
                # Calculate which chunk and frame within chunk
                chunk_len = chunk_data.shape[1]
                chunk_idx = frame_idx // chunk_len
                frame_in_chunk = frame_idx % chunk_len
                
                if chunk_idx < chunk_data.shape[0]:
                    frame_data = chunk_data[chunk_idx, frame_in_chunk, :, :3]  # pose only
                    self.plot_pose_frame(frame_data, f"{part_name.replace('_', ' ').title()}", ax)
            
            # Add info panel
            ax_info = fig.add_subplot(gs[1, 2])
            ax_info.axis('off')
            
            info_text = f"Video: {video_name}\n"
            info_text += f"Frame: {frame_idx}\n"
            info_text += f"Total Frames: {sample_data.get('num_frames', 'Unknown')}\n"
            info_text += f"Total Chunks: {sample_data.get('num_chunks', 'Unknown')}\n\n"
            info_text += "Processing Pipeline:\n"
            info_text += "1. Load pose sequence\n"
            info_text += "2. Global bbox normalization\n"
            info_text += "3. Interpolate low confidence\n"
            info_text += "4. Compute velocity\n"
            info_text += "5. Split body parts\n"
            info_text += "6. Part-wise bbox normalization\n"
            info_text += "7. Chunk sequences\n"
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.suptitle(f'{video_name} - Frame {frame_idx} - Actual Data Pipeline', 
                        fontsize=20, fontweight='bold')
            
            # Convert to image array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            temp_frames.append(img_array)
            plt.close(fig)
            
            if (frame_idx // 2) % 10 == 0:
                logger.info(f"Processed frame {frame_idx}/{max_frames}")
        
        # Write video using av
        try:
            container = av.open(str(output_path), mode='w')
            stream = container.add_stream('libx264', rate=fps)
            stream.height = temp_frames[0].shape[0]
            stream.width = temp_frames[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            
            for frame_array in temp_frames:
                frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)
            
            # Flush remaining packets
            packet = stream.encode()
            container.mux(packet)
            container.close()
            
            logger.info(f"Video saved: {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating video with av: {e}")
            # Fallback to GIF
            fallback_path = output_path.with_suffix('.gif')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            def animate(frame):
                ax.clear()
                ax.imshow(temp_frames[frame % len(temp_frames)])
                ax.axis('off')
                ax.set_title(f"Frame {frame}")
                return ax,
            
            anim = FuncAnimation(fig, animate, frames=len(temp_frames), interval=1000//fps, blit=False)
            anim.save(str(fallback_path), writer='pillow', fps=fps)
            plt.close(fig)
            logger.info(f"Fallback GIF saved: {fallback_path}")
        
        # Clean up
        del temp_frames
        gc.collect()


def create_dummy_dataset_for_movement(config: Dict[str, Any]) -> CSLDailyDataset:
    """Create dummy dataset for movement visualization."""
    logger.warning("Creating dummy dataset for movement visualization")
    
    from signstream.training.train_rvq import create_dummy_dataset
    return create_dummy_dataset(config)


def main():
    """Main function for actual pose movement visualization."""
    parser = argparse.ArgumentParser(description="Visualize actual pose movement pipeline")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to sample from")
    parser.add_argument("--sample-idx", type=int, default=None,
                       help="Specific sample index (random if not specified)")
    parser.add_argument("--output-dir", type=str, default="actual_movement_visualization",
                       help="Output directory")
    parser.add_argument("--fps", type=int, default=15,
                       help="Video frame rate")
    parser.add_argument("--create-video", action="store_true",
                       help="Create video output (requires av library)")
    parser.add_argument("--analyze-chunks", action="store_true",
                       help="Analyze chunked data as well")
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
    logger.info(f"Chunks: {sample.get('num_chunks', 'N/A')}")

    # Initialize visualizer
    visualizer = ActualPoseVisualizer()

    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Analyze chunked data structure
    chunks = sample['chunks']
    logger.info("Chunked data structure:")
    for part_name, chunk_data in chunks.items():
        logger.info(f"  {part_name}: {chunk_data.shape}")
    
    # 2. Visualize different chunks
    if args.analyze_chunks:
        num_chunks = sample['num_chunks']
        sample_chunks = [0, num_chunks//4, num_chunks//2, 3*num_chunks//4, num_chunks-1]
        
        for chunk_idx in sample_chunks:
            if chunk_idx < num_chunks:
                fig = visualizer.create_chunked_visualization(chunks, video_name, chunk_idx)
                plt.savefig(output_dir / f"{video_name}_chunk_{chunk_idx}_visualization.png", 
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    # 3. Movement trajectory analysis for each body part
    for part_name in ['body', 'face', 'left_hand', 'right_hand']:
        if part_name in chunks:
            logger.info(f"Analyzing movement for {part_name}...")
            
            chunk_data = chunks[part_name]  # [N, L, K, 5]
            
            # Reconstruct sequence from chunks (approximately)
            N, L, K, _ = chunk_data.shape
            pose_data = chunk_data[:, :, :, :3]  # Only pose (x,y,conf)
            
            # Take middle frames from each chunk to avoid edge effects
            mid_frame = L // 2
            sequence_poses = pose_data[:, mid_frame, :, :]  # [N, K, 3]
            
            fig = visualizer.create_movement_trajectory(
                sequence_poses, part_name, f"{video_name} - {part_name.title()}"
            )
            plt.savefig(output_dir / f"{video_name}_{part_name}_movement_analysis.png",
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # 4. Create video if requested
    if args.create_video:
        logger.info("Creating data pipeline video...")
        try:
            video_path = output_dir / f"{video_name}_actual_pipeline.mp4"
            visualizer.create_preprocessing_steps_video(
                sample, video_name, video_path, args.fps
            )
        except Exception as e:
            logger.error(f"Error creating video: {e}")
    
    # 5. Save processing summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "video_name": video_name,
        "sample_index": sample_idx,
        "total_frames": sample.get('num_frames', 0),
        "total_chunks": sample.get('num_chunks', 0),
        "chunk_structure": {part: list(chunks[part].shape) for part in chunks.keys()},
        "text": sample.get("text", ""),
        "gloss": sample.get("gloss", ""),
        "config_file": args.config,
        "processing_notes": [
            "Data processed through actual pipeline in datasets.py",
            "Uses process_all() from transforms.py", 
            "Global bbox normalization -> interpolation -> velocity -> split -> chunk",
            "Each body part has separate bbox normalization",
            "Final format: [N_chunks, chunk_len, keypoints, 5] with pose+velocity"
        ]
    }
    
    with open(output_dir / f"{video_name}_actual_pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Visualization complete! Results saved to: {output_dir}")
    logger.info(f"Processed sample with {sample['num_chunks']} chunks")
    
    # Clean up
    del sample, chunks
    gc.collect()


if __name__ == "__main__":
    main()