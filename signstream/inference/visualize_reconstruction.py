"""
Comprehensive visualization script for RVQ model reconstruction quality.
Loads checkpoint, samples videos, exports tokens, and visualizes original vs decoded poses.
"""

import argparse
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import gc

from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from signstream.inference.export_tokens import create_token_template


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PoseVisualizer:
    """Visualize pose sequences with body part separation."""
    
    def __init__(self, body_part_indices: Dict[str, Tuple[int, int]]):
        self.body_part_indices = body_part_indices
        self.colors = {
            'face': '#FF6B6B',      # Red
            'left_hand': '#4ECDC4',  # Teal
            'right_hand': '#45B7D1', # Blue
            'body': '#96CEB4',       # Green
            'full_body': '#FECA57'   # Yellow
        }
        
    def plot_pose_frame(self, pose: np.ndarray, title: str = "", ax=None, alpha: float = 1.0) -> plt.Axes:
        """Plot a single pose frame with different colors for body parts."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Extract 2D coordinates (assuming pose shape is [keypoints, 3] with x,y,confidence)
        if pose.shape[1] == 3:
            x, y = pose[:, 0], pose[:, 1]
            confidence = pose[:, 2]
        else:
            x, y = pose[:, 0], pose[:, 1] 
            confidence = np.ones(len(pose))
        
        # Plot each body part with different colors
        for part_name, (start_idx, end_idx) in self.body_part_indices.items():
            if part_name == 'full_body':
                continue  # Skip full_body as it overlaps with others
                
            # Adjust indices (datasets might be 1-indexed)
            start_idx = max(0, start_idx - 1) if start_idx > 0 else start_idx
            end_idx = min(len(pose), end_idx)
            
            if start_idx < len(pose) and end_idx <= len(pose):
                part_x = x[start_idx:end_idx]
                part_y = y[start_idx:end_idx]
                part_conf = confidence[start_idx:end_idx]
                
                # Filter out low confidence points
                valid_points = part_conf > 0.3
                if np.any(valid_points):
                    ax.scatter(part_x[valid_points], part_y[valid_points], 
                             c=self.colors.get(part_name, '#888888'), 
                             label=part_name.replace('_', ' ').title(),
                             alpha=alpha, s=20)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y axis for image coordinates
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def create_comparison_animation(self, original_poses: np.ndarray, decoded_poses: np.ndarray, 
                                  video_name: str, save_path: Optional[Path] = None) -> animation.FuncAnimation:
        """Create side-by-side animation comparing original and decoded poses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Pose Reconstruction Comparison: {video_name}', fontsize=16)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            if frame < len(original_poses):
                self.plot_pose_frame(original_poses[frame], f"Original (Frame {frame})", ax1)
            if frame < len(decoded_poses):
                self.plot_pose_frame(decoded_poses[frame], f"Decoded (Frame {frame})", ax2)
            
            return ax1, ax2
        
        anim = animation.FuncAnimation(fig, animate, frames=min(len(original_poses), len(decoded_poses)),
                                     interval=100, blit=False, repeat=True)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(str(save_path), writer='pillow', fps=10)
            logger.info(f"Saved animation: {save_path}")
        
        return anim
    
    def plot_reconstruction_error(self, original_poses: np.ndarray, decoded_poses: np.ndarray,
                                video_name: str, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot reconstruction error analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Reconstruction Error Analysis: {video_name}', fontsize=16)
        
        # Ensure same length for comparison
        min_len = min(len(original_poses), len(decoded_poses))
        orig_poses = original_poses[:min_len]
        dec_poses = decoded_poses[:min_len]
        
        # Calculate per-frame errors
        frame_errors = np.mean(np.square(orig_poses - dec_poses), axis=(1, 2))
        
        # Per-frame error over time
        axes[0, 0].plot(frame_errors, linewidth=2)
        axes[0, 0].set_title('Reconstruction Error Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 1].hist(frame_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].set_xlabel('Mean Squared Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-body-part error analysis
        part_errors = {}
        for part_name, (start_idx, end_idx) in self.body_part_indices.items():
            if part_name == 'full_body':
                continue
                
            start_idx = max(0, start_idx - 1) if start_idx > 0 else start_idx
            end_idx = min(orig_poses.shape[1], end_idx)
            
            if start_idx < orig_poses.shape[1] and end_idx <= orig_poses.shape[1]:
                part_orig = orig_poses[:, start_idx:end_idx]
                part_dec = dec_poses[:, start_idx:end_idx]
                part_error = np.mean(np.square(part_orig - part_dec))
                part_errors[part_name] = part_error
        
        # Bar plot of part errors
        if part_errors:
            parts = list(part_errors.keys())
            errors = list(part_errors.values())
            bars = axes[1, 0].bar(parts, errors, color=[self.colors.get(p, '#888888') for p in parts])
            axes[1, 0].set_title('Error by Body Part')
            axes[1, 0].set_ylabel('Mean Squared Error')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{error:.4f}', ha='center', va='bottom')
        
        # Sample frames comparison
        sample_frames = np.linspace(0, min_len-1, 5).astype(int)
        frame_mse = [np.mean(np.square(orig_poses[f] - dec_poses[f])) for f in sample_frames]
        axes[1, 1].plot(sample_frames, frame_mse, 'o-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Sample Frame Errors')
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('Mean Squared Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved error analysis: {save_path}")
        
        return fig


def load_model(config: Dict[str, Any], checkpoint_path: str) -> RVQModel:
    """Load RVQ model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    model = RVQModel(
        latent_dim=config["model"]["latent_dim"],
        chunk_len=config["data"]["chunk_len"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"]["commitment_beta"],
        ema_decay=config["model"]["rvq"].get("ema_decay", 0.99),
        usage_reg=config["model"]["rvq"].get("usage_reg", 1e-3),
        arch=config["model"]["arch"],
        num_layers=config["model"].get("encoder_layer", 2),
        type_embed_dim=config["model"].get("type_embed_dim", 16),
        dropout=config["model"].get("dropout", 0.1),
        temporal_aggregation=config["model"].get("temporal_aggregation", "mean"),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    del checkpoint  # Clean up immediately
    gc.collect()
    
    return model


def create_dummy_dataset_for_visualization(config: Dict[str, Any]) -> CSLDailyDataset:
    """Create dummy dataset for visualization when real data is not available."""
    logger.warning("Creating dummy dataset for visualization")
    
    from signstream.training.train_rvq import create_dummy_dataset
    return create_dummy_dataset(config)


def reconstruct_poses_from_tokens(model: RVQModel, tokens_dict: Dict[str, List], 
                                body_parts: List[str], chunk_len: int, 
                                device: torch.device) -> Dict[str, np.ndarray]:
    """Reconstruct pose sequences from exported tokens."""
    logger.info("Reconstructing poses from tokens...")
    
    reconstructed_parts = {}
    
    for part in body_parts:
        if part not in tokens_dict:
            continue
            
        part_tokens = tokens_dict[part]
        if not part_tokens:
            continue
        
        # Extract codes from token format
        codes_list = []
        for token_entry in part_tokens:
            part_key = part.upper().replace("_", "")
            if part_key in token_entry:
                codes = token_entry[part_key]
                if isinstance(codes, list) and len(codes) > 0:
                    codes_list.append(codes)
        
        if not codes_list:
            continue
        
        # Convert to tensor [N, levels]
        codes_tensor = torch.tensor(codes_list, dtype=torch.long).to(device)
        
        # Decode using model's quantizer
        with torch.no_grad():
            # Get the quantizer for this body part
            quantizer = model.quantizers[part]
            decoded_latents = quantizer.decode(codes_tensor)  # [N, latent_dim]
            
            # Decode latents to poses using the decoder
            decoded_poses = model.decode_part(decoded_latents, part)  # [N, chunk_len, K*C]
            
            # Reshape to pose format
            if part == 'face':
                K, C = 67, 3  # Face keypoints
            elif part in ['left_hand', 'right_hand']:
                K, C = 21, 3  # Hand keypoints
            elif part == 'body':
                K, C = 17, 3  # Body keypoints
            else:  # full_body
                K, C = 133, 3  # All keypoints
            
            N = decoded_poses.shape[0]
            decoded_poses = decoded_poses.view(N, chunk_len, K, C)
            
            # Convert to numpy and reshape to sequence
            pose_sequence = decoded_poses.cpu().numpy()  # [N, chunk_len, K, C]
            pose_sequence = pose_sequence.reshape(-1, K, C)  # [N*chunk_len, K, C]
            
            reconstructed_parts[part] = pose_sequence
    
    return reconstructed_parts


def export_sample_tokens(model: RVQModel, sample: Dict, body_parts: List[str], 
                        device: torch.device) -> Dict[str, Any]:
    """Export tokens for a single sample."""
    result = {
        "video_name": sample["name"],
        "text": sample.get("text", ""),
        "gloss": sample.get("gloss", ""),
        "tokens": {},
        "templates": []
    }
    
    # Process each body part
    for part in body_parts:
        if part not in sample["chunks"]:
            continue
            
        x = sample["chunks"][part]  # [N, L, K, C]
        N, L, K, C = x.shape
        x_flat = x.view(N, L, K * C).to(device)
        
        with torch.no_grad():
            _, codes, _, _, _ = model(x_flat, part)
        
        # Convert to token format
        codes_list = codes.cpu().tolist()
        part_tokens = []
        for chunk_idx, chunk_codes in enumerate(codes_list):
            token_entry = {
                "t": chunk_idx,
                part.upper().replace("_", ""): chunk_codes
            }
            part_tokens.append(token_entry)
        
        result["tokens"][part] = part_tokens
        
        # Clean up
        del x_flat, codes, codes_list
    
    # Create template visualization
    max_template_chunks = min(3, len(result["tokens"].get(body_parts[0], [])))
    for chunk_idx in range(max_template_chunks):
        codes_for_chunk = {}
        for part in body_parts:
            if part in result["tokens"]:
                codes_for_chunk[part] = [
                    token[part.upper().replace("_", "")]
                    for token in result["tokens"][part]
                ]
        
        if codes_for_chunk:
            template = create_token_template(codes_for_chunk, chunk_idx)
            result["templates"].append(template)
    
    return result


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize RVQ reconstruction quality")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to sample from")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of videos to sample and visualize")
    parser.add_argument("--body-parts", type=str, nargs="+", 
                       default=["face", "left_hand", "right_hand", "body"],
                       help="Body parts to process")
    parser.add_argument("--output-dir", type=str, default="visualization_output",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--save-animations", action="store_true",
                       help="Save comparison animations (warning: large files)")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="Maximum frames to visualize per video")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
        dataset = create_dummy_dataset_for_visualization(config)

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    model = load_model(config, args.checkpoint)
    model.to(device)
    logger.info("Model loaded successfully")

    # Initialize visualizer
    visualizer = PoseVisualizer(config["data"]["body_parts"])
    
    # Sample videos and process
    sample_indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), 
                                    replace=False)
    
    results_summary = []
    
    for i, sample_idx in enumerate(sample_indices):
        logger.info(f"Processing sample {i+1}/{len(sample_indices)}: index {sample_idx}")
        
        try:
            sample = dataset[sample_idx]
            video_name = sample["name"]
            
            # Export tokens
            logger.info(f"Exporting tokens for {video_name}")
            token_data = export_sample_tokens(model, sample, args.body_parts, device)
            
            # Save token data
            token_path = output_dir / f"{video_name}_tokens.json"
            with open(token_path, 'w', encoding='utf-8') as f:
                json.dump(token_data, f, ensure_ascii=False, indent=2)
            
            # Get original poses (use the first body part for visualization)
            primary_part = args.body_parts[0]
            if primary_part in sample["chunks"]:
                original_chunks = sample["chunks"][primary_part]  # [N, L, K, C]
                N, L, K, C = original_chunks.shape
                
                # Reshape to sequence
                original_poses = original_chunks.view(-1, K, C).numpy()  # [N*L, K, C]
                original_poses = original_poses[:args.max_frames]  # Limit frames
                
                # Reconstruct poses from tokens
                logger.info(f"Reconstructing poses for {video_name}")
                reconstructed_parts = reconstruct_poses_from_tokens(
                    model, token_data["tokens"], [primary_part], 
                    config["data"]["chunk_len"], device
                )
                
                if primary_part in reconstructed_parts:
                    decoded_poses = reconstructed_parts[primary_part][:args.max_frames]
                    
                    # Create visualizations
                    logger.info(f"Creating visualizations for {video_name}")
                    
                    # Error analysis plot
                    error_fig = visualizer.plot_reconstruction_error(
                        original_poses, decoded_poses, video_name,
                        output_dir / f"{video_name}_error_analysis.png"
                    )
                    plt.close(error_fig)
                    
                    # Sample frame comparison
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'Sample Frame Comparisons: {video_name}', fontsize=16)
                    
                    sample_frames = np.linspace(0, min(len(original_poses), len(decoded_poses))-1, 6).astype(int)
                    for j, frame_idx in enumerate(sample_frames):
                        row = j // 3
                        col = j % 3
                        ax = axes[row, col]
                        
                        # Plot original (alpha=0.7) and decoded (alpha=1.0) overlaid
                        if frame_idx < len(original_poses):
                            visualizer.plot_pose_frame(original_poses[frame_idx], 
                                                     f"Frame {frame_idx}\nOriginal (faded) + Decoded", 
                                                     ax, alpha=0.4)
                        if frame_idx < len(decoded_poses):
                            visualizer.plot_pose_frame(decoded_poses[frame_idx], "", ax, alpha=1.0)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{video_name}_sample_frames.png", 
                               dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Optional: Create animation
                    if args.save_animations:
                        anim = visualizer.create_comparison_animation(
                            original_poses, decoded_poses, video_name,
                            output_dir / f"{video_name}_comparison.gif"
                        )
                        plt.close('all')  # Close animation figures
                    
                    # Calculate summary metrics
                    min_len = min(len(original_poses), len(decoded_poses))
                    if min_len > 0:
                        mse = np.mean(np.square(original_poses[:min_len] - decoded_poses[:min_len]))
                        mae = np.mean(np.abs(original_poses[:min_len] - decoded_poses[:min_len]))
                        
                        sample_result = {
                            "video_name": video_name,
                            "num_frames": min_len,
                            "mse": float(mse),
                            "mae": float(mae),
                            "tokens": len(token_data["tokens"].get(primary_part, [])),
                            "sample_template": token_data["templates"][:2] if token_data["templates"] else []
                        }
                        results_summary.append(sample_result)
                        
                        logger.info(f"Sample {video_name}: MSE={mse:.6f}, MAE={mae:.6f}")
                
                # Clean up
                del original_poses, reconstructed_parts
                if 'decoded_poses' in locals():
                    del decoded_poses
            
            # Clean up sample
            del sample, token_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Save summary results
    summary_path = output_dir / "reconstruction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "checkpoint": args.checkpoint,
            "num_samples": len(results_summary),
            "body_parts": args.body_parts,
            "samples": results_summary,
            "overall_metrics": {
                "avg_mse": np.mean([r["mse"] for r in results_summary]) if results_summary else 0,
                "avg_mae": np.mean([r["mae"] for r in results_summary]) if results_summary else 0,
                "avg_tokens": np.mean([r["tokens"] for r in results_summary]) if results_summary else 0
            }
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Visualization complete! Results saved to: {output_dir}")
    logger.info(f"Summary: {len(results_summary)} samples processed")
    if results_summary:
        avg_mse = np.mean([r["mse"] for r in results_summary])
        avg_mae = np.mean([r["mae"] for r in results_summary])
        logger.info(f"Average MSE: {avg_mse:.6f}, Average MAE: {avg_mae:.6f}")
    
    # Clean up
    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()