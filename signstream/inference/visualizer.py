"""
Comprehensive visualization tools for RVQ model analysis and pose reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from ..models.rvq.rvq_model import RVQModel
from ..models.metrics.codebook import compute_all_codebook_metrics
from .export_tokens import load_model

logger = logging.getLogger(__name__)


class RVQVisualizer:
    """Comprehensive visualizer for RVQ model analysis."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None):
        self.config = config
        self.model = load_model(config, checkpoint_path) if checkpoint_path else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_codebook_usage(self, codes_per_level: List[torch.Tensor], 
                               codebook_size: int, part_name: str,
                               save_path: Optional[str] = None) -> None:
        """
        Create comprehensive codebook usage visualizations.
        
        Args:
            codes_per_level: List of code tensors for each RVQ level
            codebook_size: Size of each codebook
            part_name: Name of body part
            save_path: Optional path to save figure
        """
        num_levels = len(codes_per_level)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Codebook Analysis - {part_name.title()}', fontsize=16)
        
        # 1. Usage histogram for each level
        ax1 = axes[0, 0]
        for level, codes in enumerate(codes_per_level):
            counts = torch.bincount(codes.flatten(), minlength=codebook_size).float()
            ax1.bar(np.arange(codebook_size) + level * 0.1, counts.numpy(), 
                   width=0.8, alpha=0.7, label=f'Level {level}')
        
        ax1.set_xlabel('Code Index')
        ax1.set_ylabel('Usage Count')
        ax1.set_title('Code Usage Distribution by Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Utilization rate by level
        ax2 = axes[0, 1]
        utilizations = []
        for codes in codes_per_level:
            unique_codes = len(torch.unique(codes.flatten()))
            utilization = unique_codes / codebook_size
            utilizations.append(utilization)
        
        bars = ax2.bar(range(num_levels), utilizations)
        ax2.set_xlabel('RVQ Level')
        ax2.set_ylabel('Utilization Rate')
        ax2.set_title('Codebook Utilization by Level')
        ax2.set_ylim(0, 1.0)
        
        # Color bars based on utilization
        colors = ['red' if u < 0.3 else 'orange' if u < 0.7 else 'green' 
                 for u in utilizations]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (bar, util) in enumerate(zip(bars, utilizations)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{util:.2%}', ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Dead codes heatmap
        ax3 = axes[1, 0]
        dead_codes_matrix = np.zeros((num_levels, codebook_size))
        
        for level, codes in enumerate(codes_per_level):
            used_codes = set(codes.flatten().tolist())
            for code_idx in range(codebook_size):
                dead_codes_matrix[level, code_idx] = 1 if code_idx not in used_codes else 0
        
        im = ax3.imshow(dead_codes_matrix, aspect='auto', cmap='Reds')
        ax3.set_xlabel('Code Index')
        ax3.set_ylabel('RVQ Level')
        ax3.set_title('Dead Codes (Red = Unused)')
        plt.colorbar(im, ax=ax3)
        
        # 4. Usage frequency distribution
        ax4 = axes[1, 1]
        all_frequencies = []
        for codes in codes_per_level:
            counts = torch.bincount(codes.flatten(), minlength=codebook_size).float()
            frequencies = counts[counts > 0].numpy()  # Only used codes
            all_frequencies.extend(frequencies)
        
        if all_frequencies:
            ax4.hist(all_frequencies, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Usage Frequency')
            ax4.set_ylabel('Number of Codes')
            ax4.set_title('Usage Frequency Distribution')
            ax4.axvline(np.mean(all_frequencies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_frequencies):.1f}')
            ax4.legend()
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved codebook visualization: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def visualize_temporal_patterns(self, latents: torch.Tensor, part_name: str,
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize temporal patterns in the latent space.
        
        Args:
            latents: Tensor of shape [batch_size, num_chunks, latent_dim]
            part_name: Name of body part
            save_path: Optional path to save figure
        """
        if latents.dim() != 3:
            logger.warning("Expected 3D latents tensor [B, N, D]")
            return
        
        B, N, D = latents.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Temporal Analysis - {part_name.title()}', fontsize=16)
        
        # 1. Latent magnitude over time
        ax1 = axes[0, 0]
        latent_norms = torch.norm(latents, dim=-1)  # [B, N]
        
        for i in range(min(5, B)):  # Show first 5 samples
            ax1.plot(latent_norms[i].cpu().numpy(), alpha=0.7, label=f'Sample {i}')
        
        ax1.set_xlabel('Chunk Index')
        ax1.set_ylabel('Latent Norm')
        ax1.set_title('Latent Vector Magnitudes Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Temporal differences
        ax2 = axes[0, 1]
        if N > 1:
            temporal_diffs = latents[:, 1:] - latents[:, :-1]  # [B, N-1, D]
            diff_norms = torch.norm(temporal_diffs, dim=-1)  # [B, N-1]
            
            # Plot temporal difference statistics
            mean_diffs = diff_norms.mean(dim=0).cpu().numpy()
            std_diffs = diff_norms.std(dim=0).cpu().numpy()
            
            x = np.arange(len(mean_diffs))
            ax2.plot(x, mean_diffs, 'b-', label='Mean')
            ax2.fill_between(x, mean_diffs - std_diffs, mean_diffs + std_diffs, 
                           alpha=0.3, label='±1 std')
            ax2.set_xlabel('Temporal Difference Index')
            ax2.set_ylabel('Difference Magnitude')
            ax2.set_title('Temporal Smoothness')
            ax2.legend()
        
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA of latent space
        ax3 = axes[1, 0]
        from sklearn.decomposition import PCA
        
        # Flatten for PCA
        latents_flat = latents.view(-1, D).cpu().numpy()
        if latents_flat.shape[0] > 2:
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents_flat)
            
            # Color by temporal position
            colors = np.repeat(np.arange(N), B)
            scatter = ax3.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, 
                                alpha=0.6, cmap='viridis')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax3.set_title('Latent Space PCA (colored by time)')
            plt.colorbar(scatter, ax=ax3, label='Chunk Index')
        
        # 4. Latent dimension variance
        ax4 = axes[1, 1]
        latent_vars = latents.var(dim=(0, 1)).cpu().numpy()  # Variance across batch and time
        
        ax4.bar(range(len(latent_vars)), latent_vars)
        ax4.set_xlabel('Latent Dimension')
        ax4.set_ylabel('Variance')
        ax4.set_title('Latent Dimension Utilization')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temporal visualization: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def visualize_reconstruction_quality(self, original: torch.Tensor, 
                                       reconstructed: torch.Tensor,
                                       part_name: str,
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize reconstruction quality with error analysis.
        
        Args:
            original: Original pose data
            reconstructed: Reconstructed pose data  
            part_name: Name of body part
            save_path: Optional path to save figure
        """
        if original.shape != reconstructed.shape:
            logger.warning("Original and reconstructed shapes don't match")
            return
        
        # Calculate errors
        errors = (reconstructed - original).cpu().numpy()
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Reconstruction Quality - {part_name.title()}\n'
                    f'MSE: {mse:.6f}, MAE: {mae:.6f}', fontsize=16)
        
        # 1. Error distribution
        ax1 = axes[0, 0]
        ax1.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', label='Perfect reconstruction')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Original vs Reconstructed scatter
        ax2 = axes[0, 1]
        sample_indices = np.random.choice(original.numel(), min(10000, original.numel()), 
                                        replace=False)
        orig_sample = original.flatten()[sample_indices].cpu().numpy()
        recon_sample = reconstructed.flatten()[sample_indices].cpu().numpy()
        
        ax2.scatter(orig_sample, recon_sample, alpha=0.5, s=1)
        
        # Perfect reconstruction line
        min_val = min(orig_sample.min(), recon_sample.min())
        max_val = max(orig_sample.max(), recon_sample.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect reconstruction')
        
        ax2.set_xlabel('Original Values')
        ax2.set_ylabel('Reconstructed Values')
        ax2.set_title('Original vs Reconstructed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error over time (if temporal)
        ax3 = axes[1, 0]
        if len(original.shape) >= 2:
            # Assuming first dim is temporal or batch
            temporal_errors = np.mean(np.abs(errors), axis=tuple(range(1, len(errors.shape))))
            ax3.plot(temporal_errors, 'b-', linewidth=2)
            ax3.set_xlabel('Time/Batch Index')
            ax3.set_ylabel('Mean Absolute Error')
            ax3.set_title('Reconstruction Error Over Time')
            ax3.grid(True, alpha=0.3)
        
        # 4. Error heatmap (if 2D structure)
        ax4 = axes[1, 1]
        if len(original.shape) >= 3:
            # Take mean over first dimension and show as heatmap
            error_map = np.mean(np.abs(errors), axis=0)
            im = ax4.imshow(error_map, aspect='auto', cmap='Reds')
            ax4.set_xlabel('Feature Dimension')
            ax4.set_ylabel('Temporal/Spatial Dimension')
            ax4.set_title('Mean Absolute Error Heatmap')
            plt.colorbar(im, ax=ax4)
        else:
            ax4.text(0.5, 0.5, 'Error heatmap not available\nfor this data shape', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Error Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reconstruction visualization: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def create_token_timeline(self, token_data: Dict[str, Any], 
                            save_path: Optional[str] = None) -> None:
        """
        Create a timeline visualization of token sequences.
        
        Args:
            token_data: Exported token data from export_tokens
            save_path: Optional path to save figure
        """
        video_name = token_data.get('video_name', 'Unknown')
        tokens = token_data.get('tokens', {})
        
        if not tokens:
            logger.warning("No token data to visualize")
            return
        
        body_parts = list(tokens.keys())
        num_parts = len(body_parts)
        
        fig, axes = plt.subplots(num_parts, 1, figsize=(15, 3 * num_parts), 
                               sharex=True)
        if num_parts == 1:
            axes = [axes]
        
        fig.suptitle(f'Token Timeline - {video_name}', fontsize=16)
        
        for i, part in enumerate(body_parts):
            ax = axes[i]
            part_tokens = tokens[part]
            
            # Extract time indices and codes
            times = []
            code_sequences = []
            
            for token_entry in part_tokens:
                t = token_entry.get('t', 0)
                times.append(t)
                
                # Get the codes for this part
                part_key = part.upper().replace('_', '')
                codes = token_entry.get(part_key, [])
                
                if isinstance(codes, list) and len(codes) > 0:
                    if isinstance(codes[0], str) and codes[0] == "NC":
                        # No-change token
                        code_sequences.append(f"NC×{codes[1]}")
                    else:
                        # Regular codes
                        code_sequences.append('-'.join(map(str, codes)))
                else:
                    code_sequences.append("Empty")
            
            # Create timeline
            y_pos = [0] * len(times)
            colors = plt.cm.tab10(np.linspace(0, 1, len(set(code_sequences))))
            
            # Map unique code sequences to colors
            unique_codes = list(set(code_sequences))
            color_map = {code: colors[i % len(colors)] for i, code in enumerate(unique_codes)}
            
            for j, (t, codes) in enumerate(zip(times, code_sequences)):
                ax.scatter(t, y_pos[j], s=100, c=[color_map[codes]], 
                         alpha=0.8, edgecolors='black')
                
                # Add text annotation for short sequences
                if len(codes) < 20:  
                    ax.annotate(codes, (t, y_pos[j]), xytext=(0, 10), 
                              textcoords='offset points', ha='center', 
                              fontsize=8, rotation=45)
            
            ax.set_ylabel(f'{part.title()}\nCodes')
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([])
            
            # Add legend for unique codes
            if len(unique_codes) <= 10:  # Only show legend if not too many codes
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color_map[code], 
                                            markersize=8, label=code)
                                 for code in unique_codes[:5]]  # Limit to 5 for space
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                         loc='upper left')
        
        axes[-1].set_xlabel('Chunk Index')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved token timeline: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


def create_visualization_report(config_path: str, checkpoint_path: str, 
                              token_file: str, output_dir: str) -> None:
    """
    Create a comprehensive visualization report.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to trained model checkpoint
        token_file: Path to exported tokens file
        output_dir: Directory to save visualizations
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RVQVisualizer(config, checkpoint_path)
    
    # Load token data
    token_data = []
    with open(token_file, 'r') as f:
        for line in f:
            token_data.append(json.loads(line))
    
    logger.info(f"Creating visualization report with {len(token_data)} samples")
    
    # Create token timeline for first few samples
    for i, sample in enumerate(token_data[:3]):
        timeline_path = output_path / f'token_timeline_sample_{i}.png'
        visualizer.create_token_timeline(sample, str(timeline_path))
    
    logger.info(f"Visualization report created in {output_dir}")