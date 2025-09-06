"""
TensorBoard logging utilities for RVQ training.
"""

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
from PIL import Image


class RVQTensorBoardLogger:
    """TensorBoard logger for RVQ model training with visualizations."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory for tensorboard logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars in one plot."""
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of tensor values."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
        self.writer.add_histogram(tag, values, step)
    
    def log_codebook_utilization(self, codebook_metrics: Dict[str, Any], step: int):
        """Log codebook utilization metrics and visualizations."""
        
        for part_name, metrics in codebook_metrics.items():
            if 'overall' not in metrics:
                continue
                
            overall = metrics['overall']
            
            # Log overall metrics
            self.log_scalar(f'codebook/{part_name}/utilization', 
                           overall['avg_utilization'], step)
            self.log_scalar(f'codebook/{part_name}/perplexity', 
                           overall['avg_perplexity'], step)
            self.log_scalar(f'codebook/{part_name}/entropy', 
                           overall['avg_entropy'], step)
            
            # Per-level metrics
            if 'per_level' in metrics:
                for level_name, level_metrics in metrics['per_level'].items():
                    for metric_name, value in level_metrics.items():
                        if isinstance(value, (int, float)):
                            self.log_scalar(f'codebook/{part_name}/{level_name}/{metric_name}', 
                                          value, step)
            
            # Create utilization visualization
            if 'per_level' in metrics:
                self._log_codebook_utilization_plot(part_name, metrics, step)
    
    def _log_codebook_utilization_plot(self, part_name: str, metrics: Dict, step: int):
        """Create and log codebook utilization bar plot."""
        per_level = metrics['per_level']
        levels = sorted(per_level.keys())
        utilizations = [per_level[level]['utilization'] for level in levels]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(range(len(levels)), utilizations)
        ax.set_xlabel('RVQ Level')
        ax.set_ylabel('Utilization Rate')
        ax.set_title(f'{part_name.title()} Codebook Utilization')
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f'L{i}' for i in range(len(levels))])
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, util in zip(bars, utilizations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{util:.2%}', ha='center', va='bottom')
        
        # Color bars based on utilization rate
        colors = ['red' if u < 0.3 else 'orange' if u < 0.7 else 'green' 
                 for u in utilizations]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        # Convert plot to image and log
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        self.writer.add_image(f'codebook_utilization/{part_name}', 
                             image_array, step, dataformats='HWC')
        plt.close(fig)
        buf.close()
    
    def log_usage_histogram(self, codes: torch.Tensor, codebook_size: int, 
                           part_name: str, step: int):
        """Log histogram of code usage."""
        if codes.dim() > 1:
            codes = codes.flatten()
        
        # Count code frequencies
        counts = torch.bincount(codes, minlength=codebook_size).float()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(codebook_size), counts.numpy())
        ax.set_xlabel('Code Index')
        ax.set_ylabel('Usage Count')
        ax.set_title(f'{part_name.title()} Code Usage Distribution')
        
        # Highlight unused codes
        unused_mask = counts == 0
        if unused_mask.sum() > 0:
            unused_indices = torch.where(unused_mask)[0]
            ax.bar(unused_indices.numpy(), [1] * len(unused_indices), 
                   color='red', alpha=0.7, label='Unused')
            ax.legend()
        
        plt.tight_layout()
        
        # Convert and log
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        self.writer.add_image(f'code_usage/{part_name}', 
                             image_array, step, dataformats='HWC')
        plt.close(fig)
        buf.close()
    
    def log_loss_components(self, losses: Dict[str, float], part_name: str, step: int):
        """Log different loss components for a body part."""
        for loss_name, loss_value in losses.items():
            self.log_scalar(f'losses/{part_name}/{loss_name}', loss_value, step)
    
    def log_reconstruction_quality(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor, 
                                  part_name: str, step: int):
        """Log reconstruction quality metrics."""
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(reconstructed, original)
            mae = torch.nn.functional.l1_loss(reconstructed, original)
            
            # Log metrics
            self.log_scalar(f'reconstruction/{part_name}/mse', mse.item(), step)
            self.log_scalar(f'reconstruction/{part_name}/mae', mae.item(), step)
            
            # Log distribution of reconstruction errors
            errors = (reconstructed - original).flatten()
            self.log_histogram(f'reconstruction/{part_name}/error_dist', errors, step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log current learning rates."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_scalar(f'learning_rate/group_{i}', lr, step)
    
    def log_gradient_norms(self, model: torch.nn.Module, step: int):
        """Log gradient norms for monitoring training stability."""
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual parameter gradient norms (for debugging)
                if step % 100 == 0:  # Log less frequently
                    self.log_scalar(f'gradients/{name.replace(".", "/")}', 
                                   param_norm.item(), step)
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.log_scalar('gradients/total_norm', total_norm, step)
    
    def log_model_weights(self, model: torch.nn.Module, step: int):
        """Log model weight histograms."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f'weights/{name.replace(".", "/")}', 
                                  param.data, step)
    
    def log_token_samples(self, token_samples: Dict[str, Any], step: int):
        """Log sample token outputs as text."""
        for part_name, sample in token_samples.items():
            codes = sample.get('codes', [])
            video_name = sample.get('video_name', 'unknown')
            
            # Convert codes to string representation
            if codes:
                codes_str = str(codes[:5])  # Show first 5 chunks
                text = f"Video: {video_name}\nTokens: {codes_str}..."
                self.writer.add_text(f'token_samples/{part_name}', text, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as text."""
        config_str = self._dict_to_string(config)
        self.writer.add_text('config', config_str, 0)
    
    def _dict_to_string(self, d: Dict, indent: int = 0) -> str:
        """Convert dictionary to formatted string."""
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append('  ' * indent + f'{k}:')
                lines.append(self._dict_to_string(v, indent + 1))
            else:
                lines.append('  ' * indent + f'{k}: {v}')
        return '\n'.join(lines)
    
    def log_temporal_consistency(self, latents: torch.Tensor, part_name: str, step: int):
        """Log temporal consistency metrics."""
        if latents.dim() >= 3:  # [B, N, D]
            # Compute temporal differences
            temporal_diffs = latents[:, 1:] - latents[:, :-1]
            temporal_smoothness = torch.mean(torch.norm(temporal_diffs, dim=-1))
            
            self.log_scalar(f'temporal/{part_name}/smoothness', 
                           temporal_smoothness.item(), step)
            
            # Log distribution of temporal changes
            diff_norms = torch.norm(temporal_diffs, dim=-1).flatten()
            self.log_histogram(f'temporal/{part_name}/change_distribution', 
                              diff_norms, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_tensorboard_logger(config: Dict[str, Any]) -> Optional[RVQTensorBoardLogger]:
    """
    Set up TensorBoard logger based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TensorBoard logger or None if disabled
    """
    logging_config = config.get('logging', {})
    
    if not logging_config.get('use_tensorboard', False):
        return None
    
    log_dir = logging_config.get('log_dir', './logs')
    experiment_name = config.get('experiment_name', 'rvq_experiment')
    
    return RVQTensorBoardLogger(log_dir, experiment_name)