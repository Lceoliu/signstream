"""
Codebook usage and quality metrics for RVQ.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any


def codebook_utilization(codes: Tensor, codebook_size: int) -> float:
    """
    Calculate the utilization rate of a codebook.
    
    Args:
        codes: Tensor of code indices, shape [batch_size, levels] or [batch_size]
        codebook_size: Total size of the codebook
        
    Returns:
        Utilization rate as a fraction (0.0 to 1.0)
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    unique_codes = torch.unique(codes)
    return len(unique_codes) / codebook_size


def codebook_perplexity(codes: Tensor, codebook_size: int) -> float:
    """
    Calculate the perplexity of codebook usage.
    
    Args:
        codes: Tensor of code indices, shape [batch_size, levels] or [batch_size]
        codebook_size: Total size of the codebook
        
    Returns:
        Perplexity score
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    # Calculate code frequencies
    codes_flat = codes.flatten()
    counts = torch.bincount(codes_flat, minlength=codebook_size).float()
    probs = counts / counts.sum()
    
    # Remove zero probabilities for numerical stability
    probs = probs[probs > 0]
    
    # Calculate entropy and perplexity
    entropy = -torch.sum(probs * torch.log(probs))
    perplexity = torch.exp(entropy)
    
    return perplexity.item()


def codebook_entropy(codes: Tensor, codebook_size: int) -> float:
    """
    Calculate the entropy of codebook usage distribution.
    
    Args:
        codes: Tensor of code indices, shape [batch_size, levels] or [batch_size]  
        codebook_size: Total size of the codebook
        
    Returns:
        Entropy in bits
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    # Calculate code frequencies
    codes_flat = codes.flatten()
    counts = torch.bincount(codes_flat, minlength=codebook_size).float()
    probs = counts / counts.sum()
    
    # Remove zero probabilities for numerical stability
    probs = probs[probs > 0]
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log2(probs))
    
    return entropy.item()


def dead_code_ratio(codes: Tensor, codebook_size: int) -> float:
    """
    Calculate the ratio of dead codes (unused codes) in the codebook.
    
    Args:
        codes: Tensor of code indices, shape [batch_size, levels] or [batch_size]
        codebook_size: Total size of the codebook
        
    Returns:
        Dead code ratio as a fraction (0.0 to 1.0)
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    unique_codes = torch.unique(codes)
    dead_codes = codebook_size - len(unique_codes)
    
    return dead_codes / codebook_size


def usage_distribution_stats(codes: Tensor, codebook_size: int) -> Dict[str, float]:
    """
    Calculate statistics of codebook usage distribution.
    
    Args:
        codes: Tensor of code indices, shape [batch_size, levels] or [batch_size]
        codebook_size: Total size of the codebook
        
    Returns:
        Dictionary with usage statistics
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    # Calculate code frequencies
    codes_flat = codes.flatten()
    counts = torch.bincount(codes_flat, minlength=codebook_size).float()
    
    # Usage statistics
    used_codes = (counts > 0).sum().item()
    total_codes = codebook_size
    
    # Calculate statistics on usage frequencies
    used_counts = counts[counts > 0]
    
    stats = {
        'utilization': used_codes / total_codes,
        'dead_ratio': (total_codes - used_codes) / total_codes,
        'mean_usage': used_counts.mean().item() if len(used_counts) > 0 else 0.0,
        'std_usage': used_counts.std().item() if len(used_counts) > 0 else 0.0,
        'min_usage': used_counts.min().item() if len(used_counts) > 0 else 0.0,
        'max_usage': used_counts.max().item() if len(used_counts) > 0 else 0.0,
        'perplexity': codebook_perplexity(codes, codebook_size),
        'entropy': codebook_entropy(codes, codebook_size)
    }
    
    return stats


def compute_all_codebook_metrics(codes_per_level: List[Tensor], codebook_size: int) -> Dict[str, Any]:
    """
    Compute comprehensive codebook metrics for all RVQ levels.
    
    Args:
        codes_per_level: List of code tensors for each RVQ level
        codebook_size: Size of each codebook
        
    Returns:
        Dictionary with metrics for each level and overall statistics
    """
    metrics = {
        'per_level': {},
        'overall': {}
    }
    
    all_utilizations = []
    all_perplexities = []
    all_entropies = []
    
    for level, codes in enumerate(codes_per_level):
        level_stats = usage_distribution_stats(codes, codebook_size)
        metrics['per_level'][f'level_{level}'] = level_stats
        
        all_utilizations.append(level_stats['utilization'])
        all_perplexities.append(level_stats['perplexity'])
        all_entropies.append(level_stats['entropy'])
    
    # Overall statistics
    metrics['overall'] = {
        'avg_utilization': np.mean(all_utilizations),
        'avg_perplexity': np.mean(all_perplexities), 
        'avg_entropy': np.mean(all_entropies),
        'min_utilization': np.min(all_utilizations),
        'max_utilization': np.max(all_utilizations),
        'total_levels': len(codes_per_level)
    }
    
    return metrics


def usage_histogram(codes: Tensor, codebook_size: int, num_bins: int = 20) -> Dict[str, Any]:
    """
    Create a histogram of codebook usage for visualization.
    
    Args:
        codes: Tensor of code indices
        codebook_size: Size of codebook
        num_bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data
    """
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    
    # Calculate code frequencies
    codes_flat = codes.flatten()
    counts = torch.bincount(codes_flat, minlength=codebook_size).float()
    
    # Create histogram of usage frequencies
    hist_counts, bin_edges = np.histogram(counts.numpy(), bins=num_bins)
    
    return {
        'hist_counts': hist_counts.tolist(),
        'bin_edges': bin_edges.tolist(),
        'raw_counts': counts.numpy().tolist(),
        'num_used_codes': (counts > 0).sum().item(),
        'num_dead_codes': (counts == 0).sum().item()
    }