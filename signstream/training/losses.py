"""Loss utilities for RVQ training."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from signstream.models.metrics.temporal import temporal_loss as _temporal_loss


def recon_loss(
    pred: Tensor, target: Tensor, loss_type: str = "huber", ignore_index: int = 2
) -> Tensor:
    """Reconstruction loss using Huber or MSE.
    pred and target are expected to have shape [B, T, D]
    where D is the keypoint dimension (e.g., 5 for x, y, confidence, vx, vy).
    here we ignore the confidence dimension (index 2) in loss computation.
    """
    D = pred.size(-1)
    mask = torch.ones(D, device=pred.device)
    mask[ignore_index] = 0.0

    # 扩展到 pred 的 shape
    mask = mask.view(*(1,) * (pred.dim() - 1), D)

    diff = (pred - target) * mask
    if loss_type == "huber":
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))
    return F.mse_loss(diff, torch.zeros_like(diff))


def temporal_loss(latents: Tensor) -> Tensor:
    """Wrapper for temporal smoothness loss."""
    return _temporal_loss(latents)


def usage_regularization(prob_loss: Tensor, weight: float) -> Tensor:
    """Apply weighting to codebook usage regularization."""
    return weight * prob_loss
