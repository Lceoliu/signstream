"""Loss utilities for RVQ training."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from signstream.models.metrics.temporal import temporal_loss as _temporal_loss


def recon_loss(pred: Tensor, target: Tensor, loss_type: str = "huber") -> Tensor:
    """Reconstruction loss using Huber or MSE."""
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def temporal_loss(latents: Tensor) -> Tensor:
    """Wrapper for temporal smoothness loss."""
    return _temporal_loss(latents)


def usage_regularization(prob_loss: Tensor, weight: float) -> Tensor:
    """Apply weighting to codebook usage regularization."""
    return weight * prob_loss
