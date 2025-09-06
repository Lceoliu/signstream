from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean squared error loss."""
    return F.mse_loss(pred, target)


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber (smooth L1) loss."""
    return F.smooth_l1_loss(pred, target, beta=delta)
