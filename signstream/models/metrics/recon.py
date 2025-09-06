from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean squared error loss."""
    return F.mse_loss(pred, target)
