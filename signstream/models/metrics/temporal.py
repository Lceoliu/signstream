import torch
from torch import Tensor


def temporal_loss(latents: Tensor) -> Tensor:
    """Penalize differences between consecutive latent vectors.

    Args:
        latents: Tensor of shape ``[B, N, D]`` representing quantized latents
            per chunk.
    Returns:
        Scalar temporal smoothness loss.
    """
    diff = latents[:, 1:, :] - latents[:, :-1, :]
    return torch.mean(diff.pow(2))
