import torch
from torch import nn
from typing import Dict
from .encoder import PART_TYPES


class PoseDecoder(nn.Module):
    """Mirror of :class:`PoseEncoder` to reconstruct pose chunks."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)

    def forward(self, z: torch.Tensor, part: str) -> torch.Tensor:
        part_id = torch.full(
            (z.shape[0],), PART_TYPES[part], dtype=torch.long, device=z.device
        )
        h = z + self.type_embed(part_id)
        out = self.net(h)
        return out
