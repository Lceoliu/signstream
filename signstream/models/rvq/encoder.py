import torch
from torch import nn
from typing import Dict

PART_TYPES: Dict[str, int] = {
    "face": 0,
    "left_hand": 1,
    "right_hand": 2,
    "body": 3,
    "full_body": 4,
}


class PoseEncoder(nn.Module):
    """Simple MLP encoder for pose chunks.

    The encoder flattens the spatio-temporal pose representation and
    projects it into a latent space. A learnable embedding is added to
    distinguish different body parts.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)

    def forward(self, x: torch.Tensor, part: str) -> torch.Tensor:
        part_id = torch.full(
            (x.shape[0],), PART_TYPES[part], dtype=torch.long, device=x.device
        )
        h = self.net(x)
        h = h + self.type_embed(part_id)
        return h
