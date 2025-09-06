import torch
from torch import nn
from typing import Dict
from .encoder import PART_TYPES


class PoseDecoder(nn.Module):
    """Decoder matching :class:`PoseEncoder` architecture."""

    def __init__(
        self,
        latent_dim: int,
        frame_dim: int,
        chunk_len: int,
        arch: str = "mlp",
    ) -> None:
        super().__init__()
        self.arch = arch
        self.chunk_len = chunk_len
        self.frame_dim = frame_dim
        if arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, frame_dim * chunk_len),
            )
        elif arch == "transformer":
            self.pos_embed = nn.Parameter(torch.zeros(chunk_len, latent_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_proj = nn.Linear(latent_dim, frame_dim)
        else:
            raise ValueError(f"Unknown arch: {arch}")
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)

    def forward(self, z: torch.Tensor, part: str) -> torch.Tensor:
        part_id = torch.full(
            (z.shape[0],), PART_TYPES[part], dtype=torch.long, device=z.device
        )
        if self.arch == "mlp":
            h = z + self.type_embed(part_id)
            out = self.net(h)
            return out.view(z.shape[0], self.chunk_len, self.frame_dim)
        else:
            h = z.unsqueeze(1).expand(-1, self.chunk_len, -1)
            h = h + self.pos_embed.unsqueeze(0)
            h = h + self.type_embed(part_id).unsqueeze(1)
            h = self.transformer(h)
            out = self.output_proj(h)
            return out
