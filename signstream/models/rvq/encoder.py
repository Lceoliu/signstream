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
    """Encode pose chunks with either an MLP or a Transformer backbone."""

    def __init__(
        self,
        frame_dim: int,
        chunk_len: int,
        latent_dim: int,
        arch: str = "mlp",
    ) -> None:
        super().__init__()
        self.arch = arch
        self.chunk_len = chunk_len
        self.frame_dim = frame_dim
        if arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(frame_dim * chunk_len, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )
        elif arch == "transformer":
            self.input_proj = nn.Linear(frame_dim, latent_dim)
            self.pos_embed = nn.Parameter(torch.zeros(chunk_len, latent_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise ValueError(f"Unknown arch: {arch}")
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)

    def forward(self, x: torch.Tensor, part: str) -> torch.Tensor:
        """Encode a batch of pose chunks.

        Args:
            x: Tensor of shape ``[B, L, F]`` where ``F`` is frame_dim.
            part: Body part name.
        Returns:
            Tensor of shape ``[B, latent_dim]``.
        """
        part_id = torch.full(
            (x.shape[0],), PART_TYPES[part], dtype=torch.long, device=x.device
        )
        if self.arch == "mlp":
            h = self.net(x.view(x.shape[0], -1))
            h = h + self.type_embed(part_id)
        else:
            h = self.input_proj(x)
            h = h + self.pos_embed[: x.shape[1]].unsqueeze(0)
            h = h + self.type_embed(part_id).unsqueeze(1)
            h = self.transformer(h)
            h = h.mean(dim=1)
        return h
