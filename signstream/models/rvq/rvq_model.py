import torch
from torch import nn
from .encoder import PoseEncoder
from .decoder import PoseDecoder
from .quantizer import ResidualVectorQuantizer


class RVQModel(nn.Module):
    """Encoder -> Residual VQ -> Decoder pipeline."""

    def __init__(
        self,
        frame_dim: int,
        chunk_len: int,
        latent_dim: int,
        codebook_size: int,
        levels: int,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        usage_reg: float = 1e-3,
        arch: str = "mlp",
    ) -> None:
        super().__init__()
        self.encoder = PoseEncoder(frame_dim, chunk_len, latent_dim, arch)
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            levels=levels,
            commitment_beta=commitment_beta,
            ema_decay=ema_decay,
            usage_reg=usage_reg,
        )
        self.decoder = PoseDecoder(latent_dim, frame_dim, chunk_len, arch)

    def forward(self, x: torch.Tensor, part: str):
        """Encode, quantize and decode pose chunks."""
        z = self.encoder(x, part)
        z_q, codes, q_loss, usage_loss = self.quantizer(z)
        recon = self.decoder(z_q, part)
        return recon, codes, q_loss, usage_loss, z_q
