import torch
from torch import nn
from .encoder import PoseEncoder
from .decoder import PoseDecoder
from .quantizer import ResidualVectorQuantizer


class RVQModel(nn.Module):
    """Encoder -> Residual VQ -> Decoder pipeline."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        codebook_size: int,
        levels: int,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = PoseEncoder(input_dim, latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            levels=levels,
            commitment_beta=commitment_beta,
        )
        self.decoder = PoseDecoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor, part: str):
        """Encode, quantize and decode pose chunks."""
        z = self.encoder(x, part)
        z_q, codes, q_loss = self.quantizer(z)
        recon = self.decoder(z_q, part)
        return recon, codes, q_loss
