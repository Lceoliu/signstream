import torch
from torch import nn
from .encoder import PoseEncoder, PART_DIMENSIONS
from .decoder import PoseDecoder
from .quantizer import ResidualVectorQuantizer


class RVQModel(nn.Module):
    """Multi-body-part Encoder -> Residual VQ -> Decoder pipeline with shared backbone."""

    def __init__(
        self,
        latent_dim: int = 256,
        chunk_len: int = 10,
        codebook_size: int = 1024,
        levels: int = 3,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        usage_reg: float = 1e-3,
        arch: str = "transformer",
        num_layers: int = 2,
        type_embed_dim: int = 16,
        dropout: float = 0.1,
        temporal_aggregation: str = "mean",
    ) -> None:
        """
        Initialize RVQ model with shared backbone for multi-body-part pose processing.
        
        Args:
            latent_dim: Latent space dimension
            chunk_len: Number of frames per chunk
            codebook_size: Size of each RVQ codebook
            levels: Number of RVQ levels
            commitment_beta: Commitment loss weight
            ema_decay: EMA decay rate for codebook updates
            usage_reg: Usage regularization weight
            arch: Architecture type ("mlp" or "transformer")
            num_layers: Number of layers in shared backbone
            type_embed_dim: Dimension of type embeddings
            dropout: Dropout rate
            temporal_aggregation: How to aggregate temporal dimension ("mean", "max", "attention")
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.chunk_len = chunk_len
        self.arch = arch

        # Shared encoder for all body parts
        self.encoder = PoseEncoder(
            latent_dim=latent_dim,
            chunk_len=chunk_len,
            type_embed_dim=type_embed_dim,
            num_layers=num_layers,
            arch=arch,
            dropout=dropout,
            temporal_aggregation=temporal_aggregation,
        )

        # RVQ quantizer (shared across body parts)
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            levels=levels,
            commitment_beta=commitment_beta,
            ema_decay=ema_decay,
            usage_reg=usage_reg,
        )

        # Shared decoder for all body parts
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            chunk_len=chunk_len,
            type_embed_dim=type_embed_dim,
            num_layers=num_layers,
            arch=arch,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, part: str):
        """
        Forward pass: encode, quantize and decode pose chunks for a specific body part.

        Args:
            x: Input pose tensor of shape [B, L, F] where:
                - B: batch size
                - L: chunk length (frames)
                - F: frame dimension (keypoints * 5: x, y, confidence, vx, vy)
            part: Body part name ("face", "left_hand", "right_hand", "body", "full_body")

        Returns:
            recon: Reconstructed pose tensor, same shape as input
            codes: Quantization codes, shape [B, levels]
            q_loss: VQ commitment loss (scalar)
            usage_loss: Usage regularization loss (scalar)
            z_q: Quantized latent representation, shape [B, latent_dim]
        """
        # Validate part
        if part not in PART_DIMENSIONS:
            raise ValueError(f"Unknown body part: {part}. Available: {list(PART_DIMENSIONS.keys())}")

        # Support both 3-channel (x,y,conf) and 5-channel (x,y,conf,vx,vy) inputs
        B, L, F = x.shape
        K = PART_DIMENSIONS[part]
        expect5 = K * 5
        expect3 = K * 3

        input_was_3ch = False
        if F == expect5:
            x5 = x
        elif F == expect3:
            # Expand to 5D by appending zero velocities
            input_was_3ch = True
            x3 = x.view(B, L, K, 3)
            zeros_vel = torch.zeros(B, L, K, 2, device=x.device, dtype=x.dtype)
            x5 = torch.cat([x3, zeros_vel], dim=-1).view(B, L, expect5)
        else:
            raise ValueError(
                f"Unexpected frame_dim for part {part}: got {F}, expected {expect5} (5ch) or {expect3} (3ch)"
            )

        # Encode: [B, L, K*5] -> [B, D]
        z = self.encoder(x5, part)

        # Quantize: [B, D] -> [B, D], codes, losses
        z_q, codes, q_loss, usage_loss = self.quantizer(z)

        # Decode: [B, D] -> [B, L, K*5]
        recon5 = self.decoder(z_q, part)

        # If original input was 3-channel, slice back to 3 channels; else keep 5
        if input_was_3ch:
            recon5_view = recon5.view(B, L, K, 5)
            recon = recon5_view[:, :, :, :3].contiguous().view(B, L, expect3)
        else:
            recon = recon5

        return recon, codes, q_loss, usage_loss, z_q

    def encode_only(self, x: torch.Tensor, part: str) -> torch.Tensor:
        """Encode input to latent space without quantization."""
        return self.encoder(x, part)

    def quantize_only(self, z: torch.Tensor):
        """Quantize latent vectors."""
        return self.quantizer(z)

    def decode_only(self, z_q: torch.Tensor, part: str) -> torch.Tensor:
        """Decode quantized latents to pose space."""
        return self.decoder(z_q, part)

    def get_part_frame_dim(self, part: str) -> int:
        """Get frame dimension for a specific body part."""
        return self.encoder.get_part_frame_dim(part)

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            'latent_dim': self.latent_dim,
            'chunk_len': self.chunk_len,
            'arch': self.arch,
            'codebook_size': self.quantizer.codebook_size,
            'levels': self.quantizer.levels,
            'commitment_beta': self.quantizer.beta,
            'ema_decay': self.quantizer.ema_decay,
            'usage_reg': self.quantizer._usage_weight(),
            'supported_parts': list(PART_DIMENSIONS.keys()),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
