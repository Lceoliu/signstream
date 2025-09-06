import torch
from torch import nn
from typing import Dict
from .encoder import PART_TYPES, PART_DIMENSIONS, SharedPoseBackbone


class PoseDecoder(nn.Module):
    """Multi-body-part pose decoder with shared backbone, mirroring PoseEncoder architecture."""

    def __init__(
        self,
        latent_dim: int = 256,
        chunk_len: int = 10,
        type_embed_dim: int = 16,
        num_layers: int = 2,
        arch: str = "transformer",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.chunk_len = chunk_len
        self.latent_dim = latent_dim
        
        # Shared backbone (same as encoder)
        self.backbone = SharedPoseBackbone(
            latent_dim=latent_dim,
            num_layers=num_layers,
            arch=arch,
            dropout=dropout,
        )
        
        # Type embeddings for different body parts
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)
        
        # Output projection layers for each body part (different output dimensions)
        self.output_projections = nn.ModuleDict()
        for part_name, num_keypoints in PART_DIMENSIONS.items():
            frame_dim = num_keypoints * 3  # x, y, confidence
            if arch == "mlp":
                # For MLP: project to flattened output
                self.output_projections[part_name] = nn.Linear(
                    latent_dim, frame_dim * chunk_len
                )
            else:
                # For transformer: project per-frame
                self.output_projections[part_name] = nn.Linear(latent_dim, frame_dim)
        
        # Positional embeddings for transformer
        if arch == "transformer":
            self.pos_embed = nn.Parameter(torch.zeros(chunk_len, latent_dim))
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(latent_dim)

    def _get_part_output_dim(self, part: str) -> int:
        """Get output dimension for a specific body part."""
        if part not in PART_DIMENSIONS:
            raise ValueError(f"Unknown body part: {part}")
        return PART_DIMENSIONS[part] * 3  # x, y, confidence

    def _expand_to_temporal(self, z: torch.Tensor) -> torch.Tensor:
        """Expand latent vector to temporal dimension for transformer."""
        # z: [B, D] -> [B, L, D]
        B, D = z.shape
        return z.unsqueeze(1).expand(-1, self.chunk_len, -1)

    def forward(self, z: torch.Tensor, part: str) -> torch.Tensor:
        """Decode latent representation back to pose chunk for a specific body part.

        Args:
            z: Latent tensor of shape [B, latent_dim] representing chunk-level encoding
            part: Body part name ("face", "left_hand", "right_hand", "body", "full_body")
            
        Returns:
            Tensor of shape [B, L, F] where:
                - B: batch size
                - L: chunk length (number of frames)
                - F: frame dimension (num_keypoints * 3)
        """
        if len(z.shape) != 2:
            raise ValueError(f"Expected 2D latent input [B, D], got {z.shape}")
        
        B, D = z.shape
        if D != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got D={D}")
        
        # Get type embedding
        part_id = torch.full(
            (B,), PART_TYPES[part], dtype=torch.long, device=z.device
        )
        type_emb = self.type_embed(part_id)  # [B, D]
        
        # Add type embedding to input
        h = z + type_emb  # [B, D]
        
        if self.arch == "mlp":
            # Pass through shared backbone
            h = self.backbone(h)  # [B, D]
            
            # Layer norm
            h = self.layer_norm(h)
            
            # Output projection (specific to body part)
            out = self.output_projections[part](h)  # [B, L*F]
            
            # Reshape to temporal format
            frame_dim = self._get_part_output_dim(part)
            out = out.view(B, self.chunk_len, frame_dim)  # [B, L, F]
            
        elif self.arch == "transformer":
            # Expand to temporal dimension
            h = self._expand_to_temporal(h)  # [B, L, D]
            
            # Add positional embeddings
            h = h + self.pos_embed.unsqueeze(0)  # [B, L, D]
            
            # Add type embeddings (broadcast across time)
            h = h + type_emb.unsqueeze(1)  # [B, L, D]
            
            # Pass through shared backbone
            h = self.backbone(h)  # [B, L, D]
            
            # Layer norm
            h = self.layer_norm(h)
            
            # Output projection (specific to body part)
            out = self.output_projections[part](h)  # [B, L, F]
        
        return out

    def get_part_output_dim(self, part: str) -> int:
        """Get the output frame dimension for a specific body part."""
        return self._get_part_output_dim(part)
