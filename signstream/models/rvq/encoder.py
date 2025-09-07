import torch
from torch import nn
from typing import Dict, Tuple

# Body part types according to CSL-Daily format
PART_TYPES: Dict[str, int] = {
    "face": 0,           # 24-91: 68 keypoints
    "left_hand": 1,      # 92-112: 21 keypoints  
    "right_hand": 2,     # 113-133: 21 keypoints
    "body": 3,           # 1-17: 17 keypoints
    "full_body": 4,      # 1-133: 133 keypoints
}

# Body part dimensions (keypoints per part)
PART_DIMENSIONS: Dict[str, int] = {
    "face": 68,          # 91 - 24 + 1 = 68
    "left_hand": 21,     # 112 - 92 + 1 = 21
    "right_hand": 21,    # 133 - 113 + 1 = 21
    "body": 17,          # 17 - 1 + 1 = 17
    "full_body": 133,    # 133 - 1 + 1 = 133
}


class SharedPoseBackbone(nn.Module):
    """Shared backbone for all body parts with configurable architecture."""
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 2,
        arch: str = "transformer",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.arch = arch
        self.latent_dim = latent_dim
        
        if arch == "mlp":
            # Simple MLP backbone
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    # First layer will be handled by input projection
                    continue
                self.layers.append(nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
                
        elif arch == "transformer":
            # Lightweight Transformer backbone
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,  # More heads for better representation
                dim_feedforward=latent_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared backbone.
        
        Args:
            x: Input tensor of shape [B, L, D] where D is latent_dim
            
        Returns:
            Encoded tensor of shape [B, L, D] or [B, D] depending on architecture
        """
        if self.arch == "mlp":
            # For MLP, we expect flattened input [B, L*D] -> [B, D]
            h = x
            for layer in self.layers:
                h = layer(h)
            return h
        elif self.arch == "transformer":
            # For transformer, keep temporal dimension [B, L, D]
            h = self.transformer(x)
            return h


class PoseEncoder(nn.Module):
    """Multi-body-part pose encoder with shared backbone and type embeddings."""

    def __init__(
        self,
        latent_dim: int = 256,
        chunk_len: int = 10,
        type_embed_dim: int = 16,
        num_layers: int = 2,
        arch: str = "transformer",
        dropout: float = 0.1,
        temporal_aggregation: str = "mean",  # "mean", "max", "attention"
    ) -> None:
        super().__init__()
        self.arch = arch
        self.chunk_len = chunk_len
        self.latent_dim = latent_dim
        self.temporal_aggregation = temporal_aggregation

        # Input projection layers for each body part (different input dimensions)
        self.input_projections = nn.ModuleDict()
        for part_name, num_keypoints in PART_DIMENSIONS.items():
            frame_dim = num_keypoints * 5  # x, y, confidence, vx, vy
            if arch == "mlp":
                # For MLP: flatten temporal dimension
                self.input_projections[part_name] = nn.Linear(
                    frame_dim * chunk_len, latent_dim
                )
            else:
                # For transformer: project per-frame
                self.input_projections[part_name] = nn.Linear(frame_dim, latent_dim)

        # Shared backbone
        self.backbone = SharedPoseBackbone(
            latent_dim=latent_dim,
            num_layers=num_layers,
            arch=arch,
            dropout=dropout,
        )

        # Type embeddings for different body parts
        self.type_embed = nn.Embedding(len(PART_TYPES), latent_dim)

        # Positional embeddings for transformer
        if arch == "transformer":
            self.pos_embed = nn.Parameter(torch.zeros(chunk_len, latent_dim))

        # Temporal attention for aggregation
        if temporal_aggregation == "attention":
            self.temporal_attention = nn.MultiheadAttention(
                latent_dim, num_heads=4, batch_first=True
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(latent_dim)

    def _get_part_input_dim(self, part: str) -> int:
        """Get input dimension for a specific body part."""
        if part not in PART_DIMENSIONS:
            raise ValueError(f"Unknown body part: {part}")
        return PART_DIMENSIONS[part] * 5  # x, y, confidence, vx, vy

    def _validate_input_shape(self, x: torch.Tensor, part: str) -> None:
        """Validate input tensor shape for the given body part."""
        expected_frame_dim = self._get_part_input_dim(part)

        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input [B, L, F], got {x.shape}")

        B, L, F = x.shape
        if L != self.chunk_len:
            raise ValueError(f"Expected chunk_len={self.chunk_len}, got L={L}")

        if F != expected_frame_dim:
            raise ValueError(
                f"Expected frame_dim={expected_frame_dim} for {part}, got F={F}"
            )

    def _aggregate_temporal(self, h: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal dimension to get chunk-level representation."""
        if self.temporal_aggregation == "mean":
            return h.mean(dim=1)  # [B, L, D] -> [B, D]
        elif self.temporal_aggregation == "max":
            return h.max(dim=1)[0]  # [B, L, D] -> [B, D]
        elif self.temporal_aggregation == "attention":
            # Use attention with a learnable CLS token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

            # Attention: CLS token attends to all temporal positions
            attended, _ = self.temporal_attention(cls_tokens, h, h)
            return attended.squeeze(1)  # [B, 1, D] -> [B, D]
        else:
            raise ValueError(f"Unknown temporal aggregation: {self.temporal_aggregation}")

    def forward(self, x: torch.Tensor, part: str) -> torch.Tensor:
        """Encode a batch of pose chunks for a specific body part.

        Args:
            x: Tensor of shape [B, L, F] where:
                - B: batch size
                - L: chunk length (number of frames)  
                - F: frame dimension (num_keypoints * 3)
            part: Body part name ("face", "left_hand", "right_hand", "body", "full_body")
            
        Returns:
            Tensor of shape [B, latent_dim] representing chunk-level encoding.
        """
        # Validate input
        self._validate_input_shape(x, part)

        B, L, F = x.shape

        # Get type embedding
        part_id = torch.full(
            (B,), PART_TYPES[part], dtype=torch.long, device=x.device
        )
        type_emb = self.type_embed(part_id)  # [B, D]

        # Input projection (different for each body part)
        if self.arch == "mlp":
            # Flatten temporal dimension and project
            x_flat = x.view(B, -1)  # [B, L*F]
            h = self.input_projections[part](x_flat)  # [B, D]

            # Add type embedding
            h = h + type_emb  # [B, D]

            # Pass through shared backbone
            h = self.backbone(h)  # [B, D]

            # Layer norm
            h = self.layer_norm(h)

        elif self.arch == "transformer":
            # Project each frame
            h = self.input_projections[part](x)  # [B, L, D]

            # Add positional embeddings
            h = h + self.pos_embed[:L].unsqueeze(0)  # [B, L, D]

            # Add type embeddings (broadcast across time)
            h = h + type_emb.unsqueeze(1)  # [B, L, D]

            # Pass through shared backbone
            h = self.backbone(h)  # [B, L, D]

            # Temporal aggregation
            h = self._aggregate_temporal(h)  # [B, D]

            # Layer norm
            h = self.layer_norm(h)

        return h

    def get_part_frame_dim(self, part: str) -> int:
        """Get the frame dimension for a specific body part."""
        return self._get_part_input_dim(part)
