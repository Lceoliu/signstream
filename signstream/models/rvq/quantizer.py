import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with EMA updates and straight-through estimator.

    This module implements a multi-level residual vector quantization scheme
    with exponential moving average updates for the codebooks, as described
    in the Rules.md specification.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        levels: int = 3,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        usage_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.beta = commitment_beta
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.usage_reg = usage_reg
        self.dim = dim

        # Codebook embeddings
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, dim) for _ in range(levels)]
        )

        # EMA tracking for codebook updates
        self.register_buffer('cluster_size', torch.zeros(levels, codebook_size))
        self.register_buffer('embed_avg', torch.zeros(levels, codebook_size, dim))

        # Initialize codebooks
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _quantize(
        self, x: torch.Tensor, emb: nn.Embedding, level: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``x`` with the given codebook and EMA updates.

        Args:
            x: Tensor of shape ``[..., dim]`` to quantize.
            emb: Codebook embedding.
            level: Which RVQ level this codebook belongs to.

        Returns:
            quantized: Quantized tensor with same shape as ``x``.
            indices: Indices of selected codewords with shape ``[...]``.
            probs: Softmax probabilities over the codebook (for usage reg).
        """
        # Compute distances to all codebook entries
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(x, emb.weight.t())
            + emb.weight.pow(2).sum(dim=-1)
        )

        # Find nearest neighbors
        indices = distances.argmin(dim=-1)
        quantized = emb(indices)

        # Compute probabilities for usage regularization
        probs = torch.softmax(-distances, dim=-1)

        # EMA updates during training
        if self.training:
            # Flatten for EMA computation
            flat_x = x.view(-1, self.dim)
            flat_indices = indices.view(-1)

            # One-hot encoding of indices
            encodings = F.one_hot(flat_indices, num_classes=self.codebook_size).float()

            # Update cluster sizes
            self.cluster_size[level] = self.cluster_size[level] * self.ema_decay + \
                                     (1 - self.ema_decay) * encodings.sum(0)

            # Update embedding averages
            dw = torch.matmul(encodings.t(), flat_x)
            self.embed_avg[level] = self.embed_avg[level] * self.ema_decay + \
                                   (1 - self.ema_decay) * dw

            # Normalize embeddings
            n = self.cluster_size[level].sum()
            cluster_size = (self.cluster_size[level] + 1e-5) / (n + 1e-5 * self.codebook_size)
            embed_normalized = self.embed_avg[level] / cluster_size.unsqueeze(1)

            # Update codebook weights
            emb.weight.data.copy_(embed_normalized)

        return quantized, indices, probs

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``x`` and compute usage regularization.

        Args:
            x: Input tensor of shape ``[N, D]``.

        Returns:
            quantized: Quantized representation with same shape as ``x``.
            codes: Tensor of shape ``[N, levels]`` containing code indices.
            commit_loss: Commitment loss term (codebook + commitment losses).
            usage_loss: KL divergence to uniform over codebook usage.
        """
        residual = x
        quantized_sum = torch.zeros_like(x)
        codes = []
        commit_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        usage_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for level, emb in enumerate(self.codebooks):
            quantized, idx, probs = self._quantize(residual, emb, level)
            quantized_sum = quantized_sum + quantized
            codes.append(idx)

            # Standard VQ-VAE losses
            # Codebook loss (move embeddings towards inputs)
            commit_loss = commit_loss + torch.mean((residual.detach() - quantized) ** 2)
            # Commitment loss (move inputs towards embeddings)
            commit_loss = commit_loss + self.beta * torch.mean((residual - quantized.detach()) ** 2)

            # Usage regularization: KL divergence from uniform distribution
            uniform_prob = 1.0 / self.codebook_size
            avg_probs = probs.mean(dim=0)  # Average over batch
            kl_div = torch.sum(avg_probs * torch.log(avg_probs / uniform_prob + 1e-8))
            usage_reg_tensor = torch.tensor(
                float(self.usage_reg), device=x.device, dtype=x.dtype
            )
            usage_loss = usage_loss + usage_reg_tensor * kl_div

            # Update residual for next level
            residual = residual - quantized

        codes = torch.stack(codes, dim=-1)

        # Straight-through estimator: add residual gradient
        quantized_final = quantized_sum + (residual - residual.detach())

        return (
            quantized_final,
            codes,
            commit_loss,
            usage_loss,
        )
