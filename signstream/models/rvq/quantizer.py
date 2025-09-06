import torch
from torch import nn
from typing import Tuple

class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with straight-through estimator.

    This module implements a simple multi-level residual vector
    quantization scheme. For an input ``x`` it sequentially quantizes
    the residual using ``levels`` codebooks. The quantized vectors are
    summed to form the final representation and the indices of the
    selected codewords are returned.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        levels: int = 2,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.beta = commitment_beta
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, dim) for _ in range(levels)]
        )
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _quantize(
        self, x: torch.Tensor, emb: nn.Embedding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``x`` with the given codebook and return soft usage probs.

        Args:
            x: Tensor of shape ``[..., dim]`` to quantize.
            emb: Codebook embedding.

        Returns:
            quantized: Quantized tensor with same shape as ``x``.
            indices: Indices of selected codewords with shape ``[...]``.
            probs: Softmax probabilities over the codebook (for usage reg).
        """
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(x, emb.weight.t())
            + emb.weight.pow(2).sum(dim=-1)
        )
        probs = torch.softmax(-distances, dim=-1)
        indices = distances.argmin(dim=-1)
        quantized = emb(indices)
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
            commit_loss: Commitment loss term.
            usage_loss: KL divergence to uniform over codebook usage.
        """
        residual = x
        quantized_sum = torch.zeros_like(x)
        codes = []
        commit_loss = 0.0
        usage_loss = 0.0
        for emb in self.codebooks:
            quantized, idx, probs = self._quantize(residual, emb)
            quantized_sum = quantized_sum + quantized
            codes.append(idx)
            commit_loss = commit_loss + torch.mean((residual.detach() - quantized) ** 2)
            commit_loss = commit_loss + self.beta * torch.mean((residual - quantized.detach()) ** 2)
            usage_loss = usage_loss + (
                probs * (probs * emb.num_embeddings).clamp(min=1e-8).log()
            ).sum(dim=-1).mean()
            residual = residual - quantized
        codes = torch.stack(codes, dim=-1)
        return (
            quantized_sum + (residual - residual.detach()),
            codes,
            commit_loss,
            usage_loss,
        )
