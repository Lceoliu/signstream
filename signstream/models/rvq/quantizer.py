import math
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer with EMA-updated codebooks and clean single-pass STE.

    Features:
    - Multi-level residual quantization (RVQ)
    - EMA updates of codebooks (no gradient to codebooks)
    - Single-pass Straight-Through Estimator (STE) at the end
    - Usage regularization via KL(p || Uniform) == logK - H(p)
    - Softmax temperature + logits stabilization
    - Batch renormalization of EMA counts to prevent weight collapse
    - Dead-code reinit (optional)
    - Optional DDP all_reduce on EMA stats
    - Codebook orthogonality regularization (optional)
    - Perplexity tracking for monitoring
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        levels: int = 3,
        commitment_beta: float = 0.25,  # commitment weight
        ema_decay: float = 0.99,  # EMA momentum
        usage_reg: float = 1e-3,  # target weight after warmup
        usage_warmup_steps: int = 0,  # linear warmup steps for usage_reg
        temp: float = 1.0,  # softmax temperature for probs stats
        eps: float = 1e-5,
        # dead-code handling
        enable_dead_code_reinit: bool = True,
        dead_code_threshold: float = 1e-3,  # small EMA count regarded as "dead"
        dead_refresh_interval: int = 1000,  # steps between scans
        ddp_allreduce_ema: bool = True,  # all_reduce EMA stats in DDP
        init_mode: str = "uniform_spherical",  # ["uniform_spherical", "uniform_range", "kaiming"]
        init_range_scale: float = 1.0,
        # Additional features
        orthogonal_reg: float = 0.0,  # orthogonality regularization weight
        track_perplexity: bool = True,  # track codebook usage perplexity
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.levels = levels

        self.beta = float(commitment_beta)
        self.ema_decay = float(ema_decay)
        self.usage_reg = float(usage_reg)
        self.usage_reg_target = float(usage_reg)
        self.usage_warmup_steps = int(usage_warmup_steps)
        self.temp = float(temp)
        self.eps = float(eps)

        self.enable_dead_code_reinit = bool(enable_dead_code_reinit)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_refresh_interval = int(dead_refresh_interval)
        self.ddp_allreduce_ema = bool(ddp_allreduce_ema)

        self.orthogonal_reg = float(orthogonal_reg)
        self.track_perplexity = bool(track_perplexity)

        # Codebooks: turn off grad (EMA will update weights)
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, dim) for _ in range(levels)]
        )
        for emb in self.codebooks:
            emb.weight.requires_grad_(False)

        # EMA buffers
        self.register_buffer("cluster_size", torch.zeros(levels, codebook_size))
        self.register_buffer("embed_avg", torch.zeros(levels, codebook_size, dim))

        # Training step counter for warmup & dead-code schedules
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

        # Perplexity tracking buffer
        if self.track_perplexity:
            self.register_buffer("perplexity", torch.zeros(levels))

        # Initialize codebooks
        self._init_codebooks(init_mode, init_range_scale)

    # ---------- Public API ----------

    @torch.no_grad()
    def set_temperature(self, temp: float):
        """Dynamically adjust temperature during training."""
        self.temp = float(temp)

    @torch.no_grad()
    def set_usage_reg_target(self, w: float):
        """Dynamically adjust usage regularization weight."""
        self.usage_reg_target = float(w)

    @torch.no_grad()
    def get_codebook_utilization(self) -> torch.Tensor:
        """Get the utilization rate for each codebook level."""
        with torch.no_grad():
            utilization = (
                (self.cluster_size > self.dead_code_threshold).float().mean(dim=1)
            )
        return utilization  # [levels]

    def get_metrics(self) -> Dict[str, Any]:
        """Get various metrics for monitoring."""
        metrics = {}
        if self.track_perplexity:
            metrics["perplexity"] = self.perplexity.cpu().numpy()
        metrics["utilization"] = self.get_codebook_utilization().cpu().numpy()
        metrics["step"] = self.step.item()
        return metrics

    # ---------- Forward ----------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] or [B, T, D] latent vectors

        Returns:
            quantized_final: same shape as x, quantized representation
            codes: [B, levels] or [B, T, levels] selected code indices per level
            commit_loss: scalar tensor
            usage_loss: scalar tensor (includes orthogonal_loss if enabled)
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(-1, D)  # [B*T, D]
        elif x.dim() == 2:
            B, D = x.shape
            T = 1
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        assert D == self.dim, f"Dimension mismatch: expected {self.dim}, got {D}"

        device = x.device
        dtype = x.dtype

        residual = x
        quantized_sum = torch.zeros_like(x)
        codes: List[torch.Tensor] = []

        commit_loss = x.new_tensor(0.0)
        usage_loss = x.new_tensor(0.0)
        orthogonal_loss = x.new_tensor(0.0)

        # Per-level quantization
        for level, emb in enumerate(self.codebooks):
            q, idx, probs = self._quantize_and_ema(residual, emb, level)
            codes.append(idx)  # [B*T]
            quantized_sum = quantized_sum + q  # accumulate raw q
            prev_input = residual  # x before this level
            residual = residual - q  # residual for next level

            # Commitment loss (EMA path: only pull encoder outputs toward code)
            commit_loss = commit_loss + self.beta * F.mse_loss(prev_input, q.detach())

            # Usage regularization: encourage high entropy (KL to uniform)
            avg_probs = probs.detach().mean(dim=0).clamp_min(self.eps)  # [K]
            entropy = -(avg_probs * avg_probs.log()).sum()
            kl = math.log(self.codebook_size) - entropy
            usage_loss = usage_loss + self._usage_weight() * kl

            # Track perplexity
            if self.track_perplexity and self.training:
                with torch.no_grad():
                    perplexity = torch.exp(entropy)
                    self.perplexity[level] = perplexity

            # Orthogonality regularization (optional)
            if self.orthogonal_reg > 0:
                orthogonal_loss = orthogonal_loss + self._orthogonal_loss(emb.weight)

        # Average losses over levels
        L = max(self.levels, 1)
        commit_loss = commit_loss / L
        usage_loss = usage_loss / L

        if self.orthogonal_reg > 0:
            orthogonal_loss = orthogonal_loss / L
            usage_loss = usage_loss + self.orthogonal_reg * orthogonal_loss

        # Single-pass STE: forward = quantized_sum, backward = identity to x
        quantized_final = x + (quantized_sum - x).detach()

        # Stack codes and reshape back
        codes = torch.stack(codes, dim=-1).long()  # [B*T, levels]

        if original_shape[0:2] == (B, T) and T > 1:
            quantized_final = quantized_final.reshape(B, T, D)
            codes = codes.reshape(B, T, self.levels)

        # Step bookkeeping + dead-code maintenance
        if self.training:
            self.step += 1
            if self.enable_dead_code_reinit and (
                int(self.step.item()) % self.dead_refresh_interval == 0
            ):
                self._dead_code_reinit(x)

        return quantized_final, codes, commit_loss, usage_loss

    # ---------- Core ops ----------

    def _quantize_and_ema(
        self, x: torch.Tensor, emb: nn.Embedding, level: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [..., D]
        Returns:
          q: [..., D]
          idx: [...]
          probs: [..., K]  (softmax over codebook; for usage stats)
        """
        # Compute L2 distances using Einstein summation (more efficient)
        distances = (
            torch.cdist(x.unsqueeze(1), emb.weight.unsqueeze(0), p=2).squeeze(1) ** 2
        )

        # Argmin & gather
        idx = distances.argmin(dim=-1)  # [...]
        q = emb(idx)  # [..., D]

        # Softmax probs for usage stats (stabilized + temperature)
        logits = -distances
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits / max(self.temp, self.eps), dim=-1)

        # EMA updates (no grad graph)
        if self.training:
            self._update_ema(x, idx, level)

        return q, idx, probs

    @torch.no_grad()
    def _update_ema(self, x: torch.Tensor, idx: torch.Tensor, level: int):
        """Update EMA statistics for the codebook."""
        flat_x = x.detach().reshape(-1, self.dim)  # [N, D]
        flat_idx = idx.reshape(-1)  # [N]

        # One-hot encoding
        enc = F.one_hot(flat_idx, num_classes=self.codebook_size).to(
            flat_x.dtype
        )  # [N, K]
        hits = enc.sum(dim=0)  # [K]
        dw = enc.t().mm(flat_x)  # [K, D]

        # (Optional) DDP all-reduce stats
        if self.ddp_allreduce_ema and dist.is_available() and dist.is_initialized():
            dist.all_reduce(hits, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

        # Exponential moving averages with Laplace smoothing
        self.cluster_size[level].mul_(self.ema_decay).add_(
            hits, alpha=1 - self.ema_decay
        )
        self.embed_avg[level].mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Batch renormalization with stability improvements
        cs = self.cluster_size[level] + self.eps  # [K]
        total_count = cs.sum()
        if total_count > 0:
            # Scale to maintain expected batch size
            target_sum = flat_x.size(0)
            cs = cs * (target_sum / total_count.clamp_min(1.0))

        centers = self.embed_avg[level] / cs.unsqueeze(1).clamp_min(self.eps)  # [K, D]

        # Update only codes with sufficient statistics
        significant_hits = self.cluster_size[level] > self.eps
        if significant_hits.any():
            self.codebooks[level].weight.data[significant_hits] = centers[
                significant_hits
            ]

    # ---------- Helpers ----------

    def _usage_weight(self) -> float:
        """Linear warmup for usage regularization weight."""
        if self.usage_warmup_steps <= 0:
            return self.usage_reg_target
        step = int(self.step.item())
        if step >= self.usage_warmup_steps:
            return self.usage_reg_target
        return self.usage_reg_target * (step / float(self.usage_warmup_steps))

    def _orthogonal_loss(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality regularization loss."""
        # Normalize vectors
        normalized = F.normalize(weight, dim=-1)
        # Compute cosine similarity matrix
        cosine_sim = torch.mm(normalized, normalized.t())
        # Remove diagonal (self-similarity)
        eye = torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype)
        off_diagonal = (cosine_sim - eye).abs()
        # Return mean of off-diagonal elements
        return off_diagonal.mean()

    @torch.no_grad()
    def _init_codebooks(self, mode: str, scale: float):
        """Initialize codebook weights and EMA buffers."""
        for l, emb in enumerate(self.codebooks):
            if mode == "uniform_spherical":
                # Sample Gaussian then normalize to unit sphere; scale optional
                w = torch.randn_like(emb.weight)
                w = F.normalize(w, dim=-1) * scale
                emb.weight.copy_(w)
            elif mode == "uniform_range":
                bound = scale / math.sqrt(self.dim)
                nn.init.uniform_(emb.weight, -bound, bound)
            elif mode == "kaiming":
                nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))
                emb.weight.mul_(scale)
            else:
                raise ValueError(f"Unknown init_mode: {mode}")

        # Initialize EMA stats with small positive values for stability
        self.cluster_size.fill_(1.0 / self.codebook_size)
        self.embed_avg.copy_(
            self.codebooks[0].weight.unsqueeze(0).expand(self.levels, -1, -1)
        )

    @torch.no_grad()
    def _dead_code_reinit(self, x: torch.Tensor):
        """
        Reinitialize rarely used codes to random samples from current batch.
        Triggered periodically by `dead_refresh_interval`.
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))

        B = x.shape[0]
        if B == 0:
            return

        for lvl, emb in enumerate(self.codebooks):
            cs = self.cluster_size[lvl]  # [K]
            dead = cs < self.dead_code_threshold
            if not dead.any():
                continue

            num_dead = int(dead.sum().item())
            # Use different samples for each dead code
            num_samples = min(num_dead, B)
            idx = torch.randperm(B, device=x.device)[:num_samples]

            if num_samples < num_dead:
                # If we have more dead codes than batch samples, repeat samples
                idx = idx.repeat((num_dead // num_samples) + 1)[:num_dead]

            new_vecs = x[idx]  # [num_dead, D]

            # Add small noise to avoid exact duplicates
            noise = torch.randn_like(new_vecs) * 0.01
            new_vecs = new_vecs + noise

            emb.weight.data[dead] = new_vecs
            # Reset EMA stats for dead codes
            self.embed_avg[lvl][dead] = new_vecs
            self.cluster_size[lvl][dead] = 1.0 / self.codebook_size

    # ---------- Decode function ----------

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode from code indices to vectors.

        Args:
            codes: [B, levels] or [B, T, levels] code indices

        Returns:
            decoded: [B, D] or [B, T, D] decoded vectors
        """
        original_shape = codes.shape
        if codes.dim() == 3:
            B, T, L = codes.shape
            codes = codes.reshape(-1, L)  # [B*T, levels]
        elif codes.dim() == 2:
            B, L = codes.shape
            T = 1
        else:
            raise ValueError(f"Expected 2D or 3D codes, got {codes.dim()}D")

        assert L == self.levels, f"Level mismatch: expected {self.levels}, got {L}"

        decoded = torch.zeros(
            codes.size(0), self.dim, device=codes.device, dtype=torch.float32
        )

        for level in range(self.levels):
            level_codes = codes[:, level]  # [B*T]
            level_emb = self.codebooks[level](level_codes)  # [B*T, D]
            decoded = decoded + level_emb

        if T > 1:
            decoded = decoded.reshape(B, T, self.dim)

        return decoded

    # ---------- Convenience ----------

    @torch.no_grad()
    def remove_codebooks_from_optimizer(self, optim):
        """Optional helper: drop codebook params from an existing optimizer param_groups."""
        to_skip = set()
        for emb in self.codebooks:
            to_skip.add(id(emb.weight))
        for g in optim.param_groups:
            kept = []
            for p in g["params"]:
                if id(p) not in to_skip:
                    kept.append(p)
            g["params"] = kept


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Test 2D input
    B, D = 8, 16
    x = torch.randn(B, D)

    rvq = ResidualVectorQuantizer(
        dim=D,
        codebook_size=256,
        levels=3,
        commitment_beta=0.25,
        ema_decay=0.99,
        usage_reg=1e-3,
        usage_warmup_steps=1000,  # 前 1000 step 线性 warmup 到 1e-3
        temp=1.5,  # 早期平滑统计，后期可 set_temperature(1.0)
        enable_dead_code_reinit=True,
        dead_code_threshold=1e-3,
        dead_refresh_interval=1000,
        ddp_allreduce_ema=True,
        init_mode="uniform_spherical",
        orthogonal_reg=0.01,  # 添加正交性正则化
        track_perplexity=True,
    )

    rvq.train()
    q, codes, closs, uloss = rvq(x)
    print("2D Input Test:")
    print("q.shape:", q.shape)  # [B, D]
    print("codes.shape:", codes.shape)  # [B, levels]
    print("commit_loss:", closs.item(), "usage_loss:", uloss.item())

    # Test decode
    decoded = rvq.decode(codes)
    print("decoded.shape:", decoded.shape)
    print("reconstruction error:", F.mse_loss(q, decoded).item())

    # Test 3D input (sequence)
    print("\n3D Input Test:")
    B, T, D = 4, 10, 16
    x_seq = torch.randn(B, T, D)
    q_seq, codes_seq, closs_seq, uloss_seq = rvq(x_seq)
    print("q_seq.shape:", q_seq.shape)  # [B, T, D]
    print("codes_seq.shape:", codes_seq.shape)  # [B, T, levels]

    # Get metrics
    metrics = rvq.get_metrics()
    print("\nMetrics:", metrics)
