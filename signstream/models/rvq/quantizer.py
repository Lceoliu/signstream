import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer with EMA-updated codebooks and a clean single-pass STE.

    Features:
    - Multi-level residual quantization (RVQ)
    - EMA updates of codebooks (no gradient to codebooks)
    - Single-pass Straight-Through Estimator (STE) at the end
    - Usage regularization via KL(p || Uniform) == logK - H(p)
    - Softmax temperature + logits stabilization
    - Batch renormalization of EMA counts to prevent weight collapse
    - Dead-code reinit (optional)
    - Optional DDP all_reduce on EMA stats
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
        init_mode: str = "uniform_spherical",  # ["uniform_spherical", "uniform_range"]
        init_range_scale: float = 1.0,
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

        # Initialize codebooks
        self._init_codebooks(init_mode, init_range_scale)

    # ---------- Public API ----------

    @torch.no_grad()
    def set_temperature(self, temp: float):
        self.temp = float(temp)

    @torch.no_grad()
    def set_usage_reg_target(self, w: float):
        self.usage_reg_target = float(w)

    # ---------- Forward ----------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] latent vectors

        Returns:
            quantized_final: [B, D] quantized representation
            codes: [B, levels] selected code indices per level
            commit_loss: scalar tensor
            usage_loss: scalar tensor
        """
        assert x.dim() == 2 and x.size(1) == self.dim, "x must be [B, D]"

        B, D = x.shape
        device = x.device
        dtype = x.dtype

        residual = x
        quantized_sum = torch.zeros_like(x)
        codes: List[torch.Tensor] = []

        commit_loss = x.new_tensor(0.0)
        usage_loss = x.new_tensor(0.0)

        # Per-level stats for usage regularization (we use probs.detach())
        for level, emb in enumerate(self.codebooks):
            q, idx, probs = self._quantize_and_ema(residual, emb, level)
            codes.append(idx)  # [B]
            quantized_sum = quantized_sum + q  # accumulate raw q
            prev_input = residual  # x before this level
            residual = residual - q  # residual for next level

            # Commitment loss (EMA path: only pull encoder outputs toward code)
            commit_loss = (
                commit_loss + self.beta * ((prev_input - q.detach()) ** 2).mean()
            )

            # Usage regularization: encourage high entropy (KL to uniform)
            avg_probs = probs.detach().mean(dim=0).clamp_min(self.eps)  # [K]
            entropy = -(avg_probs * avg_probs.log()).sum()
            kl = math.log(self.codebook_size) - entropy
            usage_loss = usage_loss + self._usage_weight() * kl

        # Average losses over levels
        L = max(self.levels, 1)
        commit_loss = commit_loss / L
        usage_loss = usage_loss / L

        # Single-pass STE: forward = quantized_sum, backward = identity to x
        quantized_final = x + (quantized_sum - x).detach()

        # Stack codes
        codes = torch.stack(codes, dim=-1).long()  # [B, levels]

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
        # Distances: ||x||^2 - 2 x·e + ||e||^2
        # x2: [..., 1], xe: [..., K], e2: [K]
        x2 = x.pow(2).sum(dim=-1, keepdim=True)  # [..., 1]
        xe = x @ emb.weight.t()  # [..., K]
        e2 = emb.weight.pow(2).sum(dim=-1)  # [K]
        distances = x2 - 2 * xe + e2  # [..., K]

        # Argmin & gather
        idx = distances.argmin(dim=-1)  # [...]
        q = emb(idx)  # [..., D]

        # Softmax probs for usage stats (stabilized + temperature)
        logits = -distances
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits / self.temp, dim=-1)

        # EMA updates (no grad graph)
        if self.training:
            with torch.no_grad():
                flat_x = x.detach().reshape(-1, self.dim)  # [N, D]
                flat_idx = idx.reshape(-1)  # [N]
                enc = F.one_hot(flat_idx, num_classes=self.codebook_size).to(
                    flat_x.dtype
                )  # [N, K]
                hits = enc.sum(dim=0)  # [K]

                # (Optional) DDP all-reduce stats
                if (
                    self.ddp_allreduce_ema
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    dist.all_reduce(hits, op=dist.ReduceOp.SUM)
                    dw = enc.t().mm(flat_x)
                    dist.all_reduce(dw, op=dist.ReduceOp.SUM)
                else:
                    dw = enc.t().mm(flat_x)

                # Exponential moving averages
                self.cluster_size[level].mul_(self.ema_decay).add_(
                    hits, alpha=1 - self.ema_decay
                )
                self.embed_avg[level].mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )

                # Batch renormalization: keep total "mass" comparable to batch size
                cs = self.cluster_size[level] + self.eps  # [K]
                cs = cs * (flat_x.size(0) / cs.sum().clamp_min(self.eps))  # renorm
                centers = self.embed_avg[level] / cs.unsqueeze(1)  # [K, D]

                # Only write back codes hit in THIS batch (more stable than EMA>0)
                batch_hits = hits > 0
                if batch_hits.any():
                    emb.weight.data[batch_hits] = centers[batch_hits]

        return q, idx, probs

    # ---------- Helpers ----------

    def _usage_weight(self) -> float:
        """Linear warmup for usage regularization weight."""
        if self.usage_warmup_steps <= 0:
            return self.usage_reg_target
        step = int(self.step.item())
        if step >= self.usage_warmup_steps:
            return self.usage_reg_target
        return self.usage_reg_target * (step / float(self.usage_warmup_steps))

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
            else:
                raise ValueError(f"Unknown init_mode: {mode}")

        # Reset EMA stats
        self.cluster_size.zero_()
        self.embed_avg.zero_()

    @torch.no_grad()
    def _dead_code_reinit(self, x: torch.Tensor):
        """
        Reinitialize rarely used codes to random samples from current batch.
        Triggered periodically by `dead_refresh_interval`.
        """
        B = x.shape[0]
        if B == 0:
            return

        for lvl, emb in enumerate(self.codebooks):
            cs = self.cluster_size[lvl]  # [K]
            dead = cs < self.dead_code_threshold
            if not dead.any():
                continue

            num_dead = int(dead.sum().item())
            # Randomly pick batch samples to re-seed dead entries
            idx = torch.randint(0, B, (num_dead,), device=x.device)
            new_vecs = x[idx]  # [num_dead, D]

            emb.weight.data[dead] = new_vecs
            # also seed EMA stats to non-zero small values
            self.embed_avg[lvl][dead] = new_vecs
            self.cluster_size[lvl][dead] = 1.0

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
# 使用示例（放到你的训练脚本里）
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
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
    )

    rvq.train()
    q, codes, closs, uloss = rvq(x)
    print("q.shape:", q.shape)  # [B, D]
    print("codes.shape:", codes.shape)  # [B, levels]
    print("commit_loss:", closs.item(), "usage_loss:", uloss.item())
