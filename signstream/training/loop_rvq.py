"""
Training and validation loops for RVQ model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import gc

from .utils_tb import RVQTensorBoardLogger

from ..models.rvq.rvq_model import RVQModel
from ..models.metrics.codebook import compute_all_codebook_metrics
from .losses import recon_loss, temporal_loss, usage_regularization
from .improved_losses import weighted_recon_loss, adaptive_recon_loss


logger = logging.getLogger(__name__)


class RVQTrainingLoop:
    """Training loop for RVQ model with comprehensive metrics and logging."""

    def __init__(
        self,
        model: RVQModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        logger_writer: RVQTensorBoardLogger = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger_writer = logger_writer

        # Training parameters
        self.training_config = config['training']
        self.model_config = config['model']
        self.epochs = self.training_config['epochs']
        self.save_every = self.training_config.get('save_every', 5)
        self.eval_every = self.training_config.get('eval_every', 1)
        self.sample_every = self.training_config.get('sample_every', 5)
        self.num_eval_samples = self.training_config.get('num_eval_samples', 5)

        # Stability parameters
        self.max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
        self.loss_scale_factor = self.training_config.get('loss_scale_factor', 1.0)

        # Loss weights
        self.temporal_alpha = self.model_config['rvq'].get('temporal_loss_alpha', 0.05)
        self.enable_temporal_loss = self.model_config['rvq'].get(
            'enable_temporal_loss', True
        )
        self.position_weight = self.model_config['rvq'].get('position_weight', 1.0)
        self.velocity_weight = self.model_config['rvq'].get('velocity_weight', 0.1)
        self.usage_beta = self.model_config['rvq'].get('usage_reg', 1e-3)

        # Checkpoint directory
        self.checkpoint_dir = Path(config['save_path'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0

        # Body parts to train on
        self.body_parts = ['face', 'left_hand', 'right_hand', 'body', 'full_body']
        
        # DDP compatibility flag (can be overridden in subclasses)
        self.is_ddp = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            part: {
                'recon_loss': 0.0,
                'q_loss': 0.0,
                'usage_loss': 0.0,
                'temporal_loss': 0.0,
                'total_loss': 0.0,
            }
            for part in self.body_parts
        }

        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            total_loss = 0.0
            batch_metrics = {}

            # Process each body part
            for part in self.body_parts:
                if part not in batch['chunks']:
                    continue

                # Get chunks for this body part [B, N, L, K, C]
                x = batch['chunks'][part]
                B, N, L, K, C = x.shape

                # Mask padded chunks
                mask = batch.get('chunk_mask', None)
                if mask is not None:
                    mask = mask.to(self.device)  # [B, N]
                else:
                    mask = torch.ones(B, N, dtype=torch.bool, device=self.device)

                # Reshape to [B*N, L, K*C] and mask valid chunks only
                x_seq_all = x.view(B * N, L, K * C).to(self.device)
                mask_flat = mask.view(B * N)
                if mask_flat.any():
                    x_seq = x_seq_all[mask_flat]
                else:
                    continue  # no valid chunks

                # Forward pass
                recon, codes, q_loss, usage_loss, z_q = self.model(x_seq, part)

                # Reconstruction loss with stability checks - using improved weighted loss
                loss_r = weighted_recon_loss(
                    recon,
                    x_seq,
                    loss_type="mse",
                    position_weight=self.position_weight,
                    velocity_weight=self.velocity_weight,
                    ignore_confidence=True,
                )

                # Check for NaN in individual losses
                if not torch.isfinite(loss_r):
                    logger.warning(
                        f"Non-finite reconstruction loss for {part}, skipping"
                    )
                    continue
                if not torch.isfinite(q_loss):
                    logger.warning(f"Non-finite quantization loss for {part}, skipping")
                    continue
                if not torch.isfinite(usage_loss):
                    logger.warning(f"Non-finite usage loss for {part}, skipping")
                    continue

                # Temporal consistency loss (disabled by default; would require per-sample masking)
                # With padding involved, temporal loss is unreliable; keep disabled unless fully masked per sample
                loss_t = torch.tensor(0.0, device=self.device)

                # Total loss for this part
                part_loss = loss_r + q_loss + usage_loss + self.temporal_alpha * loss_t
                total_loss += part_loss

                # Track metrics
                batch_metrics[part] = {
                    'recon_loss': loss_r.item(),
                    'q_loss': q_loss.item(),
                    'usage_loss': usage_loss.item(),
                    'temporal_loss': loss_t.item(),
                    'total_loss': part_loss.item(),
                }

            # Check for NaN/Inf losses before backward pass
            if not torch.isfinite(total_loss):
                logger.warning(
                    f"Non-finite loss detected: {total_loss.item()}, skipping batch"
                )
                continue

            # Mixed precision support
            amp_mode = str(self.training_config.get('amp', 'fp32')).lower()
            use_cuda = self.device.type == 'cuda'
            if amp_mode == 'fp16' and use_cuda:
                from torch.cuda.amp import autocast, GradScaler
                if not hasattr(self, '_scaler'):
                    self._scaler = GradScaler()
                with autocast(dtype=torch.float16):
                    scaled_loss = total_loss * self.loss_scale_factor
                # Backward with scaler
                self._scaler.scale(scaled_loss).backward()
                # Unscale before clipping
                if self.max_grad_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if grad_norm > self.max_grad_norm * 2:
                        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                # bf16 or fp32 path (bf16 autocast is safe without scaler)
                if amp_mode == 'bf16' and use_cuda:
                    from torch.cuda.amp import autocast
                    with autocast(dtype=torch.bfloat16):
                        scaled_loss = total_loss * self.loss_scale_factor
                        scaled_loss.backward()
                else:
                    # fp32
                    scaled_loss = total_loss * self.loss_scale_factor
                    scaled_loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if grad_norm > self.max_grad_norm * 2:
                        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")

                self.optimizer.step()

            # Accumulate metrics
            for part in self.body_parts:
                if part in batch_metrics:
                    for metric_name, value in batch_metrics[part].items():
                        epoch_metrics[part][metric_name] += value

            # Log batch progress
            if batch_idx % 100 == 0:
                total_loss_val = float(total_loss.detach().item())
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, "
                    f"Total Loss: {total_loss_val:.4f}"
                )
            
            # Memory cleanup more frequently to prevent gradual accumulation
            if batch_idx % 50 == 0:
                # Clean up intermediate variables
                del batch_metrics
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Average metrics over batches
        for part in self.body_parts:
            for metric_name in epoch_metrics[part]:
                epoch_metrics[part][metric_name] /= total_batches

        # gc to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = {
            part: {
                'recon_loss': 0.0,
                'q_loss': 0.0,
                'usage_loss': 0.0,
                'temporal_loss': 0.0,
                'total_loss': 0.0,
            }
            for part in self.body_parts
        }

        # Collect codes for codebook analysis
        all_codes = {part: [] for part in self.body_parts}
        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}/{self.epochs}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch_metrics = {}

                # Process each body part
                for part in self.body_parts:
                    if part not in batch['chunks']:
                        continue

                    # Get chunks for this body part
                    x = batch['chunks'][part].detach()
                    B, N, L, K, C = x.shape

                    # Mask padded chunks
                    mask = batch.get('chunk_mask', None)
                    if mask is not None:
                        mask = mask.to(self.device)  # [B, N]
                    else:
                        mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                    x_seq_all = x.view(B * N, L, K * C).to(self.device)
                    mask_flat = mask.view(B * N)
                    if not mask_flat.any():
                        continue
                    x_seq = x_seq_all[mask_flat]

                    # Forward pass
                    recon, codes, q_loss, usage_loss, z_q = self.model(x_seq, part)

                    # Store codes for analysis (valid chunks only) - detach from graph
                    all_codes[part].append(codes.detach().cpu().clone())

                    # Compute losses using improved weighted loss
                    loss_r = weighted_recon_loss(
                        recon,
                        x_seq,
                        loss_type="mse",
                        position_weight=self.position_weight,
                        velocity_weight=self.velocity_weight,
                        ignore_confidence=True,
                    )
                    if N > 1 and self.enable_temporal_loss and self.temporal_alpha > 0:
                        z_q_seq = z_q.view(B, N, -1)
                        loss_t = temporal_loss(z_q_seq)
                    else:
                        loss_t = torch.tensor(0.0, device=self.device)

                    part_loss = (
                        loss_r + q_loss + usage_loss + self.temporal_alpha * loss_t
                    )

                    # Track metrics
                    batch_metrics[part] = {
                        'recon_loss': loss_r.item(),
                        'q_loss': q_loss.item(),
                        'usage_loss': usage_loss.item(),
                        'temporal_loss': loss_t.item(),
                        'total_loss': part_loss.item(),
                    }

                # Accumulate metrics
                for part in self.body_parts:
                    if part in batch_metrics:
                        for metric_name, value in batch_metrics[part].items():
                            epoch_metrics[part][metric_name] += value

        # Average metrics
        num_batches = len(self.val_loader)
        for part in self.body_parts:
            for metric_name in epoch_metrics[part]:
                epoch_metrics[part][metric_name] /= num_batches

        # Compute codebook metrics
        codebook_metrics = {}
        codebook_size = self.model_config['rvq']['codebook_size']

        for part in self.body_parts:
            if all_codes[part]:
                part_codes = torch.cat(
                    all_codes[part], dim=0
                )  # [total_samples, levels]

                # Split by levels for analysis
                codes_per_level = [part_codes[:, i] for i in range(part_codes.shape[1])]
                part_metrics = compute_all_codebook_metrics(
                    codes_per_level, codebook_size
                )
                codebook_metrics[part] = part_metrics

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_metrics, codebook_metrics

    def sample_tokens(self, epoch: int) -> Dict[str, List]:
        """Sample token outputs for a few validation examples."""
        self.model.eval()
        samples = {}

        with torch.no_grad():
            # Get a batch from validation
            batch = next(iter(self.val_loader))

            for part in self.body_parts[: self.num_eval_samples]:  # Limit samples
                if part not in batch['chunks']:
                    continue

                x = batch['chunks'][part]
                B, N, L, K, C = x.shape

                # Take first sample from batch
                x_sample = x[0:1].view(1 * N, L, K * C).to(self.device)

                # Get tokens
                _, codes, _, _, _ = self.model(x_sample, part)

                samples[part] = {
                    'codes': codes.cpu().tolist(),
                    'video_name': (
                        batch['names'][0] if 'names' in batch else f'sample_{epoch}'
                    ),
                    'num_chunks': N,
                }

        return samples

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        # Regular checkpoint
        if epoch % self.save_every == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(
            f"Model: {sum(p.numel() for p in self.model.parameters())} parameters"
        )

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.eval_every == 0:
                with torch.no_grad():
                    val_metrics, codebook_metrics = self.validate_epoch(epoch)

                    # Calculate average validation loss for checkpointing
                    avg_val_loss = np.mean(
                        [metrics['total_loss'] for metrics in val_metrics.values()]
                    )

                    is_best = avg_val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = avg_val_loss

                    # Log metrics
                    self._log_metrics(
                        epoch, train_metrics, val_metrics, codebook_metrics
                    )

                    # Save checkpoint
                    all_metrics = {
                        'train': train_metrics,
                        'val': val_metrics,
                        'codebook': codebook_metrics,
                    }
                    self.save_checkpoint(epoch, all_metrics, is_best)

            # Sample tokens
            if epoch % self.sample_every == 0:
                with torch.no_grad():
                    token_samples = self.sample_tokens(epoch)
                    self._log_token_samples(epoch, token_samples)

            # Free caches - explicitly clear tensor references
            del (
                train_metrics,
                val_metrics,
                codebook_metrics,
                all_metrics,
            )
            try:
                del token_samples
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_metrics(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict, codebook_metrics: Dict
    ):
        """Log training metrics."""
        logger.info(f"\n=== Epoch {epoch} Results ===")

        # Training metrics
        logger.info("Training Metrics:")
        for part, metrics in train_metrics.items():
            logger.info(
                f"  {part}: Recon: {metrics['recon_loss']:.4f}, "
                f"Q: {metrics['q_loss']:.4f}, "
                f"Usage: {metrics['usage_loss']:.4f}, "
                f"Temporal: {metrics['temporal_loss']:.4f}, "
                f"Total: {metrics['total_loss']:.4f}"
            )

        # Validation metrics
        logger.info("Validation Metrics:")
        for part, metrics in val_metrics.items():
            logger.info(
                f"  {part}: Recon: {metrics['recon_loss']:.4f}, "
                f"Q: {metrics['q_loss']:.4f}, "
                f"Usage: {metrics['usage_loss']:.4f}, "
                f"Temporal: {metrics['temporal_loss']:.4f}, "
                f"Total: {metrics['total_loss']:.4f}"
            )

        # Codebook metrics
        logger.info("Codebook Health:")
        for part, cb_metrics in codebook_metrics.items():
            overall = cb_metrics['overall']
            logger.info(
                f"  {part}: Util: {overall['avg_utilization']:.2%}, "
                f"Perplexity: {overall['avg_perplexity']:.1f}, "
                f"Entropy: {overall['avg_entropy']:.2f}"
            )

        # TensorBoard logging
        if self.logger_writer:
            self._write_tensorboard_metrics(
                epoch, train_metrics, val_metrics, codebook_metrics
            )

    def _log_token_samples(self, epoch: int, samples: Dict):
        """Log sample token outputs."""
        logger.info(f"\nToken Samples (Epoch {epoch}):")
        for part, sample in samples.items():
            codes = sample['codes']
            logger.info(
                f"  {part} ({sample['video_name']}): {codes[:3]}..."
            )  # Show first 3 chunks

    def _write_tensorboard_metrics(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict, codebook_metrics: Dict
    ):
        """Write metrics to TensorBoard."""
        if not self.logger_writer:
            return

        # Training metrics
        for part, metrics in train_metrics.items():
            for metric_name, value in metrics.items():
                self.logger_writer.log_scalar(
                    f'train/{part}_{metric_name}', value, epoch
                )

        # Validation metrics
        for part, metrics in val_metrics.items():
            for metric_name, value in metrics.items():
                self.logger_writer.log_scalar(f'val/{part}_{metric_name}', value, epoch)

        # Codebook metrics
        for part, cb_metrics in codebook_metrics.items():
            overall = cb_metrics['overall']
            for metric_name, value in overall.items():
                self.logger_writer.log_scalar(
                    f'codebook/{part}_{metric_name}', value, epoch
                )

            # Per-level metrics
            for level_name, level_metrics in cb_metrics['per_level'].items():
                for metric_name, value in level_metrics.items():
                    self.logger_writer.log_scalar(
                        f'codebook/{part}_{level_name}_{metric_name}', value, epoch
                    )
    
    def _train_epoch(self) -> Dict[str, float]:
        """Standard method name for DDP compatibility. Calls train_epoch.""" 
        return self.train_epoch(self.current_epoch)
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Standard method name for DDP compatibility. Calls validate_epoch."""
        val_metrics, _ = self.validate_epoch(self.current_epoch)
        return val_metrics
        
    def _process_batch(self, batch: Dict, is_training: bool = True) -> Dict[str, float]:
        """Process a single batch and return loss metrics (DDP compatibility)."""
        if is_training:
            self.optimizer.zero_grad()
            
        total_loss = 0.0
        batch_metrics = {}
        
        # Process each body part
        for part in self.body_parts:
            if part not in batch['chunks']:
                continue

            # Get chunks for this body part [B, N, L, K, C]
            x = batch['chunks'][part]
            B, N, L, K, C = x.shape

            # Mask padded chunks
            mask = batch.get('chunk_mask', None)
            if mask is not None:
                mask = mask.to(self.device)  # [B, N]
            else:
                mask = torch.ones(B, N, dtype=torch.bool, device=self.device)

            # Reshape to [B*N, L, K*C] and mask valid chunks only
            x_seq_all = x.view(B * N, L, K * C).to(self.device)
            mask_flat = mask.view(B * N)
            if mask_flat.any():
                x_seq = x_seq_all[mask_flat]
            else:
                continue  # no valid chunks

            # Forward pass
            if is_training:
                recon, codes, q_loss, usage_loss, z_q = self.model(x_seq, part)
            else:
                with torch.no_grad():
                    recon, codes, q_loss, usage_loss, z_q = self.model(x_seq, part)

            # Reconstruction loss using improved weighted loss
            from .improved_losses import weighted_recon_loss
            loss_r = weighted_recon_loss(
                recon,
                x_seq,
                loss_type="mse",
                position_weight=self.position_weight,
                velocity_weight=self.velocity_weight,
                ignore_confidence=True,
            )

            # Stability checks
            if not torch.isfinite(loss_r) or not torch.isfinite(q_loss) or not torch.isfinite(usage_loss):
                continue

            # Temporal loss (disabled by default for padding)
            loss_t = torch.tensor(0.0, device=self.device)

            # Total loss for this part
            part_loss = loss_r + q_loss + usage_loss + self.temporal_alpha * loss_t
            total_loss += part_loss

            # Track metrics
            batch_metrics[part + '_recon_loss'] = loss_r.item()
            batch_metrics[part + '_q_loss'] = q_loss.item()
            batch_metrics[part + '_usage_loss'] = usage_loss.item()
            batch_metrics[part + '_temporal_loss'] = loss_t.item()
            batch_metrics[part + '_total_loss'] = part_loss.item()

        # Backward pass for training
        if is_training and torch.isfinite(total_loss):
            # Handle mixed precision as in train_epoch
            amp_mode = str(self.training_config.get('amp', 'fp32')).lower()
            use_cuda = self.device.type == 'cuda'
            
            if amp_mode == 'fp16' and use_cuda:
                from torch.cuda.amp import autocast, GradScaler
                if not hasattr(self, '_scaler'):
                    self._scaler = GradScaler()
                with autocast(dtype=torch.float16):
                    scaled_loss = total_loss * self.loss_scale_factor
                self._scaler.scale(scaled_loss).backward()
                if self.max_grad_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                if amp_mode == 'bf16' and use_cuda:
                    from torch.cuda.amp import autocast
                    with autocast(dtype=torch.bfloat16):
                        scaled_loss = total_loss * self.loss_scale_factor
                        scaled_loss.backward()
                else:
                    scaled_loss = total_loss * self.loss_scale_factor
                    scaled_loss.backward()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        batch_metrics['total_loss'] = total_loss.item() if torch.isfinite(total_loss) else 0.0
        return batch_metrics
