"""
Optimizer and learning rate scheduling utilities.
"""

import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ExponentialLR,
    ReduceLROnPlateau
)
from typing import Dict, Any


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Training configuration dictionary
        
    Returns:
        Configured optimizer
    """
    training_config = config['training']

    optimizer_type = training_config.get('optimizer', 'adamw').lower()
    lr = float(training_config['lr'])
    weight_decay = float(training_config['wd'])

    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=training_config.get('betas', (0.9, 0.999)),
            eps=training_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=training_config.get('betas', (0.9, 0.999)),
            eps=training_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dictionary
        
    Returns:
        Configured scheduler or None
    """
    training_config = config['training']
    scheduler_config = training_config.get('scheduler', {})
    
    if not scheduler_config or scheduler_config.get('type') is None:
        return None
    
    scheduler_type = scheduler_config['type'].lower()
    
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', training_config['epochs']),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            verbose=scheduler_config.get('verbose', True)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


class WarmupScheduler:
    """
    Learning rate warmup scheduler that can wrap other schedulers.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None, metrics=None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase learning rate
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Use base scheduler after warmup
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metrics is not None:
                        self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_warmup_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """
    Create warmup scheduler with optional base scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dictionary
        
    Returns:
        WarmupScheduler or base scheduler
    """
    training_config = config['training']
    warmup_epochs = training_config.get('warmup_epochs', 0)
    
    if warmup_epochs <= 0:
        # No warmup, return base scheduler
        return create_scheduler(optimizer, config)
    
    # Create base scheduler
    base_scheduler = create_scheduler(optimizer, config)
    
    # Wrap with warmup
    return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
