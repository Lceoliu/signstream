"""
Improved loss functions that handle 5D pose data (x, y, conf, vx, vy) properly.
"""

import torch
from torch import Tensor
import torch.nn.functional as F

def weighted_recon_loss(
    pred: Tensor, 
    target: Tensor, 
    loss_type: str = "mse",
    position_weight: float = 1.0,
    velocity_weight: float = 0.1,
    ignore_confidence: bool = True,
) -> Tensor:
    """
    Improved reconstruction loss with separate weighting for position and velocity.
    
    Args:
        pred: Predicted tensor [B, T, D] where D = K * 5
        target: Target tensor [B, T, D] where D = K * 5  
        loss_type: "mse" or "huber"
        position_weight: Weight for position loss (x, y)
        velocity_weight: Weight for velocity loss (vx, vy)
        ignore_confidence: Whether to ignore confidence channel
    
    Returns:
        Weighted reconstruction loss
    """
    B, T, D = pred.shape
    
    # Check if D is divisible by 5 (x, y, conf, vx, vy per keypoint)
    if D % 5 != 0:
        # Fallback to original loss if not 5D data
        if ignore_confidence and D % 3 == 0:
            # 3D data (x, y, conf)
            K = D // 3
            mask = torch.ones(3, device=pred.device)
            mask[2] = 0.0  # Ignore confidence
            mask = mask.view(*(1,) * (pred.dim() - 1), 3).expand_as(pred.view(B, T, K, 3))
            diff = (pred.view(B, T, K, 3) - target.view(B, T, K, 3)) * mask
            diff = diff.view(B, T, D)
        else:
            diff = pred - target
            
        if loss_type == "huber":
            return F.smooth_l1_loss(diff, torch.zeros_like(diff))
        else:
            return F.mse_loss(diff, torch.zeros_like(diff))
    
    # Handle 5D data
    K = D // 5
    
    # Reshape to [B, T, K, 5] for easier processing
    pred_reshaped = pred.view(B, T, K, 5)
    target_reshaped = target.view(B, T, K, 5)
    
    # Split into components: [x, y, conf, vx, vy]
    pred_pos = pred_reshaped[:, :, :, :2]     # Position: x, y
    pred_vel = pred_reshaped[:, :, :, 3:5]    # Velocity: vx, vy
    target_pos = target_reshaped[:, :, :, :2]
    target_vel = target_reshaped[:, :, :, 3:5]
    
    # Compute separate losses
    if loss_type == "huber":
        pos_loss = F.smooth_l1_loss(pred_pos, target_pos)
        vel_loss = F.smooth_l1_loss(pred_vel, target_vel)
    else:  # mse
        pos_loss = F.mse_loss(pred_pos, target_pos)
        vel_loss = F.mse_loss(pred_vel, target_vel)
    
    # Weighted combination
    total_loss = position_weight * pos_loss + velocity_weight * vel_loss
    
    return total_loss


def adaptive_recon_loss(
    pred: Tensor,
    target: Tensor, 
    loss_type: str = "mse",
    base_pos_weight: float = 1.0,
    base_vel_weight: float = 0.1,
    adaptive_scaling: bool = True,
) -> Tensor:
    """
    Adaptive reconstruction loss that adjusts velocity weight based on velocity magnitude.
    
    Args:
        pred: Predicted tensor [B, T, D]
        target: Target tensor [B, T, D]
        loss_type: "mse" or "huber"
        base_pos_weight: Base weight for position loss
        base_vel_weight: Base weight for velocity loss
        adaptive_scaling: Whether to adaptively scale velocity weight
    
    Returns:
        Adaptive reconstruction loss
    """
    B, T, D = pred.shape
    
    if D % 5 != 0:
        # Fallback for non-5D data
        return weighted_recon_loss(pred, target, loss_type, base_pos_weight, base_vel_weight)
    
    K = D // 5
    pred_reshaped = pred.view(B, T, K, 5)
    target_reshaped = target.view(B, T, K, 5)
    
    pred_pos = pred_reshaped[:, :, :, :2]
    pred_vel = pred_reshaped[:, :, :, 3:5]
    target_pos = target_reshaped[:, :, :, :2]
    target_vel = target_reshaped[:, :, :, 3:5]
    
    # Compute losses
    if loss_type == "huber":
        pos_loss = F.smooth_l1_loss(pred_pos, target_pos)
        vel_loss = F.smooth_l1_loss(pred_vel, target_vel)
    else:
        pos_loss = F.mse_loss(pred_pos, target_pos)
        vel_loss = F.mse_loss(pred_vel, target_vel)
    
    # Adaptive velocity weight based on velocity magnitude
    vel_weight = base_vel_weight
    if adaptive_scaling:
        # Compute average velocity magnitude
        avg_vel_mag = torch.mean(torch.sqrt(target_vel.pow(2).sum(dim=-1)))
        
        # Scale down velocity weight if velocities are large
        if avg_vel_mag > 1.0:
            vel_weight = base_vel_weight / torch.clamp(avg_vel_mag, min=1.0, max=10.0)
        
        # Ensure minimum weight to still learn velocities
        vel_weight = torch.clamp(vel_weight, min=0.01, max=1.0)
    
    total_loss = base_pos_weight * pos_loss + vel_weight * vel_loss
    
    return total_loss