"""
Simple debug script to test loss function behavior with 5D data.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.training.losses import recon_loss

def test_loss_with_5d_data():
    """Test the loss function with 5D data to identify gradient explosion."""
    print("Testing loss function with 5D data...")
    
    # Create test data with different velocity scales
    batch_size, seq_len = 4, 8
    
    # Different body part sizes
    test_cases = [
        ("face", 68 * 5),      # 340 dims
        ("left_hand", 21 * 5), # 105 dims  
        ("body", 17 * 5),      # 85 dims
    ]
    
    velocity_scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for part_name, dim in test_cases:
        print(f"\nTesting {part_name} (dim={dim}):")
        
        for vel_scale in velocity_scales:
            # Create synthetic data: [B, T, D] where D = K * 5
            pred = torch.randn(batch_size, seq_len, dim)
            target = torch.randn(batch_size, seq_len, dim)
            
            # Scale velocity components (indices 3, 4, 8, 9, 13, 14, ...)
            # Every 5th and 6th element are velocities
            for i in range(0, dim, 5):
                if i + 4 < dim:  # Make sure we have vx, vy components
                    pred[:, :, i+3:i+5] *= vel_scale  # vx, vy
                    target[:, :, i+3:i+5] *= vel_scale
            
            # Test loss with different loss types
            for loss_type in ["mse", "huber"]:
                try:
                    loss = recon_loss(pred, target, loss_type=loss_type)
                    
                    # Test gradient
                    pred.requires_grad_(True)
                    loss.backward()
                    grad_norm = pred.grad.norm().item()
                    
                    print(f"  Vel scale {vel_scale:>6.1f}, {loss_type:>5s}: Loss={loss.item():>10.4f}, GradNorm={grad_norm:>10.4f}")
                    
                    if grad_norm > 1000:
                        print(f"    [CRITICAL] Gradient explosion detected!")
                    elif grad_norm > 100:
                        print(f"    [WARNING] Large gradient detected!")
                    
                    # Reset gradient
                    pred.grad = None
                    pred.requires_grad_(False)
                    
                except Exception as e:
                    print(f"    [ERROR] {loss_type} failed: {e}")

def test_loss_masking():
    """Test if the confidence masking is working correctly."""
    print("\nTesting loss masking (ignore confidence channel)...")
    
    batch_size, seq_len, dim = 2, 4, 15  # 3 keypoints * 5 channels
    
    pred = torch.randn(batch_size, seq_len, dim)
    target = torch.randn(batch_size, seq_len, dim)
    
    # Set confidence channels (indices 2, 7, 12) to very different values
    pred[:, :, 2] = 100.0   # High confidence in pred
    target[:, :, 2] = 0.0   # Low confidence in target
    pred[:, :, 7] = 100.0
    target[:, :, 7] = 0.0
    pred[:, :, 12] = 100.0
    target[:, :, 12] = 0.0
    
    loss = recon_loss(pred, target, loss_type="mse", ignore_index=2)
    print(f"Loss with confidence masking: {loss.item():.6f}")
    
    # Test without masking by changing ignore_index
    loss_no_mask = recon_loss(pred, target, loss_type="mse", ignore_index=-1)
    print(f"Loss without confidence masking: {loss_no_mask.item():.6f}")
    
    # The masked loss should be much smaller
    if loss.item() < loss_no_mask.item():
        print("✓ Confidence masking is working correctly")
    else:
        print("✗ Confidence masking may not be working properly")

def test_improved_loss_function():
    """Test an improved loss function that handles 5D data better."""
    print("\nTesting improved loss function...")
    
    def improved_recon_loss(pred, target, loss_type="mse", position_weight=1.0, velocity_weight=0.1):
        """Improved reconstruction loss with separate weighting for position and velocity."""
        B, T, D = pred.shape
        
        # Assuming D = K * 5 where each keypoint has [x, y, conf, vx, vy]
        K = D // 5
        
        # Reshape to [B, T, K, 5]
        pred_reshaped = pred.view(B, T, K, 5)
        target_reshaped = target.view(B, T, K, 5)
        
        # Split into components
        pred_pos = pred_reshaped[:, :, :, :2]     # x, y
        pred_vel = pred_reshaped[:, :, :, 3:5]    # vx, vy
        target_pos = target_reshaped[:, :, :, :2]
        target_vel = target_reshaped[:, :, :, 3:5]
        
        # Compute separate losses
        if loss_type == "mse":
            pos_loss = torch.nn.functional.mse_loss(pred_pos, target_pos)
            vel_loss = torch.nn.functional.mse_loss(pred_vel, target_vel)
        else:  # huber
            pos_loss = torch.nn.functional.smooth_l1_loss(pred_pos, target_pos)
            vel_loss = torch.nn.functional.smooth_l1_loss(pred_vel, target_vel)
        
        # Weighted combination
        total_loss = position_weight * pos_loss + velocity_weight * vel_loss
        return total_loss
    
    # Test with extreme velocity values
    pred = torch.randn(4, 8, 85)  # Body part: 17 * 5
    target = torch.randn(4, 8, 85)
    
    # Scale velocities to be extreme
    for i in range(0, 85, 5):
        if i + 4 < 85:
            pred[:, :, i+3:i+5] *= 1000  # Extreme velocities
            target[:, :, i+3:i+5] *= 1000
    
    # Compare old vs new loss
    old_loss = recon_loss(pred, target, loss_type="mse")
    new_loss = improved_recon_loss(pred, target, loss_type="mse")
    
    print(f"Old loss (equal weighting): {old_loss.item():.6f}")
    print(f"New loss (weighted): {new_loss.item():.6f}")
    
    # Test gradients
    pred1 = pred.clone().requires_grad_(True)
    pred2 = pred.clone().requires_grad_(True)
    
    old_loss = recon_loss(pred1, target, loss_type="mse")
    old_loss.backward()
    old_grad_norm = pred1.grad.norm().item()
    
    new_loss = improved_recon_loss(pred2, target, loss_type="mse")
    new_loss.backward()
    new_grad_norm = pred2.grad.norm().item()
    
    print(f"Old gradient norm: {old_grad_norm:.6f}")
    print(f"New gradient norm: {new_grad_norm:.6f}")
    
    if new_grad_norm < old_grad_norm * 0.5:
        print("✓ Improved loss function reduces gradient magnitude")
    
    return improved_recon_loss

def main():
    """Main debug function."""
    print("=" * 60)
    print("Debugging Loss Function with 5D Data (x, y, conf, vx, vy)")
    print("=" * 60)
    
    test_loss_with_5d_data()
    test_loss_masking() 
    improved_loss_fn = test_improved_loss_function()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("1. Velocity values can cause gradient explosion when they're large")
    print("2. Consider separate weighting for position vs velocity in loss")
    print("3. May need to normalize/clip velocity values in data preprocessing")
    print("4. Current masking (ignore confidence) appears to work correctly")
    print("=" * 60)

if __name__ == "__main__":
    main()