"""
Test the improved training with weighted loss for 5D data.
"""

import torch
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.training.improved_losses import weighted_recon_loss

def test_weighted_loss_stability():
    """Test the weighted loss function's stability with extreme velocity values."""
    print("Testing weighted loss function stability...")
    
    # Create test data with extreme velocity values
    batch_size, seq_len = 4, 8
    test_cases = [
        ("face", 68 * 5),      # 340 dims
        ("body", 17 * 5),      # 85 dims
    ]
    
    velocity_scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for part_name, dim in test_cases:
        print(f"\n{part_name} (dim={dim}):")
        print("VelScale    Old Loss    New Loss    Old Grad    New Grad")
        print("-" * 60)
        
        for vel_scale in velocity_scales:
            # Create data
            pred = torch.randn(batch_size, seq_len, dim)
            target = torch.randn(batch_size, seq_len, dim)
            
            # Scale velocity components
            for i in range(0, dim, 5):
                if i + 4 < dim:
                    pred[:, :, i+3:i+5] *= vel_scale
                    target[:, :, i+3:i+5] *= vel_scale
            
            # Test old approach (equal weighting)
            pred1 = pred.clone().requires_grad_(True)
            old_loss = weighted_recon_loss(pred1, target, "mse", 1.0, 1.0)  # Equal weights
            old_loss.backward()
            old_grad_norm = pred1.grad.norm().item()
            
            # Test new approach (weighted)
            pred2 = pred.clone().requires_grad_(True)
            new_loss = weighted_recon_loss(pred2, target, "mse", 1.0, 0.1)  # Weighted
            new_loss.backward()
            new_grad_norm = pred2.grad.norm().item()
            
            print(f"{vel_scale:>7.1f}  {old_loss.item():>10.4f}  {new_loss.item():>10.4f}  {old_grad_norm:>10.4f}  {new_grad_norm:>10.4f}")
            
            # Check improvement
            if new_grad_norm < old_grad_norm * 0.5:
                print("           ^^ Gradient significantly reduced!")

def test_training_simulation():
    """Test a simulated training run with the new loss function."""
    print("\nTesting simulated training with weighted loss...")
    
    # Create simple synthetic model and data
    latent_dim = 64
    face_dim = 68 * 5  # 340
    
    # Simple autoencoder-like model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(face_dim, latent_dim)
            self.decoder = torch.nn.Linear(latent_dim, face_dim)
            
        def forward(self, x):
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            latent = self.encoder(x_flat)
            recon = self.decoder(latent)
            return recon.view(B, T, D)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create data with increasing velocity magnitudes to simulate real training
    def create_batch(vel_scale):
        x = torch.randn(4, 8, face_dim)
        # Scale velocities
        for i in range(0, face_dim, 5):
            if i + 4 < face_dim:
                x[:, :, i+3:i+5] *= vel_scale
        return x
    
    print("Batch   VelScale   Loss      GradNorm")
    print("-" * 40)
    
    losses = []
    grad_norms = []
    
    for batch_idx in range(20):
        # Simulate velocity growth over time (common in real data)
        vel_scale = 1.0 + batch_idx * 0.5  # Gradually increasing velocities
        
        x = create_batch(vel_scale)
        
        optimizer.zero_grad()
        recon = model(x)
        
        # Use weighted loss
        loss = weighted_recon_loss(recon, x, "mse", 1.0, 0.1)
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        
        if batch_idx % 5 == 0:
            print(f"{batch_idx:>5}   {vel_scale:>8.1f}   {loss.item():>8.4f}   {grad_norm:>8.4f}")
    
    # Check for stability
    max_loss = max(losses)
    max_grad = max(grad_norms)
    
    print(f"\nTraining stability check:")
    print(f"  Max loss: {max_loss:.4f}")
    print(f"  Max gradient norm: {max_grad:.4f}")
    
    if max_grad < 10.0:
        print("  ✓ Gradients remained stable!")
    elif max_grad < 100.0:
        print("  ~ Gradients were manageable")
    else:
        print("  ✗ Gradient explosion still occurred")
    
    return max_grad < 100.0

def main():
    """Main test function."""
    print("=" * 70)
    print("Testing Improved Training with Weighted Loss for 5D Data")
    print("=" * 70)
    
    test_weighted_loss_stability()
    
    success = test_training_simulation()
    
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY:")
    print("1. Use weighted_recon_loss() instead of regular recon_loss()")
    print("2. Set position_weight=1.0, velocity_weight=0.1 (or lower)")
    print("3. This prevents velocity components from dominating the loss")
    print("4. Training should be stable even with large velocities")
    if success:
        print("\n✓ The weighted loss approach should solve the gradient explosion!")
    else:
        print("\n✗ May need further adjustments to the loss weights")
    print("=" * 70)

if __name__ == "__main__":
    main()