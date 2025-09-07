"""
Debug script to identify the gradient explosion cause around 200-300 batches.
"""

import torch
import yaml
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.training.train_rvq import create_dummy_dataset, create_model
from signstream.training.loop_rvq import RVQTrainingLoop
from signstream.training.optim import create_optimizer
from signstream.training.losses import recon_loss
from torch.utils.data import DataLoader
from signstream.data.datasets import CSLDailyDataModule

def analyze_data_statistics(dataset, num_samples=10):
    """Analyze the statistics of the processed data."""
    print("Analyzing data statistics...")
    
    all_values = []
    all_velocities = []
    all_positions = []
    all_confidences = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        for part_name, chunks in sample['chunks'].items():
            if part_name == 'full_body':  # Skip full_body for efficiency
                continue
                
            N, L, K, C = chunks.shape
            print(f"  {part_name}: {chunks.shape}, channels={C}")
            
            # Flatten for analysis
            data = chunks.view(-1, C).numpy()
            
            if C >= 5:  # x, y, confidence, vx, vy
                positions = data[:, :2]  # x, y
                confidence = data[:, 2]   # confidence
                velocities = data[:, 3:5] # vx, vy
                
                all_positions.append(positions)
                all_confidences.append(confidence)
                all_velocities.append(velocities)
                all_values.append(data)
                
                print(f"    Positions - mean: {positions.mean():.4f}, std: {positions.std():.4f}, min: {positions.min():.4f}, max: {positions.max():.4f}")
                print(f"    Confidence - mean: {confidence.mean():.4f}, std: {confidence.std():.4f}, min: {confidence.min():.4f}, max: {confidence.max():.4f}")
                print(f"    Velocities - mean: {velocities.mean():.4f}, std: {velocities.std():.4f}, min: {velocities.min():.4f}, max: {velocities.max():.4f}")
                
                # Check for extreme values
                extreme_vel = np.abs(velocities).max()
                if extreme_vel > 10:
                    print(f"    [WARNING] Extreme velocity detected: {extreme_vel}")
    
    if all_velocities:
        all_velocities = np.concatenate(all_velocities, axis=0)
        print(f"\nOverall velocity statistics:")
        print(f"  Mean: {all_velocities.mean():.6f}")
        print(f"  Std: {all_velocities.std():.6f}")
        print(f"  Min: {all_velocities.min():.6f}")
        print(f"  Max: {all_velocities.max():.6f}")
        print(f"  95th percentile: {np.percentile(all_velocities, 95):.6f}")
        print(f"  99th percentile: {np.percentile(all_velocities, 99):.6f}")
        
        # Count extreme values
        extreme_count = np.sum(np.abs(all_velocities) > 5)
        print(f"  Values > 5: {extreme_count}/{len(all_velocities)} ({100*extreme_count/len(all_velocities):.2f}%)")

def test_loss_stability():
    """Test loss function stability with extreme values."""
    print("\nTesting loss function stability...")
    
    # Test with normal values
    normal_pred = torch.randn(4, 8, 5) * 0.5
    normal_target = torch.randn(4, 8, 5) * 0.5
    
    loss_normal = recon_loss(normal_pred, normal_target, loss_type="mse")
    print(f"Normal values - Loss: {loss_normal.item():.6f}")
    
    # Test with large velocity values
    large_pred = normal_pred.clone()
    large_target = normal_target.clone()
    large_pred[:, :, 3:5] *= 100  # Scale velocities
    large_target[:, :, 3:5] *= 100
    
    loss_large = recon_loss(large_pred, large_target, loss_type="mse")
    print(f"Large velocity values - Loss: {loss_large.item():.6f}")
    
    # Test with very large velocity values
    huge_pred = normal_pred.clone()
    huge_target = normal_target.clone()
    huge_pred[:, :, 3:5] *= 1000  # Scale velocities
    huge_target[:, :, 3:5] *= 1000
    
    loss_huge = recon_loss(huge_pred, huge_target, loss_type="mse")
    print(f"Huge velocity values - Loss: {loss_huge.item():.6f}")
    
    # Test gradient magnitudes
    huge_pred.requires_grad_(True)
    loss_huge.backward()
    grad_norm = huge_pred.grad.norm().item()
    print(f"Gradient norm for huge values: {grad_norm:.6f}")

def simulate_training_batches():
    """Simulate training for many batches to see where explosion happens."""
    print("\nSimulating training batches...")
    
    # Load config
    config_path = "signstream/configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    config["training"]["batch_size"] = 4
    
    # Create model and data
    device = torch.device("cpu")
    train_dataset = create_dummy_dataset(config)
    model = create_model(config, device)
    optimizer = create_optimizer(model, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    model.train()
    batch_losses = []
    grad_norms = []
    
    print("Running batches...")
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 50:  # Test first 50 batches
            break
            
        optimizer.zero_grad()
        total_loss = 0.0
        
        try:
            # Process each body part
            for part in ['face', 'left_hand', 'right_hand', 'body']:
                if part not in batch['chunks']:
                    continue
                    
                x = batch['chunks'][part]
                B, N, L, K, C = x.shape
                x_seq = x.view(B * N, L, K * C).to(device)
                
                # Forward pass
                recon, codes, q_loss, usage_loss, z_q = model(x_seq, part)
                
                # Reconstruction loss
                loss_r = recon_loss(recon, x_seq, loss_type="mse")
                
                part_loss = loss_r + q_loss + usage_loss
                total_loss += part_loss
                
                # Check for issues
                if not torch.isfinite(total_loss):
                    print(f"  [BATCH {batch_idx}] Non-finite loss in {part}: {total_loss.item()}")
                    break
            
            if torch.isfinite(total_loss):
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                optimizer.step()
                
                batch_losses.append(total_loss.item())
                grad_norms.append(grad_norm.item())
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss={total_loss.item():.6f}, GradNorm={grad_norm:.6f}")
                    
                # Check for gradient explosion
                if grad_norm > 100:
                    print(f"  [WARNING] Large gradient at batch {batch_idx}: {grad_norm:.2f}")
                if grad_norm > 1000:
                    print(f"  [CRITICAL] Gradient explosion at batch {batch_idx}: {grad_norm:.2f}")
                    break
                    
        except Exception as e:
            print(f"  [ERROR] Batch {batch_idx} failed: {e}")
            break
    
    print(f"\nCompleted {len(batch_losses)} batches")
    if batch_losses:
        print(f"Loss progression: {batch_losses[0]:.4f} -> {batch_losses[-1]:.4f}")
        print(f"Max gradient norm: {max(grad_norms):.4f}")

def main():
    """Main debug function."""
    print("=" * 70)
    print("Debugging Gradient Explosion Around 200-300 Batches")
    print("=" * 70)
    
    # Create dummy dataset for analysis
    config_path = "signstream/configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    dataset = create_dummy_dataset(config)
    
    # Analyze data
    analyze_data_statistics(dataset, num_samples=5)
    
    # Test loss stability
    test_loss_stability()
    
    # Simulate training
    simulate_training_batches()

if __name__ == "__main__":
    main()