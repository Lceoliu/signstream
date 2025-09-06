"""
Test script to verify training stability fixes.
"""

import torch
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.train_rvq import create_dummy_dataset

def test_stability_fixes():
    """Test the stability fixes work correctly."""
    print("Testing stability fixes...")
    
    # Load config with stability parameters
    config_path = "signstream/configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"Stability parameters:")
    print(f"  max_grad_norm: {config['training']['max_grad_norm']}")
    print(f"  loss_scale_factor: {config['training']['loss_scale_factor']}")
    print(f"  lr: {config['training']['lr']}")
    print(f"  commitment_beta: {config['model']['rvq']['commitment_beta']}")
    print(f"  usage_reg: {config['model']['rvq']['usage_reg']}")
    
    # Create model with new parameters
    model = RVQModel(
        latent_dim=config["model"]["latent_dim"],
        chunk_len=config["data"]["chunk_len"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"]["commitment_beta"],
        ema_decay=config["model"]["rvq"]["ema_decay"],
        usage_reg=config["model"]["rvq"]["usage_reg"],
        arch=config["model"]["arch"],
        num_layers=config["model"].get("encoder_layer", 2),
        type_embed_dim=config["model"].get("type_embed_dim", 16),
        dropout=config["model"].get("dropout", 0.1),
        temporal_aggregation=config["model"].get("temporal_aggregation", "mean"),
    )
    
    print(f"Model created with {model.get_model_info()['total_parameters']:,} parameters")
    
    # Test with extreme inputs that might cause instability
    test_cases = [
        {"name": "normal", "data": torch.randn(4, 10, 204)},
        {"name": "large_values", "data": torch.randn(4, 10, 204) * 100},
        {"name": "very_large", "data": torch.randn(4, 10, 204) * 1000},
        {"name": "tiny_values", "data": torch.randn(4, 10, 204) * 0.001},
    ]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    max_grad_norm = config["training"]["max_grad_norm"]
    loss_scale_factor = config["training"]["loss_scale_factor"]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        x = test_case["data"]
        
        try:
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            recon, codes, q_loss, usage_loss, z_q = model(x, "face")
            
            # Check for finite losses
            if not torch.isfinite(q_loss):
                print(f"  Non-finite q_loss: {q_loss}")
                continue
            if not torch.isfinite(usage_loss):
                print(f"  Non-finite usage_loss: {usage_loss}")
                continue
            
            # Compute total loss
            recon_loss = torch.nn.functional.mse_loss(recon, x)
            total_loss = recon_loss + q_loss + usage_loss
            
            if not torch.isfinite(total_loss):
                print(f"  Non-finite total loss: {total_loss}")
                continue
            
            # Scale and backward
            scaled_loss = total_loss * loss_scale_factor
            scaled_loss.backward()
            
            # Check gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            print(f"  Total loss: {total_loss.item():.6f}")
            print(f"  Scaled loss: {scaled_loss.item():.6f}")
            print(f"  Grad norm: {grad_norm:.6f}")
            print(f"  Recon: {recon_loss.item():.6f}, Q: {q_loss.item():.6f}, Usage: {usage_loss.item():.6f}")
            print(f"  [PASS] Stability test passed")
            
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
    
    # Test quantizer stability specifically
    print(f"\nTesting quantizer stability...")
    quantizer = model.quantizer
    
    for i in range(5):
        x = torch.randn(8, 256) * (10 ** i)  # Exponentially increasing magnitude
        
        try:
            z_q, codes, q_loss, usage_loss = quantizer(x)
            print(f"  Magnitude 10^{i}: q_loss={q_loss.item():.6f}, usage_loss={usage_loss.item():.6f}")
            
            if not torch.isfinite(q_loss) or not torch.isfinite(usage_loss):
                print(f"    [WARN] Non-finite loss at magnitude 10^{i}")
            else:
                print(f"    [PASS] Finite losses")
                
        except Exception as e:
            print(f"    [FAIL] Error at magnitude 10^{i}: {e}")

if __name__ == "__main__":
    test_stability_fixes()
    print("\n[SUCCESS] Stability testing completed!")