"""
Test script to verify the training pipeline works end-to-end with the new shared backbone architecture.
"""

import torch
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.train_rvq import create_dummy_dataset, create_model
from signstream.training.loop_rvq import RVQTrainingLoop
from signstream.training.optim import create_optimizer

def test_training_pipeline():
    """Test the complete training pipeline."""
    print("Testing training pipeline with shared backbone architecture...")
    
    # Load default config
    config_path = "signstream/configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Override with smaller values for testing
    config["model"]["latent_dim"] = 64
    config["model"]["rvq"]["codebook_size"] = 32
    config["model"]["rvq"]["levels"] = 2
    config["training"]["batch_size"] = 2
    config["training"]["epochs"] = 3
    config["data"]["chunk_len"] = 8
    
    print(f"Config loaded: {config['experiment_name']}")
    
    # Create dummy datasets
    train_dataset = create_dummy_dataset(config)
    val_dataset = create_dummy_dataset(config)
    
    print(f"Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Test dataset samples
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Chunk keys: {list(sample['chunks'].keys())}")
    for part, chunk_tensor in sample['chunks'].items():
        print(f"  {part}: {chunk_tensor.shape}")
    
    # Create model
    device = torch.device("cpu")  # Use CPU for testing
    model = create_model(config, device)
    
    print(f"Model created with {model.get_model_info()['total_parameters']:,} parameters")
    
    # Test single forward pass with each body part
    for part_name, chunk_tensor in sample['chunks'].items():
        print(f"Testing forward pass for {part_name}...")
        
        # Prepare batch (add batch dimension)
        batch_size = 1
        N, L, K, C = chunk_tensor.shape
        x = chunk_tensor.view(N, L, K * C).unsqueeze(0)  # [1, N, L, K*C]
        
        # Forward pass
        with torch.no_grad():
            recon, codes, q_loss, usage_loss, z_q = model(x.squeeze(0), part_name)
            
        print(f"  Input shape: {x.squeeze(0).shape}")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Codes shape: {codes.shape}")
        print(f"  Q loss: {q_loss.item():.6f}")
        print(f"  Usage loss: {usage_loss.item():.6f}")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    from signstream.data.datasets import CSLDailyDataModule
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"], 
        shuffle=False,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    print(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Test batch processing
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Optimizer created: {type(optimizer).__name__}")
    
    # Create training loop
    trainer = RVQTrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        logger_writer=None,  # No logging for test
    )
    
    print("Training loop created successfully")
    
    # Test a single training step
    print("\nTesting single training step...")
    model.train()
    
    total_train_loss = 0
    total_recon_loss = 0
    total_q_loss = 0
    total_usage_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:  # Only test first 2 batches
            break
            
        batch_losses = trainer._train_step(batch, log_metrics=False)
        
        total_train_loss += batch_losses['train_loss']
        total_recon_loss += batch_losses['recon_loss']
        total_q_loss += batch_losses['q_loss']
        total_usage_loss += batch_losses['usage_loss']
        num_batches += 1
        
        print(f"  Batch {batch_idx}: loss={batch_losses['train_loss']:.6f}")
    
    # Average losses
    avg_train_loss = total_train_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_q_loss = total_q_loss / num_batches
    avg_usage_loss = total_usage_loss / num_batches
    
    print(f"Average losses over {num_batches} batches:")
    print(f"  Total: {avg_train_loss:.6f}")
    print(f"  Reconstruction: {avg_recon_loss:.6f}")
    print(f"  Quantization: {avg_q_loss:.6f}")
    print(f"  Usage: {avg_usage_loss:.6f}")
    
    # Test validation step
    print("\nTesting validation step...")
    model.eval()
    
    val_batch = next(iter(val_loader))
    val_losses = trainer._validation_step(val_batch)
    
    print(f"Validation loss: {val_losses['val_loss']:.6f}")
    print(f"Val reconstruction: {val_losses['val_recon_loss']:.6f}")
    
    return True

def main():
    """Run training pipeline test."""
    print("=" * 70)
    print("Testing SignStream RVQ Training Pipeline")
    print("=" * 70)
    
    try:
        success = test_training_pipeline()
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Training pipeline test completed successfully!")
        print("The shared backbone architecture is ready for training.")
        print("=" * 70)
        
        return success
        
    except Exception as e:
        print(f"\n[FAILED] Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)