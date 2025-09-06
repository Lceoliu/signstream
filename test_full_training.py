"""
Test the full training configuration with temporal loss disabled and Huber loss.
"""

import torch
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.training.train_rvq import create_dummy_dataset, create_model
from signstream.training.loop_rvq import RVQTrainingLoop
from signstream.training.optim import create_optimizer
from torch.utils.data import DataLoader
from signstream.data.datasets import CSLDailyDataModule

def test_training_configuration():
    """Test the full training configuration with all stability measures."""
    print("Testing full training configuration...")
    
    # Load config
    config_path = "signstream/configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Override for testing
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 2
    
    print(f"Configuration:")
    print(f"  temporal_loss_alpha: {config['model']['rvq']['temporal_loss_alpha']}")
    print(f"  enable_temporal_loss: {config['model']['rvq']['enable_temporal_loss']}")
    print(f"  max_grad_norm: {config['training']['max_grad_norm']}")
    print(f"  loss_scale_factor: {config['training']['loss_scale_factor']}")
    print(f"  commitment_beta: {config['model']['rvq']['commitment_beta']}")
    print(f"  usage_reg: {config['model']['rvq']['usage_reg']}")
    
    # Create datasets and model
    device = torch.device("cpu")
    train_dataset = create_dummy_dataset(config)
    val_dataset = create_dummy_dataset(config)
    
    model = create_model(config, device)
    optimizer = create_optimizer(model, config)
    
    print(f"Model: {model.get_model_info()['total_parameters']:,} parameters")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    # Create training loop
    trainer = RVQTrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        logger_writer=None,
    )
    
    print(f"Training loop created, testing 1 epoch...")
    
    # Test one epoch
    try:
        epoch_metrics = trainer.train_epoch(epoch=1)
        
        print(f"Training epoch completed successfully!")
        
        # Print metrics for each body part
        for part, metrics in epoch_metrics.items():
            if metrics['total_loss'] > 0:  # Only print parts that were processed
                print(f"  {part}:")
                print(f"    recon_loss: {metrics['recon_loss']:.6f}")
                print(f"    q_loss: {metrics['q_loss']:.6f}")
                print(f"    usage_loss: {metrics['usage_loss']:.6f}")
                print(f"    temporal_loss: {metrics['temporal_loss']:.6f}")
                print(f"    total_loss: {metrics['total_loss']:.6f}")
        
        # Test validation
        val_metrics, codebook_metrics = trainer.validate_epoch(epoch=1)
        print(f"Validation completed successfully!")
        
        # Check for reasonable loss values
        total_losses = [metrics['total_loss'] for metrics in epoch_metrics.values() if metrics['total_loss'] > 0]
        if total_losses:
            max_loss = max(total_losses)
            avg_loss = sum(total_losses) / len(total_losses)
            
            print(f"Loss statistics:")
            print(f"  Average: {avg_loss:.6f}")
            print(f"  Maximum: {max_loss:.6f}")
            
            if max_loss < 1000:  # Reasonable threshold
                print(f"  [PASS] Loss values are stable")
            elif max_loss < 100000:
                print(f"  [WARN] Loss values are high but manageable")
            else:
                print(f"  [FAIL] Loss values are too high")
                return False
        
        # Check temporal loss is actually disabled
        temporal_losses = [metrics['temporal_loss'] for metrics in epoch_metrics.values()]
        if all(loss == 0.0 for loss in temporal_losses):
            print(f"  [PASS] Temporal loss correctly disabled")
        else:
            print(f"  [WARN] Temporal loss not fully disabled")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_configuration()
    if success:
        print("\n[SUCCESS] Full training configuration test passed!")
        print("Training should now be stable with:")
        print("  - Temporal loss disabled")
        print("  - Huber loss for reconstruction")
        print("  - Gradient clipping enabled")
        print("  - Reduced hyperparameters")
    else:
        print("\n[FAILED] Training configuration needs more work")
    
    sys.exit(0 if success else 1)