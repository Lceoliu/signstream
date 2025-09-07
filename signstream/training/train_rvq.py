"""
Main training script for RVQ model with comprehensive logging and evaluation.
"""

import argparse
import logging
import yaml
import torch
import os
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from signstream.data.datasets import CSLDailyDataset, CSLDailyDataModule
from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.seed import set_seed
from signstream.training.loop_rvq import RVQTrainingLoop
from signstream.training.optim import create_optimizer, create_warmup_scheduler
from signstream.training.utils_tb import setup_tensorboard_logger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_directories(config: dict) -> Path:
    """Set up experiment directories."""
    experiment_name = config['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    exp_dir = Path(config['save_path']) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'exports').mkdir(exist_ok=True)
    
    # Save config copy
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Update config paths
    config['save_path'] = str(exp_dir)
    config['logging']['log_dir'] = str(exp_dir / 'logs')
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def create_dummy_dataset(config: dict) -> CSLDailyDataset:
    """Create dummy dataset for testing when CSL-Daily data is not available."""
    logger.warning("Creating dummy dataset for testing - CSL-Daily data not found")
    
    class DummyCSLDataset(CSLDailyDataset):
        def __init__(self, **kwargs):
            # Override initialization to avoid file system checks
            self.chunk_len = kwargs['chunk_len']
            self.normalize = kwargs.get('normalize', True)
            self.augment = False
            self.body_part_indices = {
                'face': (24, 91),
                'left_hand': (92, 112), 
                'right_hand': (113, 133),
                'body': (1, 17),
                'full_body': (1, 133),
            }
            self.center_indices = {
                'body': 1,
                'face': 32,
                'left_hand': 92,
                'right_hand': 113,
            }
            # Create dummy data
            self.valid_samples = []
            for i in range(100):  # 100 dummy samples
                self.valid_samples.append({
                    'name': f'dummy_video_{i}',
                    'pose_file': f'dummy_{i}.npy',
                    'annotation': {'text': f'dummy text {i}', 'gloss': f'dummy gloss {i}'}
                })
        
        def __getitem__(self, idx):
            # Generate dummy pose data
            T = torch.randint(50, 200, (1,)).item()  # Random sequence length
            poses = torch.randn(T, 133, 3)  # Random pose data
            
            # Split into body parts and chunk
            body_parts = self._split_body_parts(poses)
            chunked_parts = self._chunk_sequences(body_parts)
            
            return {
                'name': self.valid_samples[idx]['name'],
                'chunks': chunked_parts,
                'text': self.valid_samples[idx]['annotation']['text'],
                'gloss': self.valid_samples[idx]['annotation']['gloss'],
                'num_frames': T,
                'num_chunks': chunked_parts['face'].shape[0],
            }
    
    return DummyCSLDataset(
        root_dir=config["data"]["root"],
        chunk_len=config["data"]["chunk_len"],
        fps=config["data"]["fps"],
    )


def create_datasets(config: dict):
    """Create training and validation datasets."""
    try:
        # Try to create real datasets
        train_dataset = CSLDailyDataset(
            root_dir=config["data"]["root"],
            split="train",
            chunk_len=config["data"]["chunk_len"],
            fps=config["data"]["fps"],
            # normalize=config["data"]["normalize"]["center_parts"],
            augment=True,
            augment_config=config["data"]["augment"],
            body_part_indices=config["data"]["body_parts"],
            center_indices=config["data"]["center_indices"],
        )

        val_dataset = CSLDailyDataset(
            root_dir=config["data"]["root"],
            split="val",
            chunk_len=config["data"]["chunk_len"],
            fps=config["data"]["fps"],
            # normalize=config["data"]["normalize"]["center_parts"],
            augment=False,
            body_part_indices=config["data"]["body_parts"],
            center_indices=config["data"]["center_indices"],
        )

    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load real dataset: {e}")
        train_dataset = create_dummy_dataset(config)
        val_dataset = create_dummy_dataset(config)

    return train_dataset, val_dataset


def create_model(config: dict, device: torch.device) -> RVQModel:
    """Create RVQ model based on configuration."""
    
    # Create model with new shared backbone architecture
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
    
    model.to(device)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']} total parameters")
    logger.info(f"Architecture: {model_info['arch']}, Latent dim: {model_info['latent_dim']}")
    logger.info(f"Supported body parts: {model_info['supported_parts']}")
    logger.info(f"Body part dimensions: {model_info['part_dimensions']}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RVQ model")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for resuming training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Set device
    device_name = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config["training"]["seed"])
    
    # Setup directories
    exp_dir = setup_directories(config)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_warmup_scheduler(optimizer, config)
    
    # Setup TensorBoard logging
    tb_logger = setup_tensorboard_logger(config)
    if tb_logger:
        tb_logger.log_config(config)
    
    # Create training loop
    trainer = RVQTrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        logger_writer=tb_logger,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if tb_logger:
            tb_logger.close()


if __name__ == "__main__":
    main()
