"""
Multi-GPU DDP training script for RVQ model.
"""

import argparse
import logging
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from pathlib import Path
from datetime import datetime

from signstream.data.datasets import CSLDailyDataset, CSLDailyDataModule
from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.seed import set_seed
from signstream.training.loop_rvq import RVQTrainingLoop
from signstream.training.optim import create_optimizer, create_warmup_scheduler
from signstream.training.utils_tb import setup_tensorboard_logger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_ddp(rank: int, world_size: int, master_port: str = "12355"):
    """Initialize the process group for DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_directories(config: dict, rank: int) -> Path:
    """Set up experiment directories (only on rank 0)."""
    if rank != 0:
        return Path(config['save_path'])
    
    experiment_name = config['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    exp_dir = Path(config['save_path']) / f"{experiment_name}_ddp_{timestamp}"
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


def create_dummy_dataset(config: dict, rank: int) -> CSLDailyDataset:
    """Create dummy dataset for testing when CSL-Daily data is not available."""
    if rank == 0:
        logger.warning("Creating dummy dataset for testing - CSL-Daily data not found")
    
    class DummyCSLDataset(CSLDailyDataset):
        def __init__(self, **kwargs):
            # Override initialization to avoid file system checks
            self.chunk_len = kwargs['chunk_len']
            self.normalize = kwargs.get('normalize', True)
            self.augment = kwargs.get('augment', False)
            self.body_part_indices = {
                'face': (24, 91),
                'left_hand': (92, 112), 
                'right_hand': (113, 133),
                'body': (1, 17),
                'full_body': (1, 133),
            }
            # Create dummy data
            self.valid_samples = []
            for i in range(500):  # 500 dummy samples for multi-GPU training
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
        augment=kwargs.get('augment', False),
    )


def create_datasets(config: dict, rank: int):
    """Create training and validation datasets."""
    try:
        # Try to create real datasets
        train_dataset = CSLDailyDataset(
            root_dir=config["data"]["root"],
            split="train",
            chunk_len=config["data"]["chunk_len"],
            fps=config["data"]["fps"],
            normalize=config["data"]["normalize"]["center_parts"],
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
            normalize=config["data"]["normalize"]["center_parts"],
            augment=False,
            body_part_indices=config["data"]["body_parts"],
            center_indices=config["data"]["center_indices"],
        )
        
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            logger.warning(f"Could not load real dataset: {e}")
        train_dataset = create_dummy_dataset(config, rank)
        # Use same dataset for validation in dummy mode
        val_dataset = create_dummy_dataset(config, rank)
    
    return train_dataset, val_dataset


def create_model(config: dict, device: torch.device, rank: int) -> RVQModel:
    """Create RVQ model based on configuration."""
    
    # Calculate frame dimensions for full_body (default)
    body_parts = config["data"]["body_parts"]
    start, end = body_parts['full_body']
    num_points = end - start + 1
    frame_dim = num_points * 3
    
    model = RVQModel(
        frame_dim=frame_dim,
        chunk_len=config["data"]["chunk_len"],
        latent_dim=config["model"]["latent_dim"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"]["commitment_beta"],
        ema_decay=config["model"]["rvq"]["ema_decay"],
        usage_reg=config["model"]["rvq"]["usage_reg"],
        arch=config["model"]["arch"],
    )
    
    model.to(device)
    
    if rank == 0:
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


class DDPTrainingLoop(RVQTrainingLoop):
    """DDP-aware training loop."""
    
    def __init__(self, *args, **kwargs):
        self.rank = kwargs.pop('rank', 0)
        self.world_size = kwargs.pop('world_size', 1)
        super().__init__(*args, **kwargs)
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint only on rank 0."""
        if self.rank == 0:
            super().save_checkpoint(epoch, metrics, is_best)
    
    def _log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict, 
                     codebook_metrics: dict):
        """Log metrics only on rank 0."""
        if self.rank == 0:
            super()._log_metrics(epoch, train_metrics, val_metrics, codebook_metrics)


def train_ddp(rank: int, world_size: int, config: dict, args):
    """DDP training function."""
    # Setup DDP
    setup_ddp(rank, world_size, args.master_port)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    set_seed(config["training"]["seed"] + rank)  # Different seed per process
    
    # Setup directories (only on rank 0)
    exp_dir = setup_directories(config, rank)
    
    # Sync all processes after directory creation
    dist.barrier()
    
    # Broadcast config from rank 0 to ensure consistency
    if rank == 0:
        config_list = [config]
    else:
        config_list = [None]
    
    dist.broadcast_object_list(config_list, src=0)
    config = config_list[0]
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, rank)
    
    if rank == 0:
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"] // world_size,  # Scale batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] // world_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=CSLDailyDataModule.collate_fn,
    )
    
    # Create model
    model = create_model(config, device, rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_warmup_scheduler(optimizer, config)
    
    # Setup TensorBoard logging (only on rank 0)
    tb_logger = setup_tensorboard_logger(config) if rank == 0 else None
    if tb_logger:
        tb_logger.log_config(config)
    
    # Create training loop
    trainer = DDPTrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        logger_writer=tb_logger,
        rank=rank,
        world_size=world_size,
    )
    
    # Load checkpoint if specified
    if args.checkpoint and rank == 0:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Sync all processes before training
    dist.barrier()
    
    # Start training
    try:
        trainer.train()
        if rank == 0:
            logger.info("DDP training completed successfully")
    except Exception as e:
        logger.error(f"DDP training failed on rank {rank}: {e}")
        raise
    finally:
        if tb_logger:
            tb_logger.close()
        cleanup_ddp()


def main():
    """Main DDP training function."""
    parser = argparse.ArgumentParser(description="Train RVQ model with DDP")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for resuming training")
    parser.add_argument("--master-port", type=str, default="12355",
                       help="Master port for DDP communication")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get world size from environment or config
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size == 1:
        logger.warning("Running DDP with world_size=1. Consider using single GPU script instead.")
    
    # Spawn processes for DDP
    mp.spawn(
        train_ddp,
        args=(world_size, config, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()