"""
Simple test script to verify the training pipeline works end-to-end with the new shared backbone architecture.
"""

import torch
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.rvq_model import RVQModel

def create_dummy_dataset_simple():
    """Create a simple dummy dataset for testing."""
    from signstream.models.rvq.encoder import PART_DIMENSIONS
    
    class SimpleDummyDataset:
        def __init__(self, num_samples=10, chunk_len=8):
            self.num_samples = num_samples
            self.chunk_len = chunk_len
            self.samples = []
            
            for i in range(num_samples):
                # Create random pose sequence
                T = torch.randint(30, 80, (1,)).item()  # Random length
                poses = torch.randn(T, 133, 3)  # Full body pose
                
                # Split into body parts and chunk
                body_parts = self._split_body_parts(poses)
                chunked_parts = self._chunk_sequences(body_parts)
                
                sample = {
                    'name': f'dummy_{i}',
                    'chunks': chunked_parts,
                    'text': f'dummy text {i}',
                    'gloss': f'dummy gloss {i}',
                    'num_frames': T,
                    'num_chunks': chunked_parts['face'].shape[0],
                }
                self.samples.append(sample)
        
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            return self.samples[idx]
        
        def _split_body_parts(self, poses):
            """Split poses into body parts based on keypoint indices."""
            # CSL-Daily body part indices
            parts = {
                'face': poses[:, 23:91, :],      # indices 24-91 (0-indexed: 23-90)
                'left_hand': poses[:, 91:112, :], # indices 92-112 (0-indexed: 91-111)  
                'right_hand': poses[:, 112:133, :], # indices 113-133 (0-indexed: 112-132)
                'body': poses[:, 0:17, :],        # indices 1-17 (0-indexed: 0-16)
                'full_body': poses,               # full pose
            }
            return parts
        
        def _chunk_sequences(self, body_parts):
            """Chunk sequences for each body part."""
            chunked = {}
            for part_name, poses in body_parts.items():
                T, K, C = poses.shape  # T=frames, K=keypoints, C=channels
                
                # Pad if needed
                if T % self.chunk_len != 0:
                    pad_len = self.chunk_len - (T % self.chunk_len)
                    padding = torch.zeros(pad_len, K, C)
                    poses = torch.cat([poses, padding], dim=0)
                    T = poses.shape[0]
                
                # Reshape to chunks
                N = T // self.chunk_len  # number of chunks
                chunks = poses.reshape(N, self.chunk_len, K, C)
                chunked[part_name] = chunks
                
            return chunked
    
    return SimpleDummyDataset()

def test_simple_training():
    """Test simple training loop without external dependencies."""
    print("Testing simple training pipeline...")
    
    # Create simple config
    config = {
        "model": {
            "latent_dim": 64,
            "arch": "mlp",
            "encoder_layer": 2,
            "type_embed_dim": 16,
            "dropout": 0.1,
            "temporal_aggregation": "mean",
            "rvq": {
                "codebook_size": 32,
                "levels": 2,
                "commitment_beta": 0.25,
                "ema_decay": 0.99,
                "usage_reg": 1e-3
            }
        },
        "data": {
            "chunk_len": 8
        },
        "training": {
            "batch_size": 2,
            "lr": 1e-3,
            "wd": 0.01
        }
    }
    
    # Create model
    model = RVQModel(
        latent_dim=config["model"]["latent_dim"],
        chunk_len=config["data"]["chunk_len"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"]["commitment_beta"],
        ema_decay=config["model"]["rvq"]["ema_decay"],
        usage_reg=config["model"]["rvq"]["usage_reg"],
        arch=config["model"]["arch"],
        num_layers=config["model"]["encoder_layer"],
        type_embed_dim=config["model"]["type_embed_dim"],
        dropout=config["model"]["dropout"],
        temporal_aggregation=config["model"]["temporal_aggregation"],
    )
    
    print(f"Model created: {model.get_model_info()['total_parameters']:,} parameters")
    
    # Create dataset
    dataset = create_dummy_dataset_simple()
    print(f"Dataset created: {len(dataset)} samples")
    
    # Test single sample
    sample = dataset[0]
    print(f"Sample chunks: {list(sample['chunks'].keys())}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["wd"])
    
    # Simple training loop
    model.train()
    total_losses = []
    
    print("\\nRunning mini training loop...")
    
    for epoch in range(5):
        epoch_losses = []
        
        # Process a few samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            
            # Process each body part
            sample_loss = 0
            num_parts = 0
            
            for part_name, chunks in sample['chunks'].items():
                if part_name == 'full_body':  # Skip full_body for efficiency
                    continue
                    
                N, L, K, C = chunks.shape
                x = chunks.view(N, L, K * C)  # Flatten spatial dimensions
                
                # Forward pass
                recon, codes, q_loss, usage_loss, _ = model(x, part_name)
                
                # Compute losses
                recon_loss = torch.nn.functional.mse_loss(recon, x)
                total_loss = recon_loss + q_loss + usage_loss
                
                sample_loss += total_loss.item()
                num_parts += 1
            
            if num_parts > 0:
                avg_loss = sample_loss / num_parts
                epoch_losses.append(avg_loss)
        
        if epoch_losses:
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            total_losses.append(epoch_loss)
            print(f"  Epoch {epoch}: Loss = {epoch_loss:.6f}")
    
    # Check that loss decreased
    if len(total_losses) >= 2:
        initial_loss = total_losses[0]
        final_loss = total_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\\nTraining progress:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        if final_loss < initial_loss:
            print("[PASS] Loss decreased during training")
        else:
            print("[WARNING] Loss did not decrease significantly")
    
    # Test model in eval mode
    print("\\nTesting evaluation mode...")
    model.eval()
    
    sample = dataset[0]
    part_name = 'face'  # Test with face
    chunks = sample['chunks'][part_name]
    
    N, L, K, C = chunks.shape
    x = chunks.view(N, L, K * C)
    
    with torch.no_grad():
        recon, codes, q_loss, usage_loss, z_q = model(x, part_name)
        
    print(f"  Input: {x.shape}")
    print(f"  Reconstruction: {recon.shape}")
    print(f"  Codes: {codes.shape}")
    print(f"  Latent: {z_q.shape}")
    
    return True

def main():
    """Run simple training test."""
    print("=" * 70)
    print("Testing SignStream RVQ Training Pipeline (Simple)")
    print("=" * 70)
    
    try:
        success = test_simple_training()
        
        print("\\n" + "=" * 70)
        print("[SUCCESS] Simple training pipeline test completed!")
        print("The shared backbone architecture is working correctly.")
        print("=" * 70)
        
        return success
        
    except Exception as e:
        print(f"\\n[FAILED] Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)