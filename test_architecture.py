"""
Simple test script to verify the shared backbone architecture works correctly.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.quantizer import ResidualVectorQuantizer
from signstream.models.rvq.rvq_model import RVQModel
from signstream.models.rvq.encoder import PART_DIMENSIONS

def test_quantizer():
    """Test basic quantizer functionality."""
    print("Testing ResidualVectorQuantizer...")
    
    config = {
        'dim': 64,
        'codebook_size': 32,
        'levels': 3,
        'commitment_beta': 0.25,
        'ema_decay': 0.99,
        'usage_reg': 1e-3
    }
    
    quantizer = ResidualVectorQuantizer(**config)
    
    # Test initialization
    assert quantizer.levels == config['levels']
    assert quantizer.codebook_size == config['codebook_size']
    assert quantizer.dim == config['dim']
    assert len(quantizer.codebooks) == config['levels']
    print("[PASS] Quantizer initialization test passed")
    
    # Test forward pass shapes
    batch_size = 8
    x = torch.randn(batch_size, config['dim'])
    quantized, codes, commit_loss, usage_loss = quantizer(x)
    
    assert quantized.shape == x.shape
    assert codes.shape == (batch_size, config['levels'])
    assert isinstance(commit_loss, torch.Tensor)
    assert isinstance(usage_loss, torch.Tensor)
    print("[PASS] Quantizer forward pass test passed")

def test_shared_backbone_model():
    """Test RVQ model with shared backbone architecture."""
    print("\nTesting RVQModel with shared backbone...")
    
    model_config = {
        'latent_dim': 32,
        'chunk_len': 8,
        'codebook_size': 16,
        'levels': 2,
        'arch': 'mlp',
        'commitment_beta': 0.25,
        'ema_decay': 0.99,
        'usage_reg': 1e-3,
        'num_layers': 2,
        'type_embed_dim': 16,
        'dropout': 0.1,
        'temporal_aggregation': 'mean'
    }
    
    model = RVQModel(**model_config)
    
    # Test with all body parts
    test_parts = ['face', 'left_hand', 'right_hand', 'body', 'full_body']
    batch_size = 4
    
    for part_name in test_parts:
        print(f"  Testing {part_name}...")
        
        # Get correct dimensions for this body part
        num_keypoints = PART_DIMENSIONS[part_name]
        frame_dim = num_keypoints * 3  # x, y, confidence
        
        # Create test input
        x = torch.randn(batch_size, model_config['chunk_len'], frame_dim)
        
        # Forward pass
        recon, codes, q_loss, usage_loss, z_q = model(x, part_name)
        
        # Check shapes
        assert recon.shape == x.shape, f"Recon shape mismatch for {part_name}: {recon.shape} vs {x.shape}"
        assert codes.shape == (batch_size, model_config['levels']), f"Codes shape mismatch for {part_name}"
        assert z_q.shape == (batch_size, model_config['latent_dim']), f"Latent shape mismatch for {part_name}"
        
        # Check losses are scalars
        assert q_loss.dim() == 0, f"Q loss not scalar for {part_name}"
        assert usage_loss.dim() == 0, f"Usage loss not scalar for {part_name}"
        
        print(f"    [OK] {part_name}: input {x.shape} -> recon {recon.shape}, codes {codes.shape}, latent {z_q.shape}")
    
    print("[PASS] All body parts test passed")

def test_model_reconstruction():
    """Test that model can produce different reconstructions for different inputs."""
    print("\nTesting model reconstruction capabilities...")
    
    model_config = {
        'latent_dim': 32,
        'chunk_len': 8,
        'codebook_size': 16,
        'levels': 2,
        'arch': 'mlp',
        'commitment_beta': 0.25,
        'ema_decay': 0.99,
        'usage_reg': 1e-3,
        'num_layers': 2,
        'type_embed_dim': 16,
        'dropout': 0.1,
        'temporal_aggregation': 'mean'
    }
    
    model = RVQModel(**model_config)
    
    # Test with face data (largest body part)
    face_dim = PART_DIMENSIONS['face'] * 3  # 68 * 3 = 204
    
    # Create two different inputs
    x1 = torch.ones(2, 8, face_dim) * 1.0   # All ones
    x2 = torch.ones(2, 8, face_dim) * -1.0  # All negative ones
    
    # Forward passes
    recon1, codes1, _, _, _ = model(x1, "face")
    recon2, codes2, _, _, _ = model(x2, "face")
    
    # Reconstructions should be different for different inputs
    recon_diff = torch.norm(recon1 - recon2)
    assert recon_diff > 0.1, f"Reconstructions too similar: diff = {recon_diff.item()}"
    
    # Codes might be different (not guaranteed, but likely)
    codes_same = torch.equal(codes1, codes2)
    
    print(f"    [OK] Reconstruction difference: {recon_diff.item():.4f}")
    print(f"    [OK] Codes are {'same' if codes_same else 'different'}")
    print("[PASS] Reconstruction test passed")

def test_overfitting_capability():
    """Test that model can overfit to a single sample."""
    print("\nTesting model overfitting capability...")
    
    model_config = {
        'latent_dim': 32,
        'chunk_len': 8,
        'codebook_size': 16,
        'levels': 2,
        'arch': 'mlp',
        'commitment_beta': 0.25,
        'ema_decay': 0.99,
        'usage_reg': 1e-3,
        'num_layers': 2,
        'type_embed_dim': 16,
        'dropout': 0.1,
        'temporal_aggregation': 'mean'
    }
    
    model = RVQModel(**model_config)
    
    # Single sample - use body part
    body_dim = PART_DIMENSIONS['body'] * 3  # 17 * 3 = 51
    x = torch.randn(1, 8, body_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    initial_loss = None
    for epoch in range(20):
        recon, codes, q_loss, usage_loss, _ = model(x, "body")
        
        # Reconstruction + quantization loss
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        total_loss = recon_loss + q_loss + usage_loss
        
        if epoch == 0:
            initial_loss = total_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: Loss = {total_loss.item():.6f}")
    
    final_loss = total_loss.item()
    reduction = (initial_loss - final_loss) / initial_loss
    
    print(f"    Initial loss: {initial_loss:.6f}")
    print(f"    Final loss: {final_loss:.6f}")
    print(f"    Reduction: {reduction*100:.1f}%")
    
    assert final_loss < initial_loss * 0.5, f"Insufficient loss reduction: {reduction*100:.1f}%"
    print("[PASS] Overfitting test passed")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing SignStream RVQ Shared Backbone Architecture")
    print("=" * 60)
    
    try:
        test_quantizer()
        test_shared_backbone_model()
        test_model_reconstruction()
        test_overfitting_capability()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED! Shared backbone architecture is working correctly.")
        print("=" * 60)
        
        # Print model info
        model = RVQModel(
            latent_dim=256,
            chunk_len=10,
            codebook_size=1024,
            levels=3,
            arch='transformer',
            commitment_beta=0.25,
            ema_decay=0.99,
            usage_reg=1e-3,
            num_layers=2,
            type_embed_dim=16,
            dropout=0.1,
            temporal_aggregation='mean'
        )
        
        model_info = model.get_model_info()
        print(f"\nModel Information:")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Architecture: {model_info['arch']}")
        print(f"  Latent dim: {model_info['latent_dim']}")
        print(f"  Supported body parts: {model_info['supported_parts']}")
        print(f"  Part dimensions: {model_info['part_dimensions']}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)