"""
Tests for ResidualVectorQuantizer (RVQ) implementation.
"""

import torch
import pytest
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from signstream.models.rvq.quantizer import ResidualVectorQuantizer
from signstream.models.rvq.rvq_model import RVQModel
from signstream.models.metrics.codebook import (
    codebook_utilization,
    codebook_perplexity, 
    compute_all_codebook_metrics
)


class TestResidualVectorQuantizer:
    """Test suite for ResidualVectorQuantizer."""
    
    @pytest.fixture
    def quantizer_config(self) -> Dict[str, Any]:
        """Standard quantizer configuration for testing."""
        return {
            'dim': 64,
            'codebook_size': 32,
            'levels': 3,
            'commitment_beta': 0.25,
            'ema_decay': 0.99,
            'usage_reg': 1e-3
        }
    
    @pytest.fixture
    def quantizer(self, quantizer_config) -> ResidualVectorQuantizer:
        """Create quantizer instance for testing."""
        return ResidualVectorQuantizer(**quantizer_config)
    
    def test_quantizer_initialization(self, quantizer, quantizer_config):
        """Test quantizer initializes correctly."""
        assert quantizer.levels == quantizer_config['levels']
        assert quantizer.codebook_size == quantizer_config['codebook_size']
        assert quantizer.dim == quantizer_config['dim']
        assert len(quantizer.codebooks) == quantizer_config['levels']
        
        # Check buffer shapes
        assert quantizer.cluster_size.shape == (quantizer_config['levels'], quantizer_config['codebook_size'])
        assert quantizer.embed_avg.shape == (quantizer_config['levels'], quantizer_config['codebook_size'], quantizer_config['dim'])
    
    def test_forward_pass_shapes(self, quantizer, quantizer_config):
        """Test forward pass produces correct output shapes."""
        batch_size = 8
        dim = quantizer_config['dim']
        
        x = torch.randn(batch_size, dim)
        
        quantized, codes, commit_loss, usage_loss = quantizer(x)
        
        # Check output shapes
        assert quantized.shape == x.shape
        assert codes.shape == (batch_size, quantizer_config['levels'])
        assert isinstance(commit_loss, torch.Tensor)
        assert isinstance(usage_loss, torch.Tensor)
        assert commit_loss.dim() == 0  # Scalar
        assert usage_loss.dim() == 0   # Scalar
    
    def test_quantization_deterministic(self, quantizer):
        """Test that quantization is deterministic for same input."""
        x = torch.randn(4, 64)
        
        # Set to eval mode to disable EMA updates
        quantizer.eval()
        
        # First pass
        quantized1, codes1, _, _ = quantizer(x)
        
        # Second pass with same input
        quantized2, codes2, _, _ = quantizer(x)
        
        # Should be identical
        torch.testing.assert_close(quantized1, quantized2)
        torch.testing.assert_close(codes1.float(), codes2.float())
    
    def test_ema_updates_in_training(self, quantizer):
        """Test that EMA buffers update during training."""
        quantizer.train()
        
        # Get initial buffer states
        initial_cluster_size = quantizer.cluster_size.clone()
        initial_embed_avg = quantizer.embed_avg.clone()
        
        # Forward pass
        x = torch.randn(16, 64)
        quantizer(x)
        
        # Buffers should have changed
        assert not torch.equal(initial_cluster_size, quantizer.cluster_size)
        assert not torch.equal(initial_embed_avg, quantizer.embed_avg)
    
    def test_no_ema_updates_in_eval(self, quantizer):
        """Test that EMA buffers don't update during evaluation."""
        quantizer.eval()
        
        # Get initial buffer states
        initial_cluster_size = quantizer.cluster_size.clone()
        initial_embed_avg = quantizer.embed_avg.clone()
        
        # Forward pass
        x = torch.randn(16, 64)
        quantizer(x)
        
        # Buffers should be unchanged
        torch.testing.assert_close(initial_cluster_size, quantizer.cluster_size)
        torch.testing.assert_close(initial_embed_avg, quantizer.embed_avg)
    
    def test_code_ranges(self, quantizer, quantizer_config):
        """Test that generated codes are within valid ranges."""
        x = torch.randn(10, 64)
        _, codes, _, _ = quantizer(x)
        
        # Check code ranges
        assert codes.min() >= 0
        assert codes.max() < quantizer_config['codebook_size']
        assert codes.dtype == torch.long
    
    def test_gradient_flow(self, quantizer):
        """Test that gradients flow through the quantizer."""
        x = torch.randn(8, 64, requires_grad=True)
        
        quantized, _, commit_loss, usage_loss = quantizer(x)
        
        # Compute total loss
        total_loss = commit_loss + usage_loss + quantized.sum()
        total_loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_commitment_loss_properties(self, quantizer):
        """Test properties of commitment loss."""
        # Test with zero input (should have low commitment loss)
        x_zero = torch.zeros(4, 64)
        _, _, commit_loss_zero, _ = quantizer(x_zero)
        
        # Test with large input (should have higher commitment loss)
        x_large = torch.ones(4, 64) * 10
        _, _, commit_loss_large, _ = quantizer(x_large)
        
        # Both should be positive
        assert commit_loss_zero >= 0
        assert commit_loss_large >= 0
    
    def test_usage_regularization(self, quantizer_config):
        """Test usage regularization encourages uniform distribution."""
        # Create quantizer with high usage regularization
        quantizer = ResidualVectorQuantizer(
            dim=quantizer_config['dim'],
            codebook_size=8,  # Small codebook for easier testing
            levels=1,
            usage_reg=1.0  # High regularization
        )
        
        # Generate data that would naturally cluster to few codes
        x = torch.zeros(32, quantizer_config['dim'])
        x[:16] += 1.0  # First half shifted
        
        _, codes, _, usage_loss = quantizer(x)
        
        # Usage loss should be positive (encouraging more uniform usage)
        assert usage_loss > 0
        
        # Codes should utilize multiple entries
        unique_codes = len(torch.unique(codes))
        assert unique_codes > 1  # Should use more than one code
    
    def test_residual_quantization(self, quantizer_config):
        """Test that residual quantization works across levels."""
        quantizer = ResidualVectorQuantizer(
            dim=quantizer_config['dim'],
            codebook_size=quantizer_config['codebook_size'],
            levels=2,
        )
        
        x = torch.randn(8, quantizer_config['dim'])
        quantized, codes, _, _ = quantizer(x)
        
        # Check that we have codes for each level
        assert codes.shape[1] == 2
        
        # Each level should potentially use different codes
        level_0_codes = codes[:, 0]
        level_1_codes = codes[:, 1] 
        
        # They shouldn't be identical (unless by chance)
        assert not torch.all(level_0_codes == level_1_codes)


class TestCodebookMetrics:
    """Test suite for codebook analysis metrics."""
    
    def test_codebook_utilization(self):
        """Test codebook utilization calculation."""
        # Perfect utilization
        codes_full = torch.arange(10)  # Uses all codes 0-9
        util_full = codebook_utilization(codes_full, 10)
        assert util_full == 1.0
        
        # Partial utilization
        codes_partial = torch.tensor([0, 1, 0, 1, 2])  # Uses 3 out of 10 codes
        util_partial = codebook_utilization(codes_partial, 10)
        assert util_partial == 0.3
        
        # No utilization (edge case)
        codes_empty = torch.tensor([])
        util_empty = codebook_utilization(codes_empty, 10)
        assert util_empty == 0.0
    
    def test_codebook_perplexity(self):
        """Test codebook perplexity calculation."""
        # Uniform distribution (high perplexity)
        codes_uniform = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        perp_uniform = codebook_perplexity(codes_uniform, 4)
        assert perp_uniform > 3.0  # Close to 4 for uniform
        
        # Concentrated distribution (low perplexity) 
        codes_concentrated = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])
        perp_concentrated = codebook_perplexity(codes_concentrated, 4)
        assert perp_concentrated < 2.0  # Much less than 4
    
    def test_compute_all_metrics(self):
        """Test comprehensive codebook metrics computation."""
        # Create multi-level codes
        codes_level_0 = torch.tensor([0, 1, 2, 0, 1])
        codes_level_1 = torch.tensor([2, 2, 3, 3, 4])
        codes_per_level = [codes_level_0, codes_level_1]
        
        metrics = compute_all_codebook_metrics(codes_per_level, codebook_size=5)
        
        # Check structure
        assert 'per_level' in metrics
        assert 'overall' in metrics
        
        # Check per-level metrics
        assert 'level_0' in metrics['per_level']
        assert 'level_1' in metrics['per_level']
        
        # Check metric completeness
        level_0_metrics = metrics['per_level']['level_0']
        expected_keys = ['utilization', 'perplexity', 'entropy', 'dead_ratio']
        for key in expected_keys:
            assert key in level_0_metrics
            assert isinstance(level_0_metrics[key], (int, float))
        
        # Check overall metrics
        overall = metrics['overall']
        assert 'avg_utilization' in overall
        assert 'avg_perplexity' in overall
        assert 'total_levels' in overall
        assert overall['total_levels'] == 2


class TestRVQModel:
    """Test suite for complete RVQ model."""
    
    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Standard model configuration for testing."""
        return {
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
    
    @pytest.fixture
    def model(self, model_config) -> RVQModel:
        """Create RVQ model instance for testing."""
        return RVQModel(**model_config)
    
    def test_model_forward_pass(self, model, model_config):
        """Test complete model forward pass."""
        batch_size = 4
        chunk_len = model_config['chunk_len']
        
        # Test with different body parts
        from signstream.models.rvq.encoder import PART_DIMENSIONS
        
        for part_name, num_keypoints in PART_DIMENSIONS.items():
            frame_dim = num_keypoints * 3  # x, y, confidence
            x = torch.randn(batch_size, chunk_len, frame_dim)
            
            recon, codes, q_loss, usage_loss, z_q = model(x, part_name)
            
            # Check shapes
            assert recon.shape == x.shape, f"Recon shape mismatch for {part_name}"
            assert codes.shape == (batch_size, model_config['levels']), f"Codes shape mismatch for {part_name}"
            assert z_q.shape == (batch_size, model_config['latent_dim']), f"Latent shape mismatch for {part_name}"
            
            # Check losses are scalars
            assert q_loss.dim() == 0, f"Q loss not scalar for {part_name}"
            assert usage_loss.dim() == 0, f"Usage loss not scalar for {part_name}"
    
    def test_model_reconstruction_sanity(self, model):
        """Test that model can reconstruct simple patterns."""
        # Test with face (68 keypoints * 3 = 204 dims)
        from signstream.models.rvq.encoder import PART_DIMENSIONS
        
        face_dim = PART_DIMENSIONS['face'] * 3
        x = torch.zeros(2, 8, face_dim)
        x[0] = 1.0  # First sample is all ones
        x[1] = -1.0  # Second sample is all negative ones
        
        # Forward pass
        recon, _, _, _, _ = model(x, "face")
        
        # Reconstruction should be different for different inputs
        recon_diff = torch.norm(recon[0] - recon[1])
        assert recon_diff > 0.1  # Should be noticeably different
    
    def test_model_overfitting_capability(self, model):
        """Test that model can overfit to a single sample (sanity check)."""
        # Single sample to overfit - use body part (17 keypoints * 3 = 51 dims)
        from signstream.models.rvq.encoder import PART_DIMENSIONS
        
        body_dim = PART_DIMENSIONS['body'] * 3
        x = torch.randn(1, 8, body_dim)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        initial_loss = None
        for epoch in range(20):  # More epochs for better convergence
            recon, codes, q_loss, usage_loss, _ = model(x, "body")
            
            # Reconstruction + quantization loss
            recon_loss = torch.nn.functional.mse_loss(recon, x)
            total_loss = recon_loss + q_loss + usage_loss
            
            if epoch == 0:
                initial_loss = total_loss.item()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Loss should decrease significantly
        final_loss = total_loss.item()
        assert final_loss < initial_loss * 0.5  # At least 50% reduction
    
    def test_model_different_body_parts(self, model_config):
        """Test model handles different body parts correctly."""
        # Create model
        model = RVQModel(**model_config)
        
        # Test different parts with correct input dimensions
        from signstream.models.rvq.encoder import PART_DIMENSIONS
        
        parts = ["face", "left_hand", "right_hand", "body", "full_body"]
        
        results = {}
        for part in parts:
            frame_dim = PART_DIMENSIONS[part] * 3
            x = torch.randn(2, 8, frame_dim)
            recon, codes, _, _, _ = model(x, part)
            results[part] = (recon, codes, x.shape)
        
        # Results should be different for different parts (when inputs differ)
        face_recon, face_codes, face_shape = results["face"]
        hand_recon, hand_codes, hand_shape = results["left_hand"]
        
        # Shapes should be different (face has more keypoints than hand)
        assert face_shape != hand_shape
        
        # Codes should be in valid range
        for part_name, (_, codes, _) in results.items():
            assert codes.min() >= 0
            assert codes.max() < model_config['codebook_size']


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_quantizer = TestResidualVectorQuantizer()
    test_metrics = TestCodebookMetrics()
    test_model = TestRVQModel()
    
    print("Running RVQ tests...")
    
    # Create test instances
    config = {
        'dim': 64, 'codebook_size': 32, 'levels': 3,
        'commitment_beta': 0.25, 'ema_decay': 0.99, 'usage_reg': 1e-3
    }
    quantizer = ResidualVectorQuantizer(**config)
    
    model_config = {
        'chunk_len': 8, 'latent_dim': 32,
        'codebook_size': 16, 'levels': 2, 'arch': 'mlp',
        'commitment_beta': 0.25, 'ema_decay': 0.99, 'usage_reg': 1e-3,
        'num_layers': 2, 'type_embed_dim': 16, 'dropout': 0.1,
        'temporal_aggregation': 'mean'
    }
    model = RVQModel(**model_config)
    
    # Run some basic tests
    try:
        test_quantizer.test_quantizer_initialization(quantizer, config)
        test_quantizer.test_forward_pass_shapes(quantizer, config)
        test_metrics.test_codebook_utilization()
        
        # Test model with all body parts
        from signstream.models.rvq.encoder import PART_DIMENSIONS
        test_parts = ['face', 'left_hand', 'right_hand', 'body']
        
        for part in test_parts:
            print(f"Testing {part} part...")
            part_dim = PART_DIMENSIONS[part] * 3
            x = torch.randn(2, model_config['chunk_len'], part_dim)
            recon, codes, q_loss, usage_loss, z_q = model(x, part)
            assert recon.shape == x.shape, f"Shape mismatch for {part}"
            assert codes.shape == (2, model_config['levels']), f"Codes shape wrong for {part}"
            print(f"✓ {part} test passed")
        
        print("✓ All basic tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise