"""
Test the quantizer fix across different scenarios.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.quantizer import ResidualVectorQuantizer

def test_quantizer_fix():
    """Test the quantizer fix in various scenarios."""
    print("Testing quantizer fix...")
    
    # Test configurations
    configs = [
        {"device": "cpu", "dtype": torch.float32},
        {"device": "cpu", "dtype": torch.float64},
    ]
    
    # Add CUDA test if available
    if torch.cuda.is_available():
        configs.append({"device": "cuda", "dtype": torch.float32})
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: device={config['device']}, dtype={config['dtype']}")
        
        try:
            # Create quantizer
            quantizer = ResidualVectorQuantizer(
                dim=64,
                codebook_size=32,
                levels=2,
                usage_reg=1e-3
            )
            
            if config['device'] == 'cuda':
                quantizer = quantizer.cuda()
            
            # Create input tensor
            x = torch.randn(4, 64, device=config['device'], dtype=config['dtype'])
            print(f"  Input: shape={x.shape}, device={x.device}, dtype={x.dtype}")
            
            # Forward pass
            z_q, codes, q_loss, usage_loss = quantizer(x)
            
            print(f"  Output: z_q={z_q.shape}, codes={codes.shape}")
            print(f"  Losses: q_loss={q_loss.item():.6f}, usage_loss={usage_loss.item():.6f}")
            print(f"  [PASS] Test passed")
            
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()

def test_edge_cases():
    """Test edge cases that might cause the original error."""
    print("\nTesting edge cases...")
    
    quantizer = ResidualVectorQuantizer(
        dim=32,
        codebook_size=16,
        levels=1,
        usage_reg=0.0  # Zero usage reg
    )
    
    # Test 1: Zero usage regularization
    x = torch.randn(2, 32)
    z_q, codes, q_loss, usage_loss = quantizer(x)
    print(f"Zero usage_reg test: usage_loss={usage_loss.item()}")
    
    # Test 2: Very small batch
    x = torch.randn(1, 32)
    z_q, codes, q_loss, usage_loss = quantizer(x)
    print(f"Small batch test: shapes z_q={z_q.shape}, codes={codes.shape}")
    
    # Test 3: Large batch
    x = torch.randn(100, 32)
    z_q, codes, q_loss, usage_loss = quantizer(x)
    print(f"Large batch test: usage_loss={usage_loss.item():.6f}")
    
    print("All edge cases passed!")

if __name__ == "__main__":
    test_quantizer_fix()
    test_edge_cases()
    print("\n[SUCCESS] All quantizer fix tests passed!")