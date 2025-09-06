"""
Debug script to identify the quantizer issue.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.quantizer import ResidualVectorQuantizer

def debug_quantizer():
    """Debug the quantizer issue step by step."""
    print("Debugging quantizer...")
    
    # Create quantizer
    quantizer = ResidualVectorQuantizer(
        dim=64,
        codebook_size=32,
        levels=2,
        commitment_beta=0.25,
        ema_decay=0.99,
        usage_reg=1e-3
    )
    
    print(f"Usage reg type: {type(quantizer.usage_reg)}")
    print(f"Usage reg value: {quantizer.usage_reg}")
    
    # Create input
    x = torch.randn(4, 64)
    print(f"Input shape: {x.shape}")
    print(f"Input device: {x.device}")
    print(f"Input dtype: {x.dtype}")
    
    # Try forward pass with debugging
    try:
        residual = x
        quantized_sum = torch.zeros_like(x)
        codes = []
        commit_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        usage_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        print(f"Initial commit_loss: {commit_loss} (type: {type(commit_loss)}, device: {commit_loss.device})")
        print(f"Initial usage_loss: {usage_loss} (type: {type(usage_loss)}, device: {usage_loss.device})")
        
        for level, emb in enumerate(quantizer.codebooks):
            print(f"\\nLevel {level}:")
            quantized, idx, probs = quantizer._quantize(residual, emb, level)
            print(f"  Quantized shape: {quantized.shape}")
            print(f"  Probs shape: {probs.shape}")
            print(f"  Probs dtype: {probs.dtype}")
            
            # Test KL div computation step by step
            uniform_prob = 1.0 / quantizer.codebook_size
            print(f"  uniform_prob: {uniform_prob} (type: {type(uniform_prob)})")
            
            avg_probs = probs.mean(dim=0)  # Average over batch
            print(f"  avg_probs shape: {avg_probs.shape}")
            print(f"  avg_probs dtype: {avg_probs.dtype}")
            print(f"  avg_probs device: {avg_probs.device}")
            
            log_term = torch.log(avg_probs / uniform_prob + 1e-8)
            print(f"  log_term shape: {log_term.shape}")
            print(f"  log_term dtype: {log_term.dtype}")
            
            kl_div = torch.sum(avg_probs * log_term)
            print(f"  kl_div: {kl_div} (type: {type(kl_div)}, device: {kl_div.device})")
            
            # Test the problematic line
            usage_reg_tensor = torch.tensor(quantizer.usage_reg, device=x.device, dtype=x.dtype)
            print(f"  usage_reg as tensor: {usage_reg_tensor}")
            
            usage_loss_update = usage_reg_tensor * kl_div
            print(f"  usage_loss_update: {usage_loss_update}")
            
            usage_loss = usage_loss + usage_loss_update
            print(f"  Updated usage_loss: {usage_loss}")
            
            codes.append(idx)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
        
        codes = torch.stack(codes, dim=1)  # [N, levels]
        print(f"\\nFinal codes shape: {codes.shape}")
        print(f"Final quantized shape: {quantized_sum.shape}")
        print(f"Final commit_loss: {commit_loss}")
        print(f"Final usage_loss: {usage_loss}")
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_quantizer()