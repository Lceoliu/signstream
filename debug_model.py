"""
Debug script to identify the model issue.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from signstream.models.rvq.rvq_model import RVQModel
from signstream.models.rvq.encoder import PART_DIMENSIONS

def debug_model():
    """Debug the model issue."""
    print("Debugging RVQ model...")
    
    # Create model
    model = RVQModel(
        latent_dim=32,
        chunk_len=8,
        codebook_size=16,
        levels=2,
        arch='mlp',
        commitment_beta=0.25,
        ema_decay=0.99,
        usage_reg=1e-3,
        num_layers=2,
        type_embed_dim=16,
        dropout=0.1,
        temporal_aggregation='mean'
    )
    
    print(f"Model created")
    
    # Test with face data
    part_name = 'face'
    frame_dim = PART_DIMENSIONS[part_name] * 3  # 68 * 3 = 204
    print(f"Testing {part_name} with frame_dim={frame_dim}")
    
    # Create input
    batch_size = 2
    chunk_len = 8
    x = torch.randn(batch_size, chunk_len, frame_dim)
    print(f"Input shape: {x.shape}")
    
    try:
        # Step through forward pass
        print("\\nStep 1: Encoder")
        z = model.encoder(x, part_name)
        print(f"Encoder output shape: {z.shape}")
        print(f"Encoder output dtype: {z.dtype}")
        print(f"Encoder output device: {z.device}")
        
        print("\\nStep 2: Quantizer")
        print(f"Quantizer usage_reg: {model.quantizer.usage_reg}")
        print(f"Quantizer usage_reg type: {type(model.quantizer.usage_reg)}")
        
        z_q, codes, q_loss, usage_loss = model.quantizer(z)
        print(f"Quantizer outputs:")
        print(f"  z_q shape: {z_q.shape}")
        print(f"  codes shape: {codes.shape}")
        print(f"  q_loss: {q_loss}")
        print(f"  usage_loss: {usage_loss}")
        
        print("\\nStep 3: Decoder")
        recon = model.decoder(z_q, part_name)
        print(f"Decoder output shape: {recon.shape}")
        
        print("\\nFull forward pass test:")
        recon, codes, q_loss, usage_loss, z_q = model(x, part_name)
        print(f"Success! Output shapes:")
        print(f"  recon: {recon.shape}")
        print(f"  codes: {codes.shape}")
        print(f"  z_q: {z_q.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def debug_minimal():
    """Minimal debug case."""
    print("\\n" + "="*50)
    print("Minimal debug case")
    print("="*50)
    
    from signstream.models.rvq.quantizer import ResidualVectorQuantizer
    
    quantizer = ResidualVectorQuantizer(
        dim=32,
        codebook_size=16,
        levels=2,
        usage_reg=1e-3
    )
    
    x = torch.randn(2, 32)
    
    try:
        z_q, codes, q_loss, usage_loss = quantizer(x)
        print(f"Minimal test passed:")
        print(f"  z_q: {z_q.shape}")
        print(f"  codes: {codes.shape}")
        print(f"  q_loss: {q_loss}")
        print(f"  usage_loss: {usage_loss}")
    except Exception as e:
        print(f"Minimal test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()
    debug_minimal()