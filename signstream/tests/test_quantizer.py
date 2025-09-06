import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from signstream.models.rvq.quantizer import ResidualVectorQuantizer


def test_quantizer_roundtrip() -> None:
    quant = ResidualVectorQuantizer(dim=8, codebook_size=16, levels=2)
    x = torch.randn(4, 8)
    q, codes, c_loss, u_loss = quant(x)
    assert q.shape == x.shape
    assert codes.shape == (4, 2)
    assert c_loss.shape == torch.Size([])
    assert u_loss.shape == torch.Size([])
