import sys
from pathlib import Path

import json
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from signstream.inference.export_tokens import export_samples


def _create_dummy_dataset(root: Path) -> None:
    (root / "frames_512x512").mkdir(parents=True)
    data = np.zeros((10, 134, 3), dtype=np.float32)
    np.save(root / "frames_512x512" / "sample.npy", data)
    annotations = {"sample": {"text": "hi", "gloss": "hi", "num_frames": 10, "signer": 0}}
    with open(root / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    split = {"train": ["sample"], "val": [], "test": []}
    with open(root / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(split, f)


def test_export_tokens(tmp_path) -> None:
    _create_dummy_dataset(tmp_path)
    ds = CSLDailyDataset(root_dir=str(tmp_path), split="train", chunk_len=5, fps=30)
    start, end = ds.body_part_indices["full_body"]
    num_points = end - start + 1
    model = RVQModel(
        frame_dim=num_points * 3,
        chunk_len=5,
        latent_dim=16,
        codebook_size=8,
        levels=2,
    )
    samples = export_samples(model, ds, num_samples=1, enable_rle=False)
    assert len(samples) == 1
    assert "tokens" in samples[0]
