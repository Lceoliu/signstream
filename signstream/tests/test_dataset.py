import sys
from pathlib import Path

import json
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from signstream.data.datasets import CSLDailyDataset


def _create_dummy_dataset(root: Path) -> None:
    (root / "frames_512x512").mkdir(parents=True)
    data = np.zeros((20, 134, 3), dtype=np.float32)
    np.save(root / "frames_512x512" / "sample.npy", data)
    annotations = {
        "sample": {"text": "hi", "gloss": "hi", "num_frames": 20, "signer": 0}
    }
    with open(root / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    split = {"train": ["sample"], "val": [], "test": []}
    with open(root / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(split, f)


def test_dataset_chunking(tmp_path) -> None:
    _create_dummy_dataset(tmp_path)
    ds = CSLDailyDataset(root_dir=str(tmp_path), split="train", chunk_len=10, fps=30)
    sample = ds[0]
    assert sample["chunks"]["full_body"].shape[0] == 2
    from signstream.data.datasets import CSLDailyDataModule

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, collate_fn=CSLDailyDataModule.collate_fn
    )
    batch = next(iter(loader))
    assert batch["chunks"]["full_body"].shape[0] == 1
