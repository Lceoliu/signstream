import argparse
import json
import yaml
import torch
from typing import List

from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from .rle import rle_encode


def load_model(config: dict, checkpoint: str | None = None) -> RVQModel:
    part = "full_body"
    # Full body contains 133 keypoints (COCO WholeBody without global).
    num_points = 133
    frame_dim = num_points * 3
    model = RVQModel(
        frame_dim=frame_dim,
        chunk_len=config["data"]["chunk_len"],
        latent_dim=config["model"]["latent_dim"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"].get("commitment_beta", 0.25),
        arch=config["model"].get("arch", "mlp"),
    )
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
    model.eval()
    return model


def export_samples(
    model: RVQModel,
    dataset: CSLDailyDataset,
    num_samples: int,
    enable_rle: bool = False,
) -> List[dict]:
    device = next(model.parameters()).device
    results = []
    part = "full_body"
    for i in range(num_samples):
        sample = dataset[i]
        x = sample["chunks"][part]  # [N, L, K, 3]
        N, L, K, C = x.shape
        x_flat = x.view(N, L, K * C).to(device)
        with torch.no_grad():
            _, codes, _, _, _ = model(x_flat, part)
        codes_list = codes.cpu().tolist()
        tokens = [{"t": t, "FB": c} for t, c in enumerate(codes_list)]
        if enable_rle:
            fb_tokens = [tok["FB"] for tok in tokens]
            compressed = rle_encode(fb_tokens)
            tokens = [{"t": t, "FB": c} for t, c in enumerate(compressed)]
        results.append(
            {
                "video_name": sample["name"],
                "fps": dataset.fps,
                "chunk_len": dataset.chunk_len,
                "tokens": tokens,
                "rle": enable_rle,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RVQ tokens")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--out", type=str, default="tokens.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = CSLDailyDataset(
        root_dir=config["data"]["root"],
        split=args.split,
        chunk_len=config["data"]["chunk_len"],
        fps=config["data"]["fps"],
    )
    model = load_model(config, args.checkpoint)
    samples = export_samples(
        model, dataset, args.num_samples, enable_rle=config["export"]["enable_rle"]
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
