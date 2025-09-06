import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    set_seed(config["train"]["seed"])

    dataset = CSLDailyDataset(
        root_dir=config["data"]["root"],
        split=config["data"].get("split", "train"),
        chunk_len=config["data"]["chunk_len"],
        fps=config["data"]["fps"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )

    part = "full_body"
    start, end = dataset.body_part_indices[part]
    num_points = end - start + 1
    input_dim = config["data"]["chunk_len"] * num_points * 3

    model = RVQModel(
        input_dim=input_dim,
        latent_dim=config["model"]["latent_dim"],
        codebook_size=config["model"]["rvq"]["codebook_size"],
        levels=config["model"]["rvq"]["levels"],
        commitment_beta=config["model"]["rvq"].get("commitment_beta", 0.25),
    )

    device = torch.device(config["train"].get("device", "cpu"))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["wd"],
    )

    epochs = config["train"]["epochs"]
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch["chunks"][part]  # [B, N, L, K, 3]
            B, N, L, K, C = x.shape
            x_flat = x.view(B * N, L * K * C).to(device)
            recon, codes, q_loss = model(x_flat, part)
            recon = recon.view(B * N, L, K, C)
            loss_recon = F.mse_loss(recon, x.to(device))
            loss = loss_recon + q_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1} loss {loss.item():.4f}")

    torch.save({"model_state": model.state_dict()}, "rvq_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RVQ model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
