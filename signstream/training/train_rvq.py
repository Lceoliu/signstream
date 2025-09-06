import argparse
import random
import yaml
import argparse
import argparse
import random
import yaml
import torch
from torch.utils.data import DataLoader

from signstream.data.datasets import CSLDailyDataset
from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.seed import set_seed
from signstream.training.losses import recon_loss, temporal_loss, usage_regularization


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

    device = torch.device(config["train"].get("device", "cpu"))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["wd"],
    )

    epochs = config["train"]["epochs"]
    usage_coeff = config["model"]["rvq"].get("usage_beta", 0.0)
    temporal_alpha = config["train"].get("temporal_alpha", 0.0)

    best_loss = float("inf")
    code_usage = [set() for _ in range(config["model"]["rvq"]["levels"])]

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch["chunks"][part]  # [B, N, L, K, 3]
            B, N, L, K, C = x.shape
            x_seq = x.view(B * N, L, K * C).to(device)
            recon, codes, q_loss, usage_loss, z_q = model(x_seq, part)
            loss_r = recon_loss(recon, x_seq, loss_type="huber")
            z_q_seq = z_q.view(B, N, -1)
            loss_t = temporal_loss(z_q_seq) if N > 1 else torch.tensor(0.0, device=device)
            loss = loss_r + q_loss + usage_regularization(usage_loss, usage_coeff) + temporal_alpha * loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for lvl in range(codes.shape[1]):
                code_usage[lvl].update(codes[:, lvl].detach().cpu().tolist())
        avg_loss = epoch_loss / len(dataloader)
        print(f"epoch {epoch+1} loss {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": model.state_dict()}, "rvq_model_best.pt")

    # Codebook utilization
    for lvl, used in enumerate(code_usage):
        util = len(used) / config["model"]["rvq"]["codebook_size"]
        print(f"level {lvl} utilization: {util:.2%}")

    # Sample a training video and log tokens
    sample = dataset[random.randrange(len(dataset))]
    x = sample["chunks"][part]
    N, L, K, C = x.shape
    x_seq = x.view(N, L, K * C).to(device)
    with torch.no_grad():
        _, codes, _, _, _ = model(x_seq, part)
    print("sample tokens:", codes.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RVQ model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
