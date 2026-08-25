from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from self_play import generate_self_play_dataset
from model import LinithNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="number of self-play games")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[train] generating dataset from {args.games} games...")
    X, y = generate_self_play_dataset(num_games=args.games)
    print("[train] dataset:", X.shape[0], "positions")

    # Tensors
    X_t = torch.from_numpy(X)          # (N, 6, 10, 10)
    y_t = torch.from_numpy(y).view(-1, 1)  # (N, 1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LinithNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("[train] starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            policy_logits, preds = model(batch_x)  # preds: (B,1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)

        avg_loss = total_loss / max(count, 1)
        print(f"[train] epoch {epoch}/{args.epochs} - loss {avg_loss:.4f}")

    torch.save(model.state_dict(), "linith_value_net.pt")
    print("[train] saved model to linith_value_net.pt")


if __name__ == "__main__":
    main()
