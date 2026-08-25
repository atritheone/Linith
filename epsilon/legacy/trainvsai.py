# trainvsai.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LinithNet


def trainvsai(
    base_model: str,
    data_path: str,
    out_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    dev = torch.device(device)
    net = LinithNet().to(dev)
    net.load_state_dict(torch.load(base_model, map_location=dev))

    data = np.load(data_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).view(-1, 1)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        net.train()
        total = 0.0
        n = 0
        for bx, by in dl:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            _, preds = net(bx)
            loss = crit(preds, by)
            loss.backward()
            opt.step()
            total += loss.item() * bx.size(0)
            n += bx.size(0)
        print(f"[trainvsai] epoch {epoch}/{epochs} loss={total/n:.6f}")

    torch.save(net.state_dict(), out_model)
    print(f"[trainvsai] saved updated model to {out_model}")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Linith net on model-vs-ai games.")
    ap.add_argument("--base", type=str, required=True, help="Starting model .pt (e.g. linith_from_hard_200.pt)")
    ap.add_argument("--data", type=str, required=True, help="NPZ from self_play_vs_hard.py")
    ap.add_argument("--out", type=str, required=True, help="Output model path")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    trainvsai(
        base_model=args.base,
        data_path=args.data,
        out_model=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
