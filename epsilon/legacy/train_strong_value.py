# train_strong_value.py
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LinithNet
from self_play_value_mcts import generate_value_dataset_from_mcts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10,
                        help="number of self-play + train cycles")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=64,
                        help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2,
                        help="training epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load", type=str, default="",
                        help="optional starting checkpoint")
    parser.add_argument("--save-prefix", type=str, default="linith_strong")
    args = parser.parse_args()

    device = torch.device(args.device)

    net = LinithNet().to(device)
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print(f"[train_strong] loaded model from {args.load}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for it in range(1, args.iterations + 1):
        print(f"\n[train_strong] iteration {it}/{args.iterations}")

        # 1) self-play with current net
        X, y = generate_value_dataset_from_mcts(
            net,
            num_games=args.games_per_iter,
            num_simulations=args.sims,
            device=args.device,
        )
        print(f"[train_strong] dataset: {X.shape[0]} positions")

        # 2) train value head on this dataset
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y).view(-1, 1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(1, args.epochs + 1):
            net.train()
            total_loss = 0.0
            total_n = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                _, preds = net(batch_x)      # ignore policy, use value
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                n = batch_x.size(0)
                total_loss += loss.item() * n
                total_n += n

            avg_loss = total_loss / max(1, total_n)
            print(f"[train_strong] iter {it}, epoch {epoch}/{args.epochs} "
                  f"- value loss {avg_loss:.4f}")

        # 3) save checkpoint for this iteration
        out_path = f"{args.save_prefix}_iter{it}.pt"
        torch.save(net.state_dict(), out_path)
        print(f"[train_strong] saved model to {out_path}")


if __name__ == "__main__":
    main()
