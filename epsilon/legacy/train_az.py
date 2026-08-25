# train_az.py
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LinithNet
from self_play_mcts import generate_mcts_self_play_dataset
from action_space import ACTION_SIZE


def az_loss(policy_logits, value_pred, pi_target, v_target):
    """
    AlphaZero-style loss:
      L = (z - v)^2 - pi^T log p  + 1e-4 * ||theta||^2 (L2 done via optimizer)
    """
    # value loss (MSE)
    value_loss = nn.functional.mse_loss(value_pred, v_target)

    # policy loss (cross-entropy between target pi and softmax(policy_logits))
    log_probs = nn.functional.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.mean(torch.sum(pi_target * log_probs, dim=1))

    return policy_loss + value_loss, policy_loss, value_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load", type=str, default="", help="optional path to existing model")
    parser.add_argument("--save", type=str, default="linith_az_net.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    net = LinithNet().to(device)
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print(f"[train_az] loaded model from {args.load}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # 1. Generate self-play data
    print(f"[train_az] generating MCTS self-play data from {args.games} games...")
    X, pi, v = generate_mcts_self_play_dataset(
        net,
        num_games=args.games,
        num_simulations=args.sims,
        device=args.device,
    )
    print("[train_az] dataset:", X.shape[0], "positions")

    X_t = torch.from_numpy(X)                       # (N, 6, 10, 10)
    pi_t = torch.from_numpy(pi)                    # (N, ACTION_SIZE)
    v_t = torch.from_numpy(v).view(-1, 1)          # (N, 1)

    dataset = TensorDataset(X_t, pi_t, v_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("[train_az] starting training...")
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = total_pl = total_vl = 0.0
        total_n = 0

        for batch_x, batch_pi, batch_v in loader:
            batch_x = batch_x.to(device)
            batch_pi = batch_pi.to(device)
            batch_v = batch_v.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = net(batch_x)
            loss, pl, vl = az_loss(policy_logits, value_pred, batch_pi, batch_v)
            loss.backward()
            optimizer.step()

            n = batch_x.size(0)
            total_loss += loss.item() * n
            total_pl += pl.item() * n
            total_vl += vl.item() * n
            total_n += n

        avg_loss = total_loss / total_n
        avg_pl = total_pl / total_n
        avg_vl = total_vl / total_n
        print(f"[train_az] epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} "
              f"(policy {avg_pl:.4f}, value {avg_vl:.4f})")

    torch.save(net.state_dict(), args.save)
    print(f"[train_az] saved model to {args.save}")


if __name__ == "__main__":
    main()
