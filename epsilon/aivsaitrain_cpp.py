import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pv_model import LinithPVNet
import linith_selfplay_cpp as lsp


def train_pv_from_teacher_cpp(
    games: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_path: str,
    max_moves: int = 400,
    difficulty: str = "hard_train",
    value_coeff: float = 1.0,
    policy_coeff: float = 1.0,
):
    # 1) Generate teacher data using the C++ engine
    print("Generating teacher dataset with C++ self-play...")
    X, Pi, Z = lsp.generate_teacher_dataset_cpp(
        num_games=games,
        max_moves=max_moves,
        difficulty=difficulty,
    )

    # Ensure correct dtypes/shapes
    X = np.asarray(X, dtype=np.float32)
    Pi = np.asarray(Pi, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)

    X_t = torch.from_numpy(X)
    Pi_t = torch.from_numpy(Pi)
    Z_t = torch.from_numpy(Z).view(-1, 1)

    ds = TensorDataset(X_t, Pi_t, Z_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    dev = torch.device(device)
    net = LinithPVNet().to(dev)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.KLDivLoss(reduction="batchmean")

    print("Starting PV training from Hard-AI teacher data (C++ self-play)")
    print(f"Device - {dev}")
    print(f"Epochs - {epochs}, batch_size - {batch_size}, lr - {lr}")
    print(f"value_coeff - {value_coeff}, policy_coeff - {policy_coeff}")
    print()

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        n = 0

        for bx, bpi, bz in dl:
            bx = bx.to(dev)
            bpi = bpi.to(dev)
            bz = bz.to(dev)

            opt.zero_grad()
            logits, v = net(bx)

            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = ce(log_probs, bpi)
            value_loss = mse(v, bz)

            loss = policy_coeff * policy_loss + value_coeff * value_loss
            loss.backward()
            opt.step()

            bs = bx.size(0)
            total_loss += loss.item() * bs
            total_policy += policy_loss.item() * bs
            total_value += value_loss.item() * bs
            n += bs

        print(
            f"Epoch {epoch}/{epochs} "
            f"Loss - {total_loss / n:.6f} "
            f"Policy - {total_policy / n:.6f} "
            f"Value - {total_value / n:.6f}"
        )

    torch.save(net.state_dict(), out_path)
    print()
    print(f"Saved teacher-trained PV net to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Linith PV net from C++ Hard-AI self-play."
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="linith_pv_from_hard_cpp.pt")
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard_train",
        choices=["easy", "medium", "hard", "hard_train"],
    )
    parser.add_argument("--value-coeff", type=float, default=1.0)
    parser.add_argument("--policy-coeff", type=float, default=1.0)

    args = parser.parse_args()

    train_pv_from_teacher_cpp(
        games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out_path=args.out,
        max_moves=args.max_moves,
        difficulty=args.difficulty,
        value_coeff=args.value_coeff,
        policy_coeff=args.policy_coeff,
    )


if __name__ == "__main__":
    main()
