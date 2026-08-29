import argparse
from datetime import datetime
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from pv_model import LinithPVNet
from action_space import ACTION_SIZE


# ---------------------------------------------------------
# Atreyan era helper (for logging symmetry with hyperloop style)
# ---------------------------------------------------------
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class SelfPlayDataset(Dataset):
    def __init__(self, X, Pi, Z):
        self.X = X.astype(np.float32)
        self.Pi = Pi.astype(np.float32)
        self.Z = Z.astype(np.float32)

        assert self.X.shape[0] == self.Pi.shape[0] == self.Z.shape[0]
        assert self.X.shape[1:] == (8, 10, 10)
        assert self.Pi.shape[1] == ACTION_SIZE

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]        # (8,10,10)
        pi = self.Pi[idx]      # (A,)
        z = self.Z[idx]        # ()
        return (
            torch.from_numpy(x),
            torch.from_numpy(pi),
            torch.tensor(z, dtype=torch.float32),
        )


# ---------------------------------------------------------
# Symmetry augmentation (placeholder hook)
# ---------------------------------------------------------
def apply_symmetry_augmentation(X, Pi, Z):
    """
    8-way (rot/flip) symmetry augmentation hook.

    IMPORTANT:
    - If you already have proper action-index symmetry mapping for Pi,
      replace this placeholder with your real augmentation code.
    - For now this is a no-op to avoid breaking the action encoding.
    """
    return X, Pi, Z


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_one_model(

    base_model_path: str,
    data_npz_path: str,
    out_model_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str,
    use_symmetry: bool,
    device: str = "cpu",
):
    dev = torch.device(device)

    print()
    print("========================")
    print("   Training New Model  ")
    print("========================")
    print()

    print(f"Loading data from {data_npz_path}")
    d = np.load(data_npz_path)
    X = d["X"]
    Pi = d["Pi"]
    Z = d["Z"]

    print(f"Raw dataset shapes - X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")

    if use_symmetry:
        print("Applying symmetry augmentation (hook)...")
        X, Pi, Z = apply_symmetry_augmentation(X, Pi, Z)
        print(f"After symmetry - X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")

    dataset = SelfPlayDataset(X, Pi, Z)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Load base model
    print(f"Loading base model from {base_model_path}")
    net = LinithPVNet()
    net.load_state_dict(torch.load(base_model_path, map_location=dev))
    net.to(dev)

    # Optimizer with L2 weight decay
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # L2
    )

    # LR schedule
    scheduler = None
    lr_schedule = (lr_schedule or "none").lower()
    if lr_schedule == "step":
        # simple step schedule: decay by 0.5 every half of total epochs
        step_size = max(1, epochs // 2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        print(f"Using StepLR schedule: step_size={step_size}, gamma=0.5")
    elif lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        print("Using CosineAnnealingLR schedule")
    else:
        print("Using constant learning rate (no LR schedule)")

    # Loss functions
    mse_loss = nn.MSELoss()

    start_time = datetime.now()
    print(f"Training start - {format_dt_ae(start_time)}")

    net.train()
    for epoch in range(1, epochs + 1):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, target_pi, target_z = batch
            x = x.to(dev)                    # [B,8,10,10]
            target_pi = target_pi.to(dev)    # [B,A]
            target_z = target_z.to(dev)      # [B]

            optimizer.zero_grad()

            logits, value = net(x)           # logits: [B,A], value: [B,1]
            value = value.view(-1)           # [B]

            # policy loss: cross-entropy with soft targets Pi
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()

            # value loss: MSE
            value_loss = mse_loss(value, target_z)

            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            epoch_policy_loss /= n_batches
            epoch_value_loss /= n_batches
            epoch_total_loss /= n_batches

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} "
            f"LR - {current_lr:.6f}  "
            f"Policy - {epoch_policy_loss:.4f}  "
            f"Value - {epoch_value_loss:.4f}  "
            f"Total - {epoch_total_loss:.4f}"
        )

    end_time = datetime.now()
    print(f"Training end   - {format_dt_ae(end_time)}")
    print(f"Duration       - {end_time - start_time}")

    # Save model
    out_dir = os.path.dirname(out_model_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(net.state_dict(), out_model_path)
    print(f"Saved trained model to {out_model_path}")


# ---------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Linith PV net from self-play data.")

    ap.add_argument("--base", type=str, required=True,
                    help="Base model .pt to start from.")
    ap.add_argument("--data", type=str, required=True,
                    help="Dataset .npz (X, Pi, Z). Can be a replay buffer file.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output model .pt path.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    # New knobs for hyperloop
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay lambda.",
    )
    ap.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        help="LR schedule: 'none', 'step', or 'cosine'.",
    )
    ap.add_argument(
        "--symmetry-aug",
        action="store_true",
        help="Enable symmetry augmentation (rot/flip).",
    )

    args = ap.parse_args()

    train_one_model(
        base_model_path=args.base,
        data_npz_path=args.data,
        out_model_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        use_symmetry=args.symmetry_aug,
        device=args.device,
    )


if __name__ == "__main__":
    main()
