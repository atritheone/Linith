# selfplaypvloop.py
import argparse
import subprocess
from datetime import datetime
import os
import sys
import numpy as np

# Atreyan Era timestamp helper: AE = Gregorian year - 2020
# Example: 2025-11-23 03:37:00 -> 5AE-11-23 03:37:00

def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)


def build_multi_iter_replay(npz_files: list[str],
                            capacity: int,
                            out_path: str) -> None:
    """
    Build a replay dataset from multiple self-play npz files.

    - Concatenates X, Pi, Z from all npz_files.
    - If `capacity > 0` and total positions > capacity, uniformly samples.
    - Saves compressed npz to `out_path`.
    """
    if not npz_files:
        raise ValueError("No npz_files provided to build_multi_iter_replay.")

    X_list = []
    Pi_list = []
    Z_list = []

    total_positions = 0
    for path in npz_files:
        print(f"[replay] loading {path}")
        data = np.load(path)
        X = data["X"]
        Pi = data["Pi"]
        Z = data["Z"]

        X_list.append(X)
        Pi_list.append(Pi)
        Z_list.append(Z)

        total_positions += X.shape[0]

    X_all = np.concatenate(X_list, axis=0)
    Pi_all = np.concatenate(Pi_list, axis=0)
    Z_all = np.concatenate(Z_list, axis=0)

    N = X_all.shape[0]
    print(f"[replay] total positions before sampling: {N}")

    # Uniformly sample if we exceed capacity
    if capacity > 0 and N > capacity:
        idx = np.random.choice(N, size=capacity, replace=False)
        X_all = X_all[idx]
        Pi_all = Pi_all[idx]
        Z_all = Z_all[idx]
        print(f"[replay] sampled down to capacity {capacity} positions.")
    else:
        print("[replay] no sampling applied (capacity <= 0 or N <= capacity).")

    np.savez_compressed(out_path, X=X_all, Pi=Pi_all, Z=Z_all)
    print(f"[replay] saved merged replay dataset to {out_path}")



def main():
    ap = argparse.ArgumentParser(
        description="Multi-iteration PV self-play + training loop for Linith."
    )
    ap.add_argument("--start-model", type=str, required=True,
                    help="Initial PV model .pt file (e.g. linith_pv_iter1.pt)")
    ap.add_argument("--iterations", type=int, default=3,
                    help="How many self-play + train cycles to run")
    ap.add_argument("--games", type=int, default=300,
                    help="Self-play games per iteration")
    ap.add_argument("--sims", type=int, default=32,
                    help="MCTS simulations per move (if selfplaypv uses it)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Training epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-moves", type=int, default=10000)
    ap.add_argument("--tag", type=str, default="selfplaypv",
                    help="Prefix for output models and datasets")
    ap.add_argument("--replay-capacity", type=int, default=0,
                    help="Max positions to keep in replay buffer (0 = keep all)")
    args = ap.parse_args()

    start_time = datetime.now()
    print(f"Starting at {format_dt_ae(start_time)}")
    print(f"Start model - {args.start_model}")
    print(f"Iterations - {args.iterations}, Games per iteration - {args.games}, "
          f"Epochs - {args.epochs}, Sims - {args.sims}")

    current_model = args.start_model

    history_data_files: list[str] = []

    for i in range(1, args.iterations + 1):
        iter_start = datetime.now()
        print("\n====================================================")
        print(f"     ITERATION {i}/{args.iterations}")
        print(f"     Using model - {current_model}")
        print("====================================================\n")

        # Filenames for this iteration
        data_file = f"{args.tag}_iter{i}.npz"
        out_model = f"{args.tag}_iter{i+1}.pt"

        # 1) Self-play to generate dataset
        sp_cmd = [
            "python", "selfplaypv_cpp_gpu.py",
            "--model", current_model,
            "--games", str(args.games),
            "--sims", str(args.sims),
            "--device", args.device,
            "--max-moves", str(args.max_moves),
            "--out", data_file,
            "--replay-capacity", str(args.replay_capacity),
        ]
        run(sp_cmd)

        if not os.path.exists(data_file):
            print(f"[error] selfplay did not produce {data_file}, aborting.")
            break

        # ----------------------------------------------------------
        # 2) Build multi-iteration replay dataset (if enabled)
        # ----------------------------------------------------------
        history_data_files.append(data_file)

        # If replay_capacity <= 0, fall back to "this iteration only"
        if args.replay_capacity > 0 and len(history_data_files) > 0:
            # Use ALL accumulated datasets so far, then sample down
            # to args.replay_capacity positions in total.
            replay_file = f"{args.tag}_replay_iter{i}.npz"
            print(f"[replay] building multi-iteration replay to {replay_file}")
            build_multi_iter_replay(
                npz_files=history_data_files,
                capacity=args.replay_capacity,
                out_path=replay_file,
            )
            data_for_training = replay_file
        else:
            # Original behavior: only this iteration's dataset
            data_for_training = data_file
            print("[replay] disabled or capacity <= 0; "
                  "training only on current iteration's data.")

        # 3) Train from replay dataset to produce next model
        train_cmd = [
            "python", "train_pv_selfplay_gpu.py",
            "--base", current_model,
            "--data", data_for_training,
            "--out", out_model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", args.device,
        ]
        run(train_cmd)

        if not os.path.exists(out_model):
            print(f"[error] training did not produce {out_model}, aborting.")
            break

        iter_end = datetime.now()
        print(f"Iteration {i} complete.")
        print(f"Dataset - {data_file}")
        print(f"New model - {out_model}")
        print(f"Duration - {iter_end - iter_start}")

        current_model = out_model

    end_time = datetime.now()
    print("\n========== LOOP COMPLETE ==========")
    print(f"Started - {format_dt_ae(start_time)}")
    print(f"Finished - {format_dt_ae(end_time)}")
    print(f"Total duration - {end_time - start_time}")
    print(f"Final model - {current_model}")


if __name__ == "__main__":
    main()
