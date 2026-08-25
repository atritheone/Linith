import argparse
import subprocess
from datetime import datetime
import os
import sys
import re
import shutil
import numpy as np  # <-- for replay buffer handling


## ----------------------------------------------------------- ##
## hyperloop expects a starting model in 'epsilon_0.x.pt' format
## ----------------------------------------------------------- ##


# atreyan era timestamp helper
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)


def run_eval_modelvsmodel(
    eval_script: str,
    best_model: str,
    candidate_model: str,
    games: int,
    sims: int,
    device: str,
    max_moves: int,
    out_prefix: str,
) -> float:
    """
    Run modelvsmodel_cpp.py and return winrate for candidate model (Model B).

    We expect output lines like:
        Model A wins - 12
        Model B wins - 18
        Draws       - 20

    We compute:
        winrate_B = (wins_b + 0.5 * draws) / total_games
    """
    out_file = f"{out_prefix}.npz"

    cmd = [
        "python",
        eval_script,
        "--model-a",
        best_model,
        "--model-b",
        candidate_model,
        "--games",
        str(games),
        "--sims",
        str(sims),
        "--device",
        device,
        "--max-moves",
        str(max_moves),
        "--replay-capacity",
        "0",
        "--out",
        out_file,
    ]

    print("\n>>", " ".join(cmd))
    sys.stdout.flush()

    # capture output for parsing, but also echo it back to the shell
    res = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout or ""
    stderr = res.stderr or ""

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, file=sys.stderr, end="")
    sys.stdout.flush()

    # parse summary lines from stdout
    m_a = re.search(r"Model A wins\s*-\s*(\d+)", stdout)
    m_b = re.search(r"Model B wins\s*-\s*(\d+)", stdout)
    m_d = re.search(r"Draws\s*-\s*(\d+)", stdout)

    if not (m_a and m_b and m_d):
        raise RuntimeError(
            "Could not parse win/draw counts from evaluation output. "
            "Expected summary lines - 'Model A wins - X', "
            "'Model B wins - Y', 'Draws - Z'"
        )

    wins_a = int(m_a.group(1))
    wins_b = int(m_b.group(1))
    draws = int(m_d.group(1))

    total_games = wins_a + wins_b + draws
    if total_games == 0:
        raise RuntimeError("Evaluation reported zero total games.")

    # winrate of Model B, with draws counting as 0.5
    winrate_b = (wins_b + 0.5 * draws) / float(total_games)

    print(
        f"Parsed - A = {wins_a}, B = {wins_b}, Draws = {draws}, "
        f"Games = {total_games}, New model win rate = {winrate_b:.4f}"
    )

    return winrate_b


def split_name_version(stem: str) -> tuple[str, str]:
    """
    Split 'epsilon_0.21' -> ('epsilon_', '0.21').
    Assumes the filename ends in a numeric version (with optional dot).
    """
    m = re.match(r"^(.*?)(\d+(?:\.\d+)?)$", stem)
    if not m:
        raise ValueError(f"Cannot parse version from '{stem}'")
    return m.group(1), m.group(2)


def bump_minor(version: str) -> str:
    """
    Bump the *second decimal place*.

    Rules:
      '0.1'  -> '0.11'
      '0.32' -> '0.33'
      '0.99' -> '1.00'
    """
    if "." not in version:
        # No decimal part: treat as ".00" and bump second decimal -> ".01"
        major_str = version
        frac_str = "00"
    else:
        major_str, frac_str = version.split(".", 1)

    # Normalise fractional part to 2 digits
    if len(frac_str) == 0:
        frac_str = "00"
    elif len(frac_str) == 1:
        frac_str = frac_str + "0"
    else:
        frac_str = frac_str[:2]  # ignore anything beyond 2 digits

    major = int(major_str)
    frac = int(frac_str)

    # bump the 2-digit fractional part
    frac += 1
    if frac >= 100:
        frac -= 100
        major += 1

    new_frac_str = f"{frac:02d}"
    return f"{major}.{new_frac_str}"



# -------------------- replay buffer helpers --------------------


def _load_npz(path: str):
    d = np.load(path)
    return d["X"], d["Pi"], d["Z"]


def update_replay_buffer_npz(
    replay_path: str,
    new_data_path: str,
    capacity: int,
) -> str:
    """
    Multi-iteration replay buffer on disk.

    - new_data_path: npz with X, Pi, Z from this iteration.
    - replay_path:   global npz file storing all replay positions so far.
    - capacity:      max positions to keep (0 = unlimited).

    Returns the path that should be passed to the trainer (replay_path).
    """
    X_new, Pi_new, Z_new = _load_npz(new_data_path)

    if os.path.exists(replay_path):
        X_old, Pi_old, Z_old = _load_npz(replay_path)
        X = np.concatenate([X_old, X_new], axis=0)
        Pi = np.concatenate([Pi_old, Pi_new], axis=0)
        Z = np.concatenate([Z_old, Z_new], axis=0)
    else:
        X, Pi, Z = X_new, Pi_new, Z_new

    # capacity is in positions, not games
    if capacity is not None and capacity > 0 and X.shape[0] > capacity:
        X = X[-capacity:]
        Pi = Pi[-capacity:]
        Z = Z[-capacity:]

    np.savez_compressed(replay_path, X=X, Pi=Pi, Z=Z)
    return replay_path


def main():
    ap = argparse.ArgumentParser(
        description="Multi-iteration PV self-play + training + eval loop for Linith."
    )
    ap.add_argument(
        "--start-model",
        type=str,
        required=True,
        help="Initial PV model .pt file (e.g. epsilon_0.21.pt)",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="How many self-play + train cycles to run",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=300,
        help="Self-play games per iteration",
    )
    ap.add_argument(
        "--sims",
        type=int,
        default=32,
        help="MCTS simulations per move for self-play",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per iteration",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-moves", type=int, default=10000)
    ap.add_argument(
        "--tag",
        type=str,
        default="selfplaypv",
        help="Prefix for output datasets (npz files)",
    )
    ap.add_argument(
        "--replay-capacity",
        type=int,
        default=0,
        help="Max positions to keep in replay buffer (0 = keep all = unlimited)",
    )

    # ---- training hyperparameters passed through to train_pv_selfplay.py ----
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay lambda passed to trainer.",
    )
    ap.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        help="Learning rate schedule name (trainer interprets this).",
    )
    ap.add_argument(
        "--symmetry-aug",
        action="store_true",
        help="Enable 8-way (rot/flip) symmetry augmentation in trainer.",
    )

    # ---- evaluation / selection options ----
    ap.add_argument(
        "--eval-script",
        type=str,
        default="modelvsmodel_cpp_gpu.py",
        help="Script used to evaluate models (two-model self-play)",
    )
    ap.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Evaluation games per candidate",
    )
    ap.add_argument(
        "--eval-sims",
        type=int,
        default=0,
        help="MCTS sims for evaluation (0 = use --sims)",
    )
    ap.add_argument(
        "--eval-max-moves",
        type=int,
        default=10000,
        help="Max moves per eval game (can differ from training self-play)",
    )
    ap.add_argument(
        "--eval-out-prefix",
        type=str,
        default="eval_ab",
        help="Prefix for temporary eval dataset npz",
    )
    ap.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.52,
        help=(
            "Win rate (candidate vs best) needed to promote "
            "(using wins_B + 0.5*draws over total games)."
        ),
    )
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation and always promote latest candidate",
    )

    args = ap.parse_args()

    # derive naming scheme from start model
    start_model_abs = os.path.abspath(args.start_model)
    base_dir, start_file = os.path.split(start_model_abs)
    stem_no_ext, ext = os.path.splitext(start_file)  # 'epsilon_0.21', '.pt'
    name_prefix, version_str = split_name_version(stem_no_ext)
    base_stem = f"{name_prefix}{version_str}"       # 'epsilon_0.21'

    final_version = bump_minor(version_str)         # '0.22'
    final_model_name = f"{name_prefix}{final_version}{ext}"
    final_model_path = os.path.join(base_dir, final_model_name)

    eval_sims = args.eval_sims or args.sims

    start_time = datetime.now()
    print(f"Starting at {format_dt_ae(start_time)}")
    print(f"Start model - {start_model_abs}")
    print(
        f"Iterations - {args.iterations}, Games per iteration - {args.games}, "
        f"Epochs - {args.epochs}, Sims - {args.sims}"
    )
    print(f"Final bumped version will be - {final_model_path}")

    # best_model is the one used for self-play and as baseline for evaluation
    best_model = start_model_abs

    # global replay npz path (for multi-iteration buffer)
    replay_npz_path = f"{args.tag}_replay.npz"

    for i in range(1, args.iterations + 1):
        if i > 26:
            raise ValueError(
                "More than 26 iterations requested; letter suffix scheme only "
                "supports up to 26 (a–z)."
            )

        iter_start = datetime.now()
        letter = chr(ord("a") + i - 1)  # 1 -> 'a', 2 -> 'b', ...
        candidate_name = f"{base_stem}{letter}{ext}"   # epsilon_0.21a.pt, etc.
        candidate_model = os.path.join(base_dir, candidate_name)

        print("\n============================================================")
        print(f"     ITERATION {i}/{args.iterations}")
        print(f"     Using best model as base - {best_model}")
        print(f"     Candidate model this iteration - {candidate_model}")
        print("============================================================\n")

        # filenames for this iteration's dataset
        data_file = f"{args.tag}_iter{i}.npz"

        # 1) self-play to generate dataset (from current best model)
        sp_cmd = [
            "python",
            "selfplaypv_cpp_gpu.py",
            "--model",
            best_model,
            "--games",
            str(args.games),
            "--sims",
            str(args.sims),
            "--device",
            args.device,
            "--max-moves",
            str(args.max_moves),
            "--out",
            data_file,
            "--replay-capacity",
            str(args.replay_capacity),
        ]
        run(sp_cmd)

        if not os.path.exists(data_file):
            print(f"[error] Self-Play did not produce {data_file}, aborting.")
            break

        # 1b) update multi-iteration replay buffer (if enabled)
        train_data_file = data_file
        if args.replay_capacity != 0:
            try:
                train_data_file = update_replay_buffer_npz(
                    replay_npz_path,
                    data_file,
                    capacity=args.replay_capacity if args.replay_capacity > 0 else 0,
                )
                print(
                    f"Updated replay buffer {replay_npz_path} "
                    f"from {data_file}; trainer will use {train_data_file}"
                )
            except Exception as e:
                print(f"[warning] Failed to update replay buffer: {e}")
                print("[warning] Falling back to using this iteration only.")
                train_data_file = data_file

        # 2) train from replay (or current dataset) to produce candidate model
        train_cmd = [
            "python",
            "train_pv_selfplay_gpu.py",
            "--base",
            best_model,
            "--data",
            train_data_file,
            "--out",
            candidate_model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--lr-schedule",
            args.lr_schedule,
        ]
        if args.symmetry_aug:
            train_cmd.append("--symmetry-aug")

        run(train_cmd)

        if not os.path.exists(candidate_model):
            print(f"[error] Training did not produce {candidate_model}, aborting.")
            break

        # 3) evaluate candidate vs best and decide whether to promote
        promoted = False
        winrate = None

        if args.no_eval:
            print("Evaluation disabled (--no-eval). Promoting candidate unconditionally.")
            best_model = candidate_model
            promoted = True
        else:
            try:
                winrate = run_eval_modelvsmodel(
                    args.eval_script,
                    best_model,
                    candidate_model,
                    args.eval_games,
                    eval_sims,
                    args.device,
                    args.eval_max_moves,
                    f"{args.eval_out_prefix}_iter{i}",
                )
                print(f"Candidate win rate (with draws as 0.5) = {winrate:.4f}")
                if winrate >= args.promotion_threshold:
                    print(
                        f"Candidate promoted (win rate {winrate:.3f} "
                        f">= threshold {args.promotion_threshold:.3f})"
                    )
                    best_model = candidate_model
                    promoted = True
                else:
                    print(
                        f"Candidate NOT promoted (win rate {winrate:.3f} "
                        f"< threshold {args.promotion_threshold:.3f}); "
                        f"keeping best model = {best_model}"
                    )
            except Exception as e:
                print(f"[warning] Evaluation failed {e}")
                print("[warning] Keeping existing best model and continuing.")

        iter_end = datetime.now()
        print()
        print(f"Iteration {i} complete.")
        print()
        print(f"Dataset - {data_file}")
        print(f"Train dataset used - {train_data_file}")
        print(f"Candidate model - {candidate_model}")
        print(f"Promoted - {promoted}")
        if winrate is not None:
            print(f"Win rate (candidate vs best) - {winrate:.4f}")
        print(f"Duration - {iter_end - iter_start}")

    # 4) final model - copy best model to bumped version name
    end_time = datetime.now()
    if not os.path.exists(best_model):
        print(
            f"[error] Best model {best_model} does not exist at end of loop; "
            "no final model produced."
        )
    else:
        if os.path.abspath(best_model) == os.path.abspath(final_model_path):
            print(f" Best model already has final name - {final_model_path}")
        else:
            print(
                f" Copying best model\n"
                f"           from - {best_model}\n"
                f"           to -   {final_model_path}"
            )
            shutil.copyfile(best_model, final_model_path)

    print("\n========== LOOP COMPLETE ==========")
    print(f"Started - {format_dt_ae(start_time)}")
    print(f"Finished - {format_dt_ae(end_time)}")
    print(f"Total duration - {end_time - start_time}")
    print(f"Best model - {best_model}")
    print(f"Final versioned model - {final_model_path}")
    print("===================================")
    print()


if __name__ == "__main__":
    main()
