import argparse
import subprocess
from datetime import datetime
import os
import sys
import re
import shutil
import numpy as np
import time
import threading


## ----------------------------------------------------------- ##
## hyperloop expects a starting model in 'epsilon_0.x.pt' format
## ----------------------------------------------------------- ##


# Atreyan Era timestamp helper
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)

# -------------------- worker progress tracking --------------------

def _worker_reader(proc: subprocess.Popen, worker_id: int, progress: dict, lock: threading.Lock) -> None:
    """
    Read a worker's stdout, parse progress, and update shared state.

    We EXPECT lines from selfplay like:
      "Game 3/10 - Sun won (started ..., ended ..., duration ...)"
      "Games requested - 10"
    """
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue

            # Parse "Game X/Y - ..."
            # e.g. "Game 3/10 - Sun won (started ..., ended ..., duration ...)"
            m_game = re.search(r"Game\s+(\d+)\s*/\s*(\d+)\s*-", line)
            if m_game:
                cur = int(m_game.group(1))
                total = int(m_game.group(2))
                with lock:
                    prog = progress.setdefault(worker_id, {"current": 0, "total": total})
                    prog["current"] = cur
                    prog["total"] = total
                continue

            # Parse "Games requested - N" (fallback total if needed)
            m_req = re.search(r"Games requested\s*-\s*(\d+)", line)
            if m_req:
                total = int(m_req.group(1))
                with lock:
                    prog = progress.setdefault(worker_id, {"current": 0, "total": total})
                    prog["total"] = total
                continue

            # if you ever want to surface key events, you can do:
            # if "Self-Play Summary" in line:
            #     print(f"[W{worker_id}] {line}")

    except Exception as e:
        print(f"[W{worker_id}] [reader error] {e}")
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


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
        sys.executable,
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


# -------------------- parallel self-play helpers --------------------


def run_selfplay_sharded(
    best_model: str,
    games: int,
    sims: int,
    device: str,
    max_moves: int,
    tag: str,
    iteration: int,
    workers: int,
) -> str:
    """
    Run self-play in parallel shards and merge them into one dataset.

    Returns path to merged npz: f"{tag}_iter{iteration}.npz"
    """
    sp_start = datetime.now()

    # Single-worker fallback behaves as before
    if workers <= 1 or games <= 1:
        data_file = f"{tag}_iter{iteration}.npz"
        sp_cmd = [
            sys.executable,
            "-u",
            "selfplaypv_cpp_gpu.py",
            "--model",
            best_model,
            "--games",
            str(games),
            "--sims",
            str(sims),
            "--device",
            device,
            "--max-moves",
            str(max_moves),
            "--out",
            data_file,
            "--replay-capacity",
            "0",
        ]

        print(
            f"\nParallel Self-Play (single worker) - {games} games "
            f"(sims - {sims}, device - {device})"
        )
        print(f"Started - {format_dt_ae(sp_start)}")

        run(sp_cmd)
        if not os.path.exists(data_file):
            raise RuntimeError(f"Self-Play did not produce {data_file}")

        sp_end = datetime.now()
        print(f"End - {format_dt_ae(sp_end)}")
        print(f"Duration - {sp_end - sp_start}")
        return data_file

    actual_workers = min(workers, games)
    base = games // actual_workers
    rem = games % actual_workers

    print(
        f"\nParallel Self-Play - {games} games "
        f"across {actual_workers} workers "
        f"(base - {base}, remainder - {rem})"
    )
    print(f"Start - {format_dt_ae(sp_start)}")

    shard_paths: list[str] = []
    procs: list[tuple[subprocess.Popen, threading.Thread, int]] = []

    progress: dict[int, dict] = {}
    lock = threading.Lock()

    # Spawn workers
    for w in range(actual_workers):
        g_w = base + (1 if w < rem else 0)
        if g_w <= 0:
            continue

        worker_id = w + 1
        shard_out = f"{tag}_iter{iteration}_shard{worker_id}.npz"
        shard_paths.append(shard_out)

        cmd = [
            sys.executable,
            "-u",  # unbuffered so prints come through immediately
            "selfplaypv_cpp_gpu.py",
            "--model",
            best_model,
            "--games",
            str(g_w),
            "--sims",
            str(sims),
            "--device",
            device,
            "--max-moves",
            str(max_moves),
            "--out",
            shard_out,
            "--replay-capacity",
            "0",
        ]

        print("\n>>", " ".join(cmd))
        sys.stdout.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        # initialise progress entry
        with lock:
            progress[worker_id] = {"current": 0, "total": g_w}

        t = threading.Thread(
            target=_worker_reader,
            args=(proc, worker_id, progress, lock),
            daemon=True,
        )
        t.start()
        procs.append((proc, t, worker_id))

    print()
    print("Starting workers...")
    print()

    # periodically print status until all workers exit
    last_progress_snapshot = {}
    still_running = True

    while still_running:
        time.sleep(1.5)

        still_running = any(proc.poll() is None for proc, _, _ in procs)

        changed = False

        with lock:
            cols = []
            for _, _, wid in procs:
                prog = progress.get(wid, {"current": 0, "total": 0})
                cur = prog.get("current", 0)
                total = prog.get("total", 0)

                # Check for changes
                prev = last_progress_snapshot.get(wid, -1)
                if cur != prev:
                    changed = True
                last_progress_snapshot[wid] = cur

                cols.append(f"Worker {wid} - {cur}/{total}")

            status_line = " | ".join(cols)

        # Only print if changed
        if changed:
            print(f"Iteration {iteration} | {status_line}")

    # ensure threads are finished and collect exit codes
    exit_codes = []
    for proc, t, wid in procs:
        proc.wait()
        t.join(timeout=1.0)
        exit_codes.append(proc.returncode)

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more self-play workers failed {exit_codes}")

    # merge shards
    X_list = []
    Pi_list = []
    Z_list = []

    for path in shard_paths:
        if not os.path.exists(path):
            raise RuntimeError(f"Shard {path} not found after Self-Play")

        X_s, Pi_s, Z_s = _load_npz(path)
        if X_s.shape[0] == 0:
            continue
        X_list.append(X_s)
        Pi_list.append(Pi_s)
        Z_list.append(Z_s)

    if not X_list:
        raise RuntimeError(
            f"No data produced by any shard for iteration {iteration}"
        )

    X = np.concatenate(X_list, axis=0)
    Pi = np.concatenate(Pi_list, axis=0)
    Z = np.concatenate(Z_list, axis=0)

    data_file = f"{tag}_iter{iteration}.npz"
    np.savez_compressed(data_file, X=X, Pi=Pi, Z=Z)

    # optional - clean up shard files
    for path in shard_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    print()
    print(f"====== Parallel Self-Play complete for iteration {iteration} ======")
    print()
    print(
        f"Merged {len(shard_paths)} shards into {data_file} "
        f"(X={X.shape}, Pi={Pi.shape}, Z={Z.shape})"
    )

    sp_end = datetime.now()
    print(f"End - {format_dt_ae(sp_end)}")
    print(f"Duration - {sp_end - sp_start}")
    print()

    return data_file



# -------------------- main loop --------------------


def main():
    ap = argparse.ArgumentParser(
        description="Multi-iteration PV self-play + training + eval loop for Linith (parallel-ready)."
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
        help="Self-play games per iteration (total, across all workers)",
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
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel self-play workers per iteration",
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
    base_stem = f"{name_prefix}{version_str}"  # 'epsilon_0.21'

    final_version = bump_minor(version_str)  # '0.22'
    final_model_name = f"{name_prefix}{final_version}{ext}"
    final_model_path = os.path.join(base_dir, final_model_name)

    eval_sims = args.eval_sims or args.sims

    start_time = datetime.now()
    print(f"Starting at {format_dt_ae(start_time)}")
    print(f"Start model - {start_model_abs}")
    print(
        f"Iterations - {args.iterations}, Games per iteration - {args.games}, "
        f"Epochs - {args.epochs}, Sims - {args.sims}, Workers - {args.workers}"
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
        candidate_name = f"{base_stem}{letter}{ext}"  # epsilon_0.21a.pt, etc.
        candidate_model = os.path.join(base_dir, candidate_name)

        print("\n============================================================")
        print(f"     Iteration {i}/{args.iterations}")
        print(f"     Using best model as base - {best_model}")
        print(f"     Candidate model this iteration - {candidate_model}")
        print("============================================================\n")

        # 1) parallel self-play to generate dataset (from current best model)
        try:
            data_file = run_selfplay_sharded(
                best_model=best_model,
                games=args.games,
                sims=args.sims,
                device=args.device,
                max_moves=args.max_moves,
                tag=args.tag,
                iteration=i,
                workers=args.workers,
            )
        except Exception as e:
            print(f"[error] Self-Play failed in iteration {i}: {e}")
            break

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
            "--device",
            args.device,
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
        print(f"====== Iteration {i} Complete ======")
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

    print("\n========== Loop Complete ==========")
    print(f"Started - {format_dt_ae(start_time)}")
    print(f"Finished - {format_dt_ae(end_time)}")
    print(f"Total duration - {end_time - start_time}")
    print(f"Best model - {best_model}")
    print(f"Final versioned model - {final_model_path}")
    print("===================================")
    print()


if __name__ == "__main__":
    main()
