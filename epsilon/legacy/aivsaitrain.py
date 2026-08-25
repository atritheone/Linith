# aivsaitrain.py

import argparse
import time
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from linithenv import LinithEnv, SUN, MOON
from pv_model import LinithPVNet
from linithai import choose_hard_move
from action_space import ACTION_SIZE, encode_action


# ---------------- Atreyan Era timestamp helper ----------------

def format_dt_ae(dt: datetime) -> str:
    """
    Convert datetime to Atreyan Era format.
    AE year = Gregorian year - 2020.
    Example: 2025-11-21 03:23:00 -> '5AE-11-21 03:23:00'
    """
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


# ---------------- Hard-AI self-play: teacher data ----------------

def play_hard_vs_hard_game(
    max_moves: int = 400,
    difficulty: str = "hard_train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """
    Play one full game: Hard AI vs Hard AI.

    Returns:
      states: np.ndarray (T, C, H, W)
      policies: np.ndarray (T, ACTION_SIZE), one-hot on the teacher's chosen move
      values: np.ndarray (T,), value in {+1, 0, -1} from POV of side-to-move
      winner: SUN / MOON / None
    """
    env = LinithEnv(max_moves=max_moves)
    obs = env.reset()  # (C, H, W)

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []

    while not env.state.done:
        s = env.state

        legal = env.legal_actions()

        # Teacher move
        try:
            action = choose_hard_move(env, difficulty=difficulty)
        except Exception as e:
            print(f"[error] choose_hard_move failed {e}")
            traceback.print_exc()
            break

        # Encode teacher move as one-hot policy over ACTION_SIZE
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)

        try:
            idx = encode_action(env, action)
            if 0 <= idx < ACTION_SIZE:
                pi[idx] = 1.0
            else:
                raise ValueError(f"encode_action returned out-of-range index {idx}")
        except Exception as e:
            # Fallback: uniform over legal actions we can encode
            print(f"[error] {action} -> {e} (using uniform legal fallback)")
            legal_indices: list[int] = []
            for a in legal:
                try:
                    li = encode_action(env, a)
                    if 0 <= li < ACTION_SIZE:
                        legal_indices.append(li)
                except Exception:
                    continue

            if not legal_indices:
                # No legal encodable moves; bail out of this game
                print("[warning] no encodable legal actions; aborting game")
                break

            val = 1.0 / len(legal_indices)
            for li in legal_indices:
                pi[li] = val

        states.append(obs)
        policies.append(pi)
        players.append(s.current_player)

        obs, reward, done, info = env.step(action)

    winner = env.state.winner

    # Assign values from POV of player-to-move at each recorded state
    if winner is None:
        z_sun = 0.0
    elif winner == SUN:
        z_sun = 1.0
    else:
        z_sun = -1.0

    values: list[float] = []
    for p in players:
        if winner is None:
            v = 0.0
        elif p == SUN:
            v = z_sun
        else:
            v = -z_sun
        values.append(v)

    if len(states) == 0:
        return (
            np.zeros((0, 6, 10, 10), dtype=np.float32),
            np.zeros((0, ACTION_SIZE), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            winner,
        )

    states_arr = np.stack(states, axis=0).astype(np.float32)
    policies_arr = np.stack(policies, axis=0).astype(np.float32)
    values_arr = np.array(values, dtype=np.float32)

    return states_arr, policies_arr, values_arr, winner


def generate_teacher_dataset(
    num_games: int,
    max_moves: int = 400,
    difficulty: str = "hard_train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset by letting Hard AI play against itself.
    Returns:
      X: (N, C, H, W)
      Pi: (N, ACTION_SIZE)
      Z: (N,)
    """
    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    wins_sun = wins_moon = draws = 0

    batch_start = datetime.now()
    print(f"Self-Play start - {format_dt_ae(batch_start)}")
    print()

    for g in range(num_games):
        game_idx = g + 1
        game_start = datetime.now()
        print(f"Playing game {game_idx}/{num_games} (start {format_dt_ae(game_start)})")

        try:
            states, policies, values, winner = play_hard_vs_hard_game(
                max_moves=max_moves,
                difficulty=difficulty,
            )

            if states.shape[0] == 0:
                print(f"[warning] game {game_idx} produced no positions; skipping")
                continue

            all_states.append(states)
            all_policies.append(policies)
            all_values.append(values)

            # Friendly winner message and timing
            if winner is None:
                winner_msg = "Draw"
            elif winner == SUN:
                winner_msg = "Sun won"
            else:
                winner_msg = "Moon won"

            if winner == SUN:
                wins_sun += 1
            elif winner == MOON:
                wins_moon += 1
            else:
                draws += 1

            game_end = datetime.now()
            print(f"Game {game_idx}/{num_games} - {winner_msg}, Positions - {states.shape[0]}, (end {format_dt_ae(game_end)}, duration {game_end - game_start})")

        except Exception as e:
            game_end = datetime.now()
            print(f"[error] Game {game_idx}/{num_games} failed at {format_dt_ae(game_end)} {e}")
            traceback.print_exc()
            continue

    if not all_states:
        raise RuntimeError("[error] no successful games; dataset is empty")

    X = np.concatenate(all_states, axis=0).astype(np.float32)
    Pi = np.concatenate(all_policies, axis=0).astype(np.float32)
    Z = np.concatenate(all_values, axis=0).astype(np.float32)

    batch_end = datetime.now()
    print()
    print("==== Hard-AI Teacher Self-Play Summary ====")
    print(f"Games requested - {num_games}")
    print(f"Sun wins       - {wins_sun}")
    print(f"Moon wins      - {wins_moon}")
    print(f"Draws          - {draws}")
    print()
    print(f"Dataset shapes X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")
    print(f"Self-play end   {format_dt_ae(batch_end)}")
    print(f"Total duration  {batch_end - batch_start}")
    print()

    return X, Pi, Z


# ---------------- Training PV net from teacher data ----------------

def train_pv_from_teacher(
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
    """
    1) Generate (X, Pi, Z) from Hard-AI vs Hard-AI self-play.
    2) Train LinithPVNet on that dataset (policy + value).
    3) Save model to out_path.
    """
    X, Pi, Z = generate_teacher_dataset(
        num_games=games,
        max_moves=max_moves,
        difficulty=difficulty,
    )

    X_t = torch.from_numpy(X)
    Pi_t = torch.from_numpy(Pi)
    Z_t = torch.from_numpy(Z).view(-1, 1)

    ds = TensorDataset(X_t, Pi_t, Z_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    dev = torch.device(device)
    net = LinithPVNet().to(dev)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.KLDivLoss(reduction="batchmean")  # log_softmax + prob targets

    print("Starting PV training from Hard-AI teacher data")
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
            logits, v = net(bx)  # logits: (B,ACTION_SIZE), v: (B,1)

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


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Linith PV net from Hard-AI vs Hard-AI self-play (teacher imitation)."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of Hard-AI vs Hard-AI games to generate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs over the collected dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="linith_pv_from_hard.pt",
        help="Output model path.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Max moves per game before forcing end.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard_train",
        choices=["easy", "medium", "hard", "hard_train"],
        help="Hard-AI difficulty used as teacher (recommended: hard_train or hard).",
    )
    parser.add_argument(
        "--value-coeff",
        type=float,
        default=1.0,
        help="Weight for value loss term.",
    )
    parser.add_argument(
        "--policy-coeff",
        type=float,
        default=1.0,
        help="Weight for policy loss term.",
    )

    args = parser.parse_args()

    train_pv_from_teacher(
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
