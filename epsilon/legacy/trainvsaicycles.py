import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from linithenv import LinithEnv, SUN, MOON
from pv_model import LinithPVNet
from linithai import choose_hard_move
from mcts_value import ValueMCTS
from action_space import ACTION_SIZE, encode_action


NUM_ACTIONS = ACTION_SIZE


def pretty_opp_name(difficulty: str) -> str:
    """
    Convert 'hard_train' -> 'Hard Train', 'medium' -> 'Medium', etc.
    """
    name = " ".join(w.capitalize() for w in difficulty.replace("_", " ").split())
    return name or "Medium"


def format_atreyan_datetime(dt: datetime) -> str:
    """
    Format a datetime as '5AE-11-20 21:36:22' where:
      AE_year = year - 2020  (2021 -> 1AE, 2025 -> 5AE, etc.)
    """
    ae_year = dt.year - 2020
    return f"{ae_year}AE-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def generate_model_vs_ai(
    net: LinithPVNet,
    games: int,
    sims: int,
    device: torch.device,
    opponent_difficulty: str = "medium",
    max_moves: int = 400,
):
    """
    Play model-vs-ai games and return (X, Pi, Z) where:
      - X:  [num_positions, C, H, W]      (states where Epsilon moved)
      - Pi: [num_positions, ACTION_SIZE]  (MCTS-based policy targets)
      - Z:  [num_positions]               (value targets from POV of player-to-move)

    Model uses value-MCTS, AI uses linithai.
    """
    net.eval().to(device)
    mcts = ValueMCTS(net, device=device)

    all_states = []
    all_policies = []
    all_values = []

    wins_model = 0
    wins_ai = 0
    draws = 0

    opp_name = pretty_opp_name(opponent_difficulty)

    for g in range(games):
        env = LinithEnv(max_moves=max_moves)
        obs = env.reset()

        # alternate sides each game
        model_side = SUN if (g % 2 == 0) else MOON

        states_this_game = []
        policies_this_game = []
        players_this_game = []

        while not env.state.done:
            s = env.state

            if s.current_player == model_side:
                # ----- Epsilon move via MCTS -----
                visit_counts = mcts.search(env, num_simulations=sims)

                # Build policy target from visit counts
                pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                total_visits = 0.0
                legal_pairs = []

                for a, n in visit_counts.items():
                    try:
                        idx = encode_action(env, a)
                    except Exception:
                        continue
                    legal_pairs.append((a, idx, n))
                    pi[idx] += float(n)
                    total_visits += float(n)

                if total_visits > 0.0:
                    pi /= total_visits
                else:
                    # Fallback: uniform over legal actions if something went wrong with visit_counts
                    legal = env.legal_actions()
                    legal_indices = []
                    for a in legal:
                        try:
                            idx = encode_action(env, a)
                        except Exception:
                            continue
                        legal_indices.append(idx)
                    if not legal_indices:
                        # truly stuck; break defensively
                        break
                    for idx in legal_indices:
                        pi[idx] = 1.0 / len(legal_indices)

                # Choose action: argmax visit count
                if legal_pairs:
                    action = max(legal_pairs, key=lambda t: t[2])[0]
                else:
                    # fallback: pick any legal move if visit_counts was empty
                    legal = env.legal_actions()
                    if not legal:
                        break
                    action = legal[0]

                # Record training sample (state, policy, player)
                states_this_game.append(obs)
                policies_this_game.append(pi)
                players_this_game.append(s.current_player)

            else:
                # ----- AI opponent move -----
                action = choose_hard_move(env, difficulty=opponent_difficulty)

            obs, reward, done, info = env.step(action)

        winner = env.state.winner

        # Value targets: standard AlphaZero-style
        if winner is None:
            z_sun = 0.0
        elif winner == SUN:
            z_sun = 1.0
        else:
            z_sun = -1.0

        values_this_game = []
        for p in players_this_game:
            if winner is None:
                v = 0.0
            elif p == SUN:
                v = z_sun
            else:
                v = -z_sun
            values_this_game.append(v)

        if states_this_game:
            all_states.append(np.stack(states_this_game, axis=0))
            all_policies.append(np.stack(policies_this_game, axis=0))
            all_values.extend(values_this_game)

        # Logging + win counters
        if winner is None:
            draws += 1
            print(f"Game {g+1}/{games} - Draw (Epsilon was {'Sun' if model_side == SUN else 'Moon'})")
        elif winner == model_side:
            wins_model += 1
            print(f"Game {g+1}/{games} - Epsilon wins as {'Sun' if model_side == SUN else 'Moon'}")
        else:
            wins_ai += 1
            print(f"Game {g+1}/{games} - {opp_name} wins (Epsilon was {'Sun' if model_side == SUN else 'Moon'})")

    if not all_states:
        raise RuntimeError("No positions recorded from model moves")

    X = np.concatenate(all_states, axis=0).astype(np.float32)
    Pi = np.concatenate(all_policies, axis=0).astype(np.float32)
    Z = np.array(all_values, dtype=np.float32)

    print()
    print("==== Epsilon vs AI Summary ====")
    print(f"Games - {games}")
    print(f"Epsilon wins - {wins_model}")
    print(f"AI wins - {wins_ai}")
    print(f"Draws - {draws}")
    print(f"Dataset - X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")

    return X, Pi, Z, wins_model, wins_ai, draws


def fine_tune_on_dataset(
    base_net: LinithPVNet,
    X: np.ndarray,
    Pi: np.ndarray,
    Z: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    value_coeff: float = 1.0,
    policy_coeff: float = 1.0,
):
    """
    Fine-tune base_net on (X, Pi, Z) and return the updated net.

    Loss = policy_coeff * KL(policy_pred || Pi) + value_coeff * MSE(v, Z)
    """
    net = base_net.to(device)
    net.train()

    X_t = torch.from_numpy(X)
    Pi_t = torch.from_numpy(Pi)
    Z_t = torch.from_numpy(Z).view(-1, 1)

    ds = TensorDataset(X_t, Pi_t, Z_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.KLDivLoss(reduction="batchmean")  # log_softmax vs prob targets

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        n = 0

        for bx, bpi, bz in dl:
            bx = bx.to(device)
            bpi = bpi.to(device)
            bz = bz.to(device)

            opt.zero_grad()
            logits, v = net(bx)  # (policy_logits, value)

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

        avg_loss = total_loss / n
        avg_policy = total_policy / n
        avg_value = total_value / n

        print(
            f"Epoch {epoch}/{epochs} "
            f"Loss - {avg_loss:.6f} "
            f"Policy - {avg_policy:.6f} "
            f"Value - {avg_value:.6f}"
        )

    return net


def run_cycles(
    start_model: str,
    out_prefix: str,
    cycles: int,
    games_per_cycle: int,
    sims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device_str: str,
    opponent_difficulty: str,
    max_moves: int,
):
    device = torch.device(device_str)

    # load initial model
    net = LinithPVNet().to(device)
    net.load_state_dict(torch.load(start_model, map_location=device))
    print(f"Loaded start model from {start_model}")

    opp_name = pretty_opp_name(opponent_difficulty)

    for c in range(1, cycles + 1):
        print(f"\n========== CYCLE {c}/{cycles} ==========")
        cycle_start = datetime.now()
        print(f"Cycle {c} Start - {format_atreyan_datetime(cycle_start)}")
        print(f"Opponent Difficulty - {opp_name}")
        print(f"Sims per Move - {sims}")
        print()

        # 1) model-vs-ai games → dataset
        X, Pi, Z, wins_model, wins_ai, draws = generate_model_vs_ai(
            net,
            games=games_per_cycle,
            sims=sims,
            device=device,
            opponent_difficulty=opponent_difficulty,
            max_moves=max_moves,
        )

        print()

        # 2) fine-tune on those games (policy + value)
        net = fine_tune_on_dataset(
            net,
            X,
            Pi,
            Z,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

        print()

        # 3) save model for this cycle
        out_path = f"{out_prefix}_cycle{c}.pt"
        torch.save(net.state_dict(), out_path)
        cycle_end = datetime.now()
        elapsed = cycle_end - cycle_start
        total_seconds = int(elapsed.total_seconds())
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        print()
        print(
            f"Saved cycle {c} model to {out_path} "
            f"Epsilon wins - {wins_model}, AI wins - {wins_ai}, Draws - {draws}"
            f"End - {format_atreyan_datetime(cycle_end)}, Duration - {duration_str}"
        )

    print("\n========== ALL CYCLES COMPLETE ==========")
    print(f"Final model saved as - {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run repeated cycles of model-vs-ai self-play and training."
    )
    ap.add_argument(
        "--start-model",
        type=str,
        required=True,
        help="Path to starting model .pt (e.g. linith_pv_iter2.pt)",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="linith_vs_model",
        help="Prefix for output models (suffix _cycleN.pt will be added)",
    )
    ap.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of play+train cycles to run",
    )
    ap.add_argument(
        "--games-per-cycle",
        type=int,
        default=50,
        help="Number of model-vs-ai games per cycle",
    )
    ap.add_argument(
        "--sims",
        type=int,
        default=128,
        help="MCTS simulations per model move",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per cycle",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device string (e.g. cpu or cuda)",
    )
    ap.add_argument(
        "--opp",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard", "hard_train"],
        help="Opponent difficulty to use (easy, medium, hard, or hard_train)",
    )
    ap.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Max moves per game before forced end",
    )

    args = ap.parse_args()

    run_cycles(
        start_model=args.start_model,
        out_prefix=args.out_prefix,
        cycles=args.cycles,
        games_per_cycle=args.games_per_cycle,
        sims=args.sims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        opponent_difficulty=args.opp,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
