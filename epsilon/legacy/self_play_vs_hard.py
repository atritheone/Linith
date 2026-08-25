# self_play_vs_hard.py

import argparse
import numpy as np
import torch

from linithenv import LinithEnv, SUN, MOON
from model import LinithNet
from linithai import choose_hard_move
from mcts_value import ValueMCTS


def generate_model_vs_hard(
    model_path: str,
    games: int,
    sims: int,
    device: str,
    opponent_difficulty: str = "hard",
    max_moves: int = 400,
):
    dev = torch.device(device)
    net = LinithNet().to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()

    mcts = ValueMCTS(net, device=dev)

    all_states = []
    all_values = []

    wins_model = wins_hard = draws = 0

    for g in range(games):
        env = LinithEnv(max_moves=max_moves)
        obs = env.reset()

        # alternate sides so we don't overfit to Sun or Moon
        model_side = SUN if (g % 2 == 0) else MOON

        states_this_game = []
        players_this_game = []

        while not env.state.done:
            s = env.state
            if s.current_player == model_side:
                # model move via MCTS
                visit_counts = mcts.search(env, num_simulations=sims)
                action = max(visit_counts.items(), key=lambda kv: kv[1])[0]

                # record the position from model's POV
                states_this_game.append(obs)
                players_this_game.append(s.current_player)
            else:
                # Hard move
                action = choose_hard_move(env, difficulty=opponent_difficulty)

            obs, reward, done, info = env.step(action)

        winner = env.state.winner
        if winner is None:
            draws += 1
        elif winner == model_side:
            wins_model += 1
        else:
            wins_hard += 1

        # assign value targets for model's positions
        if winner is None:
            z_sun = 0.0
        elif winner == SUN:
            z_sun = 1.0
        else:
            z_sun = -1.0

        for p in players_this_game:
            if winner is None:
                v = 0.0
            elif p == SUN:
                v = z_sun
            else:
                v = -z_sun
            all_values.append(v)

        if states_this_game:
            all_states.append(np.stack(states_this_game, axis=0))

        winner_label = 'None' if winner is None else ('Sun' if winner == SUN else 'Moon')
        print(f"[mvsh] game {g+1}/{games} winner={winner_label} (model was {'Sun' if model_side==SUN else 'Moon'})")

    if not all_states:
        raise RuntimeError("No positions recorded from model moves")

    X = np.concatenate(all_states, axis=0).astype(np.float32)
    y = np.array(all_values, dtype=np.float32)

    print("==== model vs hard summary ====")
    print(f"Games: {games}")
    print(f"Model wins: {wins_model}")
    print(f"Hard wins : {wins_hard}")
    print(f"Draws     : {draws}")
    print(f"Dataset   : X={X.shape}, y={y.shape}")

    return X, y


def main():
    ap = argparse.ArgumentParser(description="Generate model-vs-Hard dataset.")
    ap.add_argument("--model", type=str, required=True, help="Path to existing model .pt")
    ap.add_argument("--games", type=int, default=50, help="Number of games vs Hard")
    ap.add_argument("--sims", type=int, default=128, help="MCTS simulations per model move")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--opp", type=str, default="hard", help="hard or hard_train")
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--out", type=str, default="mvsh_data.npz")
    args = ap.parse_args()

    X, y = generate_model_vs_hard(
        model_path=args.model,
        games=args.games,
        sims=args.sims,
        device=args.device,
        opponent_difficulty=args.opp,
        max_moves=args.max_moves,
    )

    np.savez_compressed(args.out, X=X, y=y)
    print(f"[mvsh] saved dataset to {args.out}")


if __name__ == "__main__":
    main()
