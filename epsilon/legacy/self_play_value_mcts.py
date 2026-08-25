# self_play_value_mcts.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from linithenv import LinithEnv, SUN, MOON
from model import LinithNet
from mcts_value import ValueMCTS


def play_one_game_value_mcts(
    net: LinithNet,
    num_simulations: int = 64,
    device: str = "cpu",
    max_moves: int = 500,
):
    """
    Play one full game of Linith using ValueMCTS.

    Returns:
      states:  list of np.ndarray (6,10,10)
      players: list of int (SUN or MOON) for each state
      winner:  SUN / MOON / None
    """
    env = LinithEnv(max_moves=max_moves)
    obs = env.reset()

    net.eval().to(device)
    mcts = ValueMCTS(net, device=device)

    states: List[np.ndarray] = []
    players: List[int] = []

    while not env.state.done:  # type: ignore[union-attr]
        # record state & side to move
        states.append(obs)
        players.append(env.state.current_player)  # type: ignore[union-attr]

        legal = env.legal_actions()
        if not legal:
            # no legal moves, treat as draw
            env.state.done = True  # type: ignore[union-attr]
            env.state.winner = None  # type: ignore[union-attr]
            break

        # MCTS to choose move (full rules incl. group moves)
        visit_counts = mcts.search(env, num_simulations=num_simulations)
        # pick most visited action
        best_action = max(visit_counts.items(), key=lambda kv: kv[1])[0]

        obs, reward, done, info = env.step(best_action)

    winner = env.state.winner  # type: ignore[union-attr]
    return states, players, winner


def generate_value_dataset_from_mcts(
    net: LinithNet,
    num_games: int,
    num_simulations: int = 64,
    device: str = "cpu",
    max_moves: int = 500,
):
    """
    Generate (state, value) pairs via self-play using ValueMCTS.

    Returns:
      X: (N, 6, 10, 10)
      y: (N,) in [-1, 0, 1], from player-to-move perspective.
    """
    all_states: List[np.ndarray] = []
    all_values: List[float] = []

    for g in range(num_games):
        states, players, winner = play_one_game_value_mcts(
            net,
            num_simulations=num_simulations,
            device=device,
            max_moves=max_moves,
        )

        # assign game result
        if winner is None:
            z_sun = 0.0
        elif winner == SUN:
            z_sun = 1.0
        else:  # winner == MOON
            z_sun = -1.0

        for p in players:
            if winner is None:
                v = 0.0
            elif p == SUN:
                v = z_sun
            else:
                v = -z_sun
            all_values.append(v)

        all_states.extend(states)

        winner_label = 'None' if winner is None else ('Sun' if winner == SUN else 'Moon')
        print(f"[value_mcts self-play] game {g+1}/{num_games} finished, winner={winner_label}")

    X = np.stack(all_states, axis=0).astype(np.float32)
    y = np.array(all_values, dtype=np.float32)
    return X, y


if __name__ == "__main__":
    device = "cpu"
    net = LinithNet()
    # Optionally load an existing checkpoint
    # net.load_state_dict(torch.load("linith_value_net.pt", map_location=device))

    X, y = generate_value_dataset_from_mcts(net, num_games=5, num_simulations=32, device=device)
    print("X:", X.shape, "y:", y.shape)
