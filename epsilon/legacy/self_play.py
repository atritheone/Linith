from __future__ import annotations

from typing import List, Tuple

import numpy as np

from linithenv import LinithEnv, SUN, MOON

Action = Tuple  # same convention as in linith_env


def play_one_game(env: LinithEnv, max_steps: int = 500):
    """
    Plays one full random game of Linith in the RL environment.

    Returns:
      states:  list of np.ndarray, each (8, 10, 10)
      players: list of int (SUN or MOON) corresponding to each state
      winner:  SUN / MOON / None
    """
    states: List[np.ndarray] = []
    players: List[int] = []

    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        # record state & side to move
        states.append(obs)
        players.append(env.state.current_player)  # type: ignore[union-attr]

        actions = env.legal_actions()
        if not actions:
            # no legal moves – treat as draw for now
            env.state.done = True  # type: ignore[union-attr]
            env.state.winner = None  # type: ignore[union-attr]
            break

        # random policy (later replaced by MCTS/NN)
        import random
        action = random.choice(actions)

        obs, reward, done, info = env.step(action)
        steps += 1

    winner = env.state.winner  # type: ignore[union-attr]
    return states, players, winner


def generate_self_play_dataset(
    num_games: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset from random self-play.

    Returns:
      X: np.ndarray of shape (N, 8, 10, 10) – states
      y: np.ndarray of shape (N,) – value targets in [-1, 0, 1],
         from the perspective of the player to move in each state.
    """
    env = LinithEnv()

    all_states: List[np.ndarray] = []
    all_values: List[float] = []

    for g in range(num_games):
        states, players, winner = play_one_game(env)

        # convert game result into per-state values
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
                v = z_sun          # SUN's perspective
            else:
                v = -z_sun         # MOON's perspective
            all_values.append(v)

        all_states.extend(states)

        winner_label = 'None' if winner is None else ('Sun' if winner == SUN else 'Moon')
        print(f"[self-play] game {g+1}/{num_games} finished, winner={winner_label}")

    X = np.stack(all_states, axis=0).astype(np.float32)      # (N, 8, 10, 10)
    y = np.array(all_values, dtype=np.float32)               # (N,)
    return X, y


if __name__ == "__main__":
    X, y = generate_self_play_dataset(num_games=5)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
