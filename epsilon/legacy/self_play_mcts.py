# self_play_mcts.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from linithenv import LinithEnv, SUN, MOON
from action_space import decode_action, legal_action_indices, ACTION_SIZE
from model import LinithNet
from mcts import MCTS


def mcts_self_play_game(
    net: LinithNet,
    num_simulations: int = 64,
    temperature_moves: int = 20,
    device: str = "cpu",
):
    """
    Play one game of Linith using MCTS-guided self-play.

    Returns:
      states:  list of np.ndarray (8,10,10)
      policies: list of np.ndarray (ACTION_SIZE,) visit distributions
      players: list of int (SUN or MOON) at each state
      winner:  SUN / MOON / None
    """
    env = LinithEnv()
    obs = env.reset()

    net.eval()
    mcts = MCTS(net, device=device)

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    players: List[int] = []

    move_index = 0

    while not env.state.done:  # type: ignore[union-attr]
        # Run tree search
        visit_counts = mcts.run(env, num_simulations=num_simulations)  # (ACTION_SIZE,)

        legal_idxs = legal_action_indices(env)
        if not legal_idxs:
            # no legal moves, force game end as draw
            env.state.done = True  # type: ignore[union-attr]
            env.state.winner = None  # type: ignore[union-attr]
            break

        # create a probability distribution from visit counts
        counts = visit_counts.astype(np.float32)
        # mask non-legal to 0
        mask = np.zeros_like(counts)
        mask[legal_idxs] = counts[legal_idxs]

        if move_index < temperature_moves:
            # soft selection: proportional to N^(1/tau), tau=1
            if mask.sum() <= 0:
                probs = np.ones_like(mask) / len(legal_idxs)
            else:
                probs = mask / mask.sum()
            action_idx = np.random.choice(len(probs), p=probs)
        else:
            # later in the game: greedy on visit count
            action_idx = int(np.argmax(mask))

        # record training data before applying move
        states.append(obs)
        policies.append(mask / max(1.0, mask.sum()))
        players.append(env.state.current_player)  # type: ignore[union-attr]

        # apply selected move
        action = decode_action(action_idx)
        obs, reward, done, info = env.step(action)
        move_index += 1

    winner = env.state.winner  # type: ignore[union-attr]
    return states, policies, players, winner


def generate_mcts_self_play_dataset(
    net: LinithNet,
    num_games: int,
    num_simulations: int = 64,
    device: str = "cpu",
):
    """
    Generate (state, policy, value) triples via MCTS self-play.

    Returns:
      X:  (N, 8, 10, 10)
      pi: (N, ACTION_SIZE)
      v:  (N,)
    """
    all_states: List[np.ndarray] = []
    all_policies: List[np.ndarray] = []
    all_values: List[float] = []

    for g in range(num_games):
        states, policies, players, winner = mcts_self_play_game(
            net,
            num_simulations=num_simulations,
            device=device,
        )

        # convert winner to per-state value
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
        all_policies.extend(policies)

        winner_label = 'None' if winner is None else ('Sun' if winner == SUN else 'Moon')
        print(f"[mcts self-play] game {g+1}/{num_games} finished, winner={winner_label}")

    X = np.stack(all_states, axis=0).astype(np.float32)
    pi = np.stack(all_policies, axis=0).astype(np.float32)
    v = np.array(all_values, dtype=np.float32)
    return X, pi, v


if __name__ == "__main__":
    device = "cpu"
    net = LinithNet().to(device)
    # Optionally load a previous checkpoint:
    # net.load_state_dict(torch.load("linith_value_net.pt", map_location=device))

    X, pi, v = generate_mcts_self_play_dataset(net, num_games=2, num_simulations=32, device=device)
    print("X:", X.shape, "pi:", pi.shape, "v:", v.shape)
