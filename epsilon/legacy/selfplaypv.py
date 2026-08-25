import argparse
import random

import numpy as np
import torch
import time
from datetime import datetime
import traceback

# Atreyan Era timestamp helper: AE = Gregorian year - 2020
# Example: 2025-11-20 20:35:00 -> 5AE-11-20 20:35:00
# For years before 2020, AE will be negative (e.g., 2019 -> -1AE)
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"

from linithenv import LinithEnv, SUN, MOON
from pv_model import LinithPVNet
from linith_selfplay_cpp import pv_mcts_search_cpp
from action_space import ACTION_SIZE, encode_action

NUM_ACTIONS = ACTION_SIZE

# Use root-only policy only if sims == 0; otherwise always use MCTS
ROOT_ONLY_MOVES = 0

# AlphaZero-style: sample from MCTS policy for first N moves, then go greedy
TEMPERATURE_MOVES = 30


def make_cpp_eval_fn(net: LinithPVNet, device: torch.device):
    """
    Returns eval_fn(env) -> (policy_vector, value)
    for pv_mcts_search_cpp.
    """
    def eval_fn(env: LinithEnv):
        # env.state.to_tensor() is a pybind11-exposed method on C++ GameState
        obs = env.state.to_tensor()                 # (6,10,10) numpy from C++
        obs_np = np.array(obs, dtype=np.float32, copy=False)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)  # [1,6,10,10]

        with torch.no_grad():
            logits, value = net(obs_t)              # logits: [1, ACTION_SIZE]
            logits = logits[0]
            value = float(value.item())

        policy = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
        return policy, value

    return eval_fn

# ---------------------------------------------------------
# Replay buffer for self-play positions
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.states = []
        self.policies = []
        self.values = []
        self.pos = 0

    def __len__(self):
        return len(self.states)

    def push_episode(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        T = states.shape[0]
        for i in range(T):
            s = states[i]
            p = policies[i]
            v = float(values[i])

            if len(self.states) < self.capacity:
                self.states.append(s.astype(np.float32, copy=False))
                self.policies.append(p.astype(np.float32, copy=False))
                self.values.append(np.float32(v))
            else:
                idx = self.pos
                self.states[idx] = s.astype(np.float32, copy=False)
                self.policies[idx] = p.astype(np.float32, copy=False)
                self.values[idx] = np.float32(v)
                self.pos = (self.pos + 1) % self.capacity

    def build_dataset(self):
        if not self.states:
            return (
                np.empty((0, 6, 10, 10), dtype=np.float32),
                np.empty((0, ACTION_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        X = np.stack(self.states, axis=0).astype(np.float32)
        Pi = np.stack(self.policies, axis=0).astype(np.float32)
        Z = np.array(self.values, dtype=np.float32)
        return X, Pi, Z


# ---------------------------------------------------------
# Auto-tuning temperature schedule
# ---------------------------------------------------------
def auto_temperature(
    move_idx: int,
    legal_moves: int,
    max_moves: int,
    *,
    base_tau: float = 1.2,
    min_tau: float = 0.05,
    endgame_fraction: float = 0.5,
) -> float:
    """
    Adaptive temperature schedule for self-play.
    Early game → more exploration
    Late game → more deterministic
    Also adjusts slightly to branching factor.
    """

    # Phase = fraction of game progression
    horizon = max(1, int(max_moves * endgame_fraction))
    phase = min(1.0, max(0.0, move_idx / horizon))

    # Linear decay from base_tau → min_tau
    tau = base_tau - (base_tau - min_tau) * phase

    # Branching-factor adjustment
    if legal_moves >= 80:
        tau *= 1.2
    elif legal_moves <= 20:
        tau *= 0.7

    return max(min_tau, min(2.0, tau))

def generate_self_play(
    model_path: str,
    games: int,
    sims: int,          # used for MCTS in mid/late game
    device: str,
    max_moves: int,
    replay_capacity: int | None = None,
):
    dev = torch.device(device)
    net = LinithPVNet()
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.to(dev)
    net.eval()

    # MCTS engine for the MCTS phase
    mcts = PV_MCTS(net, device=dev)

    # Optional replay buffer over *positions*
    buffer = None
    if replay_capacity is not None and replay_capacity > 0:
        buffer = ReplayBuffer(replay_capacity)

    # Fallback legacy containers if no replay buffer is used
    legacy_states = []
    legacy_policies = []
    legacy_values = []

    wins_sun = wins_moon = draws = 0

    batch_start = datetime.now()
    print(f"Self-Play start - {format_dt_ae(batch_start)}")
    print()

    for g in range(games):
        game_index = g + 1
        print(f"Playing {game_index} of {games}")
        game_start = datetime.now()

        try:
            env = LinithEnv(max_moves=max_moves)
            obs = env.reset()

            states_this_game = []
            policies_this_game = []
            players_this_game = []

            move_stats = {"place_swan": 0, "place_stone": 0, "move_group": 0}

            move_idx = 0
            while not env.state.done:
                move_idx += 1
                s = env.state

                use_mcts = (move_idx > ROOT_ONLY_MOVES and sims > 0)

                # -------------------- ROOT-ONLY POLICY PHASE --------------------
                if not use_mcts:
                    t0 = time.time()
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs).unsqueeze(0).to(dev)  # [1,6,10,10]
                        policy_logits, value = net(obs_t)  # logits: [1,ACTION_SIZE]
                        policy_logits = policy_logits[0]   # [ACTION_SIZE]
                    t1 = time.time()

                    policy_probs = torch.softmax(policy_logits, dim=0).cpu().numpy()  # [ACTION_SIZE]

                    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    legal = env.legal_actions()
                    legal_indices = []

                    # Count legal actions per kind (debug / stats)
                    legal_kind_counts = {
                        "place_swan": 0,
                        "place_stone": 0,
                        "move_group": 0,
                        "other": 0,
                    }

                    for a in legal:
                        k = a[0]
                        if k in legal_kind_counts:
                            legal_kind_counts[k] += 1
                        else:
                            legal_kind_counts["other"] += 1

                    for a in legal:
                        try:
                            idx = encode_action(env, a)
                        except Exception as e:
                            print(f"[encode_error] {a} -> {e}")
                            continue
                        legal_indices.append((a, idx))

                    if not legal_indices:
                        # no moves; defensive
                        break

                    # Extract policy mass for legal moves
                    total_p = 0.0
                    for a, idx in legal_indices:
                        p = float(policy_probs[idx])
                        pi[idx] = p
                        total_p += p

                    if total_p <= 0.0:
                        # uniform over legal moves
                        for a, idx in legal_indices:
                            pi[idx] = 1.0 / len(legal_indices)
                    else:
                        pi /= total_p

                    # Policy mass per kind (debug)
                    mass_by_kind = {
                        "place_swan": 0.0,
                        "place_stone": 0.0,
                        "move_group": 0.0,
                        "other": 0.0,
                    }
                    for a, idx in legal_indices:
                        k = a[0]
                        if k in mass_by_kind:
                            mass_by_kind[k] += float(pi[idx])
                        else:
                            mass_by_kind["other"] += float(pi[idx])

                    # ----------------- Auto-tuned temperature -----------------
                    num_legal = len(legal_indices)
                    temperature = auto_temperature(
                        move_idx=move_idx,
                        legal_moves=num_legal,
                        max_moves=max_moves,
                        base_tau=1.2,
                        min_tau=0.05,
                        endgame_fraction=0.5,
                    )

                    if temperature == 1.0:
                        pi_temp = pi.copy()
                    else:
                        logits = np.log(pi + 1e-12) / temperature
                        pi_temp = np.exp(logits)
                        ssum = pi_temp.sum()
                        if ssum > 0:
                            pi_temp /= ssum
                        else:
                            pi_temp = pi.copy()
                    # ---------------------------------------------------------

                    chosen_idx = np.random.choice(NUM_ACTIONS, p=pi_temp)

                    chosen_action = None
                    for a, idx in legal_indices:
                        if idx == chosen_idx:
                            chosen_action = a
                            break
                    if chosen_action is None:
                        chosen_action = legal_indices[0][0]

                    mode = "root"

                # -------------------- MCTS PHASE --------------------
                else:
                    t0 = time.time()
                    # (Assumes PV_MCTS.search already supports root noise args if you wired that in)
                    visit_counts = mcts.search(
                        env,
                        num_simulations=sims,
                        add_root_noise=True,
                        dirichlet_alpha=0.3,
                        dirichlet_eps=0.25,
                    )
                    t1 = time.time()

                    legal = env.legal_actions()
                    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    legal_indices = []

                    for a in legal:
                        try:
                            idx = encode_action(env, a)
                        except Exception as e:
                            print(f"[encode_error] {a} -> {e}")
                            continue
                        legal_indices.append((a, idx))

                    total_visits = float(sum(visit_counts.values()))
                    if total_visits > 0.0:
                        for a, n in visit_counts.items():
                            try:
                                idx = encode_action(env, a)
                            except Exception as e:
                                print(f"[encode_error] {a} -> {e}")
                                continue
                            pi[idx] = n / total_visits
                    else:
                        # uniform fallback
                        if legal_indices:
                            p_val = 1.0 / len(legal_indices)
                            for a, idx in legal_indices:
                                pi[idx] = p_val

                    if not legal_indices:
                        break

                    # ----------------- Auto-tuned temperature over MCTS pi ----
                    num_legal = len(legal_indices)
                    tau = auto_temperature(
                        move_idx=move_idx,
                        legal_moves=num_legal,
                        max_moves=max_moves,
                        base_tau=1.0,
                        min_tau=1e-3,
                        endgame_fraction=0.4,
                    )

                    if tau == 1.0:
                        pi_temp = pi.copy()
                    else:
                        logits = np.log(pi + 1e-12) / tau
                        pi_temp = np.exp(logits)
                        ssum = pi_temp.sum()
                        if ssum > 0:
                            pi_temp /= ssum
                        else:
                            pi_temp = pi.copy()
                    # ---------------------------------------------------------

                    chosen_idx = np.random.choice(NUM_ACTIONS, p=pi_temp)

                    chosen_action = None
                    for a, idx in legal_indices:
                        if idx == chosen_idx:
                            chosen_action = a
                            break
                    if chosen_action is None:
                        chosen_action = legal_indices[0][0]

                    mode = f"mcts[{sims}]"

                # ----- record training data -----
                states_this_game.append(obs)
                policies_this_game.append(pi)
                players_this_game.append(env.state.current_player)

                # Track chosen action kind
                kind = chosen_action[0]
                move_stats[kind] = move_stats.get(kind, 0) + 1

                obs, reward, done, info = env.step(chosen_action)

                if done:
                    break

            # ---- End of game: assign outcomes Z ----
            winner = env.state.winner
            if winner is None:
                draws += 1
                z_sun = 0.0
            elif winner == SUN:
                wins_sun += 1
                z_sun = 1.0
            else:
                wins_moon += 1
                z_sun = -1.0

            # Per-position values for this game
            values_this_game = []
            for p in players_this_game:
                if winner is None:
                    z = 0.0
                elif p == SUN:
                    z = z_sun
                else:
                    z = -z_sun
                values_this_game.append(z)

            states_np = np.stack(states_this_game, axis=0)
            policies_np = np.stack(policies_this_game, axis=0)
            values_np = np.array(values_this_game, dtype=np.float32)

            if buffer is not None:
                buffer.push_episode(states_np, policies_np, values_np)
            else:
                legacy_states.append(states_np)
                legacy_policies.append(policies_np)
                legacy_values.extend(values_this_game)

            # Friendly winner message and timing
            if winner is None:
                winner_msg = "Draw"
            elif winner == SUN:
                winner_msg = "Sun won"
            else:
                winner_msg = "Moon won"

            game_end = datetime.now()
            game_duration = game_end - game_start
            print(
                f"Game {game_index}/{games} - {winner_msg} "
                f"(started {format_dt_ae(game_start)}, "
                f"ended {format_dt_ae(game_end)}, duration {game_duration})"
            )
            print(f"  move_stats = {move_stats}")

        except Exception as e:
            game_end = datetime.now()
            print(f"[error]Game {game_index}/{games} failed at {format_dt_ae(game_end)}: {e}")
            print(traceback.format_exc())
            continue

    # ----- Build dataset from either replay buffer or legacy lists -----
    if buffer is not None:
        X, Pi, Z = buffer.build_dataset()
        if X.shape[0] == 0:
            print("\n[selfplaypv] No self-play games completed successfully; no dataset generated.")
            return (
                np.empty((0, 6, 10, 10), dtype=np.float32),
                np.empty((0, NUM_ACTIONS), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
    else:
        if not legacy_states:
            print("\n[selfplaypv] No self-play games completed successfully; no dataset generated.")
            return (
                np.empty((0, 6, 10, 10), dtype=np.float32),
                np.empty((0, NUM_ACTIONS), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        X = np.concatenate(legacy_states, axis=0).astype(np.float32)
        Pi = np.concatenate(legacy_policies, axis=0).astype(np.float32)
        Z = np.array(legacy_values, dtype=np.float32)

    batch_end = datetime.now()
    print()
    print("==== Self-Play Summary ====")
    print(f"Games requested - {games}")
    print(f"Sun wins - {wins_sun}")
    print(f"Moon wins - {wins_moon}")
    print(f"Draws - {draws}")
    print()
    print(f"Dataset shapes X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")
    print(f"Self-Play end   {format_dt_ae(batch_end)}")
    print(f"Total duration  {batch_end - batch_start}")
    print()

    return X, Pi, Z


def main():
    ap = argparse.ArgumentParser(description="Self-play generator for Linith PV net.")
    ap.add_argument("--model", type=str, required=True, help="Path to PV net .pt")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--out", type=str, default="selfplay_pv_iter1.npz")
    args = ap.parse_args()

    X, Pi, Z = generate_self_play(
        model_path=args.model,
        games=args.games,
        sims=args.sims,
        device=args.device,
        max_moves=args.max_moves,
    )

    np.savez_compressed(args.out, X=X, Pi=Pi, Z=Z)
    print(f"[selfplaypv] saved dataset to {args.out}")


if __name__ == "__main__":
    main()
