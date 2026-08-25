import argparse
import random
import time
import traceback
from datetime import datetime

import numpy as np
import torch

# Atreyan Era timestamp helper
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"

# ---- C++ engine bindings ----
from linith_selfplay_cpp import LinithEnv, SUN, MOON, encode_action_cpp
from linith_selfplay_cpp import pv_mcts_search_cpp

# ---- PV net + MCTS (Python) ----
from pv_model import LinithPVNet
from action_space import ACTION_SIZE

NUM_ACTIONS = ACTION_SIZE
encode_action = encode_action_cpp


# ---------------------------------------------------------
# Helper: wrap a PV network into eval_fn for MCTS
# ---------------------------------------------------------
def make_cpp_eval_fn(net: LinithPVNet, device: torch.device):
    def eval_fn(env: LinithEnv):
        obs = env.state.to_tensor()  # (6,10,10) numpy from C++
        obs_np = np.array(obs, dtype=np.float32, copy=False)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)  # [1,6,10,10]

        with torch.no_grad():
            logits, value = net(obs_t)
            logits = logits[0]
            value = float(value.item())

        policy = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
        return policy, value

    return eval_fn


# ---------------------------------------------------------
# Replay buffer (same as yours)
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
# Auto-tuned temperature schedule (unchanged)
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
    horizon = max(1, int(max_moves * endgame_fraction))
    phase = min(1.0, max(0.0, move_idx / horizon))
    tau = base_tau - (base_tau - min_tau) * phase

    if legal_moves >= 80:
        tau *= 1.2
    elif legal_moves <= 20:
        tau *= 0.7

    return max(min_tau, min(2.0, tau))


# =========================================================
#                 TWO-MODEL SELF PLAY
# =========================================================
def generate_self_play_two_models(
    model_a: str,
    model_b: str,
    games: int,
    sims: int,
    device: str,
    max_moves: int,
    replay_capacity: int | None = None,
):
    dev = torch.device(device)

    # Load both models
    net_sun = LinithPVNet()
    net_sun.load_state_dict(torch.load(model_a, map_location=dev))
    net_sun.to(dev).eval()
    eval_sun = make_cpp_eval_fn(net_sun, dev)

    net_moon = LinithPVNet()
    net_moon.load_state_dict(torch.load(model_b, map_location=dev))
    net_moon.to(dev).eval()
    eval_moon = make_cpp_eval_fn(net_moon, dev)

    # Treat these as "model A" and "model B"
    net_a, eval_a = net_sun, eval_sun
    net_b, eval_b = net_moon, eval_moon

    # Replay buffer
    buffer = None
    if replay_capacity is not None and replay_capacity > 0:
        buffer = ReplayBuffer(replay_capacity)

    legacy_states = []
    legacy_policies = []
    legacy_values = []

    wins_a = wins_b = draws = 0

    batch_start = datetime.now()
    print(f"Two-model Self-Play start - {format_dt_ae(batch_start)}")
    print()

    for g in range(games):
        print(f"Playing {g+1}/{games}")
        game_start = datetime.now()

        # Alternate sides each game:
        # even-indexed games: A = Sun, B = Moon
        # odd-indexed games:  B = Sun, A = Moon
        if g % 2 == 0:
            sun_label, sun_file = "Model A", model_a
            moon_label, moon_file = "Model B", model_b
            sun_net, sun_eval = net_a, eval_a
            moon_net, moon_eval = net_b, eval_b
        else:
            sun_label, sun_file = "Model B", model_b
            moon_label, moon_file = "Model A", model_a
            sun_net, sun_eval = net_b, eval_b
            moon_net, moon_eval = net_a, eval_a

        try:
            env = LinithEnv(max_moves=max_moves)
            obs = env.reset()

            states_this_game = []
            policies_this_game = []
            players_this_game = []

            move_idx = 0

            while not env.state.done:
                move_idx += 1
                player = env.state.current_player

                if player == SUN:
                    eval_fn = sun_eval
                    net = sun_net
                else:
                    eval_fn = moon_eval
                    net = moon_net

                # Determine if using root-only or MCTS
                use_mcts = (move_idx > 0 and sims > 0)

                # ---------------- ROOT ONLY ----------------
                if not use_mcts:
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        logits, value = net(obs_t)
                        logits = logits[0]

                    raw_probs = torch.softmax(logits, dim=0).cpu().numpy()
                    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)

                    legal = env.legal_actions()
                    legal_indices = []
                    for a in legal:
                        try:
                            idx = encode_action(env, a)
                            legal_indices.append((a, idx))
                        except:
                            continue

                    if not legal_indices:
                        break

                    total = 0.0
                    for a, idx in legal_indices:
                        p = float(raw_probs[idx])
                        pi[idx] = p
                        total += p

                    if total <= 0.0:
                        for a, idx in legal_indices:
                            pi[idx] = 1.0 / len(legal_indices)
                    else:
                        pi /= total

                    tau = auto_temperature(
                        move_idx=move_idx,
                        legal_moves=len(legal_indices),
                        max_moves=max_moves,
                    )

                    if tau != 1.0:
                        logits = np.log(pi + 1e-12) / tau
                        pi_temp = np.exp(logits)
                        pi_temp /= pi_temp.sum()
                    else:
                        pi_temp = pi.copy()

                    chosen_idx = np.random.choice(NUM_ACTIONS, p=pi_temp)

                    chosen_action = None
                    for a, idx in legal_indices:
                        if idx == chosen_idx:
                            chosen_action = a
                            break
                    if chosen_action is None:
                        chosen_action = legal_indices[0][0]

                # ---------------- MCTS ----------------------
                else:
                    visit_counts = pv_mcts_search_cpp(
                        env,
                        eval_fn,
                        sims,
                        c_puct=1.5,
                        add_root_noise=True,
                        dirichlet_alpha=0.3,
                        dirichlet_eps=0.25,
                    )

                    legal = env.legal_actions()
                    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    legal_indices = []

                    for a in legal:
                        try:
                            idx = encode_action(env, a)
                            legal_indices.append((a, idx))
                        except:
                            continue

                    tot = float(sum(visit_counts.values()))
                    if tot > 0:
                        for a, n in visit_counts.items():
                            try:
                                idx = encode_action(env, a)
                                pi[idx] = n / tot
                            except:
                                pass
                    else:
                        if legal_indices:
                            p = 1.0 / len(legal_indices)
                            for a, idx in legal_indices:
                                pi[idx] = p

                    if not legal_indices:
                        break

                    tau = auto_temperature(
                        move_idx=move_idx,
                        legal_moves=len(legal_indices),
                        max_moves=max_moves,
                        base_tau=1.0,
                        min_tau=1e-3,
                        endgame_fraction=0.4,
                    )

                    if tau != 1.0:
                        logits = np.log(pi + 1e-12) / tau
                        pi_temp = np.exp(logits)
                        pi_temp /= pi_temp.sum()
                    else:
                        pi_temp = pi.copy()

                    chosen_idx = np.random.choice(NUM_ACTIONS, p=pi_temp)

                    chosen_action = None
                    for a, idx in legal_indices:
                        if idx == chosen_idx:
                            chosen_action = a
                            break
                    if chosen_action is None:
                        chosen_action = legal_indices[0][0]

                # ----- record training data -----
                states_this_game.append(obs)
                policies_this_game.append(pi)
                players_this_game.append(player)

                obs, reward, done, info = env.step(chosen_action)
                if done:
                    break

            # Outcome
            winner = env.state.winner
            if winner is None:
                draws += 1
                z_sun = 0.0
            elif winner == SUN:
                # Sun side wins this game
                if sun_label == "Model A":
                    wins_a += 1
                else:
                    wins_b += 1
                z_sun = 1.0
            else:
                # Moon side wins this game
                if moon_label == "Model A":
                    wins_a += 1
                else:
                    wins_b += 1
                z_sun = -1.0

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

            game_end = datetime.now()
            duration = game_end - game_start

            if winner is None:
                wmsg = "Draw"
            elif winner == SUN:
                wmsg = f"{sun_label} ({sun_file}) won as Sun"
            else:
                wmsg = f"{moon_label} ({moon_file}) won as Moon"

            print(f"Game {g + 1}/{games}, {wmsg}, duration {duration}")

        except Exception as e:
            print(f"[error] Game {g+1} failed {e}")
            print(traceback.format_exc())
            continue

    # Final dataset
    if buffer is not None:
        X, Pi, Z = buffer.build_dataset()
    else:
        if not legacy_states:
            return (
                np.empty((0,6,10,10), np.float32),
                np.empty((0,NUM_ACTIONS), np.float32),
                np.empty((0,), np.float32),
            )
        X = np.concatenate(legacy_states, axis=0).astype(np.float32)
        Pi = np.concatenate(legacy_policies, axis=0).astype(np.float32)
        Z = np.array(legacy_values, dtype=np.float32)

    batch_end = datetime.now()
    print()
    print("==== Two-Model Self-Play Summary ====")
    print(f"Model A wins - {wins_a}")
    print(f"Model B wins - {wins_b}")
    print(f"Draws - {draws}")
    print(f"Dataset shapes  X - {X.shape}  Pi - {Pi.shape}  Z -{Z.shape}")
    print(f"Ended {format_dt_ae(batch_end)}")
    print()

    return X, Pi, Z


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Two-model PV self-play for Linith.")
    ap.add_argument("--model-a", type=str, required=True)
    ap.add_argument("--model-b", type=str, required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-moves", type=int, default=10000)
    ap.add_argument("--out", type=str, default="selfplay_two_models.npz")
    ap.add_argument("--replay-capacity", type=int, default=0)
    args = ap.parse_args()

    X, Pi, Z = generate_self_play_two_models(
        model_a=args.model_a,
        model_b=args.model_b,
        games=args.games,
        sims=args.sims,
        device=args.device,
        max_moves=args.max_moves,
        replay_capacity=args.replay_capacity,
    )

    np.savez_compressed(args.out, X=X, Pi=Pi, Z=Z)
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()
