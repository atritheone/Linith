# epsilonvsai_cpp.py
#
# Evaluate a PV model vs C++ Hard AI, with HTML-compatible replay export.

import argparse
import json
import os
import re
from datetime import datetime, UTC

import numpy as np
import torch

from pv_model import LinithPVNet
from action_space import ACTION_SIZE  # only need the constant size

# C++ engine bindings (from your linith_selfplay_cpp module)
from linith_selfplay_cpp import (
    LinithEnv,
    SUN,
    MOON,
    choose_hard_move_cpp,
    encode_action_cpp,
)

# Preserve original global
opponent_name: str = "Medium"


# ---------------------------------------------------
# Root-policy model move selection (PV-only; no MCTS)
# ---------------------------------------------------
def choose_model_move_pv(env: LinithEnv, obs: np.ndarray, net: LinithPVNet, device: torch.device):
    """
    Root move selection using the PV net:
      • forward pass through PV net
      • mask to legal moves
      • pick argmax probability

    env: C++ LinithEnv
    obs: latest encoded observation (C, H, W) from env.reset()/env.step()
    """
    obs = obs.astype(np.float32, copy=False)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = net(obs_t)  # logits: [1, ACTION_SIZE]
        policy_logits = policy_logits[0]   # -> [ACTION_SIZE]

    policy_probs = torch.softmax(policy_logits, dim=0).cpu().numpy()

    pi = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal = env.legal_actions()
    legal_indices = []

    # Build list of (action, index) pairs using the C++ encoder
    for a in legal:
        try:
            idx = encode_action_cpp(env, a)
        except Exception:
            continue
        legal_indices.append((a, idx))

    if not legal_indices:
        return None

    total_p = 0.0
    for a, idx in legal_indices:
        p = float(policy_probs[idx])
        pi[idx] = p
        total_p += p

    if total_p <= 0.0:
        # fall back to uniform over legal
        val = 1.0 / len(legal_indices)
        for a, idx in legal_indices:
            pi[idx] = val
    else:
        pi /= total_p

    # pick argmax index
    best_idx = int(np.argmax(pi))

    # recover corresponding action
    for a, idx in legal_indices:
        if idx == best_idx:
            return a

    # defensive fallback
    return legal_indices[0][0]


# ---------------------------------------------------
# Snapshot compatible with Linith HTML history/replay
# ---------------------------------------------------
def snapshot_for_html(env: LinithEnv, move_number: int, is_move: bool):
    """
    Build a single snapshot in the same shape that the HTML client uses
    via cloneState(meta) + pushHistory(meta).

    Only the fields that the browser actually consumes for import/replay
    are populated:

      - board:  10x10 grid of ints
      - turn:   'setup' | 'play' (approximate)
      - toPlace: 'sun' | 'moon' | 'stone' (approximate / best-effort)
      - current: 1 (Sun) or 2 (Moon)
      - movesLeft: small int (we default to 1 if not exposed)
      - tag:   'initial' for the very first non-move snapshot, else 'move'
      - isMove: True for real moves; False for the initial snapshot
      - moveNumber: sequential move index for real moves
    """
    s = env.state  # C++ GameState-like object

    # --- board ---
    board = None
    if hasattr(s, "board"):
        b = s.board
    elif hasattr(env, "board"):
        b = env.board
    else:
        b = None

    if b is None:
        # safety fallback: empty 10x10
        board = [[0 for _ in range(10)] for _ in range(10)]
    else:
        # b may be a numpy array or a list-of-lists; both are iterable
        if isinstance(b, np.ndarray):
            board = b.tolist()
        else:
            board = [list(row) for row in b]

    # --- turn / phase ---
    # We don't track exact setup vs play here; approximate:
    if hasattr(s, "turn"):
        turn = s.turn
    else:
        if not is_move and move_number == 0:
            turn = "setup"
        else:
            turn = "play"

    if isinstance(turn, bytes):
        turn = turn.decode("utf-8", "ignore")
    if not isinstance(turn, str) or not turn:
        if not is_move and move_number == 0:
            turn = "setup"
        else:
            turn = "play"

    # --- current player ---
    if hasattr(s, "current_player"):
        current_player = s.current_player
    elif hasattr(s, "current"):
        current_player = s.current
    else:
        current_player = SUN

    # --- toPlace (approximate, but good enough for replay UI) ---
    to_place = getattr(s, "to_place", None)
    if isinstance(to_place, bytes):
        to_place = to_place.decode("utf-8", "ignore")

    if to_place is None or (isinstance(to_place, str) and not to_place):
        if turn == "setup":
            to_place = "sun" if current_player == SUN else "moon"
        else:
            to_place = "stone"
    elif isinstance(to_place, int):
        if to_place == SUN:
            to_place = "sun"
        elif to_place == MOON:
            to_place = "moon"
        else:
            to_place = "stone"

    # --- movesLeft (used mainly for UI; 1 is a safe default) ---
    moves_left = getattr(s, "actions_left", None)
    if moves_left is None:
        moves_left = getattr(s, "movesLeft", None)
    if not isinstance(moves_left, int):
        moves_left = 1

    snap = {
        "board": board,
        "turn": turn,
        "toPlace": to_place,
        "current": int(current_player),
        "movesLeft": int(moves_left),
        "tag": "initial" if (not is_move and move_number == 0) else "move",
    }

    snap["isMove"] = bool(is_move)
    if is_move:
        snap["moveNumber"] = int(move_number)

    return snap


# ---------------------------------------------------
#  play_model_vs_ai_game (C++ env + C++ hard AI)
# ---------------------------------------------------
def play_model_vs_ai_game(
    net: LinithPVNet,
    sims: int,
    device: str,
    opponent_difficulty: str = "medium",
    max_moves: int = 400,
    model_as: int = SUN,
    record_frames: bool = False,
):
    """
    Play one game: model (with PV-root policy) vs C++ Hard AI.

    If record_frames=True, returns (winner, frames) where frames is a list of
    snapshot dicts compatible with the Linith HTML replay format.
    Otherwise returns just winner.
    """
    dev = torch.device(device)
    net.eval().to(dev)

    env = LinithEnv(max_moves=max_moves)
    obs = env.reset()  # (C, H, W)

    frames = [] if record_frames else None
    move_number = 0

    # initial snapshot (before any moves)
    if record_frames:
        frames.append(snapshot_for_html(env, move_number=0, is_move=False))

    # main loop
    while not env.state.done:
        s = env.state

        if s.current_player == model_as:
            action = choose_model_move_pv(env, obs, net, dev)
        else:
            action = choose_hard_move_cpp(env, difficulty=opponent_difficulty)

        if action is None:
            break  # defensive fallback

        obs, reward, done, info = env.step(action)

        if record_frames:
            move_number += 1
            frames.append(snapshot_for_html(env, move_number=move_number, is_move=True))

    winner = env.state.winner

    if record_frames:
        return winner, frames
    else:
        return winner


# ---------------------------------------------------
# Helpers for HTML-compatible Linith save files
# ---------------------------------------------------
def _slug_for_filename(s: str) -> str:
    """
    Match the JS slug() used in linith_0.229.html for AI style/difficulty.
    """
    s = str(s or "").lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9_-]", "", s)
    s = re.sub(r"-+", "-", s)

    # strip leading/trailing '-' but keep '_'
    if s.startswith("-"):
        s = s[1:]
    if s.endswith("-"):
        s = s[:-1]
    return s


def make_linith_filename(
    ai_style: str,
    ai_difficulty: str,
    clock_mode: str = "off",
    ts: datetime | None = None,
) -> str:
    """
    Reproduce the browser's filename generation exactly:

      linithgame_{style}_{difficulty}[_{clockSuffix}]_{AEyear}AE{MM}{DD}_{HH}{mm}{SS}.json
    """
    ts = ts or datetime.now(UTC)

    style_val = _slug_for_filename(ai_style)
    diff_val = _slug_for_filename(ai_difficulty) or "unknown"
    mode_part = f"{style_val}_{diff_val}" if (style_val or diff_val) else "local"

    # clock suffix
    clock_suffix = ""
    if clock_mode and clock_mode != "off":
        if clock_mode == "stopwatch":
            clock_suffix = "_stopwatch"
        elif str(clock_mode).startswith("chess-"):
            parts = str(clock_mode).split("-")
            mins = parts[1] if len(parts) > 1 else ""
            clock_suffix = f"_{mins}minutes" if mins else ""
        else:
            clock_suffix = f"_{_slug_for_filename(clock_mode)}"

    # Atreyan Era timestamp: 2020 => 0AE
    greg_year = ts.year
    ae_year = greg_year - 2020

    MM = f"{ts.month:02d}"
    DD = f"{ts.day:02d}"
    HH = f"{ts.hour:02d}"
    mm = f"{ts.minute:02d}"
    SS = f"{ts.second:02d}"
    stamp = f"{ae_year}AE{MM}{DD}_{HH}{mm}{SS}"

    return f"linithgame_{mode_part}{clock_suffix}_{stamp}.json"


def _parse_snapshot(snap):
    """
    Match the JS parseSnap() in saveGameRecord():
    - if string: JSON.parse(...)
    - if object: pass through
    """
    if snap is None:
        return None
    if isinstance(snap, str):
        try:
            return json.loads(snap)
        except Exception:
            return None
    if isinstance(snap, dict):
        return snap
    return None


def build_linith_payload(
    frames,
    ai_style: str,
    ai_difficulty: str,
    winner_code,
    *,
    board_size: int = 10,
    clock_mode: str = "off",
    version: str = "epsilon-cli",
) -> dict:
    """
    Build a payload bit-for-bit compatible with the HTML client.

    frames:
      list of snapshots; can be dicts or JSON strings, just like the browser uses.
      The first frame must be the initial empty board snap used for 'New Game.'
    winner_code:
      C++ env's winner (SUN, MOON, or None)
    """
    if not frames:
        raise ValueError("build_linith_payload() requires at least one frame")

    first_state = _parse_snapshot(frames[0])
    last_state = _parse_snapshot(frames[-1])

    # Normalise frames to JSON strings, same as normaliseFramesToStrings()
    norm_frames: list[str] = []
    for f in frames:
        if isinstance(f, str):
            norm_frames.append(f)
        else:
            try:
                norm_frames.append(json.dumps(f, separators=(",", ":")))
            except Exception:
                continue

    # Map winner to HTML-style outcome text.
    if winner_code == SUN:
        outcome_short = "Sun wins."
        outcome_detailed = "Sun ☼ wins by encircling the final Swan."
    elif winner_code == MOON:
        outcome_short = "Moon wins."
        outcome_detailed = "Moon ☾ wins by encircling the final Swan."
    elif winner_code in (0, None, "draw", "Draw"):
        outcome_short = "Draw."
        outcome_detailed = "Draw."
    else:
        outcome_short = "Game over."
        outcome_detailed = "Game over."

    meta = {
        "kind": "linith-game",
        "version": version,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "boardSize": int(board_size),
        "ai": {
            "difficulty": ai_difficulty,
            "style": ai_style,
        },
        "clock": clock_mode,
    }

    payload = {
        "meta": meta,
        "newGame": {
            "log": "New Game.",
            "state": first_state,
        },
        "replay": norm_frames,
        "outcome": {
            "short": outcome_short,
            "detailed": outcome_detailed,
            "state": last_state,
        },
    }
    return payload


def save_linith_game(
    frames,
    winner_code,
    *,
    ai_style: str,
    ai_difficulty: str,
    save_dir: str,
    board_size: int = 10,
    clock_mode: str = "off",
    version: str = "epsilon-cli",
) -> str:
    """
    High-level helper:
    - builds the HTML-compatible payload
    - generates the correct AE filename
    - writes the .json into save_dir
    Returns the absolute path to the written file.
    """
    os.makedirs(save_dir, exist_ok=True)

    payload = build_linith_payload(
        frames=frames,
        ai_style=ai_style,
        ai_difficulty=ai_difficulty,
        winner_code=winner_code,
        board_size=board_size,
        clock_mode=clock_mode,
        version=version,
    )

    fname = make_linith_filename(
        ai_style=ai_style,
        ai_difficulty=ai_difficulty,
        clock_mode=clock_mode,
    )
    out_path = os.path.join(save_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return out_path


# ---------------------------------------------------
# eval_model_vs_ai
# ---------------------------------------------------
def eval_model_vs_ai(
    model_path: str,
    games: int,
    sims: int,
    device: str,
    opponent_difficulty: str,
    max_moves: int,
    save_dir: str | None = None,
):
    dev = torch.device(device)

    # Set opponent_name from difficulty flag (title-cased, underscores -> spaces)
    global opponent_name
    opponent_name = (
        " ".join(
            w.capitalize() for w in opponent_difficulty.replace("_", " ").split()
        )
        or opponent_name
    )

    net = LinithPVNet().to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()

    wins_model = 0
    wins_ai = 0
    draws = 0

    print(f"Evaluating model {model_path} vs {opponent_name}")
    print()

    for g in range(games):
        game_index = g + 1
        model_side = SUN if (g % 2 == 0) else MOON
        print(f"Playing {game_index} of {games}")

        if save_dir:
            winner, frames = play_model_vs_ai_game(
                net,
                sims=sims,
                device=device,
                opponent_difficulty=opponent_difficulty,
                max_moves=max_moves,
                model_as=model_side,
                record_frames=True,
            )
        else:
            frames = None
            winner = play_model_vs_ai_game(
                net,
                sims=sims,
                device=device,
                opponent_difficulty=opponent_difficulty,
                max_moves=max_moves,
                model_as=model_side,
                record_frames=False,
            )

        if winner is None:
            draws += 1
            print(
                f"Game {g+1}/{games} - Draw (Epsilon was {'Sun' if model_side==SUN else 'Moon'})"
            )
        elif winner == model_side:
            wins_model += 1
            print(
                f"Game {g+1}/{games} - Epsilon wins as {'Sun' if model_side==SUN else 'Moon'}"
            )
        else:
            wins_ai += 1
            print(
                f"Game {g+1}/{games} - {opponent_name} wins (Epsilon was {'Sun' if model_side==SUN else 'Moon'})"
            )

        # If saving, write a Linith HTML-compatible replay JSON for this game
        if save_dir and frames:
            path = save_linith_game(
                frames=frames,
                winner_code=winner,
                ai_style=opponent_name,              # current hard AI style
                ai_difficulty=opponent_difficulty,
                save_dir=save_dir,
                board_size=10,
                clock_mode="off",
                version="0.229",
            )
            print(f"  -> saved replay to {path}")

    print()
    print("========== SUMMARY ==========")
    print(f"Games - {games}")
    print(f"Epsilon wins - {wins_model}")
    print(f"{opponent_name} wins - {wins_ai}")
    print(f"Draws - {draws}")
    print()


# ---------------------------------------------------
# main CLI
# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Linith model vs C++ Hard AI.")
    ap.add_argument(
        "--model",
        type=str,
        default="linith_pv_from_hard_cpp.pt",
        help="Path to model .pt file",
    )
    ap.add_argument(
        "--games", type=int, default=20, help="Number of evaluation games"
    )
    ap.add_argument(
        "--sims",
        type=int,
        default=0,
        help="(kept for compatibility; unused in PV-only mode)",
    )
    ap.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device (cpu or cuda)"
    )
    ap.add_argument(
        "--opp",
        type=str,
        default="hard",
        help="Opponent difficulty: easy, medium, hard, or hard_train",
    )
    ap.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Max moves per game before forced end",
    )
    ap.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to write Linith HTML replay JSON files",
    )
    args = ap.parse_args()

    eval_model_vs_ai(
        model_path=args.model,
        games=args.games,
        sims=args.sims,
        device=args.device,
        opponent_difficulty=args.opp,
        max_moves=args.max_moves,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
