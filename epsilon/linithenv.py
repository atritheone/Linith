from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .linithrules import (
        BOARD_SIZE, EMPTY, SWAN_SUN, SWAN_MOON, FROZEN_SUN, FROZEN_MOON,
        STONE, SUN, MOON, count_active_swans, count_total_swans,
        legal_group_moves, legal_stone_placements, legal_swan_placements,
        legal_push_moves, simulate_group_move, simulate_push_move,
        compute_freezes_on, extra_actions_for_player, base_actions_per_turn, DIRS8,
    )
except ImportError:
    from linithrules import (
    BOARD_SIZE,
    EMPTY,
    SWAN_SUN,
    SWAN_MOON,
    FROZEN_SUN,
    FROZEN_MOON,
    STONE,
    SUN,
    MOON,
    count_active_swans,
    count_total_swans,
    legal_group_moves,
    legal_stone_placements,
    legal_swan_placements,
    legal_push_moves,
    simulate_group_move,
    simulate_push_move,
    compute_freezes_on,
    extra_actions_for_player,
    base_actions_per_turn,
    DIRS8,
    )


@dataclass
class GameState:
    board: np.ndarray
    current_player: int     # SUN or MOON
    actions_left: int       # actions remaining in current turn for current_player
    done: bool = False
    winner: Optional[int] = None   # SUN, MOON, or None for draw
    move_count: int = 0
    max_moves: int = 500

    def to_tensor(self):
        """
        Encode the complete decision state as an (8, H, W) float32 array.
        """
        return encode_game_state(self)


def encode_game_state(state: GameState) -> np.ndarray:
    """Canonical state encoding shared by GameState and LinithEnv."""
    b = state.board
    stone = (b == STONE).astype(np.float32)
    max_moves = max(1, int(state.max_moves))
    return np.stack(
        [
            (b == SWAN_SUN).astype(np.float32),
            (b == FROZEN_SUN).astype(np.float32),
            (b == SWAN_MOON).astype(np.float32),
            (b == FROZEN_MOON).astype(np.float32),
            stone,
            np.full_like(stone, 1.0 if state.current_player == SUN else 0.0),
            np.full_like(stone, float(state.actions_left)),
            np.full_like(stone, min(1.0, float(state.move_count) / max_moves)),
        ],
        axis=0,
    )


# Action encoding:
#   ("place_swan",  r, c)
#   ("place_stone", r, c)
#   ("move_group",  subset, (dr,dc))
#   ("push",        enemy_subset, (dr,dc))
Action = Tuple


class LinithEnv:
    """
    RL-style environment that mirrors the in-browser Linith rules.

    - One env.step() = one action (place swan / place stone / group move / push).
    - The same player may act multiple times in a row if they have actions_left > 0.
    - Base actions per turn come from base_actions_per_turn(board).
    - Extra actions are granted via extra_actions_for_player(...) after freezes.
    """

    def __init__(self, max_moves: int = 500):
        self.max_moves = max_moves
        self.state: Optional[GameState] = None

    # ---------- core API ----------

    def reset(self) -> np.ndarray:
        """
        Start the game in the post-setup position, matching the HTML rules:

          1) Sun places first Swan (anywhere).
          2) Moon places first Swan, not adjacent to Sun's Swan.
          3) Moon is the first to play in the main phase.

        The two initial placements are sampled randomly under the real constraints.
        """
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # 1) Sun initial Swan: choose a random tile
        import random
        sr = random.randrange(BOARD_SIZE)
        sc = random.randrange(BOARD_SIZE)
        board[sr, sc] = SWAN_SUN

        # 2) Moon initial Swan: choose any empty, non-adjacent to Sun’s Swan
        candidates: List[Tuple[int, int]] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != EMPTY:
                    continue
                # must NOT be 8-adjacent to Sun’s Swan
                adj = False
                for dr, dc in DIRS8:
                    nr, nc = sr + dr, sc + dc
                    if (nr, nc) == (r, c):
                        adj = True
                        break
                if not adj:
                    candidates.append((r, c))

        if not candidates:
            # Extremely unlikely on 10x10 from a single Swan, but guard anyway:
            # fallback: any empty cell.
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board[r, c] == EMPTY:
                        candidates.append((r, c))

        mr, mc = random.choice(candidates)
        board[mr, mc] = SWAN_MOON

        # 3) Moon takes the first real turn
        self.state = GameState(
            board=board,
            current_player=MOON,  # matches HTML: current = MOON after setup
            actions_left=base_actions_per_turn(board),
            done=False,
            winner=None,
            move_count=0,
            max_moves=self.max_moves,
        )
        return self.encode_state()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.done:
            raise RuntimeError("Game already finished; call reset().")

        s = self.state
        acting_player = s.current_player

        # ---- apply action to board ----
        self._apply_action(action)

        # ---- resolve encirclements (freezing, sealed swans) ----
        freeze_res = compute_freezes_on(self.state.board)
        self.state.board = freeze_res.board

        # ---- check terminal conditions (win/draw) ----
        self._check_terminal_conditions(freeze_res)

        reward = 0.0

        if self.state.done:
            # game ended immediately after this action
            if self.state.winner is None:
                reward = 0.0
            elif self.state.winner == acting_player:
                reward = 1.0
            else:
                reward = -1.0

        else:
            # still ongoing: handle action economy
            # (freeze bonus + base action cost, then possibly pass turn)

            # Extra actions from freezing enemy Swans this action
            bonus = extra_actions_for_player(acting_player, freeze_res)
            self.state.actions_left += bonus

            # Base cost of this action (always 1)
            self.state.actions_left -= 1

            # If no actions left, pass turn
            if self.state.actions_left <= 0:
                self.state.current_player = SUN if acting_player == MOON else MOON
                self.state.actions_left = base_actions_per_turn(self.state.board)

        self.state.move_count += 1
        if self.state.move_count >= self.max_moves and not self.state.done:
            self.state.done = True
            self.state.winner = None  # treat as draw

        obs = self.encode_state()
        info: Dict[str, Any] = {}
        return obs, reward, self.state.done, info

    def clone(self):
        """
        Deep copy the entire environment.
        Works for LinithEnv because all fields are pure Python objects.
        """
        import copy
        return copy.deepcopy(self)

    def legal_actions(self) -> List[Action]:
        if self.state is None or self.state.done:
            return []

        s = self.state
        board = s.board
        player = s.current_player

        actions: List[Action] = []

        # ----- Swan placements (if <6 total swans for this player) -----
        if count_total_swans(player, board) < 6:
            for r, c in legal_swan_placements(board, player):
                actions.append(("place_swan", r, c))

        # ----- Stone placements -----
        for r, c in legal_stone_placements(board):
            actions.append(("place_stone", r, c))

        # ----- Group moves (1–6 active Swans) -----
        for subset, direction in legal_group_moves(board, player, max_group_size=6):
            actions.append(("move_group", tuple(subset), direction))

        # ----- Push moves -----
        for subset, direction in legal_push_moves(board, player):
            actions.append(("push", tuple(subset), direction))

        return actions

    # ---------- helpers ----------

    def encode_state(self) -> np.ndarray:
        """
        Encode current state as (C, 10, 10) tensor.

        Channels:
          0: sun swan active
          1: sun swan frozen
          2: moon swan active
          3: moon swan frozen
          4: stone
          5: current_player (1 for SUN to move, 0 for MOON to move)
          6: actions remaining in the current turn
          7: move_count / max_moves
        """
        if self.state is None:
            raise RuntimeError("Env not reset.")

        return encode_game_state(self.state)

    def _apply_action(self, action: Action) -> None:
        if self.state is None:
            raise RuntimeError("Env not reset.")
        kind = action[0]

        if kind == "place_swan":
            _, r, c = action
            self._place_swan(int(r), int(c))
        elif kind == "place_stone":
            _, r, c = action
            self._place_stone(int(r), int(c))
        elif kind == "move_group":
            _, subset, direction = action
            self._move_group(subset, direction)
        elif kind == "push":
            _, subset, direction = action
            self._push(subset, direction)
        else:
            raise ValueError(f"Unknown action kind: {kind}")

    def _place_swan(self, r: int, c: int) -> None:
        assert self.state is not None
        player = self.state.current_player
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError("Swan placement is outside the board.")
        if count_total_swans(player, self.state.board) >= 6:
            raise ValueError("A player cannot have more than six Swans.")
        if (r, c) not in legal_swan_placements(self.state.board, player):
            raise ValueError("Illegal Swan placement.")
        if player == SUN:
            self.state.board[r, c] = SWAN_SUN
        else:
            self.state.board[r, c] = SWAN_MOON

    def _place_stone(self, r: int, c: int) -> None:
        assert self.state is not None
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError("Stone placement is outside the board.")
        if self.state.board[r, c] != EMPTY:
            raise ValueError("A Stone can only be placed on an empty tile.")
        self.state.board[r, c] = STONE

    def _move_group(
        self,
        subset: List[Tuple[int, int]],
        direction: Tuple[int, int],
    ) -> None:
        assert self.state is not None
        player = self.state.current_player
        nb = simulate_group_move(self.state.board, subset, direction, player)
        if nb is None:
            raise ValueError("Illegal group move passed to _move_group.")
        self.state.board = nb

    def _push(
        self,
        subset: List[Tuple[int, int]],
        direction: Tuple[int, int],
    ) -> None:
        assert self.state is not None
        player = self.state.current_player
        nb = simulate_push_move(self.state.board, player, list(subset), direction)
        if nb is None:
            raise ValueError("Illegal push move passed to _push.")
        self.state.board = nb

    def _has_any_legal(self, player: int) -> bool:
        """
        Check if `player` has any legal action in the current position, by
        temporarily treating it as their turn and reusing legal_actions().
        """
        assert self.state is not None
        s = self.state

        saved_player = s.current_player
        saved_actions_left = s.actions_left

        # Temporarily simulate "it's this player's turn"
        s.current_player = player
        s.actions_left = 1  # enough to allow legal_actions to run normally

        has = bool(self.legal_actions())

        # Restore original turn information
        s.current_player = saved_player
        s.actions_left = saved_actions_left

        return has

    def _check_terminal_conditions(self, freeze_res) -> None:
        """
        Mirrors the Linith end conditions:

          - Mutual final encirclement in the same action => draw.
          - Single-side final encirclement => that side loses.
          - Fallback: if a side has 0 active Swans and the other has >0, other wins.
          - No-move draw: both have active Swans but neither has any legal action.
        """
        assert self.state is not None
        if self.state.done:
            return

        b = self.state.board

        # 1) Mutual encirclement (both sealed this action) -> draw
        if freeze_res.sealed_sun > 0 and freeze_res.sealed_moon > 0:
            self.state.done = True
            self.state.winner = None
            return

        # 2) Single-side final encirclement
        if freeze_res.sealed_sun > 0:
            self.state.done = True
            self.state.winner = MOON
            return

        if freeze_res.sealed_moon > 0:
            self.state.done = True
            self.state.winner = SUN
            return

        # 3) Fallback active-swan checks (should rarely differ from sealed_*)
        a_sun = count_active_swans(SUN, b)
        a_moon = count_active_swans(MOON, b)

        if a_sun == 0 and a_moon > 0:
            self.state.done = True
            self.state.winner = MOON
            return

        if a_moon == 0 and a_sun > 0:
            self.state.done = True
            self.state.winner = SUN
            return

        # 4) No-move draw: both players still have active Swans
        #    but neither has any legal move or placement.
        if a_sun > 0 and a_moon > 0:
            if not self._has_any_legal(SUN) and not self._has_any_legal(MOON):
                self.state.done = True
                self.state.winner = None
                return
