# action_space.py
from __future__ import annotations

from typing import List, Tuple, Iterable

from linithrules import BOARD_SIZE, DIRS8, SWAN_SUN, SWAN_MOON

# We fix the action space as:
#
#   0–99    : place_swan  at (r,c)
#   100–199 : place_stone at (r,c)
#   200–703 : move_group for any non-empty subset of up to 6 active Swans
#             of the current player, in any of 8 directions.
#   704–751 : push with one of up to 6 active Swans in any of 8 directions
#
# That’s:
#   100 placements for swans
#   100 placements for stones
#   63 subsets (1..6 swans) * 8 directions = 504 group moves
#   6 possible swans * 8 directions        = 48 pushes
#
# Total: 100 + 100 + 504 + 48 = 752 actions.

ACTION_SIZE = 752

Action = Tuple  # same convention as in linith_env

MAX_SWANS = 6  # per player, including frozen (rules), but only active matter here

# All non-empty bitmasks on up to 6 swans (bits 0..5)
SUBSET_MASKS: List[int] = [mask for mask in range(1, 1 << MAX_SWANS)]  # 1..63
MASK_TO_INDEX = {mask: i for i, mask in enumerate(SUBSET_MASKS)}       # mask -> 0..62

# Ranges / offsets
PLACE_SWAN_START   = 0
PLACE_SWAN_END     = 100          # exclusive
PLACE_STONE_START  = 100
PLACE_STONE_END    = 200          # exclusive
MOVE_GROUP_START   = 200
MOVE_GROUP_END     = 704          # exclusive
PUSH_START         = 704
PUSH_END           = 752          # exclusive

PUSH_ACTIONS_PER_SWAN = len(DIRS8)   # 8
MAX_PUSH_SWANS        = MAX_SWANS    # up to 6 swans considered for push encoding


def _square_index(r: int, c: int) -> int:
    """Map (r,c) to a 0..99 index on the 10×10 board."""
    return int(r) * BOARD_SIZE + int(c)


def _index_to_square(idx: int) -> Tuple[int, int]:
    """Inverse of _square_index for indices 0..99."""
    r = idx // BOARD_SIZE
    c = idx % BOARD_SIZE
    return int(r), int(c)


def _active_swans_for_current_player(env) -> List[Tuple[int, int]]:
    """
    Return a sorted list of (r,c) for *active* swans of the player to move.

    Assumptions:
      - env.state.current_player is +1 for Sun, -1 for Moon (matches linithrules.SUN/MOON).
      - Board uses SWAN_SUN / SWAN_MOON for active swans.
      - Frozen swans are ignored here (they are inert for movement).
    """
    s = env.state
    board = s.board
    player = s.current_player

    if player == 1:      # Sun
        target = SWAN_SUN
    elif player == -1:   # Moon
        target = SWAN_MOON
    else:
        raise ValueError(f"Unexpected current_player value: {player}")

    coords: List[Tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == target:
                coords.append((int(r), int(c)))

    coords.sort()
    if len(coords) > MAX_SWANS:
        # By rules this shouldn't happen; if it does, cap to the first 6 for encoding.
        coords = coords[:MAX_SWANS]
    return coords


def _subset_mask_from_coords(
    subset: Iterable[Tuple[int, int]],
    swans: List[Tuple[int, int]],
) -> int:
    """
    Given a subset of coordinates (r,c) and the sorted list of this player's
    active swans, build a bitmask where bit i corresponds to swans[i].
    """
    coord_to_index = {coord: i for i, coord in enumerate(swans)}
    mask = 0
    for (r, c) in subset:
        key = (int(r), int(c))
        if key not in coord_to_index:
            raise ValueError(f"Swan at {key} not found among active swans {swans}")
        i = coord_to_index[key]
        if i >= MAX_SWANS:
            raise ValueError(f"Swan index {i} exceeds MAX_SWANS={MAX_SWANS}")
        mask |= 1 << i

    if mask == 0:
        raise ValueError("Empty subset for move_group.")
    return mask


def _find_swan_index(swans: List[Tuple[int, int]], coord: Tuple[int, int]) -> int:
    """
    Find the index of coord in swans (sorted list of active swans).
    Raises if not found or beyond MAX_PUSH_SWANS.
    """
    try:
        i = swans.index((int(coord[0]), int(coord[1])))
    except ValueError:
        raise ValueError(f"Swan at {coord} not found among active swans {swans}")
    if i >= MAX_PUSH_SWANS:
        raise ValueError(
            f"Swan index {i} exceeds MAX_PUSH_SWANS={MAX_PUSH_SWANS} for pushes"
        )
    return i


def decode_action(idx: int, env=None) -> Action:
    """
    Convert an integer index in [0, ACTION_SIZE) to an environment Action tuple.

    For move_group and push actions (>=200), this requires env to know where the
    current player's active swans are.

    Supported decoded shapes:
      - ("place_swan",  r, c)
      - ("place_stone", r, c)
      - ("move_group",  subset, (dr,dc))
      - ("push",        (my_r, my_c), (enemy_r, enemy_c))
    """
    if not (0 <= idx < ACTION_SIZE):
        raise ValueError(f"Action index out of range: {idx}")

    # 0–99: place_swan
    if idx < PLACE_SWAN_END:
        r, c = _index_to_square(idx)
        return "place_swan", r, c

    # 100–199: place_stone
    if idx < PLACE_STONE_END:
        rc = idx - PLACE_SWAN_END
        r, c = _index_to_square(rc)
        return "place_stone", r, c

    if env is None:
        raise ValueError(
            "decode_action for move_group / push requires env to know swan positions."
        )

    # 200–703: move_group with subset of up to 6 swans
    if idx < MOVE_GROUP_END:
        move_idx = idx - MOVE_GROUP_START
        subset_index = move_idx // 8  # 0..62
        dir_index = move_idx % 8      # 0..7

        mask = SUBSET_MASKS[subset_index]

        swans = _active_swans_for_current_player(env)
        subset_coords: List[Tuple[int, int]] = []
        for i, coord in enumerate(swans):
            if mask & (1 << i):
                subset_coords.append(coord)

        dr, dc = DIRS8[dir_index]
        return "move_group", subset_coords, (dr, dc)

    # 704–751: push with one of up to 6 swans in any of 8 directions
    if idx < PUSH_END:
        push_idx = idx - PUSH_START
        swan_index = push_idx // PUSH_ACTIONS_PER_SWAN  # 0..5
        dir_index = push_idx % PUSH_ACTIONS_PER_SWAN    # 0..7

        swans = _active_swans_for_current_player(env)
        if swan_index >= len(swans):
            # This can happen if there are fewer than MAX_PUSH_SWANS active swans.
            raise ValueError(
                f"Decoded push refers to swan index {swan_index}, "
                f"but only {len(swans)} active swans are present."
            )

        my_r, my_c = swans[swan_index]
        dr, dc = DIRS8[dir_index]
        enemy_r = my_r + dr
        enemy_c = my_c + dc

        return "push", (my_r, my_c), (enemy_r, enemy_c)

    # Should be unreachable due to the initial range check.
    raise ValueError(f"Unhandled action index: {idx}")


def encode_action(env, action: Action) -> int:
    """
    Convert an environment Action into an integer index.

    Supports:
      - ("place_swan",  r, c)
      - ("place_stone", r, c)
      - ("move_group",  subset, (dr,dc)), where subset is a list of (r,c)
        belonging to the *current* player's active swans (size 1..6).
      - ("push",        (my_r, my_c), (enemy_r, enemy_c))
        my_r,my_c must be an active Swan of the current player.
    """
    kind = action[0]

    # ------------------------------------------------------------
    # place_swan
    # ------------------------------------------------------------
    if kind == "place_swan":
        _, r, c = action
        idx = _square_index(r, c)
        if not (0 <= idx < 100):
            raise ValueError(f"place_swan outside board: {action}")
        return PLACE_SWAN_START + idx

    # ------------------------------------------------------------
    # place_stone
    # ------------------------------------------------------------
    if kind == "place_stone":
        _, r, c = action
        rc = _square_index(r, c)
        if not (0 <= rc < 100):
            raise ValueError(f"place_stone outside board: {action}")
        return PLACE_STONE_START + rc

    # ------------------------------------------------------------
    # move_group
    # ------------------------------------------------------------
    if kind == "move_group":
        _, subset, direction = action
        if not (1 <= len(subset) <= MAX_SWANS):
            raise ValueError(
                f"encode_action supports subset sizes 1..{MAX_SWANS}, "
                f"got {len(subset)} in {action}"
            )

        swans = _active_swans_for_current_player(env)
        mask = _subset_mask_from_coords(subset, swans)

        try:
            subset_index = MASK_TO_INDEX[mask]  # 0..62
        except KeyError:
            raise ValueError(f"Subset mask {mask} out of supported range in {action}")

        dr, dc = direction
        try:
            dir_index = DIRS8.index((int(dr), int(dc)))
        except ValueError:
            raise ValueError(f"Unknown direction {direction} in move_group.")

        # 200 + subset_index * 8 + dir_index
        return MOVE_GROUP_START + subset_index * 8 + dir_index

    # ------------------------------------------------------------
    # push
    # ------------------------------------------------------------
    if kind == "push":
        # Expected shape: ("push", (my_r, my_c), (enemy_r, enemy_c))
        _, my_pos, enemy_pos = action
        my_r, my_c = int(my_pos[0]), int(my_pos[1])
        enemy_r, enemy_c = int(enemy_pos[0]), int(enemy_pos[1])

        swans = _active_swans_for_current_player(env)
        swan_index = _find_swan_index(swans, (my_r, my_c))  # 0..5

        dr = enemy_r - my_r
        dc = enemy_c - my_c
        try:
            dir_index = DIRS8.index((int(dr), int(dc)))
        except ValueError:
            raise ValueError(
                f"Direction from {my_pos} to {enemy_pos} is not in DIRS8 in push action."
            )

        if swan_index >= MAX_PUSH_SWANS:
            raise ValueError(
                f"Swan index {swan_index} exceeds MAX_PUSH_SWANS={MAX_PUSH_SWANS}"
            )

        push_idx = swan_index * PUSH_ACTIONS_PER_SWAN + dir_index
        return PUSH_START + push_idx

    raise ValueError(f"Unknown action kind: {kind}")


def legal_action_indices(env) -> List[int]:
    """
    Return the list of action indices that are legal in the given env state.

    Any legal env action that cannot be encoded (e.g. >6 swans somehow,
    or unexpected subsets) is discarded.
    """
    indices: List[int] = []
    for a in env.legal_actions():
        try:
            idx = encode_action(env, a)
        except Exception as e:
            print(f"[encode_error] {a} -> {e}")
            continue
        else:
            indices.append(idx)
    return indices


class ActionSpace:
    """
    Thin helper wrapper around the module-level functions.
    Mainly for convenience / backwards compatibility.
    """

    def __init__(self):
        self.num_actions = ACTION_SIZE

    def encode(self, env, action):
        return encode_action(env, action)

    def decode(self, env, idx):
        return decode_action(idx, env)

    def legal_indices(self, env):
        return legal_action_indices(env)
