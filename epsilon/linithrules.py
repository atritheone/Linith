from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core constants (match HTML / JS implementation)
# ---------------------------------------------------------------------------

BOARD_SIZE = 10

# Board contents
EMPTY       = 0
SWAN_SUN    = 1
SWAN_MOON   = 2
STONE       = 3
FROZEN_SUN  = 4
FROZEN_MOON = 5

# Player identifiers used by the RL code.
# The original HTML uses 1 (Sun) and 2 (Moon).  Here we keep that mapping
# via SWAN_* but expose SUN / MOON as +/-1 as a convenience for learning code.
SUN  = 1
MOON = -1

# Directions
DIRS8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

DIRS4: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def neigh8(r: int, c: int) -> Iterable[Tuple[int, int]]:
    """Yield all in–bounds 8-neighbours of (r, c)."""
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            yield nr, nc


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def is_swan(v: int) -> bool:
    return v in (SWAN_SUN, SWAN_MOON, FROZEN_SUN, FROZEN_MOON)


def is_active_swan(player: int, v: int) -> bool:
    """Active (unfrozen) swan for the given player."""
    if player == SUN:
        return v == SWAN_SUN
    else:
        return v == SWAN_MOON


def same_player_swan(player: int, v: int) -> bool:
    """Any swan (active or frozen) belonging to player."""
    if player == SUN:
        return v in (SWAN_SUN, FROZEN_SUN)
    else:
        return v in (SWAN_MOON, FROZEN_MOON)


def enemy_swan(player: int, v: int) -> bool:
    """Any swan (active or frozen) belonging to the opponent."""
    if player == SUN:
        return v in (SWAN_MOON, FROZEN_MOON)
    else:
        return v in (SWAN_SUN, FROZEN_SUN)


def count_active_swans(player: int, board: np.ndarray) -> int:
    if player == SUN:
        return int(np.sum(board == SWAN_SUN))
    else:
        return int(np.sum(board == SWAN_MOON))


def count_total_swans(player: int, board: np.ndarray) -> int:
    if player == SUN:
        return int(np.sum((board == SWAN_SUN) | (board == FROZEN_SUN)))
    else:
        return int(np.sum((board == SWAN_MOON) | (board == FROZEN_MOON)))


def any_empty(board: np.ndarray) -> bool:
    """True if the position has at least one empty (grey) tile."""
    return bool(np.any(board == EMPTY))


# ---------------------------------------------------------------------------
# "Silver shield" forbidden zones (Rule 5C movement restriction)
# ---------------------------------------------------------------------------

def _forbidden_zone_mask_for_player(board: np.ndarray, player: int) -> np.ndarray:
    """
    Tiles that this player's Swans are not allowed to move into, based on:

        "A Swan cannot move into any of the eight tiles surrounding an
         opponent’s Swan that has no Stones in any of its own eight
         surrounding tiles."

    Implementation details:

    * For every enemy Swan (active or frozen) with zero adjacent Stones,
      we mark all 8 surrounding tiles as forbidden.
    * The Swan's own tile is NOT forbidden by this rule.
    * The mask is computed from the perspective of `player`, but the
      underlying rule is symmetric: for a move of a Swan owned by X, we
      call this helper with `player = X`.
    """
    assert board.shape == (BOARD_SIZE, BOARD_SIZE)

    mask = np.zeros(board.shape, dtype=bool)

    if player == SUN:
        enemy_vals = (SWAN_MOON, FROZEN_MOON)
    else:
        enemy_vals = (SWAN_SUN, FROZEN_SUN)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(board[r, c])
            if v not in enemy_vals:
                continue

            # Does this enemy Swan have any adjacent Stone?
            has_stone = False
            for nr, nc in neigh8(r, c):
                if int(board[nr, nc]) == STONE:
                    has_stone = True
                    break

            if has_stone:
                continue

            # No adjacent stones – mark its 8-neighbour tiles as forbidden.
            for nr, nc in neigh8(r, c):
                mask[nr, nc] = True

    return mask


# ---------------------------------------------------------------------------
# Placement legality (rules section 5A & 5B)
# ---------------------------------------------------------------------------

def legal_swan_placements(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """All legal Swan placements for *normal* turns.

    Rules encoded:

    * Each player may have at most 6 Swans on the board (active + frozen).
    * Place on an empty (grey) tile.
    * The tile must be orthogonally adjacent to at least one of your Swans
      (active or frozen).
    * The tile must NOT be orthogonally adjacent to any opponent Swan
      (active or frozen).

    Initial setup Swans (Sun's first Swan anywhere; Moon's Swan anywhere
    not adjacent orthogonally or diagonally to Sun's first Swan) are
    handled by the environment, not this helper.
    """
    assert board.shape == (BOARD_SIZE, BOARD_SIZE)

    if count_total_swans(player, board) >= 6:
        return []

    placements: List[Tuple[int, int]] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            has_adj_mine = False
            has_adj_enemy = False

            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                v = int(board[nr, nc])
                if not is_swan(v):
                    continue
                if same_player_swan(player, v):
                    has_adj_mine = True
                elif enemy_swan(player, v):
                    has_adj_enemy = True

            if not has_adj_mine:
                continue
            if has_adj_enemy:
                continue

            placements.append((r, c))

    return placements


def legal_stone_placements(board: np.ndarray) -> List[Tuple[int, int]]:
    """All legal Stone placements (rule 5B): any empty tile."""
    assert board.shape == (BOARD_SIZE, BOARD_SIZE)
    coords: List[Tuple[int, int]] = []
    empties = np.where(board == EMPTY)
    for r, c in zip(*empties):
        coords.append((int(r), int(c)))
    return coords


# ---------------------------------------------------------------------------
# Swan group movement (rules section 5C) + movement restriction
# ---------------------------------------------------------------------------

def _group_move_valid(
    board: np.ndarray,
    player: int,
    subset: List[Tuple[int, int]],
    dr: int,
    dc: int,
) -> bool:
    """Legacy boolean wrapper kept for compatibility.

    The canonical implementation lives in `_legal_move_subset_local`.
    """
    return _legal_move_subset_local(board, subset, (dr, dc), player) is not None


def _legal_move_subset_local(
    board: np.ndarray,
    subset: List[Tuple[int, int]],
    direction: Tuple[int, int],
    player: int,
    apply_forbidden: bool = True,
):
    """Local legality check for a multi-Swan move.

    This is a faithful port of the HTML helper
    `legalMoveSubsetLocal(subset, dir)` plus the extra movement
    restriction from the rules:

        "A Swan cannot move into any of the eight tiles surrounding an
         opponent’s Swan that has no Stones in any of its own eight
         surrounding tiles."

    * All Swans in `subset` are moved by the same (dr, dc) in one step.
    * Stones orthogonally or diagonally adjacent to any moving Swan
      are dragged along, *except*:
        - Stones also adjacent to an unmoved friendly *active* Swan.
        - Stones also adjacent to any enemy Swan.
    * Swans and Stones may only move into tiles that are effectively empty
      after considering vacated source squares and moved Stones.
    * No collisions: Stones cannot be pushed into each other or off board.
    * Swan targets may not lie in forbidden "silver shield" tiles.
    """
    dr, dc = direction

    def inb(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    # Movement restriction mask for this side's Swans
    forbidden = _forbidden_zone_mask_for_player(board, player)

    # Encode moving Swans as scalar positions for quick membership checks
    moving: Set[int] = {r * BOARD_SIZE + c for (r, c) in subset}

    # Stones that will move, and their destinations
    stones_from: Set[Tuple[int, int]] = set()
    stones_to: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # 1) Collect stones dragged by the moving subset (ignoring shared stones)
    for (r, c) in subset:
        for er, ec in DIRS8:
            sr, sc = r + er, c + ec
            if not inb(sr, sc):
                continue
            if board[sr, sc] != STONE:
                continue

            # Is this stone shared between moving swans and
            # an unmoved friendly active swan or an enemy swan?
            shared = False
            for ar, ac in DIRS8:
                xr, xc = sr + ar, sc + ac
                if not inb(xr, xc):
                    continue
                vv = int(board[xr, xc])
                if not is_swan(vv):
                    continue
                if enemy_swan(player, vv):
                    shared = True
                    break
                if (
                    same_player_swan(player, vv)
                    and is_active_swan(player, vv)
                    and (xr * BOARD_SIZE + xc) not in moving
                ):
                    shared = True
                    break

            if shared:
                continue

            tr, tc = sr + dr, sc + dc
            if not inb(tr, tc):
                # dragged stone would go off board → illegal
                return None

            stones_from.add((sr, sc))
            stones_to[(sr, sc)] = (tr, tc)

    def is_vacant_after_move(r: int, c: int) -> bool:
        """Test whether (r, c) will be empty after the move is applied."""
        if not inb(r, c):
            return False
        v = int(board[r, c])
        if v == EMPTY:
            return True
        # Tile currently occupied by a moving Swan – it will vacate.
        if (r * BOARD_SIZE + c) in moving:
            return True
        # Tile currently occupied by a Stone that is being moved away.
        if (r, c) in stones_from:
            return True
        return False

    # 2) Validate Swan target squares (including forbidden zones)
    for (r, c) in subset:
        nr, nc = r + dr, c + dc
        if not inb(nr, nc):
            return None
        # Rule: cannot move into forbidden "silver shield" tiles
        # for voluntary moves. Pushes set apply_forbidden=False and
        # bypass this check.
        if apply_forbidden and forbidden[nr, nc]:
            return None

        occ = int(board[nr, nc])
        if occ == EMPTY:
            continue
        if is_swan(occ):
            # Cannot move into a non-moving Swan
            if (nr * BOARD_SIZE + nc) not in moving:
                return None
        elif occ == STONE:
            # The Stone in front of us must itself be dragged by the move,
            # and its destination must be vacant.
            src = (nr, nc)
            if src not in stones_to:
                return None
            tr, tc = stones_to[src]
            if not is_vacant_after_move(tr, tc):
                return None
        else:
            # Any other code is illegal to enter
            return None

    # 3) Validate all dragged Stone targets: no overlap and must be vacant
    seen_targets: Set[Tuple[int, int]] = set()
    for _, (tr, tc) in stones_to.items():
        if not is_vacant_after_move(tr, tc):
            return None
        if (tr, tc) in seen_targets:
            # two Stones trying to occupy same tile
            return None
        seen_targets.add((tr, tc))

    return stones_from, stones_to


def simulate_group_move(
    board: np.ndarray,
    subset: List[Tuple[int, int]],
    direction: Tuple[int, int],
    player: int,
    apply_forbidden: bool = True,
) -> Optional[np.ndarray]:
    """Apply a legal multi-Swan move and return the new board.

    If the move is illegal, returns ``None``.

    ``apply_forbidden`` controls whether the "silver shield"
    restriction is applied. Normal Swan moves use the default
    (True); pushes will pass False.
    """
    res = _legal_move_subset_local(board, subset, direction, player, apply_forbidden)
    if res is None:
        return None

    stones_from, stones_to = res
    dr, dc = direction

    nb = board.copy()

    # 1) Clear original Swan positions
    for (r, c) in subset:
        nb[r, c] = EMPTY

    # 2) Clear original Stone positions
    for (sr, sc) in stones_from:
        nb[sr, sc] = EMPTY

    # 3) Place Stones at their targets
    for (sr, sc), (tr, tc) in stones_to.items():
        nb[tr, tc] = STONE

    # 4) Place Swans at their new positions
    swan_code = SWAN_SUN if player == SUN else SWAN_MOON
    for (r, c) in subset:
        nr, nc = r + dr, c + dc
        nb[nr, nc] = swan_code

    return nb


def legal_group_moves(
    board: np.ndarray,
    player: int,
    max_group_size: int = 6,
):
    """Yield all legal group moves for the given player.

    Yields tuples ``(subset, (dr, dc))`` where:

    * ``subset`` is a list of coordinates of active Swans for that player.
    * ``(dr, dc)`` is one of the 8 directions in ``DIRS8``.
    """
    active: List[Tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if is_active_swan(player, v):
                active.append((r, c))

    n = len(active)
    if n == 0:
        return

    # Enumerate all non-empty subsets of active swans up to max_group_size.
    # We iterate by bitmask for deterministic ordering.
    for mask in range(1, 1 << n):
        # quick popcount limit
        if mask.bit_count() > max_group_size:
            continue
        subset = [active[i] for i in range(n) if (mask & (1 << i))]
        for (dr, dc) in DIRS8:
            if _legal_move_subset_local(board, subset, (dr, dc), player) is not None:
                yield subset, (dr, dc)


# ---------------------------------------------------------------------------
# Push actions (Rule 5D)
# ---------------------------------------------------------------------------

def legal_push_moves(
    board: np.ndarray,
    player: int,
):
    """
    Yield all legal single-Swan push moves for `player`.

    Rule fragment:

        "Push Swan(s)
         Push an enemy Swan away from your own Swan.
         The enemy Swan must be active and in an adjacent tile
         (orthogonal or diagonal)."

    Each yielded entry is:

        ((my_r, my_c), (enemy_r, enemy_c), (dr, dc))

    where (dr, dc) is the push direction from (my_r, my_c) through
    (enemy_r, enemy_c). When the enemy Swan is pushed, all Stones
    unique to that Swan move with it using the same rules as normal
    Swan movement.
    """
    assert board.shape == (BOARD_SIZE, BOARD_SIZE)

    enemy_player = MOON if player == SUN else SUN

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_active_swan(player, int(board[r, c])):
                continue

            # Consider each adjacent tile for a potential enemy Swan.
            for dr, dc in DIRS8:
                er, ec = r + dr, c + dc
                if not in_bounds(er, ec):
                    continue

                v = int(board[er, ec])
                # Enemy Swan must be active (cannot push frozen Swans).
                if not is_active_swan(enemy_player, v):
                    continue

                # Destination of the enemy Swan after the push.
                tr, tc = er + dr, ec + dc
                if not in_bounds(tr, tc):
                    continue

                # Check legality of moving the enemy Swan (and its Stones)
                # from its own perspective, including collisions.
                # IMPORTANT: pushed Swans are allowed to enter "silver shield"
                # tiles, so we set apply_forbidden=False.
                if _legal_move_subset_local(
                    board,
                    [(er, ec)],
                    (dr, dc),
                    enemy_player,
                    apply_forbidden=False,
                ) is None:
                    continue

                yield (r, c), (er, ec), (dr, dc)


def simulate_push_move(
    board: np.ndarray,
    player: int,
    my_pos: Tuple[int, int],
    enemy_pos: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Apply a single-Swan push and return the new board.

    The caller is expected to ensure that `(my_pos, enemy_pos, …)` is one
    of the results of `legal_push_moves`. If the push is illegal, returns
    ``None``.
    """
    (mr, mc) = my_pos
    (er, ec) = enemy_pos

    dr = er - mr
    dc = ec - mc

    if (dr, dc) not in DIRS8:
        return None

    enemy_player = MOON if player == SUN else SUN

    if not in_bounds(mr, mc) or not in_bounds(er, ec):
        return None
    if not is_active_swan(player, int(board[mr, mc])):
        return None
    if not is_active_swan(enemy_player, int(board[er, ec])):
        return None

    # Reuse group-move simulation from enemy's perspective: we are
    # moving their Swan (and its Stones) while our Swan remains in place.
    return simulate_group_move(
        board,
        [(er, ec)],
        (dr, dc),
        enemy_player,
        apply_forbidden=False,
    )


# ---------------------------------------------------------------------------
# Encirclement / freezing (rules section 6 & 7)
# ---------------------------------------------------------------------------

def _collect_active_groups(board: np.ndarray) -> Dict[int, List[List[Tuple[int, int]]]]:
    """Collect 8-connected components of *active* Swans for each player."""
    seen = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    groups: Dict[int, List[List[Tuple[int, int]]]] = {SUN: [], MOON: []}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(board[r, c])
            if not (is_active_swan(SUN, v) or is_active_swan(MOON, v)):
                continue
            if seen[r][c]:
                continue

            owner = SUN if is_active_swan(SUN, v) else MOON
            stack: List[Tuple[int, int]] = [(r, c)]
            comp: List[Tuple[int, int]] = []
            seen[r][c] = True

            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dr, dc in DIRS8:
                    nx, ny = x + dr, y + dc
                    if not in_bounds(nx, ny) or seen[nx][ny]:
                        continue
                    if is_active_swan(owner, int(board[nx, ny])):
                        seen[nx][ny] = True
                        stack.append((nx, ny))

            groups[owner].append(comp)

    return groups


def _group_encircled(
    board: np.ndarray,
    comp: List[Tuple[int, int]],
    owner: int,
) -> bool:
    """Check whether an active Swan group is fully encircled.

    A group is encircled iff *every* neighbouring tile (8-neighbourhood of
    all group members), that is not itself in the group and is on the board,
    is non-empty and is not occupied by a friendly active Swan.

    This matches the rule text:

        "A Swan or connected group of Swans is encircled when all eight
         surrounding tiles (orthogonal and diagonal) are occupied by Stones,
         Enemy Swans, and/or the board’s edge."

    The board edge counts implicitly because out-of-bounds tiles are never
    required to be empty; only in-bounds neighbours can fail the test.
    """
    inside: Set[Tuple[int, int]] = set(comp)
    for (r, c) in comp:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                # board edge – always counts as blocking
                continue
            if (nr, nc) in inside:
                continue
            v = int(board[nr, nc])
            if v == EMPTY:
                # open grey tile → not encircled
                return False
            # A neighbouring friendly *active* Swan extends the active group;
            # that situation should have merged into this component instead.
            if same_player_swan(owner, v) and is_active_swan(owner, v):
                return False
    return True


@dataclass
class FreezeResult:
    board: np.ndarray
    froze_sun: int      # number of Sun Swans frozen (but not final group)
    froze_moon: int     # number of Moon Swans frozen (but not final group)
    sealed_sun: int     # number of Sun Swans in the final active group
    sealed_moon: int    # number of Moon Swans in the final active group


def compute_freezes_on(board: np.ndarray) -> FreezeResult:
    """Freeze all newly encircled active Swan groups.

    Returns a new board and counts that the environment can use to
    implement:

    * Extra actions (one per enemy Swan frozen, including the final group).
    * Immediate loss when a player's *final* active Swan group is encircled.
    """
    nb = board.copy()
    groups = _collect_active_groups(nb)

    froze_sun = froze_moon = 0
    sealed_sun = sealed_moon = 0

    for owner in (SUN, MOON):
        for comp in groups[owner]:
            if not _group_encircled(nb, comp, owner):
                continue

            # Freeze all active Swans in this component
            active_before = count_active_swans(owner, nb)
            for (r, c) in comp:
                v = int(nb[r, c])
                if v == SWAN_SUN:
                    nb[r, c] = FROZEN_SUN
                elif v == SWAN_MOON:
                    nb[r, c] = FROZEN_MOON

            # If this encircled group contained *all* of that player's
            # remaining active Swans, it is their final active group.
            if len(comp) == active_before:
                if owner == SUN:
                    sealed_sun += len(comp)
                else:
                    sealed_moon += len(comp)
            else:
                if owner == SUN:
                    froze_sun += len(comp)
                else:
                    froze_moon += len(comp)

    return FreezeResult(
        board=nb,
        froze_sun=froze_sun,
        froze_moon=froze_moon,
        sealed_sun=sealed_sun,
        sealed_moon=sealed_moon,
    )


# ---------------------------------------------------------------------------
# Small helpers for rule-7 style end-condition handling
# ---------------------------------------------------------------------------

def extra_actions_for_player(current_player: int, freeze: FreezeResult) -> int:
    """How many extra actions should *current_player* receive this turn?

    Rule:

        "Encircling an opponent's Swan grants one extra action per Swan
         frozen."

    The environment is responsible for calling ``compute_freezes_on`` on
    the board *after* the current player's move and then passing the
    resulting ``FreezeResult`` into this helper.

    ``sealed_*`` are also Swans frozen this turn and therefore count
    as extra actions.
    """
    if current_player == SUN:
        # Sun just moved; any Moon Swans that became frozen grant actions.
        return freeze.froze_moon + freeze.sealed_moon
    else:
        return freeze.froze_sun + freeze.sealed_sun


def base_actions_per_turn(board: np.ndarray) -> int:
    """
    Base number of actions per turn *before* adding freeze bonuses.

    Rule:

        "Once both players have six Swans on the board, each turn
         has two actions."

    We count total Swans (active + frozen) for each side and switch
    from 1 → 2 actions once both are at six or more.
    """
    sun_total = count_total_swans(SUN, board)
    moon_total = count_total_swans(MOON, board)
    if sun_total >= 6 and moon_total >= 6:
        return 2
    return 1


def has_any_legal_action(board: np.ndarray, player: int) -> bool:
    """Convenience helper for draw detection (rule 7).

    A position is a draw if *neither* player has any legal Swan placement,
    Stone placement, or Swan move and both still have active Swans.
    This function checks the "any legal action" part for a single player.
    """
    if legal_swan_placements(board, player):
        return True
    if legal_stone_placements(board):
        return True
    for _subset, _dir in legal_group_moves(board, player):
        return True
    # Pushes are also legal actions if available.
    for _ in legal_push_moves(board, player):
        return True
    return False
