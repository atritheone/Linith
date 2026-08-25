# linithai.py  – updated to match current JS linithAI (naked zones, territory, pushes)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import random
import numpy as np
import time

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
    legal_push_moves
)

DEBUG_HARD_AI = False

# Per-player last action and repeat counter
_LAST_ACTION_PER_PLAYER: Dict[int, Optional[Tuple[Any, ...]]] = {
    SUN: None,
    MOON: None,
}
_REPEAT_COUNT_PER_PLAYER: Dict[int, int] = {
    SUN: 0,
    MOON: 0,
}

MAX_REPEAT_SAME_ACTION = 3

# Directions – must match JS
DIRS8: List[Tuple[int, int]] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

DIRS4: List[Tuple[int, int]] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]

# “decisive stone” thresholds
DECISIVE_CHOKE = 3
DECISIVE_RING = 3.0


# ---------- low-level helpers ----------


def set_hard_ai_debug(enabled: bool) -> None:
    global DEBUG_HARD_AI
    DEBUG_HARD_AI = bool(enabled)


def inb(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get(b: np.ndarray, r: int, c: int) -> int:
    return int(b[r, c])


def setv(b: np.ndarray, r: int, c: int, v: int) -> None:
    b[r, c] = v


def clone(b: np.ndarray) -> np.ndarray:
    return b.copy()


def is_empty(b: np.ndarray, r: int, c: int) -> bool:
    return get(b, r, c) == EMPTY


def is_active_swan(p: int, v: int) -> bool:
    return (p == SUN and v == SWAN_SUN) or (p == MOON and v == SWAN_MOON)


def same_player_swan(p: int, v: int) -> bool:
    if p == SUN:
        return v == SWAN_SUN or v == FROZEN_SUN
    else:
        return v == SWAN_MOON or v == FROZEN_MOON


def enemy_swan(p: int, v: int) -> bool:
    if p == SUN:
        return v == SWAN_MOON or v == FROZEN_MOON
    else:
        return v == SWAN_SUN or v == FROZEN_SUN


def neigh8(r: int, c: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if inb(nr, nc):
            out.append((nr, nc))
    return out


def neigh4(r: int, c: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if inb(nr, nc):
            out.append((nr, nc))
    return out


def shuffled(seq):
    arr = list(seq)
    random.shuffle(arr)
    return arr


def count_active_swans(p: int, b: np.ndarray) -> int:
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_active_swan(p, get(b, r, c)):
                n += 1
    return n


def liberties_for(p: int, b: np.ndarray) -> int:
    """Count distinct empty tiles adjacent (8-neighbours) to any active Swan for player p."""
    seen = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = get(b, r, c)
            if not is_active_swan(p, v):
                continue
            for nr, nc in neigh8(r, c):
                if is_empty(b, nr, nc):
                    seen.add((nr, nc))
    return len(seen)


def both_at_six(b: np.ndarray) -> bool:
    def total(p: int) -> int:
        n = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if same_player_swan(p, get(b, r, c)):
                    n += 1
        return n

    return total(SUN) >= 6 and total(MOON) >= 6


# ---------- encirclement & pressure ----------


def collect_active_groups(b: np.ndarray) -> Dict[int, List[List[Tuple[int, int]]]]:
    seen = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    groups: Dict[int, List[List[Tuple[int, int]]]] = {SUN: [], MOON: []}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = get(b, r, c)
            if not is_active_swan(SUN, v) and not is_active_swan(MOON, v):
                continue
            if seen[r][c]:
                continue
            owner = SUN if is_active_swan(SUN, v) else MOON
            q = [(r, c)]
            comp: List[Tuple[int, int]] = []
            seen[r][c] = True
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for nx, ny in neigh8(x, y):
                    if seen[nx][ny]:
                        continue
                    if is_active_swan(owner, get(b, nx, ny)):
                        seen[nx][ny] = True
                        q.append((nx, ny))
            groups[owner].append(comp)
    return groups


def group_encircled(b: np.ndarray, comp: List[Tuple[int, int]], owner: int) -> bool:
    inside = set(comp)
    for r, c in comp:
        for nr, nc in neigh8(r, c):
            if (nr, nc) in inside:
                continue
            v = get(b, nr, nc)
            if v == EMPTY:
                return False
            if same_player_swan(owner, v) and is_active_swan(owner, v):
                return False
    return True


@dataclass
class FreezeResult:
    frozeSun: int = 0
    frozeMoon: int = 0
    sealedSun: int = 0
    sealedMoon: int = 0


def freeze_encircled(b: np.ndarray) -> FreezeResult:
    res = FreezeResult()
    groups = collect_active_groups(b)

    for owner in (SUN, MOON):
        for comp in groups[owner]:
            if group_encircled(b, comp, owner):
                active_before = count_active_swans(owner, b)
                for r, c in comp:
                    v = get(b, r, c)
                    if v == SWAN_SUN:
                        setv(b, r, c, FROZEN_SUN)
                    elif v == SWAN_MOON:
                        setv(b, r, c, FROZEN_MOON)
                if len(comp) == active_before:
                    if owner == SUN:
                        res.sealedSun += len(comp)
                    else:
                        res.sealedMoon += len(comp)
                else:
                    if owner == SUN:
                        res.frozeSun += len(comp)
                    else:
                        res.frozeMoon += len(comp)
    return res


def enemy_ring_pressure(b: np.ndarray, player: int) -> float:
    foe = SUN if player == MOON else MOON

    def is_enemy_active(v: int) -> bool:
        return (foe == SUN and v == SWAN_SUN) or (foe == MOON and v == SWAN_MOON)

    seen = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    groups: List[List[Tuple[int, int]]] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = get(b, r, c)
            if not is_enemy_active(v) or seen[r][c]:
                continue
            Q = [(r, c)]
            seen[r][c] = True
            comp: List[Tuple[int, int]] = []
            while Q:
                x, y = Q.pop()
                comp.append((x, y))
                for dr, dc in DIRS8:
                    nx, ny = x + dr, y + dc
                    if not inb(nx, ny) or seen[nx][ny]:
                        continue
                    if is_enemy_active(get(b, nx, ny)):
                        seen[nx][ny] = True
                        Q.append((nx, ny))
            groups.append(comp)

    score = 0.0
    for comp in groups:
        rim = set()
        for r, c in comp:
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not inb(nr, nc):
                    continue
                if get(b, nr, nc) == EMPTY:
                    rim.add((nr, nc))
        k = len(rim)
        if k <= 6:
            score += (6 - k) * 1.0
        if k <= 3:
            score += 2.0
        if k <= 1:
            score += 4.0
    return score


# ---------- naked enemy swan helpers (global) ----------


def is_enemy_swan_naked_global(b: np.ndarray, r: int, c: int, player: int) -> bool:
    v = get(b, r, c)
    if not enemy_swan(player, v):
        return False
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if not inb(nr, nc):
            continue
        if get(b, nr, nc) == STONE:
            return False
    return True


def is_in_naked_enemy_zone_global(b: np.ndarray, r: int, c: int, player: int) -> bool:
    for dr, dc in DIRS8:
        er, ec = r + dr, c + dc
        if not inb(er, ec):
            continue
        if is_enemy_swan_naked_global(b, er, ec, player):
            return True
    return False


# ---------- evaluation helpers ----------


def stone_advances_game(b_before: np.ndarray, b_after: np.ndarray, player: int) -> bool:
    nb = clone(b_after)
    res = freeze_encircled(nb)
    froze_enemy = (res.frozeMoon + res.sealedMoon) if player == SUN else (res.frozeSun + res.sealedSun)
    if froze_enemy > 0:
        return True

    opp = SUN if player == MOON else MOON

    my_lib_before = liberties_for(player, b_before)
    opp_lib_before = liberties_for(opp, b_before)
    my_lib_after = liberties_for(player, nb)
    opp_lib_after = liberties_for(opp, nb)

    if opp_lib_after <= opp_lib_before - 1:
        return True
    if my_lib_after >= my_lib_before + 1:
        return True

    pr_before = enemy_ring_pressure(b_before, player)
    pr_after = enemy_ring_pressure(nb, player)
    if pr_after >= pr_before + 1.0:
        return True

    return False


def territory_advantage(b: np.ndarray, current: int) -> float:
    """
    Mirror of JS territoryAdvantage:
      - BFS from empty neighbours of our / their active Swans
      - respect naked enemy zones (cannot flow through them)
      - compare reachability & distance.
    """
    opp = SUN if current == MOON else MOON
    INF = 99
    MAXD = 6
    EDGE = 10.0
    SCALE = 2.0

    def bfs(player: int):
        dist = [[INF] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        q: List[Tuple[int, int]] = []

        # seed: empty neighbours of active swans for this player
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = get(b, r, c)
                if not is_active_swan(player, v):
                    continue
                for dr, dc in DIRS8:
                    nr, nc = r + dr, c + dc
                    if not inb(nr, nc):
                        continue
                    if not is_empty(b, nr, nc):
                        continue
                    if is_in_naked_enemy_zone_global(b, nr, nc, player):
                        continue
                    if dist[nr][nc] > 1:
                        dist[nr][nc] = 1
                        q.append((nr, nc))

        # flood-fill over empty tiles, respecting naked enemy zones
        head = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            d = dist[r][c]
            if d >= MAXD:
                continue
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not inb(nr, nc):
                    continue
                if not is_empty(b, nr, nc):
                    continue
                if is_in_naked_enemy_zone_global(b, nr, nc, player):
                    continue
                if dist[nr][nc] > d + 1:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc))

        return dist

    d_me = bfs(current)
    d_opp = bfs(opp)

    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_empty(b, r, c):
                continue

            dm = d_me[r][c]
            do = d_opp[r][c]

            me_far = dm > MAXD
            opp_far = do > MAXD
            if me_far and opp_far:
                continue

            # treat unreachable as MAXD+1
            if dm > MAXD:
                dm = MAXD + 1
            if do > MAXD:
                do = MAXD + 1

            if dm <= MAXD and do == MAXD + 1:
                score += EDGE
            elif do <= MAXD and dm == MAXD + 1:
                score -= EDGE
            else:
                diff = do - dm  # >0 means we are closer
                score += diff * SCALE

    return score


def evaluate_styled(
    b_before: np.ndarray,
    b_after: np.ndarray,
    current: int,
    style_name: str = "doctrinal",
    my_lib_before_override: Optional[int] = None,
    opp_lib_before_override: Optional[int] = None,
) -> float:
    nb = clone(b_after)
    res = freeze_encircled(nb)

    my_active_after = count_active_swans(current, nb)
    opp = SUN if current == MOON else MOON
    opp_active_after = count_active_swans(opp, nb)

    if opp_active_after == 0 and my_active_after > 0:
        return 1e9
    if my_active_after == 0 and opp_active_after > 0:
        return -1e9

    my_lib_before = my_lib_before_override if my_lib_before_override is not None else liberties_for(current, b_before)
    opp_lib_before = opp_lib_before_override if opp_lib_before_override is not None else liberties_for(opp, b_before)
    my_lib_after = liberties_for(current, nb)
    opp_lib_after = liberties_for(opp, nb)

    my_delta = my_lib_after - my_lib_before
    opp_delta = opp_lib_after - opp_lib_before

    froze_gain = (res.frozeMoon + res.sealedMoon) if current == SUN else (res.frozeSun + res.sealedSun)
    self_loss = (res.frozeSun + res.sealedSun) if current == SUN else (res.frozeMoon + res.sealedMoon)
    ring = enemy_ring_pressure(nb, current)
    momentum = 10 if both_at_six(b_before) else 0

    # style weights (defaults; STYLE[style_name] can override in JS – here we keep fixed)
    wFreeze = 500.0
    wSelfFreeze = -600.0
    wMyLib = 5.0
    wOpLib = -9.0
    wRing = 0.0
    wMomentum = float(momentum)
    wSpace = 0.0  # hook for territory, default off (matches JS default)

    tot_frozen = (res.frozeSun + res.sealedSun) + (res.frozeMoon + res.sealedMoon)
    phase = max(0.0, min(1.0, tot_frozen / 6.0))
    freeze_boost = 1.0 + (0.25 * phase if style_name == "blizzard" else 0.12 * phase)
    ring_boost = 1.0 + (0.25 * phase if style_name == "fortress" else 0.08 * phase)

    terr_delta = 0.0
    if wSpace != 0.0:
        terr_before = territory_advantage(b_before, current)
        terr_after = territory_advantage(nb, current)
        terr_delta = terr_after - terr_before

    return (
        froze_gain * (wFreeze * freeze_boost)
        + self_loss * wSelfFreeze
        + my_delta * wMyLib
        + opp_delta * wOpLib
        + ring * (wRing * ring_boost)
        + momentum * froze_gain
        + terr_delta * wSpace
    )


def decisive_stone(b: np.ndarray, r: int, c: int, me: int, style_name: str = "doctrinal") -> bool:
    b2 = clone(b)
    setv(b2, r, c, STONE)
    nb = clone(b2)
    res = freeze_encircled(nb)
    froze_enemy = (res.frozeMoon + res.sealedMoon) if me == SUN else (res.frozeSun + res.sealedSun)
    if froze_enemy > 0:
        return True
    opp = SUN if me == MOON else MOON
    opp_lib_before = liberties_for(opp, b)
    opp_lib_after = liberties_for(opp, nb)
    if opp_lib_after <= opp_lib_before - DECISIVE_CHOKE:
        return True
    pr_before = enemy_ring_pressure(b, me)
    pr_after = enemy_ring_pressure(nb, me)
    if pr_after >= pr_before + DECISIVE_RING:
        return True
    return False


# ---------- legal placements & movement ----------


def legal_swan_placements(b: np.ndarray, p: int) -> List[Tuple[int, int]]:
    """
    Swan placement rules:
      - If the player has no active swans yet, prefer central 2x2 cluster.
      - Otherwise, adjacency to own / no-adjacent-enemy (JS behaviour).
    """
    my_swans = active_swans_of(b, p)

    if not my_swans:
        centres = [(4, 4), (4, 5), (5, 4), (5, 5)]
        out = [(r, c) for (r, c) in centres if is_empty(b, r, c)]
        if out:
            return out
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if is_empty(b, r, c)
        ]

    out: List[Tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_empty(b, r, c):
                continue
            has_adj_mine = False
            adj_enemy = False
            for nr, nc in neigh4(r, c):
                v = get(b, nr, nc)
                if same_player_swan(p, v):
                    has_adj_mine = True
            for nr, nc in neigh8(r, c):
                v = get(b, nr, nc)
                if enemy_swan(p, v):
                    adj_enemy = True
                    break
            if has_adj_mine and not adj_enemy:
                out.append((r, c))
    return out


def legal_stone_placements(b: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """
    Stone placement heuristic as in JS:
      1) squares adjacent to an enemy swan,
      2) otherwise 'frontier' squares,
      3) otherwise any empty square.
    """
    adj_enemy: List[Tuple[int, int]] = []
    frontier: List[Tuple[int, int]] = []
    all_cells: List[Tuple[int, int]] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not is_empty(b, r, c):
                continue
            all_cells.append((r, c))

            near_any = False
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not inb(nr, nc):
                    continue
                v = get(b, nr, nc)
                if v != EMPTY:
                    near_any = True
                if enemy_swan(player, v):
                    adj_enemy.append((r, c))
                    near_any = True
                    break
            if near_any:
                frontier.append((r, c))

    if adj_enemy:
        return adj_enemy
    if frontier:
        return frontier
    return all_cells


def active_swans_of(b: np.ndarray, p: int) -> List[Tuple[int, int]]:
    arr: List[Tuple[int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = get(b, r, c)
            if is_active_swan(p, v):
                arr.append((r, c))
    return arr


def simulate_move_subset(
    b: np.ndarray,
    p: int,
    subset: List[Tuple[int, int]],
    direction: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Move a subset of *our* active Swans by one step in direction,
    with stone-follow rules and the naked enemy-swan zone restriction.
    """
    dr, dc = direction
    moving = {f"{r},{c}" for r, c in subset}

    # local naked enemy helpers (as in JS simulateMoveSubset)
    def is_enemy_swan_naked_local(sr: int, sc: int) -> bool:
        v = get(b, sr, sc)
        if not enemy_swan(p, v):
            return False
        for dr8, dc8 in DIRS8:
            rr, cc = sr + dr8, sc + dc8
            if not inb(rr, cc):
                continue
            if get(b, rr, cc) == STONE:
                return False
        return True

    def is_in_naked_enemy_zone_local(r: int, c: int) -> bool:
        for dr8, dc8 in DIRS8:
            er, ec = r + dr8, c + dc8
            if not inb(er, ec):
                continue
            if is_enemy_swan_naked_local(er, ec):
                return True
        return False

    # 1) check swan targets, including naked-zone rule
    for r, c in subset:
        nr, nc = r + dr, c + dc
        if not inb(nr, nc):
            return None
        if is_in_naked_enemy_zone_local(nr, nc):
            return None
        v = get(b, nr, nc)
        if v != EMPTY and f"{nr},{nc}" not in moving:
            return None

    # 2) collect stones that move with them
    stones_from: List[Tuple[int, int]] = []
    stone_seen = set()

    for r, c in subset:
        for sr, sc in neigh8(r, c):
            if not inb(sr, sc) or get(b, sr, sc) != STONE:
                continue
            key = f"{sr},{sc}"
            if key in stone_seen:
                continue

            shared = False
            for ar, ac in neigh8(sr, sc):
                if not inb(ar, ac):
                    continue
                vv = get(b, ar, ac)
                if enemy_swan(p, vv):
                    shared = True
                    break
                if same_player_swan(p, vv) and is_active_swan(p, vv) and f"{ar},{ac}" not in moving:
                    shared = True
                    break
            if not shared:
                stone_seen.add(key)
                stones_from.append((sr, sc))

    stones_to = [(sr + dr, sc + dc) for (sr, sc) in stones_from]

    to_set = set()
    for tr, tc in stones_to:
        if not inb(tr, tc):
            return None
        occ = get(b, tr, tc)
        vac = (
            occ == EMPTY
            or f"{tr},{tc}" in moving
            or any(sr == tr and sc == tc for (sr, sc) in stones_from)
        )
        if not vac:
            return None
        tkey = f"{tr},{tc}"
        if tkey in to_set:
            return None
        to_set.add(tkey)

    nb = clone(b)
    # clear swans
    for r, c in subset:
        setv(nb, r, c, EMPTY)
    # clear moved stones
    for sr, sc in stones_from:
        setv(nb, sr, sc, EMPTY)
    # place stones
    for tr, tc in stones_to:
        setv(nb, tr, tc, STONE)
    # place moved swans
    for r, c in subset:
        setv(nb, r + dr, c + dc, SWAN_SUN if p == SUN else SWAN_MOON)

    return nb


def all_swan_subsets(coords: List[Tuple[int, int]]):
    n = len(coords)
    total = 1 << n
    for mask in range(1, total):
        subset = [coords[i] for i in range(n) if mask & (1 << i)]
        yield subset


def simulate_push_subset(
    b: np.ndarray,
    p: int,
    subset: List[Tuple[int, int]],
    direction: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Simulate pushing a subset of ENEMY active Swans in the given direction.
    Mirrors JS simulatePushSubset semantics.
    """
    dr, dc = direction

    def has_friendly_pusher(r: int, c: int) -> bool:
        for nr, nc in neigh8(r, c):
            v = get(b, nr, nc)
            if is_active_swan(p, v):
                return True
        return False

    # all subset must be enemy active swans with at least one friendly pusher
    for r, c in subset:
        v = get(b, r, c)
        if not enemy_swan(p, v) or v in (FROZEN_SUN, FROZEN_MOON):
            return None
        if not has_friendly_pusher(r, c):
            return None

    moving_set = {r * BOARD_SIZE + c for (r, c) in subset}
    stones_from: set[str] = set()
    stones_to: Dict[str, Tuple[int, int]] = {}

    def stone_key(r_: int, c_: int) -> str:
        return f"s:{r_},{c_}"

    def is_vacant_after_move(b_: np.ndarray, r_: int, c_: int) -> bool:
        if not inb(r_, c_):
            return False
        v_ = get(b_, r_, c_)
        if v_ == EMPTY:
            return True
        if (r_ * BOARD_SIZE + c_) in moving_set:
            return True
        if stone_key(r_, c_) in stones_from:
            return True
        return False

    # collect following stones
    for r, c in subset:
        for dr8, dc8 in DIRS8:
            sr, sc = r + dr8, c + dc8
            if not inb(sr, sc) or get(b, sr, sc) != STONE:
                continue

            shared = False
            for ar, ac in DIRS8:
                xr, xc = sr + ar, sc + ac
                if not inb(xr, xc):
                    continue
                vv = get(b, xr, xc)
                if vv == EMPTY:
                    continue
                # Adjacent to friendly of p → shared
                if is_active_swan(p, vv) or same_player_swan(p, vv):
                    shared = True
                    break
                # Adjacent to unmoving ally of the moving side (enemy of p)
                if enemy_swan(p, vv) and vv not in (FROZEN_SUN, FROZEN_MOON) and (xr * BOARD_SIZE + xc) not in moving_set:
                    shared = True
                    break
            if shared:
                continue

            tr, tc = sr + dr, sc + dc
            if not inb(tr, tc):
                return None
            sk = stone_key(sr, sc)
            stones_from.add(sk)
            stones_to[sk] = (tr, tc)

    # validate stone targets
    seen_targets: set[Tuple[int, int]] = set()
    for sk, (tr, tc) in stones_to.items():
        if not is_vacant_after_move(b, tr, tc):
            return None
        key = (tr, tc)
        if key in seen_targets:
            return None
        seen_targets.add(key)

    # validate pushed swan destinations
    dest_set: set[Tuple[int, int]] = set()
    for r, c in subset:
        nr, nc = r + dr, c + dc
        if not inb(nr, nc):
            return None
        occ = get(b, nr, nc)
        dkey = (nr, nc)
        if occ == EMPTY:
            if dkey in dest_set:
                return None
            dest_set.add(dkey)
            continue
        if occ == STONE:
            sk = stone_key(nr, nc)
            if sk not in stones_to:
                return None
            tr, tc = stones_to[sk]
            if not is_vacant_after_move(b, tr, tc):
                return None
            if dkey in dest_set:
                return None
            dest_set.add(dkey)
            continue
        # another enemy swan: must itself be in the pushed subset
        if (nr * BOARD_SIZE + nc) in moving_set:
            if dkey in dest_set:
                return None
            dest_set.add(dkey)
            continue
        return None

    # apply to clone
    nb = clone(b)
    # clear original enemy swans
    for r, c in subset:
        setv(nb, r, c, EMPTY)
    # move stones
    for sk in list(stones_from):
        sr, sc = map(int, sk[2:].split(","))
        setv(nb, sr, sc, EMPTY)
    for _, (tr, tc) in stones_to.items():
        setv(nb, tr, tc, STONE)
    # place enemy swans at destination
    for r, c in subset:
        v = get(b, r, c)
        setv(nb, r + dr, c + dc, v)

    return nb


# ---------- liners for greedy candidates ----------


def after_board(board: np.ndarray, a: Dict[str, Any], current: int) -> Optional[np.ndarray]:
    nb = clone(board)
    t = a["type"]
    if t == "stone":
        r, c = a["r"], a["c"]
        setv(nb, r, c, STONE)
        return nb
    if t == "swan":
        r, c = a["r"], a["c"]
        setv(nb, r, c, SWAN_SUN if current == SUN else SWAN_MOON)
        return nb
    if t == "move":
        swans = [(s["r"], s["c"]) for s in a["swans"]]
        dir_ = tuple(a["dir"])
        nb2 = simulate_move_subset(board, current, swans, dir_)
        return nb2
    if t == "push":
        swans = [(s["r"], s["c"]) for s in a["swans"]]
        dir_ = tuple(a["dir"])
        nb2 = simulate_push_subset(board, current, swans, dir_)
        return nb2
    return None


def freeze_delta_for_player(b_before: np.ndarray, a: Dict[str, Any], player: int) -> int:
    b2 = after_board(b_before, a, player)
    if b2 is None:
        return 0
    nb = clone(b2)
    res = freeze_encircled(nb)
    return (res.frozeMoon + res.sealedMoon) if player == SUN else (res.frozeSun + res.sealedSun)


def generate_greedy_candidates(b: np.ndarray, player: int, style_name: str = "doctrinal") -> List[Dict[str, Any]]:
    """
    Simple 1-ply look for opponent in oppHasFreezeInOne:
    stones, swan placements, single-swan moves (no pushes).
    """
    out: List[Dict[str, Any]] = []
    my_swans = active_swans_of(b, player)

    # stones
    for r, c in legal_stone_placements(b, player):
        b2 = clone(b)
        setv(b2, r, c, STONE)
        sc = evaluate_styled(b, b2, player, style_name)
        out.append({"type": "stone", "r": r, "c": c, "score": sc})

    # placements
    if count_active_swans(player, b) < 6:
        for r, c in legal_swan_placements(b, player):
            b2 = clone(b)
            setv(b2, r, c, SWAN_SUN if player == SUN else SWAN_MOON)
            sc = evaluate_styled(b, b2, player, style_name)
            out.append({"type": "swan", "r": r, "c": c, "score": sc})

    # single moves
    for r, c in my_swans:
        for dir_ in DIRS8:
            b2 = simulate_move_subset(b, player, [(r, c)], dir_)
            if b2 is None:
                continue
            sc = evaluate_styled(b, b2, player, style_name)
            out.append({"type": "move", "dir": dir_, "swans": [{"r": r, "c": c}], "score": sc})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ---------- main AI function ----------


def linith_ai(
    board: np.ndarray,
    current: int,
    difficulty: str = "hard",
    style_name: str = "doctrinal",
    debug: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pure Python port of linithAI(board, current, difficulty).
    Returns an action dict:
      { "type": "stone"/"swan"/"move"/"push", ... }
    """

    me = current
    opp = SUN if me == MOON else MOON

    # decide whether to log this move
    if debug is None:
        debug_enabled = DEBUG_HARD_AI
    else:
        debug_enabled = bool(debug)

    # ---- timing & stats ----
    t_start = time.time()
    stats = {
        "stones_tested": 0,
        "stone_advancing": 0,
        "swan_placements": 0,
        "subsets": 0,
        "move_candidates": 0,
    }

    def finalize(action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        t_end = time.time()
        elapsed = t_end - t_start
        if debug_enabled and elapsed > 0.3:
            print(
                f"[HARD AI] move {difficulty} took {elapsed:.3f}s | "
                f"stones_tested={stats['stones_tested']} "
                f"stone_advancing={stats['stone_advancing']} "
                f"swan_placements={stats['swan_placements']} "
                f"subsets={stats['subsets']} "
                f"move_candidates={stats['move_candidates']}",
                flush=True,
            )
        return action

    # ---- capability profiles ----
    if difficulty == "hard_train":
        CAP = {
            "MAX_SUBSET": 2,
            "LOCAL_R": 5,
            "MAX_STONES": 24,
            "BEAM": 40,
            "PROBE": 0,
            "MUST_TACTICS": True,
        }
    elif difficulty == "hard":
        CAP = {
            "MAX_SUBSET": 99,
            "LOCAL_R": 99,
            "MAX_STONES": 999,
            "BEAM": 999,
            "PROBE": 0,
            "MUST_TACTICS": True,
        }
    elif difficulty == "medium":
        CAP = {"MAX_SUBSET": 2, "LOCAL_R": 3, "MAX_STONES": 18, "BEAM": 10, "PROBE": 3, "MUST_TACTICS": True}
    else:  # easy
        CAP = {"MAX_SUBSET": 1, "LOCAL_R": 2, "MAX_STONES": 10, "BEAM": 6, "PROBE": 0, "MUST_TACTICS": True}

    SUBSET_LIMIT = 30 if difficulty == "hard_train" else 200

    def in_locality(r: int, c: int) -> bool:
        if CAP["LOCAL_R"] >= 90:
            return True
        coords = active_swans_of(board, me)
        for sr, sc in coords:
            if max(abs(sr - r), abs(sc - c)) <= CAP["LOCAL_R"]:
                return True
        return False

    cands: List[Dict[str, Any]] = []

    def push_stone(r: int, c: int, score: float):
        if stats["move_candidates"] >= CAP["BEAM"]:
            return
        cands.append({"type": "stone", "r": r, "c": c, "score": score})
        stats["move_candidates"] += 1

    def push_swan(r: int, c: int, score: float):
        if stats["move_candidates"] >= CAP["BEAM"]:
            return
        cands.append({"type": "swan", "r": r, "c": c, "score": score})
        stats["move_candidates"] += 1

    def push_move(subset: List[Tuple[int, int]], dir_: Tuple[int, int], score: float):
        if stats["move_candidates"] >= CAP["BEAM"]:
            return
        cands.append(
            {
                "type": "move",
                "dir": dir_,
                "swans": [{"r": r, "c": c} for (r, c) in subset],
                "score": score,
            }
        )
        stats["move_candidates"] += 1

    def push_push(subset: List[Tuple[int, int]], dir_: Tuple[int, int], score: float):
        if stats["move_candidates"] >= CAP["BEAM"]:
            return
        cands.append(
            {
                "type": "push",
                "dir": dir_,
                "swans": [{"r": r, "c": c} for (r, c) in subset],
                "score": score,
            }
        )
        stats["move_candidates"] += 1

    # precompute root libs for this board (used when scoring stones)
    root_my_lib = liberties_for(me, board)
    root_opp_lib = liberties_for(opp, board)

    # ----- unconditional tactical pre-pass: stone freeze/seal in one -----
    wins: List[Dict[str, Any]] = []
    for r, c in legal_stone_placements(board, current):
        b2 = clone(board)
        setv(b2, r, c, STONE)
        res = freeze_encircled(clone(b2))
        if me == SUN:
            enemy_sealed = res.sealedMoon
            enemy_frozen = res.frozeMoon + res.sealedMoon
        else:
            enemy_sealed = res.sealedSun
            enemy_frozen = res.frozeSun + res.sealedSun
        if enemy_frozen > 0:
            sc = (
                (1e9 if enemy_sealed > 0 else 1e7)
                + enemy_frozen * 1000.0
                + evaluate_styled(board, b2, me, style_name, root_my_lib, root_opp_lib)
            )
            wins.append({"type": "stone", "r": r, "c": c, "score": sc})
    if wins:
        wins.sort(key=lambda a: a["score"], reverse=True)
        return finalize(wins[0])

    # count total swans (active+frozen)
    my_total_swans = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = get(board, r, c)
            if same_player_swan(me, v):
                my_total_swans += 1

    # ----- stones -----
    all_stones = shuffled(legal_stone_placements(board, current))
    advancing: List[Tuple[int, int]] = []
    for r, c in all_stones:
        if len(advancing) >= CAP["MAX_STONES"]:
            break
        stats["stones_tested"] += 1
        if my_total_swans < 6 and not decisive_stone(board, r, c, me, style_name):
            continue
        b2 = clone(board)
        setv(b2, r, c, STONE)
        if stone_advances_game(board, b2, me):
            advancing.append((r, c))
            stats["stone_advancing"] += 1

    for r, c in advancing:
        if stats["move_candidates"] >= CAP["BEAM"]:
            break
        b2 = clone(board)
        setv(b2, r, c, STONE)
        sc = evaluate_styled(board, b2, me, style_name, root_my_lib, root_opp_lib)
        sc += 0.6 * enemy_ring_pressure(b2, me)
        prox = 0.0
        for dr, dc in DIRS8:
            rr, cc = r + dr, c + dc
            if inb(rr, cc) and get(board, rr, cc) != EMPTY:
                prox += 0.15
        push_stone(r, c, sc + prox)

    # ----- swan placements -----
    if my_total_swans < 6 and stats["move_candidates"] < CAP["BEAM"]:
        for r, c in shuffled(legal_swan_placements(board, me)):
            if stats["move_candidates"] >= CAP["BEAM"]:
                break
            if not in_locality(r, c):
                continue
            stats["swan_placements"] += 1
            b2 = clone(board)
            setv(b2, r, c, SWAN_SUN if me == SUN else SWAN_MOON)
            push_swan(r, c, evaluate_styled(board, b2, me, style_name))

    # ----- multi-swan moves -----
    coords = active_swans_of(board, me)
    if coords and stats["move_candidates"] < CAP["BEAM"]:
        subset_count = 0
        for subset in all_swan_subsets(coords):
            subset_count += 1
            stats["subsets"] += 1
            if subset_count > SUBSET_LIMIT:
                break
            if len(subset) > CAP["MAX_SUBSET"]:
                continue
            if not any(in_locality(r, c) for (r, c) in subset):
                continue
            for dir_ in DIRS8:
                if stats["move_candidates"] >= CAP["BEAM"]:
                    break
                b2 = simulate_move_subset(board, me, subset, dir_)
                if b2 is None:
                    continue
                push_move(subset, dir_, evaluate_styled(board, b2, me, style_name))

    # ----- pushes: subsets of ENEMY active swans -----
    enemy_coords = active_swans_of(board, opp)
    if enemy_coords and stats["move_candidates"] < CAP["BEAM"]:
        for subset in all_swan_subsets(enemy_coords):
            if len(subset) > CAP["MAX_SUBSET"]:
                continue
            if not any(in_locality(r, c) for (r, c) in subset):
                continue
            for dir_ in DIRS8:
                if stats["move_candidates"] >= CAP["BEAM"]:
                    break
                b2 = simulate_push_subset(board, me, subset, dir_)
                if b2 is None:
                    continue
                sc = evaluate_styled(board, b2, me, style_name)
                sc += 0.4 * enemy_ring_pressure(b2, me)
                push_push(subset, dir_, sc)

    # ----- no candidates fallbacks -----
    if not cands:
        if my_total_swans < 6:
            sp = [pos for pos in legal_swan_placements(board, me) if in_locality(*pos)]
            if sp:
                r, c = sp[0]
                return finalize({"type": "swan", "r": r, "c": c, "score": 0.0})
        for r, c in coords:
            for dir_ in DIRS8:
                b2 = simulate_move_subset(board, me, [(r, c)], dir_)
                if b2 is not None:
                    return finalize({"type": "move", "dir": dir_, "swans": [{"r": r, "c": c}], "score": 0.0})
        if all_stones:
            r, c = all_stones[0]
            return finalize({"type": "stone", "r": r, "c": c, "score": 0.0})
        return finalize(None)

    # ----- tactical layer -----

    def opp_has_freeze_in_one(b: np.ndarray) -> bool:
        # stones that immediately freeze or seal us
        for r, c in legal_stone_placements(b, opp):
            b2 = clone(b)
            setv(b2, r, c, STONE)
            res = freeze_encircled(clone(b2))
            if me == SUN:
                our_frozen = res.frozeSun + res.sealedSun
            else:
                our_frozen = res.frozeMoon + res.sealedMoon
            if our_frozen > 0:
                return True

        # greedy sample of other opponent actions
        opp_acts = generate_greedy_candidates(b, opp, style_name)[:12]
        for oa in opp_acts:
            delta = freeze_delta_for_player(b, oa, opp)
            if delta > 0:
                return True
        return False

    # winning now?
    winning_now = [a for a in cands if freeze_delta_for_player(board, a, me) > 0]
    if CAP["MUST_TACTICS"] and winning_now:
        winning_now.sort(key=lambda x: x["score"], reverse=True)
        return finalize(winning_now[0])

    # sorted candidates
    sorted_cands = sorted(cands, key=lambda x: x["score"], reverse=True)
    epsilon = 0.25
    best = sorted_cands[0]
    best_non_stone = next((a for a in sorted_cands if a["type"] != "stone"), None)
    if (
        best is not None
        and best["type"] == "stone"
        and best_non_stone is not None
        and (best["score"] - best_non_stone["score"]) <= epsilon
    ):
        top_best = best_non_stone
    else:
        top_best = best

    # defenders
    defenders: List[Dict[str, Any]] = []
    for a in sorted_cands:
        b2 = after_board(board, a, me)
        if b2 is None:
            continue
        if not opp_has_freeze_in_one(b2):
            defenders.append(a)

    if CAP["MUST_TACTICS"] and defenders and opp_has_freeze_in_one(board):
        defenders_sorted = sorted(defenders[: CAP["BEAM"]], key=lambda x: x["score"], reverse=True)
        return finalize(defenders_sorted[0])

    # Helper: avoid choosing a stone that makes our position worse if a non-stone with non-negative score exists
    def avoid_self_harm(pick: Optional[Dict[str, Any]], pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if pick is None:
            return pick
        if pick.get("type") != "stone" or pick.get("score", 0.0) >= 0.0:
            return pick
        for a in pool:
            if a.get("type") != "stone" and a.get("score", 0.0) >= 0.0:
                return a
        return pick

    # Additional guard: avoid any pick that newly gives the opponent a freeze-in-one,
    # when there exists a safe alternative in the current pool.
    def avoid_creating_tactic_loss(pick: Optional[Dict[str, Any]], pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if pick is None:
            return pick
        nb = after_board(board, pick, me)
        if nb is not None and opp_has_freeze_in_one(nb):
            for a in pool:
                b2 = after_board(board, a, me)
                if b2 is not None and not opp_has_freeze_in_one(b2):
                    return a
        return pick

    # ----- difficulty-specific selection -----
    if difficulty in ("hard", "hard_train"):
        safe_pick = avoid_creating_tactic_loss(avoid_self_harm(top_best, sorted_cands), sorted_cands)
        return finalize(safe_pick)

    if difficulty == "medium":
        beam = sorted_cands[: CAP["BEAM"]]
        best_by_probe = None
        best_probe_score = float("-inf")
        for i in range(min(len(beam), CAP["PROBE"])):
            a = beam[i]
            b2 = after_board(board, a, me)
            if b2 is None:
                continue
            opp_acts = generate_greedy_candidates(b2, opp, style_name)
            opp_best = opp_acts[0] if opp_acts else None
            final_score = -opp_best["score"] if opp_best else a["score"]
            if final_score > best_probe_score:
                best_probe_score = final_score
                best_by_probe = a
        if best_by_probe is not None and random.random() < 0.70:
            pick = best_by_probe
        else:
            pick = beam[0] if beam else top_best
        pool = beam if beam else sorted_cands
        safe_pick = avoid_creating_tactic_loss(avoid_self_harm(pick, pool), pool)
        return finalize(safe_pick)

    # easy
    E_BEAM = sorted_cands[: CAP["BEAM"]]
    pref = [a for a in E_BEAM if a["type"] != "move" or (a.get("swans") and len(a["swans"]) > 1)]
    if pref:
        idx = min(len(pref) - 1, int(len(pref) * 0.6))
        return finalize(pref[idx])
    if E_BEAM:
        idx = min(len(E_BEAM) - 1, int(len(E_BEAM) * 0.7))
        return finalize(E_BEAM[idx])
    return finalize(top_best)


# ---------- env wrapper ----------


import random
from typing import Any, Tuple
from linithrules import legal_push_moves, SUN, MOON


def choose_hard_move(env, difficulty: str = "hard") -> Tuple[str, Any, Any]:
    """
    Wrapper to use with LinithEnv:

      action = choose_hard_move(env)
      obs, reward, done, info = env.step(action)

    Returns an action tuple compatible with env.step:

      ("place_swan",  r, c)
      ("place_stone", r, c)
      ("move_group",  subset, dir)
      ("push",        (my_r, my_c), (enemy_r, enemy_c))

    Anti-repeat behaviour:

      - A player cannot play the exact same action tuple more than
        MAX_REPEAT_SAME_ACTION times in a row.
      - A player also cannot indefinitely repeat the same 2-step pattern
        (A,B,A,B,...) or 3-step pattern (A,B,C,A,B,C,...). If the chosen
        action would continue such a pattern, a different legal action is
        selected if possible.
    """
    assert env.state is not None
    board = env.state.board
    current = env.state.current_player

    MAX_REPEAT_SAME_ACTION = 3  # tweak as you like

    # ---------- helpers ----------

    def fallback_random_legal():
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("Hard AI has no legal actions")
        return random.choice(legal)

    def _would_cause_cycle(history, candidate) -> bool:
        """
        Given existing history (list of actions for this player) and a candidate
        action, check if appending candidate would create a repeating 2- or
        3-step pattern at the end of the history.
        """
        h2 = history + [candidate]
        n = len(h2)

        # 2-cycle: ... A, B, A, B
        if n >= 4:
            if h2[-1] == h2[-3] and h2[-2] == h2[-4]:
                return True

        # 3-cycle: ... A, B, C, A, B, C
        if n >= 6:
            if (
                h2[-1] == h2[-4] and
                h2[-2] == h2[-5] and
                h2[-3] == h2[-6]
            ):
                return True

        return False

    def anti_repeat(action):
        """
        Prevent:
          - Same action > MAX_REPEAT_SAME_ACTION times in a row for this player
          - Repeating ABAB or ABCABC cycles for this player.

        State is stored on the env instance.
        """
        # lazy init of tracking on env
        if not hasattr(env, "_hard_ai_last_action"):
            env._hard_ai_last_action = {SUN: None, MOON: None}
            env._hard_ai_repeat_count = {SUN: 0, MOON: 0}
            env._hard_ai_history = {SUN: [], MOON: []}

        last = env._hard_ai_last_action.get(current)
        rep = env._hard_ai_repeat_count.get(current, 0)
        history = env._hard_ai_history.get(current, [])

        def is_bad(candidate) -> bool:
            # same-action repetition check
            same = (candidate == last)
            too_many_same = same and (rep + 1 > MAX_REPEAT_SAME_ACTION)

            # cycle check (2- or 3-step)
            causes_cycle = _would_cause_cycle(history, candidate)

            return too_many_same or causes_cycle

        candidate = action

        if is_bad(candidate):
            # try to find an alternative legal action that is not "bad"
            legal = env.legal_actions()
            # randomise order so we don't always pick the first lexicographically
            random.shuffle(legal)
            for alt in legal:
                if alt != candidate and not is_bad(alt):
                    candidate = alt
                    break
            # if all legal moves are "bad", we keep the original candidate

        # update tracking for the chosen action
        if candidate == last:
            rep_new = rep + 1
        else:
            rep_new = 1

        env._hard_ai_last_action[current] = candidate
        env._hard_ai_repeat_count[current] = rep_new

        # update per-player history (keep last few entries)
        history.append(candidate)
        if len(history) > 8:
            history = history[-8:]
        env._hard_ai_history[current] = history

        return candidate

    # ---------- main mapping from linith_ai -> env action ----------

    a = linith_ai(board, current, difficulty=difficulty)

    if a is None:
        return anti_repeat(fallback_random_legal())

    t = a["type"]

    if t == "swan":
        action = ("place_swan", a["r"], a["c"])
        return anti_repeat(action)

    if t == "stone":
        action = ("place_stone", a["r"], a["c"])
        return anti_repeat(action)

    if t == "move":
        subset = [(s["r"], s["c"]) for s in a["swans"]]
        dir_ = tuple(a["dir"])
        action = ("move_group", tuple(subset), dir_)
        return anti_repeat(action)

    if t == "push":
        # AI push candidate stores the ENEMY swans in a["swans"] and the push
        # direction in a["dir"]. We need to convert that into (my_pos, enemy_pos)
        # as used by LinithEnv / simulate_push_move.

        swans = a.get("swans", [])
        if not swans:
            # Shouldn't happen, but fall back safely.
            legal = env.legal_actions()
            pushes = [act for act in legal if act[0] == "push"]
            if pushes:
                return anti_repeat(random.choice(pushes))
            if legal:
                return anti_repeat(random.choice(legal))
            raise RuntimeError("Hard AI produced empty push candidate and no legal moves")

        # Take the first enemy Swan in the pushed subset as the contact point.
        er, ec = swans[0]["r"], swans[0]["c"]
        dir_ = tuple(a["dir"])

        # Use rules-level generator to find the matching legal push and extract my_pos.
        for my_pos, enemy_pos, d in legal_push_moves(board, current):
            if enemy_pos == (er, ec) and d == dir_:
                action = ("push", my_pos, enemy_pos)
                return anti_repeat(action)

        # If we somehow didn't find an exact match, fall back to any legal push,
        # then to any legal move, rather than crashing.
        legal = env.legal_actions()
        pushes = [act for act in legal if act[0] == "push"]
        if pushes:
            return anti_repeat(random.choice(pushes))
        if legal:
            return anti_repeat(random.choice(legal))
        raise RuntimeError("Hard AI chose a push with no matching legal_push_moves and no legal fallback")

    # fallback for unknown types
    return anti_repeat(fallback_random_legal())
