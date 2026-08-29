// linith_selfplay_cpp.cpp
//
// C++ port of the Linith rules, environment, hard AI, action space,
// and Hard-vs-Hard self-play dataset generator, exposed via pybind11.
//
// Build as a Python extension named "linith_selfplay_cpp".

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <array>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <cstring>
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
namespace py = pybind11;

// ============================================================
//  Basic Linith rules and types (port of linithrules.py)
// ============================================================

constexpr int BOARD_SIZE = 10;

enum Cell {
    EMPTY       = 0,
    SWAN_SUN    = 1,
    SWAN_MOON   = 2,
    STONE       = 3,
    FROZEN_SUN  = 4,
    FROZEN_MOON = 5
};

constexpr int SUN  = 1;
constexpr int MOON = -1;

using Coord = std::pair<int,int>;

static const std::array<Coord,8> DIRS8 = {{
    {-1,-1}, {-1,0}, {-1,1},
    {0,-1},          {0,1},
    {1,-1},  {1,0},  {1,1}
}};

static const std::array<Coord,4> DIRS4 = {{
    {-1,0}, {1,0}, {0,-1}, {0,1}
}};

inline bool in_bounds(int r, int c) {
    return 0 <= r && r < BOARD_SIZE && 0 <= c && c < BOARD_SIZE;
}

using Board = std::array<int, BOARD_SIZE * BOARD_SIZE>;

inline int idx(int r, int c) { return r * BOARD_SIZE + c; }

inline int get_cell(const Board &b, int r, int c) {
    return b[idx(r,c)];
}
inline void set_cell(Board &b, int r, int c, int v) {
    b[idx(r,c)] = v;
}

inline bool is_swan(int v) {
    return v == SWAN_SUN || v == SWAN_MOON || v == FROZEN_SUN || v == FROZEN_MOON;
}

inline bool is_active_swan(int player, int v) {
    if (player == SUN) return v == SWAN_SUN;
    else              return v == SWAN_MOON;
}

inline bool same_player_swan(int player, int v) {
    if (player == SUN) {
        return v == SWAN_SUN || v == FROZEN_SUN;
    } else {
        return v == SWAN_MOON || v == FROZEN_MOON;
    }
}

inline bool enemy_swan(int player, int v) {
    if (player == SUN) {
        return v == SWAN_MOON || v == FROZEN_MOON;
    } else {
        return v == SWAN_SUN || v == FROZEN_SUN;
    }
}

inline int count_active_swans_int(int player, const Board &board) {
    int cnt = 0;
    int target = (player == SUN) ? SWAN_SUN : SWAN_MOON;
    for (int v : board) if (v == target) ++cnt;
    return cnt;
}

inline int count_total_swans_int(int player, const Board &board) {
    int a = (player == SUN) ? SWAN_SUN : SWAN_MOON;
    int f = (player == SUN) ? FROZEN_SUN : FROZEN_MOON;
    int cnt = 0;
    for (int v : board) if (v == a || v == f) ++cnt;
    return cnt;
}

inline bool any_empty(const Board &board) {
    for (int v : board) if (v == EMPTY) return true;
    return false;
}

// ------------------------------------------------------------
// "Free Swan" / silver-shield forbidden tiles
//
// A Swan belonging to `player` may NOT MOVE into any of the
// eight tiles surrounding an opponent's Swan that has no Stones
// in any of its own eight surrounding tiles.
//
// We build a boolean mask for tiles forbidden to `player`'s
// Swan moves. Note: this is for *voluntary* Swan moves only;
// pushes may ignore this mask.
// ------------------------------------------------------------
std::array<std::array<bool, BOARD_SIZE>, BOARD_SIZE>
forbidden_zone_mask_for_player(const Board &board, int player)
{
    std::array<std::array<bool, BOARD_SIZE>, BOARD_SIZE> mask{};
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c)
            mask[r][c] = false;

    int enemy_active = (player == SUN) ? SWAN_MOON   : SWAN_SUN;
    int enemy_frozen = (player == SUN) ? FROZEN_MOON : FROZEN_SUN;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int v = get_cell(board, r, c);
            if (v != enemy_active && v != enemy_frozen) continue;

            // Does this enemy Swan have any adjacent Stone?
            bool has_stone = false;
            for (auto [dr, dc] : DIRS8) {
                int nr = r + dr;
                int nc = c + dc;
                if (!in_bounds(nr, nc)) continue;
                if (get_cell(board, nr, nc) == STONE) {
                    has_stone = true;
                    break;
                }
            }
            if (has_stone) continue;

            // No adjacent Stone → mark its 8 neighbours as forbidden.
            for (auto [dr, dc] : DIRS8) {
                int nr = r + dr;
                int nc = c + dc;
                if (!in_bounds(nr, nc)) continue;
                mask[nr][nc] = true;
            }
        }
    }

    return mask;
}


// ---- placements ----

std::vector<Coord> legal_swan_placements(const Board &board, int player) {
    std::vector<Coord> placements;
    if (count_total_swans_int(player, board) >= 6)
        return placements;

    for (int r=0; r<BOARD_SIZE; ++r) {
        for (int c=0; c<BOARD_SIZE; ++c) {
            if (get_cell(board,r,c) != EMPTY) continue;

            bool has_adj_mine = false;
            bool has_adj_enemy = false;

            for (auto [dr,dc] : DIRS4) {
                int nr=r+dr, nc=c+dc;
                if (!in_bounds(nr,nc)) continue;
                int v = get_cell(board,nr,nc);
                if (!is_swan(v)) continue;
                if (same_player_swan(player,v)) has_adj_mine = true;
                else if (enemy_swan(player,v)) has_adj_enemy = true;
            }

            if (!has_adj_mine) continue;
            if (has_adj_enemy) continue;

            placements.emplace_back(r,c);
        }
    }
    return placements;
}

std::vector<Coord> legal_stone_placements(const Board &board, int player) {
    std::vector<Coord> adj_enemy;
    std::vector<Coord> frontier;
    std::vector<Coord> all_cells;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (get_cell(board, r, c) != EMPTY)
                continue;

            all_cells.emplace_back(r, c);

            bool near_any = false;
            for (auto [dr, dc] : DIRS8) {
                int nr = r + dr;
                int nc = c + dc;
                if (!in_bounds(nr,nc)) continue;

                int v = get_cell(board, nr, nc);
                if (v != EMPTY)
                    near_any = true;
                if (enemy_swan(player, v)) {
                    // prefer squares next to an enemy Swan
                    adj_enemy.emplace_back(r, c);
                    near_any = true;
                    break;
                }
            }
            if (near_any)
                frontier.emplace_back(r, c);
        }
    }

    if (!adj_enemy.empty())  return adj_enemy;
    if (!frontier.empty())   return frontier;
    return all_cells;
}

// ---- group moves ----

struct MoveLocalResult {
    std::set<Coord> stones_from;
    std::map<Coord,Coord> stones_to;
    bool valid;
};

MoveLocalResult legal_move_subset_local(
    const Board &board,
    const std::vector<Coord> &subset,
    Coord dir,
    int player,
    bool apply_forbidden = true)
{
    MoveLocalResult out;
    out.valid = false;

    int dr = dir.first;
    int dc = dir.second;

    // Movement restriction mask for this side's Swan moves
    std::array<std::array<bool, BOARD_SIZE>, BOARD_SIZE> forbidden{};
    if (apply_forbidden) {
        forbidden = forbidden_zone_mask_for_player(board, player);
    }


    std::set<int> moving;
    for (auto [r, c] : subset) {
        moving.insert(idx(r, c));
    }

    std::set<Coord>        stones_from;
    std::map<Coord,Coord>  stones_to;

    // collect stones pulled by moving subset
    for (auto [r, c] : subset) {
        for (auto [er, ec] : DIRS8) {
            int sr = r + er;
            int sc = c + ec;
            if (!in_bounds(sr, sc)) continue;
            if (get_cell(board, sr, sc) != STONE) continue;

            Coord stone(sr, sc);
            if (stones_from.count(stone)) continue;

            bool shared = false;
            for (auto [ar, ac] : DIRS8) {
                int xr = sr + ar;
                int xc = sc + ac;
                if (!in_bounds(xr, xc)) continue;

                int vv = get_cell(board, xr, xc);
                if (!is_swan(vv)) continue;

                // shared with enemy Swan
                if (enemy_swan(player, vv)) {
                    shared = true;
                    break;
                }

                // shared with unmoved friendly active Swan
                if (same_player_swan(player, vv) &&
                    is_active_swan(player, vv) &&
                    !moving.count(idx(xr, xc)))
                {
                    shared = true;
                    break;
                }
            }
            if (shared) continue;

            int tr = sr + dr;
            int tc = sc + dc;
            if (!in_bounds(tr, tc)) {
                // stone would leave the board → move invalid
                return out;
            }

            stones_from.insert(stone);
            stones_to[stone] = Coord(tr, tc);
        }
    }

    auto is_vacant_after_move = [&](int r, int c) -> bool {
        if (!in_bounds(r, c)) return false;
        int v = get_cell(board, r, c);
        if (v == EMPTY) return true;
        if (moving.count(idx(r, c))) return true;          // Swan vacates
        if (stones_from.count(Coord(r, c))) return true;   // Stone vacates
        return false;
    };

    // validate swan targets
    for (auto [r, c] : subset) {
        int nr = r + dr;
        int nc = c + dc;
        if (!in_bounds(nr, nc)) return out;

        // Free-Swan rule: only for voluntary moves, not pushes.
        if (apply_forbidden && forbidden[nr][nc]) {
            return out;
        }

        int occ = get_cell(board, nr, nc);
        if (occ == EMPTY) continue;

        if (is_swan(occ)) {
            // must be another Swan in the moving subset
            if (!moving.count(idx(nr, nc))) return out;
        } else if (occ == STONE) {
            // must be one of the stones we’re moving, and its destination must be vacant
            Coord src(nr, nc);
            auto it = stones_to.find(src);
            if (it == stones_to.end()) return out;
            Coord target = it->second;
            if (!is_vacant_after_move(target.first, target.second)) return out;
        } else {
            // cannot move into any other piece type
            return out;
        }
    }

    // validate stone targets (no collisions, all are truly vacant-after-move)
    std::set<Coord> seen_targets;
    for (auto &kv : stones_to) {
        Coord target = kv.second;
        if (!is_vacant_after_move(target.first, target.second)) return out;
        if (seen_targets.count(target)) return out;
        seen_targets.insert(target);
    }

    out.stones_from = stones_from;
    out.stones_to   = stones_to;
    out.valid       = true;
    return out;
}


Board simulate_group_move(
    const Board &board,
    const std::vector<Coord> &subset,
    Coord dir,
    int player,
    bool &ok,
    bool apply_forbidden = true)
{
    ok = false;
    MoveLocalResult res = legal_move_subset_local(
    board, subset, dir, player, apply_forbidden);
    if (!res.valid) {
        return board;
    }

    Board nb = board;
    int dr = dir.first, dc = dir.second;

    for (auto [r,c] : subset) set_cell(nb,r,c,EMPTY);
    for (auto [sr,sc] : res.stones_from) set_cell(nb,sr,sc,EMPTY);
    for (auto &kv : res.stones_to) {
        Coord t = kv.second;
        set_cell(nb,t.first,t.second,STONE);
    }

    int swan_code = (player==SUN) ? SWAN_SUN : SWAN_MOON;
    for (auto [r,c] : subset) {
        int nr=r+dr, nc=c+dc;
        set_cell(nb,nr,nc,swan_code);
    }

    ok = true;
    return nb;
}

std::vector<std::pair<std::vector<Coord>,Coord>>
legal_group_moves(const Board &board, int player, int max_group_size=6)
{
    std::vector<std::pair<std::vector<Coord>,Coord>> moves;

    std::vector<Coord> active;
    for (int r=0;r<BOARD_SIZE;++r)
        for (int c=0;c<BOARD_SIZE;++c) {
            int v=get_cell(board,r,c);
            if (is_active_swan(player,v)) active.emplace_back(r,c);
        }

    int n = (int)active.size();
    if (n==0) return moves;

    int total_masks = 1<<n;
    for (int mask=1; mask<total_masks; ++mask) {
        std::vector<Coord> subset;
        for (int i=0;i<n;++i)
            if (mask&(1<<i))
                subset.push_back(active[i]);
        if ((int)subset.size()>max_group_size) continue;

        for (auto dir : DIRS8) {
            MoveLocalResult res = legal_move_subset_local(board, subset, dir, player);
            if (res.valid) {
                moves.emplace_back(subset, dir);
            }
        }
    }

    return moves;
}

// ---- freezing & encirclement ----

struct FreezeResult {
    Board board;
    int froze_sun;
    int froze_moon;
    int sealed_sun;
    int sealed_moon;
};

std::map<int,std::vector<std::vector<Coord>>> collect_active_groups(const Board &board) {
    std::map<int,std::vector<std::vector<Coord>>> groups;
    groups[SUN] = {};
    groups[MOON] = {};

    std::array<std::array<bool,BOARD_SIZE>,BOARD_SIZE> seen{};
    for (auto &row : seen) row.fill(false);

    for (int r=0;r<BOARD_SIZE;++r) {
        for (int c=0;c<BOARD_SIZE;++c) {
            int v = get_cell(board,r,c);
            bool sun_active = is_active_swan(SUN,v);
            bool moon_active = is_active_swan(MOON,v);
            if (!sun_active && !moon_active) continue;
            if (seen[r][c]) continue;

            int owner = sun_active ? SUN : MOON;
            std::vector<Coord> comp;
            std::vector<Coord> stack;
            stack.emplace_back(r,c);
            seen[r][c] = true;

            while (!stack.empty()) {
                auto [x,y] = stack.back(); stack.pop_back();
                comp.emplace_back(x,y);
                for (auto [dr,dc] : DIRS8) {
                    int nx=x+dr, ny=y+dc;
                    if (!in_bounds(nx,ny) || seen[nx][ny]) continue;
                    if (is_active_swan(owner, get_cell(board,nx,ny))) {
                        seen[nx][ny] = true;
                        stack.emplace_back(nx,ny);
                    }
                }
            }

            groups[owner].push_back(std::move(comp));
        }
    }
    return groups;
}

bool group_encircled(const Board &board,
                     const std::vector<Coord> &comp,
                     int owner)
{
    std::set<Coord> inside(comp.begin(), comp.end());
    for (auto [r,c] : comp) {
        for (auto [dr,dc] : DIRS8) {
            int nr=r+dr, nc=c+dc;
            if (!in_bounds(nr,nc)) continue;
            if (inside.count(Coord(nr,nc))) continue;
            int v = get_cell(board,nr,nc);
            if (v==EMPTY) return false;
            if (same_player_swan(owner,v) && is_active_swan(owner,v)) return false;
        }
    }
    return true;
}

FreezeResult compute_freezes_on(const Board &board) {
    Board nb = board;
    auto groups = collect_active_groups(nb);

    int froze_sun=0, froze_moon=0, sealed_sun=0, sealed_moon=0;

    for (int owner : {SUN,MOON}) {
        for (auto &comp : groups[owner]) {
            if (!group_encircled(nb, comp, owner)) continue;

            int active_before = count_active_swans_int(owner, nb);
            for (auto [r,c] : comp) {
                int v = get_cell(nb,r,c);
                if (v==SWAN_SUN) set_cell(nb,r,c,FROZEN_SUN);
                else if (v==SWAN_MOON) set_cell(nb,r,c,FROZEN_MOON);
            }

            if ((int)comp.size() == active_before) {
                if (owner==SUN) sealed_sun += (int)comp.size();
                else            sealed_moon += (int)comp.size();
            } else {
                if (owner==SUN) froze_sun += (int)comp.size();
                else            froze_moon += (int)comp.size();
            }
        }
    }

    return FreezeResult{nb,froze_sun,froze_moon,sealed_sun,sealed_moon};
}

// ============================================================
//  Hard AI (port of hard_ai.py, using above rules)
// ============================================================

static bool DEBUG_HARD_AI = false;

void set_hard_ai_debug(bool enabled) {
    DEBUG_HARD_AI = enabled;
}

inline bool inb(int r,int c) { return in_bounds(r,c); }

inline int getB(const Board &b,int r,int c) { return get_cell(b,r,c); }
inline void setB(Board &b,int r,int c,int v) { set_cell(b,r,c,v); }

Board clone_board(const Board &b) { return b; }

// liberties_for, enemy ring pressure, etc. mirror hard_ai.py logic.

int count_active_swans_board(int p, const Board &b) {
    return count_active_swans_int(p,b);
}

int liberties_for_player(int p, const Board &b) {
    std::set<Coord> seen;
    for (int r=0;r<BOARD_SIZE;++r) {
        for (int c=0;c<BOARD_SIZE;++c) {
            int v = get_cell(b,r,c);
            if (!is_active_swan(p,v)) continue;
            for (auto [dr,dc] : DIRS8) {
                int nr=r+dr, nc=c+dc;
                if (!in_bounds(nr,nc)) continue;
                if (get_cell(b,nr,nc)==EMPTY)
                    seen.emplace(nr,nc);
            }
        }
    }
    return (int)seen.size();
}

double enemy_ring_pressure(const Board &b, int player) {
    int foe = (player==MOON)? SUN:MOON;

    auto is_enemy_active = [&](int v)->bool{
        return (foe==SUN && v==SWAN_SUN) || (foe==MOON && v==SWAN_MOON);
    };

    std::array<std::array<bool,BOARD_SIZE>,BOARD_SIZE> seen{};
    for (auto &row:seen) row.fill(false);
    std::vector<std::vector<Coord>> groups;

    for (int r=0;r<BOARD_SIZE;++r) {
        for (int c=0;c<BOARD_SIZE;++c) {
            int v=get_cell(b,r,c);
            if (!is_enemy_active(v) || seen[r][c]) continue;
            std::vector<Coord> Q;
            std::vector<Coord> comp;
            Q.emplace_back(r,c);
            seen[r][c]=true;
            while (!Q.empty()) {
                auto [x,y]=Q.back(); Q.pop_back();
                comp.emplace_back(x,y);
                for (auto [dr,dc] : DIRS8) {
                    int nx=x+dr, ny=y+dc;
                    if (!in_bounds(nx,ny)||seen[nx][ny]) continue;
                    if (is_enemy_active(get_cell(b,nx,ny))) {
                        seen[nx][ny]=true;
                        Q.emplace_back(nx,ny);
                    }
                }
            }
            groups.push_back(std::move(comp));
        }
    }

    double score=0.0;
    for (auto &comp : groups) {
        std::set<Coord> rim;
        for (auto [r,c] : comp) {
            for (auto [dr,dc] : DIRS8) {
                int nr=r+dr, nc=c+dc;
                if (!in_bounds(nr,nc)) continue;
                if (get_cell(b,nr,nc)==EMPTY)
                    rim.emplace(nr,nc);
            }
        }
        int k=(int)rim.size();
        if (k<=6) score += (6-k)*1.0;
        if (k<=3) score += 2.0;
        if (k<=1) score += 4.0;
    }
    return score;
}

bool both_at_six_board(const Board &b) {
    return (count_total_swans_int(SUN,b)>=6 &&
            count_total_swans_int(MOON,b)>=6);
}

bool has_friendly_pusher(const Board &board, int player, Coord enemy_swan) {
    for (auto [dr, dc] : DIRS8) {
        int r = enemy_swan.first + dr;
        int c = enemy_swan.second + dc;
        if (in_bounds(r, c) && is_active_swan(player, get_cell(board, r, c)))
            return true;
    }
    return false;
}

Board simulate_push_subset_board(
    const Board &board,
    int player,
    const std::vector<Coord> &subset,
    Coord direction,
    bool &ok)
{
    ok = false;
    if (subset.empty() || std::find(DIRS8.begin(), DIRS8.end(), direction) == DIRS8.end())
        return board;
    std::set<Coord> unique(subset.begin(), subset.end());
    if (unique.size() != subset.size()) return board;
    int enemy = (player == SUN) ? MOON : SUN;
    for (const Coord &coord : subset) {
        if (!in_bounds(coord.first, coord.second) ||
            !is_active_swan(enemy, get_cell(board, coord.first, coord.second)) ||
            !has_friendly_pusher(board, player, coord))
            return board;
    }
    return simulate_group_move(
        board, subset, direction, enemy, ok, /*apply_forbidden=*/false
    );
}

std::vector<std::pair<std::vector<Coord>, Coord>> legal_push_moves_board(
    const Board &board,
    int player)
{
    int enemy = (player == SUN) ? MOON : SUN;
    std::vector<Coord> candidates;
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Coord coord(r, c);
            if (is_active_swan(enemy, get_cell(board, r, c)) &&
                has_friendly_pusher(board, player, coord))
                candidates.push_back(coord);
        }

    std::vector<std::pair<std::vector<Coord>, Coord>> result;
    for (int mask = 1; mask < (1 << static_cast<int>(candidates.size())); ++mask) {
        std::vector<Coord> subset;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
            if (mask & (1 << i)) subset.push_back(candidates[i]);
        for (Coord direction : DIRS8) {
            bool ok = false;
            simulate_push_subset_board(board, player, subset, direction, ok);
            if (ok) result.emplace_back(subset, direction);
        }
    }
    return result;
}

bool has_any_legal_action_board(const Board &board, int player) {
    // Any legal Swan placement?
    if (!legal_swan_placements(board, player).empty()) {
        return true;
    }

    // Any legal Stone placement?
    if (!legal_stone_placements(board, player).empty()) {
        return true;
    }

    // Any legal group move?
    auto gm = legal_group_moves(board, player, /*max_group_size=*/6);
    if (!gm.empty()) {
        return true;
    }

    if (!legal_push_moves_board(board, player).empty()) {
        return true;
    }

    return false;
}


struct FreezeResultSimple {
    int frozeSun=0;
    int frozeMoon=0;
    int sealedSun=0;
    int sealedMoon=0;
};

FreezeResultSimple freeze_encircled_board(Board &b) {
    auto fr = compute_freezes_on(b);
    FreezeResultSimple out;
    out.frozeSun = fr.froze_sun;
    out.frozeMoon= fr.froze_moon;
    out.sealedSun= fr.sealed_sun;
    out.sealedMoon=fr.sealed_moon;
    b = fr.board;
    return out;
}

bool stone_advances_game(const Board &b_before, const Board &b_after, int player) {
    Board nb = clone_board(b_after);
    auto res = freeze_encircled_board(nb);
    int froze_enemy = (player==SUN) ? (res.frozeMoon+res.sealedMoon)
                                    : (res.frozeSun+res.sealedSun);
    if (froze_enemy>0) return true;

    int opp = (player==MOON)?SUN:MOON;

    int my_lib_before  = liberties_for_player(player,b_before);
    int opp_lib_before = liberties_for_player(opp,b_before);
    int my_lib_after   = liberties_for_player(player,nb);
    int opp_lib_after  = liberties_for_player(opp,nb);

    if (opp_lib_after <= opp_lib_before-1) return true;
    if (my_lib_after  >= my_lib_before +1) return true;

    double pr_before = enemy_ring_pressure(b_before,player);
    double pr_after  = enemy_ring_pressure(nb,player);
    if (pr_after >= pr_before+1.0) return true;

    return false;
}

// ----- territory helpers (space evaluation) -----

bool is_enemy_swan_naked_global(const Board &b, int r, int c, int player) {
    int v = get_cell(b, r, c);
    if (!enemy_swan(player, v)) return false;
    // if any adjacent stone, not naked
    for (auto [dr, dc] : DIRS8) {
        int nr = r + dr;
        int nc = c + dc;
        if (!in_bounds(nr, nc)) continue;
        if (get_cell(b, nr, nc) == STONE) return false;
    }
    return true;
}

bool is_in_naked_enemy_zone_global(const Board &b, int r, int c, int player) {
    for (auto [dr, dc] : DIRS8) {
        int er = r + dr;
        int ec = c + dc;
        if (!in_bounds(er, ec)) continue;
        if (is_enemy_swan_naked_global(b, er, ec, player)) return true;
    }
    return false;
}

double territory_advantage(const Board &b, int current) {
    const int OPP = (current == SUN ? MOON : SUN);
    const int INF  = 99;
    const int MAXD = 6;   // depth horizon in Swan moves
    const int EDGE = 10;  // strong territory bonus per safe tile
    const int SCALE = 2;  // distance differential scaling

    using DistGrid = std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>;

    auto bfs = [&](int player) -> DistGrid {
        DistGrid dist;
        for (auto &row : dist) row.fill(INF);

        std::vector<Coord> q;
        q.reserve(BOARD_SIZE * BOARD_SIZE);

        // seed: empty neighbours of active Swans for this player
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                int v = get_cell(b, r, c);
                if (!is_active_swan(player, v)) continue;

                for (auto [dr, dc] : DIRS8) {
                    int nr = r + dr;
                    int nc = c + dc;
                    if (!in_bounds(nr, nc)) continue;
                    if (get_cell(b, nr, nc) != EMPTY) continue;
                    if (is_in_naked_enemy_zone_global(b, nr, nc, player)) continue;
                    if (dist[nr][nc] > 1) {
                        dist[nr][nc] = 1;
                        q.emplace_back(nr, nc);
                    }
                }
            }
        }

        // flood-fill over empty tiles, respecting naked enemy zones
        std::size_t qi = 0;
        while (qi < q.size()) {
            auto [r, c] = q[qi++];
            int d = dist[r][c];
            if (d >= MAXD) continue; // don't expand too far

            for (auto [dr, dc] : DIRS8) {
                int nr = r + dr;
                int nc = c + dc;
                if (!in_bounds(nr, nc)) continue;
                if (get_cell(b, nr, nc) != EMPTY) continue;
                if (is_in_naked_enemy_zone_global(b, nr, nc, player)) continue;
                if (dist[nr][nc] > d + 1) {
                    dist[nr][nc] = d + 1;
                    q.emplace_back(nr, nc);
                }
            }
        }
        return dist;
    };

    DistGrid dMe  = bfs(current);
    DistGrid dOpp = bfs(OPP);

    double score = 0.0;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (get_cell(b, r, c) != EMPTY) continue;

            int dm  = dMe[r][c];
            int dOp = dOpp[r][c];

            bool meFar  = (dm  > MAXD);
            bool oppFar = (dOp > MAXD);

            if (meFar && oppFar) continue; // nobody reaches soon

            // treat unreachable as MAXD+1 so it's valuable but not explosive
            if (dm  > MAXD) dm  = MAXD + 1;
            if (dOp > MAXD) dOp = MAXD + 1;

            if (dm <= MAXD && dOp == MAXD + 1) {
                // strong safe territory for current
                score += EDGE;
            } else if (dOp <= MAXD && dm == MAXD + 1) {
                // strong safe territory for opponent
                score -= EDGE;
            } else {
                // both can reach within horizon: compare distances
                int diff = dOp - dm; // >0 means we're closer
                score += diff * SCALE;
            }
        }
    }

    return score;
}


// -----------------------------------------
// Tunable evaluation weights
// -----------------------------------------

struct EvalWeights {
    double wFreeze;
    double wSelfFreeze;
    double wMyLib;
    double wOpLib;
    double wRing;
    double wMomentum;
    double wSpace;
    double freeze_phase_default;
    double freeze_phase_blizzard;
    double ring_phase_default;
    double ring_phase_fortress;
};

// Default = JS Hard AI values
static EvalWeights g_default_eval_weights {
    500.0,    // wFreeze
   -600.0,    // wSelfFreeze
      5.0,    // wMyLib
     -9.0,    // wOpLib
      0.0,    // wRing
     10.0,    // wMomentum (bothAtSix)
      0.0,    // wSpace (JS default)
      0.12,   // freeze_phase_default
      0.25,   // freeze_phase_blizzard
      0.08,   // ring_phase_default
      0.25    // ring_phase_fortress
};

// Tunable weights (start identical)
static EvalWeights g_tuned_eval_weights = g_default_eval_weights;

// Thread-local pointer to active weights
static thread_local EvalWeights* g_current_eval_weights = &g_default_eval_weights;

// RAII guard for switching persona
struct EvalScopeGuard {
    EvalWeights* prev;
    explicit EvalScopeGuard(EvalWeights* w) : prev(g_current_eval_weights) {
        g_current_eval_weights = w;
    }
    ~EvalScopeGuard() {
        g_current_eval_weights = prev;
    }
};

// Python exposure helpers
EvalWeights get_default_eval_weights_cpp() { return g_default_eval_weights; }
EvalWeights get_tuned_eval_weights_cpp()   { return g_tuned_eval_weights; }
void set_tuned_eval_weights_cpp(const EvalWeights& w) { g_tuned_eval_weights = w; }
void reset_tuned_eval_weights_cpp() { g_tuned_eval_weights = g_default_eval_weights; }


double evaluate_styled(const Board &b_before,
                       const Board &b_after,
                       int current,
                       const std::string &style_name)
{
    // use the currently-active weight set (default or tuned)
    const EvalWeights& W = *g_current_eval_weights;

    Board nb = clone_board(b_after);
    auto res = freeze_encircled_board(nb);

    int my_active_after = count_active_swans_board(current, nb);
    int opp             = (current == MOON) ? SUN : MOON;
    int opp_active_after= count_active_swans_board(opp, nb);

    if (opp_active_after == 0 && my_active_after > 0) return 1e9;
    if (my_active_after == 0 && opp_active_after > 0) return -1e9;

    int my_lib_before  = liberties_for_player(current, b_before);
    int opp_lib_before = liberties_for_player(opp,     b_before);
    int my_lib_after   = liberties_for_player(current, nb);
    int opp_lib_after  = liberties_for_player(opp,     nb);

    int my_delta  = my_lib_after  - my_lib_before;
    int opp_delta = opp_lib_after - opp_lib_before;

    int froze_gain = (current == SUN)
                     ? (res.frozeMoon + res.sealedMoon)
                     : (res.frozeSun  + res.sealedSun);
    int self_loss  = (current == SUN)
                     ? (res.frozeSun  + res.sealedSun)
                     : (res.frozeMoon + res.sealedMoon);

    double ring = enemy_ring_pressure(nb, current);

    // momentum term: only active when both are at six
    double momentum = both_at_six_board(b_before) ? W.wMomentum : 0.0;

    int tot_frozen = (res.frozeSun + res.sealedSun) +
                     (res.frozeMoon + res.sealedMoon);
    double phase = std::max(0.0, std::min(1.0, tot_frozen / 6.0));

    double freeze_boost =
        1.0 + ((style_name == "blizzard")
               ? (W.freeze_phase_blizzard * phase)
               : (W.freeze_phase_default  * phase));

    double ring_boost =
        1.0 + ((style_name == "fortress")
               ? (W.ring_phase_fortress * phase)
               : (W.ring_phase_default  * phase));

    double terr_delta = 0.0;
    if (W.wSpace != 0.0) {
        double terr_before = territory_advantage(b_before, current);
        double terr_after  = territory_advantage(nb,       current);
        terr_delta = terr_after - terr_before;
    }

    return froze_gain * (W.wFreeze * freeze_boost)
         + self_loss  * (W.wSelfFreeze)
         + my_delta   * (W.wMyLib)
         + opp_delta  * (W.wOpLib)
         + ring       * (W.wRing * ring_boost)
         + momentum   * froze_gain
         + terr_delta * W.wSpace;
}



bool decisive_stone(const Board &b, int r,int c, int me,
                    const std::string &style_name="doctrinal")
{
    Board b2 = clone_board(b);
    setB(b2,r,c,STONE);
    Board nb = clone_board(b2);
    auto res = freeze_encircled_board(nb);
    int froze_enemy = (me==SUN)
                      ? (res.frozeMoon+res.sealedMoon)
                      : (res.frozeSun+res.sealedSun);
    if (froze_enemy>0) return true;

    int opp = (me==MOON)?SUN:MOON;
    int opp_lib_before = liberties_for_player(opp,b);
    int opp_lib_after  = liberties_for_player(opp,nb);
    if (opp_lib_after <= opp_lib_before-3) return true;

    double pr_before = enemy_ring_pressure(b,me);
    double pr_after  = enemy_ring_pressure(nb,me);
    if (pr_after >= pr_before+3.0) return true;

    return false;
}

std::vector<Coord> active_swans_of(const Board &b, int p) {
    std::vector<Coord> arr;
    for (int r=0;r<BOARD_SIZE;++r)
        for (int c=0;c<BOARD_SIZE;++c) {
            int v=get_cell(b,r,c);
            if (is_active_swan(p,v)) arr.emplace_back(r,c);
        }
    return arr;
}

std::vector<std::vector<Coord>> all_swan_subsets(const std::vector<Coord> &coords) {
    std::vector<std::vector<Coord>> out;
    int n=(int)coords.size();
    int total=1<<n;
    for (int mask=1; mask<total; ++mask) {
        std::vector<Coord> subset;
        for (int i=0;i<n;++i)
            if (mask&(1<<i))
                subset.push_back(coords[i]);
        out.push_back(std::move(subset));
    }
    return out;
}

struct MoveAction {
    // type: "stone", "swan", "move", "push"
    std::string type;
    int r=0, c=0;          // placement square where applicable
    Coord dir{0,0};        // movement direction (for move/push)
    std::vector<Coord> swans;  // for move: our subset; for push: enemy Swan(s)
    double score=0.0;
};

Board after_board(const Board &board, const MoveAction &a, int current) {
    Board nb = clone_board(board);
    if (a.type=="stone") {
        setB(nb,a.r,a.c,STONE);
        return nb;
    } else if (a.type=="swan") {
        setB(nb,a.r,a.c, (current==SUN)?SWAN_SUN:SWAN_MOON);
        return nb;
    } else if (a.type=="move") {
        bool ok=false;
        Board nb2 = simulate_group_move(board,a.swans,a.dir,current,ok);
        if (!ok) return board; // caller must check
        return nb2;
    } else if (a.type=="push") {
        if (a.swans.empty()) return board;
        bool ok=false;
        Board nb2 = simulate_push_subset_board(board, current, a.swans, a.dir, ok);
        if (!ok) return board;
        return nb2;
    }
    return board;
}

int freeze_delta_for_player(const Board &b_before, const MoveAction &a, int player) {
    Board b2 = after_board(b_before,a,player);
    if (&b2 == &b_before) return 0; // if we used invalid sentinel
    Board nb = clone_board(b2);
    auto res = freeze_encircled_board(nb);
    return (player==SUN)
           ? (res.frozeMoon+res.sealedMoon)
           : (res.frozeSun+res.sealedSun);
}

std::vector<MoveAction> generate_greedy_candidates(
    const Board &b,
    int player,
    const std::string &style_name = "doctrinal")
{
    std::vector<MoveAction> out;
    auto my_swans = active_swans_of(b,player);

    // stones
    for (auto [r,c] : legal_stone_placements(b, player)) {
        Board b2 = clone_board(b);
        setB(b2, r, c, STONE);
        double sc = evaluate_styled(b, b2, player, style_name);
        MoveAction a;
        a.type = "stone";
        a.r = r; a.c = c;
        a.score = sc;
        out.push_back(std::move(a));
    }

    // placements
    if (count_active_swans_board(player,b)<6) {
        for (auto [r,c] : legal_swan_placements(b,player)) {
            Board b2 = clone_board(b);
            setB(b2,r,c,(player==SUN)?SWAN_SUN:SWAN_MOON);
            double sc = evaluate_styled(b,b2,player,style_name);
            MoveAction a;
            a.type="swan"; a.r=r; a.c=c; a.score=sc;
            out.push_back(std::move(a));
        }
    }

    // single moves
    for (auto [r,c] : my_swans) {
        for (auto dir : DIRS8) {
            bool ok=false;
            Board b2 = simulate_group_move(b, {Coord(r,c)}, dir, player, ok);
            if (!ok) continue;
            double sc = evaluate_styled(b,b2,player,style_name);
            MoveAction a;
            a.type="move"; a.dir=dir; a.swans={Coord(r,c)}; a.score=sc;
            out.push_back(std::move(a));
        }
    }

    // Pushes may move any eligible enemy subset in any common direction.
    for (const auto &[subset, dir] : legal_push_moves_board(b, player)) {
        bool ok=false;
        Board b2 = simulate_push_subset_board(b, player, subset, dir, ok);
        if (!ok) continue;
        MoveAction a;
        a.type="push";
        a.dir=dir;
        a.swans=subset;
        a.score=evaluate_styled(b,b2,player,style_name);
        out.push_back(std::move(a));
    }

    std::sort(out.begin(),out.end(),
              [](const MoveAction &x,const MoveAction &y){return x.score>y.score;});
    return out;
}

// ----- main AI selection: linith_ai, choose_hard_move -----
//
// To keep this manageable in one file, we implement the same logic
// but in a compact C++ style. Behaviour should closely mirror
// hard_ai.linith_ai for "hard", "hard_train", "medium", "easy".

MoveAction linith_ai(
    const Board &board,
    int current,
    const std::string &difficulty="hard",
    const std::string &style_name="doctrinal",
    bool debug_flag = false)
{
    int me = current;
    int opp = (me==MOON)?SUN:MOON;
    bool debug_enabled = debug_flag || DEBUG_HARD_AI;

    EvalScopeGuard guard(
    (difficulty == "very_hard")
        ? &g_tuned_eval_weights
        : &g_default_eval_weights
);

    auto t_start = std::chrono::steady_clock::now();

    struct Stats {
        int stones_tested=0;
        int stone_advancing=0;
        int swan_placements=0;
        int subsets=0;
        int move_candidates=0;
    } stats;

    auto finalize = [&](const MoveAction &action)->MoveAction{
        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end-t_start).count();
        if (debug_enabled && elapsed>0.3) {
            std::cout << "[HARD AI] move " << difficulty << " took " << elapsed
                      << "s | stones_tested=" << stats.stones_tested
                      << " stone_advancing=" << stats.stone_advancing
                      << " swan_placements=" << stats.swan_placements
                      << " subsets=" << stats.subsets
                      << " move_candidates=" << stats.move_candidates
                      << std::endl;
        }
        return action;
    };

    struct Cap {
        int MAX_SUBSET;
        int LOCAL_R;
        int MAX_STONES;
        int BEAM;
        int PROBE;
        bool MUST_TACTICS;
    } CAP;


    if (difficulty=="hard" || difficulty=="very_hard") {
        CAP = {99,99,999,2048,0,true};
    } else if (difficulty=="hard_train") {
        CAP = {2,5,24,40,0,true};
    } else if (difficulty=="medium") {
        CAP = {2,3,18,10,3,true};
    } else { // easy
        CAP = {1,2,10,6,0,true};
    }

    int SUBSET_LIMIT = (difficulty=="hard_train") ? 30 : 200;

    auto in_locality = [&](int r,int c)->bool{
        if (CAP.LOCAL_R>=90) return true;
        auto coords = active_swans_of(board,me);
        for (auto [sr,sc] : coords)
            if (std::max(std::abs(sr-r), std::abs(sc-c)) <= CAP.LOCAL_R)
                return true;
        return false;
    };

    std::vector<MoveAction> cands;

    auto push_stone = [&](int r,int c,double score){
        if (stats.move_candidates>=CAP.BEAM) return;
        MoveAction a; a.type="stone"; a.r=r; a.c=c; a.score=score;
        cands.push_back(std::move(a));
        stats.move_candidates++;
    };
    auto push_swan = [&](int r,int c,double score){
        if (stats.move_candidates>=CAP.BEAM) return;
        MoveAction a; a.type="swan"; a.r=r; a.c=c; a.score=score;
        cands.push_back(std::move(a));
        stats.move_candidates++;
    };
    auto push_move = [&](const std::vector<Coord> &subset, Coord dir, double score){
        if (stats.move_candidates>=CAP.BEAM) return;
        MoveAction a; a.type="move"; a.dir=dir; a.swans=subset; a.score=score;
        cands.push_back(std::move(a));
        stats.move_candidates++;
    };
    auto push_push = [&](const std::vector<Coord> &subset, Coord dir, double score){
        if (stats.move_candidates>=CAP.BEAM) return;
        MoveAction a;
        a.type = "push";
        a.dir = dir;
        a.swans = subset;
        a.score = score;
        cands.push_back(std::move(a));
        stats.move_candidates++;
    };

    // tactical pre-pass: stones freezing/sealing in one
    std::vector<MoveAction> wins;
    for (auto [r,c] : legal_stone_placements(board, me)) {
        Board b2 = clone_board(board);
        setB(b2, r, c, STONE);
        Board nb = clone_board(b2);
        auto res = freeze_encircled_board(nb);
        int enemy_sealed = (me==SUN)?res.sealedMoon:res.sealedSun;
        int enemy_frozen = (me==SUN)?(res.frozeMoon+res.sealedMoon)
                                    :(res.frozeSun+res.sealedSun);
        if (enemy_frozen>0) {
            double sc = ((enemy_sealed>0)?1e9:1e7)
                      + enemy_frozen*1000.0
                      + evaluate_styled(board,b2,me,style_name);
            MoveAction a; a.type="stone"; a.r=r; a.c=c; a.score=sc;
            wins.push_back(std::move(a));
        }
    }
    if (!wins.empty()) {
        std::sort(wins.begin(),wins.end(),
                  [](const MoveAction &x,const MoveAction &y){return x.score>y.score;});
        return finalize(wins[0]);
    }

    int my_total_swans=0;
    for (int r=0;r<BOARD_SIZE;++r)
        for (int c=0;c<BOARD_SIZE;++c)
            if (same_player_swan(me,get_cell(board,r,c)))
                my_total_swans++;

    // stones
    std::vector<Coord> all_stones = legal_stone_placements(board, me);
    // random shuffle for variety
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all_stones.begin(),all_stones.end(),rng);

    std::vector<Coord> advancing;
    for (auto [r,c] : all_stones) {
        if (!in_locality(r,c)) continue;
        if ((int)advancing.size()>=CAP.MAX_STONES) break;
        stats.stones_tested++;
        if (my_total_swans<6 && !decisive_stone(board,r,c,me,style_name))
            continue;
        Board b2 = clone_board(board);
        setB(b2,r,c,STONE);
        if (stone_advances_game(board,b2,me)) {
            advancing.emplace_back(r,c);
            stats.stone_advancing++;
        }
    }

    for (auto [r,c] : advancing) {
        if (stats.move_candidates>=CAP.BEAM) break;
        Board b2 = clone_board(board);
        setB(b2,r,c,STONE);
        double sc = evaluate_styled(board,b2,me,style_name);
        sc += 0.6*enemy_ring_pressure(b2,me);
        double prox=0.0;
        for (auto [dr,dc] : DIRS8) {
            int rr=r+dr, cc=c+dc;
            if (in_bounds(rr,cc) && get_cell(board,rr,cc)!=EMPTY)
                prox += 0.15;
        }
        push_stone(r,c,sc+prox);
    }

    // swan placements
    if (my_total_swans<6 && stats.move_candidates<CAP.BEAM) {
        auto sp = legal_swan_placements(board,me);
        std::shuffle(sp.begin(),sp.end(),rng);
        for (auto [r,c] : sp) {
            if (stats.move_candidates>=CAP.BEAM) break;
            if (!in_locality(r,c)) continue;
            stats.swan_placements++;
            Board b2 = clone_board(board);
            setB(b2,r,c,(me==SUN)?SWAN_SUN:SWAN_MOON);
            push_swan(r,c,evaluate_styled(board,b2,me,style_name));
        }
    }

    // multi-swan moves
    auto coords = active_swans_of(board,me);
    if (!coords.empty() && stats.move_candidates<CAP.BEAM) {
        int subset_count=0;
        auto subsets = all_swan_subsets(coords);
        for (auto &subset : subsets) {
            subset_count++;
            stats.subsets++;
            if (subset_count>SUBSET_LIMIT) break;
            if ((int)subset.size()>CAP.MAX_SUBSET) continue;
            bool any_local=false;
            for (auto [r,c] : subset)
                if (in_locality(r,c)) {any_local=true;break;}
            if (!any_local) continue;
            for (auto dir : DIRS8) {
                if (stats.move_candidates>=CAP.BEAM) break;
                bool ok=false;
                Board b2 = simulate_group_move(board,subset,dir,me,ok);
                if (!ok) continue;
                push_move(subset,dir,evaluate_styled(board,b2,me,style_name));
            }
        }
    }

    // push moves: any eligible enemy subset in any common direction
    if (stats.move_candidates < CAP.BEAM) {
        for (const auto &[subset, dir] : legal_push_moves_board(board, me)) {
            if (stats.move_candidates >= CAP.BEAM) break;
            bool local = std::any_of(subset.begin(), subset.end(), [&](Coord p) {
                return in_locality(p.first, p.second);
            });
            if (!local) continue;
            bool ok = false;
            Board b2 = simulate_push_subset_board(board, me, subset, dir, ok);
            if (!ok) continue;
            push_push(subset, dir, evaluate_styled(board, b2, me, style_name));
        }
    }


    if (cands.empty()) {
        if (my_total_swans<6) {
            auto sp = legal_swan_placements(board,me);
            std::vector<Coord> filtered;
            for (auto &p : sp) if (in_locality(p.first,p.second)) filtered.push_back(p);
            if (!filtered.empty()) {
                MoveAction a; a.type="swan"; a.r=filtered[0].first; a.c=filtered[0].second; a.score=0.0;
                return finalize(a);
            }
        }
        for (auto [r,c] : coords) {
            for (auto dir : DIRS8) {
                bool ok=false;
                Board b2 = simulate_group_move(board,{Coord(r,c)},dir,me,ok);
                if (ok) {
                    MoveAction a; a.type="move"; a.dir=dir; a.swans={Coord(r,c)}; a.score=0.0;
                    return finalize(a);
                }
            }
        }
        if (!all_stones.empty()) {
            MoveAction a; a.type="stone"; a.r=all_stones[0].first; a.c=all_stones[0].second; a.score=0.0;
            return finalize(a);
        }
        MoveAction none; none.type="none"; return finalize(none);
    }

    auto opp_has_freeze_in_one = [&](const Board &b)->bool{
        auto ls = legal_stone_placements(b, opp);
        for (auto [r,c] : ls) {
            Board b2 = clone_board(b);
            setB(b2, r, c, STONE);
            Board nb = clone_board(b2);
            auto res = freeze_encircled_board(nb);
            int our_frozen = (me==SUN)?(res.frozeSun+res.sealedSun)
                                      :(res.frozeMoon+res.sealedMoon);
            if (our_frozen>0) return true;
        }
        auto opp_acts = generate_greedy_candidates(b,opp,style_name);
        if ((int)opp_acts.size()>12) opp_acts.resize(12);
        for (auto &oa : opp_acts) {
            int delta = freeze_delta_for_player(b,oa,opp);
            if (delta>0) return true;
        }
        return false;
    };

    std::vector<MoveAction> winning_now;
    for (auto &a : cands) {
        int delta = freeze_delta_for_player(board,a,me);
        if (delta>0) winning_now.push_back(a);
    }
    if (CAP.MUST_TACTICS && !winning_now.empty()) {
        std::sort(winning_now.begin(),winning_now.end(),
                  [](const MoveAction &x,const MoveAction &y){return x.score>y.score;});
        return finalize(winning_now[0]);
    }

    std::vector<MoveAction> sorted_cands = cands;
    std::sort(sorted_cands.begin(),sorted_cands.end(),
              [](const MoveAction &x,const MoveAction &y){return x.score>y.score;});

    double epsilon = 0.25;
    MoveAction best = sorted_cands[0];
    MoveAction *best_non_stone_ptr = nullptr;
    for (auto &a : sorted_cands) {
        if (a.type!="stone") { best_non_stone_ptr=&a; break;}
    }
    MoveAction top_best = best;
    if (best.type=="stone" && best_non_stone_ptr &&
        (best.score - best_non_stone_ptr->score)<=epsilon)
        top_best = *best_non_stone_ptr;

    std::vector<MoveAction> defenders;
    for (auto &a : sorted_cands) {
        Board b2 = after_board(board,a,me);
        if (opp_has_freeze_in_one(b2)) continue;
        defenders.push_back(a);
    }

    if (CAP.MUST_TACTICS && !defenders.empty() && opp_has_freeze_in_one(board)) {
        if ((int)defenders.size()>CAP.BEAM) defenders.resize(CAP.BEAM);
        std::sort(defenders.begin(),defenders.end(),
                  [](const MoveAction &x,const MoveAction &y){return x.score>y.score;});
        return finalize(defenders[0]);
    }

    auto avoid_self_harm = [&](const MoveAction &pick,
                               const std::vector<MoveAction> &pool)->MoveAction{
        if (pick.type!="stone" || pick.score>=0.0) return pick;
        for (auto &a : pool)
            if (a.type!="stone" && a.score>=0.0)
                return a;
        return pick;
    };

    auto avoid_creating_tactic_loss = [&](const MoveAction &pick,
                                          const std::vector<MoveAction> &pool)->MoveAction{
        Board nb = after_board(board,pick,me);
        if (opp_has_freeze_in_one(nb)) {
            for (auto &a : pool) {
                Board b2 = after_board(board,a,me);
                if (!opp_has_freeze_in_one(b2))
                    return a;
            }
        }
        return pick;
    };

    if (difficulty=="hard" || difficulty=="hard_train") {
        MoveAction safe = avoid_creating_tactic_loss(
            avoid_self_harm(top_best,sorted_cands), sorted_cands);
        return finalize(safe);
    }

    if (difficulty=="medium") {
        std::vector<MoveAction> beam = sorted_cands;
        if ((int)beam.size()>CAP.BEAM) beam.resize(CAP.BEAM);
        MoveAction best_by_probe; best_by_probe.type="none";
        double best_probe_score=-1e100;
        for (int i=0;i<std::min((int)beam.size(),CAP.PROBE);++i) {
            auto &a = beam[i];
            Board b2 = after_board(board,a,me);
            auto opp_acts = generate_greedy_candidates(b2,opp,style_name);
            MoveAction opp_best;
            if (!opp_acts.empty()) opp_best = opp_acts[0];
            double final_score = (!opp_acts.empty()) ? -opp_best.score : a.score;
            if (final_score>best_probe_score) {
                best_probe_score = final_score;
                best_by_probe = a;
            }
        }
        std::uniform_real_distribution<double> u01(0.0,1.0);
        MoveAction pick;
        if (best_by_probe.type!="none" && u01(rng)<0.70) {
            pick = best_by_probe;
        } else {
            pick = !beam.empty() ? beam[0] : top_best;
        }
        auto &pool = !beam.empty() ? beam : sorted_cands;
        MoveAction safe = avoid_creating_tactic_loss(
            avoid_self_harm(pick,pool), pool);
        return finalize(safe);
    }

    // easy
    std::vector<MoveAction> E_BEAM = sorted_cands;
    if ((int)E_BEAM.size()>CAP.BEAM) E_BEAM.resize(CAP.BEAM);
    std::vector<MoveAction> pref;
    for (auto &a : E_BEAM) {
        if (a.type!="move") pref.push_back(a);
        else if (!a.swans.empty() && a.swans.size()>1) pref.push_back(a);
    }
    std::uniform_real_distribution<double> u01(0.0,1.0);
    if (!pref.empty()) {
        int idx = std::min((int)pref.size()-1,
                           (int)(pref.size()*0.6));
        return finalize(pref[idx]);
    }
    if (!E_BEAM.empty()) {
        int idx = std::min((int)E_BEAM.size()-1,
                           (int)(E_BEAM.size()*0.7));
        return finalize(E_BEAM[idx]);
    }

    return finalize(top_best);
}

// ============================================================
//  Environment (LinithEnv) – port of linithenv.py
// ============================================================

struct GameState;  // forward

template<typename R>
void encode_state_from_gamestate(const GameState& st, R& r);

struct GameState {
    Board board;
    int current_player;  // SUN or MOON
    int actions_left;
    bool done;
    int winner; // SUN / MOON / 0 (None)
    int move_count;
    int max_moves;

    py::array_t<float> to_tensor() const {
        auto result = py::array_t<float>({8, BOARD_SIZE, BOARD_SIZE});
        auto r = result.mutable_unchecked<3>();

        encode_state_from_gamestate(*this, r);

        return result;
    }
};

template<typename R>
void encode_state_from_gamestate(const GameState& st, R& r) {
    for (int rr = 0; rr < BOARD_SIZE; ++rr) {
        for (int cc = 0; cc < BOARD_SIZE; ++cc) {
            int v = get_cell(st.board, rr, cc);
            // channels 0–4: same mapping as encode_state()
            r(0, rr, cc) = (v == SWAN_SUN    ? 1.0f : 0.0f);
            r(1, rr, cc) = (v == FROZEN_SUN  ? 1.0f : 0.0f);
            r(2, rr, cc) = (v == SWAN_MOON   ? 1.0f : 0.0f);
            r(3, rr, cc) = (v == FROZEN_MOON ? 1.0f : 0.0f);
            r(4, rr, cc) = (v == STONE       ? 1.0f : 0.0f);
            // channel 5: current player; fill below
            r(5, rr, cc) = 0.0f;
            r(6, rr, cc) = static_cast<float>(st.actions_left);
            r(7, rr, cc) = std::min(
                1.0f,
                static_cast<float>(st.move_count) / std::max(1, st.max_moves)
            );
        }
    }

    if (st.current_player == SUN) {
        for (int rr = 0; rr < BOARD_SIZE; ++rr) {
            for (int cc = 0; cc < BOARD_SIZE; ++cc) {
                r(5, rr, cc) = 1.0f;
            }
        }
    } else {
        for (int rr = 0; rr < BOARD_SIZE; ++rr) {
            for (int cc = 0; cc < BOARD_SIZE; ++cc) {
                r(5, rr, cc) = 0.0f;
            }
        }
    }
}

class LinithEnv {
public:
    explicit LinithEnv(int max_moves=500)
        : max_moves_(max_moves)
    {
        reset();
    }

    py::array_t<float> reset() {
        // start with empty board
        for (auto &v : state_.board) v = EMPTY;

        // Sun initial Swan
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0,BOARD_SIZE-1);
        int sr = dist(rng);
        int sc = dist(rng);
        set_cell(state_.board,sr,sc,SWAN_SUN);

        // Moon initial Swan (not 8-adjacent if possible)
        std::vector<Coord> candidates;
        for (int r=0;r<BOARD_SIZE;++r)
            for (int c=0;c<BOARD_SIZE;++c) {
                if (get_cell(state_.board,r,c)!=EMPTY) continue;
                bool adj=false;
                for (auto [dr,dc] : DIRS8) {
                    int nr=sr+dr,nc=sc+dc;
                    if (nr==r && nc==c) {adj=true;break;}
                }
                if (!adj) candidates.emplace_back(r,c);
            }
        if (candidates.empty()) {
            for (int r=0;r<BOARD_SIZE;++r)
                for (int c=0;c<BOARD_SIZE;++c)
                    if (get_cell(state_.board,r,c)==EMPTY)
                        candidates.emplace_back(r,c);
        }
        std::uniform_int_distribution<int> dist2(0,(int)candidates.size()-1);
        Coord m = candidates[dist2(rng)];
        set_cell(state_.board,m.first,m.second,SWAN_MOON);

        state_.current_player = MOON;
        state_.actions_left  = 1;
        state_.done          = false;
        state_.winner        = 0;
        state_.move_count    = 0;
        state_.max_moves     = max_moves_;

        return encode_state();
    }

    py::tuple step_py(const py::object &action_obj) {
        // action is a Python tuple:
        // ("place_swan",  r, c)
        // ("place_stone", r, c)
        // ("move_group",  subset, (dr,dc))
        // ("push",        enemy_subset, (dr,dc))
        if (state_.done) {
            throw std::runtime_error("Game already finished; call reset().");
        }
        std::string kind = py::cast<std::string>(action_obj.attr("__getitem__")(0));
        int acting_player = state_.current_player;

        if (kind=="place_swan") {
            int r = py::cast<int>(action_obj.attr("__getitem__")(1));
            int c = py::cast<int>(action_obj.attr("__getitem__")(2));
            place_swan(r,c);
        } else if (kind=="place_stone") {
            int r = py::cast<int>(action_obj.attr("__getitem__")(1));
            int c = py::cast<int>(action_obj.attr("__getitem__")(2));
            place_stone(r,c);
        } else if (kind=="move_group") {
            auto subset_obj = action_obj.attr("__getitem__")(1);
            auto dir_obj    = action_obj.attr("__getitem__")(2);
            std::vector<Coord> subset;
            int subset_len = static_cast<int>(py::len(subset_obj));
            subset.reserve(subset_len);
            for (int i=0;i<subset_len;++i) {
                auto t = subset_obj.attr("__getitem__")(i);
                int r = py::cast<int>(t.attr("__getitem__")(0));
                int c = py::cast<int>(t.attr("__getitem__")(1));
                subset.emplace_back(r,c);
            }
            int dr = py::cast<int>(dir_obj.attr("__getitem__")(0));
            int dc = py::cast<int>(dir_obj.attr("__getitem__")(1));
            move_group(subset,Coord(dr,dc));
        } else if (kind=="push") {
            auto subset_obj = action_obj.attr("__getitem__")(1);
            auto dir_obj = action_obj.attr("__getitem__")(2);
            std::vector<Coord> subset;
            for (int i=0; i<static_cast<int>(py::len(subset_obj)); ++i) {
                auto t = subset_obj.attr("__getitem__")(i);
                subset.emplace_back(
                    py::cast<int>(t.attr("__getitem__")(0)),
                    py::cast<int>(t.attr("__getitem__")(1))
                );
            }
            push(
                subset,
                Coord(
                    py::cast<int>(dir_obj.attr("__getitem__")(0)),
                    py::cast<int>(dir_obj.attr("__getitem__")(1))
                )
            );
        } else {
            throw std::runtime_error("Unknown action kind in step_py");
        }


        auto fr = compute_freezes_on(state_.board);
        state_.board = fr.board;

        check_terminal_conditions(fr);

        float reward = 0.0f;
        if (state_.done) {
            if (state_.winner==0) reward = 0.0f;
            else if (state_.winner==acting_player) reward=1.0f;
            else reward=-1.0f;
        } else {
            // Extra actions equal to number of enemy Swans frozen/sealed this action
            int extra_actions = (acting_player==SUN)
                ? (fr.froze_moon+fr.sealed_moon)
                : (fr.froze_sun+fr.sealed_sun);

            state_.actions_left += extra_actions;

            // Base cost of this action
            state_.actions_left -= 1;

            // If no actions left, pass the turn
            if (state_.actions_left<=0) {
                state_.current_player = (acting_player==SUN)?MOON:SUN;
                state_.actions_left = both_at_six_board(state_.board)?2:1;
            }
        }

        state_.move_count += 1;
        if (state_.move_count>=max_moves_ && !state_.done) {
            state_.done = true;
            state_.winner = 0;
        }

        auto obs = encode_state();
        py::dict info;
        return py::make_tuple(obs, reward, state_.done, info);
    }

    const GameState &state() const { return state_; }

    std::vector<py::object> legal_actions_py() const {
        if (state_.done) return {};
        std::vector<py::object> acts;
        int player = state_.current_player;
        const Board &board = state_.board;

        // swan placements
        if (count_total_swans_int(player,board)<6) {
            for (auto [r,c] : legal_swan_placements(board,player)) {
                acts.push_back(py::make_tuple("place_swan",r,c));
            }
        }

        // stone placements
        for (auto [r,c] : legal_stone_placements(board, player)) {
            acts.push_back(py::make_tuple("place_stone", r, c));
        }

        // group moves
        auto gmoves = legal_group_moves(board,player,6);
        for (auto &entry : gmoves) {
            auto &subset = entry.first;
            auto &dir = entry.second;
            py::list s_list;
            for (auto [r,c] : subset)
                s_list.append(py::make_tuple(r,c));
            acts.push_back(py::make_tuple(
                "move_group",
                s_list,
                py::make_tuple(dir.first,dir.second)
            ));
        }

        for (const auto &[subset, dir] : legal_push_moves_board(board, player)) {
            py::list s_list;
            for (auto [r,c] : subset) s_list.append(py::make_tuple(r,c));
            acts.push_back(py::make_tuple(
                "push", s_list, py::make_tuple(dir.first, dir.second)
            ));
        }

        return acts;
    }

    // Create a deep copy of this environment (used by Python MCTS)
    LinithEnv clone() const {
        LinithEnv copy(max_moves_);
        copy.state_ = state_;  // GameState is trivially copyable
        return copy;
    }

private:
    int max_moves_;
    GameState state_;

    void place_swan(int r,int c) {
        int player = state_.current_player;
        if (!in_bounds(r,c) || count_total_swans_int(player,state_.board)>=6)
            throw std::runtime_error("Illegal Swan placement.");
        auto legal = legal_swan_placements(state_.board, player);
        if (std::find(legal.begin(), legal.end(), Coord(r,c)) == legal.end())
            throw std::runtime_error("Illegal Swan placement.");
        set_cell(state_.board,r,c,(player==SUN)?SWAN_SUN:SWAN_MOON);
    }

    void place_stone(int r,int c) {
        if (!in_bounds(r,c) || get_cell(state_.board,r,c)!=EMPTY)
            throw std::runtime_error("A Stone can only be placed on an empty tile.");
        set_cell(state_.board,r,c,STONE);
    }

    void move_group(const std::vector<Coord> &subset, Coord dir) {
        bool ok=false;
        Board nb = simulate_group_move(state_.board,subset,dir,state_.current_player,ok);
        if (!ok)
            throw std::runtime_error("Illegal group move passed to move_group.");
        state_.board = nb;
    }

    void push(const std::vector<Coord> &subset, Coord direction) {
        bool ok=false;
        Board nb = simulate_push_subset_board(
            state_.board, state_.current_player, subset, direction, ok
        );
        if (!ok)
            throw std::runtime_error("Illegal push move passed to push.");
        state_.board = nb;
    }

    py::array_t<float> encode_state() const {
        // Channels:
        // 0: sun active
        // 1: sun frozen
        // 2: moon active
        // 3: moon frozen
        // 4: stone
        // 6: actions_left; 7: move_count/max_moves
        auto result = py::array_t<float>({8,BOARD_SIZE,BOARD_SIZE});
        auto r = result.mutable_unchecked<3>();
        encode_state_from_gamestate(state_, r);
        return result;
    }

    void check_terminal_conditions(const FreezeResult &fr) {
        if (state_.done) return;

        const Board &b = state_.board;

        if (fr.sealed_sun>0 && fr.sealed_moon>0) {
            state_.done=true;
            state_.winner=0;
            return;
        }
        if (fr.sealed_sun>0) {
            state_.done=true;
            state_.winner=MOON;
            return;
        }
        if (fr.sealed_moon>0) {
            state_.done=true;
            state_.winner=SUN;
            return;
        }

        int a_sun = count_active_swans_int(SUN,b);
        int a_moon= count_active_swans_int(MOON,b);

        if (a_sun==0 && a_moon>0) {
            state_.done=true;
            state_.winner=MOON;
            return;
        }
        if (a_moon==0 && a_sun>0) {
            state_.done=true;
            state_.winner=SUN;
            return;
        }

        if (a_sun > 0 && a_moon > 0) {
            if (!has_any_legal_action_board(b, SUN) &&
                !has_any_legal_action_board(b, MOON)) {
                state_.done = true;
                state_.winner = 0;
                return;
                }
        }
    }
};

// ============================================================
//  Action space mapping (encode_action env->index)
//  0–99   : place_swan
//  100–199: place_stone
//  200–703: move_group (1..6 swans, 8 dirs)
//  704–1207: push      (enemy subset, 8 dirs)
// ============================================================

constexpr int ACTION_SIZE = 1208;
constexpr int MAX_SWANS   = 6;

inline int square_index(int r,int c) {
    return r*BOARD_SIZE + c;
}

std::vector<Coord> active_swans_for_player(const LinithEnv &env, int player) {
    const GameState &s = env.state();
    const Board &board = s.board;
    int target = (player==SUN)?SWAN_SUN:SWAN_MOON;
    std::vector<Coord> coords;
    for (int r=0;r<BOARD_SIZE;++r)
        for (int c=0;c<BOARD_SIZE;++c)
            if (get_cell(board,r,c)==target)
                coords.emplace_back(r,c);
    std::sort(coords.begin(),coords.end());
    if ((int)coords.size()>MAX_SWANS)
        coords.resize(MAX_SWANS);
    return coords;
}

std::vector<Coord> active_swans_for_current_player(const LinithEnv &env) {
    return active_swans_for_player(env, env.state().current_player);
}

int subset_mask_from_coords(const std::vector<Coord> &subset,
                            const std::vector<Coord> &swans)
{
    std::map<Coord,int> coord_to_index;
    for (int i=0;i<(int)swans.size();++i)
        coord_to_index[swans[i]] = i;
    int mask=0;
    for (auto [r,c] : subset) {
        Coord key(r,c);
        auto it = coord_to_index.find(key);
        if (it==coord_to_index.end())
            throw std::runtime_error("Swan coord not found in active swans");
        int i = it->second;
        if (i>=MAX_SWANS)
            throw std::runtime_error("Swan index exceeds MAX_SWANS");
        mask |= (1<<i);
    }
    if (mask==0)
        throw std::runtime_error("Empty subset for move_group");
    return mask;
}

std::vector<int> build_subset_masks() {
    std::vector<int> masks;
    for (int m=1; m<(1<<MAX_SWANS); ++m)
        masks.push_back(m);
    return masks;
}
static const std::vector<int> SUBSET_MASKS = build_subset_masks();

std::map<int,int> build_mask_to_index() {
    std::map<int,int> m;
    for (int i=0;i<(int)SUBSET_MASKS.size();++i)
        m[SUBSET_MASKS[i]] = i;
    return m;
}
static const std::map<int,int> MASK_TO_INDEX = build_mask_to_index();

int encode_action_cpp(const LinithEnv &env, const py::object &action_obj) {
    std::string kind = py::cast<std::string>(action_obj.attr("__getitem__")(0));

    if (kind=="place_swan") {
        int r = py::cast<int>(action_obj.attr("__getitem__")(1));
        int c = py::cast<int>(action_obj.attr("__getitem__")(2));
        if (!in_bounds(r,c))
            throw std::runtime_error("place_swan outside board");
        int idx = square_index(r,c);
        if (!(0<=idx && idx<100))
            throw std::runtime_error("place_swan outside board");
        return idx;
    }
    if (kind=="place_stone") {
        int r = py::cast<int>(action_obj.attr("__getitem__")(1));
        int c = py::cast<int>(action_obj.attr("__getitem__")(2));
        if (!in_bounds(r,c))
            throw std::runtime_error("place_stone outside board");
        int rc = square_index(r,c);
        if (!(0<=rc && rc<100))
            throw std::runtime_error("place_stone outside board");
        return 100+rc;
    }
    if (kind=="move_group") {
        auto subset_obj = action_obj.attr("__getitem__")(1);
        auto dir_obj    = action_obj.attr("__getitem__")(2);

        int subset_len = static_cast<int>(py::len(subset_obj));
        if (!(1<=subset_len && subset_len<=MAX_SWANS))
            throw std::runtime_error("subset size out of 1..MAX_SWANS");
        std::vector<Coord> subset;
        subset.reserve(subset_len);
        for (int i=0;i<subset_len;++i) {
            auto t = subset_obj.attr("__getitem__")(i);
            int r = py::cast<int>(t.attr("__getitem__")(0));
            int c = py::cast<int>(t.attr("__getitem__")(1));
            subset.emplace_back(r,c);
        }

        auto swans = active_swans_for_current_player(env);
        int mask = subset_mask_from_coords(subset,swans);
        auto it = MASK_TO_INDEX.find(mask);
        if (it==MASK_TO_INDEX.end())
            throw std::runtime_error("Subset mask out of supported range");
        int subset_index = it->second;

        int dr = py::cast<int>(dir_obj.attr("__getitem__")(0));
        int dc = py::cast<int>(dir_obj.attr("__getitem__")(1));
        int dir_index=-1;
        for (int i=0;i<8;++i) {
            if (DIRS8[i].first==dr && DIRS8[i].second==dc) {
                dir_index=i; break;
            }
        }
        if (dir_index<0)
            throw std::runtime_error("Unknown direction in move_group.");

        return 200 + subset_index*8 + dir_index;
    }

    if (kind=="push") {
        auto subset_obj = action_obj.attr("__getitem__")(1);
        auto dir_obj = action_obj.attr("__getitem__")(2);
        int subset_len = static_cast<int>(py::len(subset_obj));
        if (!(1 <= subset_len && subset_len <= MAX_SWANS))
            throw std::runtime_error("push subset size out of range");
        std::vector<Coord> subset;
        for (int i=0; i<subset_len; ++i) {
            auto t = subset_obj.attr("__getitem__")(i);
            subset.emplace_back(
                py::cast<int>(t.attr("__getitem__")(0)),
                py::cast<int>(t.attr("__getitem__")(1))
            );
        }
        int enemy = env.state().current_player == SUN ? MOON : SUN;
        auto swans = active_swans_for_player(env, enemy);
        int mask = subset_mask_from_coords(subset, swans);
        int subset_index = MASK_TO_INDEX.at(mask);

        int dr = py::cast<int>(dir_obj.attr("__getitem__")(0));
        int dc = py::cast<int>(dir_obj.attr("__getitem__")(1));
        int dir_index = -1;
        for (int i=0;i<8;++i) {
            if (DIRS8[i].first==dr && DIRS8[i].second==dc) {
                dir_index = i;
                break;
            }
        }
        if (dir_index<0)
            throw std::runtime_error("push: direction not in DIRS8");

        return 704 + subset_index*8 + dir_index;
    }

    throw std::runtime_error("Unknown action kind in encode_action_cpp");
}

std::vector<int> legal_action_indices_cpp(const LinithEnv &env) {
    std::vector<int> indices;
    auto acts = env.legal_actions_py();
    for (auto &a : acts) {
        try {
            int idx = encode_action_cpp(env,a);
            indices.push_back(idx);
        } catch (...) {
            // skip unencodable
        }
    }
    return indices;
}

// ============================================================
//  PV MCTS (AlphaZero-style) in C++
//  - Uses LinithEnv for simulation
//  - Uses Python callback eval_fn(env) -> (policy_vector, value)
// ============================================================

struct PVNodeCpp {
    PVNodeCpp* parent;
    py::object action_from_parent;  // Python action that led here (None for root)
    int player;                     // player to move at this node
    double N;
    double W;
    double Q;
    double P;                       // prior probability from policy net

    std::vector<std::unique_ptr<PVNodeCpp>> children;
    std::vector<py::object> child_actions;  // parallel to children

    PVNodeCpp(PVNodeCpp* parent_,
              const py::object& action_,
              int player_)
        : parent(parent_),
          action_from_parent(action_),
          player(player_),
          N(0.0),
          W(0.0),
          Q(0.0),
          P(0.0)
    {}
};

// simple RNG for Dirichlet noise
static std::mt19937& pv_rng() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

// -------- helpers: terminal value, evaluation, expansion, selection, backprop --------

double pv_terminal_value_cpp(const LinithEnv& env, int root_player) {
    int winner = env.state().winner;
    if (winner == 0) return 0.0;
    return (winner == root_player) ? 1.0 : -1.0;
}

// eval_fn(env) must return (policy_vector, value) from Python:
// policy_vector: np.ndarray [ACTION_SIZE], float32
// value: scalar float from the current player-to-move's perspective
std::pair<std::vector<double>, double>
pv_evaluate_cpp(LinithEnv& env, const py::function& eval_fn) {
    py::tuple out = eval_fn(env);
    if (out.size() != 2) {
        throw std::runtime_error("eval_fn must return (policy_vector, value)");
    }

    py::array policy_arr = out[0].cast<py::array>();
    double value = out[1].cast<double>();

    py::buffer_info buf = policy_arr.request();
    if (buf.ndim != 1 || buf.shape[0] != ACTION_SIZE) {
        throw std::runtime_error("policy_vector must be 1D of length ACTION_SIZE");
    }

    std::vector<double> policy(ACTION_SIZE);
    if (buf.format == py::format_descriptor<float>::format()) {
        float* ptr = static_cast<float*>(buf.ptr);
        for (ssize_t i = 0; i < buf.shape[0]; ++i) {
            policy[i] = static_cast<double>(ptr[i]);
        }
    } else if (buf.format == py::format_descriptor<double>::format()) {
        double* ptr = static_cast<double*>(buf.ptr);
        for (ssize_t i = 0; i < buf.shape[0]; ++i) {
            policy[i] = ptr[i];
        }
    } else {
        // fallback: convert element-wise through Python
        for (ssize_t i = 0; i < buf.shape[0]; ++i) {
            policy[i] = policy_arr.attr("__getitem__")(i).cast<double>();
        }
    }

    return {policy, value};
}

// EXPAND with policy-guided pruning, same logic as Python PV_MCTS._expand
void pv_expand_cpp(
    PVNodeCpp* node,
    LinithEnv& env,
    const std::vector<py::object>& legal_actions,
    const std::vector<double>& policy_vector)
{
    if (!node->children.empty()) return;
    node->player = env.state().current_player;
    struct Scored {
        double p;
        py::object action;
        int idx;
    };

    std::vector<Scored> scored;
    scored.reserve(legal_actions.size());

    for (auto& a : legal_actions) {
        try {
            int idx = encode_action_cpp(env, a);
            if (idx < 0 || idx >= ACTION_SIZE) continue;
            double p = policy_vector[idx];
            scored.push_back({p, a, idx});
        } catch (const std::exception&) {
            // skip unencodable actions
            continue;
        }
    }

    if (scored.empty()) {
        return;  // nothing to expand
    }

    // sort by policy prob descending
    std::sort(
        scored.begin(),
        scored.end(),
        [](const Scored& x, const Scored& y) {
            return x.p > y.p;
        }
    );

    const int MAX_K = 48;
    const int MIN_K = 12;

    int k = std::min<int>(MAX_K, static_cast<int>(scored.size()));
    if (k < MIN_K) {
        k = std::min<int>(static_cast<int>(scored.size()), MIN_K);
    }
    scored.resize(k);

    double total_p = 0.0;
    for (const auto& s : scored) {
        total_p += std::max(0.0, s.p);
    }

    if (total_p <= 0.0) {
        // uniform fallback
        double prior = 1.0 / static_cast<double>(scored.size());
        for (const auto& s : scored) {
            auto child = std::make_unique<PVNodeCpp>(node, s.action, /*player*/0);
            child->P = prior;
            node->child_actions.push_back(s.action);
            node->children.push_back(std::move(child));
        }
        return;
    }

    for (const auto& s : scored) {
        double prior = std::max(0.0, s.p) / total_p;
        auto child = std::make_unique<PVNodeCpp>(node, s.action, /*player*/0);
        child->P = prior;
        node->child_actions.push_back(s.action);
        node->children.push_back(std::move(child));
    }
}

// SELECT: traverse tree from root until leaf using UCB, simulating in a cloned env
std::pair<PVNodeCpp*, LinithEnv>
pv_select_cpp(LinithEnv& root_env, PVNodeCpp* root, double c_puct, int root_player) {
    PVNodeCpp* node = root;
    LinithEnv sim_env = root_env.clone();

    while (!node->children.empty()) {
        // total visit count
        double total_N = 1e-8;
        for (auto& child_ptr : node->children) {
            total_N += child_ptr->N;
        }

        double best_score = -1e9;
        int best_index = -1;

        for (int i = 0; i < static_cast<int>(node->children.size()); ++i) {
            auto& child = *node->children[i];
            double U = c_puct * child.P * std::sqrt(total_N) / (1.0 + child.N);
            double exploitation = (node->player == root_player) ? child.Q : -child.Q;
            double score = exploitation + U;
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }

        if (best_index < 0) {
            break;
        }

        py::object action = node->child_actions[best_index];

        // step the environment
        auto step_out = sim_env.step_py(action);
        const GameState& s = sim_env.state();
        node = node->children[best_index].get();
        node->player = s.current_player;

        if (s.done) {
            break;
        }
    }

    return {node, sim_env};
}

void pv_backpropagate_cpp(PVNodeCpp* node, double value, int root_player) {
    PVNodeCpp* cur = node;
    (void)root_player; // we already pass value in root_player's perspective

    while (cur != nullptr) {
        cur->N += 1.0;
        cur->W += value;
        cur->Q = cur->W / cur->N;
        cur = cur->parent;
    }
}

void pv_expand_cpp(
    PVNodeCpp* node,
    LinithEnv& env,
    const std::vector<py::object>& legal_actions,
    const std::vector<double>& policy_vector);


// Convert any Python list inside an action structure to tuples,
// recursively, so that the whole action becomes hashable.
py::object make_hashable_action(const py::object& obj) {
    if (py::isinstance<py::list>(obj)) {
        py::list lst = obj.cast<py::list>();
        py::tuple tup(lst.size());
        for (size_t i = 0; i < lst.size(); ++i) {
            tup[i] = make_hashable_action(lst[i]); // recurse
        }
        return tup;
    }
    else if (py::isinstance<py::tuple>(obj)) {
        py::tuple t = obj.cast<py::tuple>();
        py::tuple out(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            out[i] = make_hashable_action(t[i]);
        }
        return out;
    }
    else {
        return obj; // primitive: int,str,None...
    }
}

// Batch evaluation: envs -> [(policy, value), ...]
static std::vector<std::pair<std::vector<double>, double>>
pv_evaluate_batch_cpp(const std::vector<LinithEnv>& envs, py::function eval_fn)
{
    py::gil_scoped_acquire gil;

    py::list py_envs;
    for (const auto& e : envs) {
        py_envs.append(e);  // LinithEnv is a pybind-exposed type
    }

    // eval_fn is implemented in Python to accept a list and return a list
    py::object out = eval_fn(py_envs);
    py::sequence seq = out;

    if (py::len(seq) != static_cast<py::ssize_t>(envs.size())) {
        throw std::runtime_error("eval_fn batch length mismatch");
    }

    std::vector<std::pair<std::vector<double>, double>> results;
    results.reserve(envs.size());

    for (py::handle item : seq) {
        py::tuple t = py::cast<py::tuple>(item);
        std::vector<double> policy = py::cast<std::vector<double>>(t[0]);
        double value = py::cast<double>(t[1]);
        results.emplace_back(std::move(policy), value);
    }

    return results;
}


// Main search entry point: returns dict {action -> visit_count}
py::dict pv_mcts_search_cpp(
    LinithEnv& env,
    py::function eval_fn,
    int num_simulations,
    double c_puct = 1.5,
    bool add_root_noise = false,
    double dirichlet_alpha = 0.3,
    double dirichlet_eps = 0.25)
{
    if (num_simulations <= 0) {
        py::dict empty;
        return empty;
    }

    const GameState& root_state = env.state();
    int root_player = root_state.current_player;

    PVNodeCpp root(nullptr, py::none(), root_player);

    // ----- initial expansion at root -----
    auto eval_root = pv_evaluate_cpp(env, eval_fn);
    std::vector<double>& policy_root = eval_root.first;
    double value_root = eval_root.second;

    auto legal_root = env.legal_actions_py();
    pv_expand_cpp(&root, env, legal_root, policy_root);

    root.W += value_root;
    root.N += 1.0;
    root.Q = root.W / root.N;

    // optional Dirichlet noise on root priors
    if (add_root_noise && !root.children.empty()) {
        std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
        std::vector<double> noise(root.children.size());
        double sum = 0.0;
        for (size_t i = 0; i < noise.size(); ++i) {
            double x = gamma(pv_rng());
            noise[i] = x;
            sum += x;
        }
        if (sum > 0.0) {
            for (double& x : noise) x /= sum;
        }

        for (size_t i = 0; i < root.children.size(); ++i) {
            auto& child = *root.children[i];
            double p_old = child.P;
            double p_new = (1.0 - dirichlet_eps) * p_old + dirichlet_eps * noise[i];
            child.P = p_new;
        }
    }

    // Sequential selection/evaluation guarantees that each simulation observes
    // the visits produced by the previous one and that a leaf expands once.
    for (int sim = 1; sim < num_simulations; ++sim) {
        auto sel = pv_select_cpp(env, &root, c_puct, root_player);
        PVNodeCpp* leaf = sel.first;
        LinithEnv sim_env = std::move(sel.second);

        double v;
        if (sim_env.state().done) {
            v = pv_terminal_value_cpp(sim_env, root_player);
        } else {
            auto evaluated = pv_evaluate_cpp(sim_env, eval_fn);
            auto legal = sim_env.legal_actions_py();
            pv_expand_cpp(leaf, sim_env, legal, evaluated.first);
            v = evaluated.second;
            if (sim_env.state().current_player != root_player) v = -v;
        }
        pv_backpropagate_cpp(leaf, v, root_player);
    }

    // ----- collect visit counts at root -----
    py::dict visits;
    for (size_t i = 0; i < root.children.size(); ++i) {
        auto& child = *root.children[i];
        py::object action_raw = root.child_actions[i];
        py::object action = make_hashable_action(action_raw);
        visits[action] = child.N;
    }

    return visits;
}

// ============================================================
//  Hard-vs-Hard self-play and dataset generator
// ============================================================

std::string format_dt_ae(const std::chrono::system_clock::time_point &tp) {
    using namespace std::chrono;
    auto tt = system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm,&tt);
#else
    localtime_r(&tt,&tm);
#endif
    int ae = (tm.tm_year+1900)-2020;
    char buf[64];
    std::snprintf(buf,sizeof(buf),"%dAE-%02d-%02d %02d:%02d:%02d",
                  ae, tm.tm_mon+1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

struct GameData {
    std::vector<float> states;   // [T,8,10,10]
    std::vector<float> policies; // [T,ACTION_SIZE]
    std::vector<float> values;   // [T]
    int winner;
};

struct MoveKey {
    std::size_t h;
    int player;
};

py::object choose_hard_move_py(LinithEnv &env, const std::string &difficulty) {
    const GameState &s = env.state();
    int current = s.current_player;
    Board board = s.board;
    MoveAction a = linith_ai(board,current,difficulty,"doctrinal",false);
    if (a.type=="stone") {
        return py::make_tuple("place_stone",a.r,a.c);
    } else if (a.type=="swan") {
        return py::make_tuple("place_swan",a.r,a.c);
    } else if (a.type=="move") {
        py::list sw;
        for (auto [r,c] : a.swans) sw.append(py::make_tuple(r,c));
        return py::make_tuple("move_group",sw,py::make_tuple(a.dir.first,a.dir.second));
    } else if (a.type=="push") {
        py::list sw;
        for (auto [r,c] : a.swans) sw.append(py::make_tuple(r,c));
        return py::make_tuple("push", sw, py::make_tuple(a.dir.first,a.dir.second));
    }
    // fallback: any legal action
    auto legal = env.legal_actions_py();
    if (legal.empty())
        throw std::runtime_error("Hard AI has no legal actions");
    return legal[0];
}


GameData play_hard_vs_hard_game_cpp(
    int max_moves,
    const std::string &difficulty)
{
    GameData gd;
    LinithEnv env(max_moves);
    auto obs = env.reset(); // (8,10,10) float32

    std::vector<int> players;

    // (player, encoded_action_index) history for repetition checks
    std::vector<std::pair<int,int>> history;

    // how strict we are
    const int MAX_SAME_ACTION_CHAIN    = 2;   // same (player,idx) in a row
    const int HISTORY_MAX              = 32;  // max length of history vector
    const int MAX_SAME_ACTION_GLOBAL   = 3;   // total uses of same (player,idx) in a game
    const int WINDOW_PLIES             = 24;  // how many recent plies we inspect
    const int MAX_WINDOW_ACTION_REUSE  = 1;   // max times same (player,idx) may appear in that window

    // total usage of (player, idx) across the whole game
    std::map<std::pair<int,int>, int> action_use;

    auto would_cause_repeat = [&](int player, int idx) -> bool {
        auto key = std::make_pair(player, idx);

        // --- 0) global cap: don't let a player spam exactly the same action forever ---
        {
            auto it  = action_use.find(key);
            int used = (it == action_use.end()) ? 0 : it->second;
            if (used >= MAX_SAME_ACTION_GLOBAL) {
                return true;
            }
        }

        int n = static_cast<int>(history.size());
        if (n == 0) {
            return false;
        }

        // --- 1) same-action chain for the same player (in a row) ---
        {
            int chain = 0;
            for (int i = n - 1; i >= 0; --i) {
                auto e = history[i];
                if (e.first == player && e.second == idx) {
                    ++chain;
                } else {
                    break;
                }
            }
            if (chain >= MAX_SAME_ACTION_CHAIN) {
                return true;
            }
        }

        // --- 2) density in the recent window: too many uses in last WINDOW_PLIES plies ---
        {
            int start = std::max(0, n - WINDOW_PLIES);
            int count = 0;
            for (int i = start; i < n; ++i) {
                auto e = history[i];
                if (e.first == player && e.second == idx) {
                    ++count;
                    if (count >= MAX_WINDOW_ACTION_REUSE) {
                        return true;
                    }
                }
            }
        }

        // --- 3) ABAB ping-pong pattern (optional but still useful) ---
        if (n >= 3) {
            auto e0 = history[n - 3];
            auto e1 = history[n - 2];
            auto e2 = history[n - 1];

            if (e0.first == e2.first &&
                e0.second == e2.second &&
                e0.first != e1.first)
            {
                if (player == e1.first && idx == e1.second) {
                    return true;
                }
            }
        }

        return false;
    };

    int move_no = 0;

    while (!env.state().done) {
        const GameState &s = env.state();
        auto legal = env.legal_actions_py();
        if (legal.empty()) break;

        py::object action_obj;
        int idx = -1;

        // 1) ask Hard AI for its move and encode it
        try {
            action_obj = choose_hard_move_py(env, difficulty);
            idx = encode_action_cpp(env, action_obj);
        } catch (const std::exception &e) {
            std::cerr << "[error] choose_hard_move/encode_action_cpp failed: "
                      << e.what() << "\n";
            break;
        }

        // 2) if this move would repeat a bad pattern, look for an alternative legal action
        if (idx < 0 || idx >= ACTION_SIZE || would_cause_repeat(s.current_player, idx)) {
            bool found_alt = false;

            for (auto &cand : legal) {
                try {
                    int idx2 = encode_action_cpp(env, cand);
                    if (idx2 < 0 || idx2 >= ACTION_SIZE) continue;
                    if (idx2 == idx) continue;
                    if (would_cause_repeat(s.current_player, idx2)) continue;

                    action_obj = cand;
                    idx = idx2;
                    found_alt = true;
                    break;
                } catch (...) {
                    // if encoding some candidate blows up, just skip it
                    continue;
                }
            }

            // If no alternative avoids repetition but idx is still valid,
            // we just accept the original move (better to continue than crash).
            if (!found_alt) {
                if (idx < 0 || idx >= ACTION_SIZE) {
                    std::cerr << "[error] no non-repeating alternative and idx invalid; aborting game\n";
                    break;
                }
            }
        }

        // 3) debug-print the move
        //try {
        //    std::string act_str = py::str(action_obj).cast<std::string>();
        //   std::cout << "Move " << move_no
        //              << "  P" << (s.current_player == SUN ? "SUN" : "MOON")
        //              << "  idx=" << idx
        //              << "  " << act_str
        //              << "\n";
        //} catch (...) {
            // ignore printing issues
        //}

        // 4) build one-hot policy at idx
        std::vector<float> pi(ACTION_SIZE, 0.0f);
        if (idx >= 0 && idx < ACTION_SIZE) {
            pi[idx] = 1.0f;
        } else {
            std::cerr << "[warn] idx out-of-range when building policy: " << idx << "\n";
        }

        // 5) store obs + policy
        auto buf = obs.request();
        float *obs_ptr = static_cast<float*>(buf.ptr);
        int obs_size = 8 * BOARD_SIZE * BOARD_SIZE;
        gd.states.insert(gd.states.end(), obs_ptr, obs_ptr + obs_size);
        gd.policies.insert(gd.policies.end(), pi.begin(), pi.end());
        players.push_back(s.current_player);

        // 6) update repetition history
        history.emplace_back(s.current_player, idx);
        if (static_cast<int>(history.size()) > HISTORY_MAX) {
            history.erase(history.begin());
        }

        // 6b) update global usage count
        auto key = std::make_pair(s.current_player, idx);
        action_use[key] += 1;

        // 7) step the environment
        auto step_out = env.step_py(action_obj);
        obs = step_out.cast<py::tuple>()[0].cast<py::array_t<float>>();

        ++move_no;
        if (move_no >= max_moves) break;
    }

    int winner = env.state().winner;
    gd.winner = winner;

    float z_sun = 0.0f;
    if (winner == SUN)      z_sun = 1.0f;
    else if (winner == MOON) z_sun = -1.0f;

    for (int p : players) {
        if (winner == 0) {
            gd.values.push_back(0.0f);
        } else if (p == SUN) {
            gd.values.push_back(z_sun);
        } else {
            gd.values.push_back(-z_sun);
        }
    }

    return gd;
}


py::tuple generate_teacher_dataset_cpp(
    int num_games,
    int max_moves,
    const std::string &difficulty)
{
    std::vector<float> all_states;
    std::vector<float> all_policies;
    std::vector<float> all_values;

    int wins_sun=0, wins_moon=0, draws=0;
    int total_positions=0;

    auto batch_start = std::chrono::system_clock::now();
    std::cout << "Self-Play start - " << format_dt_ae(batch_start) << "\n\n";

    for (int g=0; g<num_games; ++g) {
        int game_idx = g+1;
        auto game_start = std::chrono::system_clock::now();
        std::cout << "Playing game " << game_idx << "/" << num_games
                  << " - Start " << format_dt_ae(game_start) << "\n";

        try {
            GameData gd = play_hard_vs_hard_game_cpp(max_moves,difficulty);
            int T = (int)gd.values.size();
            if (T==0) {
                std::cout << "[warning] game " << game_idx
                          << " produced no positions; skipping\n";
                continue;
            }

            all_states.insert(all_states.end(), gd.states.begin(), gd.states.end());
            all_policies.insert(all_policies.end(), gd.policies.begin(), gd.policies.end());
            all_values.insert(all_values.end(), gd.values.begin(), gd.values.end());
            total_positions += T;

            std::string winner_msg;
            if (gd.winner==0) { winner_msg="Draw"; draws++; }
            else if (gd.winner==SUN) { winner_msg="Sun won"; wins_sun++; }
            else if (gd.winner==MOON){ winner_msg="Moon won"; wins_moon++; }
            else winner_msg="Unknown result";

            auto game_end = std::chrono::system_clock::now();
            double dur = std::chrono::duration<double>(game_end-game_start).count();
            std::cout << "Game " << game_idx << "/" << num_games
                      << " - " << winner_msg
                      << ", Positions - " << T
                      << ", end " << format_dt_ae(game_end)
                      << ", duration " << dur << "s\n";
        } catch (const std::exception &e) {
            auto game_end = std::chrono::system_clock::now();
            std::cerr << "[error] Game " << game_idx << "/" << num_games
                      << " failed at " << format_dt_ae(game_end)
                      << " : " << e.what() << "\n";
        }
    }

    if (all_states.empty())
        throw std::runtime_error("[error] no successful games; dataset is empty");

    int T = total_positions;
    auto batch_end = std::chrono::system_clock::now();

    std::cout << "\n==== Hard-AI Teacher Self-Play Summary ====\n";
    std::cout << "Games requested - " << num_games << "\n";
    std::cout << "Sun wins       - " << wins_sun << "\n";
    std::cout << "Moon wins      - " << wins_moon << "\n";
    std::cout << "Draws          - " << draws << "\n\n";
    std::cout << "Dataset positions N = " << T << "\n";
    std::cout << "Self-play end   " << format_dt_ae(batch_end) << "\n";
    std::cout << "Total duration  "
              << std::chrono::duration_cast<std::chrono::seconds>(batch_end-batch_start).count()
              << "s\n\n";

    int state_size = 8*BOARD_SIZE*BOARD_SIZE;

    if ((int)all_states.size()  != T*state_size)
        throw std::runtime_error("states size mismatch");
    if ((int)all_policies.size()!= T*ACTION_SIZE)
        throw std::runtime_error("policies size mismatch");
    if ((int)all_values.size()  != T)
        throw std::runtime_error("values size mismatch");

    py::array_t<float> X({T,8,BOARD_SIZE,BOARD_SIZE});
    py::array_t<float> Pi({T,ACTION_SIZE});
    py::array_t<float> Z({T});

    std::memcpy(X.mutable_data(),  all_states.data(),   all_states.size()*sizeof(float));
    std::memcpy(Pi.mutable_data(), all_policies.data(),all_policies.size()*sizeof(float));
    std::memcpy(Z.mutable_data(),  all_values.data(),  all_values.size()*sizeof(float));

    return py::make_tuple(X,Pi,Z);
}

// ============================================================
//  Pybind11 module
// ============================================================

PYBIND11_MODULE(linith_selfplay_cpp, m) {
    m.doc() = "C++ Hard-vs-Hard self-play generator for Linith";

    // ------------------------------
    // Constants (OK)
    // ------------------------------
    m.attr("SUN")  = SUN;
    m.attr("MOON") = MOON;
    m.attr("ACTION_SIZE") = ACTION_SIZE;

    // ------------------------------
    // 1. Bind GameState FIRST
    // ------------------------------
    py::class_<GameState>(m, "GameState")
        .def_readwrite("current_player", &GameState::current_player)
        .def_readwrite("actions_left", &GameState::actions_left)
        .def_readwrite("done", &GameState::done)
        .def_property(
            "winner",
            [](const GameState &s) -> py::object {
                return s.winner == 0 ? py::none() : py::cast(s.winner);
            },
            [](GameState &s, const py::object &winner) {
                s.winner = winner.is_none() ? 0 : winner.cast<int>();
            }
        )
        .def_readwrite("move_count", &GameState::move_count)
        .def_readwrite("max_moves", &GameState::max_moves)
        .def("to_tensor", &GameState::to_tensor);

    // ------------------------------
    // 2. Bind LinithEnv SECOND
    // ------------------------------
    py::class_<LinithEnv>(m, "LinithEnv")
        .def(py::init<int>(), py::arg("max_moves") = 20000)
        .def("reset", &LinithEnv::reset)
        .def("step", &LinithEnv::step_py)
        .def("legal_actions", &LinithEnv::legal_actions_py)
        .def_property_readonly(
            "state",
            &LinithEnv::state,
            py::return_value_policy::reference_internal
        )
        .def("clone", &LinithEnv::clone);

    py::class_<EvalWeights>(m, "EvalWeights")
        .def(py::init<>())
        .def_readwrite("wFreeze",              &EvalWeights::wFreeze)
        .def_readwrite("wSelfFreeze",          &EvalWeights::wSelfFreeze)
        .def_readwrite("wMyLib",               &EvalWeights::wMyLib)
        .def_readwrite("wOpLib",               &EvalWeights::wOpLib)
        .def_readwrite("wRing",                &EvalWeights::wRing)
        .def_readwrite("wMomentum",            &EvalWeights::wMomentum)
        .def_readwrite("wSpace",               &EvalWeights::wSpace)
        .def_readwrite("freeze_phase_default", &EvalWeights::freeze_phase_default)
        .def_readwrite("freeze_phase_blizzard",&EvalWeights::freeze_phase_blizzard)
        .def_readwrite("ring_phase_default",   &EvalWeights::ring_phase_default)
        .def_readwrite("ring_phase_fortress",  &EvalWeights::ring_phase_fortress);

        m.def("get_default_eval_weights", &get_default_eval_weights_cpp);
        m.def("get_tuned_eval_weights",   &get_tuned_eval_weights_cpp);
        m.def("set_tuned_eval_weights",   &set_tuned_eval_weights_cpp);
        m.def("reset_tuned_eval_weights", &reset_tuned_eval_weights_cpp);

    // ------------------------------
    // 3. Bind Hard AI & encoders
    // ------------------------------
    m.def(
        "choose_hard_move_cpp",
        &choose_hard_move_py,
        py::arg("env"),
        py::arg("difficulty") = "hard"
    );

    m.def(
        "encode_action_cpp",
        &encode_action_cpp,
        py::arg("env"),
        py::arg("action")
    );

    m.def(
        "legal_action_indices_cpp",
        &legal_action_indices_cpp,
        py::arg("env")
    );

    m.def(
        "pv_mcts_search_cpp",
        &pv_mcts_search_cpp,
        py::arg("env"),
        py::arg("eval_fn"),
        py::arg("num_simulations"),
        py::arg("c_puct") = 1.5,
        py::arg("add_root_noise") = false,
        py::arg("dirichlet_alpha") = 0.3,
        py::arg("dirichlet_eps") = 0.25,
        "C++ PV-MCTS search: eval_fn(env) -> (policy_vector, value); "
        "returns dict {action -> visit_count}"
    );

    // ------------------------------
    // 4. Dataset generator
    // ------------------------------
    m.def(
        "generate_teacher_dataset_cpp",
        &generate_teacher_dataset_cpp,
        py::arg("num_games"),
        py::arg("max_moves"),
        py::arg("difficulty"),
        "Generate (X, Pi, Z) from Hard-AI vs Hard-AI self-play in C++"
    );
}
