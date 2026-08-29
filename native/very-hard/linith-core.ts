/**
 * Linith's browser-native rules kernel.
 *
 * This file is AssemblyScript, not application TypeScript.  It deliberately
 * has no dependency on the old Epsilon C++ port: that port implements several
 * historical rule variants.  The exported API is intentionally small so a
 * Web Worker can keep one instance alive and run an entire search without
 * crossing the JS/Wasm boundary at every node.
 *
 * Build with scripts/build-wasm.mjs in this directory.
 */

import {
  personalityContainment,
  personalityDevelopment,
  personalityEarlyStone,
  personalityFragmentation,
  personalityFreezeUrgency,
  personalityLibertyBalance,
  personalityMobility,
  personalitySelfPreservation,
  personalitySacrificeTolerance,
  personalityStructure,
  personalityTerritory
} from "./personality.generated";

const BOARD_CELLS: i32 = 100;
const MAX_ACTIONS: i32 = 1216; // 100 stones + 100 Swans + 2 * 63 groups * 8 dirs
const MAX_PLY: i32 = 24;

const EMPTY: u8 = 0;
const SWAN_SUN: u8 = 1;
const SWAN_MOON: u8 = 2;
const STONE: u8 = 3;
const FROZEN_SUN: u8 = 4;
const FROZEN_MOON: u8 = 5;

const SUN: i32 = 1;
const MOON: i32 = 2;

const ACTION_STONE: u8 = 0;
const ACTION_SWAN: u8 = 1;
const ACTION_MOVE: u8 = 2;
const ACTION_PUSH: u8 = 3;

// Must match rulesEngine.DIRECTIONS exactly (action ordering is compatibility data).
const DIR_R = StaticArray.fromArray<i8>([-1, 1, 0, 0, -1, -1, 1, 1]);
const DIR_C = StaticArray.fromArray<i8>([0, 0, -1, 1, -1, 1, -1, 1]);
const ALL_DIR_R = StaticArray.fromArray<i8>([-1, -1, -1, 0, 0, 1, 1, 1]);
const ALL_DIR_C = StaticArray.fromArray<i8>([-1, 0, 1, -1, 1, -1, 0, 1]);
const ORTH_R = StaticArray.fromArray<i8>([-1, 1, 0, 0]);
const ORTH_C = StaticArray.fromArray<i8>([0, 0, -1, 1]);

const board = new StaticArray<u8>(BOARD_CELLS);
let currentPlayer: i32 = SUN;
let actionsLeft: i32 = 1;
let gameOutcome: i32 = 0; // 0 ongoing, 1 Sun, 2 Moon, 3 draw

// Action lists are stored per ply, allowing recursive perft/search without
// allocating objects or regenerating a parent's action list.
const actionCounts = new StaticArray<i32>(MAX_PLY);
const actionTypes = new StaticArray<u8>(MAX_PLY * MAX_ACTIONS);
const actionDirs = new StaticArray<u8>(MAX_PLY * MAX_ACTIONS);
const actionCells = new StaticArray<u8>(MAX_PLY * MAX_ACTIONS);
const actionMaskLo = new StaticArray<u64>(MAX_PLY * MAX_ACTIONS);
const actionMaskHi = new StaticArray<u64>(MAX_PLY * MAX_ACTIONS);

// Whole-position snapshots are only 103 bytes. Copying these into a fixed undo
// stack is faster and much simpler than heap allocation, and is ready to be
// replaced by a delta stack if profiling ever shows it matters.
const undoBoards = new StaticArray<u8>(MAX_PLY * BOARD_CELLS);
const undoCurrent = new StaticArray<i32>(MAX_PLY);
const undoMoves = new StaticArray<i32>(MAX_PLY);
const undoOutcome = new StaticArray<i32>(MAX_PLY);
const undoOpponentLoss = new StaticArray<i32>(MAX_PLY);
let undoDepth: i32 = 0;

// Fixed scratch space. Linith has at most six Swans per player in legal play.
const activeCells = new StaticArray<u8>(6);
const groupCells = new StaticArray<u8>(BOARD_CELLS);
const groupStack = new StaticArray<u8>(BOARD_CELLS);

// The worker supplies performance.now. Search checks it only periodically, so
// wall-clock enforcement does not turn into a per-node JS/Wasm call.
@external("env", "now")
declare function monotonicNow(): f64;

const MATE_SCORE: i32 = 1_000_000_000;
const SEARCH_INFINITY: i32 = 1_000_000_001;
const TT_SIZE: i32 = 65_536;
const TT_MASK: u64 = 65_535;
const TT_EXACT: u8 = 1;
const TT_LOWER: u8 = 2;
const TT_UPPER: u8 = 3;

// Persistent, direct-mapped transposition table. A generation byte avoids a
// full clear between searches while allowing completed-turn continuations to
// reuse deep entries.
const ttHashes = new StaticArray<u64>(TT_SIZE);
const ttScores = new StaticArray<i32>(TT_SIZE);
const ttDepths = new StaticArray<i8>(TT_SIZE);
const ttFlags = new StaticArray<u8>(TT_SIZE);
const ttBestActions = new StaticArray<u64>(TT_SIZE);
const ttGenerations = new StaticArray<u8>(TT_SIZE);
let ttGeneration: u8 = 1;

const orderedIndexes = new StaticArray<u16>(MAX_PLY * MAX_ACTIONS);
const orderedScores = new StaticArray<i32>(MAX_PLY * MAX_ACTIONS);
const actionTactical = new StaticArray<u8>(MAX_PLY * MAX_ACTIONS);
const rootOrderedIndexes = new StaticArray<u16>(MAX_ACTIONS);
const rootTactical = new StaticArray<u8>(MAX_ACTIONS);
const rootDepthScores = new StaticArray<i32>(MAX_ACTIONS);
const rootPersonalityBonuses = new StaticArray<i32>(MAX_ACTIONS);
const rootDepthObjectiveScores = new StaticArray<i32>(MAX_ACTIONS);
const pathHashes = new StaticArray<u64>(MAX_PLY);
const ROOT_HISTORY_SIZE: i32 = 32;
const rootHistoryHashes = new StaticArray<u64>(ROOT_HISTORY_SIZE);
const rootHistoryActions = new StaticArray<u64>(ROOT_HISTORY_SIZE);
const rootHistoryResults = new StaticArray<u64>(ROOT_HISTORY_SIZE);
let rootHistoryCursor: i32 = 0;
let rootHistoryCount: i32 = 0;
let repeatedRootAction: u64 = 0;
let activeRootHistoryHash: u64 = 0;
let rootPositionRepeated: bool = false;

// The searched principal variation is exposed only while the same player keeps
// the turn. A browser worker can therefore play the already-searched second
// scheduled action (and any bonus chain) without another search boundary.
const continuationTypes = new StaticArray<u8>(MAX_PLY);
const continuationDirs = new StaticArray<u8>(MAX_PLY);
const continuationCells = new StaticArray<u8>(MAX_PLY);
const continuationMaskLo = new StaticArray<u64>(MAX_PLY);
const continuationMaskHi = new StaticArray<u64>(MAX_PLY);
const continuationInputHashes = new StaticArray<u64>(MAX_PLY);
const continuationPostHashes = new StaticArray<u64>(MAX_PLY);
let searchContinuationCount: i32 = 0;
let searchRootPostHash: u64 = 0;
let searchRootActionCount: i32 = 0;

function priorRootResult(hash: u64): bool {
  for (let ago = 1; ago <= rootHistoryCount; ago++) {
    const slot = (rootHistoryCursor - ago + ROOT_HISTORY_SIZE) % ROOT_HISTORY_SIZE;
    if (unchecked(rootHistoryResults[slot]) == hash) return true;
  }
  return false;
}

function priorObservedPositionCount(hash: u64): i32 {
  let count = 0;
  for (let ago = 1; ago <= rootHistoryCount; ago++) {
    const slot = (rootHistoryCursor - ago + ROOT_HISTORY_SIZE) % ROOT_HISTORY_SIZE;
    if (unchecked(rootHistoryHashes[slot]) == hash) count++;
    if (unchecked(rootHistoryResults[slot]) == hash) count++;
  }
  return count;
}

/**
 * Count prior occurrences of a just-applied child while action ordering still
 * has the complete legal list. `pathHashes[0]` and `activeRootHistoryHash`
 * describe the same current search root, so only deeper ancestors are added.
 * This is deliberately used before selective width: otherwise an opponent's
 * quiet repetition-closing reply can be discarded before alphaBeta gets a
 * chance to apply the history score.
 */
function observedResultCount(hash: u64, parentPly: i32): i32 {
  let count = priorObservedPositionCount(hash);
  if (hash == activeRootHistoryHash) count++;
  for (let ancestor = 1; ancestor <= parentPly; ancestor++) {
    if (unchecked(pathHashes[ancestor]) == hash) count++;
  }
  return count;
}

let sameTurnHistoryActions: i32 = 0;
const SAME_TURN_HISTORY_ACTION_CAP: i32 = 12_000;

/**
 * Determine whether a translation can finish an exact played/search-history
 * cycle later in the same uninterrupted turn. A Stone or Swan placement is
 * irreversible, and a freeze changes an active Swan irreversibly, so neither
 * can occur in an exact-position restoration. Restricting the probe to
 * non-freezing moves/pushes keeps it bounded while covering every possible
 * exact cycle under the live rules.
 */
function reachesObservedPositionWithinTurn(ply: i32, actor: i32, ancestorPly: i32): bool {
  if (ply >= MAX_PLY - 1 || currentPlayer != actor || gameOutcome != 0) return false;
  const count = generateAt(ply);
  searchGenerated += count;
  for (let index = 0; index < count; index++) {
    if (sameTurnHistoryActions >= SAME_TURN_HISTORY_ACTION_CAP) return false;
    const type = unchecked(actionTypes[actionOffset(ply, index)]);
    if (type != ACTION_MOVE && type != ACTION_PUSH) continue;
    sameTurnHistoryActions++;
    if ((sameTurnHistoryActions & 31) == 0 && searchDeadline > 0.0 && monotonicNow() >= searchDeadline) {
      searchStopped = true;
      searchStopReason = 1;
      return false;
    }
    searchClassificationActions++;
    if (!applyAt(ply, index)) continue;
    const opponentLoss = lastOpponentLoss;
    const repeats = gameOutcome == 0 && observedResultCount(hashPosition(), ancestorPly) > 0;
    undo_position();
    if (repeats && opponentLoss == 0) return true;
  }
  return false;
}

let lastOpponentLoss: i32 = 0;
let searchRootPlayer: i32 = SUN;
let searchStyle: i32 = 0;
let searchTacticalDepth: i32 = 1;
let configuredTacticalDepth: i32 = 1;
let configuredExactDepth: i32 = 0;
let searchDeadline: f64 = 0.0;
let searchNodeLimit: i32 = 0;
let searchNodes: i32 = 0;
let searchGenerated: i32 = 0;
let searchEvaluations: i32 = 0;
let searchTtHits: i32 = 0;
let searchCutoffs: i32 = 0;
let searchReSearches: i32 = 0;
let searchRootHistoryHits: i32 = 0;
let searchClassificationActions: i32 = 0;
let searchQuiescenceForcedActions: i32 = 0;
let searchRootThreatDetected: i32 = 0;
let searchExactExtensions: i32 = 0;
let searchExactSolved: i32 = 0;
let searchReturnedSolved: bool = false;
let searchReturnedHistoryDependent: bool = false;
let searchRootIterationSolved: bool = false;
let searchCompletedDepth: i32 = 0;
let searchAttemptedDepth: i32 = 0;
let searchBestRootIndex: i32 = -1;
let searchBestScore: i32 = 0;
let searchBestObjectiveScore: i32 = 0;
let searchStrongestObjectiveScore: i32 = 0;
let searchBestPersonalityBonus: i32 = 0;
let searchStopReason: i32 = 0; // 0 max depth, 1 deadline, 2 node budget
let searchStopped: bool = false;

@inline
function inBounds(r: i32, c: i32): bool {
  return r >= 0 && r < 10 && c >= 0 && c < 10;
}

@inline
function cellAt(r: i32, c: i32): i32 {
  return r * 10 + c;
}

@inline
function rowOf(cell: i32): i32 {
  return cell / 10;
}

@inline
function colOf(cell: i32): i32 {
  return cell % 10;
}

@inline
function tileAt(cell: i32): u8 {
  return unchecked(board[cell]);
}

@inline
function activeTile(player: i32): u8 {
  return player == SUN ? SWAN_SUN : SWAN_MOON;
}

@inline
function frozenTile(player: i32): u8 {
  return player == SUN ? FROZEN_SUN : FROZEN_MOON;
}

@inline
function other(player: i32): i32 {
  return player == SUN ? MOON : SUN;
}

@inline
function isActiveFor(tile: u8, player: i32): bool {
  return tile == activeTile(player);
}

@inline
function isSwanFor(tile: u8, player: i32): bool {
  return tile == activeTile(player) || tile == frozenTile(player);
}

@inline
function maskHas(lo: u64, hi: u64, cell: i32): bool {
  if (cell < 64) return (lo & (u64(1) << cell)) != 0;
  return (hi & (u64(1) << (cell - 64))) != 0;
}

@inline
function addLow(lo: u64, cell: i32): u64 {
  return cell < 64 ? lo | (u64(1) << cell) : lo;
}

@inline
function addHigh(hi: u64, cell: i32): u64 {
  return cell >= 64 ? hi | (u64(1) << (cell - 64)) : hi;
}

function countTotal(player: i32): i32 {
  const active = activeTile(player);
  const frozen = frozenTile(player);
  let count = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    const tile = tileAt(cell);
    if (tile == active || tile == frozen) count++;
  }
  return count;
}

function countActive(player: i32): i32 {
  const target = activeTile(player);
  let count = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) == target) count++;
  }
  return count;
}

function anyEmpty(): bool {
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) == EMPTY) return true;
  }
  return false;
}

function countEmptyCells(): i32 {
  let empty = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) if (tileAt(cell) == EMPTY) empty++;
  return empty;
}

function legalSwanPlacement(player: i32, cell: i32): bool {
  if (tileAt(cell) != EMPTY || countTotal(player) >= 6) return false;
  const r = rowOf(cell);
  const c = colOf(cell);
  let touchesFriendly = false;
  for (let d = 0; d < 4; d++) {
    const nr = r + unchecked(ORTH_R[d]);
    const nc = c + unchecked(ORTH_C[d]);
    if (inBounds(nr, nc) && isSwanFor(tileAt(cellAt(nr, nc)), player)) touchesFriendly = true;
  }
  if (!touchesFriendly) return false;
  const enemy = other(player);
  for (let d = 0; d < 8; d++) {
    const nr = r + unchecked(ALL_DIR_R[d]);
    const nc = c + unchecked(ALL_DIR_C[d]);
    if (inBounds(nr, nc) && isSwanFor(tileAt(cellAt(nr, nc)), enemy)) return false;
  }
  return true;
}

function nakedEnemyZone(player: i32, r: i32, c: i32): bool {
  const enemy = other(player);
  for (let d = 0; d < 8; d++) {
    const er = r + unchecked(ALL_DIR_R[d]);
    const ec = c + unchecked(ALL_DIR_C[d]);
    if (!inBounds(er, ec) || !isSwanFor(tileAt(cellAt(er, ec)), enemy)) continue;
    let hasStone = false;
    for (let s = 0; s < 8; s++) {
      const sr = er + unchecked(ALL_DIR_R[s]);
      const sc = ec + unchecked(ALL_DIR_C[s]);
      if (inBounds(sr, sc) && tileAt(cellAt(sr, sc)) == STONE) {
        hasStone = true;
        break;
      }
    }
    if (!hasStone) return true;
  }
  return false;
}

/** Build the following-Stone mask. An impossible translation returns false. */
function followingStones(
  movingPlayer: i32,
  movingLo: u64,
  movingHi: u64,
  dr: i32,
  dc: i32,
  result: StaticArray<u64>
): bool {
  let stonesLo: u64 = 0;
  let stonesHi: u64 = 0;
  for (let movingCell = 0; movingCell < BOARD_CELLS; movingCell++) {
    if (!maskHas(movingLo, movingHi, movingCell)) continue;
    const mr = rowOf(movingCell);
    const mc = colOf(movingCell);
    for (let d = 0; d < 8; d++) {
      const sr = mr + unchecked(ALL_DIR_R[d]);
      const sc = mc + unchecked(ALL_DIR_C[d]);
      if (!inBounds(sr, sc)) continue;
      const stoneCell = cellAt(sr, sc);
      if (tileAt(stoneCell) != STONE || maskHas(stonesLo, stonesHi, stoneCell)) continue;

      let shared = false;
      for (let a = 0; a < 8; a++) {
        const ar = sr + unchecked(ALL_DIR_R[a]);
        const ac = sc + unchecked(ALL_DIR_C[a]);
        if (!inBounds(ar, ac)) continue;
        const adjacentCell = cellAt(ar, ac);
        const adjacent = tileAt(adjacentCell);
        if (isSwanFor(adjacent, other(movingPlayer)) ||
            (isSwanFor(adjacent, movingPlayer) && !maskHas(movingLo, movingHi, adjacentCell))) {
          shared = true;
          break;
        }
      }
      if (!shared) {
        stonesLo = addLow(stonesLo, stoneCell);
        stonesHi = addHigh(stonesHi, stoneCell);
      }
    }
  }

  for (let stoneCell = 0; stoneCell < BOARD_CELLS; stoneCell++) {
    if (!maskHas(stonesLo, stonesHi, stoneCell)) continue;
    const tr = rowOf(stoneCell) + dr;
    const tc = colOf(stoneCell) + dc;
    if (!inBounds(tr, tc)) return false;
    const destination = cellAt(tr, tc);
    const tile = tileAt(destination);
    if (tile != EMPTY && !maskHas(movingLo, movingHi, destination) && !maskHas(stonesLo, stonesHi, destination)) {
      return false;
    }
  }
  unchecked(result[0] = stonesLo);
  unchecked(result[1] = stonesHi);
  return true;
}

const stoneScratch = new StaticArray<u64>(2);

function legalTranslation(player: i32, movingLo: u64, movingHi: u64, dir: i32, pushing: bool): bool {
  const dr = i32(unchecked(DIR_R[dir]));
  const dc = i32(unchecked(DIR_C[dir]));
  const movingPlayer = pushing ? other(player) : player;
  if (!followingStones(movingPlayer, movingLo, movingHi, dr, dc, stoneScratch)) return false;
  const stonesLo = unchecked(stoneScratch[0]);
  const stonesHi = unchecked(stoneScratch[1]);
  let destinationLo: u64 = 0;
  let destinationHi: u64 = 0;

  for (let movingCell = 0; movingCell < BOARD_CELLS; movingCell++) {
    if (!maskHas(movingLo, movingHi, movingCell)) continue;
    const r = rowOf(movingCell);
    const c = colOf(movingCell);
    const tr = r + dr;
    const tc = c + dc;
    if (!inBounds(tr, tc) || (!pushing && nakedEnemyZone(player, tr, tc))) return false;
    const destination = cellAt(tr, tc);
    if (maskHas(destinationLo, destinationHi, destination)) return false;
    destinationLo = addLow(destinationLo, destination);
    destinationHi = addHigh(destinationHi, destination);
    const tile = tileAt(destination);
    if (tile != EMPTY && !maskHas(movingLo, movingHi, destination) && !maskHas(stonesLo, stonesHi, destination)) {
      return false;
    }
  }

  // A translated following Stone may not land on a translated Swan.
  for (let stoneCell = 0; stoneCell < BOARD_CELLS; stoneCell++) {
    if (!maskHas(stonesLo, stonesHi, stoneCell)) continue;
    const destination = cellAt(rowOf(stoneCell) + dr, colOf(stoneCell) + dc);
    if (maskHas(destinationLo, destinationHi, destination)) return false;
  }
  return true;
}

function pusherPresent(player: i32, movingLo: u64, movingHi: u64): bool {
  const actor = activeTile(player);
  for (let movingCell = 0; movingCell < BOARD_CELLS; movingCell++) {
    if (!maskHas(movingLo, movingHi, movingCell)) continue;
    const r = rowOf(movingCell);
    const c = colOf(movingCell);
    let found = false;
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (inBounds(nr, nc) && tileAt(cellAt(nr, nc)) == actor) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

function appendAction(ply: i32, type: u8, cell: i32, dir: i32, lo: u64, hi: u64): void {
  const count = unchecked(actionCounts[ply]);
  if (count >= MAX_ACTIONS) unreachable();
  const offset = ply * MAX_ACTIONS + count;
  unchecked(actionTypes[offset] = type);
  unchecked(actionCells[offset] = u8(cell));
  unchecked(actionDirs[offset] = u8(dir));
  unchecked(actionMaskLo[offset] = lo);
  unchecked(actionMaskHi[offset] = hi);
  unchecked(actionCounts[ply] = count + 1);
}

function collectActive(player: i32): i32 {
  const target = activeTile(player);
  let count = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) == target) {
      // Legal states never exceed six. Trap corrupted inputs rather than write past scratch.
      if (count >= 6) unreachable();
      unchecked(activeCells[count++] = u8(cell));
    }
  }
  return count;
}

function generateAt(ply: i32): i32 {
  if (ply < 0 || ply >= MAX_PLY || gameOutcome != 0) return 0;
  unchecked(actionCounts[ply] = 0);
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) == EMPTY) appendAction(ply, ACTION_STONE, cell, 0, 0, 0);
  }
  if (countTotal(currentPlayer) < 6) {
    for (let cell = 0; cell < BOARD_CELLS; cell++) {
      if (legalSwanPlacement(currentPlayer, cell)) appendAction(ply, ACTION_SWAN, cell, 0, 0, 0);
    }
  }

  let count = collectActive(currentPlayer);
  const subsets = 1 << count;
  for (let subset = 1; subset < subsets; subset++) {
    let lo: u64 = 0;
    let hi: u64 = 0;
    for (let i = 0; i < count; i++) {
      if ((subset & (1 << i)) == 0) continue;
      const cell = i32(unchecked(activeCells[i]));
      lo = addLow(lo, cell);
      hi = addHigh(hi, cell);
    }
    for (let dir = 0; dir < 8; dir++) {
      if (legalTranslation(currentPlayer, lo, hi, dir, false)) {
        appendAction(ply, ACTION_MOVE, 0, dir, lo, hi);
      }
    }
  }

  count = collectActive(other(currentPlayer));
  const enemySubsets = 1 << count;
  for (let subset = 1; subset < enemySubsets; subset++) {
    let lo: u64 = 0;
    let hi: u64 = 0;
    for (let i = 0; i < count; i++) {
      if ((subset & (1 << i)) == 0) continue;
      const cell = i32(unchecked(activeCells[i]));
      lo = addLow(lo, cell);
      hi = addHigh(hi, cell);
    }
    if (!pusherPresent(currentPlayer, lo, hi)) continue;
    for (let dir = 0; dir < 8; dir++) {
      if (legalTranslation(currentPlayer, lo, hi, dir, true)) {
        appendAction(ply, ACTION_PUSH, 0, dir, lo, hi);
      }
    }
  }
  return unchecked(actionCounts[ply]);
}

function pushUndo(): bool {
  if (undoDepth >= MAX_PLY) return false;
  const offset = undoDepth * BOARD_CELLS;
  for (let cell = 0; cell < BOARD_CELLS; cell++) unchecked(undoBoards[offset + cell] = tileAt(cell));
  unchecked(undoCurrent[undoDepth] = currentPlayer);
  unchecked(undoMoves[undoDepth] = actionsLeft);
  unchecked(undoOutcome[undoDepth] = gameOutcome);
  unchecked(undoOpponentLoss[undoDepth] = lastOpponentLoss);
  undoDepth++;
  return true;
}

function translate(player: i32, movingLo: u64, movingHi: u64, dir: i32, pushing: bool): void {
  const dr = i32(unchecked(DIR_R[dir]));
  const dc = i32(unchecked(DIR_C[dir]));
  const movingPlayer = pushing ? other(player) : player;
  followingStones(movingPlayer, movingLo, movingHi, dr, dc, stoneScratch);
  const stonesLo = unchecked(stoneScratch[0]);
  const stonesHi = unchecked(stoneScratch[1]);

  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (maskHas(movingLo, movingHi, cell) || maskHas(stonesLo, stonesHi, cell)) unchecked(board[cell] = EMPTY);
  }
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (maskHas(stonesLo, stonesHi, cell)) {
      unchecked(board[cellAt(rowOf(cell) + dr, colOf(cell) + dc)] = STONE);
    }
  }
  const swan = activeTile(movingPlayer);
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (maskHas(movingLo, movingHi, cell)) {
      unchecked(board[cellAt(rowOf(cell) + dr, colOf(cell) + dc)] = swan);
    }
  }
}

/** Returns [frozeSun, frozeMoon, sealedSun, sealedMoon] packed in 8-bit lanes. */
function computeFreezes(): u32 {
  let seenLo: u64 = 0;
  let seenHi: u64 = 0;
  let frozeSun = 0;
  let frozeMoon = 0;
  let sealedSun = 0;
  let sealedMoon = 0;

  for (let start = 0; start < BOARD_CELLS; start++) {
    const startTile = tileAt(start);
    if ((startTile != SWAN_SUN && startTile != SWAN_MOON) || maskHas(seenLo, seenHi, start)) continue;
    const owner = startTile == SWAN_SUN ? SUN : MOON;
    const target = activeTile(owner);
    let stackSize = 0;
    let groupSize = 0;
    unchecked(groupStack[stackSize++] = u8(start));
    seenLo = addLow(seenLo, start);
    seenHi = addHigh(seenHi, start);

    while (stackSize > 0) {
      const cell = i32(unchecked(groupStack[--stackSize]));
      unchecked(groupCells[groupSize++] = u8(cell));
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (!inBounds(nr, nc)) continue;
        const next = cellAt(nr, nc);
        if (tileAt(next) == target && !maskHas(seenLo, seenHi, next)) {
          seenLo = addLow(seenLo, next);
          seenHi = addHigh(seenHi, next);
          unchecked(groupStack[stackSize++] = u8(next));
        }
      }
    }

    let encircled = true;
    for (let i = 0; i < groupSize && encircled; i++) {
      const cell = i32(unchecked(groupCells[i]));
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (inBounds(nr, nc) && tileAt(cellAt(nr, nc)) == EMPTY) {
          encircled = false;
          break;
        }
      }
    }
    if (!encircled) continue;

    const remaining = countActive(owner);
    for (let i = 0; i < groupSize; i++) unchecked(board[i32(unchecked(groupCells[i]))] = frozenTile(owner));
    if (remaining == groupSize) {
      if (owner == SUN) sealedSun += groupSize;
      else sealedMoon += groupSize;
    } else if (owner == SUN) frozeSun += groupSize;
    else frozeMoon += groupSize;
  }
  return u32(frozeSun) | (u32(frozeMoon) << 8) | (u32(sealedSun) << 16) | (u32(sealedMoon) << 24);
}

function applyAt(ply: i32, index: i32): bool {
  const count = unchecked(actionCounts[ply]);
  if (index < 0 || index >= count || !pushUndo()) return false;
  const offset = ply * MAX_ACTIONS + index;
  const type = unchecked(actionTypes[offset]);
  const actor = currentPlayer;
  if (type == ACTION_STONE) {
    unchecked(board[i32(unchecked(actionCells[offset]))] = STONE);
  } else if (type == ACTION_SWAN) {
    unchecked(board[i32(unchecked(actionCells[offset]))] = activeTile(actor));
  } else {
    translate(actor, unchecked(actionMaskLo[offset]), unchecked(actionMaskHi[offset]), i32(unchecked(actionDirs[offset])), type == ACTION_PUSH);
  }

  const freeze = computeFreezes();
  const frozeSun = i32(freeze & 0xff);
  const frozeMoon = i32((freeze >> 8) & 0xff);
  const sealedSun = i32((freeze >> 16) & 0xff);
  const sealedMoon = i32((freeze >> 24) & 0xff);
  if (sealedSun > 0 && sealedMoon > 0) gameOutcome = 3;
  else if (sealedSun > 0) gameOutcome = MOON;
  else if (sealedMoon > 0) gameOutcome = SUN;
  else {
    const activeSun = countActive(SUN);
    const activeMoon = countActive(MOON);
    if (activeSun == 0 && activeMoon == 0) gameOutcome = 3;
    else if (activeMoon == 0) gameOutcome = SUN;
    else if (activeSun == 0) gameOutcome = MOON;
    else if (!anyEmpty()) gameOutcome = 3;
    else gameOutcome = 0;
  }

  const opponentLoss = actor == SUN ? frozeMoon + sealedMoon : frozeSun + sealedSun;
  lastOpponentLoss = opponentLoss;
  const remaining = actionsLeft + opponentLoss - 1;
  if (remaining > 0) {
    actionsLeft = remaining;
  } else {
    currentPlayer = other(actor);
    actionsLeft = countTotal(SUN) >= 6 && countTotal(MOON) >= 6 ? 2 : 1;
  }
  return true;
}

function perftAt(depth: i32, ply: i32): u64 {
  if (depth <= 0 || gameOutcome != 0) return 1;
  const count = generateAt(ply);
  if (count == 0) return 1;
  let nodes: u64 = 0;
  for (let i = 0; i < count; i++) {
    if (!applyAt(ply, i)) continue;
    nodes += perftAt(depth - 1, ply + 1);
    undo_position();
  }
  return nodes;
}

// -------------------------- classical search core --------------------------

function hashPosition(): u64 {
  let hash: u64 = 14695981039346656037;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    hash ^= u64(tileAt(cell));
    hash *= 1099511628211;
  }
  hash ^= u64(currentPlayer);
  hash *= 1099511628211;
  hash ^= u64(actionsLeft);
  hash *= 1099511628211;
  hash ^= u64(gameOutcome);
  return hash * 1099511628211;
}

function hashSearchPosition(): u64 {
  let hash = hashPosition();
  hash = (hash ^ u64(searchRootPlayer)) * 1099511628211;
  hash = (hash ^ u64(searchStyle + 17)) * 1099511628211;
  hash = (hash ^ u64(configuredTacticalDepth + 31)) * 1099511628211;
  hash = (hash ^ u64(configuredExactDepth + 47)) * 1099511628211;
  return hash;
}

@inline
function scoreToTransposition(score: i32, ply: i32): i32 {
  if (score > MATE_SCORE / 2) return score + ply;
  if (score < -MATE_SCORE / 2) return score - ply;
  return score;
}

@inline
function scoreFromTransposition(score: i32, ply: i32): i32 {
  if (score > MATE_SCORE / 2) return score - ply;
  if (score < -MATE_SCORE / 2) return score + ply;
  return score;
}

@inline
function actionOffset(ply: i32, index: i32): i32 {
  return ply * MAX_ACTIONS + index;
}

function actionSignature(ply: i32, index: i32): u64 {
  const offset = actionOffset(ply, index);
  let hash: u64 = 14695981039346656037;
  hash = (hash ^ u64(unchecked(actionTypes[offset]))) * 1099511628211;
  hash = (hash ^ u64(unchecked(actionCells[offset]))) * 1099511628211;
  hash = (hash ^ u64(unchecked(actionDirs[offset]))) * 1099511628211;
  hash = (hash ^ unchecked(actionMaskLo[offset])) * 1099511628211;
  hash = (hash ^ unchecked(actionMaskHi[offset])) * 1099511628211;
  return hash == 0 ? 1 : hash;
}

function libertiesFor(player: i32): i32 {
  const target = activeTile(player);
  let low: u64 = 0;
  let high: u64 = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) != target) continue;
    const r = rowOf(cell);
    const c = colOf(cell);
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (!inBounds(nr, nc)) continue;
      const next = cellAt(nr, nc);
      if (tileAt(next) == EMPTY) {
        low = addLow(low, next);
        high = addHigh(high, next);
      }
    }
  }
  return i32(popcnt<u64>(low) + popcnt<u64>(high));
}

function frozenCount(player: i32): i32 {
  const target = frozenTile(player);
  let count = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) if (tileAt(cell) == target) count++;
  return count;
}

function stoneContacts(player: i32): i32 {
  const target = activeTile(player);
  let contacts = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) != target) continue;
    const r = rowOf(cell);
    const c = colOf(cell);
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (inBounds(nr, nc) && tileAt(cellAt(nr, nc)) == STONE) contacts++;
    }
  }
  return contacts;
}

function centreControl(player: i32): i32 {
  const target = activeTile(player);
  let value = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) != target) continue;
    value += 18 - abs(9 - rowOf(cell) * 2) - abs(9 - colOf(cell) * 2);
  }
  return value;
}

function groupTightness(player: i32): i32 {
  const target = activeTile(player);
  let seenLo: u64 = 0;
  let seenHi: u64 = 0;
  let total = 0;
  for (let start = 0; start < BOARD_CELLS; start++) {
    if (tileAt(start) != target || maskHas(seenLo, seenHi, start)) continue;
    let stackSize = 0;
    let groupSize = 0;
    let rimLo: u64 = 0;
    let rimHi: u64 = 0;
    unchecked(groupStack[stackSize++] = u8(start));
    seenLo = addLow(seenLo, start);
    seenHi = addHigh(seenHi, start);
    while (stackSize > 0) {
      const cell = i32(unchecked(groupStack[--stackSize]));
      groupSize++;
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (!inBounds(nr, nc)) continue;
        const next = cellAt(nr, nc);
        const tile = tileAt(next);
        if (tile == EMPTY) {
          rimLo = addLow(rimLo, next);
          rimHi = addHigh(rimHi, next);
        } else if (tile == target && !maskHas(seenLo, seenHi, next)) {
          seenLo = addLow(seenLo, next);
          seenHi = addHigh(seenHi, next);
          unchecked(groupStack[stackSize++] = u8(next));
        }
      }
    }
    const liberties = i32(popcnt<u64>(rimLo) + popcnt<u64>(rimHi));
    const missing = max<i32>(0, 8 + groupSize * 2 - liberties);
    total += missing * missing;
    if (liberties <= 2) total += 80;
  }
  return total;
}

function groupUrgency(player: i32): i32 {
  const target = activeTile(player);
  let seenLo: u64 = 0;
  let seenHi: u64 = 0;
  let score = 0;
  for (let start = 0; start < BOARD_CELLS; start++) {
    if (tileAt(start) != target || maskHas(seenLo, seenHi, start)) continue;
    let stackSize = 0;
    let groupSize = 0;
    let rimLo: u64 = 0;
    let rimHi: u64 = 0;
    unchecked(groupStack[stackSize++] = u8(start));
    seenLo = addLow(seenLo, start);
    seenHi = addHigh(seenHi, start);
    while (stackSize > 0) {
      const cell = i32(unchecked(groupStack[--stackSize]));
      groupSize++;
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (!inBounds(nr, nc)) continue;
        const next = cellAt(nr, nc);
        const tile = tileAt(next);
        if (tile == EMPTY) {
          rimLo = addLow(rimLo, next);
          rimHi = addHigh(rimHi, next);
        } else if (tile == target && !maskHas(seenLo, seenHi, next)) {
          seenLo = addLow(seenLo, next);
          seenHi = addHigh(seenHi, next);
          unchecked(groupStack[stackSize++] = u8(next));
        }
      }
    }
    const liberties = i32(popcnt<u64>(rimLo) + popcnt<u64>(rimHi));
    if (liberties <= 1) score += 14_000 + groupSize * 2_500;
    else if (liberties == 2) score += 5_000 + groupSize * 900;
    else if (liberties == 3) score += 1_500 + groupSize * 250;
  }
  return score;
}

function activeGroupCount(player: i32): i32 {
  const target = activeTile(player);
  let seenLo: u64 = 0;
  let seenHi: u64 = 0;
  let groups = 0;
  for (let start = 0; start < BOARD_CELLS; start++) {
    if (tileAt(start) != target || maskHas(seenLo, seenHi, start)) continue;
    groups++;
    let stackSize = 0;
    unchecked(groupStack[stackSize++] = u8(start));
    seenLo = addLow(seenLo, start);
    seenHi = addHigh(seenHi, start);
    while (stackSize > 0) {
      const cell = i32(unchecked(groupStack[--stackSize]));
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (!inBounds(nr, nc)) continue;
        const next = cellAt(nr, nc);
        if (tileAt(next) == target && !maskHas(seenLo, seenHi, next)) {
          seenLo = addLow(seenLo, next);
          seenHi = addHigh(seenHi, next);
          unchecked(groupStack[stackSize++] = u8(next));
        }
      }
    }
  }
  return groups;
}

function localMobility(player: i32): i32 {
  const target = activeTile(player);
  let score = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) != target) continue;
    const r = rowOf(cell);
    const c = colOf(cell);
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (!inBounds(nr, nc)) continue;
      const tile = tileAt(cellAt(nr, nc));
      if (tile == EMPTY) score += 2;
      else if (tile == STONE) score += 1;
    }
  }
  return score;
}

function minimumGroupLiberties(player: i32): i32 {
  const target = activeTile(player);
  let seenLo: u64 = 0;
  let seenHi: u64 = 0;
  let minimum = 100;
  for (let start = 0; start < BOARD_CELLS; start++) {
    if (tileAt(start) != target || maskHas(seenLo, seenHi, start)) continue;
    let stackSize = 0;
    let rimLo: u64 = 0;
    let rimHi: u64 = 0;
    unchecked(groupStack[stackSize++] = u8(start));
    seenLo = addLow(seenLo, start);
    seenHi = addHigh(seenHi, start);
    while (stackSize > 0) {
      const cell = i32(unchecked(groupStack[--stackSize]));
      const r = rowOf(cell);
      const c = colOf(cell);
      for (let d = 0; d < 8; d++) {
        const nr = r + unchecked(ALL_DIR_R[d]);
        const nc = c + unchecked(ALL_DIR_C[d]);
        if (!inBounds(nr, nc)) continue;
        const next = cellAt(nr, nc);
        const tile = tileAt(next);
        if (tile == EMPTY) {
          rimLo = addLow(rimLo, next);
          rimHi = addHigh(rimHi, next);
        } else if (tile == target && !maskHas(seenLo, seenHi, next)) {
          seenLo = addLow(seenLo, next);
          seenHi = addHigh(seenHi, next);
          unchecked(groupStack[stackSize++] = u8(next));
        }
      }
    }
    minimum = min<i32>(minimum, i32(popcnt<u64>(rimLo) + popcnt<u64>(rimHi)));
  }
  return minimum;
}

function territoryBalance(mine: i32, theirs: i32): i32 {
  if (countActive(mine) == 0 || countActive(theirs) == 0) return 0;
  let score = 0;
  const myTile = activeTile(mine);
  const theirTile = activeTile(theirs);
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    if (tileAt(cell) != EMPTY) continue;
    const r = rowOf(cell);
    const c = colOf(cell);
    let myDistance = 10;
    let theirDistance = 10;
    for (let swan = 0; swan < BOARD_CELLS; swan++) {
      const tile = tileAt(swan);
      if (tile != myTile && tile != theirTile) continue;
      const distance = max<i32>(abs(rowOf(swan) - r), abs(colOf(swan) - c));
      if (tile == myTile && distance < myDistance) myDistance = distance;
      if (tile == theirTile && distance < theirDistance) theirDistance = distance;
    }
    if (myDistance < theirDistance) score += min<i32>(4, theirDistance - myDistance);
    else if (theirDistance < myDistance) score -= min<i32>(4, myDistance - theirDistance);
  }
  return score;
}

function evaluateStyledPosition(perspective: i32, style: i32): i32 {
  const opponent = other(perspective);
  const myActive = countActive(perspective);
  const theirActive = countActive(opponent);
  const myFrozen = frozenCount(perspective);
  const theirFrozen = frozenCount(opponent);
  const myLiberties = libertiesFor(perspective);
  const theirLiberties = libertiesFor(opponent);
  const frozen = theirFrozen * 18_000 - myFrozen * 19_500;
  const activity = (myActive - theirActive) * 4_800;
  const libertyScore = (myLiberties - theirLiberties) * 92;
  const contactDifference = stoneContacts(opponent) - stoneContacts(perspective);
  const pressure = (groupTightness(opponent) - groupTightness(perspective)) * 42 +
    contactDifference * 210;
  const urgency = groupUrgency(opponent) - groupUrgency(perspective);
  const territory = territoryBalance(perspective, opponent);
  const territoryScore = territory * 18;
  const totalMine = countTotal(perspective);
  const totalTheirs = countTotal(opponent);
  const development = (totalMine - totalTheirs) * 260;
  const centre = (centreControl(perspective) - centreControl(opponent)) * 6;
  const mobilityDifference = localMobility(perspective) - localMobility(opponent);
  const mobility = mobilityDifference * 115;
  const deployed = min<f64>(1.0, f64(totalMine + totalTheirs) / 12.0);
  const phaseRaw = (1.0 - deployed) * f64(development + centre) * 0.2 +
    deployed * f64(pressure + urgency) * 0.08;
  const phase = i32(Math.floor(phaseRaw + 0.5));
  const tempo = (currentPlayer == perspective ? 1 : -1) * max<i32>(1, actionsLeft) * 70;

  const libertyDifference = myLiberties - theirLiberties;
  const pressureSignal = max<f64>(-80.0, min<f64>(80.0, f64(pressure) / 100.0));
  const territorySignal = max<f64>(-60.0, min<f64>(60.0, f64(territoryScore) / 20.0));
  const urgencySignal = max<f64>(-60.0, min<f64>(60.0, f64(urgency) / 500.0));
  const fragmentation = personalityFragmentation(style);
  const fragmentationBalance = fragmentation == 0.0
    ? 0
    : activeGroupCount(opponent) - activeGroupCount(perspective);
  const styleRaw = 3.0 * (personalityFreezeUrgency(style) * f64(theirFrozen - myFrozen) * 35.0 +
    personalitySelfPreservation(style) * urgencySignal +
    personalityLibertyBalance(style) * f64(libertyDifference) * 4.0 +
    personalityContainment(style) * pressureSignal +
    personalityTerritory(style) * territorySignal +
    fragmentation * f64(fragmentationBalance) * 25.0 +
    personalityDevelopment(style) * f64(totalMine - totalTheirs) * 20.0 +
    personalityStructure(style) * f64(contactDifference) * 3.0 +
    personalityMobility(style) * f64(mobilityDifference) * 1.5);
  const styleDelta = max<i32>(-900, min<i32>(900, i32(Math.floor(styleRaw + 0.5))));
  const value = frozen + activity + libertyScore + pressure + territoryScore + development +
    centre + mobility + urgency + phase + tempo + styleDelta;
  return max<i32>(-MATE_SCORE / 4, min<i32>(MATE_SCORE / 4, value));
}

function evaluatePosition(perspective: i32): i32 {
  return evaluateStyledPosition(perspective, 0);
}

function positionStyleScore(perspective: i32, style: i32): i32 {
  if (style == 0) return 0;
  const opponent = other(perspective);
  const myFrozen = frozenCount(perspective);
  const theirFrozen = frozenCount(opponent);
  const myLiberties = libertiesFor(perspective);
  const theirLiberties = libertiesFor(opponent);
  const contactDifference = stoneContacts(opponent) - stoneContacts(perspective);
  const pressure = (groupTightness(opponent) - groupTightness(perspective)) * 42 +
    contactDifference * 210;
  const urgency = groupUrgency(opponent) - groupUrgency(perspective);
  const territoryScore = territoryBalance(perspective, opponent) * 18;
  const totalMine = countTotal(perspective);
  const totalTheirs = countTotal(opponent);
  const mobilityDifference = localMobility(perspective) - localMobility(opponent);
  const pressureSignal = max<f64>(-80.0, min<f64>(80.0, f64(pressure) / 100.0));
  const territorySignal = max<f64>(-60.0, min<f64>(60.0, f64(territoryScore) / 20.0));
  const urgencySignal = max<f64>(-60.0, min<f64>(60.0, f64(urgency) / 500.0));
  const fragmentation = personalityFragmentation(style);
  const fragmentationBalance = fragmentation == 0.0
    ? 0
    : activeGroupCount(opponent) - activeGroupCount(perspective);
  const raw = 3.0 * (personalityFreezeUrgency(style) * f64(theirFrozen - myFrozen) * 35.0 +
    personalitySelfPreservation(style) * urgencySignal +
    personalityLibertyBalance(style) * f64(myLiberties - theirLiberties) * 4.0 +
    personalityContainment(style) * pressureSignal +
    personalityTerritory(style) * territorySignal +
    fragmentation * f64(fragmentationBalance) * 25.0 +
    personalityDevelopment(style) * f64(totalMine - totalTheirs) * 20.0 +
    personalityStructure(style) * f64(contactDifference) * 3.0 +
    personalityMobility(style) * f64(mobilityDifference) * 1.5);
  return max<i32>(-900, min<i32>(900, i32(Math.floor(raw + 0.5))));
}

function prepareRootPersonalityBonuses(count: i32): void {
  for (let index = 0; index < count; index++) unchecked(rootPersonalityBonuses[index] = 0);
  if (searchStyle == 0 || count <= 0) return;
  const beforeStyle = positionStyleScore(searchRootPlayer, searchStyle);
  const undeployed = max<i32>(0, 6 - countTotal(searchRootPlayer));
  const ownFrozenBefore = frozenCount(searchRootPlayer);
  for (let index = 0; index < count; index++) {
    const type = unchecked(actionTypes[actionOffset(0, index)]);
    if (!applyAt(0, index)) continue;
    let bonus = 0;
    if (gameOutcome == 0) {
      const afterStyle = positionStyleScore(searchRootPlayer, searchStyle);
      let placement = 0.0;
      if (type == ACTION_SWAN) placement = personalityDevelopment(searchStyle) * f64(undeployed) * 40.0;
      else if (type == ACTION_STONE) placement = personalityEarlyStone(searchStyle) * f64(undeployed) * 40.0;
      const freezeIntent = personalityFreezeUrgency(searchStyle) * f64(lastOpponentLoss) * 160.0;
      const ownFrozen = max<i32>(0, frozenCount(searchRootPlayer) - ownFrozenBefore);
      const concreteSacrifice = min<i32>(lastOpponentLoss, ownFrozen);
      const sacrificeIntent = personalitySacrificeTolerance(searchStyle) * f64(concreteSacrifice) * 120.0;
      const raw = f64(afterStyle - beforeStyle) + placement + freezeIntent + sacrificeIntent;
      bonus = max<i32>(-450, min<i32>(450, i32(Math.floor(raw + 0.5))));
    }
    unchecked(rootPersonalityBonuses[index] = bonus);
    undo_position();
  }
}

/** Cheap ordering-only evaluation; avoids territory and group flood-fills. */
function fastOrderEvaluation(perspective: i32): i32 {
  const opponent = other(perspective);
  const mine = activeTile(perspective);
  const theirs = activeTile(opponent);
  const myFrozenTile = frozenTile(perspective);
  const theirFrozenTile = frozenTile(opponent);
  let myActive = 0;
  let theirActive = 0;
  let myFrozen = 0;
  let theirFrozen = 0;
  let myLiberties = 0;
  let theirLiberties = 0;
  let myContacts = 0;
  let theirContacts = 0;
  let myCentre = 0;
  let theirCentre = 0;
  for (let cell = 0; cell < BOARD_CELLS; cell++) {
    const tile = tileAt(cell);
    if (tile == myFrozenTile) { myFrozen++; continue; }
    if (tile == theirFrozenTile) { theirFrozen++; continue; }
    if (tile != mine && tile != theirs) continue;
    const isMine = tile == mine;
    if (isMine) {
      myActive++;
      myCentre += 18 - abs(9 - rowOf(cell) * 2) - abs(9 - colOf(cell) * 2);
    } else {
      theirActive++;
      theirCentre += 18 - abs(9 - rowOf(cell) * 2) - abs(9 - colOf(cell) * 2);
    }
    const r = rowOf(cell);
    const c = colOf(cell);
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (!inBounds(nr, nc)) continue;
      const neighbour = tileAt(cellAt(nr, nc));
      if (neighbour == EMPTY) {
        if (isMine) myLiberties++;
        else theirLiberties++;
      } else if (neighbour == STONE) {
        if (isMine) myContacts++;
        else theirContacts++;
      }
    }
  }
  return (theirFrozen - myFrozen) * 15_000 +
    (myActive - theirActive) * 4_000 +
    myLiberties * 72 - theirLiberties * 96 +
    (theirContacts - myContacts) * 150 +
    (myCentre - theirCentre) * 5;
}

@inline
function terminalScore(outcome: i32, ply: i32): i32 {
  if (outcome == 3 || outcome == 0) return 0;
  return outcome == searchRootPlayer ? MATE_SCORE - ply : -MATE_SCORE + ply;
}

function geometricActionScore(ply: i32, index: i32): i32 {
  const offset = actionOffset(ply, index);
  const type = unchecked(actionTypes[offset]);
  if (type == ACTION_STONE || type == ACTION_SWAN) {
    const cell = i32(unchecked(actionCells[offset]));
    const r = rowOf(cell);
    const c = colOf(cell);
    let enemyContact = 0;
    let friendlyContact = 0;
    for (let d = 0; d < 8; d++) {
      const nr = r + unchecked(ALL_DIR_R[d]);
      const nc = c + unchecked(ALL_DIR_C[d]);
      if (!inBounds(nr, nc)) continue;
      const tile = tileAt(cellAt(nr, nc));
      if (isSwanFor(tile, other(currentPlayer))) enemyContact++;
      if (isActiveFor(tile, currentPlayer)) friendlyContact++;
    }
    const centre = 18 - abs(9 - r * 2) - abs(9 - c * 2);
    return type == ACTION_STONE
      ? 50_000 + enemyContact * 12_000 + friendlyContact * 600 + centre
      : 35_000 + friendlyContact * 1_000 + centre * 3;
  }
  const groupSize = i32(popcnt<u64>(unchecked(actionMaskLo[offset])) + popcnt<u64>(unchecked(actionMaskHi[offset])));
  return (type == ACTION_PUSH ? 65_000 : 45_000) + groupSize * 1_500;
}

let turnThreatActions: i32 = 0;
const TURN_THREAT_ACTION_CAP: i32 = 12_000;

/**
 * Existential, bounded solver for an uninterrupted scheduled/bonus-action
 * chain. It scans every immediate action, but only extends quiet setup actions
 * that measurably reduce the target's minimum liberties. This catches the
 * common quiet Stone -> seal and freeze -> bonus -> seal patterns without
 * opening a second full game tree inside every horizon node.
 */
function winWithinCurrentTurnAt(ply: i32, player: i32, remainingActions: i32): i32 {
  if (remainingActions <= 0 || ply >= MAX_PLY - 1 || currentPlayer != player || gameOutcome != 0) return 0;
  const savedOpponentLoss = lastOpponentLoss;
  const targetMinimumBefore = minimumGroupLiberties(other(player));
  const count = generateAt(ply);
  searchGenerated += count;
  for (let index = 0; index < count; index++) {
    if (turnThreatActions >= TURN_THREAT_ACTION_CAP) break;
    turnThreatActions++;
    if (searchNodes == 0) searchClassificationActions++;
    if (!applyAt(ply, index)) continue;
    const actionOpponentLoss = lastOpponentLoss;
    if (gameOutcome == player) {
      undo_position();
      lastOpponentLoss = savedOpponentLoss;
      return 1;
    }
    let continuation = 0;
    if (gameOutcome == 0 && currentPlayer == player && remainingActions > 1) {
      const targetMinimumAfter = minimumGroupLiberties(other(player));
      const progresses = actionOpponentLoss > 0 ||
        (targetMinimumAfter <= 2 && targetMinimumAfter < targetMinimumBefore);
      if (progresses) continuation = winWithinCurrentTurnAt(ply + 1, player, remainingActions - 1);
    }
    undo_position();
    lastOpponentLoss = savedOpponentLoss;
    if (continuation > 0) return continuation + 1;
  }
  lastOpponentLoss = savedOpponentLoss;
  return 0;
}

function winWithinCurrentTurn(ply: i32, player: i32, maximumActions: i32): i32 {
  turnThreatActions = 0;
  return winWithinCurrentTurnAt(ply, player, maximumActions);
}

function scoreActions(ply: i32, preferred: u64): i32 {
  const count = generateAt(ply);
  searchGenerated += count;
  const base = ply * MAX_ACTIONS;
  const actorAtNode = currentPlayer;
  const actorMinimumBefore = minimumGroupLiberties(actorAtNode);
  const opponentMinimumBefore = minimumGroupLiberties(other(actorAtNode));
  const initialRootScan = ply == 0 && searchNodes == 0;
  let rootOpponentHasImmediateWin = false;
  if (initialRootScan) {
    const savedPlayer = currentPlayer;
    const savedMoves = actionsLeft;
    currentPlayer = other(savedPlayer);
    actionsLeft = countTotal(SUN) >= 6 && countTotal(MOON) >= 6 ? 2 : 1;
    const targetMinimum = minimumGroupLiberties(savedPlayer);
    rootOpponentHasImmediateWin = winWithinCurrentTurn(1, currentPlayer, targetMinimum <= 2 ? 4 : 1) > 0;
    searchRootThreatDetected = rootOpponentHasImmediateWin ? 1 : 0;
    currentPlayer = savedPlayer;
    actionsLeft = savedMoves;
  }
  // Stage one is allocation-free and cheap. It gives the expensive positional
  // deltas a deterministic, per-class-aware candidate window instead of
  // flood-filling every quiet child in a 300-action node.
  for (let index = 0; index < count; index++) {
    unchecked(orderedIndexes[base + index] = u16(index));
    unchecked(actionTactical[base + index] = 0);
    const geometric = geometricActionScore(ply, index);
    let score = initialRootScan ? geometric / 100 : geometric;
    const signature = actionSignature(ply, index);
    if (preferred != 0 && signature == preferred) score += 500_000_000;
    if (ply == 0 && repeatedRootAction != 0 && signature == repeatedRootAction) score -= 120_000_000;
    const rootActionType = unchecked(actionTypes[actionOffset(ply, index)]);
    if (ply == 0 && rootPositionRepeated &&
        (rootActionType == ACTION_MOVE || rootActionType == ACTION_PUSH)) score -= 100_000_000;
    unchecked(orderedScores[base + index] = score);
  }
  if (count > 1) sortOrdered(base, 0, count - 1);

  // Stage two still applies every action, so immediate wins, captures, and
  // unique defenses can never disappear outside a selective width. Only the
  // leading quiet candidates pay for a positional ordering evaluation.
  for (let rank = 0; rank < count; rank++) {
    if (rank > 0 && (rank & 31) == 0 && searchNodes > 0 &&
        searchDeadline > 0.0 && monotonicNow() >= searchDeadline) {
      searchStopped = true;
      searchStopReason = 1;
      break;
    }
    const index = i32(unchecked(orderedIndexes[base + rank]));
    const actionType = unchecked(actionTypes[actionOffset(ply, index)]);
    let score = unchecked(orderedScores[base + rank]);
    const actor = currentPlayer;
    if (applyAt(ply, index)) {
      const outcome = gameOutcome;
      const actionOpponentLoss = lastOpponentLoss;
      const resultHash = hashPosition();
      let repeatsObservedPosition = outcome == 0 && observedResultCount(resultHash, ply) > 0;
      if (!repeatsObservedPosition && outcome == 0 && currentPlayer == actor &&
          (actionType == ACTION_MOVE || actionType == ACTION_PUSH) &&
          actor != searchRootPlayer && rootPositionRepeated && ply <= 2) {
        repeatsObservedPosition = reachesObservedPositionWithinTurn(ply + 1, actor, ply);
      }
      if (repeatsObservedPosition) searchRootHistoryHits++;
      let concedesImmediateWin = false;
      // The initial root scan also powers the zero-budget fallback. It examines
      // every legal reply, so a geometrically quiet only-defense cannot be
      // displaced by an attractive move that loses immediately.
      if (initialRootScan && rootOpponentHasImmediateWin && outcome == 0 && currentPlayer != actor) {
        const targetMinimum = minimumGroupLiberties(actor);
        concedesImmediateWin = winWithinCurrentTurn(1, currentPlayer, targetMinimum <= 2 ? 4 : 1) > 0;
      }
      if (outcome == actor) score += 900_000_000;
      else if (outcome != 0 && outcome != 3) score -= 900_000_000;
      else if (outcome == 3) score -= fastOrderEvaluation(actor) / 4;
      if (concedesImmediateWin) score -= 400_000_000;
      // Ordering is always descending. A repetition is unattractive for the
      // root actor, but must be searched first when the opponent can choose it.
      // The eventual value remains the root-relative -100k/-250k alphaBeta
      // history score; this large term affects ordering only.
      if (repeatsObservedPosition) {
        score += actor == searchRootPlayer ? -350_000_000 : 350_000_000;
      }
      score += actionOpponentLoss * 2_000_000;
      if (currentPlayer == actor && outcome == 0) score += 150_000;
      if (outcome == 0 && (initialRootScan || rank < 64)) score += fastOrderEvaluation(actor);
      let improvesThreat = false;
      let createsThreat = false;
      if (outcome == 0 && actionOpponentLoss == 0 &&
          (initialRootScan || rank < 96) &&
          (actorMinimumBefore <= 3 || opponentMinimumBefore <= 4)) {
        const actorMinimumAfter = minimumGroupLiberties(actor);
        const opponentMinimumAfter = minimumGroupLiberties(other(actor));
        improvesThreat = actorMinimumBefore <= 3 && actorMinimumAfter > actorMinimumBefore;
        createsThreat = opponentMinimumAfter <= 2 && opponentMinimumAfter < opponentMinimumBefore;
      }
      if (improvesThreat || createsThreat) score += 300_000;
      if (outcome != 0 || actionOpponentLoss > 0 || repeatsObservedPosition || improvesThreat || createsThreat ||
          (initialRootScan && rootOpponentHasImmediateWin && !concedesImmediateWin)) {
        unchecked(actionTactical[base + index] = 1);
      }
      undo_position();
    }
    unchecked(orderedScores[base + rank] = score);
  }
  if (count > 1 && !searchStopped) sortOrdered(base, 0, count - 1);
  return count;
}

function orderBefore(base: i32, left: i32, right: i32): bool {
  const leftScore = unchecked(orderedScores[base + left]);
  const rightScore = unchecked(orderedScores[base + right]);
  if (leftScore != rightScore) return leftScore > rightScore;
  return unchecked(orderedIndexes[base + left]) < unchecked(orderedIndexes[base + right]);
}

function swapOrder(base: i32, left: i32, right: i32): void {
  const leftIndex = unchecked(orderedIndexes[base + left]);
  const leftScore = unchecked(orderedScores[base + left]);
  unchecked(orderedIndexes[base + left] = orderedIndexes[base + right]);
  unchecked(orderedScores[base + left] = orderedScores[base + right]);
  unchecked(orderedIndexes[base + right] = leftIndex);
  unchecked(orderedScores[base + right] = leftScore);
}

function sortOrdered(base: i32, first: i32, last: i32): void {
  let left = first;
  let right = last;
  const pivotAt = (first + last) >> 1;
  const pivotScore = unchecked(orderedScores[base + pivotAt]);
  const pivotIndex = unchecked(orderedIndexes[base + pivotAt]);
  while (left <= right) {
    while (unchecked(orderedScores[base + left]) > pivotScore ||
      (unchecked(orderedScores[base + left]) == pivotScore && unchecked(orderedIndexes[base + left]) < pivotIndex)) left++;
    while (unchecked(orderedScores[base + right]) < pivotScore ||
      (unchecked(orderedScores[base + right]) == pivotScore && unchecked(orderedIndexes[base + right]) > pivotIndex)) right--;
    if (left <= right) {
      swapOrder(base, left, right);
      left++;
      right--;
    }
  }
  if (first < right) sortOrdered(base, first, right);
  if (left < last) sortOrdered(base, left, last);
}

function checkSearchNode(): bool {
  if (searchStopped) return false;
  if (searchNodes >= searchNodeLimit) {
    searchStopped = true;
    searchStopReason = 2;
    return false;
  }
  searchNodes++;
  if ((searchNodes & 255) == 0 && searchDeadline > 0.0 && monotonicNow() >= searchDeadline) {
    searchStopped = true;
    searchStopReason = 1;
    return false;
  }
  return true;
}

function quiescence(
  alphaInput: i32,
  betaInput: i32,
  remaining: i32,
  ply: i32,
  mustCompleteTurn: bool
): i32 {
  if (!checkSearchNode()) return evaluatePosition(searchRootPlayer);
  unchecked(pathHashes[ply] = hashPosition());
  searchEvaluations++;
  const standPat = evaluatePosition(searchRootPlayer);
  if ((!mustCompleteTurn && remaining <= 0) || ply >= MAX_PLY - 1) return standPat;
  const maximizing = currentPlayer == searchRootPlayer;
  let best = mustCompleteTurn ? (maximizing ? -SEARCH_INFINITY : SEARCH_INFINITY) : standPat;
  let alpha = alphaInput;
  let beta = betaInput;
  if (!mustCompleteTurn) {
    if (maximizing) {
      if (best >= beta) return best;
      if (best > alpha) alpha = best;
    } else {
      if (best <= alpha) return best;
      if (best < beta) beta = best;
    }
  }
  const count = scoreActions(ply, 0);
  const base = ply * MAX_ACTIONS;
  let children = 0;
  for (let order = 0; order < count && !searchStopped; order++) {
    const index = i32(unchecked(orderedIndexes[base + order]));
    const classifiedTactical = unchecked(actionTactical[base + index]) != 0;
    const actor = currentPlayer;
    if (!applyAt(ply, index)) continue;
    const tactical = classifiedTactical || gameOutcome != 0 || lastOpponentLoss > 0;
    if (!mustCompleteTurn && !tactical) {
      undo_position();
      continue;
    }
    if (mustCompleteTurn && !tactical) searchQuiescenceForcedActions++;
    children++;
    const sameTurn = gameOutcome == 0 && currentPlayer == actor;
    const score = gameOutcome != 0
      ? terminalScore(gameOutcome, ply + 1)
      : quiescence(alpha, beta, sameTurn ? remaining : remaining - 1, ply + 1, sameTurn);
    undo_position();
    if (maximizing) {
      if (score > best) best = score;
      if (best > alpha) alpha = best;
    } else {
      if (score < best) best = score;
      if (best < beta) beta = best;
    }
    if (alpha >= beta) {
      searchCutoffs++;
      break;
    }
  }
  return children == 0 ? standPat : best;
}

function alphaBeta(
  turnDepthInput: i32,
  alphaInput: i32,
  betaInput: i32,
  ply: i32,
  exactRemainingInput: i32
): i32 {
  searchReturnedSolved = false;
  searchReturnedHistoryDependent = false;
  if (!checkSearchNode()) return evaluatePosition(searchRootPlayer);
  if (gameOutcome != 0) {
    searchReturnedSolved = true;
    return terminalScore(gameOutcome, ply);
  }
  const repetitionKey = hashPosition();
  let observedRepetitions = priorObservedPositionCount(repetitionKey);
  if (repetitionKey == activeRootHistoryHash) observedRepetitions++;
  for (let ancestor = 0; ancestor < ply; ancestor++) {
    if (unchecked(pathHashes[ancestor]) == repetitionKey) {
      searchEvaluations++;
      searchRootHistoryHits++;
      searchReturnedHistoryDependent = true;
      const value = evaluatePosition(searchRootPlayer);
      return value - (observedRepetitions >= 2 ? 250_000 : 100_000);
    }
  }
  if (observedRepetitions > 0) {
    searchEvaluations++;
    searchRootHistoryHits++;
    searchReturnedHistoryDependent = true;
    return evaluatePosition(searchRootPlayer) - (observedRepetitions >= 2 ? 250_000 : 100_000);
  }
  unchecked(pathHashes[ply] = repetitionKey);

  let turnDepth = turnDepthInput;
  let exactRemaining = exactRemainingInput;
  if (turnDepth <= 0 && configuredExactDepth > 0 && ply < MAX_PLY - 1) {
    if (exactRemaining > 0) {
      turnDepth = 1;
      exactRemaining--;
      searchExactExtensions++;
    } else if (exactRemaining < 0 && countEmptyCells() <= 6) {
      const exactCount = generateAt(ply);
      searchGenerated += exactCount;
      if (exactCount <= 12) {
        turnDepth = 1;
        exactRemaining = configuredExactDepth - 1;
        searchExactExtensions++;
      }
    }
  }
  if (turnDepth <= 0) {
    if (minimumGroupLiberties(other(currentPlayer)) <= 2) {
      const winningActions = winWithinCurrentTurn(ply, currentPlayer, 4);
      if (winningActions > 0) {
        searchReturnedSolved = true;
        return terminalScore(currentPlayer, ply + winningActions);
      }
    }
    const volatile = lastOpponentLoss > 0 ||
      minimumGroupLiberties(currentPlayer) <= 2 ||
      minimumGroupLiberties(other(currentPlayer)) <= 2;
    if (searchTacticalDepth <= 0 || !volatile) {
      searchEvaluations++;
      return evaluatePosition(searchRootPlayer);
    }
    return quiescence(alphaInput, betaInput, searchTacticalDepth, ply, false);
  }
  if (ply >= MAX_PLY - 1) {
    searchEvaluations++;
    return evaluatePosition(searchRootPlayer);
  }

  const exactMode = exactRemaining >= 0;
  const key = hashSearchPosition();
  const slot = i32(key & TT_MASK);
  const originalAlpha = alphaInput;
  const originalBeta = betaInput;
  let alpha = alphaInput;
  let beta = betaInput;
  let preferred: u64 = 0;
  if (!exactMode && unchecked(ttHashes[slot]) == key) {
    preferred = unchecked(ttBestActions[slot]);
    if (i32(unchecked(ttDepths[slot])) >= turnDepth) {
      searchTtHits++;
      const score = scoreFromTransposition(unchecked(ttScores[slot]), ply);
      const flag = unchecked(ttFlags[slot]);
      if (flag == TT_EXACT) return score;
      if (flag == TT_LOWER && score > alpha) alpha = score;
      else if (flag == TT_UPPER && score < beta) beta = score;
      if (alpha >= beta) return score;
    }
  }

  const count = scoreActions(ply, preferred);
  if (count == 0) {
    searchReturnedSolved = true;
    return 0;
  }
  const base = ply * MAX_ACTIONS;
  const maximizing = currentPlayer == searchRootPlayer;
  let best = maximizing ? -SEARCH_INFINITY : SEARCH_INFINITY;
  let bestSignature: u64 = 0;
  let children = 0;
  let allChildrenSolved = true;
  let historyDependent = false;
  const quietWidth = exactMode
    ? count
    : currentPlayer == searchRootPlayer && ply > 0
      ? 12
      : ply <= 1 ? 20 : turnDepth >= 4 ? 10 : turnDepth >= 3 ? 14 : turnDepth >= 2 ? 16 : 20;

  for (let order = 0; order < count && !searchStopped; order++) {
    const index = i32(unchecked(orderedIndexes[base + order]));
    const tactical = unchecked(actionTactical[base + index]) != 0;
    if (!tactical && order >= quietWidth) {
      allChildrenSolved = false;
      continue;
    }
    const actor = currentPlayer;
    if (!applyAt(ply, index)) continue;
    const childDepth = currentPlayer == actor && gameOutcome == 0 ? turnDepth : turnDepth - 1;
    const reduced = !exactMode && !tactical && order >= 16 && turnDepth >= 2 &&
      currentPlayer != actor && childDepth > 0 && gameOutcome == 0;
    const firstDepth = reduced ? childDepth - 1 : childDepth;
    let score: i32;
    let childSolved = true;
    let childHistoryDependent = false;
    if (gameOutcome != 0) {
      score = terminalScore(gameOutcome, ply + 1);
    } else if (children == 0 || beta - alpha <= 1) {
      score = alphaBeta(firstDepth, alpha, beta, ply + 1, exactRemaining);
      childSolved = searchReturnedSolved;
      childHistoryDependent = searchReturnedHistoryDependent;
      if (reduced && !searchStopped &&
          ((maximizing && score > alpha) || (!maximizing && score < beta))) {
        searchReSearches++;
        score = alphaBeta(childDepth, alpha, beta, ply + 1, exactRemaining);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
      } else if (reduced) childSolved = false;
    } else if (maximizing) {
      score = alphaBeta(firstDepth, alpha, alpha + 1, ply + 1, exactRemaining);
      childSolved = searchReturnedSolved;
      childHistoryDependent = searchReturnedHistoryDependent;
      if (reduced && !searchStopped && score > alpha) {
        searchReSearches++;
        score = alphaBeta(childDepth, alpha, alpha + 1, ply + 1, exactRemaining);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
      } else if (reduced) childSolved = false;
      if (!searchStopped && score > alpha && score < beta) {
        searchReSearches++;
        score = alphaBeta(childDepth, alpha, beta, ply + 1, exactRemaining);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
      }
    } else {
      score = alphaBeta(firstDepth, beta - 1, beta, ply + 1, exactRemaining);
      childSolved = searchReturnedSolved;
      childHistoryDependent = searchReturnedHistoryDependent;
      if (reduced && !searchStopped && score < beta) {
        searchReSearches++;
        score = alphaBeta(childDepth, beta - 1, beta, ply + 1, exactRemaining);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
      } else if (reduced) childSolved = false;
      if (!searchStopped && score < beta && score > alpha) {
        searchReSearches++;
        score = alphaBeta(childDepth, alpha, beta, ply + 1, exactRemaining);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
      }
    }
    undo_position();
    children++;
    allChildrenSolved = allChildrenSolved && childSolved;
    historyDependent = historyDependent || childHistoryDependent;
    if (searchStopped) break;

    if ((maximizing && score > best) || (!maximizing && score < best) ||
        (score == best && (bestSignature == 0 || actionSignature(ply, index) < bestSignature))) {
      best = score;
      bestSignature = actionSignature(ply, index);
    }
    if (maximizing) {
      if (best > alpha) alpha = best;
    } else if (best < beta) beta = best;
    if (alpha >= beta) {
      searchCutoffs++;
      break;
    }
  }

  if (children == 0 || searchStopped) {
    searchReturnedHistoryDependent = historyDependent;
    if (children == 0) searchEvaluations++;
    return children == 0 ? evaluatePosition(searchRootPlayer) : best;
  }
  searchReturnedSolved = allChildrenSolved && !historyDependent;
  searchReturnedHistoryDependent = historyDependent;
  const flag = best <= originalAlpha ? TT_UPPER : best >= originalBeta ? TT_LOWER : TT_EXACT;
  const previousGeneration = unchecked(ttGenerations[slot]);
  const previousDepth = i32(unchecked(ttDepths[slot]));
  if (exactMode && !historyDependent) {
    // Retain only the proven line's move for same-turn continuation recovery;
    // depth -1 prevents this path-sensitive exact-search entry from serving a
    // value in a later normal probe.
    unchecked(ttHashes[slot] = key);
    unchecked(ttDepths[slot] = -1);
    unchecked(ttBestActions[slot] = bestSignature);
    unchecked(ttGenerations[slot] = ttGeneration);
  } else if (!historyDependent &&
      (unchecked(ttHashes[slot]) == key || previousGeneration != ttGeneration || turnDepth >= previousDepth)) {
    unchecked(ttHashes[slot] = key);
    unchecked(ttScores[slot] = scoreToTransposition(best, ply));
    unchecked(ttDepths[slot] = i8(turnDepth));
    unchecked(ttFlags[slot] = flag);
    unchecked(ttBestActions[slot] = bestSignature);
    unchecked(ttGenerations[slot] = ttGeneration);
  }
  return best;
}

function searchRootAtDepth(turnDepth: i32): i32 {
  searchRootIterationSolved = false;
  if (!checkSearchNode()) return searchBestScore;
  unchecked(pathHashes[0] = hashPosition());
  const count = searchRootActionCount;
  if (count == 0) return 0;
  const maximizing = true; // rootPlayer is captured from currentPlayer at entry.
  let alpha = -SEARCH_INFINITY;
  const beta = SEARCH_INFINITY;
  let best = -SEARCH_INFINITY;
  let bestIndex = -1;
  let bestObjective = -SEARCH_INFINITY;
  let strongestObjective = -SEARCH_INFINITY;
  let bestPersonalityBonus = 0;
  let children = 0;
  let allChildrenSolved = true;
  let historyDependent = false;
  let stoneSeen = 0;
  let swanSeen = 0;
  let moveSeen = 0;
  let pushSeen = 0;
  for (let order = 0; order < count && !searchStopped; order++) {
    const index = i32(unchecked(rootOrderedIndexes[order]));
    const type = unchecked(actionTypes[actionOffset(0, index)]);
    let classRank = 0;
    if (type == ACTION_STONE) classRank = stoneSeen++;
    else if (type == ACTION_SWAN) classRank = swanSeen++;
    else if (type == ACTION_MOVE) classRank = moveSeen++;
    else classRank = pushSeen++;
    // Depth one remains exhaustive. Deeper iterations retain the best full
    // static candidates, a quota from every action class, and every action the
    // root scan proved tactical/defensive.
    if (turnDepth >= 2 && order >= 32 && classRank >= 4 && unchecked(rootTactical[index]) == 0) {
      allChildrenSolved = false;
      continue;
    }
    const actor = currentPlayer;
    if (!applyAt(0, index)) continue;
    const personalityBonus = unchecked(rootPersonalityBonuses[index]);
    const repeatsPriorResult = priorRootResult(hashPosition());
    const childDepth = currentPlayer == actor && gameOutcome == 0 ? turnDepth : turnDepth - 1;
    let score: i32;
    let childSolved = true;
    let childHistoryDependent = false;
    if (gameOutcome != 0) score = terminalScore(gameOutcome, 1);
    else if (children == 0) {
      score = alphaBeta(childDepth, alpha - personalityBonus, beta - personalityBonus, 1, -1);
      childSolved = searchReturnedSolved;
      childHistoryDependent = searchReturnedHistoryDependent;
      score += personalityBonus;
    }
    else {
      score = alphaBeta(
        childDepth,
        alpha - personalityBonus,
        alpha + 1 - personalityBonus,
        1,
        -1
      );
      childSolved = searchReturnedSolved;
      childHistoryDependent = searchReturnedHistoryDependent;
      score += personalityBonus;
      if (!searchStopped && score > alpha && score < beta) {
        searchReSearches++;
        score = alphaBeta(childDepth, alpha - personalityBonus, beta - personalityBonus, 1, -1);
        childSolved = searchReturnedSolved;
        childHistoryDependent = searchReturnedHistoryDependent;
        score += personalityBonus;
      }
    }
    undo_position();
    children++;
    allChildrenSolved = allChildrenSolved && childSolved;
    historyDependent = historyDependent || childHistoryDependent;
    if (searchStopped) break;
    if (repeatedRootAction != 0 && actionSignature(0, index) == repeatedRootAction &&
        score > -MATE_SCORE / 2 && score < MATE_SCORE / 2) {
      score -= 12_000;
      allChildrenSolved = false;
      historyDependent = true;
    }
    if (repeatsPriorResult && score > -MATE_SCORE / 2 && score < MATE_SCORE / 2) {
      score -= 14_000;
      allChildrenSolved = false;
      historyDependent = true;
    }
    if (rootPositionRepeated && (type == ACTION_MOVE || type == ACTION_PUSH) &&
        score > -MATE_SCORE / 2 && score < MATE_SCORE / 2) {
      // A placement irreversibly breaks an exact board cycle. Prefer that
      // escape when the live game has already visited this root, unless a
      // translation carries a mating-scale tactical value.
      score -= 100_000;
      allChildrenSolved = false;
      historyDependent = true;
    }
    const objectiveScore = score - personalityBonus;
    unchecked(rootDepthScores[index] = score);
    unchecked(rootDepthObjectiveScores[index] = objectiveScore);
    if (objectiveScore > strongestObjective) strongestObjective = objectiveScore;
    if (score > best || (score == best && (bestIndex < 0 || index < bestIndex))) {
      best = score;
      bestIndex = index;
      bestObjective = objectiveScore;
      bestPersonalityBonus = personalityBonus;
    }
    if (best > alpha) alpha = best;
  }
  if (!searchStopped && bestIndex >= 0) {
    searchBestRootIndex = bestIndex;
    searchBestScore = best;
    searchBestObjectiveScore = bestObjective;
    searchStrongestObjectiveScore = strongestObjective;
    searchBestPersonalityBonus = bestPersonalityBonus;
    if (turnDepth == 1) {
      // The exhaustive completed-turn scores are a much stronger widening
      // order than action geometry, particularly once a turn has two actions.
      for (let order = 0; order < count; order++) {
        const index = i32(unchecked(rootOrderedIndexes[order]));
        unchecked(orderedIndexes[order] = u16(index));
        // Personality selects among the canonically strongest candidates; it
        // never controls which candidates survive deeper root widening.
        unchecked(orderedScores[order] = rootDepthObjectiveScores[index]);
      }
      if (count > 1) sortOrdered(0, 0, count - 1);
      for (let order = 0; order < count; order++) unchecked(rootOrderedIndexes[order] = orderedIndexes[order]);
    } else {
      let bestOrder = 0;
      while (bestOrder < count && i32(unchecked(rootOrderedIndexes[bestOrder])) != bestIndex) bestOrder++;
      while (bestOrder > 0) {
        unchecked(rootOrderedIndexes[bestOrder] = rootOrderedIndexes[bestOrder - 1]);
        bestOrder--;
      }
      unchecked(rootOrderedIndexes[0] = u16(bestIndex));
    }
    searchRootIterationSolved = best >= MATE_SCORE - 1 || (allChildrenSolved && !historyDependent);
  }
  return best;
}

function captureRootPlan(index: i32, followContinuation: bool): void {
  searchContinuationCount = 0;
  searchRootPostHash = 0;
  if (index < 0) return;
  const startingUndoDepth = undoDepth;
  if (!applyAt(0, index)) return;
  searchRootPostHash = hashPosition();
  let ply = 1;
  while (followContinuation && gameOutcome == 0 && currentPlayer == searchRootPlayer &&
      ply < MAX_PLY && searchContinuationCount < MAX_PLY) {
    const key = hashSearchPosition();
    const slot = i32(key & TT_MASK);
    if (unchecked(ttHashes[slot]) != key) break;
    const preferred = unchecked(ttBestActions[slot]);
    if (preferred == 0) break;
    const count = generateAt(ply);
    let selected = -1;
    for (let indexAtPly = 0; indexAtPly < count; indexAtPly++) {
      if (actionSignature(ply, indexAtPly) == preferred) {
        selected = indexAtPly;
        break;
      }
    }
    if (selected < 0) break;
    const offset = actionOffset(ply, selected);
    const continuation = searchContinuationCount;
    unchecked(continuationTypes[continuation] = actionTypes[offset]);
    unchecked(continuationDirs[continuation] = actionDirs[offset]);
    unchecked(continuationCells[continuation] = actionCells[offset]);
    unchecked(continuationMaskLo[continuation] = actionMaskLo[offset]);
    unchecked(continuationMaskHi[continuation] = actionMaskHi[offset]);
    unchecked(continuationInputHashes[continuation] = hashPosition());
    if (!applyAt(ply, selected)) break;
    unchecked(continuationPostHashes[continuation] = hashPosition());
    searchContinuationCount++;
    ply++;
  }
  while (undoDepth > startingUndoDepth) undo_position();
}

function commitRootIndex(index: i32): i32 {
  if (index < 0) return 0;
  const rootHash = hashPosition();
  const signature = actionSignature(0, index);
  if (!applyAt(0, index)) return 0;
  const resultHash = hashPosition();
  undo_position();
  unchecked(rootHistoryHashes[rootHistoryCursor] = rootHash);
  unchecked(rootHistoryActions[rootHistoryCursor] = signature);
  unchecked(rootHistoryResults[rootHistoryCursor] = resultHash);
  rootHistoryCursor = (rootHistoryCursor + 1) % ROOT_HISTORY_SIZE;
  rootHistoryCount = min<i32>(ROOT_HISTORY_SIZE, rootHistoryCount + 1);
  return 1;
}

// ------------------------------ exported ABI ------------------------------

export function reset_state(player: i32, movesLeft: i32): void {
  for (let cell = 0; cell < BOARD_CELLS; cell++) unchecked(board[cell] = EMPTY);
  currentPlayer = player;
  actionsLeft = movesLeft;
  gameOutcome = 0;
  undoDepth = 0;
}

export function set_cell(cell: i32, tile: i32): void {
  if (cell < 0 || cell >= BOARD_CELLS || tile < 0 || tile > 5) return;
  unchecked(board[cell] = u8(tile));
}

export function get_cell(cell: i32): i32 {
  return cell >= 0 && cell < BOARD_CELLS ? i32(tileAt(cell)) : -1;
}

export function get_current_player(): i32 { return currentPlayer; }
export function get_actions_left(): i32 { return actionsLeft; }
export function get_outcome(): i32 { return gameOutcome; }

export function generate_actions(): i32 { return generateAt(0); }
export function get_action_type(index: i32): i32 { return i32(unchecked(actionTypes[index])); }
export function get_action_cell(index: i32): i32 { return i32(unchecked(actionCells[index])); }
export function get_action_direction(index: i32): i32 { return i32(unchecked(actionDirs[index])); }
export function get_action_mask_low(index: i32): u64 { return unchecked(actionMaskLo[index]); }
export function get_action_mask_high(index: i32): u64 { return unchecked(actionMaskHi[index]); }

/** Record only an action the caller actually selected and applied. */
export function commit_root_action(
  type: i32,
  cell: i32,
  direction: i32,
  maskLow: u64,
  maskHigh: u64
): i32 {
  const count = generateAt(0);
  for (let index = 0; index < count; index++) {
    const offset = actionOffset(0, index);
    if (i32(unchecked(actionTypes[offset])) != type ||
        i32(unchecked(actionCells[offset])) != cell ||
        i32(unchecked(actionDirs[offset])) != direction ||
        unchecked(actionMaskLo[offset]) != maskLow ||
        unchecked(actionMaskHi[offset]) != maskHigh) continue;
    return commitRootIndex(index);
  }
  return 0;
}

export function apply_generated_action(index: i32): i32 {
  return applyAt(0, index) ? 1 : 0;
}

export function undo_position(): i32 {
  if (undoDepth <= 0) return 0;
  undoDepth--;
  const offset = undoDepth * BOARD_CELLS;
  for (let cell = 0; cell < BOARD_CELLS; cell++) unchecked(board[cell] = undoBoards[offset + cell]);
  currentPlayer = unchecked(undoCurrent[undoDepth]);
  actionsLeft = unchecked(undoMoves[undoDepth]);
  gameOutcome = unchecked(undoOutcome[undoDepth]);
  lastOpponentLoss = unchecked(undoOpponentLoss[undoDepth]);
  return 1;
}

export function position_hash(): u64 {
  return hashPosition();
}

export function current_turn_win_distance(maximumActions: i32): i32 {
  return winWithinCurrentTurn(0, currentPlayer, max<i32>(1, min<i32>(8, maximumActions)));
}

export function perft(depth: i32): u64 {
  if (depth < 0 || depth >= MAX_PLY) return 0;
  return perftAt(depth, 0);
}

/**
 * Iterative-deepening completed-turn PVS. `budgetMs <= 0` disables the clock;
 * the deterministic node budget remains authoritative in tests and arenas.
 * Returns an index into the root action list exposed by get_action_*.
 */
export function search_best(
  maxTurnDepth: i32,
  nodeBudget: i32,
  budgetMs: f64,
  style: i32,
  tacticalDepth: i32,
  exactDepth: i32
): i32 {
  searchRootPlayer = currentPlayer;
  searchStyle = max<i32>(0, min<i32>(6, style));
  configuredTacticalDepth = max<i32>(0, min<i32>(3, tacticalDepth));
  configuredExactDepth = max<i32>(0, min<i32>(16, exactDepth));
  searchTacticalDepth = configuredTacticalDepth;
  searchDeadline = budgetMs > 0.0 ? monotonicNow() + budgetMs : 0.0;
  searchNodeLimit = max<i32>(0, nodeBudget);
  searchNodes = 0;
  searchGenerated = 0;
  searchEvaluations = 0;
  searchTtHits = 0;
  searchCutoffs = 0;
  searchReSearches = 0;
  searchRootHistoryHits = 0;
  searchClassificationActions = 0;
  searchQuiescenceForcedActions = 0;
  sameTurnHistoryActions = 0;
  searchRootThreatDetected = 0;
  searchExactExtensions = 0;
  searchExactSolved = 0;
  searchContinuationCount = 0;
  searchRootPostHash = 0;
  searchCompletedDepth = 0;
  searchAttemptedDepth = 0;
  searchBestRootIndex = -1;
  searchBestScore = evaluatePosition(searchRootPlayer);
  searchBestObjectiveScore = searchBestScore;
  searchStrongestObjectiveScore = searchBestScore;
  searchBestPersonalityBonus = 0;
  searchStopReason = 0;
  searchStopped = false;
  undoDepth = 0;
  ttGeneration++;
  if (ttGeneration == 0) ttGeneration = 1;

  activeRootHistoryHash = hashPosition();
  rootPositionRepeated = priorObservedPositionCount(activeRootHistoryHash) > 0;
  repeatedRootAction = 0;
  for (let ago = 1; ago <= rootHistoryCount; ago++) {
    const slot = (rootHistoryCursor - ago + ROOT_HISTORY_SIZE) % ROOT_HISTORY_SIZE;
    if (unchecked(rootHistoryHashes[slot]) == activeRootHistoryHash) {
      repeatedRootAction = unchecked(rootHistoryActions[slot]);
      searchRootHistoryHits++;
      break;
    }
  }

  const rootCount = scoreActions(0, 0);
  searchRootActionCount = rootCount;
  prepareRootPersonalityBonuses(rootCount);
  for (let order = 0; order < rootCount; order++) {
    unchecked(rootOrderedIndexes[order] = orderedIndexes[order]);
    const index = i32(unchecked(orderedIndexes[order]));
    unchecked(rootTactical[index] = actionTactical[index]);
  }
  if (rootCount <= 0) return -1;
  searchBestRootIndex = i32(unchecked(rootOrderedIndexes[0]));
  if (applyAt(0, searchBestRootIndex)) {
    const immediateRootWin = gameOutcome == searchRootPlayer;
    if (immediateRootWin) searchBestScore = terminalScore(gameOutcome, 1);
    undo_position();
    if (immediateRootWin) {
      searchBestObjectiveScore = searchBestScore;
      searchStrongestObjectiveScore = searchBestScore;
      searchCompletedDepth = 1;
      searchAttemptedDepth = 1;
      searchExactSolved = 1;
      captureRootPlan(searchBestRootIndex, false);
      return searchBestRootIndex;
    }
  }
  if (rootCount == 1) {
    if (applyAt(0, searchBestRootIndex)) {
      searchBestScore = gameOutcome != 0 ? terminalScore(gameOutcome, 1) : evaluatePosition(searchRootPlayer);
      searchBestObjectiveScore = searchBestScore;
      searchStrongestObjectiveScore = searchBestScore;
      undo_position();
    }
    captureRootPlan(searchBestRootIndex, false);
    return searchBestRootIndex;
  }
  maxTurnDepth = max<i32>(1, min<i32>(12, maxTurnDepth));

  // A zero deterministic budget intentionally returns the tactical/geometric
  // fallback without entering a partial iteration.
  if (searchNodeLimit <= 0) {
    searchStopReason = 2;
    captureRootPlan(searchBestRootIndex, false);
    return searchBestRootIndex;
  }
  for (let depth = 1; depth <= maxTurnDepth; depth++) {
    searchAttemptedDepth = depth;
    // The first completed-turn iteration is the guaranteed safety floor. Root
    // tactical scanning already catches immediate wins and only-defenses; a
    // horizon extension is enabled from depth two onward.
    searchTacticalDepth = depth == 1 ? 0 : configuredTacticalDepth;
    searchRootAtDepth(depth);
    if (searchStopped) break;
    searchCompletedDepth = depth;
    searchExactSolved = searchRootIterationSolved ? 1 : 0;
    captureRootPlan(searchBestRootIndex, true);
    if (searchExactSolved != 0 || abs(searchBestScore) > MATE_SCORE / 2) break;
  }
  if (searchRootPostHash == 0) captureRootPlan(searchBestRootIndex, false);
  return searchBestRootIndex;
}

export function clear_transposition_table(): void {
  for (let slot = 0; slot < TT_SIZE; slot++) {
    unchecked(ttHashes[slot] = 0);
    unchecked(ttGenerations[slot] = 0);
  }
  ttGeneration = 1;
  rootHistoryCursor = 0;
  rootHistoryCount = 0;
  repeatedRootAction = 0;
}

export function evaluate_loaded_position(perspective: i32, style: i32): i32 {
  return evaluateStyledPosition(perspective, max<i32>(0, min<i32>(6, style)));
}

/** Test/diagnostic entry point for the completed-turn tactical horizon. */
export function tactical_probe(perspective: i32, style: i32, tacticalDepth: i32): i32 {
  searchRootPlayer = perspective;
  searchStyle = max<i32>(0, min<i32>(6, style));
  searchTacticalDepth = max<i32>(0, min<i32>(3, tacticalDepth));
  searchDeadline = 0.0;
  searchNodeLimit = 2_000_000;
  searchNodes = 0;
  searchGenerated = 0;
  searchEvaluations = 0;
  searchCutoffs = 0;
  searchClassificationActions = 0;
  searchQuiescenceForcedActions = 0;
  searchStopped = false;
  undoDepth = 0;
  activeRootHistoryHash = hashPosition();
  rootPositionRepeated = false;
  return quiescence(-SEARCH_INFINITY, SEARCH_INFINITY, searchTacticalDepth, 0, false);
}

export function get_search_score(): i32 { return searchBestScore; }
export function get_search_objective_score(): i32 { return searchBestObjectiveScore; }
export function get_search_strongest_objective_score(): i32 { return searchStrongestObjectiveScore; }
export function get_search_personality_bonus(): i32 { return searchBestPersonalityBonus; }
export function get_search_nodes(): i32 { return searchNodes; }
export function get_search_generated_actions(): i32 { return searchGenerated; }
export function get_search_evaluations(): i32 { return searchEvaluations; }
export function get_search_tt_hits(): i32 { return searchTtHits; }
export function get_search_cutoffs(): i32 { return searchCutoffs; }
export function get_search_researches(): i32 { return searchReSearches; }
export function get_search_root_history_hits(): i32 { return searchRootHistoryHits; }
export function get_search_classification_actions(): i32 { return searchClassificationActions; }
export function get_search_quiescence_forced_actions(): i32 { return searchQuiescenceForcedActions; }
export function get_search_root_threat_detected(): i32 { return searchRootThreatDetected; }
export function get_search_exact_extensions(): i32 { return searchExactExtensions; }
export function get_search_exact_solved(): i32 { return searchExactSolved; }
export function get_search_root_post_hash(): u64 { return searchRootPostHash; }
export function get_search_continuation_count(): i32 { return searchContinuationCount; }
export function get_search_continuation_type(index: i32): i32 {
  return index >= 0 && index < searchContinuationCount ? i32(unchecked(continuationTypes[index])) : -1;
}
export function get_search_continuation_cell(index: i32): i32 {
  return index >= 0 && index < searchContinuationCount ? i32(unchecked(continuationCells[index])) : -1;
}
export function get_search_continuation_direction(index: i32): i32 {
  return index >= 0 && index < searchContinuationCount ? i32(unchecked(continuationDirs[index])) : -1;
}
export function get_search_continuation_mask_low(index: i32): u64 {
  return index >= 0 && index < searchContinuationCount ? unchecked(continuationMaskLo[index]) : 0;
}
export function get_search_continuation_mask_high(index: i32): u64 {
  return index >= 0 && index < searchContinuationCount ? unchecked(continuationMaskHi[index]) : 0;
}
export function get_search_continuation_input_hash(index: i32): u64 {
  return index >= 0 && index < searchContinuationCount ? unchecked(continuationInputHashes[index]) : 0;
}
export function get_search_continuation_post_hash(index: i32): u64 {
  return index >= 0 && index < searchContinuationCount ? unchecked(continuationPostHashes[index]) : 0;
}
export function get_search_completed_depth(): i32 { return searchCompletedDepth; }
export function get_search_attempted_depth(): i32 { return searchAttemptedDepth; }
export function get_search_stop_reason(): i32 { return searchStopReason; }
