import test from "node:test";
import assert from "node:assert/strict";
import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  computeFreezesOn,
  type Board,
  type Player,
  type Tile
} from "../src/renderer/game/encirclement";
import {
  actionKey,
  applyAction,
  generateLegalActions,
  type ActionCoordinate,
  type Direction,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";

// This is an intentionally independent, typed port of the live executor's
// placement/move/push validation.  Differential testing it against the compact
// rules engine prevents search-only rule changes from silently drifting away
// from what game.ts will actually execute.

const SUN = 1 as const;
const MOON = 2 as const;
const SIZE = 10;
const DIRS: Direction[] = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [-1, 1], [1, -1], [1, 1]
];
const DIRS4: Direction[] = [[-1, 0], [1, 0], [0, -1], [0, 1]];

interface LiveApplied extends SearchState {
  outcome: "sun" | "moon" | "draw" | null;
  opponentLoss: number;
}

test("rules engine action generation and execution match the live executor corpus", () => {
  let state = initialState();
  const random = seededRandom(0x1a11ce);
  let comparedActions = 0;

  for (let step = 0; step < 36; step += 1) {
    const compact = generateLegalActions(state);
    const live = liveGenerateActions(state);
    assert.deepEqual(
      compact.map(actionKey).sort(),
      live.map(actionKey).sort(),
      `legal action set drifted at corpus step ${step}`
    );

    const probes = uniqueNumbers([
      0,
      Math.floor(live.length / 3),
      Math.floor(live.length / 2),
      live.length - 1,
      Math.floor(random() * live.length)
    ]).filter((index) => index >= 0 && index < live.length);
    for (const index of probes) {
      const action = live[index];
      const expected = liveApplyAction(state, action);
      const actual = applyAction(state, action);
      assert.ok(expected, `live action unexpectedly rejected: ${actionKey(action)}`);
      assert.ok(actual, `compact action unexpectedly rejected: ${actionKey(action)}`);
      assert.deepEqual(actual.board, expected.board, `board drift for ${actionKey(action)} at step ${step}`);
      assert.equal(actual.current, expected.current, `side-to-move drift for ${actionKey(action)}`);
      assert.equal(actual.movesLeft, expected.movesLeft, `momentum drift for ${actionKey(action)}`);
      assert.equal(actual.opponentLoss, expected.opponentLoss, `freeze bonus drift for ${actionKey(action)}`);
      assert.equal(actual.outcome, expected.outcome, `terminal outcome drift for ${actionKey(action)}`);
      comparedActions += 1;
    }

    if (live.length === 0) break;
    const selected = live[Math.floor(random() * live.length)];
    const next = liveApplyAction(state, selected);
    assert.ok(next);
    state = next.outcome
      ? initialState(step % 2 === 0)
      : { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }

  assert.ok(comparedActions >= 100, `expected a meaningful parity corpus, compared ${comparedActions}`);
});

function liveGenerateActions(state: SearchState): LinithAction[] {
  const actions: LinithAction[] = [];
  for (let r = 0; r < SIZE; r += 1) for (let c = 0; c < SIZE; c += 1) {
    if (state.board[r][c] !== EMPTY) continue;
    actions.push({ type: "stone", r, c });
    if (liveLegalSwanPlacement(state.board, state.current, r, c)) actions.push({ type: "swan", r, c });
  }

  const own = activeCoordinates(state.board, state.current);
  for (const subset of subsets(own)) for (const dir of DIRS) {
    if (liveSimulateMove(state.board, state.current, subset, dir, false)) {
      actions.push({ type: "move", swans: subset, dir });
    }
  }

  const enemy = activeCoordinates(state.board, other(state.current));
  for (const subset of subsets(enemy)) for (const dir of DIRS) {
    if (liveSimulateMove(state.board, state.current, subset, dir, true)) {
      actions.push({ type: "push", swans: subset, dir });
    }
  }
  return actions;
}

function liveApplyAction(state: SearchState, action: LinithAction): LiveApplied | null {
  let moved: Board | null = null;
  if (action.type === "stone") {
    if (!inBounds(action.r, action.c) || state.board[action.r][action.c] !== EMPTY) return null;
    moved = clone(state.board);
    moved[action.r][action.c] = STONE;
  } else if (action.type === "swan") {
    if (!liveLegalSwanPlacement(state.board, state.current, action.r, action.c)) return null;
    moved = clone(state.board);
    moved[action.r][action.c] = state.current === SUN ? SWAN_SUN : SWAN_MOON;
  } else {
    moved = liveSimulateMove(
      state.board,
      state.current,
      action.swans,
      action.dir,
      action.type === "push"
    );
  }
  if (!moved) return null;

  const freeze = computeFreezesOn(moved);
  const outcome = liveOutcome(freeze.nb, freeze);
  const opponentLoss = state.current === SUN
    ? freeze.frozeMoon + freeze.sealedMoon
    : freeze.frozeSun + freeze.sealedSun;
  const remaining = state.movesLeft + opponentLoss - 1;
  if (remaining > 0) {
    return { board: freeze.nb, current: state.current, movesLeft: remaining, outcome, opponentLoss };
  }
  const current = other(state.current);
  const bothAtSix = countSwans(freeze.nb, SUN) >= 6 && countSwans(freeze.nb, MOON) >= 6;
  return { board: freeze.nb, current, movesLeft: bothAtSix ? 2 : 1, outcome, opponentLoss };
}

function liveOutcome(
  board: Board,
  freeze: ReturnType<typeof computeFreezesOn>
): LiveApplied["outcome"] {
  if (freeze.sealedSun > 0 && freeze.sealedMoon > 0) return "draw";
  if (freeze.sealedSun > 0) return "moon";
  if (freeze.sealedMoon > 0) return "sun";
  const activeSun = activeCoordinates(board, SUN).length;
  const activeMoon = activeCoordinates(board, MOON).length;
  if (activeSun === 0 && activeMoon === 0) return "draw";
  if (activeMoon === 0) return "sun";
  if (activeSun === 0) return "moon";
  return board.every((row) => row.every((tile) => tile !== EMPTY)) ? "draw" : null;
}

function liveLegalSwanPlacement(board: Board, player: Player, r: number, c: number): boolean {
  if (!inBounds(r, c) || board[r][c] !== EMPTY || countSwans(board, player) >= 6) return false;
  if (!DIRS4.some(([dr, dc]) => inBounds(r + dr, c + dc) && same(board[r + dr][c + dc], player))) {
    return false;
  }
  return !DIRS.some(([dr, dc]) => inBounds(r + dr, c + dc) && enemy(board[r + dr][c + dc], player));
}

function liveSimulateMove(
  board: Board,
  actor: Player,
  swans: readonly ActionCoordinate[],
  dir: Direction,
  pushing: boolean
): Board | null {
  if (swans.length === 0 || new Set(swans.map(key)).size !== swans.length) return null;
  const movingOwner = pushing ? other(actor) : actor;
  if (swans.some(({ r, c }) => !inBounds(r, c) || !active(board[r][c], movingOwner))) return null;
  if (pushing && swans.some((swan) => !neighbours(swan.r, swan.c).some(({ r, c }) => active(board[r][c], actor)))) {
    return null;
  }

  const moving = new Set(swans.map(key));
  const stoneOrigins = new Map<string, ActionCoordinate>();
  for (const swan of swans) for (const stone of neighbours(swan.r, swan.c)) {
    if (board[stone.r][stone.c] !== STONE || stoneOrigins.has(key(stone))) continue;
    const shared = neighbours(stone.r, stone.c).some((adjacent) => {
      const tile = board[adjacent.r][adjacent.c];
      if (enemy(tile, movingOwner)) return true;
      return same(tile, movingOwner) && (!active(tile, movingOwner) || !moving.has(key(adjacent)));
    });
    if (!shared) stoneOrigins.set(key(stone), stone);
  }

  const stonesFrom = new Set(stoneOrigins.keys());
  const stonesTo = new Map<string, ActionCoordinate>();
  for (const [originKey, stone] of stoneOrigins) {
    const destination = { r: stone.r + dir[0], c: stone.c + dir[1] };
    if (!liveVacantAfterMove(board, destination, moving, stonesFrom)) return null;
    if ([...stonesTo.values()].some((otherDestination) => key(otherDestination) === key(destination))) return null;
    stonesTo.set(originKey, destination);
  }

  const swanDestinations = new Set<string>();
  for (const swan of swans) {
    const destination = { r: swan.r + dir[0], c: swan.c + dir[1] };
    if (!inBounds(destination.r, destination.c)) return null;
    if (!pushing && inNakedEnemyZone(board, destination.r, destination.c, actor)) return null;
    const destinationKey = key(destination);
    if (swanDestinations.has(destinationKey)) return null;
    const tile = board[destination.r][destination.c];
    if (tile !== EMPTY && !moving.has(destinationKey) && !stonesFrom.has(destinationKey)) return null;
    swanDestinations.add(destinationKey);
  }
  if ([...stonesTo.values()].some((destination) => swanDestinations.has(key(destination)))) return null;

  const next = clone(board);
  for (const swan of swans) next[swan.r][swan.c] = EMPTY;
  for (const stone of stoneOrigins.values()) next[stone.r][stone.c] = EMPTY;
  for (const stone of stonesTo.values()) next[stone.r][stone.c] = STONE;
  for (const swan of swans) next[swan.r + dir[0]][swan.c + dir[1]] = movingOwner === SUN ? SWAN_SUN : SWAN_MOON;
  return next;
}

function liveVacantAfterMove(
  board: Board,
  coordinate: ActionCoordinate,
  moving: ReadonlySet<string>,
  stonesFrom: ReadonlySet<string>
): boolean {
  if (!inBounds(coordinate.r, coordinate.c)) return false;
  return board[coordinate.r][coordinate.c] === EMPTY || moving.has(key(coordinate)) || stonesFrom.has(key(coordinate));
}

function inNakedEnemyZone(board: Board, r: number, c: number, actor: Player): boolean {
  return neighbours(r, c).some((coordinate) => {
    if (!enemy(board[coordinate.r][coordinate.c], actor)) return false;
    return neighbours(coordinate.r, coordinate.c).every((adjacent) => board[adjacent.r][adjacent.c] !== STONE);
  });
}

function* subsets<T>(items: readonly T[]): Generator<T[]> {
  for (let mask = 1; mask < (1 << items.length); mask += 1) {
    const subset: T[] = [];
    for (let index = 0; index < items.length; index += 1) if (mask & (1 << index)) subset.push(items[index]);
    yield subset;
  }
}

function countSwans(board: Board, player: Player): number {
  let count = 0;
  for (const row of board) for (const tile of row) if (same(tile, player)) count += 1;
  return count;
}

function activeCoordinates(board: Board, player: Player): ActionCoordinate[] {
  const coordinates: ActionCoordinate[] = [];
  for (let r = 0; r < SIZE; r += 1) for (let c = 0; c < SIZE; c += 1) {
    if (active(board[r][c], player)) coordinates.push({ r, c });
  }
  return coordinates;
}

function neighbours(r: number, c: number): ActionCoordinate[] {
  return DIRS.map(([dr, dc]) => ({ r: r + dr, c: c + dc })).filter(({ r: nr, c: nc }) => inBounds(nr, nc));
}

function same(tile: Tile, player: Player): boolean {
  return player === SUN ? tile === SWAN_SUN || tile === FROZEN_SUN : tile === SWAN_MOON || tile === FROZEN_MOON;
}

function enemy(tile: Tile, player: Player): boolean {
  return same(tile, other(player));
}

function active(tile: Tile, player: Player): boolean {
  return tile === (player === SUN ? SWAN_SUN : SWAN_MOON);
}

function other(player: Player): Player {
  return player === SUN ? MOON : SUN;
}

function inBounds(r: number, c: number): boolean {
  return r >= 0 && c >= 0 && r < SIZE && c < SIZE;
}

function key({ r, c }: ActionCoordinate): string {
  return `${r},${c}`;
}

function clone(board: Board): Board {
  return board.map((row) => [...row]);
}

function initialState(reverse = false): SearchState {
  const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(EMPTY) as Board[number]);
  board[reverse ? 5 : 4][reverse ? 5 : 4] = SWAN_SUN;
  board[reverse ? 3 : 6][reverse ? 7 : 6] = SWAN_MOON;
  return { board, current: MOON, movesLeft: 1 };
}

function uniqueNumbers(numbers: number[]): number[] {
  return [...new Set(numbers)];
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}
