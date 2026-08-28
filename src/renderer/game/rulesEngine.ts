import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  computeFreezesOn,
  type Board,
  type FreezeResult,
  type Player,
  type Tile
} from "./encirclement";

export type Direction = readonly [number, number];
export interface ActionCoordinate { r: number; c: number }

export type LinithAction =
  | { type: "stone"; r: number; c: number; score?: number }
  | { type: "swan"; r: number; c: number; score?: number }
  | { type: "move"; swans: ActionCoordinate[]; dir: Direction; score?: number }
  | { type: "push"; swans: ActionCoordinate[]; dir: Direction; score?: number };

export interface SearchState {
  board: Board;
  current: Player;
  movesLeft: number;
}

export type GameOutcome = "sun" | "moon" | "draw" | null;

export interface AppliedAction extends SearchState {
  freeze: FreezeResult;
  outcome: GameOutcome;
  actor: Player;
  opponentLoss: number;
}

export interface MoveSimulation {
  board: Board;
  stonesFrom: ActionCoordinate[];
  stonesTo: ActionCoordinate[];
}

export const BOARD_SIZE = 10;
export const SUN = 1 as const;
export const MOON = 2 as const;
export const DIRECTIONS: Direction[] = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [-1, 1], [1, -1], [1, 1]
];

const ORTHOGONAL_DIRECTIONS: Direction[] = [[-1, 0], [1, 0], [0, -1], [0, 1]];

export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

export function opponentOf(player: Player): Player {
  return player === SUN ? MOON : SUN;
}

export function inBounds(row: number, column: number): boolean {
  return row >= 0 && column >= 0 && row < BOARD_SIZE && column < BOARD_SIZE;
}

export function isActiveSwan(tile: Tile, player?: Player): boolean {
  if (player === SUN) return tile === SWAN_SUN;
  if (player === MOON) return tile === SWAN_MOON;
  return tile === SWAN_SUN || tile === SWAN_MOON;
}

export function isPlayerSwan(tile: Tile, player: Player): boolean {
  return player === SUN
    ? tile === SWAN_SUN || tile === FROZEN_SUN
    : tile === SWAN_MOON || tile === FROZEN_MOON;
}

export function isEnemySwan(tile: Tile, player: Player): boolean {
  return isPlayerSwan(tile, opponentOf(player));
}

export function countTotalSwans(board: Board, player: Player): number {
  let count = 0;
  for (const row of board) for (const tile of row) if (isPlayerSwan(tile, player)) count += 1;
  return count;
}

export function countActiveSwans(board: Board, player: Player): number {
  let count = 0;
  for (const row of board) for (const tile of row) if (isActiveSwan(tile, player)) count += 1;
  return count;
}

export function activeSwanCoordinates(board: Board, player: Player): ActionCoordinate[] {
  const coordinates: ActionCoordinate[] = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (isActiveSwan(board[r][c], player)) coordinates.push({ r, c });
    }
  }
  return coordinates;
}

export function isLegalSwanPlacement(board: Board, player: Player, r: number, c: number): boolean {
  if (!inBounds(r, c) || board[r][c] !== EMPTY || countTotalSwans(board, player) >= 6) return false;

  let touchesFriendlyOrthogonally = false;
  for (const [dr, dc] of ORTHOGONAL_DIRECTIONS) {
    const nr = r + dr;
    const nc = c + dc;
    if (inBounds(nr, nc) && isPlayerSwan(board[nr][nc], player)) touchesFriendlyOrthogonally = true;
  }
  if (!touchesFriendlyOrthogonally) return false;

  for (const [dr, dc] of DIRECTIONS) {
    const nr = r + dr;
    const nc = c + dc;
    if (inBounds(nr, nc) && isEnemySwan(board[nr][nc], player)) return false;
  }
  return true;
}

export function legalSwanPlacements(board: Board, player: Player): ActionCoordinate[] {
  const actions: ActionCoordinate[] = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (isLegalSwanPlacement(board, player, r, c)) actions.push({ r, c });
    }
  }
  return actions;
}

export function legalStonePlacements(board: Board): ActionCoordinate[] {
  const actions: ActionCoordinate[] = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (board[r][c] === EMPTY) actions.push({ r, c });
    }
  }
  return actions;
}

function coordinateKey({ r, c }: ActionCoordinate): string {
  return `${r},${c}`;
}

function neighbours(r: number, c: number): ActionCoordinate[] {
  return DIRECTIONS
    .map(([dr, dc]) => ({ r: r + dr, c: c + dc }))
    .filter(({ r: nr, c: nc }) => inBounds(nr, nc));
}

function isNakedEnemyZone(board: Board, player: Player, r: number, c: number): boolean {
  for (const enemy of neighbours(r, c)) {
    if (!isEnemySwan(board[enemy.r][enemy.c], player)) continue;
    const hasStone = neighbours(enemy.r, enemy.c).some(({ r: sr, c: sc }) => board[sr][sc] === STONE);
    if (!hasStone) return true;
  }
  return false;
}

function collectFollowingStones(
  board: Board,
  movingPlayer: Player,
  movingSwans: ActionCoordinate[],
  dir: Direction
): { from: ActionCoordinate[]; to: ActionCoordinate[] } | null {
  const moving = new Set(movingSwans.map(coordinateKey));
  const from = new Map<string, ActionCoordinate>();

  for (const swan of movingSwans) {
    for (const stone of neighbours(swan.r, swan.c)) {
      if (board[stone.r][stone.c] !== STONE) continue;
      const key = coordinateKey(stone);
      if (from.has(key)) continue;

      const shared = neighbours(stone.r, stone.c).some((adjacent) => {
        const tile = board[adjacent.r][adjacent.c];
        if (isEnemySwan(tile, movingPlayer)) return true;
        return isPlayerSwan(tile, movingPlayer) && !moving.has(coordinateKey(adjacent));
      });
      if (!shared) from.set(key, stone);
    }
  }

  const origins = [...from.values()];
  const originKeys = new Set(origins.map(coordinateKey));
  const destinations = origins.map(({ r, c }) => ({ r: r + dir[0], c: c + dir[1] }));
  const destinationKeys = new Set<string>();

  for (const destination of destinations) {
    if (!inBounds(destination.r, destination.c)) return null;
    const key = coordinateKey(destination);
    if (destinationKeys.has(key)) return null;
    destinationKeys.add(key);

    const tile = board[destination.r][destination.c];
    if (tile !== EMPTY && !moving.has(key) && !originKeys.has(key)) return null;
  }
  return { from: origins, to: destinations };
}

function normalizeCoordinates(swans: readonly ActionCoordinate[]): ActionCoordinate[] | null {
  const normalized = swans.map(({ r, c }) => ({ r, c }));
  const unique = new Set(normalized.map(coordinateKey));
  return normalized.length > 0 && unique.size === normalized.length ? normalized : null;
}

export function simulateSwanMove(
  board: Board,
  player: Player,
  swans: readonly ActionCoordinate[],
  dir: Direction
): MoveSimulation | null {
  const movingSwans = normalizeCoordinates(swans);
  if (!movingSwans || !DIRECTIONS.some(([dr, dc]) => dr === dir[0] && dc === dir[1])) return null;
  if (movingSwans.some(({ r, c }) => !inBounds(r, c) || !isActiveSwan(board[r][c], player))) return null;

  const movingKeys = new Set(movingSwans.map(coordinateKey));
  const stones = collectFollowingStones(board, player, movingSwans, dir);
  if (!stones) return null;
  const stoneOrigins = new Set(stones.from.map(coordinateKey));
  const swanDestinations = new Set<string>();

  for (const { r, c } of movingSwans) {
    const destination = { r: r + dir[0], c: c + dir[1] };
    if (!inBounds(destination.r, destination.c) || isNakedEnemyZone(board, player, destination.r, destination.c)) {
      return null;
    }
    const key = coordinateKey(destination);
    if (swanDestinations.has(key)) return null;
    swanDestinations.add(key);

    const tile = board[destination.r][destination.c];
    if (tile === EMPTY || movingKeys.has(key) || (tile === STONE && stoneOrigins.has(key))) continue;
    return null;
  }

  if (stones.to.some((destination) => swanDestinations.has(coordinateKey(destination)))) return null;

  const next = cloneBoard(board);
  for (const { r, c } of movingSwans) next[r][c] = EMPTY;
  for (const { r, c } of stones.from) next[r][c] = EMPTY;
  for (const { r, c } of stones.to) next[r][c] = STONE;
  for (const { r, c } of movingSwans) next[r + dir[0]][c + dir[1]] = player === SUN ? SWAN_SUN : SWAN_MOON;
  return { board: next, stonesFrom: stones.from, stonesTo: stones.to };
}

export function simulatePush(
  board: Board,
  player: Player,
  swans: readonly ActionCoordinate[],
  dir: Direction
): MoveSimulation | null {
  const pushedSwans = normalizeCoordinates(swans);
  if (!pushedSwans || !DIRECTIONS.some(([dr, dc]) => dr === dir[0] && dc === dir[1])) return null;
  const pushedPlayer = opponentOf(player);
  if (pushedSwans.some(({ r, c }) => !inBounds(r, c) || !isActiveSwan(board[r][c], pushedPlayer))) return null;

  for (const swan of pushedSwans) {
    const hasPusher = neighbours(swan.r, swan.c).some(({ r, c }) => isActiveSwan(board[r][c], player));
    if (!hasPusher) return null;
  }

  const movingKeys = new Set(pushedSwans.map(coordinateKey));
  const stones = collectFollowingStones(board, pushedPlayer, pushedSwans, dir);
  if (!stones) return null;
  const stoneOrigins = new Set(stones.from.map(coordinateKey));
  const swanDestinations = new Set<string>();

  for (const { r, c } of pushedSwans) {
    const destination = { r: r + dir[0], c: c + dir[1] };
    if (!inBounds(destination.r, destination.c)) return null;
    const key = coordinateKey(destination);
    if (swanDestinations.has(key)) return null;
    swanDestinations.add(key);

    const tile = board[destination.r][destination.c];
    if (tile === EMPTY || movingKeys.has(key) || (tile === STONE && stoneOrigins.has(key))) continue;
    return null;
  }

  if (stones.to.some((destination) => swanDestinations.has(coordinateKey(destination)))) return null;

  const next = cloneBoard(board);
  for (const { r, c } of pushedSwans) next[r][c] = EMPTY;
  for (const { r, c } of stones.from) next[r][c] = EMPTY;
  for (const { r, c } of stones.to) next[r][c] = STONE;
  for (const { r, c } of pushedSwans) next[r + dir[0]][c + dir[1]] = pushedPlayer === SUN ? SWAN_SUN : SWAN_MOON;
  return { board: next, stonesFrom: stones.from, stonesTo: stones.to };
}

function* nonEmptySubsets<T>(items: readonly T[]): Generator<T[]> {
  const count = 1 << items.length;
  for (let mask = 1; mask < count; mask += 1) {
    const subset: T[] = [];
    for (let index = 0; index < items.length; index += 1) {
      if (mask & (1 << index)) subset.push(items[index]);
    }
    yield subset;
  }
}

export function generateLegalActions(state: SearchState): LinithAction[] {
  const { board, current } = state;
  const actions: LinithAction[] = legalStonePlacements(board).map(({ r, c }) => ({ type: "stone", r, c }));
  actions.push(...legalSwanPlacements(board, current).map(({ r, c }) => ({ type: "swan" as const, r, c })));

  const own = activeSwanCoordinates(board, current);
  for (const subset of nonEmptySubsets(own)) {
    for (const dir of DIRECTIONS) {
      if (simulateSwanMove(board, current, subset, dir)) actions.push({ type: "move", swans: subset, dir });
    }
  }

  const enemy = activeSwanCoordinates(board, opponentOf(current));
  for (const subset of nonEmptySubsets(enemy)) {
    for (const dir of DIRECTIONS) {
      if (simulatePush(board, current, subset, dir)) actions.push({ type: "push", swans: subset, dir });
    }
  }
  return actions;
}

export function applyActionToBoard(board: Board, player: Player, action: LinithAction): Board | null {
  if (action.type === "stone") {
    if (!inBounds(action.r, action.c) || board[action.r][action.c] !== EMPTY) return null;
    const next = cloneBoard(board);
    next[action.r][action.c] = STONE;
    return next;
  }
  if (action.type === "swan") {
    if (!isLegalSwanPlacement(board, player, action.r, action.c)) return null;
    const next = cloneBoard(board);
    next[action.r][action.c] = player === SUN ? SWAN_SUN : SWAN_MOON;
    return next;
  }
  const simulation = action.type === "move"
    ? simulateSwanMove(board, player, action.swans, action.dir)
    : simulatePush(board, player, action.swans, action.dir);
  return simulation?.board ?? null;
}

export function outcomeFromFreeze(freeze: FreezeResult): GameOutcome {
  if (freeze.sealedSun > 0 && freeze.sealedMoon > 0) return "draw";
  if (freeze.sealedSun > 0) return "moon";
  if (freeze.sealedMoon > 0) return "sun";
  return null;
}

export function nextTurnAfterFreeze(
  board: Board,
  current: Player,
  movesLeft: number,
  freeze: Pick<FreezeResult, "frozeSun" | "frozeMoon" | "sealedSun" | "sealedMoon">
): { current: Player; movesLeft: number; opponentLoss: number } {
  const opponentLoss = current === SUN
    ? freeze.frozeMoon + freeze.sealedMoon
    : freeze.frozeSun + freeze.sealedSun;
  const remaining = movesLeft + opponentLoss - 1;
  if (remaining > 0) return { current, movesLeft: remaining, opponentLoss };
  const next = opponentOf(current);
  const bothAtSix = countTotalSwans(board, SUN) >= 6 && countTotalSwans(board, MOON) >= 6;
  return { current: next, movesLeft: bothAtSix ? 2 : 1, opponentLoss };
}

export function applyAction(state: SearchState, action: LinithAction): AppliedAction | null {
  const moved = applyActionToBoard(state.board, state.current, action);
  if (!moved) return null;
  const freeze = computeFreezesOn(moved);
  const outcome = outcomeFromFreeze(freeze);
  const turn = nextTurnAfterFreeze(freeze.nb, state.current, state.movesLeft, freeze);
  return {
    board: freeze.nb,
    current: turn.current,
    movesLeft: turn.movesLeft,
    freeze,
    outcome,
    actor: state.current,
    opponentLoss: turn.opponentLoss
  };
}

export function actionKey(action: LinithAction): string {
  if (action.type === "stone" || action.type === "swan") return `${action.type}:${action.r},${action.c}`;
  const swans = action.swans.map(coordinateKey).sort().join(";");
  return `${action.type}:${swans}:${action.dir[0]},${action.dir[1]}`;
}

export function boardKey(state: SearchState): string {
  return `${state.current}:${state.movesLeft}:${state.board.flat().join("")}`;
}
