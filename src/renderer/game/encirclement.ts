export const EMPTY = 0 as const;
export const SWAN_SUN = 1 as const;
export const SWAN_MOON = 2 as const;
export const STONE = 3 as const;
export const FROZEN_SUN = 4 as const;
export const FROZEN_MOON = 5 as const;

export type Tile = 0 | 1 | 2 | 3 | 4 | 5;
export type Player = 1 | 2;
export type Coordinate = [number, number];
export type Board = Tile[][];

export interface FrozenGroup {
  owner: Player;
  tiles: Coordinate[];
}

export interface FreezeResult {
  nb: Board;
  frozeSun: number;
  frozeMoon: number;
  sealedSun: number;
  sealedMoon: number;
  frozenGroups: FrozenGroup[];
}

const SUN = 1 as const;
const MOON = 2 as const;
const DIRECTIONS: Coordinate[] = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1], [0, 1],
  [1, -1], [1, 0], [1, 1]
];

export function computeFreezesOn(board: Board): FreezeResult {
  const size = board.length;
  const nextBoard = board.map((row) => [...row]);
  const seen = Array.from({ length: size }, () => Array<boolean>(size).fill(false));
  const frozenGroups: FrozenGroup[] = [];
  let frozeSun = 0;
  let frozeMoon = 0;
  let sealedSun = 0;
  let sealedMoon = 0;

  const inBounds = (row: number, column: number): boolean =>
    row >= 0 && column >= 0 && row < size && column < size;

  const playerOf = (tile: Tile): Player | null => {
    if (tile === SWAN_SUN || tile === FROZEN_SUN) return SUN;
    if (tile === SWAN_MOON || tile === FROZEN_MOON) return MOON;
    return null;
  };

  const collectGroup = (startRow: number, startColumn: number): FrozenGroup => {
    const owner = playerOf(nextBoard[startRow][startColumn]);
    if (!owner) throw new Error("Cannot collect a group from a non-Swan tile.");

    const queue: Coordinate[] = [[startRow, startColumn]];
    const tiles: Coordinate[] = [];
    seen[startRow][startColumn] = true;

    while (queue.length > 0) {
      const [row, column] = queue.pop()!;
      tiles.push([row, column]);
      for (const [rowDelta, columnDelta] of DIRECTIONS) {
        const neighbourRow = row + rowDelta;
        const neighbourColumn = column + columnDelta;
        if (!inBounds(neighbourRow, neighbourColumn) || seen[neighbourRow][neighbourColumn]) continue;

        const tile = nextBoard[neighbourRow][neighbourColumn];
        if ((tile === SWAN_SUN || tile === SWAN_MOON) && playerOf(tile) === owner) {
          seen[neighbourRow][neighbourColumn] = true;
          queue.push([neighbourRow, neighbourColumn]);
        }
      }
    }

    return { owner, tiles };
  };

  const isEncircled = ({ owner, tiles }: FrozenGroup): boolean => {
    const group = new Set(tiles.map(([row, column]) => `${row},${column}`));
    for (const [row, column] of tiles) {
      for (const [rowDelta, columnDelta] of DIRECTIONS) {
        const neighbourRow = row + rowDelta;
        const neighbourColumn = column + columnDelta;
        if (!inBounds(neighbourRow, neighbourColumn)) continue;

        const tile = nextBoard[neighbourRow][neighbourColumn];
        if (tile === EMPTY) return false;

        const isWall =
          tile === STONE ||
          tile === FROZEN_SUN ||
          tile === FROZEN_MOON ||
          ((tile === SWAN_SUN || tile === SWAN_MOON) && playerOf(tile) !== owner);
        if (!isWall && !group.has(`${neighbourRow},${neighbourColumn}`)) return false;
      }
    }
    return true;
  };

  const countActiveSwans = (owner: Player): number =>
    nextBoard.flat().filter((tile) => tile === (owner === SUN ? SWAN_SUN : SWAN_MOON)).length;

  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const tile = nextBoard[row][column];
      if ((tile !== SWAN_SUN && tile !== SWAN_MOON) || seen[row][column]) continue;

      const group = collectGroup(row, column);
      if (!isEncircled(group)) continue;

      const remaining = countActiveSwans(group.owner);
      for (const [groupRow, groupColumn] of group.tiles) {
        nextBoard[groupRow][groupColumn] = group.owner === SUN ? FROZEN_SUN : FROZEN_MOON;
      }
      frozenGroups.push(group);

      if (remaining === group.tiles.length) {
        if (group.owner === SUN) sealedSun += group.tiles.length;
        else sealedMoon += group.tiles.length;
      } else if (group.owner === SUN) {
        frozeSun += group.tiles.length;
      } else {
        frozeMoon += group.tiles.length;
      }
    }
  }

  return { nb: nextBoard, frozeSun, frozeMoon, sealedSun, sealedMoon, frozenGroups };
}
