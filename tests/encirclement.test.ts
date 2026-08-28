import test from "node:test";
import assert from "node:assert/strict";
import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board,
  computeFreezesOn
} from "../src/renderer/game/encirclement";

test("an open Swan remains active", () => {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;

  const result = computeFreezesOn(board);

  assert.equal(result.nb[5][5], SWAN_SUN);
  assert.equal(result.frozeSun, 0);
  assert.equal(result.sealedSun, 0);
});

test("a player's last encircled Swan is sealed", () => {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  surroundWithStones(board, 5, 5);

  const result = computeFreezesOn(board);

  assert.equal(result.nb[5][5], FROZEN_SUN);
  assert.equal(result.sealedSun, 1);
  assert.equal(result.frozeSun, 0);
  assert.deepEqual(result.frozenGroups, [{ owner: 1, tiles: [[5, 5]] }]);
});

test("an encircled Swan freezes when another active Swan remains", () => {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_SUN;
  surroundWithStones(board, 5, 5);

  const result = computeFreezesOn(board);

  assert.equal(result.nb[5][5], FROZEN_SUN);
  assert.equal(result.nb[0][0], SWAN_SUN);
  assert.equal(result.frozeSun, 1);
  assert.equal(result.sealedSun, 0);
});

test("diagonally connected Swans freeze as one group without mutating the input", () => {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[4][4] = SWAN_SUN;
  board[5][5] = SWAN_SUN;
  surroundGroupWithStones(board, [[4, 4], [5, 5]]);
  const original = board.map((row) => [...row]);

  const result = computeFreezesOn(board);

  assert.deepEqual(board, original);
  assert.equal(result.nb[4][4], FROZEN_SUN);
  assert.equal(result.nb[5][5], FROZEN_SUN);
  assert.equal(result.frozeSun, 2);
  assert.deepEqual(result.frozenGroups, [{ owner: 1, tiles: [[4, 4], [5, 5]] }]);
});

test("the board edge closes an otherwise surrounded group", () => {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[0][1] = STONE;
  board[1][0] = STONE;
  board[1][1] = STONE;

  const result = computeFreezesOn(board);

  assert.equal(result.nb[0][0], FROZEN_SUN);
  assert.equal(result.sealedSun, 1);
});

test("both players can be sealed by the same resolved position", () => {
  const board = emptyBoard();
  board[2][2] = SWAN_SUN;
  board[7][7] = SWAN_MOON;
  surroundWithStones(board, 2, 2);
  surroundWithStones(board, 7, 7);

  const result = computeFreezesOn(board);

  assert.equal(result.nb[2][2], FROZEN_SUN);
  assert.equal(result.nb[7][7], FROZEN_MOON);
  assert.equal(result.sealedSun, 1);
  assert.equal(result.sealedMoon, 1);
});

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function surroundWithStones(board: Board, row: number, column: number): void {
  for (let rowDelta = -1; rowDelta <= 1; rowDelta += 1) {
    for (let columnDelta = -1; columnDelta <= 1; columnDelta += 1) {
      if (rowDelta !== 0 || columnDelta !== 0) {
        board[row + rowDelta][column + columnDelta] = STONE;
      }
    }
  }
}

function surroundGroupWithStones(board: Board, group: Array<[number, number]>): void {
  const groupCoordinates = new Set(group.map(([row, column]) => `${row},${column}`));
  for (const [row, column] of group) {
    for (let rowDelta = -1; rowDelta <= 1; rowDelta += 1) {
      for (let columnDelta = -1; columnDelta <= 1; columnDelta += 1) {
        const neighbourRow = row + rowDelta;
        const neighbourColumn = column + columnDelta;
        if (
          (rowDelta !== 0 || columnDelta !== 0) &&
          neighbourRow >= 0 && neighbourRow < board.length &&
          neighbourColumn >= 0 && neighbourColumn < board.length &&
          !groupCoordinates.has(`${neighbourRow},${neighbourColumn}`)
        ) {
          board[neighbourRow][neighbourColumn] = STONE;
        }
      }
    }
  }
}
