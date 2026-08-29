import test from "node:test";
import assert from "node:assert/strict";
import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board
} from "../src/renderer/game/encirclement";
import {
  applyAction,
  applyActionToBoard,
  actionKey,
  generateLegalActions,
  isLegalSwanPlacement,
  outcomeAfterAction,
  simulatePush,
  simulateSwanMove
} from "../src/renderer/game/rulesEngine";

test("documented Swan placement rejects diagonal enemy contact", () => {
  const board = emptyBoard();
  board[5][4] = SWAN_SUN;
  board[4][6] = SWAN_MOON;
  assert.equal(isLegalSwanPlacement(board, 1, 5, 5), false);
});

test("a Swan may move into a Stone square when that Stone follows out", () => {
  const board = emptyBoard();
  board[5][4] = SWAN_SUN;
  board[5][5] = STONE;
  const result = simulateSwanMove(board, 1, [{ r: 5, c: 4 }], [0, 1]);
  assert.ok(result);
  assert.equal(result.board[5][5], SWAN_SUN);
  assert.equal(result.board[5][6], STONE);
});

test("a frozen friendly Swan anchors a shared Stone", () => {
  const board = emptyBoard();
  board[5][4] = SWAN_SUN;
  board[5][5] = STONE;
  board[4][5] = FROZEN_SUN;
  const result = simulateSwanMove(board, 1, [{ r: 5, c: 4 }], [0, -1]);
  assert.ok(result);
  assert.equal(result.board[5][5], STONE);
  assert.deepEqual(result.stonesFrom, []);
});

test("push simulation preserves the pushed owner and moves an unshared Stone", () => {
  const board = emptyBoard();
  board[5][4] = SWAN_SUN;
  board[5][5] = SWAN_MOON;
  board[5][6] = STONE;
  const result = simulatePush(board, 1, [{ r: 5, c: 5 }], [0, 1]);
  assert.ok(result);
  assert.equal(result.board[5][6], SWAN_MOON);
  assert.equal(result.board[5][7], STONE);
});

test("simultaneous final encirclement resolves as a draw", () => {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[9][9] = SWAN_MOON;
  board[0][1] = STONE;
  board[1][0] = STONE;
  board[8][8] = STONE;
  board[8][9] = STONE;
  board[9][8] = STONE;
  const result = applyAction({ board, current: 1, movesLeft: 1 }, { type: "stone", r: 1, c: 1 });
  assert.ok(result);
  // The Sun placement seals itself while the already closed Moon group resolves
  // in the same position.
  assert.equal(result.outcome, "draw");
  assert.equal(result.freeze.sealedSun, 1);
  assert.equal(result.freeze.sealedMoon, 1);
});

test("live-executor saturation declares a full occupied board a draw", () => {
  const board = Array.from({ length: 10 }, () => Array(10).fill(STONE) as Board[number]);
  board[0][0] = SWAN_SUN;
  board[9][9] = SWAN_MOON;
  const noFreeze = {
    nb: board,
    frozeSun: 0,
    frozeMoon: 0,
    sealedSun: 0,
    sealedMoon: 0,
    frozenGroups: []
  };
  assert.equal(outcomeAfterAction(board, noFreeze), "draw");

  board[5][5] = EMPTY;
  assert.equal(outcomeAfterAction(board, { ...noFreeze, nb: board }), null);
});

test("live-executor active-Swan safety outcome is preserved", () => {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[6][6] = FROZEN_MOON;
  const noFreeze = {
    nb: board,
    frozeSun: 0,
    frozeMoon: 0,
    sealedSun: 0,
    sealedMoon: 0,
    frozenGroups: []
  };
  assert.equal(outcomeAfterAction(board, noFreeze), "sun");
});

test("every generated action applies without mutating its input", () => {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  board[4][5] = STONE;
  const original = board.map((row) => [...row]);
  const actions = generateLegalActions({ board, current: 1, movesLeft: 1 });
  assert.ok(actions.length > 0);
  for (const action of actions) assert.ok(applyActionToBoard(board, 1, action), JSON.stringify(action));
  assert.deepEqual(board, original);
});

test("multi-freezes award one bonus action per enemy Swan", () => {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[5][5] = SWAN_MOON;
  board[5][6] = SWAN_MOON;
  board[9][9] = SWAN_MOON;
  for (const [r, c] of [[4,4],[4,5],[4,6],[4,7],[5,4],[5,7],[6,4],[6,5],[6,6],[6,7]] as Array<[number,number]>) {
    board[r][c] = STONE;
  }
  const result = applyAction({ board, current: 1, movesLeft: 1 }, { type: "stone", r: 0, c: 1 });
  assert.ok(result);
  assert.equal(result.opponentLoss, 2);
  assert.equal(result.current, 1);
  assert.equal(result.movesLeft, 2);
});

test("Swan cap counts frozen Swans", () => {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[5][4] = FROZEN_SUN;
  board[4][5] = FROZEN_SUN;
  board[4][4] = FROZEN_SUN;
  board[6][5] = FROZEN_SUN;
  board[6][4] = FROZEN_SUN;
  assert.equal(isLegalSwanPlacement(board, 1, 5, 6), false);
});

test("reachable-state action generation remains unique, legal, and immutable", () => {
  let state = initialState();
  const random = seededRandom(0x51a7e);
  for (let step = 0; step < 120; step += 1) {
    const before = state.board.map((row) => [...row]);
    const actions = generateLegalActions(state);
    assert.equal(new Set(actions.map(actionKey)).size, actions.length, `duplicates at step ${step}`);
    assert.deepEqual(state.board, before, `generation mutated step ${step}`);
    if (actions.length === 0) {
      state = initialState();
      continue;
    }
    for (const index of [0, Math.floor(actions.length / 2), actions.length - 1]) {
      assert.ok(applyActionToBoard(state.board, state.current, actions[index]), `illegal generated action at step ${step}`);
    }
    const chosen = actions[Math.floor(random() * actions.length)];
    const next = applyAction(state, chosen);
    assert.ok(next);
    state = next.outcome ? initialState() : { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }
});

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function initialState(): { board: Board; current: 1 | 2; movesLeft: number } {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  return { board, current: 2, movesLeft: 1 };
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}
