import test from "node:test";
import assert from "node:assert/strict";
import {
  EMPTY,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board
} from "../src/renderer/game/encirclement";
import {
  chooseVeryHardTimeBudget,
  detectVeryHardPlatform
} from "../src/renderer/game/veryHard/timeManager";

test("Very Hard gives sparse openings a short allowance", () => {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;

  assert.deepEqual(
    chooseVeryHardTimeBudget({ board, current: 2, movesLeft: 1 }, "desktop"),
    {
      budgetMs: 160,
      hardLimitMs: 300,
      complexity: "opening",
      reasons: ["sparse opening"]
    }
  );
});

test("Very Hard spends its largest bounded allowance near an encirclement", () => {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[0][1] = STONE;
  board[5][5] = SWAN_MOON;
  board[5][6] = STONE;
  board[7][7] = STONE;

  const timing = chooseVeryHardTimeBudget({ board, current: 1, movesLeft: 1 }, "android");
  assert.equal(timing.complexity, "critical");
  assert.equal(timing.budgetMs, 900);
  assert.equal(timing.hardLimitMs, 1_000);
  assert.ok(timing.reasons.includes("group near encirclement"));
});

test("the two-action phase is classified as complex without exceeding desktop latency bounds", () => {
  const board = emptyBoard();
  for (let c = 0; c < 6; c += 1) {
    board[2][c] = SWAN_SUN;
    board[7][c] = SWAN_MOON;
  }

  const timing = chooseVeryHardTimeBudget({ board, current: 1, movesLeft: 2 }, "desktop");
  assert.equal(timing.complexity, "complex");
  assert.equal(timing.budgetMs, 475);
  assert.ok(timing.hardLimitMs <= 750);
  assert.ok(timing.reasons.includes("two-action phase"));
  assert.ok(timing.reasons.includes("same-turn continuation"));
});

test("platform detection gives Android priority over a desktop bridge", () => {
  assert.equal(detectVeryHardPlatform("Mozilla/5.0 (Linux; Android 15)", true), "android");
  assert.equal(detectVeryHardPlatform("Mozilla/5.0", true), "desktop");
  assert.equal(detectVeryHardPlatform("Mozilla/5.0", false), "browser");
});

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}
