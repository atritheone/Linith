import test from "node:test";
import assert from "node:assert/strict";
import { linithAI } from "../src/renderer/game/ai";
import type { Board } from "../src/renderer/game/encirclement";
import { applyActionToBoard } from "../src/renderer/game/rulesEngine";

const STYLES = ["doctrinal", "constrictor", "rupture", "blizzard", "librarian", "swarm", "fortress"] as const;
const DIFFICULTIES = ["easy", "medium", "hard"] as const;

const fixtures: Array<{ name: string; board: Board; current: 1 | 2 }> = [
  {
    name: "early development",
    current: 1,
    board: [
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,2,0],
      [0,0,0,0,1,0,2,0,0,0], [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]
    ]
  },
  {
    name: "late development",
    current: 1,
    board: [
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,1,0,0,2,0], [0,0,0,0,0,0,0,0,2,0],
      [0,0,0,1,1,0,2,0,0,0], [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,1,0,0,2,0], [0,0,0,0,0,1,0,2,0,0],
      [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]
    ]
  }
];

test("unaffected Easy, Medium, and Hard decisions retain the captured 0.232 behavior", () => {
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  let style = "doctrinal";
  globalThis.window = { linithGetStyle: () => style } as Window & typeof globalThis;
  Math.random = () => 0.9; // stable shuffles; Medium deliberately skips its 70% probe

  try {
    for (const fixture of fixtures) {
      for (const styleName of STYLES) {
        style = styleName;
        for (const difficulty of DIFFICULTIES) {
          const selected = linithAI(fixture.board, fixture.current, difficulty);
          const actual = normalize(selected);
          assert.deepEqual(actual, expected(fixture.name, styleName, difficulty),
            `${fixture.name} / ${styleName} / ${difficulty}`);
          assert.ok(selected && applyActionToBoard(fixture.board, fixture.current, selected),
            `selected action must remain legal: ${fixture.name} / ${styleName} / ${difficulty}`);
        }
      }
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

test("an unknown difficulty is normalized to Medium rather than Easy", () => {
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  globalThis.window = { linithGetStyle: () => "doctrinal" } as Window & typeof globalThis;
  Math.random = () => 0.9;
  try {
    const fixture = fixtures[0];
    assert.deepEqual(
      normalize(linithAI(fixture.board, fixture.current, "unknown")),
      normalize(linithAI(fixture.board, fixture.current, "medium"))
    );
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

test("corrected Easy, Medium, and Hard decisions are frozen on a Stone-and-freeze position", () => {
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  let style = "doctrinal";
  globalThis.window = { linithGetStyle: () => style } as Window & typeof globalThis;
  Math.random = () => 0.9;
  try {
    for (const styleName of STYLES) {
      style = styleName;
      for (const difficulty of DIFFICULTIES) {
        const action = linithAI(correctedFixture, 1, difficulty);
        assert.deepEqual(normalize(action), correctedExpected(styleName, difficulty), `${styleName} / ${difficulty}`);
        assert.ok(action && applyActionToBoard(correctedFixture, 1, action));
      }
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

const correctedFixture: Board = [
  [0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,2,0],
  [0,0,0,0,0,1,0,0,0,0], [0,1,3,0,0,0,0,0,0,0],
  [0,3,3,0,1,0,0,3,2,0], [0,3,3,3,0,0,3,3,3,0],
  [0,3,4,3,0,0,3,5,3,3], [3,3,3,3,0,0,3,3,5,3],
  [3,1,3,0,2,0,3,5,3,3], [0,0,0,0,0,0,3,3,3,0]
];

function correctedExpected(style: string, difficulty: string): unknown {
  if (style === "doctrinal") {
    if (difficulty === "easy") return { type: "stone", r: 9, c: 3 };
    return { type: "move", swans: [{ r: 2, c: 5 }], dir: [-1, 1] };
  }
  if (difficulty === "easy") return { type: "stone", r: 1, c: 7 };
  if (difficulty === "medium") return { type: "stone", r: 2, c: 8 };
  if (style === "fortress") {
    return {
      type: "move",
      swans: [{ r: 1, c: 3 }, { r: 2, c: 5 }, { r: 3, c: 1 }, { r: 4, c: 4 }],
      dir: [0, 1]
    };
  }
  return { type: "stone", r: 0, c: 9 };
}

function expected(fixture: string, style: string, difficulty: string): unknown {
  const doctrinal = style === "doctrinal";
  if (fixture === "early development") {
    if (difficulty === "easy") return { type: "swan", r: 2, c: 4 };
    return doctrinal
      ? { type: "move", swans: [{ r: 3, c: 4 }], dir: [-1, -1] }
      : { type: "move", swans: [{ r: 4, c: 4 }], dir: [1, 0] };
  }
  if (difficulty === "easy") return doctrinal
    ? { type: "swan", r: 1, c: 5 }
    : { type: "swan", r: 2, c: 6 };
  return doctrinal
    ? { type: "move", swans: [{ r: 4, c: 3 }, { r: 7, c: 5 }], dir: [1, -1] }
    : { type: "move", swans: [{ r: 2, c: 5 }], dir: [-1, 1] };
}

function normalize(action: ReturnType<typeof linithAI>): unknown {
  if (!action) return null;
  if (action.type === "stone" || action.type === "swan") return { type: action.type, r: action.r, c: action.c };
  return { type: action.type, swans: action.swans, dir: action.dir };
}
