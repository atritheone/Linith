import test from "node:test";
import assert from "node:assert/strict";
import { linithAI } from "../src/renderer/game/ai";
import { AI_STYLE_IDS } from "../src/renderer/game/aiStyles";
import type { Board } from "../src/renderer/game/encirclement";
import { applyActionToBoard } from "../src/renderer/game/rulesEngine";

const STYLES = AI_STYLE_IDS;
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

test("Doctrinal retains captured 0.232 decisions while every personality remains legal", () => {
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
          if (styleName === "doctrinal") {
            assert.deepEqual(actual, expected(fixture.name, styleName, difficulty),
              `${fixture.name} / ${styleName} / ${difficulty}`);
          }
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

test("corrected Doctrinal decisions stay frozen and every corrected personality action is legal", () => {
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
        if (styleName === "doctrinal") {
          assert.deepEqual(normalize(action), correctedExpected(styleName, difficulty), `${styleName} / ${difficulty}`);
        }
        assert.ok(action && applyActionToBoard(correctedFixture, 1, action));
      }
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

test("Swarm keeps its early-development character at Medium and Hard", () => {
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  globalThis.window = { linithGetStyle: () => "swarm" } as Window & typeof globalThis;
  Math.random = () => 0.9;
  try {
    for (const fixture of fixtures) {
      for (const difficulty of ["medium", "hard"] as const) {
        const action = linithAI(fixture.board, fixture.current, difficulty);
        assert.equal(action?.type, "swan", `${fixture.name} / ${difficulty}`);
        assert.ok(action && applyActionToBoard(fixture.board, fixture.current, action));
      }
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

test("Easy and Medium keep a legal global Stone fallback when locality has no candidates", () => {
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  globalThis.window = { linithGetStyle: () => "doctrinal" } as Window & typeof globalThis;
  Math.random = () => 0.9;
  try {
    for (const difficulty of ["easy", "medium"] as const) {
      const action = linithAI(localityExhaustionFixture, 2, difficulty);
      assert.ok(action, `${difficulty} must not return null while a Stone placement is legal`);
      assert.equal(action.type, "stone");
      assert.ok(applyActionToBoard(localityExhaustionFixture, 2, action), `${difficulty} fallback must be legal`);
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

const localityExhaustionFixture: Board = [
  [3,3,3,0,3,5,3,0,0,0], [3,5,3,0,3,3,3,3,3,3],
  [3,3,3,0,0,0,0,3,5,3], [3,5,3,0,0,1,0,3,3,3],
  [3,5,3,3,1,3,0,0,0,0], [3,3,3,3,3,3,3,1,3,0],
  [3,3,3,0,0,0,3,0,0,0], [3,2,0,0,0,3,3,0,3,0],
  [3,3,3,3,0,3,1,0,1,0], [0,3,4,3,0,3,0,0,0,3]
];

const correctedFixture: Board = [
  [0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,2,0],
  [0,0,0,0,0,1,0,0,0,0], [0,1,3,0,0,0,0,0,0,0],
  [0,3,3,0,1,0,0,3,2,0], [0,3,3,3,0,0,3,3,3,0],
  [0,3,4,3,0,0,3,5,3,3], [3,3,3,3,0,0,3,3,5,3],
  [3,1,3,0,2,0,3,5,3,3], [0,0,0,0,0,0,3,3,3,0]
];

function correctedExpected(style: string, difficulty: string): unknown {
  assert.equal(style, "doctrinal");
  if (difficulty === "easy") return { type: "stone", r: 9, c: 3 };
  return { type: "move", swans: [{ r: 2, c: 5 }], dir: [-1, 1] };
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
