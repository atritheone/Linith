import test from "node:test";
import assert from "node:assert/strict";
import {
  AI_STYLE_IDS,
  AI_STYLE_LIST,
  AI_STYLES,
  aiPersonality,
  type AiPersonalityTraits
} from "../src/renderer/game/aiStyles";
import {
  EMPTY,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board
} from "../src/renderer/game/encirclement";
import {
  applyAction,
  generateLegalActions,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  evaluateVeryHardRootPersonality,
  explainVeryHardPosition,
  VERY_HARD_ROOT_PERSONALITY_LIMIT
} from "../src/renderer/game/veryHard/evaluate";

test("the personality registry is complete, ordered, and falls back to Doctrinal", () => {
  assert.deepEqual(AI_STYLE_LIST.map(({ id }) => id), [...AI_STYLE_IDS]);
  assert.equal(aiPersonality("future-style"), AI_STYLES.doctrinal);
  assert.deepEqual(AI_STYLES.doctrinal.traits, {});
});

test("each named character owns its defining strongest trait", () => {
  assert.equal(strongest("fragmentation"), "rupture");
  assert.equal(strongest("freezeUrgency"), "blizzard");
  assert.equal(strongest("sacrificeTolerance"), "blizzard");
  assert.equal(strongest("libertyBalance"), "librarian");
  assert.equal(strongest("development"), "swarm");
  assert.equal(weakest("earlyStone"), "swarm");
  assert.equal(strongest("containment"), "fortress");
  assert.equal(strongest("structure"), "fortress");
  assert.equal(strongest("selfPreservation"), "fortress");
  assert.ok((AI_STYLES.constrictor.traits.containment ?? 0) > 1);
});

test("Doctrinal is a true zero-bias baseline across personality feature positions", () => {
  for (const state of personalityStates()) {
    assert.equal(explainVeryHardPosition(state, 1, "doctrinal").style, 0);
    assert.equal(explainVeryHardPosition(state, 2, "doctrinal").style, 0);
  }
});

test("Very Hard's development and fragmentation tie-breaks express Swarm and Rupture", () => {
  const [development, fragmentation] = personalityStates();
  assert.equal(highestStyle(development), "swarm");
  assert.equal(highestStyle(fragmentation), "rupture");
});

test("Constrictor and Fortress dominate a nearly closed containment position", () => {
  const containment = personalityStates()[2];
  const ranked = rankedStyles(containment);
  assert.deepEqual(new Set(ranked.slice(0, 2)), new Set(["constrictor", "fortress"]));
});

test("personality scores remain exactly antisymmetric", () => {
  for (const state of personalityStates()) {
    for (const style of AI_STYLE_IDS) {
      const sun = explainVeryHardPosition(state, 1, style);
      const moon = explainVeryHardPosition(state, 2, style);
      assert.equal(sun.style + moon.style, 0, style);
      assert.equal(sun.total + moon.total, 0, style);
    }
  }
});

test("root personality is action-aware, bounded, and leaves Doctrinal exactly untouched", () => {
  const scores = new Map<string, number[]>();
  for (const style of AI_STYLE_IDS) scores.set(style, []);
  for (const state of personalityStates()) {
    for (const action of generateLegalActions(state)) {
      const next = applyAction(state, action);
      assert.ok(next);
      for (const style of AI_STYLE_IDS) {
        const score = evaluateVeryHardRootPersonality(state, action, next, state.current, style);
        assert.ok(Math.abs(score) <= VERY_HARD_ROOT_PERSONALITY_LIMIT, `${style}: ${score}`);
        scores.get(style)!.push(score);
      }
    }
  }

  assert.deepEqual(new Set(scores.get("doctrinal")), new Set([0]));
  for (const style of AI_STYLE_IDS.slice(1)) {
    const values = scores.get(style)!;
    assert.ok(Math.max(...values) > Math.min(...values), `${style} must distinguish concrete actions`);
  }
});

function strongest(trait: keyof AiPersonalityTraits): string {
  return [...AI_STYLE_LIST]
    .filter(({ id }) => id !== "doctrinal")
    .sort((a, b) => (b.traits[trait] ?? 0) - (a.traits[trait] ?? 0))[0].id;
}

function weakest(trait: keyof AiPersonalityTraits): string {
  return [...AI_STYLE_LIST]
    .filter(({ id }) => id !== "doctrinal")
    .sort((a, b) => (a.traits[trait] ?? 0) - (b.traits[trait] ?? 0))[0].id;
}

function rankedStyles(state: SearchState): string[] {
  return AI_STYLE_IDS
    .filter((style) => style !== "doctrinal")
    .map((style) => ({ style, score: explainVeryHardPosition(state, 1, style).style }))
    .sort((a, b) => b.score - a.score)
    .map(({ style }) => style);
}

function highestStyle(state: SearchState): string {
  return rankedStyles(state)[0];
}

function personalityStates(): SearchState[] {
  const development = emptyBoard();
  for (const [r, c] of [[4, 4], [4, 5], [5, 4], [5, 5]] as const) development[r][c] = SWAN_SUN;
  development[8][8] = SWAN_MOON;

  const fragmentation = emptyBoard();
  fragmentation[4][4] = SWAN_SUN;
  fragmentation[4][5] = SWAN_SUN;
  for (const [r, c] of [[1, 1], [1, 8], [8, 8]] as const) fragmentation[r][c] = SWAN_MOON;

  const containment = emptyBoard();
  containment[8][8] = SWAN_SUN;
  containment[4][4] = SWAN_MOON;
  for (const [r, c] of [[3, 3], [3, 4], [3, 5], [4, 3], [4, 5], [5, 3]] as const) {
    containment[r][c] = STONE;
  }

  return [
    { board: development, current: 1, movesLeft: 1 },
    { board: fragmentation, current: 1, movesLeft: 1 },
    { board: containment, current: 1, movesLeft: 1 }
  ];
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}
