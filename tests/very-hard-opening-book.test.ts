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
  actionKey,
  applyAction,
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  D4_SYMMETRIES,
  OPENING_BOOK_SCHEMA_VERSION,
  canonicalizeOpeningPosition,
  inverseSymmetry,
  lookupOpeningBookAction,
  swapStateColors,
  transformAction,
  transformBoard,
  transformDirection,
  transformState,
  type OpeningBookEntry
} from "../src/renderer/game/veryHard/openingBook";
import {
  DEFAULT_OPENING_BOOK_SEARCH_PLANS,
  generateVerifiedOpeningBook,
  serializeGeneratedOpeningBook,
  serializeShippingOpeningBookData,
  verifyOpeningBookCandidate,
  type OpeningBookSearchRunner
} from "../src/renderer/game/veryHard/openingBookBuilder";

const ACTIONS: readonly LinithAction[] = [
  { type: "stone", r: 2, c: 7 },
  { type: "swan", r: 4, c: 5 },
  { type: "move", swans: [{ r: 3, c: 4 }, { r: 3, c: 5 }], dir: [-1, 1] },
  { type: "push", swans: [{ r: 7, c: 2 }, { r: 8, c: 2 }], dir: [0, -1] }
];

test("all D4 action and direction transforms round-trip exactly", () => {
  for (const symmetry of D4_SYMMETRIES) {
    const inverse = inverseSymmetry(symmetry);
    for (const action of ACTIONS) {
      const transformed = transformAction(action, symmetry);
      const roundTrip = transformAction(transformed, inverse);
      assert.equal(actionKey(roundTrip), actionKey(action), `${symmetry} failed for ${action.type}`);
    }
    for (const direction of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]] as const) {
      assert.deepEqual(
        transformDirection(transformDirection(direction, symmetry), inverse),
        [...direction],
        `${symmetry} failed for direction ${direction.join(",")}`
      );
    }
  }
});

test("all D4 board transforms round-trip without changing tiles", () => {
  const state = asymmetricState();
  for (const symmetry of D4_SYMMETRIES) {
    const transformed = transformBoard(state.board, symmetry);
    assert.deepEqual(transformBoard(transformed, inverseSymmetry(symmetry)), state.board, symmetry);
  }
});

test("D4 action transforms are exactly equivariant with the live executor", () => {
  const state = asymmetricState();
  const actions = generateLegalActions(state);
  assert.ok(actions.length > 0);
  for (const action of actions) {
    const applied = applyAction(state, action);
    assert.ok(applied);
    for (const symmetry of D4_SYMMETRIES) {
      const transformed = applyAction(
        transformState(state, symmetry),
        transformAction(action, symmetry)
      );
      assert.ok(transformed, `${symmetry} made ${actionKey(action)} illegal`);
      assert.deepEqual(transformed.board, transformBoard(applied.board, symmetry));
      assert.equal(transformed.current, applied.current);
      assert.equal(transformed.movesLeft, applied.movesLeft);
      assert.equal(transformed.outcome, applied.outcome);
    }
  }
});

test("canonical keys normalize D4 geometry and the side-to-move color", () => {
  const state = asymmetricState();
  const expected = canonicalizeOpeningPosition(state).key;
  for (const symmetry of D4_SYMMETRIES) {
    const transformed = transformState(state, symmetry);
    assert.equal(canonicalizeOpeningPosition(transformed).key, expected, symmetry);
    assert.equal(canonicalizeOpeningPosition(swapStateColors(transformed)).key, expected, `${symmetry} color swap`);
  }
});

test("book lookup returns the correctly untransformed legal action", () => {
  const state = asymmetricState();
  const canonical = canonicalizeOpeningPosition(state);
  const originalAction: LinithAction = { type: "stone", r: 4, c: 4 };
  assert.ok(applyAction(state, originalAction));
  const canonicalAction = transformAction(originalAction, canonical.toCanonical);
  const entry = entryFor(canonical.key, canonicalAction);

  for (const symmetry of D4_SYMMETRIES) {
    const query = transformState(state, symmetry);
    const hit = lookupOpeningBookAction(query, { entries: [entry] });
    assert.ok(hit, `missing ${symmetry} hit`);
    assert.equal(
      actionKey(hit.action),
      actionKey(transformAction(originalAction, symmetry)),
      `wrong ${symmetry} action`
    );
    assert.ok(applyAction(query, hit.action));
  }

  const colorHit = lookupOpeningBookAction(swapStateColors(state), { entries: [entry] });
  assert.ok(colorHit);
  assert.equal(actionKey(colorHit.action), actionKey(originalAction));
});

test("book lookup rejects stale, illegal, malformed, and low-confidence records", () => {
  const state = asymmetricState();
  const canonical = canonicalizeOpeningPosition(state);
  const legalCanonical = transformAction({ type: "stone", r: 4, c: 4 }, canonical.toCanonical);
  const good = entryFor(canonical.key, legalCanonical);

  assert.equal(lookupOpeningBookAction(state, { entries: [{ ...good, key: `${good.key}:stale` }] }), null);

  let occupied: { r: number; c: number } | null = null;
  for (let r = 0; r < canonical.state.board.length && !occupied; r += 1) {
    for (let c = 0; c < canonical.state.board.length; c += 1) {
      if (canonical.state.board[r][c] !== EMPTY) {
        occupied = { r, c };
        break;
      }
    }
  }
  assert.ok(occupied);
  assert.equal(lookupOpeningBookAction(state, {
    entries: [entryFor(canonical.key, { type: "stone", ...occupied })]
  }), null);
  assert.equal(lookupOpeningBookAction(state, {
    entries: [{ ...good, confidence: 1.01 }]
  }), null);
  assert.equal(lookupOpeningBookAction(state, {
    entries: [{ ...good, action: { type: "move" } as LinithAction }]
  }), null);
  assert.equal(lookupOpeningBookAction(state, {
    entries: [{ ...good, confidence: 0.5 }]
  }), null);
  assert.ok(lookupOpeningBookAction(state, { entries: [{ ...good, confidence: 0.5 }], minConfidence: 0.5 }));
});

test("generator is reproducible and deduplicates color/symmetry-equivalent positions", () => {
  const state = asymmetricState();
  const canonical = canonicalizeOpeningPosition(state);
  const canonicalAction = firstEmptyStone(canonical.state);
  const search: OpeningBookSearchRunner = (_runState, plan) => ({
    action: transformAction(canonicalAction, plan.symmetry),
    score: 1200 + plan.maxDepth,
    completedDepth: plan.maxDepth,
    nodes: plan.nodeBudget,
    stopped: false
  });
  const equivalent = swapStateColors(transformState(state, "rotate270"));
  const generationOptions = { artifactFingerprint: "sha256-final-test-artifact" };
  const first = generateVerifiedOpeningBook([state, equivalent], search, generationOptions);
  const second = generateVerifiedOpeningBook([equivalent, state], search, generationOptions);

  assert.equal(first.positionsConsidered, 1);
  assert.equal(first.entries.length, 1);
  assert.equal(first.rejections.length, 0);
  assert.equal(first.entries[0].verification.agreements, DEFAULT_OPENING_BOOK_SEARCH_PLANS.length);
  assert.equal(actionKey(first.entries[0].action), actionKey(canonicalAction));
  assert.equal(serializeGeneratedOpeningBook(first), serializeGeneratedOpeningBook(second));
  assert.equal(first.artifactFingerprint, "sha256-final-test-artifact");

  const shippingModule = serializeShippingOpeningBookData(first);
  assert.match(shippingModule, /GENERATED_VERY_HARD_OPENING_BOOK/);
  assert.match(shippingModule, /sha256-final-test-artifact/);
  assert.match(shippingModule, new RegExp(`"key": "${first.entries[0].key}"`));
  assert.doesNotMatch(shippingModule, /Object\.freeze\(\[\]\)/);
  assert.throws(
    () => serializeShippingOpeningBookData({ ...first, artifactFingerprint: null }),
    /fingerprint/
  );
});

test("generator rejects action disagreement and a tactical-verifier disagreement", () => {
  const canonical = canonicalizeOpeningPosition(asymmetricState());
  const selected = firstEmptyStone(canonical.state);
  const alternative = secondEmptyStone(canonical.state);
  const disagreement: OpeningBookSearchRunner = (_state, plan) => ({
    action: transformAction(plan.name === "corroborator" ? alternative : selected, plan.symmetry),
    score: 0,
    completedDepth: 8,
    nodes: plan.nodeBudget,
    stopped: false
  });
  assert.equal(
    verifyOpeningBookCandidate(canonical.state, disagreement).reason,
    "search-disagreement"
  );

  const tacticalFailure: OpeningBookSearchRunner = (_state, plan) => ({
    action: transformAction(plan.name === "tactical" ? alternative : selected, plan.symmetry),
    score: 0,
    completedDepth: 8,
    nodes: plan.nodeBudget,
    stopped: false
  });
  assert.equal(
    verifyOpeningBookCandidate(canonical.state, tacticalFailure).reason,
    "tactical-verification-failed"
  );
});

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function asymmetricState(): SearchState {
  const board = emptyBoard();
  board[0][1] = SWAN_SUN;
  board[2][8] = SWAN_MOON;
  board[1][1] = STONE;
  board[6][3] = FROZEN_SUN;
  board[8][9] = FROZEN_MOON;
  return { board, current: 1, movesLeft: 2 };
}

function entryFor(key: string, action: LinithAction): OpeningBookEntry {
  return {
    schema: OPENING_BOOK_SCHEMA_VERSION,
    key,
    action,
    score: 125,
    confidence: 0.97,
    verification: {
      agreements: 3,
      independentRuns: 3,
      minCompletedDepth: 5,
      tacticalDepth: 4,
      minNodeBudget: 1_000_000
    },
    generator: "test-generator"
  };
}

function firstEmptyStone(state: SearchState): LinithAction {
  for (let r = 0; r < state.board.length; r += 1) {
    for (let c = 0; c < state.board.length; c += 1) {
      if (state.board[r][c] === EMPTY) return { type: "stone", r, c };
    }
  }
  throw new Error("fixture has no empty cell");
}

function secondEmptyStone(state: SearchState): LinithAction {
  let seen = false;
  for (let r = 0; r < state.board.length; r += 1) {
    for (let c = 0; c < state.board.length; c += 1) {
      if (state.board[r][c] !== EMPTY) continue;
      if (seen) return { type: "stone", r, c };
      seen = true;
    }
  }
  throw new Error("fixture has fewer than two empty cells");
}
