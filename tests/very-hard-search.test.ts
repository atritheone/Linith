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
  type Player
} from "../src/renderer/game/encirclement";
import {
  actionKey,
  applyAction,
  boardKey,
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  chooseVeryHardAction,
  createVeryHardSearchEngine,
  explainVeryHardPosition,
  VERY_HARD_STYLE_TIE_BREAK_LIMIT,
  searchVeryHard,
  type VeryHardSearchOptions
} from "../src/renderer/game/veryHard";

const TACTICAL_OPTIONS: VeryHardSearchOptions = {
  budgetMs: 30_000,
  nodeBudget: 20_000,
  maxDepth: 3,
  tacticalDepth: 1,
  style: "doctrinal"
};

test("Very Hard takes an immediate final encirclement", () => {
  const state = immediateWinState();
  const before = cloneBoard(state.board);
  const result = chooseVeryHardAction(state, { ...TACTICAL_OPTIONS, maxDepth: 2 });

  assert.ok(result.action, "a legal winning action should be returned");
  const next = applyAction(state, result.action);
  assert.ok(next);
  assert.equal(next.outcome, "sun");
  assert.deepEqual(state.board, before, "search must not mutate the caller's board");
});

test("Very Hard finds the unique defense against mate in one", () => {
  const state = forcedDefenseState();
  const safeActions = actionsAvoidingImmediateLoss(state);
  assert.deepEqual(safeActions.map(actionKey), ["move:0,0:1,1"], "fixture must have one defense");

  const result = chooseVeryHardAction(state, { ...TACTICAL_OPTIONS, maxDepth: 1 });
  assert.ok(result.action);
  assert.equal(result.completedDepth, 1, "one completed turn plus forcing search must verify the reply");
  assert.equal(actionKey(result.action), actionKey(safeActions[0]));

  const defended = applyAction(state, result.action);
  assert.ok(defended && !defended.outcome);
  assert.equal(immediateWins(asSearchState(defended), 2).length, 0);
});

test("root classification preserves a quiet only-defense at zero budget", () => {
  const state = forcedDefenseState();
  const safeActions = actionsAvoidingImmediateLoss(state);
  assert.deepEqual(safeActions.map(actionKey), ["move:0,0:1,1"]);

  const result = chooseVeryHardAction(state, {
    budgetMs: 0,
    nodeBudget: 0,
    maxDepth: 8,
    tacticalDepth: 0
  });
  assert.ok(result.action);
  assert.equal(actionKey(result.action), actionKey(safeActions[0]));
  assert.equal(result.forcedMove, true);
  assert.equal(result.nodes, 0);
});

test("Very Hard treats simultaneous final encirclement as a forced draw", () => {
  const state = forcedDrawState();
  const legal = generateLegalActions(state);
  assert.deepEqual(legal.map(actionKey), ["stone:1,1"], "fixture must force the drawing action");

  const result = chooseVeryHardAction(state, { ...TACTICAL_OPTIONS, maxDepth: 2 });
  assert.ok(result.action);
  const next = applyAction(state, result.action);
  assert.ok(next);
  assert.equal(next.outcome, "draw");
});

test("Very Hard searches both actions in a two-action turn", () => {
  const state = twoActionWinState();
  const result = chooseVeryHardAction(state, { ...TACTICAL_OPTIONS, maxDepth: 1 });
  assert.ok(result.action);
  assert.equal(result.completedDepth, 1, "one depth unit must cover the complete two-action turn");
  assert.ok(
    ["stone:0,1", "stone:1,0"].includes(actionKey(result.action)),
    `expected the first half of the forced two-action seal, received ${actionKey(result.action)}`
  );

  const afterFirst = applyAction(state, result.action);
  assert.ok(afterFirst && !afterFirst.outcome);
  assert.equal(afterFirst.current, 1, "the same player must retain the second scheduled action");
  assert.equal(afterFirst.movesLeft, 1);

  assert.ok(result.principalVariation.length >= 2, "the first search must retain the same-turn continuation");
  const pvFinish = applyAction(asSearchState(afterFirst), result.principalVariation[1]);
  assert.ok(pvFinish);
  assert.equal(pvFinish.outcome, "sun");

  const continuation = chooseVeryHardAction(asSearchState(afterFirst), {
    ...TACTICAL_OPTIONS,
    maxDepth: 2
  });
  assert.ok(continuation.action);
  const finish = applyAction(asSearchState(afterFirst), continuation.action);
  assert.ok(finish);
  assert.equal(finish.outcome, "sun");
});

test("Very Hard follows a freeze bonus into a same-player winning continuation", () => {
  const state = bonusChainWinState();
  const result = chooseVeryHardAction(state, { ...TACTICAL_OPTIONS, maxDepth: 1 });
  assert.ok(result.action);
  assert.equal(result.completedDepth, 1, "a bonus chain must remain inside its completed turn");
  assert.ok(
    ["stone:1,1", "stone:8,8"].includes(actionKey(result.action)),
    `expected a bonus-earning seal, received ${actionKey(result.action)}`
  );

  const afterFreeze = applyAction(state, result.action);
  assert.ok(afterFreeze && !afterFreeze.outcome);
  assert.equal(afterFreeze.opponentLoss, 1);
  assert.equal(afterFreeze.current, 1, "the freeze bonus must keep control with Sun");
  assert.equal(afterFreeze.movesLeft, 1);

  assert.ok(result.principalVariation.length >= 2, "the first search must retain the bonus continuation");
  const pvFinish = applyAction(asSearchState(afterFreeze), result.principalVariation[1]);
  assert.ok(pvFinish);
  assert.equal(pvFinish.outcome, "sun");

  const continuation = chooseVeryHardAction(asSearchState(afterFreeze), {
    ...TACTICAL_OPTIONS,
    maxDepth: 2
  });
  assert.ok(continuation.action);
  const finish = applyAction(asSearchState(afterFreeze), continuation.action);
  assert.ok(finish);
  assert.equal(finish.outcome, "sun");
});

test("a fixed node budget produces deterministic fallback and diagnostics", () => {
  const state = developmentState();
  const options: VeryHardSearchOptions = {
    budgetMs: 30_000,
    nodeBudget: 24,
    maxDepth: 8,
    tacticalDepth: 1,
    style: "doctrinal"
  };
  const first = chooseVeryHardAction(state, options);
  const second = chooseVeryHardAction(state, options);

  assert.ok(first.action && second.action);
  assert.equal(actionKey(first.action), actionKey(second.action));
  assert.equal(first.score, second.score);
  assert.equal(first.completedDepth, second.completedDepth);
  assert.equal(first.nodes, second.nodes);
  assert.deepEqual(first.principalVariation.map(actionKey), second.principalVariation.map(actionKey));
  assert.equal(first.stopReason, "node-budget");
  assert.ok(first.nodes <= options.nodeBudget!);
  assert.ok(applyAction(state, first.action));
});

test("an expired deterministic deadline still returns a legal fallback", () => {
  const state = developmentState();
  let tick = 0;
  const options: VeryHardSearchOptions = {
    budgetMs: 2,
    nodeBudget: 1_000_000,
    maxDepth: 8,
    now: () => tick++
  };
  const result = chooseVeryHardAction(state, options);
  tick = 0;
  const repeat = chooseVeryHardAction(state, options);

  assert.ok(result.action);
  assert.ok(applyAction(state, result.action));
  assert.ok(repeat.action);
  assert.equal(actionKey(result.action), actionKey(repeat.action));
  assert.equal(result.nodes, repeat.nodes);
  assert.equal(result.completedDepth, repeat.completedDepth);
  assert.equal(result.stopReason, "deadline");
  assert.equal(result.timedOut, true);
});

test("the request API echoes identity and returns a legal deterministic zero-budget fallback", () => {
  const state = developmentState();
  const before = cloneBoard(state.board);
  const positionHash = boardKey(state);
  const result = searchVeryHard({
    requestId: 73,
    positionHash,
    ...state,
    style: "fortress",
    budgetMs: 0,
    nodeBudget: 0,
    maxDepth: 8
  });

  assert.equal(result.requestId, 73);
  assert.equal(result.positionHash, positionHash);
  assert.ok(result.action);
  assert.ok(applyAction(state, result.action));
  assert.equal(result.completedDepth, 0);
  assert.equal(result.timedOut, true);
  assert.deepEqual(state.board, before);
});

test("root tactical classification takes a forced win without spending a search node", () => {
  const state = immediateWinState();
  const result = chooseVeryHardAction(state, {
    budgetMs: 0,
    nodeBudget: 0,
    maxDepth: 8
  });

  assert.ok(result.action);
  assert.equal(applyAction(state, result.action)?.outcome, "sun");
  assert.equal(result.forcedMove, true);
  assert.equal(result.nodes, 0);
  assert.equal(result.completedDepth, 1);
});

test("root widening retains a best quiet Stone outside the geometric shortlist", () => {
  const board: Board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
  ];
  const state: SearchState = { board, current: 1, movesLeft: 1 };
  const result = chooseVeryHardAction(state, {
    budgetMs: Infinity,
    nodeBudget: 5_000,
    maxDepth: 1,
    tacticalDepth: 0
  });

  assert.equal(generateLegalActions(state).length, 139, "fixture must exceed the selective root width");
  assert.ok(result.action);
  assert.equal(actionKey(result.action), "stone:9,8");
  const applied = applyAction(state, result.action);
  assert.ok(applied && !applied.outcome && applied.opponentLoss === 0, "the retained move must be quiet");
  assert.ok(result.rootActionsSearched < result.rootActions, "the fixture must exercise selective widening");
});

test("a reusable engine carries safe transposition knowledge across requests", () => {
  const state = developmentState();
  const engine = createVeryHardSearchEngine();
  const options: VeryHardSearchOptions = {
    budgetMs: Infinity,
    nodeBudget: 20_000,
    maxDepth: 1,
    tacticalDepth: 0
  };
  const first = engine.search(state, options);
  const retained = engine.transpositionSize;
  const second = engine.search(state, options);

  assert.ok(first.action && second.action);
  assert.equal(actionKey(second.action), actionKey(first.action));
  assert.ok(retained > 0);
  assert.ok(second.transpositionHits > 0);
  engine.clear();
  assert.equal(engine.transpositionSize, 0);
});

test("a reusable engine avoids returning to a previously searched game state", () => {
  const state = developmentState();
  const engine = createVeryHardSearchEngine();
  const options: VeryHardSearchOptions = {
    budgetMs: Infinity,
    nodeBudget: 10_000,
    maxDepth: 1,
    tacticalDepth: 0
  };
  const first = engine.search(state, options);
  assert.ok(first.action);
  const visited = applyAction(state, first.action);
  assert.ok(visited && !visited.outcome);
  engine.search(asSearchState(visited), { ...options, budgetMs: 0, nodeBudget: 0 });

  const reconsidered = engine.search(state, options);
  assert.ok(reconsidered.action);
  assert.notEqual(
    actionKey(reconsidered.action),
    actionKey(first.action),
    "an equivalent alternative should be preferred to a known cycle"
  );
});

test("personalities are bounded evaluator tie-breaks with a zero-bias Doctrinal baseline", () => {
  const state = developmentState();
  const doctrinal = explainVeryHardPosition(state, 1, "doctrinal");
  assert.equal(doctrinal.style, 0);
  for (const style of ["constrictor", "rupture", "blizzard", "librarian", "swarm", "fortress"]) {
    const styled = explainVeryHardPosition(state, 1, style);
    assert.equal(styled.total - doctrinal.total, styled.style - doctrinal.style);
    assert.ok(Math.abs(styled.total - doctrinal.total) <= VERY_HARD_STYLE_TIE_BREAK_LIMIT);
  }
});

test("small late positions receive selective exact extensions", () => {
  const board = Array.from({ length: 10 }, () => Array(10).fill(STONE) as Board[number]);
  board[0][0] = SWAN_SUN;
  board[9][9] = SWAN_MOON;
  for (const [r, c] of [[0, 1], [1, 0], [8, 9], [9, 8]] as Array<[number, number]>) {
    board[r][c] = EMPTY;
  }
  for (let c = 0; c < 5; c += 1) board[4][c] = FROZEN_SUN;
  for (let c = 5; c < 10; c += 1) board[5][c] = FROZEN_MOON;
  const state: SearchState = { board, current: 1, movesLeft: 1 };
  assert.ok(generateLegalActions(state).length <= 12, "fixture must enter the selective exact solver");

  const result = chooseVeryHardAction(state, {
    budgetMs: Infinity,
    nodeBudget: 100_000,
    maxDepth: 1,
    tacticalDepth: 0,
    exactDepth: 8
  });
  assert.ok(result.action);
  assert.ok(result.exactExtensions > 0);
  assert.equal(result.exactSolved, true, "the bounded late fixture should be fully resolved");
  assert.ok(applyAction(state, result.action));
});

test("Very Hard returns legal actions without mutating a reachable-state corpus", () => {
  const random = seededRandom(0x0b5e55ed);
  let state = developmentState();

  for (let sample = 0; sample < 12; sample += 1) {
    const before = cloneBoard(state.board);
    const result = chooseVeryHardAction(state, {
      budgetMs: 30_000,
      nodeBudget: 96,
      maxDepth: 6,
      style: sample % 2 === 0 ? "doctrinal" : "constrictor"
    });
    assert.ok(result.action, `missing action for reachable sample ${sample}`);
    assert.deepEqual(state.board, before, `search mutated reachable sample ${sample}`);

    const selected = applyAction(state, result.action);
    assert.ok(selected, `illegal selected action for reachable sample ${sample}`);
    assertPrincipalVariationIsLegal(state, result.principalVariation);

    const actions = generateLegalActions(state);
    const randomAction = actions[Math.floor(random() * actions.length)];
    const next = applyAction(state, randomAction);
    assert.ok(next);
    state = next.outcome ? developmentState() : asSearchState(next);
  }
});

function immediateWinState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[0][1] = STONE;
  board[1][0] = STONE;
  return { board, current: 1, movesLeft: 1 };
}

function forcedDefenseState(): SearchState {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[0][1] = STONE;
  board[1][0] = STONE;
  // Frozen friendly Swans anchor both Stones, and five frozen Swans enforce
  // the six-Swan cap. Moving diagonally is therefore the only escape.
  for (const [r, c] of [[0, 2], [2, 0], [9, 0], [9, 1], [9, 2]] as Array<[number, number]>) {
    board[r][c] = FROZEN_SUN;
  }
  board[5][5] = SWAN_MOON;
  return { board, current: 1, movesLeft: 1 };
}

function forcedDrawState(): SearchState {
  const board = Array.from({ length: 10 }, () => Array(10).fill(STONE) as Board[number]);
  board[0][0] = SWAN_SUN;
  board[0][2] = SWAN_MOON;
  board[1][1] = EMPTY;
  return { board, current: 1, movesLeft: 1 };
}

function twoActionWinState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[1][1] = STONE;
  return { board, current: 1, movesLeft: 2 };
}

function bonusChainWinState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[9][9] = SWAN_MOON;
  board[0][1] = STONE;
  board[1][0] = STONE;
  board[8][9] = STONE;
  board[9][8] = STONE;
  return { board, current: 1, movesLeft: 1 };
}

function developmentState(): SearchState {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  board[4][5] = STONE;
  return { board, current: 1, movesLeft: 1 };
}

function actionsAvoidingImmediateLoss(state: SearchState): LinithAction[] {
  const opponent = other(state.current);
  return generateLegalActions(state).filter((action) => {
    const next = applyAction(state, action);
    if (!next || next.outcome) return false;
    return immediateWins(asSearchState(next), opponent).length === 0;
  });
}

function immediateWins(state: SearchState, player: Player): LinithAction[] {
  const expected = player === 1 ? "sun" : "moon";
  return generateLegalActions(state).filter((action) => applyAction(state, action)?.outcome === expected);
}

function assertPrincipalVariationIsLegal(state: SearchState, variation: readonly LinithAction[]): void {
  let cursor = state;
  for (const action of variation) {
    const next = applyAction(cursor, action);
    assert.ok(next, `illegal principal-variation action ${actionKey(action)}`);
    if (next.outcome) return;
    cursor = asSearchState(next);
  }
}

function asSearchState(state: Pick<SearchState, "board" | "current" | "movesLeft">): SearchState {
  return { board: state.board, current: state.current, movesLeft: state.movesLeft };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

function other(player: Player): Player {
  return player === 1 ? 2 : 1;
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}
