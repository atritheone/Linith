import test from "node:test";
import assert from "node:assert/strict";
import {
  createNativeVeryHardCoreSync
} from "../native/very-hard/node-adapter";
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
  boardKey,
  generateLegalActions,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  evaluateVeryHardPosition,
  evaluateVeryHardRootPersonality,
  VERY_HARD_ROOT_PERSONALITY_LIMIT
} from "../src/renderer/game/veryHard/evaluate";

const STYLES = ["doctrinal", "constrictor", "rupture", "blizzard", "librarian", "swarm", "fortress"];

test("native zero-budget fallback takes a terminal win", () => {
  const core = createNativeVeryHardCoreSync();
  const state = immediateWinState();
  const result = core.search(state, { nodeBudget: 0, budgetMs: 0 });
  assert.ok(result.action);
  assert.equal(applyAction(state, result.action)?.outcome, "sun");
});

test("native zero-budget root scan preserves a quiet only-defense", () => {
  const core = createNativeVeryHardCoreSync();
  const state = onlyDefenseState();
  const result = core.search(state, { nodeBudget: 0, budgetMs: 0 });
  assert.ok(result.action);
  assert.equal(actionKey(result.action), "move:0,0:1,1");
});

test("no personality can override an immediate win or the unique root defense", () => {
  for (const style of STYLES) {
    const winState = immediateWinState();
    const win = createNativeVeryHardCoreSync().search(winState, { style, nodeBudget: 0, budgetMs: 0 });
    assert.ok(win.action);
    assert.equal(applyAction(winState, win.action)?.outcome, "sun", `${style} win`);

    const defense = createNativeVeryHardCoreSync().search(onlyDefenseState(), {
      style,
      nodeBudget: 0,
      budgetMs: 0
    });
    assert.ok(defense.action);
    assert.equal(actionKey(defense.action), "move:0,0:1,1", `${style} defense`);
  }
});

test("native turn depth includes both scheduled actions", () => {
  const core = createNativeVeryHardCoreSync();
  const state = twoActionWinState();
  assert.equal(core.currentTurnWinDistance(state), 2);
  assert.equal(core.currentTurnWinDistance({ ...state, movesLeft: 1 }), 0);
  const result = core.search(state, {
    maxTurnDepth: 1,
    tacticalDepth: 0,
    nodeBudget: 250_000,
    budgetMs: 0
  });
  assert.ok(result.action);
  assert.ok(["stone:0,1", "stone:1,0"].includes(actionKey(result.action)));
  assert.equal(result.completedTurnDepth, 1);
  assert.ok(result.score > 500_000_000, "the completed turn must see the forced mate");
  assert.ok(result.continuation.length >= 1, "the searched second scheduled action must be exported");
  const afterRoot = applyAction(state, result.action);
  assert.ok(afterRoot && !afterRoot.outcome && afterRoot.current === state.current);
  assert.equal(core.positionHash(afterRoot), result.continuation[0].inputHash);
  assert.ok(applyAction(afterRoot, result.continuation[0].action));
});

test("native continuation includes a searched bonus-action chain", () => {
  const core = createNativeVeryHardCoreSync();
  const state = bonusActionState();
  const result = core.search(state, {
    maxTurnDepth: 1,
    tacticalDepth: 0,
    nodeBudget: 250_000,
    budgetMs: 0
  });
  assert.ok(result.action);
  const afterRoot = applyAction(state, result.action);
  assert.ok(afterRoot);
  assert.equal(afterRoot.opponentLoss, 1, "the root action must earn a bonus by freezing an enemy Swan");
  assert.equal(afterRoot.current, state.current);
  assert.ok(result.continuation.length >= 1);
  assert.equal(result.continuation[0].inputHash, core.positionHash(afterRoot));
  assert.ok(applyAction(afterRoot, result.continuation[0].action));
});

test("native tactical horizon completes both scheduled actions before evaluating", () => {
  const core = createNativeVeryHardCoreSync();
  const twoActions = core.tacticalProbe(twoActionWinState(), 1, 1);
  const oneAction = core.tacticalProbe({ ...twoActionWinState(), movesLeft: 1 }, 1, 1);
  assert.ok(twoActions.score > 500_000_000, "the tactical horizon must see the second-action seal");
  assert.ok(oneAction.score < 500_000_000, "one remaining action cannot complete the same seal");
});

test("native tactical horizon cannot stand pat during a freeze bonus chain", () => {
  const core = createNativeVeryHardCoreSync();
  const result = core.tacticalProbe(bonusActionState(), 1, 1);
  assert.ok(result.forcedSameTurnActions > 0, "quiet legal bonus continuations must be evaluated before stand-pat");
});

test("native current-turn oracle preserves a quiet two-action seal defense", () => {
  const core = createNativeVeryHardCoreSync();
  const state = twoActionThreatDefenseState();
  assert.equal(
    core.currentTurnWinDistance({ board: state.board, current: 2, movesLeft: 2 }),
    2,
    "Moon must have a quiet setup then seal if Sun passes"
  );
  const result = core.search(state, {
    maxTurnDepth: 1,
    tacticalDepth: 0,
    exactDepth: 0,
    nodeBudget: 500_000,
    budgetMs: 0
  });
  assert.equal(result.rootThreatDetected, true);
  assert.ok(result.action);
  const defended = applyAction(state, result.action);
  assert.ok(defended && defended.outcome !== "moon");
  if (!defended.outcome) {
    assert.equal(defended.current, 2);
    assert.equal(core.currentTurnWinDistance(defended), 0, "selected move must remove the two-action seal");
  }
});

test("native bounded exact endgame reports a real proof", () => {
  const core = createNativeVeryHardCoreSync();
  const result = core.search(exactEndgameState(), {
    maxTurnDepth: 1,
    tacticalDepth: 0,
    exactDepth: 8,
    nodeBudget: 100_000,
    budgetMs: 0
  });
  assert.ok(result.action);
  assert.ok(result.exactExtensions > 0);
  assert.equal(result.exactSolved, true);
  assert.equal(result.completedTurnDepth, 1);
});

test("persistent native TT is isolated by root perspective and style", () => {
  const reused = createNativeVeryHardCoreSync();
  const fresh = createNativeVeryHardCoreSync();
  const first = developmentState(1);
  const target = developmentState(2);
  const options = {
    maxTurnDepth: 2,
    tacticalDepth: 0,
    nodeBudget: 500_000,
    budgetMs: 0,
    style: "fortress"
  } as const;

  reused.search(first, { ...options, style: "constrictor" });
  const afterOtherPerspective = reused.search(target, options);
  const withoutHistory = fresh.search(target, options);
  assert.equal(actionKey(afterOtherPerspective.action!), actionKey(withoutHistory.action!));
  assert.equal(afterOtherPerspective.score, withoutHistory.score);
  assert.equal(afterOtherPerspective.completedTurnDepth, withoutHistory.completedTurnDepth);
});

test("persistent native root history breaks an avoidable exact-position loop", () => {
  const core = createNativeVeryHardCoreSync();
  const state = developmentState(1);
  const options = { nodeBudget: 0, budgetMs: 0 } as const;
  const first = core.search(state, options);
  const uncommitted = core.search(state, options);
  assert.ok(first.action && uncommitted.action);
  assert.equal(actionKey(uncommitted.action), actionKey(first.action), "search proposals must not mutate played history");
  assert.equal(uncommitted.rootHistoryHits, 0);
  assert.equal(core.commitAction(state, first.action), true);
  const repeated = core.search(state, options);
  assert.ok(repeated.action);
  assert.notEqual(actionKey(repeated.action), actionKey(first.action));
  assert.ok(repeated.rootHistoryHits > 0);
  assert.ok(applyAction(state, repeated.action));
});

test("native selective search retains an opponent reply that closes a played-history cycle", () => {
  const core = createNativeVeryHardCoreSync();
  let state = stageTwoRepetitionOpening();
  const played = [
    "swan:2,7",
    "swan:3,5",
    "move:2,7:-1,-1",
    "swan:3,4",
    "swan:3,8",
    "swan:2,4",
    "swan:4,8",
    "swan:1,4",
    "move:4,8:1,-1",
    "move:3,4;4,5;5,5:1,-1",
    "swan:5,6",
    "stone:2,7",
    "move:5,7:1,1",
    "stone:2,6",
    "move:3,8;5,6;6,8:1,0",
    "stone:6,7",
    "move:3,7;6,6:1,-1",
    "push:7,5:-1,1",
    "move:6,6:1,-1",
    "push:4,6:-1,0",
    "move:3,6;4,8;7,8:1,0",
    "stone:6,7",
    "push:3,5:0,-1",
    "push:7,5:-1,1",
    "move:6,6:1,-1"
  ];
  for (const key of played) {
    const action = generateLegalActions(state).find((candidate) => actionKey(candidate) === key);
    assert.ok(action, `missing extracted arena action ${key}`);
    if (state.current === 1) assert.equal(core.commitAction(state, action), true);
    const next = applyAction(state, action);
    assert.ok(next && !next.outcome, `invalid extracted arena action ${key}`);
    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }

  const repeatedRootKey = boardKey(state);
  const cycleRoot = generateLegalActions(state).find((action) => actionKey(action) === "push:7,5:1,1");
  assert.ok(cycleRoot);
  const afterCycleRoot = applyAction(state, cycleRoot);
  assert.ok(afterCycleRoot && !afterCycleRoot.outcome);
  const closingReply = generateLegalActions(afterCycleRoot).find(
    (action) => actionKey(action) === "move:8,6:-1,-1"
  );
  assert.ok(closingReply);
  const closed = applyAction(afterCycleRoot, closingReply);
  assert.ok(closed && !closed.outcome);
  assert.equal(boardKey(closed), repeatedRootKey, "fixture must reproduce the opponent's exact cycle-closing reply");

  const result = core.search(state, {
    maxTurnDepth: 6,
    tacticalDepth: 1,
    exactDepth: 0,
    nodeBudget: 3_000,
    budgetMs: 0
  });
  assert.ok(result.action);
  assert.notEqual(
    actionKey(result.action),
    actionKey(cycleRoot),
    "a selectively searched opponent reply must expose and reject the imminent third occurrence"
  );
  assert.ok(result.rootHistoryHits > 0);
});

test("native selective search retains the first action of a two-action history-closing reply", () => {
  const core = createNativeVeryHardCoreSync();
  let state = stageFourTwoActionCycleOpening();
  const played = [
    "swan:7,6", "swan:4,5", "move:6,6:-1,1", "swan:5,4", "swan:4,7",
    "swan:6,4", "move:4,7:-1,1", "swan:7,4", "swan:2,8", "swan:8,4",
    "move:2,8:-1,-1", "move:4,4;7,4:0,-1", "swan:8,6", "stone:2,7",
    "move:5,7;8,6:1,1", "stone:2,8", "swan:1,6", "stone:8,6", "stone:8,7",
    "move:1,6:1,-1", "move:2,5:0,-1", "stone:8,8",
    "move:4,5;5,4;6,4:-1,1", "stone:4,4", "move:7,6;9,7:-1,-1",
    "move:7,3;8,4:-1,0", "move:3,6;4,3;4,5;5,5:1,1",
    "push:5,6;7,4:-1,0", "stone:5,3", "move:4,6;5,4;6,6:-1,1",
    "move:3,7;4,5;4,7;5,7;6,3:1,-1", "move:1,7;2,4;3,8;6,5:1,-1",
    "move:2,6;4,7:-1,1", "move:4,6;7,2:0,-1", "push:7,4:-1,1",
    "move:3,8;6,5:1,-1", "move:7,4:0,-1", "push:7,3:0,1",
    "push:7,4:-1,1", "move:6,5:1,-1", "move:7,4:0,-1"
  ];
  for (const key of played) {
    const action = generateLegalActions(state).find((candidate) => actionKey(candidate) === key);
    assert.ok(action, `missing extracted two-action-cycle action ${key}`);
    if (state.current === 1) assert.equal(core.commitAction(state, action), true);
    const next = applyAction(state, action);
    assert.ok(next && !next.outcome, `invalid extracted two-action-cycle action ${key}`);
    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }

  const cycleRoot = generateLegalActions(state).find(
    (action) => actionKey(action) === "push:4,7;7,3:0,1"
  );
  assert.ok(cycleRoot);
  const afterRoot = applyAction(state, cycleRoot);
  assert.ok(afterRoot && !afterRoot.outcome);
  const continuation = generateLegalActions(afterRoot).find(
    (action) => actionKey(action) === "push:7,4:-1,1"
  );
  assert.ok(continuation);
  const beforeOpponent = applyAction(afterRoot, continuation);
  assert.ok(beforeOpponent && !beforeOpponent.outcome && beforeOpponent.current === 2);
  const firstReply = generateLegalActions(beforeOpponent).find(
    (action) => actionKey(action) === "move:6,5:1,-1"
  );
  assert.ok(firstReply);
  const afterFirstReply = applyAction(beforeOpponent, firstReply);
  assert.ok(afterFirstReply && !afterFirstReply.outcome && afterFirstReply.current === 2);
  const closingReply = generateLegalActions(afterFirstReply).find(
    (action) => actionKey(action) === "move:4,8;7,4:0,-1"
  );
  assert.ok(closingReply);
  const closed = applyAction(afterFirstReply, closingReply);
  assert.ok(closed && !closed.outcome);
  assert.equal(boardKey(closed), boardKey(state), "the second opponent action must restore the played root");

  const result = core.search(state, {
    maxTurnDepth: 6,
    tacticalDepth: 1,
    exactDepth: 0,
    nodeBudget: 3_000,
    budgetMs: 0,
    style: "librarian"
  });
  assert.ok(result.action);
  assert.notEqual(actionKey(result.action), actionKey(cycleRoot), JSON.stringify(result.diagnostics));
  assert.ok(result.rootHistoryHits > 0);
});

test("a Hard-floor override cannot commit a phantom native proposal", () => {
  const core = createNativeVeryHardCoreSync();
  const state = developmentState(1);
  const options = { nodeBudget: 0, budgetMs: 0 } as const;
  const proposal = core.search(state, options);
  assert.ok(proposal.action);
  const floor = generateLegalActions(state).find((action) => actionKey(action) !== actionKey(proposal.action!));
  assert.ok(floor);
  assert.equal(core.commitAction(state, floor), true);

  const revisited = core.search(state, options);
  assert.ok(revisited.action);
  assert.equal(
    actionKey(revisited.action),
    actionKey(proposal.action),
    "only the actual floor action, not the discarded native proposal, belongs to history"
  );
  assert.notEqual(actionKey(revisited.action), actionKey(floor));
});

test("native evaluation exactly matches TypeScript across features and styles", () => {
  const core = createNativeVeryHardCoreSync();
  const featureBoard = emptyBoard();
  for (const [r, c] of [[2, 2], [2, 3], [5, 4], [7, 7]] as const) featureBoard[r][c] = SWAN_SUN;
  for (const [r, c] of [[1, 7], [4, 7], [6, 6], [8, 2]] as const) featureBoard[r][c] = SWAN_MOON;
  for (const [r, c] of [[1, 1], [1, 2], [1, 3], [2, 1], [3, 1], [3, 2], [3, 3], [4, 4], [5, 5], [7, 6]] as const) {
    featureBoard[r][c] = STONE;
  }
  featureBoard[9][0] = FROZEN_SUN;
  featureBoard[0][9] = FROZEN_MOON;
  const states: SearchState[] = [
    developmentState(1),
    { board: featureBoard, current: 2, movesLeft: 2 },
    onlyDefenseState()
  ];
  for (const state of states) {
    for (const perspective of [1, 2] as const) {
      for (const style of STYLES) {
        assert.equal(
          core.evaluate(state, perspective, style),
          evaluateVeryHardPosition(state, perspective, style),
          `${style} perspective ${perspective}`
        );
      }
    }
  }
});

test("native root personality diagnostics preserve the bounded objective-first contract", () => {
  const state = developmentState(1);
  for (const style of STYLES) {
    const result = createNativeVeryHardCoreSync().search(state, {
      style,
      maxTurnDepth: 2,
      tacticalDepth: 1,
      exactDepth: 0,
      nodeBudget: 10_000,
      budgetMs: 0
    });
    assert.ok(result.action);
    const applied = applyAction(state, result.action);
    assert.ok(applied);
    assert.equal(
      result.personalityBonus,
      evaluateVeryHardRootPersonality(state, result.action, applied, state.current, style),
      style
    );
    assert.equal(result.score, result.objectiveScore + result.personalityBonus, style);
    assert.ok(Math.abs(result.personalityBonus) <= VERY_HARD_ROOT_PERSONALITY_LIMIT, style);
    assert.ok(result.objectiveRegret <= VERY_HARD_ROOT_PERSONALITY_LIMIT * 2, style);
    if (style === "doctrinal") {
      assert.equal(result.personalityBonus, 0);
      assert.equal(result.objectiveRegret, 0);
    }
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

function onlyDefenseState(): SearchState {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[0][1] = STONE;
  board[1][0] = STONE;
  for (const [r, c] of [[0, 2], [2, 0], [9, 0], [9, 1], [9, 2]] as const) board[r][c] = FROZEN_SUN;
  board[5][5] = SWAN_MOON;
  return { board, current: 1, movesLeft: 1 };
}

function twoActionWinState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[1][1] = STONE;
  return { board, current: 1, movesLeft: 2 };
}

function bonusActionState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[9][9] = SWAN_MOON;
  board[1][0] = STONE;
  board[1][1] = STONE;
  return { board, current: 1, movesLeft: 1 };
}

function twoActionThreatDefenseState(): SearchState {
  const board = emptyBoard();
  board[0][0] = SWAN_SUN;
  board[9][9] = SWAN_MOON;
  board[1][1] = STONE;
  for (let c = 0; c < 5; c += 1) board[4][c] = FROZEN_SUN;
  for (let c = 5; c < 10; c += 1) board[5][c] = FROZEN_MOON;
  return { board, current: 1, movesLeft: 1 };
}

function exactEndgameState(): SearchState {
  const board = Array.from({ length: 10 }, () => Array(10).fill(STONE) as Board[number]);
  board[0][0] = SWAN_SUN;
  board[9][9] = SWAN_MOON;
  for (const [r, c] of [[0, 1], [1, 0], [8, 9], [9, 8]] as const) board[r][c] = EMPTY;
  for (let c = 0; c < 5; c += 1) board[4][c] = FROZEN_SUN;
  for (let c = 5; c < 10; c += 1) board[5][c] = FROZEN_MOON;
  return { board, current: 1, movesLeft: 1 };
}

function stageTwoRepetitionOpening(): SearchState {
  const board = emptyBoard();
  board[3][7] = SWAN_MOON;
  board[4][5] = SWAN_SUN;
  board[4][7] = STONE;
  board[5][5] = SWAN_SUN;
  for (const [r, c] of [[6, 2], [6, 4], [6, 9], [7, 0], [7, 7]] as const) board[r][c] = STONE;
  return { board, current: 2, movesLeft: 1 };
}

function stageFourTwoActionCycleOpening(): SearchState {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  return { board, current: 2, movesLeft: 1 };
}

function developmentState(current: 1 | 2): SearchState {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  board[4][5] = STONE;
  return { board, current, movesLeft: 1 };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}
