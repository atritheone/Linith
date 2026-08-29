import test from "node:test";
import assert from "node:assert/strict";
import { createNativeVeryHardCoreSync } from "../native/very-hard/node-adapter";
import {
  EMPTY,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board
} from "../src/renderer/game/encirclement";
import {
  actionKey,
  applyAction,
  boardKey,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import type { NativeVeryHardResult } from "../src/renderer/game/veryHard/native/core";
import { createVeryHardSearchEngine } from "../src/renderer/game/veryHard/search";
import type {
  VeryHardSearchEngine,
  VeryHardWorkerInboundMessage,
  VeryHardWorkerOutboundMessage,
  VeryHardWorkerSearchMessage
} from "../src/renderer/game/veryHard/types";
import { createVeryHardWorkerRuntime } from "../src/renderer/game/veryHard/workerRuntime";

class FakeNativeCore {
  readonly commits: Array<{ state: SearchState; action: LinithAction }> = [];
  searchCalls = 0;
  clearCalls = 0;

  constructor(
    private readonly resultFor: (state: SearchState, call: number) => NativeVeryHardResult,
    private readonly hashFor: (state: SearchState) => bigint = stableHash
  ) {}

  search(state: SearchState): NativeVeryHardResult {
    this.searchCalls += 1;
    return this.resultFor(state, this.searchCalls);
  }

  commitAction(state: SearchState, action: LinithAction): boolean {
    if (!applyAction(state, action)) return false;
    this.commits.push({ state, action });
    return true;
  }

  positionHash(state: SearchState): bigint {
    return this.hashFor(state);
  }

  clearCache(): void {
    this.clearCalls += 1;
  }
}

function fakeEngine(): VeryHardSearchEngine {
  return {
    search: () => { throw new Error("unexpected TypeScript search"); },
    searchRequest: () => { throw new Error("unexpected TypeScript search"); },
    clear: () => undefined,
    transpositionSize: 0
  };
}

function makeHarness(core: FakeNativeCore) {
  const output: VeryHardWorkerOutboundMessage[] = [];
  const runtime = createVeryHardWorkerRuntime({
    engine: fakeEngine(),
    nativeCore: Promise.resolve(core),
    lookupBook: () => null,
    postMessage: (message) => output.push(message),
    now: () => 0
  });
  return { output, runtime };
}

function request(state: SearchState, requestId = 1, sessionId = 7): VeryHardWorkerSearchMessage {
  return {
    type: "search",
    requestId,
    sessionId,
    fingerprint: boardKey(state),
    state,
    style: "doctrinal",
    budgetMs: 100,
    nodeBudget: 100_000
  };
}

function commitFor(
  search: VeryHardWorkerSearchMessage,
  action: LinithAction,
  overrides: Partial<VeryHardWorkerInboundMessage> = {}
): VeryHardWorkerInboundMessage {
  return {
    type: "commit",
    requestId: search.requestId,
    sessionId: search.sessionId,
    fingerprint: search.fingerprint,
    preActionState: search.state,
    action,
    ...overrides
  } as VeryHardWorkerInboundMessage;
}

test("scheduled two-action turns reuse an exact native continuation and commit both accepted actions", async () => {
  const state = twoActionWinState();
  const root: LinithAction = { type: "stone", r: 0, c: 1 };
  const follow: LinithAction = { type: "stone", r: 1, c: 0 };
  const afterRoot = ongoingState(state, root);
  const core = new FakeNativeCore(() => nativeResult(root, 1, stableHash(afterRoot), [{
    action: follow,
    inputHash: stableHash(afterRoot),
    postActionHash: 999n
  }]));
  const { output, runtime } = makeHarness(core);

  const first = request(state, 1);
  await runtime.handleMessage(first);
  assert.equal(output.at(-1)?.type, "result");
  await runtime.handleMessage(commitFor(first, root));

  const second = request(afterRoot, 2);
  await runtime.handleMessage(second);
  const continued = output.at(-1);
  assert.equal(continued?.type, "result");
  assert.equal(continued?.type === "result" ? continued.engine : null, "native-continuation");
  assert.equal(continued?.type === "result" ? actionKey(continued.action!) : null, actionKey(follow));
  assert.equal(core.searchCalls, 1, "the scheduled second action should not launch another search");
  await runtime.handleMessage(commitFor(second, follow));
  assert.deepEqual(core.commits.map(({ action }) => actionKey(action)), [actionKey(root), actionKey(follow)]);
});

test("the compiled native core feeds its searched continuation through the shipping runtime", async () => {
  const state = twoActionWinState();
  const core = createNativeVeryHardCoreSync();
  const output: VeryHardWorkerOutboundMessage[] = [];
  const runtime = createVeryHardWorkerRuntime({
    engine: fakeEngine(),
    nativeCore: Promise.resolve(core),
    lookupBook: () => null,
    postMessage: (message) => output.push(message),
    now: () => 0
  });
  const first = { ...request(state, 5), budgetMs: 100, nodeBudget: 250_000 };

  await runtime.handleMessage(first);
  const rootResult = output.at(-1);
  assert.equal(rootResult?.type, "result");
  assert.ok(rootResult?.type === "result" && rootResult.action);
  const afterRoot = ongoingState(state, rootResult.action);
  assert.equal(afterRoot.current, state.current);
  await runtime.handleMessage(commitFor(first, rootResult.action));

  await runtime.handleMessage(request(afterRoot, 6));
  const continued = output.at(-1);
  assert.equal(continued?.type === "result" ? continued.engine : null, "native-continuation");
  assert.ok(continued?.type === "result" && continued.action);
  assert.ok(applyAction(afterRoot, continued.action));
});

test("a freeze-bonus action chain uses the validated continuation immediately", async () => {
  const state = bonusChainWinState();
  const root: LinithAction = { type: "stone", r: 1, c: 1 };
  const follow: LinithAction = { type: "stone", r: 8, c: 8 };
  const firstApplied = applyAction(state, root);
  assert.ok(firstApplied && !firstApplied.outcome);
  assert.equal(firstApplied.opponentLoss, 1);
  assert.equal(firstApplied.current, state.current);
  const afterRoot = ongoingState(state, root);
  const core = new FakeNativeCore(() => nativeResult(root, 1, stableHash(afterRoot), [{
    action: follow,
    inputHash: stableHash(afterRoot),
    postActionHash: 777n
  }]));
  const { output, runtime } = makeHarness(core);

  const first = request(state, 10);
  await runtime.handleMessage(first);
  await runtime.handleMessage(commitFor(first, root));
  const second = request(afterRoot, 11);
  await runtime.handleMessage(second);

  const result = output.at(-1);
  assert.equal(result?.type === "result" ? result.engine : null, "native-continuation");
  assert.equal(result?.type === "result" ? actionKey(result.action!) : null, actionKey(follow));
  assert.equal(core.searchCalls, 1);
});

test("continuations require both the exact state and native hash", async () => {
  const state = twoActionWinState();
  const root: LinithAction = { type: "stone", r: 0, c: 1 };
  const follow: LinithAction = { type: "stone", r: 1, c: 0 };
  const afterRoot = ongoingState(state, root);
  const core = new FakeNativeCore(
    (_state, call) => call === 1
      ? nativeResult(root, 1, 1n, [{ action: follow, inputHash: 1n, postActionHash: 1n }])
      : nativeResult(null, 0, null, []),
    () => 1n
  );
  const { output, runtime } = makeHarness(core);
  const first = request(state, 20);
  await runtime.handleMessage(first);
  await runtime.handleMessage(commitFor(first, root));

  const mismatched = cloneState(afterRoot);
  mismatched.board[9][9] = STONE;
  await runtime.handleMessage(request(mismatched, 21));
  assert.equal(core.searchCalls, 2, "a hash collision must not bypass the full-state check");
  const mismatchedResult = output.at(-1);
  assert.equal(mismatchedResult?.type, "result");
  assert.equal(mismatchedResult?.type === "result" ? mismatchedResult.action : undefined, null);
});

test("an executor fallback commits the actual action and invalidates the proposal continuation", async () => {
  const state = twoActionWinState();
  const proposal: LinithAction = { type: "stone", r: 0, c: 1 };
  const continuation: LinithAction = { type: "stone", r: 1, c: 0 };
  const actualFallback: LinithAction = { type: "stone", r: 9, c: 9 };
  const afterProposal = ongoingState(state, proposal);
  const core = new FakeNativeCore((_state, call) => call === 1
    ? nativeResult(proposal, 1, stableHash(afterProposal), [{
        action: continuation,
        inputHash: stableHash(afterProposal),
        postActionHash: 123n
      }])
    : nativeResult(null, 0, null, []));
  const { runtime } = makeHarness(core);
  const first = request(state, 25);

  await runtime.handleMessage(first);
  await runtime.handleMessage(commitFor(first, actualFallback));
  assert.deepEqual(core.commits.map(({ action }) => actionKey(action)), [actionKey(actualFallback)]);
  assert.notEqual(actionKey(core.commits[0].action), actionKey(proposal));

  await runtime.handleMessage(request(afterProposal, 26));
  assert.equal(core.searchCalls, 2, "discarded-proposal PV must not survive the actual fallback commit");
});

test("a legal depth-zero native proposal is returned directly and committed", async () => {
  const state = developmentState();
  const proposal: LinithAction = { type: "stone", r: 0, c: 0 };
  const core = new FakeNativeCore(() => nativeResult(proposal, 0, stableHash(ongoingState(state, proposal)), []));
  const { output, runtime } = makeHarness(core);
  const search = request(state, 30);

  await runtime.handleMessage(search);
  const result = output.at(-1);
  assert.equal(result?.type, "result");
  assert.equal(result?.type === "result" ? result.engine : null, "native");
  assert.equal(result?.type === "result" ? result.completedDepth : null, 0);
  assert.equal(result?.type === "result" ? actionKey(result.action!) : null, actionKey(proposal));
  assert.equal(core.commits.length, 0, "a search proposal must never enter played history");
  await runtime.handleMessage(commitFor(search, proposal));
  assert.deepEqual(core.commits.map(({ action }) => actionKey(action)), [actionKey(proposal)]);
});

test("quiet Doctrinal book moves do not suppress a named personality", async () => {
  const state = developmentState();
  const bookAction: LinithAction = { type: "swan", r: 3, c: 4 };
  const personalityAction: LinithAction = { type: "stone", r: 0, c: 0 };
  assert.ok(applyAction(state, bookAction));
  const core = new FakeNativeCore(() => nativeResult(
    personalityAction,
    1,
    stableHash(ongoingState(state, personalityAction)),
    []
  ));
  const output: VeryHardWorkerOutboundMessage[] = [];
  const runtime = createVeryHardWorkerRuntime({
    engine: fakeEngine(),
    nativeCore: Promise.resolve(core),
    lookupBook: () => ({ action: bookAction }),
    postMessage: (message) => output.push(message),
    now: () => 0
  });

  await runtime.handleMessage({ ...request(state, 31), style: "swarm" });
  const result = output.at(-1);
  assert.equal(result?.type === "result" ? result.engine : null, "native");
  assert.equal(result?.type === "result" ? actionKey(result.action!) : null, actionKey(personalityAction));
  assert.equal(core.searchCalls, 1);
});

test("a legal depth-zero TypeScript fallback is returned without consulting Hard", async () => {
  const state = developmentState();
  const output: VeryHardWorkerOutboundMessage[] = [];
  const runtime = createVeryHardWorkerRuntime({
    engine: createVeryHardSearchEngine(),
    nativeCore: Promise.resolve(null),
    lookupBook: () => null,
    postMessage: (message) => output.push(message),
    now: () => 0
  });
  const search = { ...request(state, 35), budgetMs: 0, nodeBudget: 0 };

  await runtime.handleMessage(search);
  const result = output.at(-1);
  assert.equal(result?.type, "result");
  assert.equal(result?.type === "result" ? result.engine : null, "typescript");
  assert.equal(result?.type === "result" ? result.completedDepth : null, 0);
  assert.ok(result?.type === "result" && result.action && applyAction(state, result.action));
});

test("stale and mismatched acknowledgements cannot alter history", async () => {
  const state = developmentState();
  const proposal: LinithAction = { type: "stone", r: 0, c: 0 };
  const core = new FakeNativeCore(() => nativeResult(proposal, 0, stableHash(ongoingState(state, proposal)), []));
  const { output, runtime } = makeHarness(core);
  const search = request(state, 40);
  await runtime.handleMessage(search);

  assert.equal(output.at(-1)?.type, "result");
  await runtime.handleMessage(commitFor(search, proposal, { sessionId: 999 }));
  assert.equal(core.commits.length, 0, "an acknowledgement from another session must be ignored");
  await runtime.handleMessage(commitFor(search, proposal, { fingerprint: "stale-position" }));
  await runtime.handleMessage(commitFor(search, proposal));
  assert.equal(core.commits.length, 0);
});

test("queued commit acknowledgement is processed before the next search", async () => {
  const state = developmentState();
  const firstAction: LinithAction = { type: "stone", r: 0, c: 0 };
  const after = ongoingState(state, firstAction);
  let searchedAfterCommit = false;
  const core = new FakeNativeCore((_state, call) => {
    if (call === 2) searchedAfterCommit = core.commits.length === 1;
    return nativeResult(firstAction, 2, stableHash(after), []);
  });
  const { runtime } = makeHarness(core);
  const first = request(state, 50);
  await runtime.handleMessage(first);

  const commit = runtime.handleMessage(commitFor(first, firstAction));
  const search = runtime.handleMessage(request(state, 51));
  await Promise.all([commit, search]);
  assert.equal(searchedAfterCommit, true);
});

function nativeResult(
  action: LinithAction | null,
  completedTurnDepth: number,
  postActionHash: bigint | null,
  continuation: NativeVeryHardResult["continuation"]
): NativeVeryHardResult {
  const diagnostics = {
    score: 0,
    nodes: 1,
    generatedActions: 1,
    evaluatedPositions: 1,
    transpositionHits: 0,
    cutoffs: 0,
    reSearches: 0,
    rootHistoryHits: 0,
    classificationActions: 0,
    rootThreatDetected: false,
    exactExtensions: 0,
    exactSolved: false,
    elapsedMs: 0,
    completedTurnDepth,
    attemptedTurnDepth: completedTurnDepth,
    stopReason: "max-depth" as const
  };
  return { action, postActionHash, continuation, ...diagnostics, diagnostics };
}

function ongoingState(state: SearchState, action: LinithAction): SearchState {
  const applied = applyAction(state, action);
  assert.ok(applied && !applied.outcome);
  return { board: applied.board, current: applied.current, movesLeft: applied.movesLeft };
}

function stableHash(state: SearchState): bigint {
  let value = 1469598103934665603n;
  for (const char of boardKey(state)) {
    value ^= BigInt(char.charCodeAt(0));
    value = BigInt.asUintN(64, value * 1099511628211n);
  }
  return value;
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

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => row.slice()), current: state.current, movesLeft: state.movesLeft };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}
