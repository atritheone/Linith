import {
  actionKey,
  applyAction,
  boardKey,
  type LinithAction,
  type SearchState
} from "../rulesEngine";
import type {
  NativeVeryHardContinuation,
  NativeVeryHardCore,
  NativeVeryHardResult
} from "./native/core";
import type {
  VeryHardSearchEngine,
  VeryHardWorkerCommitMessage,
  VeryHardWorkerInboundMessage,
  VeryHardWorkerOutboundMessage,
  VeryHardWorkerSearchMessage
} from "./types";

type NativeCorePort = Pick<
  NativeVeryHardCore,
  "search" | "commitAction" | "positionHash" | "clearCache"
>;

interface WorkerRuntimeDependencies {
  engine: VeryHardSearchEngine;
  nativeCore: Promise<NativeCorePort | null>;
  lookupBook(state: SearchState): { action: LinithAction } | null;
  postMessage(message: VeryHardWorkerOutboundMessage): void;
  now?: () => number;
}

interface CachedContinuationStep {
  action: LinithAction;
  inputHash: bigint;
  postActionHash: bigint;
  stateKey: string;
}

interface PendingCommit {
  requestId: number;
  sessionId: number;
  fingerprint: string;
  state: SearchState;
  action: LinithAction;
  core: NativeCorePort | null;
  continuation: CachedContinuationStep[];
}

export interface VeryHardWorkerRuntime {
  /** Messages are deliberately serialized so commit always precedes the next search. */
  handleMessage(message: VeryHardWorkerInboundMessage): Promise<void>;
}

function cloneState(state: SearchState): SearchState {
  return {
    board: state.board.map((row) => row.slice()),
    current: state.current,
    movesLeft: state.movesLeft
  };
}

function cloneAction(action: LinithAction): LinithAction {
  if (action.type === "stone" || action.type === "swan") return { ...action };
  return {
    type: action.type,
    swans: action.swans.map(({ r, c }) => ({ r, c })),
    dir: [action.dir[0], action.dir[1]]
  };
}

function sameIdentity(
  left: { requestId: number; sessionId: number; fingerprint: string },
  right: { requestId: number; sessionId: number; fingerprint: string }
): boolean {
  return left.requestId === right.requestId &&
    left.sessionId === right.sessionId &&
    left.fingerprint === right.fingerprint;
}

function validateContinuation(
  core: NativeCorePort,
  rootState: SearchState,
  rootAction: LinithAction,
  postActionHash: bigint | null,
  continuation: readonly NativeVeryHardContinuation[]
): CachedContinuationStep[] {
  if (postActionHash === null || continuation.length === 0) return [];
  const rootApplied = applyAction(rootState, rootAction);
  if (!rootApplied || rootApplied.outcome || rootApplied.current !== rootState.current) return [];

  let cursor: SearchState = {
    board: rootApplied.board,
    current: rootApplied.current,
    movesLeft: rootApplied.movesLeft
  };
  if (core.positionHash(cursor) !== postActionHash) return [];

  const validated: CachedContinuationStep[] = [];
  for (const candidate of continuation) {
    if (cursor.current !== rootState.current) break;
    const inputHash = core.positionHash(cursor);
    if (inputHash !== candidate.inputHash) break;
    const applied = applyAction(cursor, candidate.action);
    if (!applied) break;
    const nextState: SearchState = {
      board: applied.board,
      current: applied.current,
      movesLeft: applied.movesLeft
    };
    if (!applied.outcome && core.positionHash(nextState) !== candidate.postActionHash) break;
    validated.push({
      action: cloneAction(candidate.action),
      inputHash: candidate.inputHash,
      postActionHash: candidate.postActionHash,
      stateKey: boardKey(cursor)
    });
    if (applied.outcome) break;
    cursor = nextState;
  }
  return validated;
}

export function createVeryHardWorkerRuntime(
  dependencies: WorkerRuntimeDependencies
): VeryHardWorkerRuntime {
  const now = dependencies.now ?? (() => globalThis.performance?.now?.() ?? Date.now());
  let queue: Promise<void> = Promise.resolve();
  let activeSessionId: number | null = null;
  let cachedContinuation: CachedContinuationStep[] = [];
  let pendingCommit: PendingCommit | null = null;

  function clearProtocolState(): void {
    cachedContinuation = [];
    pendingCommit = null;
  }

  async function beginSearchSession(sessionId: number): Promise<NativeCorePort | null> {
    const core = await dependencies.nativeCore;
    if (activeSessionId === null) {
      activeSessionId = sessionId;
    } else if (activeSessionId !== sessionId) {
      clearProtocolState();
      dependencies.engine.clear();
      core?.clearCache();
      activeSessionId = sessionId;
    }
    return core;
  }

  function postError(request: VeryHardWorkerSearchMessage, error: unknown): void {
    dependencies.postMessage({
      type: "error",
      requestId: request.requestId,
      sessionId: request.sessionId,
      fingerprint: request.fingerprint,
      message: error instanceof Error ? error.message : String(error)
    });
  }

  function postFinalResult(
    request: VeryHardWorkerSearchMessage,
    state: SearchState,
    action: LinithAction | null,
    engine: "book" | "native" | "native-continuation" | "typescript",
    completedDepth: number | undefined,
    core: NativeCorePort | null,
    continuation: CachedContinuationStep[] = []
  ): void {
    if (pendingCommit && pendingCommit.sessionId === request.sessionId) {
      throw new Error("Very Hard received a new result before the previous action was committed.");
    }
    cachedContinuation = [];
    pendingCommit = action ? {
      requestId: request.requestId,
      sessionId: request.sessionId,
      fingerprint: request.fingerprint,
      state: cloneState(state),
      action: cloneAction(action),
      core,
      continuation
    } : null;
    dependencies.postMessage({
      type: "result",
      requestId: request.requestId,
      sessionId: request.sessionId,
      fingerprint: request.fingerprint,
      action,
      engine,
      completedDepth
    });
  }

  function tryCachedContinuation(
    request: VeryHardWorkerSearchMessage,
    state: SearchState,
    core: NativeCorePort
  ): boolean {
    const next = cachedContinuation[0];
    if (!next) return false;
    const applied = applyAction(state, next.action);
    const exact = next.stateKey === boardKey(state) &&
      core.positionHash(state) === next.inputHash &&
      !!applied;
    if (!exact) {
      cachedContinuation = [];
      return false;
    }
    if (!applied.outcome) {
      const postState: SearchState = {
        board: applied.board,
        current: applied.current,
        movesLeft: applied.movesLeft
      };
      if (core.positionHash(postState) !== next.postActionHash) {
        cachedContinuation = [];
        return false;
      }
    }
    const remaining = cachedContinuation.slice(1);
    postFinalResult(
      request,
      state,
      cloneAction(next.action),
      "native-continuation",
      undefined,
      core,
      remaining
    );
    return true;
  }

  async function handleSearch(request: VeryHardWorkerSearchMessage): Promise<void> {
    const state = cloneState(request.state);
    const startedAt = now();
    const core = await beginSearchSession(request.sessionId);
    if (pendingCommit) {
      throw new Error("Very Hard protocol requires the previous result to be resolved before another search.");
    }
    if (core && tryCachedContinuation(request, state, core)) return;

    const book = dependencies.lookupBook(state);
    const bookResult = book ? applyAction(state, book.action) : null;
    const universalBookMove = request.style === "doctrinal"
      || !!bookResult?.outcome
      || (bookResult?.opponentLoss ?? 0) > 0;
    if (book && bookResult && universalBookMove) {
      postFinalResult(request, state, cloneAction(book.action), "book", undefined, core);
      return;
    }

    const afterLoad = now();
    const remainingMs = Math.max(0, request.budgetMs - (afterLoad - startedAt));
    let action: LinithAction | null;
    let completedDepth: number;
    let proposedEngine: "native" | "typescript";
    let continuation: CachedContinuationStep[] = [];

    if (core) {
      const result: NativeVeryHardResult = core.search(state, {
        style: request.style,
        budgetMs: remainingMs,
        nodeBudget: remainingMs > 0 ? request.nodeBudget : 0
      });
      action = result.action;
      completedDepth = result.completedTurnDepth;
      proposedEngine = "native";
      if (action) {
        continuation = validateContinuation(
          core,
          state,
          action,
          result.postActionHash,
          result.continuation
        );
      }
    } else {
      const result = dependencies.engine.searchRequest({
        requestId: request.requestId,
        positionHash: request.fingerprint,
        board: state.board,
        current: state.current,
        movesLeft: state.movesLeft,
        style: request.style,
        budgetMs: remainingMs,
        nodeBudget: remainingMs > 0 ? request.nodeBudget : 0
      });
      action = result.action;
      completedDepth = result.completedDepth;
      proposedEngine = "typescript";
    }

    if (!action || !applyAction(state, action)) {
      postFinalResult(request, state, null, proposedEngine, completedDepth, core);
      return;
    }

    postFinalResult(request, state, action, proposedEngine, completedDepth, core, continuation);
  }

  function handleCommit(message: VeryHardWorkerCommitMessage): void {
    const pending = pendingCommit;
    if (!pending) return;
    if (!sameIdentity(pending, message) || boardKey(message.preActionState) !== boardKey(pending.state)) {
      if (message.sessionId === pending.sessionId) {
        cachedContinuation = [];
        pendingCommit = null;
      }
      return;
    }

    pendingCommit = null;
    const legal = applyAction(pending.state, message.action);
    if (!legal) {
      cachedContinuation = [];
      return;
    }
    const committed = pending.core?.commitAction(pending.state, message.action) ?? false;
    const selectedActionMatches = actionKey(message.action) === actionKey(pending.action);
    cachedContinuation = committed && selectedActionMatches
      ? pending.continuation
      : [];
  }

  async function processMessage(message: VeryHardWorkerInboundMessage): Promise<void> {
    if (!message || typeof message !== "object") return;
    if (message.type === "search") {
      try {
        await handleSearch(message);
      } catch (error) {
        clearProtocolState();
        postError(message, error);
      }
      return;
    }
    try {
      handleCommit(message);
    } catch {
      clearProtocolState();
    }
  }

  return {
    handleMessage(message): Promise<void> {
      const next = queue.then(() => processMessage(message));
      queue = next.catch(() => undefined);
      return next;
    }
  };
}
