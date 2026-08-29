import type { Board, Player } from "../encirclement";
import type { LinithAction, SearchState } from "../rulesEngine";

export type VeryHardStopReason =
  | "max-depth"
  | "deadline"
  | "node-budget"
  | "aborted"
  | "no-legal-actions";

export interface VeryHardSearchOptions {
  /** AI personality. Unknown names use the doctrinal profile. */
  style?: string;
  /** Relative wall-clock allowance. Set to Infinity to disable it. */
  budgetMs?: number;
  /** Deterministic cap on visited search nodes. Set to Infinity to disable it. */
  nodeBudget?: number;
  /** Maximum completed iterative-deepening depth, measured in complete turns. */
  maxDepth?: number;
  /** Forcing completed turns searched at the normal horizon. */
  tacticalDepth?: number;
  /** Maximum selective endgame extensions beyond the normal horizon. */
  exactDepth?: number;
  /** Maximum number of transposition entries retained by the search. */
  transpositionCapacity?: number;
  /** Optional absolute deadline expressed in the same clock as `now`. */
  deadlineMs?: number;
  /** Injectable monotonic clock, primarily for deterministic tests. */
  now?: () => number;
  /** Cooperative cancellation for direct (non-worker) callers. */
  signal?: AbortSignal;
}

export interface VeryHardSearchRequest extends VeryHardSearchOptions {
  requestId: number;
  positionHash: string;
  board: Board;
  current: Player;
  movesLeft: number;
}

export interface VeryHardDiagnostics {
  completedDepth: number;
  attemptedDepth: number;
  nodes: number;
  elapsedMs: number;
  generatedActions: number;
  evaluatedPositions: number;
  transpositionHits: number;
  transpositionStores: number;
  cutoffs: number;
  reSearches: number;
  aspirationReSearches: number;
  selectivePrunes: number;
  futilityPrunes: number;
  lateMoveReductions: number;
  repetitions: number;
  exactExtensions: number;
  exactSolved: boolean;
  forcedMove: boolean;
  rootActions: number;
  rootActionsSearched: number;
  timedOut: boolean;
  nodeBudgetReached: boolean;
  aborted: boolean;
  stopReason: VeryHardStopReason;
  principalVariation: LinithAction[];
}

/** A reusable engine keeps safe move-ordering and transposition knowledge between turns. */
export interface VeryHardSearchEngine {
  search(state: SearchState, options?: VeryHardSearchOptions): VeryHardSearchResult;
  searchRequest(request: VeryHardSearchRequest): VeryHardSearchResult;
  clear(): void;
  readonly transpositionSize: number;
}

export interface VeryHardSearchResult extends VeryHardDiagnostics {
  requestId?: number;
  positionHash: string;
  action: LinithAction | null;
  score: number;
  diagnostics: VeryHardDiagnostics;
}

export interface VeryHardWorkerSearchMessage {
  type: "search";
  requestId: number;
  /** Monotonic live-match/timeline identity assigned by the renderer. */
  sessionId: number;
  fingerprint: string;
  state: {
    board: Board;
    current: Player;
    movesLeft: number;
  };
  style: string;
  budgetMs: number;
  nodeBudget?: number;
}

/** Acknowledges the action that the live executor actually accepted. */
export interface VeryHardWorkerCommitMessage {
  type: "commit";
  requestId: number;
  sessionId: number;
  fingerprint: string;
  preActionState: {
    board: Board;
    current: Player;
    movesLeft: number;
  };
  action: LinithAction;
}

export interface VeryHardWorkerResultMessage {
  type: "result";
  requestId: number;
  sessionId: number;
  fingerprint: string;
  action: LinithAction | null;
  engine?: "book" | "native" | "native-continuation" | "typescript";
  completedDepth?: number;
}

export interface VeryHardWorkerErrorMessage {
  type: "error";
  requestId: number;
  sessionId: number;
  fingerprint: string;
  message: string;
}

export type VeryHardWorkerInboundMessage =
  | VeryHardWorkerSearchMessage
  | VeryHardWorkerCommitMessage;
export type VeryHardWorkerOutboundMessage =
  | VeryHardWorkerResultMessage
  | VeryHardWorkerErrorMessage;

export type VeryHardPosition = SearchState;
