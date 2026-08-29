import type { Board, Player } from "../../encirclement";
import { AI_STYLE_IDS } from "../../aiStyles";
import type { LinithAction, SearchState } from "../../rulesEngine";

interface WasmCoreExports {
  reset_state(player: number, movesLeft: number): void;
  set_cell(cell: number, tile: number): void;
  get_cell(cell: number): number;
  generate_actions(): number;
  get_action_type(index: number): number;
  get_action_cell(index: number): number;
  get_action_direction(index: number): number;
  get_action_mask_low(index: number): bigint;
  get_action_mask_high(index: number): bigint;
  commit_root_action(type: number, cell: number, direction: number, maskLow: bigint, maskHigh: bigint): number;
  position_hash(): bigint;
  current_turn_win_distance(maximumActions: number): number;
  tactical_probe(perspective: number, style: number, tacticalDepth: number): number;
  get_search_quiescence_forced_actions(): number;
  search_best(
    maxTurnDepth: number,
    nodeBudget: number,
    budgetMs: number,
    style: number,
    tacticalDepth: number,
    exactDepth: number
  ): number;
  get_search_score(): number;
  get_search_nodes(): number;
  get_search_generated_actions(): number;
  get_search_evaluations(): number;
  get_search_tt_hits(): number;
  get_search_cutoffs(): number;
  get_search_researches(): number;
  get_search_root_history_hits(): number;
  get_search_classification_actions(): number;
  get_search_root_threat_detected(): number;
  get_search_exact_extensions(): number;
  get_search_exact_solved(): number;
  get_search_root_post_hash(): bigint;
  get_search_continuation_count(): number;
  get_search_continuation_type(index: number): number;
  get_search_continuation_cell(index: number): number;
  get_search_continuation_direction(index: number): number;
  get_search_continuation_mask_low(index: number): bigint;
  get_search_continuation_mask_high(index: number): bigint;
  get_search_continuation_input_hash(index: number): bigint;
  get_search_continuation_post_hash(index: number): bigint;
  get_search_completed_depth(): number;
  get_search_attempted_depth(): number;
  get_search_stop_reason(): number;
  evaluate_loaded_position(perspective: number, style: number): number;
  clear_transposition_table(): void;
  memory: WebAssembly.Memory;
}

export interface NativeVeryHardOptions {
  /** Completed turns, not individual actions. */
  maxTurnDepth?: number;
  nodeBudget?: number;
  budgetMs?: number;
  style?: string;
  tacticalDepth?: number;
  /** Bounded completed-turn extensions for genuinely small late positions. */
  exactDepth?: number;
}

export interface NativeVeryHardDiagnostics {
  score: number;
  nodes: number;
  generatedActions: number;
  evaluatedPositions: number;
  transpositionHits: number;
  cutoffs: number;
  reSearches: number;
  rootHistoryHits: number;
  classificationActions: number;
  rootThreatDetected: boolean;
  exactExtensions: number;
  exactSolved: boolean;
  elapsedMs: number;
  completedTurnDepth: number;
  attemptedTurnDepth: number;
  stopReason: "max-depth" | "deadline" | "node-budget";
}

export interface NativeVeryHardResult extends NativeVeryHardDiagnostics {
  action: LinithAction | null;
  postActionHash: bigint | null;
  continuation: NativeVeryHardContinuation[];
  diagnostics: NativeVeryHardDiagnostics;
}

export interface NativeVeryHardContinuation {
  action: LinithAction;
  /** Hash of the exact board/player/actions-left state on which this action is valid. */
  inputHash: bigint;
  postActionHash: bigint;
}

const DIRECTIONS = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [-1, 1], [1, -1], [1, 1]
] as const;

const STYLE_CODES: Readonly<Record<string, number>> = Object.freeze(
  Object.fromEntries(AI_STYLE_IDS.map((style, index) => [style, index]))
);

function defaultNow(): number {
  return globalThis.performance?.now?.() ?? Date.now();
}

function groupFromMasks(lowValue: bigint, highValue: bigint): Array<{ r: number; c: number }> {
  const low = BigInt.asUintN(64, lowValue);
  const high = BigInt.asUintN(64, highValue);
  const swans: Array<{ r: number; c: number }> = [];
  for (let cell = 0; cell < 100; cell += 1) {
    const present = cell < 64
      ? (low & (1n << BigInt(cell))) !== 0n
      : (high & (1n << BigInt(cell - 64))) !== 0n;
    if (present) swans.push({ r: Math.floor(cell / 10), c: cell % 10 });
  }
  return swans;
}

function actionAt(core: WasmCoreExports, index: number): LinithAction {
  const type = core.get_action_type(index);
  if (type === 0 || type === 1) {
    const cell = core.get_action_cell(index);
    return {
      type: type === 0 ? "stone" : "swan",
      r: Math.floor(cell / 10),
      c: cell % 10
    };
  }
  const direction = DIRECTIONS[core.get_action_direction(index)];
  if (!direction || (type !== 2 && type !== 3)) throw new Error("Native Very Hard returned a corrupt action.");
  return {
    type: type === 2 ? "move" : "push",
    swans: groupFromMasks(core.get_action_mask_low(index), core.get_action_mask_high(index)),
    dir: direction
  };
}

function continuationActionAt(core: WasmCoreExports, index: number): LinithAction {
  const type = core.get_search_continuation_type(index);
  if (type === 0 || type === 1) {
    const cell = core.get_search_continuation_cell(index);
    return {
      type: type === 0 ? "stone" : "swan",
      r: Math.floor(cell / 10),
      c: cell % 10
    };
  }
  const direction = DIRECTIONS[core.get_search_continuation_direction(index)];
  if (!direction || (type !== 2 && type !== 3)) throw new Error("Native Very Hard returned a corrupt continuation.");
  return {
    type: type === 2 ? "move" : "push",
    swans: groupFromMasks(
      core.get_search_continuation_mask_low(index),
      core.get_search_continuation_mask_high(index)
    ),
    dir: direction
  };
}

function encodeAction(action: LinithAction): [number, number, number, bigint, bigint] {
  if (action.type === "stone" || action.type === "swan") {
    return [action.type === "stone" ? 0 : 1, action.r * 10 + action.c, 0, 0n, 0n];
  }
  const direction = DIRECTIONS.findIndex(([dr, dc]) => dr === action.dir[0] && dc === action.dir[1]);
  if (direction < 0) throw new Error("Cannot commit a Very Hard action with an invalid direction.");
  let low = 0n;
  let high = 0n;
  for (const { r, c } of action.swans) {
    const cell = r * 10 + c;
    if (cell < 64) low |= 1n << BigInt(cell);
    else high |= 1n << BigInt(cell - 64);
  }
  return [action.type === "move" ? 2 : 3, 0, direction, low, high];
}

function validateState(state: SearchState): void {
  if ((state.current !== 1 && state.current !== 2) || !Number.isInteger(state.movesLeft) || state.movesLeft < 1) {
    throw new Error("Native Very Hard received invalid turn state.");
  }
  if (!Array.isArray(state.board) || state.board.length !== 10 || state.board.some((row) => !Array.isArray(row) || row.length !== 10)) {
    throw new Error("Native Very Hard requires a 10x10 board.");
  }
  for (const tile of state.board.flat()) {
    if (!Number.isInteger(tile) || tile < 0 || tile > 5) throw new Error(`Native Very Hard received invalid tile ${String(tile)}.`);
  }
}

function loadPosition(core: WasmCoreExports, board: Board, current: Player, movesLeft: number): void {
  core.reset_state(current, movesLeft);
  for (let r = 0; r < 10; r += 1) {
    for (let c = 0; c < 10; c += 1) core.set_cell(r * 10 + c, board[r][c]);
  }
}

export class NativeVeryHardCore {
  constructor(
    private readonly core: WasmCoreExports,
    private readonly now: () => number = defaultNow
  ) {}

  search(state: SearchState, options: NativeVeryHardOptions = {}): NativeVeryHardResult {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    const maxTurnDepth = Math.max(1, Math.min(12, Math.floor(options.maxTurnDepth ?? 6)));
    const nodeBudget = Math.max(0, Math.min(2_000_000_000, Math.floor(options.nodeBudget ?? 2_000_000)));
    const budgetMs = Number.isFinite(options.budgetMs) ? Math.max(0, options.budgetMs!) : 0;
    const style = STYLE_CODES[options.style ?? "doctrinal"] ?? STYLE_CODES.doctrinal;
    const tacticalDepth = Math.max(0, Math.min(3, Math.floor(options.tacticalDepth ?? 1)));
    const exactDepth = Math.max(0, Math.min(16, Math.floor(options.exactDepth ?? 12)));
    const startedAt = this.now();
    const bestIndex = this.core.search_best(maxTurnDepth, nodeBudget, budgetMs, style, tacticalDepth, exactDepth);
    const elapsedMs = Math.max(0, this.now() - startedAt);
    const stop = this.core.get_search_stop_reason();
    const continuation = Array.from(
      { length: this.core.get_search_continuation_count() },
      (_, index): NativeVeryHardContinuation => ({
        action: continuationActionAt(this.core, index),
        inputHash: this.core.get_search_continuation_input_hash(index),
        postActionHash: this.core.get_search_continuation_post_hash(index)
      })
    );
    const diagnostics: NativeVeryHardDiagnostics = {
      score: this.core.get_search_score(),
      nodes: this.core.get_search_nodes(),
      generatedActions: this.core.get_search_generated_actions(),
      evaluatedPositions: this.core.get_search_evaluations(),
      transpositionHits: this.core.get_search_tt_hits(),
      cutoffs: this.core.get_search_cutoffs(),
      reSearches: this.core.get_search_researches(),
      rootHistoryHits: this.core.get_search_root_history_hits(),
      classificationActions: this.core.get_search_classification_actions(),
      rootThreatDetected: this.core.get_search_root_threat_detected() !== 0,
      exactExtensions: this.core.get_search_exact_extensions(),
      exactSolved: this.core.get_search_exact_solved() !== 0,
      elapsedMs,
      completedTurnDepth: this.core.get_search_completed_depth(),
      attemptedTurnDepth: this.core.get_search_attempted_depth(),
      stopReason: stop === 1 ? "deadline" : stop === 2 ? "node-budget" : "max-depth"
    };
    return {
      action: bestIndex < 0 ? null : actionAt(this.core, bestIndex),
      postActionHash: bestIndex < 0 ? null : this.core.get_search_root_post_hash(),
      continuation,
      ...diagnostics,
      diagnostics
    };
  }

  /** Commit game history only after the caller has selected this legal action. */
  commitAction(state: SearchState, action: LinithAction): boolean {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    const [type, cell, direction, low, high] = encodeAction(action);
    return this.core.commit_root_action(type, cell, direction, low, high) !== 0;
  }

  positionHash(state: SearchState): bigint {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    return this.core.position_hash();
  }

  currentTurnWinDistance(state: SearchState, maximumActions = 4): number {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    return this.core.current_turn_win_distance(Math.max(1, Math.min(8, Math.floor(maximumActions))));
  }

  tacticalProbe(state: SearchState, perspective: Player, tacticalDepth = 1, style = "doctrinal") {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    const score = this.core.tactical_probe(
      perspective,
      STYLE_CODES[style] ?? STYLE_CODES.doctrinal,
      Math.max(0, Math.min(3, Math.floor(tacticalDepth)))
    );
    return { score, forcedSameTurnActions: this.core.get_search_quiescence_forced_actions() };
  }

  clearCache(): void {
    this.core.clear_transposition_table();
  }

  evaluate(state: SearchState, perspective: Player, style = "doctrinal"): number {
    validateState(state);
    loadPosition(this.core, state.board, state.current, state.movesLeft);
    return this.core.evaluate_loaded_position(perspective, STYLE_CODES[style] ?? STYLE_CODES.doctrinal);
  }
}

export function instantiateNativeVeryHardCore(
  bytes: BufferSource,
  now: () => number = defaultNow
): NativeVeryHardCore {
  const module = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(module, {
    env: {
      abort(_message: number, _file: number, line: number, column: number): never {
        throw new Error(`Native Very Hard aborted at ${line}:${column}.`);
      },
      now
    }
  });
  return new NativeVeryHardCore(instance.exports as unknown as WasmCoreExports, now);
}
