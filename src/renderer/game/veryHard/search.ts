import type { Board, Player } from "../encirclement";
import {
  BOARD_SIZE,
  DIRECTIONS,
  actionKey,
  applyAction,
  boardKey,
  cloneBoard,
  generateLegalActions,
  inBounds,
  isActiveSwan,
  isEnemySwan,
  opponentOf,
  type AppliedAction,
  type LinithAction,
  type SearchState
} from "../rulesEngine";
import {
  evaluateVeryHardPosition,
  evaluateVeryHardRootPersonality,
  VERY_HARD_MATE_SCORE
} from "./evaluate";
import { lookupOpeningBookAction } from "./openingBook";
import type {
  VeryHardDiagnostics,
  VeryHardSearchEngine,
  VeryHardSearchOptions,
  VeryHardSearchRequest,
  VeryHardSearchResult,
  VeryHardStopReason
} from "./types";

export const VERY_HARD_DEFAULT_BUDGET_MS = 750;
export const VERY_HARD_DEFAULT_NODE_BUDGET = 60_000;
/** Depth is expressed in completed turns, not individual actions. */
export const VERY_HARD_DEFAULT_MAX_DEPTH = 4;
export const VERY_HARD_DEFAULT_TACTICAL_DEPTH = 1;
export const VERY_HARD_DEFAULT_EXACT_DEPTH = 12;
export const VERY_HARD_DEFAULT_TRANSPOSITION_CAPACITY = 50_000;

const NEGATIVE_INFINITY = -VERY_HARD_MATE_SCORE - 1;
const POSITIVE_INFINITY = VERY_HARD_MATE_SCORE + 1;
const MATE_BAND = VERY_HARD_MATE_SCORE / 2;
const PREPARED_CACHE_CAPACITY = 64;
const MAX_QUIESCENCE_ACTIONS = 10;
const ASPIRATION_MINIMUM = 4_000;
const REPETITION_CONTEMPT = 128;
const ROOT_HISTORY_PENALTY = 4_000;

type TranspositionBound = "exact" | "lower" | "upper";

interface TranspositionEntry {
  depth: number;
  score: number;
  bound: TranspositionBound;
  bestActionKey: string | null;
  principalVariation: LinithAction[];
  generation: number;
  solved: boolean;
}

interface SearchValue {
  score: number;
  principalVariation: LinithAction[];
  solved: boolean;
}

interface PreparedAction {
  action: LinithAction;
  key: string;
  applied: AppliedAction;
  orderScore: number;
  staticScore: number | null;
  terminalScore: number | null;
  tactical: boolean;
  critical: boolean;
  quiet: boolean;
  changesTurn: boolean;
}

interface PreparedActions {
  actions: PreparedAction[];
  allLegalCount: number;
}

interface NormalizedOptions {
  style: string;
  budgetMs: number;
  nodeBudget: number;
  maxDepth: number;
  tacticalDepth: number;
  exactDepth: number;
  transpositionCapacity: number;
  deadlineMs: number;
  now: () => number;
  signal?: AbortSignal;
}

interface EngineMemory {
  table: Map<string, TranspositionEntry>;
  history: Map<string, number>;
  killers: Map<number, string[]>;
  rootVisits: Map<string, number>;
  generation: number;
}

interface SearchContext {
  options: NormalizedOptions;
  memory: EngineMemory;
  rootPlayer: Player;
  rootKey: string;
  rootPreferredActionKey: string | null;
  rootPrepared: PreparedActions;
  startedAt: number;
  deadline: number;
  nodes: number;
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
  rootActionsSearched: Set<string>;
  prepared: Map<string, PreparedActions>;
  path: Map<string, number>;
}

class SearchInterrupted extends Error {
  constructor(readonly reason: Extract<VeryHardStopReason, "deadline" | "node-budget" | "aborted">) {
    super(reason);
  }
}

function defaultNow(): number {
  return globalThis.performance?.now?.() ?? Date.now();
}

function normalizedNonNegative(value: number | undefined, fallback: number): number {
  if (value === Infinity) return Infinity;
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : fallback;
}

function normalizedInteger(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.max(minimum, Math.min(maximum, Math.floor(value)));
}

function normalizeOptions(options: VeryHardSearchOptions): NormalizedOptions {
  return {
    style: options.style || "doctrinal",
    budgetMs: normalizedNonNegative(options.budgetMs, VERY_HARD_DEFAULT_BUDGET_MS),
    nodeBudget: normalizedNonNegative(options.nodeBudget, VERY_HARD_DEFAULT_NODE_BUDGET),
    maxDepth: normalizedInteger(options.maxDepth, VERY_HARD_DEFAULT_MAX_DEPTH, 1, 32),
    tacticalDepth: normalizedInteger(options.tacticalDepth, VERY_HARD_DEFAULT_TACTICAL_DEPTH, 0, 8),
    exactDepth: normalizedInteger(options.exactDepth, VERY_HARD_DEFAULT_EXACT_DEPTH, 0, 32),
    transpositionCapacity: normalizedInteger(
      options.transpositionCapacity,
      VERY_HARD_DEFAULT_TRANSPOSITION_CAPACITY,
      0,
      1_000_000
    ),
    deadlineMs: normalizedNonNegative(options.deadlineMs, Infinity),
    now: options.now ?? defaultNow,
    signal: options.signal
  };
}

function validateState(state: SearchState): void {
  if (state.current !== 1 && state.current !== 2) throw new Error("Very Hard requires current to be 1 or 2.");
  if (!Number.isInteger(state.movesLeft) || state.movesLeft < 1) {
    throw new Error("Very Hard requires movesLeft to be a positive integer.");
  }
  if (!Array.isArray(state.board) || state.board.length !== BOARD_SIZE) {
    throw new Error(`Very Hard requires a ${BOARD_SIZE}x${BOARD_SIZE} board.`);
  }
  for (const row of state.board) {
    if (!Array.isArray(row) || row.length !== BOARD_SIZE) {
      throw new Error(`Very Hard requires a ${BOARD_SIZE}x${BOARD_SIZE} board.`);
    }
    for (const tile of row) {
      if (!Number.isInteger(tile) || tile < 0 || tile > 5) {
        throw new Error(`Very Hard received an invalid board tile: ${String(tile)}.`);
      }
    }
  }
}

function cloneAction(action: LinithAction): LinithAction {
  if (action.type === "stone" || action.type === "swan") {
    return { type: action.type, r: action.r, c: action.c };
  }
  return {
    type: action.type,
    swans: action.swans.map(({ r, c }) => ({ r, c })),
    dir: [action.dir[0], action.dir[1]]
  };
}

function clonePrincipalVariation(actions: readonly LinithAction[]): LinithAction[] {
  return actions.map(cloneAction);
}

function compareKeys(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function checkForStop(context: SearchContext, countNode: boolean): void {
  if (context.options.signal?.aborted) throw new SearchInterrupted("aborted");
  if (context.options.now() >= context.deadline) throw new SearchInterrupted("deadline");
  if (!countNode) return;
  if (context.nodes >= context.options.nodeBudget) throw new SearchInterrupted("node-budget");
  context.nodes += 1;
}

function outcomeScore(outcome: AppliedAction["outcome"], perspective: Player, ply: number): number {
  if (outcome === "draw" || outcome === null) return 0;
  const winner: Player = outcome === "sun" ? 1 : 2;
  return winner === perspective
    ? VERY_HARD_MATE_SCORE - ply
    : -VERY_HARD_MATE_SCORE + ply;
}

function scoreForTransposition(score: number, ply: number): number {
  if (score > MATE_BAND) return score + ply;
  if (score < -MATE_BAND) return score - ply;
  return score;
}

function scoreFromTransposition(score: number, ply: number): number {
  if (score > MATE_BAND) return score - ply;
  if (score < -MATE_BAND) return score + ply;
  return score;
}

function evaluate(state: SearchState, perspective: Player, context: SearchContext): number {
  context.evaluatedPositions += 1;
  // Recursive strength is canonical for every character. Personality is a
  // bounded, action-aware root preference applied only after a line is valued.
  return evaluateVeryHardPosition(state, perspective, "doctrinal");
}

function positionKey(state: SearchState, context: SearchContext): string {
  return `${context.options.style}|${boardKey(state)}`;
}

function pathKey(state: SearchState): string {
  return boardKey(state);
}

function adjacentSwanCount(board: Board, r: number, c: number, player: Player, enemy: boolean): number {
  let count = 0;
  for (const [dr, dc] of DIRECTIONS) {
    const nr = r + dr;
    const nc = c + dc;
    if (!inBounds(nr, nc)) continue;
    if (enemy ? isEnemySwan(board[nr][nc], player) : isActiveSwan(board[nr][nc], player)) count += 1;
  }
  return count;
}

function geometricOrderScore(state: SearchState, action: LinithAction): number {
  if (action.type === "stone" || action.type === "swan") {
    const enemyContact = adjacentSwanCount(state.board, action.r, action.c, state.current, true);
    const friendlyContact = adjacentSwanCount(state.board, action.r, action.c, state.current, false);
    const centre = 18 - Math.abs(9 - action.r * 2) - Math.abs(9 - action.c * 2);
    if (action.type === "stone") return 50_000 + enemyContact * 14_000 + friendlyContact * 700 + centre;
    return 38_000 + friendlyContact * 1_200 + centre * 4;
  }

  let destinationPressure = 0;
  for (const { r, c } of action.swans) {
    const nr = r + action.dir[0];
    const nc = c + action.dir[1];
    destinationPressure += adjacentSwanCount(
      state.board,
      nr,
      nc,
      action.type === "push" ? opponentOf(state.current) : state.current,
      true
    );
  }
  const typeScore = action.type === "push" ? 68_000 : 52_000;
  return typeScore + action.swans.length * 1_800 + destinationPressure * 400;
}

function squareKey(r: number, c: number): number {
  return r * BOARD_SIZE + c;
}

interface FragileShape {
  swans: Set<number>;
  liberties: Set<number>;
}

function fragileShape(board: Board, player: Player): FragileShape {
  const seen = new Set<number>();
  const fragileSwans = new Set<number>();
  const fragileLiberties = new Set<number>();

  for (let startR = 0; startR < BOARD_SIZE; startR += 1) {
    for (let startC = 0; startC < BOARD_SIZE; startC += 1) {
      if (!isActiveSwan(board[startR][startC], player)) continue;
      const start = squareKey(startR, startC);
      if (seen.has(start)) continue;
      const group: Array<[number, number]> = [[startR, startC]];
      const liberties = new Set<number>();
      seen.add(start);
      for (let index = 0; index < group.length; index += 1) {
        const [r, c] = group[index];
        for (const [dr, dc] of DIRECTIONS) {
          const nr = r + dr;
          const nc = c + dc;
          if (!inBounds(nr, nc)) continue;
          const key = squareKey(nr, nc);
          if (board[nr][nc] === 0) liberties.add(key);
          else if (isActiveSwan(board[nr][nc], player) && !seen.has(key)) {
            seen.add(key);
            group.push([nr, nc]);
          }
        }
      }
      if (liberties.size > 3) continue;
      for (const [r, c] of group) fragileSwans.add(squareKey(r, c));
      for (const liberty of liberties) fragileLiberties.add(liberty);
    }
  }
  return { swans: fragileSwans, liberties: fragileLiberties };
}

function actionFootprint(action: LinithAction): Set<number> {
  if (action.type === "stone" || action.type === "swan") return new Set([squareKey(action.r, action.c)]);
  const result = new Set<number>();
  for (const { r, c } of action.swans) {
    result.add(squareKey(r, c));
    result.add(squareKey(r + action.dir[0], c + action.dir[1]));
  }
  return result;
}

function actionIsCritical(action: LinithAction, mine: FragileShape, theirs: FragileShape): boolean {
  for (const square of actionFootprint(action)) {
    if (
      mine.swans.has(square) || mine.liberties.has(square) ||
      theirs.swans.has(square) || theirs.liberties.has(square)
    ) return true;
  }
  return false;
}

function prepareActions(state: SearchState, context: SearchContext): PreparedActions {
  const key = boardKey(state);
  const cached = context.prepared.get(key);
  if (cached) return cached;

  const legal = generateLegalActions(state);
  context.generatedActions += legal.length;
  const mine = fragileShape(state.board, state.current);
  const theirs = fragileShape(state.board, opponentOf(state.current));
  const prepared: PreparedAction[] = [];

  for (let index = 0; index < legal.length; index += 1) {
    if (index > 0 && index % 32 === 0 && key !== context.rootKey) checkForStop(context, false);
    const action = legal[index];
    const applied = applyAction(state, action);
    if (!applied) continue;
    const terminalScore = applied.outcome ? outcomeScore(applied.outcome, state.current, 1) : null;
    const critical = actionIsCritical(action, mine, theirs);
    const winningTerminal = terminalScore !== null && terminalScore > MATE_BAND;
    const drawingTerminal = terminalScore === 0;
    const tactical = winningTerminal || drawingTerminal || applied.opponentLoss > 0;
    let orderScore = geometricOrderScore(state, action);
    if (winningTerminal) orderScore += 2_000_000_000;
    else if (drawingTerminal) orderScore += 8_000_000;
    else if (terminalScore !== null) orderScore -= 2_000_000_000;
    if (applied.opponentLoss > 0) orderScore += 40_000_000 + applied.opponentLoss * 5_000_000;
    if (applied.current === state.current) orderScore += 2_000_000;
    if (critical) orderScore += 5_000_000;
    prepared.push({
      action,
      key: actionKey(action),
      applied,
      orderScore,
      staticScore: null,
      terminalScore,
      tactical,
      critical,
      quiet: terminalScore === null && applied.opponentLoss === 0,
      changesTurn: applied.current !== state.current
    });
  }

  prepared.sort((left, right) => right.orderScore - left.orderScore || compareKeys(left.key, right.key));
  const result = { actions: prepared, allLegalCount: legal.length };
  if (context.prepared.size >= PREPARED_CACHE_CAPACITY) {
    const oldest = context.prepared.keys().next().value as string | undefined;
    if (oldest !== undefined && oldest !== context.rootKey) context.prepared.delete(oldest);
  }
  context.prepared.set(key, result);
  return result;
}

function hasImmediateWin(state: SearchState, player: Player, context: SearchContext): boolean {
  const actions = generateLegalActions(state);
  context.generatedActions += actions.length;
  const expected = player === 1 ? "sun" : "moon";
  for (const action of actions) {
    if (applyAction(state, action)?.outcome === expected) return true;
  }
  return false;
}

/**
 * Run the expensive response probe only when the opponent demonstrably has a
 * win on the untouched board. This preserves quiet only-defenses without
 * imposing an O(B^2) tax on every ordinary root.
 */
function classifyRootDefenses(
  state: SearchState,
  prepared: PreparedActions,
  context: SearchContext
): PreparedAction | null {
  const opponent = opponentOf(state.current);
  const hypotheticalThreat: SearchState = { board: state.board, current: opponent, movesLeft: 1 };
  if (!hasImmediateWin(hypotheticalThreat, opponent, context)) return null;

  const safe: PreparedAction[] = [];
  let hasUnresolvedContinuation = false;
  for (const info of prepared.actions) {
    let isSafe = false;
    let classified = true;
    if (info.applied.outcome) {
      isSafe = info.terminalScore === 0 || (info.terminalScore !== null && info.terminalScore > MATE_BAND);
    } else if (!info.changesTurn) {
      // A scheduled/bonus continuation still has another chance to answer the
      // threat; completed-turn search will decide whether it really does.
      classified = false;
      hasUnresolvedContinuation = true;
    } else {
      isSafe = !hasImmediateWin(info.applied, info.applied.current, context);
    }

    if (isSafe) {
      info.orderScore += 600_000_000;
      info.critical = true;
      safe.push(info);
    } else if (classified) {
      info.orderScore -= 600_000_000;
    }
  }
  prepared.actions.sort((left, right) => right.orderScore - left.orderScore || compareKeys(left.key, right.key));
  return !hasUnresolvedContinuation && safe.length === 1 ? safe[0] : null;
}

function candidateWidth(state: SearchState, depth: number, ply: number, pvNode: boolean): number {
  if (ply === 0) {
    const base = depth <= 1 ? 32 : depth === 2 ? 30 : depth === 3 ? 28 : 24;
    return base + (pvNode ? 8 : 0);
  }
  const continuation = state.movesLeft > 1 ? 36 : 0;
  const base = depth <= 1 ? 38 : depth === 2 ? 30 : 24;
  return base + continuation + (pvNode ? 8 : 0);
}

function staticOrderScore(info: PreparedAction, state: SearchState, context: SearchContext): number {
  if (info.staticScore === null) {
    info.staticScore = info.terminalScore ?? evaluate(info.applied, state.current, context);
  }
  const priorVisits = context.memory.rootVisits.get(boardKey(info.applied)) ?? 0;
  return info.orderScore + Math.max(-500_000, Math.min(500_000, info.staticScore * 4))
    - priorVisits * ROOT_HISTORY_PENALTY * 4;
}

function orderedCandidates(
  state: SearchState,
  prepared: PreparedActions,
  depth: number,
  ply: number,
  pvNode: boolean,
  preferredActionKey: string | null,
  context: SearchContext,
  forcingOnly = false,
  mustCompleteTurn = false
): PreparedAction[] {
  const killers = context.memory.killers.get(ply) ?? [];
  const width = candidateWidth(state, depth, ply, pvNode);
  if (ply === 0) {
    // The root is special: score every legal successor before selective
    // widening, so a quiet positional move is not hidden by its action type.
    for (const info of prepared.actions) staticOrderScore(info, state, context);
  }
  const roughOrder = ply === 0
    ? [...prepared.actions].sort((left, right) =>
        staticOrderScore(right, state, context) - staticOrderScore(left, state, context)
        || compareKeys(left.key, right.key)
      )
    : prepared.actions;
  const mandatory = prepared.actions.filter((info) =>
    info.tactical || info.critical || info.key === preferredActionKey
  );
  let pool: PreparedAction[];
  if (forcingOnly && !mustCompleteTurn) {
    pool = mandatory;
  } else if (prepared.actions.length <= width) {
    pool = prepared.actions;
  } else {
    const selected = new Map<string, PreparedAction>();
    for (const info of mandatory) selected.set(info.key, info);
    for (const info of roughOrder) {
      if (selected.size >= width && !info.tactical && !info.critical) break;
      selected.set(info.key, info);
      if (selected.size >= width && mandatory.every((entry) => selected.has(entry.key))) break;
    }
    pool = [...selected.values()];
    context.selectivePrunes += Math.max(0, prepared.actions.length - pool.length);
  }

  const scoreAll = ply === 0 || pool.length <= width * 2;
  const scored = scoreAll ? pool : pool.slice(0, width * 2);
  const scoredKeys = new Set(scored.map((info) => info.key));
  const decorated = pool.map((info) => {
    let score = scoredKeys.has(info.key) ? staticOrderScore(info, state, context) : info.orderScore;
    if (info.key === preferredActionKey) score += 4_000_000_000;
    const killerIndex = killers.indexOf(info.key);
    if (killerIndex >= 0) score += 3_000_000 - killerIndex * 100_000;
    score += context.memory.history.get(`${state.current}:${info.key}`) ?? 0;
    return { info, score };
  });
  decorated.sort((left, right) => right.score - left.score || compareKeys(left.info.key, right.info.key));
  return decorated.map(({ info }) => info);
}

function rememberCutoff(info: PreparedAction, state: SearchState, depth: number, ply: number, context: SearchContext): void {
  if (!info.quiet) return;
  const killers = context.memory.killers.get(ply) ?? [];
  if (killers[0] !== info.key) {
    context.memory.killers.set(ply, [info.key, ...killers.filter((value) => value !== info.key)].slice(0, 2));
  }
  const historyKey = `${state.current}:${info.key}`;
  const previous = context.memory.history.get(historyKey) ?? 0;
  context.memory.history.set(historyKey, Math.min(20_000_000, previous + depth * depth * 40));
}

function lookupTransposition(
  key: string,
  depth: number,
  alpha: number,
  beta: number,
  ply: number,
  context: SearchContext,
  allowValue = true
): { value?: SearchValue; alpha: number; beta: number; preferredActionKey: string | null } {
  const entry = context.memory.table.get(key);
  if (!entry) return { alpha, beta, preferredActionKey: null };
  entry.generation = context.memory.generation;
  const preferredActionKey = entry.bestActionKey;
  if (!entry.solved && entry.depth < depth) return { alpha, beta, preferredActionKey };

  context.transpositionHits += 1;
  if (!allowValue) return { alpha, beta, preferredActionKey };
  const score = scoreFromTransposition(entry.score, ply);
  if (entry.bound === "exact") {
    return {
      value: { score, principalVariation: clonePrincipalVariation(entry.principalVariation), solved: entry.solved },
      alpha,
      beta,
      preferredActionKey
    };
  }
  if (entry.bound === "lower") alpha = Math.max(alpha, score);
  else beta = Math.min(beta, score);
  if (alpha >= beta) {
    return {
      value: { score, principalVariation: clonePrincipalVariation(entry.principalVariation), solved: false },
      alpha,
      beta,
      preferredActionKey
    };
  }
  return { alpha, beta, preferredActionKey };
}

function trimTranspositions(memory: EngineMemory, capacity: number): void {
  if (capacity === 0) {
    memory.table.clear();
    return;
  }
  while (memory.table.size > capacity) {
    let victimKey: string | null = null;
    let victim: TranspositionEntry | null = null;
    for (const [key, entry] of memory.table) {
      if (!victim || entry.generation < victim.generation ||
          (entry.generation === victim.generation && entry.depth < victim.depth)) {
        victimKey = key;
        victim = entry;
      }
      if (victim && victim.generation < memory.generation - 2) break;
    }
    if (victimKey === null) break;
    memory.table.delete(victimKey);
  }
}

function storeTransposition(key: string, entry: TranspositionEntry, context: SearchContext): void {
  const capacity = context.options.transpositionCapacity;
  if (capacity === 0) return;
  const previous = context.memory.table.get(key);
  if (previous && previous.solved && !entry.solved) return;
  if (previous && previous.depth > entry.depth && !entry.solved) return;
  if (!previous && context.memory.table.size >= capacity) {
    trimTranspositions(context.memory, Math.max(0, capacity - Math.max(1, Math.floor(capacity / 100))));
  }
  context.memory.table.set(key, entry);
  context.transpositionStores += 1;
}

function countEmpty(board: Board): number {
  let empty = 0;
  for (const row of board) for (const tile of row) if (tile === 0) empty += 1;
  return empty;
}

function canStartExactExtension(state: SearchState, prepared: PreparedActions): boolean {
  return countEmpty(state.board) <= 6 && prepared.actions.length <= 12;
}

function isVolatile(state: SearchState): boolean {
  const mine = fragileShape(state.board, state.current);
  const theirs = fragileShape(state.board, opponentOf(state.current));
  return mine.swans.size > 0 || theirs.swans.size > 0;
}

function enterPath(state: SearchState, context: SearchContext): string | null {
  const key = pathKey(state);
  if ((context.path.get(key) ?? 0) > 0) {
    context.repetitions += 1;
    return null;
  }
  context.path.set(key, 1);
  return key;
}

function leavePath(key: string, context: SearchContext): void {
  context.path.delete(key);
}

function repetitionScore(state: SearchState, context: SearchContext): number {
  // There is no rules-level repetition draw. A small root-relative contempt
  // discourages the engine from manufacturing unresolved arena/game loops,
  // while remaining tiny compared with any tactical or terminal value.
  return state.current === context.rootPlayer ? -REPETITION_CONTEMPT : REPETITION_CONTEMPT;
}

function terminalValue(info: PreparedAction, actor: Player, ply: number): SearchValue | null {
  if (!info.applied.outcome) return null;
  return {
    score: outcomeScore(info.applied.outcome, actor, ply),
    principalVariation: [],
    solved: true
  };
}

function quiescenceChild(
  state: SearchState,
  info: PreparedAction,
  alpha: number,
  beta: number,
  remaining: number,
  ply: number,
  qPly: number,
  context: SearchContext
): SearchValue {
  const terminal = terminalValue(info, state.current, ply + 1);
  if (terminal) return terminal;
  if (info.changesTurn) {
    if (remaining <= 0) {
      return { score: -evaluate(info.applied, info.applied.current, context), principalVariation: [], solved: false };
    }
    const child = quiescence(
      info.applied,
      -beta,
      -alpha,
      remaining - 1,
      ply + 1,
      qPly + 1,
      context,
      false,
      false
    );
    return { score: -child.score, principalVariation: child.principalVariation, solved: child.solved };
  }
  return quiescence(
    info.applied,
    alpha,
    beta,
    remaining,
    ply + 1,
    qPly + 1,
    context,
    true,
    false
  );
}

function quiescence(
  state: SearchState,
  alpha: number,
  beta: number,
  remaining: number,
  ply: number,
  qPly: number,
  context: SearchContext,
  mustCompleteTurn: boolean,
  pathAlreadyEntered: boolean
): SearchValue {
  let entered: string | null = "";
  if (!pathAlreadyEntered) {
    checkForStop(context, true);
    entered = enterPath(state, context);
    if (entered === null) {
      return { score: repetitionScore(state, context), principalVariation: [], solved: false };
    }
  }

  try {
    const standPat = evaluate(state, state.current, context);
    if (qPly >= MAX_QUIESCENCE_ACTIONS || (!mustCompleteTurn && (remaining <= 0 || !isVolatile(state)))) {
      return { score: standPat, principalVariation: [], solved: false };
    }
    let bestScore = mustCompleteTurn ? NEGATIVE_INFINITY : standPat;
    let bestAction: PreparedAction | null = null;
    let bestContinuation: LinithAction[] = [];
    if (!mustCompleteTurn) {
      if (standPat >= beta) return { score: standPat, principalVariation: [], solved: false };
      alpha = Math.max(alpha, standPat);
    }

    const prepared = prepareActions(state, context);
    if (prepared.actions.length === 0) return { score: 0, principalVariation: [], solved: true };
    const actions = orderedCandidates(
      state,
      prepared,
      1,
      ply,
      false,
      null,
      context,
      true,
      mustCompleteTurn
    );
    if (actions.length === 0) return { score: standPat, principalVariation: [], solved: false };

    for (const info of actions) {
      checkForStop(context, false);
      const child = quiescenceChild(state, info, alpha, beta, remaining, ply, qPly, context);
      if (child.score > bestScore ||
          (child.score === bestScore && bestAction !== null && compareKeys(info.key, bestAction.key) < 0)) {
        bestScore = child.score;
        bestAction = info;
        bestContinuation = child.principalVariation;
      }
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) {
        context.cutoffs += 1;
        break;
      }
    }
    if (!bestAction) return { score: standPat, principalVariation: [], solved: false };
    return { score: bestScore, principalVariation: [bestAction.action, ...bestContinuation], solved: false };
  } finally {
    if (!pathAlreadyEntered && entered) leavePath(entered, context);
  }
}

function searchChild(
  state: SearchState,
  info: PreparedAction,
  depth: number,
  alpha: number,
  beta: number,
  ply: number,
  exactRemaining: number,
  context: SearchContext,
  reduction = 0
): SearchValue {
  const terminal = terminalValue(info, state.current, ply + 1);
  if (terminal) return terminal;
  const childDepth = info.changesTurn
    ? Math.max(0, depth - 1 - reduction)
    : depth;
  let result: SearchValue;
  if (info.changesTurn) {
    const child = alphaBeta(info.applied, childDepth, -beta, -alpha, ply + 1, exactRemaining, context, false);
    result = { score: -child.score, principalVariation: child.principalVariation, solved: child.solved };
  } else {
    result = alphaBeta(info.applied, childDepth, alpha, beta, ply + 1, exactRemaining, context, false);
  }
  return result;
}

function alphaBeta(
  state: SearchState,
  requestedDepth: number,
  alpha: number,
  beta: number,
  ply: number,
  requestedExactRemaining: number,
  context: SearchContext,
  pvNode: boolean
): SearchValue {
  checkForStop(context, true);
  const entered = enterPath(state, context);
  if (entered === null) {
    return { score: repetitionScore(state, context), principalVariation: [], solved: false };
  }
  const repetitionsAtEntry = context.repetitions;

  try {
    let depth = requestedDepth;
    let exactRemaining = requestedExactRemaining;
    let preparedAtHorizon: PreparedActions | null = null;
    if (
      depth === 0 && context.options.exactDepth > 0
      && (exactRemaining > 0 || countEmpty(state.board) <= 6)
    ) {
      preparedAtHorizon = prepareActions(state, context);
      if (exactRemaining > 0 || canStartExactExtension(state, preparedAtHorizon)) {
        if (exactRemaining === 0) exactRemaining = context.options.exactDepth;
        if (exactRemaining > 0) {
          depth = 1;
          exactRemaining -= 1;
          context.exactExtensions += 1;
        }
      }
    }
    if (depth === 0) {
      return quiescence(
        state,
        alpha,
        beta,
        context.options.tacticalDepth,
        ply,
        0,
        context,
        false,
        true
      );
    }

    const key = positionKey(state, context);
    // Root scores depend on the engine's game-history contempt. Reuse the TT
    // move for ordering, but recompute its value against the current history.
    const lookup = lookupTransposition(key, depth, alpha, beta, ply, context, ply !== 0);
    if (lookup.value) return lookup.value;
    alpha = lookup.alpha;
    beta = lookup.beta;
    const originalAlpha = alpha;
    const originalBeta = beta;

    const prepared = preparedAtHorizon ?? prepareActions(state, context);
    if (prepared.actions.length === 0) return { score: 0, principalVariation: [], solved: true };
    const actions = orderedCandidates(
      state,
      prepared,
      depth,
      ply,
      pvNode,
      ply === 0 ? context.rootPreferredActionKey ?? lookup.preferredActionKey : lookup.preferredActionKey,
      context
    );
    if (actions.length === 0) return { score: evaluate(state, state.current, context), principalVariation: [], solved: false };

    let bestScore = NEGATIVE_INFINITY;
    let bestAction: PreparedAction | null = null;
    let bestContinuation: LinithAction[] = [];
    let allChildrenSolved = actions.length === prepared.actions.length;
    const staticScore = depth === 1 && ply > 0 ? evaluate(state, state.current, context) : 0;

    for (let index = 0; index < actions.length; index += 1) {
      checkForStop(context, false);
      const info = actions[index];
      if (ply === 0) context.rootActionsSearched.add(info.key);
      const rootPersonality = ply === 0
        ? evaluateVeryHardRootPersonality(
            state,
            info.action,
            info.applied,
            context.rootPlayer,
            context.options.style
          )
        : 0;
      const rootVisits = ply === 0
        ? context.memory.rootVisits.get(boardKey(info.applied)) ?? 0
        : 0;
      const rootAdjustment = rootPersonality - rootVisits * ROOT_HISTORY_PENALTY;

      const canFutilityPrune =
        !pvNode && ply > 0 && depth === 1 && index >= 8 && info.quiet && info.changesTurn &&
        staticScore + 7_500 <= alpha;
      if (canFutilityPrune) {
        context.futilityPrunes += 1;
        allChildrenSolved = false;
        continue;
      }

      const canReduce =
        !pvNode && depth >= 3 && index >= 10 && info.quiet && info.changesTurn && exactRemaining === 0;
      const reduction = canReduce ? 1 : 0;
      if (reduction > 0) context.lateMoveReductions += 1;

      let child: SearchValue;
      if (index === 0) {
        child = searchChild(
          state,
          info,
          depth,
          alpha - rootAdjustment,
          beta - rootAdjustment,
          ply,
          exactRemaining,
          context,
          reduction
        );
      } else {
        child = searchChild(
          state,
          info,
          depth,
          alpha - rootAdjustment,
          alpha + 1 - rootAdjustment,
          ply,
          exactRemaining,
          context,
          reduction
        );
        if (rootAdjustment !== 0) child = { ...child, score: child.score + rootAdjustment };
        if (reduction > 0 && child.score > alpha) {
          context.reSearches += 1;
          child = searchChild(
            state,
            info,
            depth,
            alpha - rootAdjustment,
            alpha + 1 - rootAdjustment,
            ply,
            exactRemaining,
            context,
            0
          );
          if (rootAdjustment !== 0) child = { ...child, score: child.score + rootAdjustment };
        }
        if (child.score > alpha && child.score < beta) {
          context.reSearches += 1;
          child = searchChild(
            state,
            info,
            depth,
            alpha - rootAdjustment,
            beta - rootAdjustment,
            ply,
            exactRemaining,
            context,
            0
          );
          if (rootAdjustment !== 0) child = { ...child, score: child.score + rootAdjustment };
        }
      }
      allChildrenSolved &&= child.solved;

      if (index === 0 && rootAdjustment !== 0) {
        child = { ...child, score: child.score + rootAdjustment };
      }

      if (child.score > bestScore ||
          (child.score === bestScore && bestAction !== null && compareKeys(info.key, bestAction.key) < 0)) {
        bestScore = child.score;
        bestAction = info;
        bestContinuation = child.principalVariation;
      }
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) {
        context.cutoffs += 1;
        rememberCutoff(info, state, depth, ply, context);
        break;
      }
    }

    if (!bestAction) return { score: evaluate(state, state.current, context), principalVariation: [], solved: false };
    const principalVariation = [bestAction.action, ...bestContinuation];
    const bound: TranspositionBound = bestScore <= originalAlpha
      ? "upper"
      : bestScore >= originalBeta
        ? "lower"
        : "exact";
    // `solved` means the bound was derived without a heuristic horizon. A
    // normal alpha-beta cutoff is still a rigorous game-theoretic bound; only
    // selective omission, reduction, futility, repetition contempt, or a
    // heuristic leaf invalidates that proof.
    const solved = allChildrenSolved;
    // A repetition-contempt value depends on the current path, so it must
    // never escape into the path-independent transposition table.
    if (context.repetitions === repetitionsAtEntry) {
      storeTransposition(key, {
        depth,
        score: scoreForTransposition(bestScore, ply),
        bound,
        bestActionKey: bestAction.key,
        principalVariation: clonePrincipalVariation(principalVariation),
        generation: context.memory.generation,
        solved
      }, context);
    }
    return { score: bestScore, principalVariation, solved };
  } finally {
    leavePath(entered, context);
  }
}

function makeDiagnostics(
  context: SearchContext,
  completedDepth: number,
  attemptedDepth: number,
  stopReason: VeryHardStopReason,
  principalVariation: readonly LinithAction[],
  elapsedMs: number,
  exactSolved: boolean,
  forcedMove: boolean
): VeryHardDiagnostics {
  return {
    completedDepth,
    attemptedDepth,
    nodes: context.nodes,
    elapsedMs,
    generatedActions: context.generatedActions,
    evaluatedPositions: context.evaluatedPositions,
    transpositionHits: context.transpositionHits,
    transpositionStores: context.transpositionStores,
    cutoffs: context.cutoffs,
    reSearches: context.reSearches,
    aspirationReSearches: context.aspirationReSearches,
    selectivePrunes: context.selectivePrunes,
    futilityPrunes: context.futilityPrunes,
    lateMoveReductions: context.lateMoveReductions,
    repetitions: context.repetitions,
    exactExtensions: context.exactExtensions,
    exactSolved,
    forcedMove,
    rootActions: context.rootPrepared.actions.length,
    rootActionsSearched: context.rootActionsSearched.size,
    timedOut: stopReason === "deadline",
    nodeBudgetReached: stopReason === "node-budget",
    aborted: stopReason === "aborted",
    stopReason,
    principalVariation: clonePrincipalVariation(principalVariation)
  };
}

function decayHistory(memory: EngineMemory): void {
  for (const [key, value] of memory.history) {
    const decayed = Math.floor(value * 0.75);
    if (decayed === 0) memory.history.delete(key);
    else memory.history.set(key, decayed);
  }
}

function runSearch(
  input: SearchState,
  positionHash: string,
  options: VeryHardSearchOptions,
  memory: EngineMemory,
  requestId?: number
): VeryHardSearchResult {
  validateState(input);
  const state: SearchState = {
    board: cloneBoard(input.board),
    current: input.current,
    movesLeft: input.movesLeft
  };
  const normalized = normalizeOptions(options);
  memory.generation += 1;
  decayHistory(memory);
  trimTranspositions(memory, normalized.transpositionCapacity);
  const startedAt = normalized.now();
  const rootKey = boardKey(state);
  memory.rootVisits.set(rootKey, (memory.rootVisits.get(rootKey) ?? 0) + 1);
  const bookHint = lookupOpeningBookAction(state);
  const placeholder: PreparedActions = { actions: [], allLegalCount: 0 };
  const context: SearchContext = {
    options: normalized,
    memory,
    rootPlayer: state.current,
    rootKey,
    rootPreferredActionKey: bookHint ? actionKey(bookHint.action) : null,
    rootPrepared: placeholder,
    startedAt,
    deadline: Math.min(normalized.deadlineMs, startedAt + normalized.budgetMs),
    nodes: 0,
    generatedActions: 0,
    evaluatedPositions: 0,
    transpositionHits: 0,
    transpositionStores: 0,
    cutoffs: 0,
    reSearches: 0,
    aspirationReSearches: 0,
    selectivePrunes: 0,
    futilityPrunes: 0,
    lateMoveReductions: 0,
    repetitions: 0,
    exactExtensions: 0,
    rootActionsSearched: new Set(),
    prepared: new Map(),
    path: new Map()
  };

  context.rootPrepared = prepareActions(state, context);
  const forcedDefense = classifyRootDefenses(state, context.rootPrepared, context);
  const rootActions = context.rootPrepared.actions;
  const rootOrder = orderedCandidates(
    state,
    context.rootPrepared,
    1,
    0,
    true,
    context.rootPreferredActionKey,
    context
  );
  let bestAction = rootOrder[0]?.action ?? null;
  let bestScore = rootOrder[0]
    ? rootOrder[0].terminalScore ?? rootOrder[0].staticScore ?? evaluate(rootOrder[0].applied, state.current, context)
    : 0;
  let principalVariation: LinithAction[] = bestAction ? [bestAction] : [];
  let completedDepth = 0;
  let attemptedDepth = 0;
  let exactSolved = false;
  let forcedMove = false;
  let stopReason: VeryHardStopReason = rootActions.length === 0 ? "no-legal-actions" : "max-depth";

  const immediateWins = rootActions.filter((info) => info.terminalScore !== null && info.terminalScore > MATE_BAND);
  const forced = immediateWins[0] ?? forcedDefense ?? (rootActions.length === 1 ? rootActions[0] : null);
  if (forced) {
    forcedMove = true;
    bestAction = forced.action;
    if (forced.terminalScore !== null) bestScore = forced.terminalScore;
    else {
      if (forced.staticScore === null) forced.staticScore = evaluate(forced.applied, state.current, context);
      bestScore = forced.staticScore;
    }
    principalVariation = [forced.action];
    completedDepth = 1;
    attemptedDepth = 1;
    exactSolved = forced.terminalScore !== null || rootActions.length === 0;
    context.rootActionsSearched.add(forced.key);
  } else if (rootActions.length > 0) {
    let previousScore = bestScore;
    const rootExactRemaining = canStartExactExtension(state, context.rootPrepared)
      ? normalized.exactDepth
      : 0;
    for (let depth = 1; depth <= normalized.maxDepth; depth += 1) {
      attemptedDepth = depth;
      try {
        let alpha = NEGATIVE_INFINITY;
        let beta = POSITIVE_INFINITY;
        if (completedDepth > 0 && Math.abs(previousScore) < MATE_BAND) {
          const window = Math.max(ASPIRATION_MINIMUM, Math.floor(Math.abs(previousScore) * 0.08));
          alpha = Math.max(NEGATIVE_INFINITY, previousScore - window);
          beta = Math.min(POSITIVE_INFINITY, previousScore + window);
        }
        let iteration = alphaBeta(state, depth, alpha, beta, 0, rootExactRemaining, context, true);
        if (iteration.score <= alpha || iteration.score >= beta) {
          context.aspirationReSearches += 1;
          iteration = alphaBeta(
            state,
            depth,
            NEGATIVE_INFINITY,
            POSITIVE_INFINITY,
            0,
            rootExactRemaining,
            context,
            true
          );
        }
        const iterationAction = iteration.principalVariation[0];
        if (iterationAction) {
          bestAction = iterationAction;
          bestScore = iteration.score;
          previousScore = iteration.score;
          principalVariation = iteration.principalVariation;
        }
        completedDepth = depth;
        exactSolved = iteration.solved;
        if (iteration.solved || Math.abs(iteration.score) > MATE_BAND) break;
      } catch (error) {
        if (!(error instanceof SearchInterrupted)) throw error;
        stopReason = error.reason;
        break;
      }
    }
  }

  const elapsedMs = Math.max(0, normalized.now() - startedAt);
  const diagnostics = makeDiagnostics(
    context,
    completedDepth,
    attemptedDepth,
    stopReason,
    principalVariation,
    elapsedMs,
    exactSolved,
    forcedMove
  );
  return {
    requestId,
    positionHash,
    action: bestAction ? cloneAction(bestAction) : null,
    score: bestScore,
    ...diagnostics,
    diagnostics
  };
}

class ReusableVeryHardSearchEngine implements VeryHardSearchEngine {
  private readonly memory: EngineMemory = {
    table: new Map(),
    history: new Map(),
    killers: new Map(),
    rootVisits: new Map(),
    generation: 0
  };

  get transpositionSize(): number {
    return this.memory.table.size;
  }

  search(state: SearchState, options: VeryHardSearchOptions = {}): VeryHardSearchResult {
    return runSearch(state, boardKey(state), options, this.memory);
  }

  searchRequest(request: VeryHardSearchRequest): VeryHardSearchResult {
    const { requestId, positionHash, board, current, movesLeft, ...options } = request;
    return runSearch({ board, current, movesLeft }, positionHash, options, this.memory, requestId);
  }

  clear(): void {
    this.memory.table.clear();
    this.memory.history.clear();
    this.memory.killers.clear();
    this.memory.rootVisits.clear();
    this.memory.generation = 0;
  }
}

export function createVeryHardSearchEngine(): VeryHardSearchEngine {
  return new ReusableVeryHardSearchEngine();
}

export function chooseVeryHardAction(
  state: SearchState,
  options: VeryHardSearchOptions = {}
): VeryHardSearchResult {
  return createVeryHardSearchEngine().search(state, options);
}

export function searchVeryHard(request: VeryHardSearchRequest): VeryHardSearchResult {
  return createVeryHardSearchEngine().searchRequest(request);
}
