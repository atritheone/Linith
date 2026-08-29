import { linithAI } from "../src/renderer/game/ai";
import {
  EMPTY,
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
import { chooseVeryHardAction, type VeryHardStopReason } from "../src/renderer/game/veryHard";

interface Config {
  nodeBudget: number;
  budgetMs: number;
  maxDepth: number;
  games: number;
  maxActions: number;
  style: string;
  minScore: number | null;
}

interface ArenaResult {
  winner: Player | null;
  drawReason: "terminal" | "repetition" | "action-limit" | null;
  invalid: boolean;
  actions: number;
  veryHardNodes: number;
  veryHardDepth: number;
  veryHardSearches: number;
  veryHardElapsedMs: number;
  stops: Partial<Record<VeryHardStopReason, number>>;
}

const config = readConfig(process.argv.slice(2));
const originalWindow = globalThis.window;
const originalRandom = Math.random;
globalThis.window = { linithGetStyle: () => config.style } as Window & typeof globalThis;

try {
  const benchmark = benchmarkSearch(config);
  const arena = runArena(config);
  console.log(JSON.stringify({ kind: "very-hard-benchmark", config, ...benchmark }));
  console.log(JSON.stringify({ kind: "very-hard-vs-hard-arena", config, ...arena }));
  if (config.minScore !== null && Number(arena.veryHardScoreRate) < config.minScore) {
    console.error(
      `Very Hard score ${arena.veryHardScoreRate} did not meet required score ${config.minScore}.`
    );
    process.exitCode = 1;
  }
} finally {
  Math.random = originalRandom;
  globalThis.window = originalWindow;
}

function benchmarkSearch(settings: Config): Record<string, unknown> {
  const positions = benchmarkPositions();
  let nodes = 0;
  let elapsedMs = 0;
  let completedDepth = 0;
  let transpositionHits = 0;
  let cutoffs = 0;
  let deterministic = true;
  let legal = true;
  const stops: Partial<Record<VeryHardStopReason, number>> = {};

  for (const state of positions) {
    const options = {
      style: settings.style,
      budgetMs: settings.budgetMs,
      nodeBudget: settings.nodeBudget,
      maxDepth: settings.maxDepth
    };
    const first = chooseVeryHardAction(state, options);
    // A node cap, unlike a wall-clock cap, makes these two searches a genuine
    // reproducibility check rather than a timing coincidence.
    const deterministicOptions = { ...options, budgetMs: Infinity };
    const deterministicFirst = chooseVeryHardAction(state, deterministicOptions);
    const repeat = chooseVeryHardAction(state, deterministicOptions);
    nodes += first.nodes;
    elapsedMs += first.elapsedMs;
    completedDepth += first.completedDepth;
    transpositionHits += first.transpositionHits;
    cutoffs += first.cutoffs;
    stops[first.stopReason] = (stops[first.stopReason] ?? 0) + 1;
    deterministic &&= sameAction(deterministicFirst.action, repeat.action) &&
      deterministicFirst.score === repeat.score &&
      deterministicFirst.completedDepth === repeat.completedDepth &&
      deterministicFirst.nodes === repeat.nodes;
    legal &&= first.action !== null && applyAction(state, first.action) !== null;
  }

  return {
    positions: positions.length,
    deterministic,
    legal,
    totalNodes: nodes,
    totalElapsedMs: round(elapsedMs),
    nodesPerSecond: Math.round(nodes / Math.max(0.001, elapsedMs / 1000)),
    averageCompletedDepth: round(completedDepth / positions.length),
    transpositionHits,
    cutoffs,
    stops
  };
}

function runArena(settings: Config): Record<string, unknown> {
  const openings: Array<[[number, number], [number, number]]> = [
    [[4, 4], [6, 6]],
    [[5, 5], [3, 7]],
    [[3, 3], [6, 5]],
    [[6, 3], [3, 6]]
  ];
  const summary = {
    veryHardWins: 0,
    hardWins: 0,
    draws: 0,
    terminalDraws: 0,
    repetitionDraws: 0,
    actionLimitDraws: 0,
    invalid: 0,
    actions: 0,
    veryHardNodes: 0,
    veryHardDepth: 0,
    veryHardSearches: 0,
    veryHardElapsedMs: 0,
    stops: {} as Partial<Record<VeryHardStopReason, number>>
  };

  for (let game = 0; game < settings.games; game += 1) {
    // Adjacent games use the exact same opening and seed with the engines on
    // opposite colours, preventing an opening imbalance from masquerading as
    // playing strength.
    const pair = Math.floor(game / 2);
    const opening = openings[pair % openings.length];
    const veryHardSide = (game % 2 === 0 ? 1 : 2) as Player;
    const result = playGame(opening, veryHardSide, settings, 0x51a700 + pair * 997);
    summary.actions += result.actions;
    summary.veryHardNodes += result.veryHardNodes;
    summary.veryHardDepth += result.veryHardDepth;
    summary.veryHardSearches += result.veryHardSearches;
    summary.veryHardElapsedMs += result.veryHardElapsedMs;
    summary.invalid += Number(result.invalid);
    mergeStops(summary.stops, result.stops);
    if (result.winner === null) {
      summary.draws += 1;
      if (result.drawReason === "terminal") summary.terminalDraws += 1;
      else if (result.drawReason === "repetition") summary.repetitionDraws += 1;
      else if (result.drawReason === "action-limit") summary.actionLimitDraws += 1;
    }
    else if (result.winner === veryHardSide) summary.veryHardWins += 1;
    else summary.hardWins += 1;
  }

  return {
    games: settings.games,
    veryHardWins: summary.veryHardWins,
    hardWins: summary.hardWins,
    draws: summary.draws,
    terminalDraws: summary.terminalDraws,
    repetitionDraws: summary.repetitionDraws,
    actionLimitDraws: summary.actionLimitDraws,
    veryHardScore: round(summary.veryHardWins + summary.draws / 2),
    veryHardScoreRate: (summary.veryHardWins + summary.draws / 2) / settings.games,
    invalid: summary.invalid,
    averageActions: round(summary.actions / settings.games),
    veryHardSearches: summary.veryHardSearches,
    veryHardNodes: summary.veryHardNodes,
    veryHardElapsedMs: round(summary.veryHardElapsedMs),
    veryHardNodesPerSecond: Math.round(
      summary.veryHardNodes / Math.max(0.001, summary.veryHardElapsedMs / 1000)
    ),
    averageVeryHardDepth: round(summary.veryHardDepth / Math.max(1, summary.veryHardSearches)),
    stops: summary.stops
  };
}

function playGame(
  opening: [[number, number], [number, number]],
  veryHardSide: Player,
  settings: Config,
  seed: number
): ArenaResult {
  const board = emptyBoard();
  board[opening[0][0]][opening[0][1]] = SWAN_SUN;
  board[opening[1][0]][opening[1][1]] = SWAN_MOON;
  let state: SearchState = { board, current: 2, movesLeft: 1 };
  const repetitions = new Map<string, number>();
  const result: ArenaResult = {
    winner: null,
    drawReason: null,
    invalid: false,
    actions: 0,
    veryHardNodes: 0,
    veryHardDepth: 0,
    veryHardSearches: 0,
    veryHardElapsedMs: 0,
    stops: {}
  };

  for (let actionNumber = 0; actionNumber < settings.maxActions; actionNumber += 1) {
    let selected: LinithAction | null;
    if (state.current === veryHardSide) {
      const search = chooseVeryHardAction(state, {
        style: settings.style,
        budgetMs: settings.budgetMs,
        nodeBudget: settings.nodeBudget,
        maxDepth: settings.maxDepth
      });
      selected = search.action;
      result.veryHardNodes += search.nodes;
      result.veryHardDepth += search.completedDepth;
      result.veryHardElapsedMs += search.elapsedMs;
      result.veryHardSearches += 1;
      result.stops[search.stopReason] = (result.stops[search.stopReason] ?? 0) + 1;
    } else {
      Math.random = seededRandom(seed + actionNumber * 31 + state.current);
      selected = linithAI(state.board, state.current, "hard");
    }

    if (!selected) {
      result.invalid = true;
      result.winner = other(state.current);
      result.actions = actionNumber;
      return result;
    }
    const next = applyAction(state, selected);
    if (!next) {
      result.invalid = true;
      result.winner = other(state.current);
      result.actions = actionNumber;
      return result;
    }
    result.actions = actionNumber + 1;
    if (next.outcome) {
      result.winner = next.outcome === "draw" ? null : next.outcome === "sun" ? 1 : 2;
      result.drawReason = next.outcome === "draw" ? "terminal" : null;
      return result;
    }

    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
    const key = boardKey(state);
    const count = (repetitions.get(key) ?? 0) + 1;
    repetitions.set(key, count);
    if (count >= 3) {
      result.drawReason = "repetition";
      return result;
    }
  }
  result.drawReason = "action-limit";
  return result;
}

function benchmarkPositions(): SearchState[] {
  const first = emptyBoard();
  first[4][4] = SWAN_SUN;
  first[6][6] = SWAN_MOON;

  const second: Board = [
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,2,0], [0,0,0,0,0,0,0,0,2,0],
    [0,0,0,1,1,0,2,0,0,0], [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,2,0], [0,0,0,0,0,1,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]
  ];

  const random = seededRandom(0xc0decafe);
  let cursor: SearchState = { board: first, current: 2, movesLeft: 1 };
  for (let step = 0; step < 10; step += 1) {
    const actions = generateLegalActions(cursor);
    const next = applyAction(cursor, actions[Math.floor(random() * actions.length)]);
    if (!next || next.outcome) break;
    cursor = { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }
  return [
    { board: first, current: 2, movesLeft: 1 },
    { board: second, current: 1, movesLeft: 1 },
    cursor
  ];
}

function readConfig(args: string[]): Config {
  const value = (name: string, fallback: string): string =>
    args.find((arg) => arg.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
  const positiveInteger = (name: string, fallback: number): number => {
    const parsed = Number(value(name, String(fallback)));
    if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`--${name} must be a positive integer`);
    return parsed;
  };
  return {
    nodeBudget: positiveInteger("node-budget", 2_000),
    budgetMs: positiveInteger("budget-ms", 30_000),
    maxDepth: positiveInteger("depth", 4),
    games: positiveInteger("games", 4),
    maxActions: positiveInteger("max-actions", 120),
    style: value("style", "doctrinal"),
    minScore: optionalRate("min-score")
  };

  function optionalRate(name: string): number | null {
    const raw = args.find((arg) => arg.startsWith(`--${name}=`))?.slice(name.length + 3);
    if (raw === undefined) return null;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
      throw new Error(`--${name} must be between 0 and 1`);
    }
    return parsed;
  }
}

function mergeStops(
  target: Partial<Record<VeryHardStopReason, number>>,
  source: Partial<Record<VeryHardStopReason, number>>
): void {
  for (const [reason, count] of Object.entries(source) as Array<[VeryHardStopReason, number]>) {
    target[reason] = (target[reason] ?? 0) + count;
  }
}

function sameAction(left: LinithAction | null, right: LinithAction | null): boolean {
  return left === null || right === null ? left === right : actionKey(left) === actionKey(right);
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function other(player: Player): Player {
  return player === 1 ? 2 : 1;
}

function round(value: number): number {
  return Number(value.toFixed(2));
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}
