import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createNativeVeryHardCoreSync } from "../native/very-hard/node-adapter";
import { linithAI } from "../src/renderer/game/ai";
import { AI_STYLE_IDS, type AiStyleId } from "../src/renderer/game/aiStyles";
import {
  EMPTY,
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
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  evaluateVeryHardRootPersonality,
  VERY_HARD_ROOT_PERSONALITY_LIMIT
} from "../src/renderer/game/veryHard/evaluate";
import type { NativeVeryHardCore } from "../src/renderer/game/veryHard/native/core";

type Difficulty = "easy" | "medium" | "hard" | "very-hard";

interface Selection {
  action: LinithAction | null;
  completedDepth: number | null;
  personalityBonus: number | null;
  objectiveRegret: number | null;
  diagnosticsValid: boolean;
}

interface CombinationAudit {
  decisions: number;
  legal: number;
  deterministic: number;
  immutable: number;
  immediateWins: number;
  uniqueDefenses: number;
  divergenceFromDoctrinal: number;
  preferenceWins: number;
  preferenceTies: number;
  preferenceLosses: number;
  averagePreferenceGain: number;
  averageCompletedDepth: number | null;
  shallowSearches: number;
  maximumAbsolutePersonalityBonus: number | null;
  maximumObjectiveRegret: number | null;
  validDiagnostics: number | null;
  actionTypes: Record<string, number>;
}

const DIFFICULTIES: readonly Difficulty[] = ["easy", "medium", "hard", "very-hard"];
const SAMPLED_PLIES = [4, 18, 30] as const;
const NODE_BUDGET = 10_000;
const corpus = reachableCorpus();
const issues: string[] = [];
const artifact = inspectArtifacts();
const characterSignals = measureCharacterSignals();
const combinations: Record<string, CombinationAudit> = {};
const pairwiseDivergence: Record<string, number> = {};
const originalWindow = globalThis.window;
const originalRandom = Math.random;
const startedAt = performance.now();

if (!artifact.provenanceMatches) {
  issues.push(
    `opening book declares ${artifact.declaredWasmSha256 ?? "no WASM fingerprint"}; `
    + `active WASM is sha256-${artifact.wasmSha256}`
  );
} else {
  try {
    for (const difficulty of DIFFICULTIES) qualifyDifficulty(difficulty);
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
}

for (const style of AI_STYLE_IDS.slice(1)) {
  if (characterSignals[style].spread <= 0) issues.push(`${style} has no action-aware character signal`);
}
if (characterSignals.doctrinal.minimum !== 0 || characterSignals.doctrinal.maximum !== 0) {
  issues.push("Doctrinal has a non-zero character signal");
}

const report = {
  kind: "linith-ai-matrix-qualification",
  formatVersion: 1,
  passed: issues.length === 0,
  corpus: {
    states: corpus.length,
    starts: 4,
    sampledPlies: SAMPLED_PLIES,
    repeatedSelections: 2,
    veryHardNodeBudget: NODE_BUDGET,
    veryHardMaxTurnDepth: 2
  },
  coverage: {
    difficulties: DIFFICULTIES,
    personalities: AI_STYLE_IDS,
    combinations: DIFFICULTIES.length * AI_STYLE_IDS.length,
    matchedDecisions: corpus.length * DIFFICULTIES.length * AI_STYLE_IDS.length,
    tacticalDecisions: 2 * DIFFICULTIES.length * AI_STYLE_IDS.length
  },
  elapsedMs: performance.now() - startedAt,
  artifact,
  characterSignals,
  combinations,
  pairwiseDivergence,
  issues
};

console.log(JSON.stringify(report, null, 2));
if (issues.length > 0) process.exitCode = 1;

function qualifyDifficulty(difficulty: Difficulty): void {
  const actionsByStyle = new Map<AiStyleId, LinithAction[]>();

  for (const style of AI_STYLE_IDS) {
    const firstCore = difficulty === "very-hard" ? createNativeVeryHardCoreSync() : null;
    const repeatCore = difficulty === "very-hard" ? createNativeVeryHardCoreSync() : null;
    const actions: LinithAction[] = [];
    let legal = 0;
    let deterministic = 0;
    let immutable = 0;
    let depthTotal = 0;
    let shallowSearches = 0;
    let maximumAbsolutePersonalityBonus = 0;
    let maximumObjectiveRegret = 0;
    let validDiagnostics = 0;
    const actionTypes: Record<string, number> = { stone: 0, swan: 0, move: 0, push: 0 };

    corpus.forEach((state, stateIndex) => {
      const seed = mixSeed(0x51a700, DIFFICULTIES.indexOf(difficulty), stateIndex);
      const before = JSON.stringify(state.board);
      const first = selectAction(state, difficulty, style, seed, firstCore);
      const afterFirst = JSON.stringify(state.board);
      const repeat = selectAction(state, difficulty, style, seed, repeatCore);
      if (afterFirst === before && JSON.stringify(state.board) === before) immutable += 1;
      if (actionIdentity(first.action) === actionIdentity(repeat.action)) deterministic += 1;
      if (first.action) {
        const applied = applyAction(state, first.action);
        if (applied) {
          legal += 1;
          actions.push(first.action);
          actionTypes[first.action.type] += 1;
        }
      }
      if (difficulty === "very-hard") {
        depthTotal += first.completedDepth ?? 0;
        if ((first.completedDepth ?? 0) < 1) shallowSearches += 1;
        maximumAbsolutePersonalityBonus = Math.max(
          maximumAbsolutePersonalityBonus,
          Math.abs(first.personalityBonus ?? Infinity)
        );
        maximumObjectiveRegret = Math.max(maximumObjectiveRegret, first.objectiveRegret ?? Infinity);
        if (first.diagnosticsValid) validDiagnostics += 1;
      }
    });

    const tactical = qualifyTactics(difficulty, style);
    const key = `${difficulty}:${style}`;
    combinations[key] = {
      decisions: corpus.length,
      legal,
      deterministic,
      immutable,
      immediateWins: tactical.immediateWins,
      uniqueDefenses: tactical.uniqueDefenses,
      divergenceFromDoctrinal: 0,
      preferenceWins: 0,
      preferenceTies: 0,
      preferenceLosses: 0,
      averagePreferenceGain: 0,
      averageCompletedDepth: difficulty === "very-hard" ? depthTotal / corpus.length : null,
      shallowSearches,
      maximumAbsolutePersonalityBonus: difficulty === "very-hard" ? maximumAbsolutePersonalityBonus : null,
      maximumObjectiveRegret: difficulty === "very-hard" ? maximumObjectiveRegret : null,
      validDiagnostics: difficulty === "very-hard" ? validDiagnostics : null,
      actionTypes
    };
    actionsByStyle.set(style, actions);

    if (legal !== corpus.length) issues.push(`${key} returned ${corpus.length - legal} illegal or null corpus actions`);
    if (deterministic !== corpus.length) issues.push(`${key} was nondeterministic on ${corpus.length - deterministic} corpus states`);
    if (immutable !== corpus.length) issues.push(`${key} mutated ${corpus.length - immutable} caller boards`);
    if (tactical.immediateWins !== 1) issues.push(`${key} missed the immediate win`);
    if (tactical.uniqueDefenses !== 1) issues.push(`${key} missed the unique defense`);
    if (difficulty === "very-hard" && shallowSearches > 0) issues.push(`${key} had ${shallowSearches} shallow corpus searches`);
    if (difficulty === "very-hard" && maximumAbsolutePersonalityBonus > VERY_HARD_ROOT_PERSONALITY_LIMIT) {
      issues.push(`${key} exceeded the root personality limit`);
    }
    if (difficulty === "very-hard" && maximumObjectiveRegret > VERY_HARD_ROOT_PERSONALITY_LIMIT * 2) {
      issues.push(`${key} exceeded the objective-regret limit`);
    }
    if (difficulty === "very-hard" && validDiagnostics !== corpus.length) {
      issues.push(`${key} failed ${corpus.length - validDiagnostics} native/TypeScript personality parity checks`);
    }
    if (difficulty === "very-hard" && style === "doctrinal" &&
        (maximumAbsolutePersonalityBonus !== 0 || maximumObjectiveRegret !== 0)) {
      issues.push("very-hard:doctrinal did not remain an exact zero-bias search");
    }
  }

  const doctrinal = actionsByStyle.get("doctrinal")!;
  for (const style of AI_STYLE_IDS) {
    const actions = actionsByStyle.get(style)!;
    const audit = combinations[`${difficulty}:${style}`];
    const differences = actions.filter((action, index) => actionIdentity(action) !== actionIdentity(doctrinal[index])).length;
    audit.divergenceFromDoctrinal = differences / corpus.length;

    if (style !== "doctrinal") {
      let preferenceTotal = 0;
      let comparisons = 0;
      actions.forEach((action, index) => {
        const selectedPreference = successorPreference(corpus[index], action, style);
        const doctrinalPreference = successorPreference(corpus[index], doctrinal[index], style);
        if (selectedPreference === null || doctrinalPreference === null) return;
        const gain = selectedPreference - doctrinalPreference;
        preferenceTotal += gain;
        comparisons += 1;
        if (gain > 0) audit.preferenceWins += 1;
        else if (gain < 0) audit.preferenceLosses += 1;
        else audit.preferenceTies += 1;
      });
      audit.averagePreferenceGain = comparisons > 0 ? preferenceTotal / comparisons : 0;
    }
  }

  for (let left = 0; left < AI_STYLE_IDS.length; left += 1) {
    for (let right = left + 1; right < AI_STYLE_IDS.length; right += 1) {
      const leftStyle = AI_STYLE_IDS[left];
      const rightStyle = AI_STYLE_IDS[right];
      const a = actionsByStyle.get(leftStyle)!;
      const b = actionsByStyle.get(rightStyle)!;
      const divergence = a.filter((action, index) => actionIdentity(action) !== actionIdentity(b[index])).length / corpus.length;
      pairwiseDivergence[`${difficulty}:${leftStyle}:${rightStyle}`] = divergence;
    }
  }
}

function qualifyTactics(difficulty: Difficulty, style: AiStyleId): { immediateWins: number; uniqueDefenses: number } {
  const winState = immediateWinState();
  const win = selectAction(
    winState,
    difficulty,
    style,
    mixSeed(0x0a11ce, DIFFICULTIES.indexOf(difficulty), AI_STYLE_IDS.indexOf(style)),
    difficulty === "very-hard" ? createNativeVeryHardCoreSync() : null
  ).action;
  const expectedOutcome = winState.current === 1 ? "sun" : "moon";
  const immediateWins = win && applyAction(winState, win)?.outcome === expectedOutcome ? 1 : 0;

  const defenseState = forcedDefenseState();
  const safe = actionsAvoidingImmediateLoss(defenseState);
  if (safe.length !== 1) throw new Error(`Forced-defense fixture has ${safe.length} safe actions instead of one.`);
  const defense = selectAction(
    defenseState,
    difficulty,
    style,
    mixSeed(0xdefe115e, DIFFICULTIES.indexOf(difficulty), AI_STYLE_IDS.indexOf(style)),
    difficulty === "very-hard" ? createNativeVeryHardCoreSync() : null
  ).action;
  const uniqueDefenses = defense && actionIdentity(defense) === actionIdentity(safe[0]) ? 1 : 0;
  return { immediateWins, uniqueDefenses };
}

function selectAction(
  state: SearchState,
  difficulty: Difficulty,
  style: AiStyleId,
  seed: number,
  nativeCore: NativeVeryHardCore | null
): Selection {
  globalThis.window = { linithGetStyle: () => style } as Window & typeof globalThis;
  if (difficulty === "very-hard") {
    const core = nativeCore ?? createNativeVeryHardCoreSync();
    const result = core.search(state, {
      style,
      maxTurnDepth: 2,
      tacticalDepth: 1,
      exactDepth: 0,
      nodeBudget: NODE_BUDGET,
      budgetMs: 0
    });
    const applied = result.action ? applyAction(state, result.action) : null;
    const expectedPersonality = result.action && applied
      ? evaluateVeryHardRootPersonality(state, result.action, applied, state.current, style)
      : 0;
    return {
      action: result.action,
      completedDepth: result.completedTurnDepth,
      personalityBonus: result.personalityBonus,
      objectiveRegret: result.objectiveRegret,
      diagnosticsValid: result.score === result.objectiveScore + result.personalityBonus
        && result.personalityBonus === expectedPersonality
    };
  }
  Math.random = seededRandom(seed);
  return {
    action: linithAI(state.board, state.current, difficulty),
    completedDepth: null,
    personalityBonus: null,
    objectiveRegret: null,
    diagnosticsValid: true
  };
}

function successorPreference(state: SearchState, action: LinithAction, style: AiStyleId): number | null {
  const next = applyAction(state, action);
  if (!next) return null;
  return evaluateVeryHardRootPersonality(state, action, next, state.current, style);
}

function measureCharacterSignals(): Record<AiStyleId, { minimum: number; maximum: number; spread: number }> {
  return Object.fromEntries(AI_STYLE_IDS.map((style) => {
    const scores: number[] = [];
    for (const state of corpus) {
      for (const action of generateLegalActions(state)) {
        const next = applyAction(state, action);
        if (next) scores.push(evaluateVeryHardRootPersonality(state, action, next, state.current, style));
      }
    }
    const minimum = Math.min(...scores);
    const maximum = Math.max(...scores);
    return [style, { minimum, maximum, spread: maximum - minimum }];
  })) as Record<AiStyleId, { minimum: number; maximum: number; spread: number }>;
}

function reachableCorpus(): SearchState[] {
  const starts: Array<readonly [readonly [number, number], readonly [number, number]]> = [
    [[4, 4], [6, 6]],
    [[5, 5], [3, 7]],
    [[3, 3], [6, 5]],
    [[6, 3], [3, 6]]
  ];
  const result: SearchState[] = [];
  starts.forEach(([sun, moon], startIndex) => {
    let state = initialState(sun, moon);
    const random = seededRandom(0x51a700 + startIndex * 0x9e37);
    for (let ply = 0; ply <= Math.max(...SAMPLED_PLIES); ply += 1) {
      if ((SAMPLED_PLIES as readonly number[]).includes(ply)) result.push(cloneState(state));
      const actions = generateLegalActions(state);
      if (actions.length === 0) break;
      const developing = actions.filter((action) => action.type === "swan");
      const candidates = ply < 12 && developing.length > 0 ? developing : actions;
      const offset = Math.floor(random() * candidates.length);
      let nextState: SearchState | null = null;
      for (let index = 0; index < candidates.length; index += 1) {
        const next = applyAction(state, candidates[(offset + index) % candidates.length]);
        if (!next || next.outcome) continue;
        nextState = { board: next.board, current: next.current, movesLeft: next.movesLeft };
        break;
      }
      if (!nextState) break;
      state = nextState;
    }
  });
  return result;
}

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
  for (const [r, c] of [[0, 2], [2, 0], [9, 0], [9, 1], [9, 2]] as Array<[number, number]>) {
    board[r][c] = FROZEN_SUN;
  }
  board[5][5] = SWAN_MOON;
  return { board, current: 1, movesLeft: 1 };
}

function actionsAvoidingImmediateLoss(state: SearchState): LinithAction[] {
  const opponent = other(state.current);
  return generateLegalActions(state).filter((action) => {
    const next = applyAction(state, action);
    if (!next || next.outcome) return false;
    const expected = opponent === 1 ? "sun" : "moon";
    return !generateLegalActions(asSearchState(next))
      .some((reply) => applyAction(asSearchState(next), reply)?.outcome === expected);
  });
}

function initialState(sun: readonly [number, number], moon: readonly [number, number]): SearchState {
  const board = emptyBoard();
  board[sun[0]][sun[1]] = SWAN_SUN;
  board[moon[0]][moon[1]] = SWAN_MOON;
  return { board, current: 2, movesLeft: 1 };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
}

function asSearchState(state: Pick<SearchState, "board" | "current" | "movesLeft">): SearchState {
  return { board: state.board, current: state.current, movesLeft: state.movesLeft };
}

function actionIdentity(action: LinithAction | null | undefined): string {
  return action ? actionKey(action) : "null";
}

function other(player: Player): Player {
  return player === 1 ? 2 : 1;
}

function mixSeed(...values: number[]): number {
  let mixed = 0x9e3779b9;
  for (const value of values) {
    mixed ^= value >>> 0;
    mixed = Math.imul(mixed ^ (mixed >>> 16), 0x21f0aaad);
    mixed = Math.imul(mixed ^ (mixed >>> 15), 0x735a2d97);
    mixed ^= mixed >>> 15;
  }
  return mixed >>> 0;
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 0x1_0000_0000;
  };
}

function inspectArtifacts(): {
  wasmSha256: string;
  openingBookSha256: string;
  declaredWasmSha256: string | null;
  provenanceMatches: boolean;
} {
  const wasm = readFileSync(resolve("src/renderer/game/veryHard/native/linith-core.wasm"));
  const openingBook = readFileSync(resolve("src/renderer/game/veryHard/openingBookData.ts"));
  const wasmSha256 = createHash("sha256").update(wasm).digest("hex");
  const openingBookText = openingBook.toString("utf8");
  const declaredWasmSha256 = openingBookText.match(/artifactFingerprint":"sha256-([a-f0-9]{64})"/)?.[1] ?? null;
  return {
    wasmSha256,
    openingBookSha256: createHash("sha256").update(openingBook).digest("hex"),
    declaredWasmSha256,
    provenanceMatches: declaredWasmSha256 === wasmSha256
  };
}
