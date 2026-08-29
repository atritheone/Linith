import { performance } from "node:perf_hooks";
import { createNativeVeryHardCoreSync } from "../native/very-hard/node-adapter";
import { AI_STYLE_IDS, type AiStyleId } from "../src/renderer/game/aiStyles";
import { EMPTY, SWAN_MOON, SWAN_SUN, type Board } from "../src/renderer/game/encirclement";
import {
  actionKey,
  applyAction,
  generateLegalActions,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  evaluateVeryHardRootPersonality,
  explainVeryHardPosition,
  VERY_HARD_ROOT_PERSONALITY_LIMIT
} from "../src/renderer/game/veryHard/evaluate";

interface StyleAudit {
  decisions: number;
  legal: number;
  divergenceFromDoctrinal: number;
  averageStyleScore: number;
  averageAbsoluteStyleScore: number;
  averageCompletedTurnDepth: number;
  shallowSearches: number;
  maximumAbsolutePersonalityBonus: number;
  maximumObjectiveRegret: number;
  personalityParityFailures: number;
  actionTypes: Record<string, number>;
}

const EXPECTED_DOCTRINAL_FINGERPRINT = [
  "swan:4,6", "swan:4,6", "move:7,7;9,8:0,1", "move:6,6;6,8;8,5;8,6;9,7:-1,-1",
  "swan:3,6", "swan:3,6", "swan:4,7", "swan:4,8",
  "swan:5,5", "swan:5,5", "swan:7,5", "swan:7,5",
  "swan:4,6", "swan:5,6", "move:2,5;3,5;5,6:1,-1", "move:3,5:-1,-1",
  "swan:7,4", "swan:8,5", "swan:7,5", "swan:6,6",
  "swan:4,7", "swan:4,7", "swan:6,7", "stone:4,1"
] as const;

const sampledPlies = [4, 8, 18, 30] as const;
const corpus = reachableCorpus();
const nodeBudget = 10_000;
const decisions = new Map<AiStyleId, string[]>();
const audits = new Map<AiStyleId, StyleAudit>();
const startedAt = performance.now();

for (const style of AI_STYLE_IDS) {
  const core = createNativeVeryHardCoreSync();
  const keys: string[] = [];
  let legal = 0;
  let styleTotal = 0;
  let absoluteStyleTotal = 0;
  let completedDepthTotal = 0;
  let shallowSearches = 0;
  let maximumAbsolutePersonalityBonus = 0;
  let maximumObjectiveRegret = 0;
  let personalityParityFailures = 0;
  const actionTypes: Record<string, number> = { stone: 0, swan: 0, move: 0, push: 0 };
  for (const state of corpus) {
    const evaluation = explainVeryHardPosition(state, state.current, style);
    styleTotal += evaluation.style;
    absoluteStyleTotal += Math.abs(evaluation.style);
    const result = core.search(state, {
      style,
      maxTurnDepth: 2,
      tacticalDepth: 1,
      exactDepth: 0,
      nodeBudget,
      budgetMs: 0
    });
    completedDepthTotal += result.completedTurnDepth;
    if (result.completedTurnDepth < 1) shallowSearches += 1;
    const key = result.action ? actionKey(result.action) : "null";
    keys.push(key);
    const applied = result.action ? applyAction(state, result.action) : null;
    if (result.action && applied) {
      legal += 1;
      actionTypes[result.action.type] += 1;
      const expectedBonus = evaluateVeryHardRootPersonality(
        state,
        result.action,
        applied,
        state.current,
        style
      );
      if (result.personalityBonus !== expectedBonus ||
          result.score !== result.objectiveScore + result.personalityBonus) {
        personalityParityFailures += 1;
      }
    }
    maximumAbsolutePersonalityBonus = Math.max(maximumAbsolutePersonalityBonus, Math.abs(result.personalityBonus));
    maximumObjectiveRegret = Math.max(maximumObjectiveRegret, result.objectiveRegret);
  }
  decisions.set(style, keys);
  audits.set(style, {
    decisions: corpus.length,
    legal,
    divergenceFromDoctrinal: 0,
    averageStyleScore: styleTotal / corpus.length,
    averageAbsoluteStyleScore: absoluteStyleTotal / corpus.length,
    averageCompletedTurnDepth: completedDepthTotal / corpus.length,
    shallowSearches,
    maximumAbsolutePersonalityBonus,
    maximumObjectiveRegret,
    personalityParityFailures,
    actionTypes
  });
}

const doctrinal = decisions.get("doctrinal")!;
for (const style of AI_STYLE_IDS) {
  const keys = decisions.get(style)!;
  audits.get(style)!.divergenceFromDoctrinal = keys.filter((key, index) => key !== doctrinal[index]).length / corpus.length;
}

const pairwiseDivergence: Record<string, number> = {};
for (let left = 0; left < AI_STYLE_IDS.length; left += 1) {
  for (let right = left + 1; right < AI_STYLE_IDS.length; right += 1) {
    const a = decisions.get(AI_STYLE_IDS[left])!;
    const b = decisions.get(AI_STYLE_IDS[right])!;
    pairwiseDivergence[`${AI_STYLE_IDS[left]}:${AI_STYLE_IDS[right]}`] =
      a.filter((key, index) => key !== b[index]).length / corpus.length;
  }
}

const report = {
  kind: "linith-ai-personality-audit",
  formatVersion: 2,
  corpus: {
    states: corpus.length,
    generator: "six-curated-starts-plus-three-seeded-opening-policies",
    openingPolicies: ["development-first", "construction-first", "mixed"],
    sampledPlies,
    nodeBudget,
    maxTurnDepth: 2
  },
  elapsedMs: performance.now() - startedAt,
  styles: Object.fromEntries(AI_STYLE_IDS.map((style) => [style, audits.get(style)])),
  characterSignals: measureCharacterSignals(),
  doctrinalFingerprint: doctrinal,
  pairwiseDivergence
};

console.log(JSON.stringify(report, null, 2));
if ([...audits.values()].some(({ legal, decisions }) => legal !== decisions)) process.exitCode = 1;
if ([...audits.values()].some(({ shallowSearches }) => shallowSearches > 0)) process.exitCode = 1;
if (doctrinal.join("\n") !== EXPECTED_DOCTRINAL_FINGERPRINT.join("\n")) process.exitCode = 1;
if ([...audits.values()].some(({ maximumAbsolutePersonalityBonus }) =>
  maximumAbsolutePersonalityBonus > VERY_HARD_ROOT_PERSONALITY_LIMIT)) process.exitCode = 1;
if ([...audits.values()].some(({ maximumObjectiveRegret }) =>
  maximumObjectiveRegret > VERY_HARD_ROOT_PERSONALITY_LIMIT * 2)) process.exitCode = 1;
if ([...audits.values()].some(({ personalityParityFailures }) => personalityParityFailures > 0)) process.exitCode = 1;
if (audits.get("doctrinal")!.maximumAbsolutePersonalityBonus !== 0 ||
    audits.get("doctrinal")!.maximumObjectiveRegret !== 0) process.exitCode = 1;
if (Object.entries(report.characterSignals).some(([style, signal]) =>
  style === "doctrinal" ? signal.spread !== 0 : signal.spread <= 0)) process.exitCode = 1;

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
    [[4, 4], [6, 6]], [[5, 5], [3, 7]], [[3, 3], [6, 5]],
    [[6, 3], [3, 6]], [[2, 4], [7, 5]], [[4, 2], [5, 7]]
  ];
  const result: SearchState[] = [];
  starts.forEach(([sun, moon], startIndex) => {
    let state = initialState(sun, moon);
    const random = seededRandom(0x51a700 + startIndex * 0x9e37);
    for (let ply = 0; ply < 31; ply += 1) {
      if ((sampledPlies as readonly number[]).includes(ply)) result.push(cloneState(state));
      const actions = generateLegalActions(state);
      if (actions.length === 0) break;
      const developing = actions.filter((action) => action.type === "swan");
      const constructing = actions.filter((action) => action.type === "stone");
      const policy = startIndex % 3;
      const candidates = policy === 0 && ply < 12 && developing.length > 0
        ? developing
        : policy === 1 && ply < 8 && constructing.length > 0
          ? constructing
          : actions;
      let advanced = false;
      const offset = Math.floor(random() * candidates.length);
      for (let index = 0; index < candidates.length; index += 1) {
        const next = applyAction(state, candidates[(offset + index) % candidates.length]);
        if (!next || next.outcome) continue;
        state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
        advanced = true;
        break;
      }
      if (!advanced) break;
    }
  });
  return result;
}

function initialState(sun: readonly [number, number], moon: readonly [number, number]): SearchState {
  const board = Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
  board[sun[0]][sun[1]] = SWAN_SUN;
  board[moon[0]][moon[1]] = SWAN_MOON;
  return { board, current: 2, movesLeft: 1 };
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 0x1_0000_0000;
  };
}
