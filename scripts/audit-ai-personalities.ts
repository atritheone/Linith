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
import { explainVeryHardPosition } from "../src/renderer/game/veryHard/evaluate";

interface StyleAudit {
  decisions: number;
  legal: number;
  divergenceFromDoctrinal: number;
  averageStyleScore: number;
  averageAbsoluteStyleScore: number;
  averageCompletedTurnDepth: number;
  shallowSearches: number;
  actionTypes: Record<string, number>;
}

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
    if (result.action && applyAction(state, result.action)) {
      legal += 1;
      actionTypes[result.action.type] += 1;
    }
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
  formatVersion: 1,
  corpus: {
    states: corpus.length,
    generator: "six-curated-starts-plus-seeded-reachable-plies",
    sampledPlies,
    nodeBudget,
    maxTurnDepth: 2
  },
  elapsedMs: performance.now() - startedAt,
  styles: Object.fromEntries(AI_STYLE_IDS.map((style) => [style, audits.get(style)])),
  pairwiseDivergence
};

console.log(JSON.stringify(report, null, 2));
if ([...audits.values()].some(({ legal, decisions }) => legal !== decisions)) process.exitCode = 1;
if ([...audits.values()].some(({ shallowSearches }) => shallowSearches > 0)) process.exitCode = 1;
if (AI_STYLE_IDS.slice(1).some((style) => audits.get(style)!.divergenceFromDoctrinal === 0)) process.exitCode = 1;

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
      const candidates = ply < 12 && developing.length > 0 ? developing : actions;
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
