import { execFileSync } from "node:child_process";
import { AI_STYLES } from "../src/renderer/game/aiStyles";
import { linithAI as correctedAI } from "../src/renderer/game/ai";
import { EMPTY, SWAN_MOON, SWAN_SUN, type Board, type Player } from "../src/renderer/game/encirclement";
import { actionKey, applyAction, boardKey, type LinithAction, type SearchState } from "../src/renderer/game/rulesEngine";

const BASELINE_AI_BLOB = "ecc53b036170a63cb018b80ba2f3d6c966a23835";
const baselineAI = loadBaselineAI();
const difficulties = process.argv.slice(2).filter((arg) => ["easy", "medium", "hard"].includes(arg));
const selectedDifficulties = difficulties.length > 0 ? difficulties : ["easy", "medium", "hard"];
const openings: Array<[[number, number], [number, number]]> = [
  [[4, 4], [6, 6]],
  [[5, 5], [3, 7]]
];
const style = "doctrinal";
const originalWindow = globalThis.window;
const originalRandom = Math.random;
globalThis.window = { linithGetStyle: () => style } as Window & typeof globalThis;

try {
  for (const difficulty of selectedDifficulties) {
    const summary = {
      correctedWins: 0,
      baselineWins: 0,
      draws: 0,
      invalid: 0,
      games: 0,
      actions: 0,
      comparedDecisions: 0,
      changedDecisions: 0
    };
    for (let openingIndex = 0; openingIndex < openings.length; openingIndex += 1) {
      for (const correctedSide of [1, 2] as Player[]) {
        const result = playGame(openings[openingIndex], correctedSide, difficulty, openingIndex * 10 + correctedSide);
        summary.games += 1;
        summary.actions += result.actions;
        summary.comparedDecisions += result.comparedDecisions;
        summary.changedDecisions += result.changedDecisions;
        if (result.invalid) summary.invalid += 1;
        if (result.winner === null) summary.draws += 1;
        else if (result.winner === correctedSide) summary.correctedWins += 1;
        else summary.baselineWins += 1;
      }
    }
    console.log(JSON.stringify({
      difficulty,
      style,
      ...summary,
      averageActions: Number((summary.actions / summary.games).toFixed(1)),
      changedDecisionRate: Number((summary.changedDecisions / Math.max(1, summary.comparedDecisions)).toFixed(3))
    }));
  }
} finally {
  Math.random = originalRandom;
  globalThis.window = originalWindow;
}

function playGame(
  opening: [[number, number], [number, number]],
  correctedSide: Player,
  difficulty: string,
  seed: number
): { winner: Player | null; actions: number; invalid: boolean; comparedDecisions: number; changedDecisions: number } {
  const board = emptyBoard();
  board[opening[0][0]][opening[0][1]] = SWAN_SUN;
  board[opening[1][0]][opening[1][1]] = SWAN_MOON;
  let state: SearchState = { board, current: 2, movesLeft: 1 };
  const repetitions = new Map<string, number>();
  let comparedDecisions = 0;
  let changedDecisions = 0;

  for (let actionNumber = 0; actionNumber < 160; actionNumber += 1) {
    const decisionSeed = seed * 10_000 + actionNumber * 17 + state.current;
    Math.random = seededRandom(decisionSeed);
    const selector = state.current === correctedSide ? correctedAI : baselineAI;
    const action = selector(state.board, state.current, difficulty);
    if (actionNumber < 24) {
      Math.random = seededRandom(decisionSeed);
      const comparison = selector === correctedAI
        ? baselineAI(state.board, state.current, difficulty)
        : correctedAI(state.board, state.current, difficulty);
      comparedDecisions += 1;
      if ((!action && comparison) || (action && !comparison) ||
          (action && comparison && actionKey(action) !== actionKey(comparison))) changedDecisions += 1;
    }
    if (!action) return {
      winner: other(state.current), actions: actionNumber, invalid: true, comparedDecisions, changedDecisions
    };
    const next = applyAction(state, action);
    if (!next) {
      console.error(`Invalid ${difficulty} action: ${actionKey(action)} on ${boardKey(state)}`);
      return { winner: other(state.current), actions: actionNumber, invalid: true, comparedDecisions, changedDecisions };
    }
    if (next.outcome) {
      const winner = next.outcome === "draw" ? null : next.outcome === "sun" ? 1 : 2;
      return { winner, actions: actionNumber + 1, invalid: false, comparedDecisions, changedDecisions };
    }
    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
    const key = boardKey(state);
    const count = (repetitions.get(key) ?? 0) + 1;
    repetitions.set(key, count);
    if (count >= 3) return { winner: null, actions: actionNumber + 1, invalid: false, comparedDecisions, changedDecisions };
  }
  return { winner: null, actions: 160, invalid: false, comparedDecisions, changedDecisions };
}

function loadBaselineAI(): (board: Board, current: Player, difficulty: string) => LinithAction | null {
  let source = execFileSync("git", ["cat-file", "-p", BASELINE_AI_BLOB], { encoding: "utf8" });
  source = source
    .replace(/import \{ AI_STYLES \} from [^;]+;/, "")
    .replace("export function linithAI", "function linithAI");
  return new Function("AI_STYLES", `${source}\nreturn linithAI;`)(AI_STYLES) as ReturnType<typeof loadBaselineAI>;
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1_664_525) + 1_013_904_223) >>> 0;
    return value / 4_294_967_296;
  };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function other(player: Player): Player {
  return player === 1 ? 2 : 1;
}
