import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { linithAI } from "../src/renderer/game/ai";
import {
  actionKey,
  applyAction,
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import { createNativeVeryHardCoreSync } from "../native/very-hard/node-adapter";

interface TraceEntry {
  actionNumber: number;
  player: "sun" | "moon";
  source: "book" | "native" | "typescript" | "hard-floor" | "continuation";
  action: string;
  completedDepth: number | null;
  attemptedDepth: number | null;
  score: number | null;
  nodes: number;
  budgetMs: number;
  engineElapsedMs: number;
  elapsedMs: number;
  stopReason: string;
}

interface ReportGame {
  id: string;
  challengerSide: 1 | 2;
  challengerStyle: string;
  transcript: string[];
  decisionTrace?: TraceEntry[];
}

interface ReportManifestGame {
  id: string;
  opening: SearchState;
  decisionSeed: number;
}

interface ArenaReport {
  manifestId: string;
  games: ReportGame[];
  manifest: { games: ReportManifestGame[] };
}

interface DiagnosticConfig {
  report: string;
  output: string | null;
  maxDivergences: number;
  deepProbes: number;
  deepBudgetMs: number;
  deepNodeBudget: number;
}

const config = readConfig(process.argv.slice(2));
const report = JSON.parse(readFileSync(resolve(config.report), "utf8")) as ArenaReport;
const core = createNativeVeryHardCoreSync();
const originalRandom = Math.random;
const originalWindow = globalThis.window;

try {
  const divergences: Record<string, unknown>[] = [];
  const traceTotals = { decisions: 0, native: 0, hardFloor: 0, book: 0, continuation: 0, typescript: 0 };
  for (const game of report.games) {
    const manifestGame = report.manifest.games.find((candidate) => candidate.id === game.id);
    if (!manifestGame) throw new Error(`Manifest is missing game ${game.id}.`);
    const traceByAction = new Map((game.decisionTrace ?? []).map((entry) => [entry.actionNumber, entry]));
    let state = cloneState(manifestGame.opening);
    for (let index = 0; index < game.transcript.length; index += 1) {
      const actionNumber = index + 1;
      const playedKey = transcriptActionKey(game.transcript[index]);
      const played = findAction(state, playedKey);
      const trace = traceByAction.get(actionNumber);
      if (trace) {
        traceTotals.decisions += 1;
        if (trace.source === "native") traceTotals.native += 1;
        else if (trace.source === "hard-floor") traceTotals.hardFloor += 1;
        else if (trace.source === "book") traceTotals.book += 1;
        else if (trace.source === "continuation") traceTotals.continuation += 1;
        else traceTotals.typescript += 1;

        let hard: LinithAction | null = null;
        if (divergences.length < config.maxDivergences) {
          const seed = mixSeed(manifestGame.decisionSeed, index, state.current);
          globalThis.window = { linithGetStyle: () => game.challengerStyle } as Window & typeof globalThis;
          Math.random = seededRandom(mixSeed(seed, 0xfa11ba));
          hard = linithAI(state.board, state.current, "hard");
        }
        const hardKey = hard ? actionKey(hard) : null;
        if (hardKey !== playedKey && divergences.length < config.maxDivergences) {
          const rootPlayer = state.current;
          const playedNext = applyAction(state, played);
          const hardNext = hard ? applyAction(state, hard) : null;
          const diagnostic: Record<string, unknown> = {
            gameId: game.id,
            actionNumber,
            player: rootPlayer === 1 ? "sun" : "moon",
            movesLeft: state.movesLeft,
            position: compactState(state),
            selected: playedKey,
            hard: hardKey,
            selectedStaticScore: playedNext && !playedNext.outcome
              ? core.evaluate(stateFromApplied(playedNext), rootPlayer, game.challengerStyle)
              : terminalScore(playedNext?.outcome, rootPlayer),
            hardStaticScore: hardNext && !hardNext.outcome
              ? core.evaluate(stateFromApplied(hardNext), rootPlayer, game.challengerStyle)
              : terminalScore(hardNext?.outcome, rootPlayer),
            trace
          };
          if (divergences.length < config.deepProbes) {
            core.clearCache();
            const deep = core.search(state, {
              style: game.challengerStyle,
              budgetMs: config.deepBudgetMs,
              nodeBudget: config.deepNodeBudget,
              maxTurnDepth: 8,
              tacticalDepth: 2
            });
            diagnostic.deep = {
              action: deep.action ? actionKey(deep.action) : null,
              score: deep.score,
              completedDepth: deep.completedTurnDepth,
              attemptedDepth: deep.attemptedTurnDepth,
              nodes: deep.nodes,
              stopReason: deep.stopReason
            };
          }
          divergences.push(diagnostic);
        }
      }
      const next = applyAction(state, played);
      if (!next || next.outcome) break;
      state = stateFromApplied(next);
    }
  }

  const output = {
    kind: "linith-arena-decision-diagnostic",
    formatVersion: 1,
    report: resolve(config.report),
    manifestId: report.manifestId,
    traceTotals,
    divergenceCount: divergences.length,
    divergences
  };
  const serialized = `${JSON.stringify(output, null, 2)}\n`;
  if (config.output) writeFileSync(resolve(config.output), serialized, "utf8");
  console.log(serialized.trimEnd());
} finally {
  Math.random = originalRandom;
  globalThis.window = originalWindow;
}

function stateFromApplied(applied: { board: SearchState["board"]; current: 1 | 2; movesLeft: number }): SearchState {
  return { board: applied.board, current: applied.current, movesLeft: applied.movesLeft };
}

function terminalScore(outcome: string | null | undefined, root: 1 | 2): number | null {
  if (!outcome) return null;
  if (outcome === "draw") return 0;
  return (outcome === "sun") === (root === 1) ? 1_000_000 : -1_000_000;
}

function transcriptActionKey(entry: string): string {
  const parts = entry.split(":");
  if (parts.length < 6) throw new Error(`Malformed transcript entry: ${entry}`);
  return parts.slice(4).join(":");
}

function findAction(state: SearchState, key: string): LinithAction {
  const action = generateLegalActions(state).find((candidate) => actionKey(candidate) === key);
  if (!action) throw new Error(`Transcript action ${key} is illegal in the reconstructed position.`);
  return action;
}

function compactState(state: SearchState): Record<string, unknown> {
  return {
    current: state.current,
    movesLeft: state.movesLeft,
    board: state.board.map((row) => row.join("")).join("/")
  };
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
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
    return value / 4_294_967_296;
  };
}

function readConfig(arguments_: string[]): DiagnosticConfig {
  const raw = new Map<string, string>();
  for (const argument of arguments_) {
    const equals = argument.indexOf("=");
    if (!argument.startsWith("--") || equals < 0) throw new Error(`Expected --name=value, received ${argument}`);
    raw.set(argument.slice(2, equals), argument.slice(equals + 1));
  }
  const report = raw.get("report");
  if (!report) throw new Error("--report is required");
  return {
    report,
    output: raw.get("output") ?? null,
    maxDivergences: integer(raw, "max-divergences", 20, 1),
    deepProbes: integer(raw, "deep-probes", 0, 0),
    deepBudgetMs: integer(raw, "deep-budget-ms", 3_000, 1),
    deepNodeBudget: integer(raw, "deep-node-budget", 10_000_000, 1)
  };
}

function integer(raw: Map<string, string>, key: string, fallback: number, minimum: number): number {
  const value = Number(raw.get(key) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${key} must be an integer >= ${minimum}`);
  return value;
}
