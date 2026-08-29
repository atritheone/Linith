import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  actionKey,
  applyAction,
  boardKey,
  generateLegalActions,
  type SearchState
} from "../src/renderer/game/rulesEngine";

interface DecisionTrace {
  actionNumber: number;
  source: string;
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

interface ArenaGame {
  id: string;
  pairId: string;
  openingId: string;
  challengerSide: 1 | 2;
  challengerStyle: string;
  classification: string;
  actions: number;
  repetitions: number;
  transcript: string[];
  decisionTrace: DecisionTrace[];
  search: {
    searches: number;
    continuationHits: number;
    fallbackSearches: number;
    completedDepth: number;
    engineElapsedMs: number;
    elapsedMs: number;
  };
}

interface ArenaReport {
  manifestId: string;
  veryHardArtifact: { sha256: string; bytes: number } | null;
  games: ArenaGame[];
  manifest: { games: Array<{ id: string; opening: SearchState }> };
}

const raw = new Map<string, string>();
for (const argument of process.argv.slice(2)) {
  const equals = argument.indexOf("=");
  if (!argument.startsWith("--") || equals < 0) throw new Error(`Expected --name=value, received ${argument}`);
  raw.set(argument.slice(2, equals), argument.slice(equals + 1));
}
const directory = resolve(raw.get("reports") ?? "native/very-hard/ladder-stage2-32-shards");
const outputPath = resolve(raw.get("output") ?? "native/very-hard/stage2-failure-analysis.json");
const reportPaths = readdirSync(directory)
  .filter((name) => name.endsWith(".json"))
  .sort()
  .map((name) => resolve(directory, name));

const failures: Record<string, unknown>[] = [];
const outcomeGroups = new Map<string, ArenaGame[]>();
let totalWins = 0;
let totalLosses = 0;
let totalDraws = 0;
const artifacts = new Map<string, number>();

for (const reportPath of reportPaths) {
  const report = JSON.parse(readFileSync(reportPath, "utf8")) as ArenaReport;
  if (report.veryHardArtifact) artifacts.set(report.veryHardArtifact.sha256, report.veryHardArtifact.bytes);
  for (const game of report.games) {
    const group = outcomeGroups.get(game.classification) ?? [];
    group.push(game);
    outcomeGroups.set(game.classification, group);
    if (game.classification === "win") totalWins += 1;
    else if (game.classification === "loss") totalLosses += 1;
    else if (game.classification === "terminal-draw") totalDraws += 1;
    if (game.classification === "win" || game.classification === "terminal-draw") continue;

    const manifestGame = report.manifest.games.find((entry) => entry.id === game.id);
    if (!manifestGame) throw new Error(`Missing manifest game ${game.id} in ${reportPath}`);
    const replay = replayGame(manifestGame.opening, game);
    const stem = reportPath.split(/[\\/]/).at(-1)!.replace(/\.json$/, "");
    const category = game.classification === "loss" ? "loss" : "repetition";
    failures.push({
      report: reportPath,
      manifestId: report.manifestId,
      gameId: game.id,
      pairId: game.pairId,
      pairManifest: resolve(
        `native/very-hard/stage2-failures/${category}/${stem}-${game.pairId}.manifest.json`
      ),
      classification: game.classification,
      openingId: game.openingId,
      challengerSide: game.challengerSide === 1 ? "sun" : "moon",
      style: game.challengerStyle,
      actions: game.actions,
      maximumRepetition: game.repetitions,
      search: summarizeGameSearch(game),
      repeatedCycle: replay.repeatedCycle,
      lastVeryHardDecisions: game.decisionTrace.slice(-8)
    });
  }
}

const resolved = totalWins + totalLosses + totalDraws;
const resolvedPoints = totalWins + totalDraws * 0.5;
const output = {
  kind: "linith-arena-failure-analysis",
  formatVersion: 1,
  reports: reportPaths,
  artifacts: [...artifacts].map(([sha256, bytes]) => ({ sha256, bytes })),
  resolved: {
    games: resolved,
    wins: totalWins,
    losses: totalLosses,
    draws: totalDraws,
    scoreRate: resolved > 0 ? resolvedPoints / resolved : null,
    oneSided95Lower: resolved > 0 ? wilsonLower(resolvedPoints, resolved, 1.6448536269514722) : null
  },
  byOutcome: [...outcomeGroups].map(([classification, games]) => ({
    classification,
    games: games.length,
    averageActions: average(games.map((game) => game.actions)),
    searches: sum(games.map((game) => game.search.searches)),
    continuationHits: sum(games.map((game) => game.search.continuationHits)),
    floorDecisions: sum(games.map((game) => game.search.fallbackSearches)),
    floorRate: ratio(
      sum(games.map((game) => game.search.fallbackSearches)),
      sum(games.map((game) => game.search.searches))
    ),
    averageCompletedDepth: ratio(
      sum(games.map((game) => game.search.completedDepth)),
      sum(games.map((game) => game.search.searches))
    )
  })),
  failures
};

writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
console.log(JSON.stringify({
  output: outputPath,
  resolved: output.resolved,
  byOutcome: output.byOutcome,
  failures: failures.length
}, null, 2));

function replayGame(opening: SearchState, game: ArenaGame): { repeatedCycle: Record<string, unknown> | null } {
  let state = cloneState(opening);
  const occurrences = new Map<string, number[]>([[boardKey(state), [0]]]);
  for (let index = 0; index < game.transcript.length; index += 1) {
    const key = transcriptActionKey(game.transcript[index]);
    const action = generateLegalActions(state).find((candidate) => actionKey(candidate) === key);
    if (!action) throw new Error(`Cannot replay ${game.id} action ${index + 1}: ${key}`);
    const next = applyAction(state, action);
    if (!next || next.outcome) break;
    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
    const position = boardKey(state);
    const positions = occurrences.get(position) ?? [];
    positions.push(index + 1);
    occurrences.set(position, positions);
  }
  const repeated = [...occurrences.entries()]
    .filter(([, actions]) => actions.length >= 3)
    .sort((left, right) => right[1].at(-1)! - left[1].at(-1)!)[0];
  if (!repeated) return { repeatedCycle: null };
  const [position, actions] = repeated;
  const start = actions.at(-2)!;
  const end = actions.at(-1)!;
  const trace = new Map(game.decisionTrace.map((entry) => [entry.actionNumber, entry.source]));
  return {
    repeatedCycle: {
      position,
      occurrences: actions,
      cycleLength: end - start,
      cycle: game.transcript.slice(start, end).map((entry, offset) => ({
        actionNumber: start + offset + 1,
        transcript: entry,
        veryHardSource: trace.get(start + offset + 1) ?? null
      }))
    }
  };
}

function summarizeGameSearch(game: ArenaGame): Record<string, unknown> {
  return {
    searches: game.search.searches,
    continuationHits: game.search.continuationHits,
    floorDecisions: game.search.fallbackSearches,
    floorRate: ratio(game.search.fallbackSearches, game.search.searches),
    averageCompletedDepth: ratio(game.search.completedDepth, game.search.searches),
    averageEngineElapsedMs: ratio(game.search.engineElapsedMs, game.search.searches),
    averageSelectorElapsedMs: ratio(game.search.elapsedMs, game.search.searches)
  };
}

function transcriptActionKey(entry: string): string {
  const parts = entry.split(":");
  if (parts.length < 6) throw new Error(`Malformed transcript entry: ${entry}`);
  return parts.slice(4).join(":");
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function average(values: number[]): number | null {
  return values.length > 0 ? sum(values) / values.length : null;
}

function ratio(numerator: number, denominator: number): number | null {
  return denominator > 0 ? numerator / denominator : null;
}

function wilsonLower(successes: number, trials: number, z: number): number {
  const p = successes / trials;
  const z2 = z * z;
  const denominator = 1 + z2 / trials;
  const centre = (p + z2 / (2 * trials)) / denominator;
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator;
  return Math.max(0, centre - margin);
}
