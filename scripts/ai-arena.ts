import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { linithAI } from "../src/renderer/game/ai";
import { AI_STYLES } from "../src/renderer/game/aiStyles";
import {
  EMPTY,
  SWAN_MOON,
  SWAN_SUN,
  type Board,
  type Player
} from "../src/renderer/game/encirclement";
import {
  MOON,
  SUN,
  actionKey,
  applyAction,
  boardKey,
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../src/renderer/game/rulesEngine";
import {
  createVeryHardSearchEngine,
  type VeryHardSearchEngine,
  type VeryHardStopReason
} from "../src/renderer/game/veryHard";
import { lookupOpeningBookAction } from "../src/renderer/game/veryHard/openingBook";
import {
  chooseVeryHardTimeBudget,
  type VeryHardPlatform
} from "../src/renderer/game/veryHard/timeManager";
import { loadNativeVeryHardCoreSync } from "../native/very-hard/node-adapter";
import type { NativeVeryHardContinuation } from "../src/renderer/game/veryHard/native/core";

type Difficulty = "easy" | "medium" | "hard" | "very-hard";
type OpeningMode = "curated" | "generated" | "mixed";
type TimingMode = "shipping" | "fixed";
type FloorMode = "reply" | "depth1" | "never";
type GameClassification =
  | "win"
  | "loss"
  | "terminal-draw"
  | "repetition-unresolved"
  | "action-limit-unresolved"
  | "invalid"
  | "crash";

interface ArenaConfig {
  seed: number;
  games: number;
  maxActions: number;
  repetitionCount: number;
  openingMode: OpeningMode;
  openingCount: number;
  openingPlies: number;
  challenger: Difficulty;
  opponent: Difficulty;
  challengerStyles: string[];
  opponentStyles: string[];
  styleOffset: number;
  veryHardBudgetMs: number;
  veryHardNodeBudget: number;
  veryHardMaxDepth: number;
  veryHardEngine: "native" | "typescript";
  veryHardTiming: TimingMode;
  veryHardPlatform: VeryHardPlatform;
  veryHardFloor: FloorMode;
  manifestIn: string | null;
  manifestOut: string | null;
  output: string | null;
  summaryOnly: boolean;
  progressEvery: number;
  minPessimisticScore: number | null;
  minOneSidedLower: number | null;
  maxUnresolvedRate: number | null;
  maxOperationalFailures: number | null;
  searchTrace: boolean;
}

interface Opening {
  id: string;
  source: "curated" | "generated";
  state: SearchState;
  generationSeed: number;
  generationActions: string[];
}

interface ManifestGame {
  id: string;
  pairId: string;
  openingId: string;
  opening: SearchState;
  challengerSide: Player;
  challengerStyle: string;
  opponentStyle: string;
  decisionSeed: number;
}

interface ArenaManifest {
  format: "linith-ai-arena-manifest";
  version: 1;
  seed: number;
  challenger: Difficulty;
  opponent: Difficulty;
  maxActions: number;
  repetitionCount: number;
  games: ManifestGame[];
}

interface SearchTotals {
  searches: number;
  bookHits: number;
  continuationHits: number;
  nodes: number;
  engineElapsedMs: number;
  elapsedMs: number;
  completedDepth: number;
  fallbackSearches: number;
  stops: Partial<Record<VeryHardStopReason, number>>;
}

interface GameResult {
  id: string;
  pairId: string;
  openingId: string;
  challengerSide: Player;
  challengerStyle: string;
  opponentStyle: string;
  classification: GameClassification;
  winner: Player | null;
  actions: number;
  repetitions: number;
  illegalActions: number;
  nullActions: number;
  crashes: number;
  error: string | null;
  transcript: string[];
  search: SearchTotals;
  decisionTrace?: DecisionTrace[];
}

interface DecisionTrace {
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
  stopReason: VeryHardStopReason | "book" | "continuation";
  /** Native/TS proposal replaced by the selected Hard floor, when applicable. */
  proposalAction?: string;
}

type DecisionTracePayload = Omit<DecisionTrace, "actionNumber" | "player" | "action">;

interface Counter {
  games: number;
  wins: number;
  losses: number;
  terminalDraws: number;
  repetitionUnresolved: number;
  actionLimitUnresolved: number;
  invalid: number;
  crashes: number;
}

interface Selection {
  action: LinithAction | null;
  search: SearchTotals;
  trace: DecisionTracePayload | null;
}

interface GameEngines {
  typescript: VeryHardSearchEngine;
  nativeContinuation: NativeVeryHardContinuation[];
}

const originalWindow = globalThis.window;
const originalRandom = Math.random;
const config = readConfig(process.argv.slice(2));
const nativeVeryHard = config.veryHardEngine === "native"
  && (config.challenger === "very-hard" || config.opponent === "very-hard")
  ? loadNativeVeryHardCoreSync()
  : null;
const nativeVeryHardArtifact = nativeVeryHard ? describeNativeArtifact() : null;

try {
  globalThis.window = { linithGetStyle: () => "doctrinal" } as Window & typeof globalThis;
  const manifest = config.manifestIn ? readManifest(config.manifestIn) : buildManifest(config);
  validateManifest(manifest, config);
  if (config.manifestOut) writeJson(config.manifestOut, manifest);

  const results: GameResult[] = [];
  for (let index = 0; index < manifest.games.length; index += 1) {
    results.push(playGame(manifest.games[index], manifest, config));
    if (config.progressEvery > 0 && (index + 1) % config.progressEvery === 0) {
      process.stderr.write(`Linith arena: ${index + 1}/${manifest.games.length} games complete\n`);
    }
  }

  const report = buildReport(config, manifest, results);
  const serialized = JSON.stringify(report, null, config.output ? 2 : 0);
  if (config.output) writeFileSync(resolve(config.output), `${serialized}\n`, "utf8");
  console.log(serialized);
  applyGates(config, report.summary as ReturnType<typeof summarize>);
} finally {
  Math.random = originalRandom;
  globalThis.window = originalWindow;
}

function buildManifest(settings: ArenaConfig): ArenaManifest {
  const openings = buildOpenings(settings);
  const stylePairs = settings.challengerStyles.flatMap((challengerStyle) =>
    settings.opponentStyles.map((opponentStyle) => ({ challengerStyle, opponentStyle }))
  );
  const games: ManifestGame[] = [];
  for (let gameIndex = 0; gameIndex < settings.games; gameIndex += 1) {
    const pairIndex = Math.floor(gameIndex / 2);
    const opening = openings[pairIndex % openings.length];
    const stylePair = stylePairs[
      (Math.floor(pairIndex / openings.length) + settings.styleOffset) % stylePairs.length
    ];
    games.push({
      id: `g${String(gameIndex + 1).padStart(5, "0")}`,
      pairId: `p${String(pairIndex + 1).padStart(5, "0")}`,
      openingId: opening.id,
      opening: cloneState(opening.state),
      challengerSide: gameIndex % 2 === 0 ? 1 : 2,
      challengerStyle: stylePair.challengerStyle,
      opponentStyle: stylePair.opponentStyle,
      decisionSeed: mixSeed(settings.seed, pairIndex, 0xa11ce)
    });
  }
  return {
    format: "linith-ai-arena-manifest",
    version: 1,
    seed: settings.seed,
    challenger: settings.challenger,
    opponent: settings.opponent,
    maxActions: settings.maxActions,
    repetitionCount: settings.repetitionCount,
    games
  };
}

function buildOpenings(settings: ArenaConfig): Opening[] {
  const curated = curatedOpenings();
  const openings: Opening[] = [];
  for (let index = 0; index < settings.openingCount; index += 1) {
    const generated = settings.openingMode === "generated" || (settings.openingMode === "mixed" && index % 2 === 1);
    if (!generated) {
      const opening = curated[index % curated.length];
      openings.push({ ...opening, id: `curated-${String(index + 1).padStart(3, "0")}`, state: cloneState(opening.state) });
      continue;
    }
    openings.push(generateOpening(settings, index, curated[index % curated.length].state));
  }
  return openings;
}

function curatedOpenings(): Opening[] {
  const coordinates: Array<[[number, number], [number, number]]> = [
    [[4, 4], [6, 6]], [[5, 5], [3, 7]], [[3, 3], [6, 5]], [[6, 3], [3, 6]],
    [[2, 4], [7, 5]], [[4, 2], [5, 7]], [[2, 2], [7, 7]], [[7, 2], [2, 7]],
    [[1, 4], [8, 5]], [[4, 1], [5, 8]], [[3, 5], [6, 4]], [[5, 3], [4, 6]]
  ];
  return coordinates.map((coordinate, index) => ({
    id: `curated-${String(index + 1).padStart(3, "0")}`,
    source: "curated",
    state: stateFromCoordinates(coordinate),
    generationSeed: 0,
    generationActions: []
  }));
}

function generateOpening(settings: ArenaConfig, index: number, base: SearchState): Opening {
  const generationSeed = mixSeed(settings.seed, index, 0x0f3a1);
  const random = seededRandom(generationSeed);
  let state = cloneState(base);
  const generationActions: string[] = [];
  const plies = Math.max(1, settings.openingPlies + (index % 3) - 1);
  for (let ply = 0; ply < plies; ply += 1) {
    const actions = generateLegalActions(state);
    if (actions.length === 0) break;
    const action = actions[Math.floor(random() * actions.length)];
    const next = applyAction(state, action);
    if (!next || next.outcome) break;
    generationActions.push(actionKey(action));
    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
  }
  return {
    id: `generated-${String(index + 1).padStart(3, "0")}`,
    source: "generated",
    state,
    generationSeed,
    generationActions
  };
}

function playGame(game: ManifestGame, manifest: ArenaManifest, settings: ArenaConfig): GameResult {
  let state = cloneState(game.opening);
  nativeVeryHard?.clearCache();
  const engines: GameEngines = {
    typescript: createVeryHardSearchEngine(),
    nativeContinuation: []
  };
  const repetitions = new Map<string, number>([[boardKey(state), 1]]);
  const result: GameResult = {
    id: game.id,
    pairId: game.pairId,
    openingId: game.openingId,
    challengerSide: game.challengerSide,
    challengerStyle: game.challengerStyle,
    opponentStyle: game.opponentStyle,
    classification: "action-limit-unresolved",
    winner: null,
    actions: 0,
    repetitions: 1,
    illegalActions: 0,
    nullActions: 0,
    crashes: 0,
    error: null,
    transcript: [],
    search: emptySearchTotals()
  };
  if (settings.searchTrace) result.decisionTrace = [];

  for (let actionNumber = 0; actionNumber < manifest.maxActions; actionNumber += 1) {
    const challengerTurn = state.current === game.challengerSide;
    const difficulty = challengerTurn ? manifest.challenger : manifest.opponent;
    const style = challengerTurn ? game.challengerStyle : game.opponentStyle;
    let selection: Selection;
    try {
      selection = selectAction(
        difficulty,
        style,
        state,
        mixSeed(game.decisionSeed, actionNumber, state.current),
        settings,
        engines
      );
    } catch (error) {
      result.classification = "crash";
      result.crashes = 1;
      result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      result.winner = other(state.current);
      return result;
    }
    mergeSearch(result.search, selection.search);
    if (settings.searchTrace && selection.trace && selection.action) {
      result.decisionTrace!.push({
        actionNumber: actionNumber + 1,
        player: state.current === SUN ? "sun" : "moon",
        action: actionKey(selection.action),
        ...selection.trace
      });
    }
    if (!selection.action) {
      result.classification = "invalid";
      result.nullActions = 1;
      result.winner = other(state.current);
      result.error = "engine returned no action in a non-terminal position";
      return result;
    }

    const next = applyAction(state, selection.action);
    if (!next) {
      result.classification = "invalid";
      result.illegalActions = 1;
      result.winner = other(state.current);
      result.error = `illegal action: ${actionKey(selection.action)}`;
      return result;
    }
    result.transcript.push(
      `${actionNumber + 1}:${state.current === SUN ? "sun" : "moon"}:${difficulty}:${style}:${actionKey(selection.action)}`
    );
    result.actions = actionNumber + 1;
    if (next.outcome) {
      result.winner = next.outcome === "sun" ? 1 : next.outcome === "moon" ? 2 : null;
      result.classification = next.outcome === "draw"
        ? "terminal-draw"
        : result.winner === game.challengerSide ? "win" : "loss";
      return result;
    }

    state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
    const key = boardKey(state);
    const repetition = (repetitions.get(key) ?? 0) + 1;
    repetitions.set(key, repetition);
    result.repetitions = Math.max(result.repetitions, repetition);
    if (repetition >= manifest.repetitionCount) {
      result.classification = "repetition-unresolved";
      return result;
    }
  }
  return result;
}

function selectAction(
  difficulty: Difficulty,
  style: string,
  state: SearchState,
  seed: number,
  settings: ArenaConfig,
  engines: GameEngines
): Selection {
  globalThis.window = { linithGetStyle: () => style } as Window & typeof globalThis;
  if (difficulty === "very-hard") {
    const startedAt = performance.now();
    if (nativeVeryHard && engines.nativeContinuation.length > 0) {
      const continuation = engines.nativeContinuation[0];
      const exactState = nativeVeryHard.positionHash(state) === continuation.inputHash;
      const legal = exactState && applyAction(state, continuation.action) !== null;
      if (legal) {
        if (!nativeVeryHard.commitAction(state, continuation.action)) {
          throw new Error("Native Very Hard could not commit its exact-state continuation.");
        }
        engines.nativeContinuation.shift();
        const elapsedMs = performance.now() - startedAt;
        return {
          action: continuation.action,
          search: {
            ...emptySearchTotals(),
            continuationHits: 1,
            elapsedMs
          },
          trace: {
            source: "continuation",
            completedDepth: null,
            attemptedDepth: null,
            score: null,
            nodes: 0,
            budgetMs: 0,
            engineElapsedMs: 0,
            elapsedMs,
            stopReason: "continuation"
          }
        };
      }
      engines.nativeContinuation = [];
    }
    const budgetMs = settings.veryHardTiming === "shipping"
      ? chooseVeryHardTimeBudget(state, settings.veryHardPlatform).budgetMs
      : settings.veryHardBudgetMs;
    const book = lookupOpeningBookAction(state);
    if (book) {
      engines.nativeContinuation = [];
      if (nativeVeryHard && !nativeVeryHard.commitAction(state, book.action)) {
        throw new Error("Native Very Hard could not commit a legal opening-book action.");
      }
      const elapsedMs = performance.now() - startedAt;
      return {
        action: book.action,
        search: {
          ...emptySearchTotals(),
          bookHits: 1,
          elapsedMs
        },
        trace: {
          source: "book",
          completedDepth: null,
          attemptedDepth: null,
          score: book.score,
          nodes: 0,
          budgetMs,
          engineElapsedMs: 0,
          elapsedMs,
          stopReason: "book"
        }
      };
    }
    if (nativeVeryHard) {
      const engineStartedAt = performance.now();
      const search = nativeVeryHard.search(state, {
        style,
        budgetMs,
        nodeBudget: settings.veryHardNodeBudget,
        maxTurnDepth: settings.veryHardMaxDepth
      });
      const engineElapsedMs = performance.now() - engineStartedAt;
      const applied = search.action ? applyAction(state, search.action) : null;
      const completedOnlyOwnTurn = search.completedTurnDepth < 2
        && !!applied
        && !applied.outcome
        && applied.current !== state.current;
      const needsFloor = settings.veryHardFloor !== "never"
        && (!search.action
          || search.completedTurnDepth === 0
          || (settings.veryHardFloor === "reply" && completedOnlyOwnTurn));
      Math.random = seededRandom(mixSeed(seed, 0xfa11ba));
      const hardFloor = needsFloor ? linithAI(state.board, state.current, "hard") : null;
      const useFloor = needsFloor && !!hardFloor;
      const selectedAction = useFloor ? hardFloor : search.action;
      engines.nativeContinuation = useFloor ? [] : [...search.continuation];
      if (selectedAction && !nativeVeryHard.commitAction(state, selectedAction)) {
        throw new Error("Native Very Hard could not commit the arena-selected action.");
      }
      const elapsedMs = performance.now() - startedAt;
      return {
        action: selectedAction,
        search: {
          searches: 1,
          bookHits: 0,
          continuationHits: 0,
          nodes: search.nodes,
          engineElapsedMs,
          elapsedMs,
          completedDepth: search.completedTurnDepth,
          fallbackSearches: Number(useFloor),
          stops: { [search.stopReason]: 1 }
        },
        trace: {
          source: useFloor ? "hard-floor" : "native",
          completedDepth: search.completedTurnDepth,
          attemptedDepth: search.attemptedTurnDepth,
          score: search.score,
          nodes: search.nodes,
          budgetMs,
          engineElapsedMs,
          elapsedMs,
          stopReason: search.stopReason,
          proposalAction: useFloor && search.action ? actionKey(search.action) : undefined
        }
      };
    }
    const engineStartedAt = performance.now();
    const search = engines.typescript.search(state, {
      style,
      budgetMs,
      nodeBudget: settings.veryHardNodeBudget,
      maxDepth: settings.veryHardMaxDepth
    });
    const engineElapsedMs = performance.now() - engineStartedAt;
    const applied = search.action ? applyAction(state, search.action) : null;
    const completedOnlyOwnTurn = search.completedDepth < 2
      && !!applied
      && !applied.outcome
      && applied.current !== state.current;
    const needsFloor = settings.veryHardFloor !== "never"
      && (!search.action
        || search.completedDepth === 0
        || (settings.veryHardFloor === "reply" && completedOnlyOwnTurn));
    Math.random = seededRandom(mixSeed(seed, 0xfa11ba));
    const hardFloor = needsFloor ? linithAI(state.board, state.current, "hard") : null;
    const useFloor = needsFloor && !!hardFloor;
    const elapsedMs = performance.now() - startedAt;
    return {
      action: useFloor ? hardFloor : search.action,
      search: {
        searches: 1,
        bookHits: 0,
        continuationHits: 0,
        nodes: search.nodes,
        engineElapsedMs,
        elapsedMs,
        completedDepth: search.completedDepth,
        fallbackSearches: Number(useFloor),
        stops: { [search.stopReason]: 1 }
      },
      trace: {
        source: useFloor ? "hard-floor" : "typescript",
        completedDepth: search.completedDepth,
        attemptedDepth: search.attemptedDepth,
        score: search.score,
        nodes: search.nodes,
        budgetMs,
        engineElapsedMs,
        elapsedMs,
        stopReason: search.stopReason,
        proposalAction: useFloor && search.action ? actionKey(search.action) : undefined
      }
    };
  }

  engines.nativeContinuation = [];
  Math.random = seededRandom(seed);
  return {
    action: linithAI(state.board, state.current, difficulty),
    search: emptySearchTotals(),
    trace: null
  };
}

function buildReport(config: ArenaConfig, manifest: ArenaManifest, results: GameResult[]): Record<string, unknown> {
  const summary = summarize(results);
  const byColor = ([SUN, MOON] as Player[]).map((side) => ({
    challengerSide: side === SUN ? "sun" : "moon",
    ...summarize(results.filter((result) => result.challengerSide === side))
  }));
  const styleKeys = [...new Set(results.map((result) => `${result.challengerStyle}\u0000${result.opponentStyle}`))];
  const byStyleMatchup = styleKeys.map((key) => {
    const [challengerStyle, opponentStyle] = key.split("\u0000");
    return {
      challengerStyle,
      opponentStyle,
      ...summarize(results.filter((result) =>
        result.challengerStyle === challengerStyle && result.opponentStyle === opponentStyle
      ))
    };
  });

  const report: Record<string, unknown> = {
    kind: "linith-ai-arena",
    formatVersion: 1,
    config: publicConfig(config),
    manifestId: hashString(stableStringify(manifest)),
    veryHardArtifact: nativeVeryHardArtifact,
    summary,
    byColor,
    byStyleMatchup,
    search: summarizeSearch(results)
  };
  if (!config.summaryOnly) {
    report.manifest = manifest;
    report.games = results;
  }
  return report;
}

function summarize(results: GameResult[]): Counter & Record<string, unknown> {
  const counter: Counter = {
    games: results.length,
    wins: count(results, "win"),
    losses: count(results, "loss"),
    terminalDraws: count(results, "terminal-draw"),
    repetitionUnresolved: count(results, "repetition-unresolved"),
    actionLimitUnresolved: count(results, "action-limit-unresolved"),
    invalid: count(results, "invalid"),
    crashes: count(results, "crash")
  };
  const resolved = counter.wins + counter.losses + counter.terminalDraws;
  const decisive = counter.wins + counter.losses;
  const unresolved = counter.repetitionUnresolved + counter.actionLimitUnresolved;
  const operationalFailures = counter.invalid + counter.crashes;
  const points = counter.wins + counter.terminalDraws * 0.5;
  const resolvedScoreRate = resolved > 0 ? points / resolved : null;
  const pessimisticScoreRate = counter.games > 0 ? points / counter.games : null;
  return {
    ...counter,
    resolved,
    decisive,
    unresolved,
    operationalFailures,
    resolvedScoreRate,
    decisiveWinRate: decisive > 0 ? counter.wins / decisive : null,
    pessimisticScoreRate,
    unresolvedRate: counter.games > 0 ? unresolved / counter.games : null,
    operationalFailureRate: counter.games > 0 ? operationalFailures / counter.games : null,
    resolvedScore95: resolved > 0 ? wilson(points, resolved, 1.959963984540054) : null,
    resolvedScoreOneSided95Lower: resolved > 0 ? wilson(points, resolved, 1.6448536269514722)[0] : null,
    pessimisticScore95: counter.games > 0 ? wilson(points, counter.games, 1.959963984540054) : null,
    pessimisticScoreOneSided95Lower: counter.games > 0
      ? wilson(points, counter.games, 1.6448536269514722)[0]
      : null
  };
}

function summarizeSearch(results: GameResult[]): Record<string, unknown> {
  const total = emptySearchTotals();
  for (const result of results) mergeSearch(total, result.search);
  return {
    ...total,
    engineElapsedMs: round(total.engineElapsedMs),
    elapsedMs: round(total.elapsedMs),
    averageEngineElapsedMs: total.searches > 0 ? round(total.engineElapsedMs / total.searches) : null,
    averageSelectorElapsedMs: total.searches > 0 ? round(total.elapsedMs / total.searches) : null,
    averageCompletedDepth: total.searches > 0 ? round(total.completedDepth / total.searches) : null,
    nodesPerSecond: total.elapsedMs > 0 ? Math.round(total.nodes / (total.elapsedMs / 1000)) : null,
    fallbackRate: total.searches > 0 ? total.fallbackSearches / total.searches : null
  };
}

function applyGates(config: ArenaConfig, summary: ReturnType<typeof summarize>): void {
  const failures: string[] = [];
  const pessimistic = summary.pessimisticScoreRate as number | null;
  const lower = summary.pessimisticScoreOneSided95Lower as number | null;
  const unresolved = summary.unresolvedRate as number | null;
  const operationalFailures = summary.operationalFailures as number;
  if (config.minPessimisticScore !== null && (pessimistic === null || pessimistic < config.minPessimisticScore)) {
    failures.push(`pessimistic score ${formatRate(pessimistic)} < ${config.minPessimisticScore}`);
  }
  if (config.minOneSidedLower !== null && (lower === null || lower < config.minOneSidedLower)) {
    failures.push(`one-sided 95% lower bound ${formatRate(lower)} < ${config.minOneSidedLower}`);
  }
  if (config.maxUnresolvedRate !== null && (unresolved === null || unresolved > config.maxUnresolvedRate)) {
    failures.push(`unresolved rate ${formatRate(unresolved)} > ${config.maxUnresolvedRate}`);
  }
  if (config.maxOperationalFailures !== null && operationalFailures > config.maxOperationalFailures) {
    failures.push(`operational failures ${operationalFailures} > ${config.maxOperationalFailures}`);
  }
  if (failures.length > 0) {
    process.stderr.write(`Arena gate failed: ${failures.join("; ")}\n`);
    process.exitCode = 1;
  }
}

function readConfig(args: string[]): ArenaConfig {
  const raw = new Map<string, string>();
  for (const arg of args) {
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const equals = arg.indexOf("=");
    raw.set(equals < 0 ? arg.slice(2) : arg.slice(2, equals), equals < 0 ? "true" : arg.slice(equals + 1));
  }
  if (raw.has("help")) {
    printHelp();
    process.exit(0);
  }
  const games = integer(raw, "games", 8, 2);
  if (games % 2 !== 0) throw new Error("--games must be even so every opening is played with colors swapped");
  const openingMode = text(raw, "opening-mode", "mixed") as OpeningMode;
  if (!["curated", "generated", "mixed"].includes(openingMode)) {
    throw new Error("--opening-mode must be curated, generated, or mixed");
  }
  const allStyles = boolean(raw, "all-styles", false);
  const styleNames = Object.keys(AI_STYLES);
  const challengerStyles = allStyles ? styleNames : list(raw, "challenger-styles", ["doctrinal"]);
  const opponentStyles = allStyles ? styleNames : list(raw, "opponent-styles", ["doctrinal"]);
  validateStyles(challengerStyles);
  validateStyles(opponentStyles);
  return {
    seed: integer(raw, "seed", 0x51a700, 0),
    games,
    maxActions: integer(raw, "max-actions", 240, 1),
    repetitionCount: integer(raw, "repetition", 3, 2),
    openingMode,
    openingCount: integer(raw, "opening-count", 12, 1),
    openingPlies: integer(raw, "opening-plies", 8, 1),
    challenger: difficulty(raw, "challenger", "very-hard"),
    opponent: difficulty(raw, "opponent", "hard"),
    challengerStyles,
    opponentStyles,
    styleOffset: integer(raw, "style-offset", 0, 0),
    veryHardBudgetMs: finiteNumber(raw, "budget-ms", 500, 1),
    veryHardNodeBudget: finiteNumber(raw, "node-budget", 2_000_000, 1),
    veryHardMaxDepth: integer(raw, "depth", 6, 1),
    veryHardEngine: veryHardEngine(raw),
    veryHardTiming: timingMode(raw),
    veryHardPlatform: platform(raw),
    veryHardFloor: floorMode(raw),
    manifestIn: nullableText(raw, "manifest-in"),
    manifestOut: nullableText(raw, "manifest-out"),
    output: nullableText(raw, "output"),
    summaryOnly: boolean(raw, "summary-only", false),
    progressEvery: integer(raw, "progress-every", 0, 0),
    minPessimisticScore: optionalRate(raw, "min-pessimistic-score"),
    minOneSidedLower: optionalRate(raw, "min-one-sided-lower"),
    maxUnresolvedRate: optionalRate(raw, "max-unresolved-rate"),
    maxOperationalFailures: optionalInteger(raw, "max-operational-failures", 0),
    searchTrace: boolean(raw, "search-trace", false)
  };
}

function readManifest(path: string): ArenaManifest {
  return JSON.parse(readFileSync(resolve(path), "utf8")) as ArenaManifest;
}

function validateManifest(manifest: ArenaManifest, config: ArenaConfig): void {
  if (!manifest || typeof manifest !== "object" ||
      manifest.format !== "linith-ai-arena-manifest" || manifest.version !== 1) {
    throw new Error("unsupported arena manifest format");
  }
  if (!Array.isArray(manifest.games) || manifest.games.length === 0 || manifest.games.length % 2 !== 0) {
    throw new Error("arena manifest must contain a non-empty, even number of games");
  }
  const replayIdentity: Array<[string, unknown, unknown]> = [
    ["challenger", manifest.challenger, config.challenger],
    ["opponent", manifest.opponent, config.opponent],
    ["seed", manifest.seed, config.seed],
    ["maxActions", manifest.maxActions, config.maxActions],
    ["repetitionCount", manifest.repetitionCount, config.repetitionCount],
    ["games", manifest.games.length, config.games]
  ];
  for (const [field, actual, requested] of replayIdentity) {
    if (actual !== requested) {
      throw new Error(`arena manifest ${field} (${String(actual)}) does not match requested ${field} (${String(requested)})`);
    }
  }

  const gameIds = new Set<string>();
  const pairs = new Map<string, ManifestGame[]>();
  for (const game of manifest.games) {
    if (!game || typeof game !== "object") throw new Error("arena manifest contains a malformed game");
    if (typeof game.id !== "string" || game.id.length === 0) throw new Error("arena manifest contains a game without an id");
    if (gameIds.has(game.id)) throw new Error(`arena manifest contains duplicate game id ${game.id}`);
    gameIds.add(game.id);
    if (typeof game.pairId !== "string" || game.pairId.length === 0) {
      throw new Error(`manifest game ${game.id} does not contain a pair id`);
    }
    if (typeof game.openingId !== "string" || game.openingId.length === 0) {
      throw new Error(`manifest game ${game.id} does not contain an opening id`);
    }
    if (game.challengerSide !== SUN && game.challengerSide !== MOON) {
      throw new Error(`manifest game ${game.id} has an invalid challenger side`);
    }
    if (!Number.isSafeInteger(game.decisionSeed) || game.decisionSeed < 0) {
      throw new Error(`manifest game ${game.id} has an invalid decision seed`);
    }
    if (!game.opening || !Array.isArray(game.opening.board) ||
        game.opening.board.length !== 10 ||
        game.opening.board.some((row) => !Array.isArray(row) || row.length !== 10)) {
      throw new Error(`manifest game ${game.id} does not contain a 10x10 board`);
    }
    if (game.opening.current !== SUN && game.opening.current !== MOON) {
      throw new Error(`manifest game ${game.id} has an invalid opening player`);
    }
    if (!Number.isSafeInteger(game.opening.movesLeft) || game.opening.movesLeft < 1) {
      throw new Error(`manifest game ${game.id} has invalid movesLeft`);
    }
    validateStyles([game.challengerStyle, game.opponentStyle]);
    const pair = pairs.get(game.pairId) ?? [];
    pair.push(game);
    pairs.set(game.pairId, pair);
  }

  for (const [pairId, pair] of pairs) {
    if (pair.length !== 2) throw new Error(`arena manifest pair ${pairId} must contain exactly two games`);
    const [first, second] = pair;
    if (first.challengerSide === second.challengerSide) {
      throw new Error(`arena manifest pair ${pairId} must swap challenger colors`);
    }
    if (first.openingId !== second.openingId ||
        stableStringify(first.opening) !== stableStringify(second.opening)) {
      throw new Error(`arena manifest pair ${pairId} must use the exact same opening`);
    }
    if (first.challengerStyle !== second.challengerStyle || first.opponentStyle !== second.opponentStyle) {
      throw new Error(`arena manifest pair ${pairId} must use the exact same styles`);
    }
    if (first.decisionSeed !== second.decisionSeed) {
      throw new Error(`arena manifest pair ${pairId} must use the exact same decision seed`);
    }
  }
}

function publicConfig(config: ArenaConfig): Record<string, unknown> {
  const { manifestIn, manifestOut, output, ...portable } = config;
  return { ...portable, manifestIn, manifestOut, output };
}

function stateFromCoordinates(coordinates: [[number, number], [number, number]]): SearchState {
  const board = emptyBoard();
  board[coordinates[0][0]][coordinates[0][1]] = SWAN_SUN;
  board[coordinates[1][0]][coordinates[1][1]] = SWAN_MOON;
  return { board, current: MOON, movesLeft: 1 };
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function other(player: Player): Player {
  return player === SUN ? MOON : SUN;
}

function emptySearchTotals(): SearchTotals {
  return {
    searches: 0,
    bookHits: 0,
    continuationHits: 0,
    nodes: 0,
    engineElapsedMs: 0,
    elapsedMs: 0,
    completedDepth: 0,
    fallbackSearches: 0,
    stops: {}
  };
}

function mergeSearch(target: SearchTotals, source: SearchTotals): void {
  target.searches += source.searches;
  target.bookHits += source.bookHits;
  target.continuationHits += source.continuationHits;
  target.nodes += source.nodes;
  target.engineElapsedMs += source.engineElapsedMs;
  target.elapsedMs += source.elapsedMs;
  target.completedDepth += source.completedDepth;
  target.fallbackSearches += source.fallbackSearches;
  for (const [reason, amount] of Object.entries(source.stops) as Array<[VeryHardStopReason, number]>) {
    target.stops[reason] = (target.stops[reason] ?? 0) + amount;
  }
}

function count(results: GameResult[], classification: GameClassification): number {
  return results.filter((result) => result.classification === classification).length;
}

function wilson(successes: number, trials: number, z: number): [number, number] {
  const p = successes / trials;
  const z2 = z * z;
  const denominator = 1 + z2 / trials;
  const centre = (p + z2 / (2 * trials)) / denominator;
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator;
  return [Math.max(0, centre - margin), Math.min(1, centre + margin)];
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

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, child]) => `${JSON.stringify(key)}:${stableStringify(child)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(resolve(path), `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function describeNativeArtifact(): {
  kind: "wasm";
  sha256: string;
  bytes: number;
  openingBookSha256: string;
  openingBookBytes: number;
} {
  const bytes = readFileSync(resolve("src/renderer/game/veryHard/native/linith-core.wasm"));
  const openingBook = readFileSync(resolve("src/renderer/game/veryHard/openingBookData.ts"));
  return {
    kind: "wasm",
    sha256: createHash("sha256").update(bytes).digest("hex"),
    bytes: bytes.byteLength,
    openingBookSha256: createHash("sha256").update(openingBook).digest("hex"),
    openingBookBytes: openingBook.byteLength
  };
}

function difficulty(raw: Map<string, string>, name: string, fallback: Difficulty): Difficulty {
  const value = text(raw, name, fallback) as Difficulty;
  if (!["easy", "medium", "hard", "very-hard"].includes(value)) {
    throw new Error(`--${name} must be easy, medium, hard, or very-hard`);
  }
  return value;
}

function veryHardEngine(raw: Map<string, string>): "native" | "typescript" {
  const value = text(raw, "very-hard-engine", "native");
  if (value !== "native" && value !== "typescript") {
    throw new Error("--very-hard-engine must be native or typescript");
  }
  return value;
}

function timingMode(raw: Map<string, string>): TimingMode {
  const value = text(raw, "timing", "shipping");
  if (value !== "shipping" && value !== "fixed") {
    throw new Error("--timing must be shipping or fixed");
  }
  return value;
}

function platform(raw: Map<string, string>): VeryHardPlatform {
  const value = text(raw, "platform", "desktop");
  if (value !== "desktop" && value !== "browser" && value !== "android") {
    throw new Error("--platform must be desktop, browser, or android");
  }
  return value;
}

function floorMode(raw: Map<string, string>): FloorMode {
  const value = text(raw, "floor", "never");
  if (value !== "reply" && value !== "depth1" && value !== "never") {
    throw new Error("--floor must be reply, depth1, or never");
  }
  return value;
}

function validateStyles(styles: string[]): void {
  for (const style of styles) if (!(style in AI_STYLES)) throw new Error(`unknown AI style: ${style}`);
}

function text(raw: Map<string, string>, name: string, fallback: string): string {
  return raw.get(name) ?? fallback;
}

function nullableText(raw: Map<string, string>, name: string): string | null {
  return raw.get(name) ?? null;
}

function list(raw: Map<string, string>, name: string, fallback: string[]): string[] {
  const value = raw.get(name);
  if (value === undefined) return fallback;
  const values = value.split(",").map((item) => item.trim()).filter(Boolean);
  if (values.length === 0) throw new Error(`--${name} cannot be empty`);
  return values;
}

function boolean(raw: Map<string, string>, name: string, fallback: boolean): boolean {
  const value = raw.get(name);
  if (value === undefined) return fallback;
  if (value === "true" || value === "1") return true;
  if (value === "false" || value === "0") return false;
  throw new Error(`--${name} must be true or false`);
}

function integer(raw: Map<string, string>, name: string, fallback: number, minimum: number): number {
  const value = Number(raw.get(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${name} must be an integer >= ${minimum}`);
  return value;
}

function optionalInteger(raw: Map<string, string>, name: string, minimum: number): number | null {
  if (!raw.has(name)) return null;
  return integer(raw, name, minimum, minimum);
}

function finiteNumber(raw: Map<string, string>, name: string, fallback: number, minimum: number): number {
  const value = Number(raw.get(name) ?? fallback);
  if (!Number.isFinite(value) || value < minimum) throw new Error(`--${name} must be a number >= ${minimum}`);
  return value;
}

function optionalRate(raw: Map<string, string>, name: string): number | null {
  if (!raw.has(name)) return null;
  const value = Number(raw.get(name));
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`--${name} must be between 0 and 1`);
  return value;
}

function formatRate(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(4);
}

function round(value: number): number {
  return Number(value.toFixed(2));
}

function printHelp(): void {
  console.log(`Linith reproducible paired AI arena

Core:
  --challenger=very-hard       easy | medium | hard | very-hard
  --opponent=hard              easy | medium | hard | very-hard
  --games=8                    even; adjacent games swap colors
  --seed=5351168               deterministic corpus and decisions
  --max-actions=240            unresolved rather than scored as a draw
  --repetition=3               repeated positions become unresolved

Corpus/styles:
  --opening-mode=mixed         curated | generated | mixed
  --opening-count=12
  --opening-plies=8
  --challenger-styles=doctrinal,constrictor
  --opponent-styles=doctrinal
  --all-styles=true            full style cross-product
  --style-offset=0             deterministic matchup rotation for sharding

Very Hard search:
  --very-hard-engine=native   native | typescript
  --timing=shipping           shipping adaptive clock | fixed
  --platform=desktop          desktop | browser | android
  --budget-ms=500             used only by --timing=fixed
  --node-budget=2000000 --depth=6
  --floor=never                never (shipping) | reply | depth1

Reproduction/output:
  --manifest-out=arena-manifest.json
  --manifest-in=arena-manifest.json
  --output=arena-report.json
  --summary-only=true --progress-every=20
  --search-trace=true          per-Very-Hard decision diagnostics

CI gates:
  --min-pessimistic-score=0.70
  --min-one-sided-lower=0.65
  --max-unresolved-rate=0.005
  --max-operational-failures=0`);
}
