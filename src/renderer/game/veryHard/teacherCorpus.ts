import { EMPTY, SWAN_MOON, SWAN_SUN, type Board, type Player } from "../encirclement";
import {
  actionKey,
  applyAction,
  boardKey,
  generateLegalActions,
  type LinithAction,
  type SearchState
} from "../rulesEngine";

export const TEACHER_CORPUS_GENERATOR_VERSION = "linith-classical-teacher-1";

export interface TeacherSearchConfig {
  nodeBudget: number;
  maxDepth: number;
  tacticalDepth: number;
  exactDepth: number;
  style: string;
}

export interface TeacherSearchObservation {
  action: LinithAction | null;
  score: number;
  completedDepth: number;
  attemptedDepth: number;
  nodes: number;
  stopReason: string;
  exactSolved: boolean;
}

export type TeacherSearchRunner = (
  state: SearchState,
  config: TeacherSearchConfig
) => TeacherSearchObservation;

export interface TeacherCorpusHooks {
  /** Reset engine-local game history before this independent logical game. */
  beforeGame?: (gameIndex: number) => void;
  /** Commit only the action accepted by the live rule model. */
  afterAction?: (state: SearchState, action: LinithAction) => boolean | void;
}

export interface TeacherCorpusOptions {
  seed?: number;
  /** Total logical game count across every shard. */
  games?: number;
  /** Zero-based modulo shard.  Shards are mergeable independent jobs. */
  shardIndex?: number;
  shardCount?: number;
  openingPlies?: number;
  maxActions?: number;
  repetitionCount?: number;
  sampleStride?: number;
  samplesPerGame?: number;
  minCompletedDepth?: number;
  search?: Partial<TeacherSearchConfig>;
  startingStates?: readonly SearchState[];
  generator?: string;
  /** Exact SHA-256 (or equivalent) of the teacher engine artifact. */
  artifactFingerprint?: string;
}

export interface TeacherCorpusSample {
  id: string;
  /** Stable game-level grouping key used to prevent train/holdout leakage. */
  groupId: string;
  state: SearchState;
  perspective: Player;
  /** Actual terminal game outcome from `perspective`, never an adjudication. */
  target: -1 | 0 | 1;
  weight: number;
  terminalOutcome: "sun" | "moon" | "draw";
  teacher: {
    action: LinithAction;
    score: number;
    completedDepth: number;
    attemptedDepth: number;
    nodes: number;
    stopReason: string;
    exactSolved: boolean;
  };
}

export interface TeacherCorpus {
  format: "linith-evaluation-corpus";
  version: 1;
  generator: string;
  artifactFingerprint: string | null;
  config: {
    seed: number;
    games: number;
    shardIndex: number;
    shardCount: number;
    openingPlies: number;
    maxActions: number;
    repetitionCount: number;
    sampleStride: number;
    samplesPerGame: number;
    minCompletedDepth: number;
    startingStatesFingerprint: string;
    search: TeacherSearchConfig;
  };
  gamesAttempted: number;
  completedGames: number;
  discarded: {
    repetition: number;
    actionLimit: number;
    invalid: number;
    insufficientDepth: number;
  };
  samples: TeacherCorpusSample[];
}

interface TracePosition {
  state: SearchState;
  observation: TeacherSearchObservation;
}

interface NormalizedOptions {
  seed: number;
  games: number;
  shardIndex: number;
  shardCount: number;
  openingPlies: number;
  maxActions: number;
  repetitionCount: number;
  sampleStride: number;
  samplesPerGame: number;
  minCompletedDepth: number;
  search: TeacherSearchConfig;
  startingStates: readonly SearchState[];
  startingStatesFingerprint: string;
  generator: string;
  artifactFingerprint: string | null;
}

const CURATED_STARTS: ReadonlyArray<readonly [readonly [number, number], readonly [number, number]]> = [
  [[4, 4], [6, 6]], [[5, 5], [3, 7]], [[3, 3], [6, 5]], [[6, 3], [3, 6]],
  [[2, 4], [7, 5]], [[4, 2], [5, 7]], [[2, 2], [7, 7]], [[7, 2], [2, 7]],
  [[1, 4], [8, 5]], [[4, 1], [5, 8]], [[3, 5], [6, 4]], [[5, 3], [4, 6]]
];

function positiveInteger(value: number | undefined, fallback: number, minimum = 1): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= minimum ? value : fallback;
}

function nonNegativeInteger(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : fallback;
}

function cloneState(state: SearchState): SearchState {
  return {
    board: state.board.map((row) => [...row]),
    current: state.current,
    movesLeft: state.movesLeft
  };
}

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function curatedStartingStates(): SearchState[] {
  return CURATED_STARTS.map(([sun, moon]) => {
    const board = emptyBoard();
    board[sun[0]][sun[1]] = SWAN_SUN;
    board[moon[0]][moon[1]] = SWAN_MOON;
    // The live setup executor starts Moon after the two initial placements.
    return { board, current: 2 as const, movesLeft: 1 };
  });
}

function normalize(options: TeacherCorpusOptions): NormalizedOptions {
  const search: TeacherSearchConfig = {
    nodeBudget: positiveInteger(options.search?.nodeBudget, 1_000_000, 100_000),
    maxDepth: positiveInteger(options.search?.maxDepth, 7, 4),
    tacticalDepth: positiveInteger(options.search?.tacticalDepth, 3),
    exactDepth: positiveInteger(options.search?.exactDepth, 12),
    style: options.search?.style || "doctrinal"
  };
  const startingStates = options.startingStates?.length
    ? options.startingStates.map(cloneState)
    : curatedStartingStates();
  const shardCount = positiveInteger(options.shardCount, 1);
  const shardIndex = nonNegativeInteger(options.shardIndex, 0);
  if (shardIndex >= shardCount) {
    throw new Error(`Teacher shard index ${shardIndex} is outside shard count ${shardCount}.`);
  }
  return {
    seed: nonNegativeInteger(options.seed, 0x7ea4c3) >>> 0,
    games: positiveInteger(options.games, 8),
    shardIndex,
    shardCount,
    openingPlies: positiveInteger(options.openingPlies, 8, 0),
    maxActions: positiveInteger(options.maxActions, 240),
    repetitionCount: positiveInteger(options.repetitionCount, 3, 2),
    sampleStride: positiveInteger(options.sampleStride, 3),
    samplesPerGame: positiveInteger(options.samplesPerGame, 24),
    minCompletedDepth: positiveInteger(options.minCompletedDepth, 3),
    search,
    startingStates,
    startingStatesFingerprint: hashString(startingStates.map(boardKey).join("|")),
    generator: options.generator ?? TEACHER_CORPUS_GENERATOR_VERSION,
    artifactFingerprint: options.artifactFingerprint?.trim() || null
  };
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
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function reachableOpening(base: SearchState, plies: number, seed: number): SearchState {
  let state = cloneState(base);
  const random = seededRandom(seed);
  for (let ply = 0; ply < plies; ply += 1) {
    const actions = generateLegalActions(state);
    if (actions.length === 0) break;
    // Generation is random only before teacher play, and is fully seeded.  It
    // broadens the corpus without assigning labels or adjudicating positions.
    const offset = Math.floor(random() * actions.length);
    let advanced = false;
    for (let index = 0; index < actions.length; index += 1) {
      const next = applyAction(state, actions[(offset + index) % actions.length]);
      if (!next || next.outcome) continue;
      state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
      advanced = true;
      break;
    }
    if (!advanced) break;
  }
  return state;
}

function outcomeTarget(outcome: "sun" | "moon" | "draw", perspective: Player): -1 | 0 | 1 {
  if (outcome === "draw") return 0;
  return (outcome === "sun" ? 1 : 2) === perspective ? 1 : -1;
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function samplesFromTerminalTrace(
  trace: readonly TracePosition[],
  outcome: "sun" | "moon" | "draw",
  gameIndex: number,
  options: NormalizedOptions
): TeacherCorpusSample[] {
  const candidates = trace.filter((_position, index) => index % options.sampleStride === 0);
  const selected = candidates.length <= options.samplesPerGame
    ? candidates
    : Array.from({ length: options.samplesPerGame }, (_unused, index) => {
      const sourceIndex = Math.floor(index * candidates.length / options.samplesPerGame);
      return candidates[sourceIndex];
    });
  const result: TeacherCorpusSample[] = [];
  const groupId = `g${String(gameIndex + 1).padStart(6, "0")}`;
  for (let index = 0; index < selected.length; index += 1) {
    const { state, observation } = selected[index];
    for (const perspective of [1, 2] as const) {
      const key = `${boardKey(state)}:${perspective}`;
      result.push({
        id: `${groupId}-s${String(index + 1).padStart(3, "0")}`
          + `-p${perspective}-${hashString(key)}`,
        groupId,
        state: cloneState(state),
        perspective,
        target: outcomeTarget(outcome, perspective),
        weight: 1,
        terminalOutcome: outcome,
        teacher: {
          action: observation.action!,
          score: perspective === state.current ? observation.score : -observation.score,
          completedDepth: observation.completedDepth,
          attemptedDepth: observation.attemptedDepth,
          nodes: observation.nodes,
          stopReason: observation.stopReason,
          exactSolved: observation.exactSolved
        }
      });
    }
  }
  return result;
}

/**
 * Produce labels only from a terminal result returned by the live rules model.
 * Repetitions, action limits, shallow searches, illegal actions and crashes add
 * no samples; the discarded counters make that filtering auditable.
 */
export function generateTeacherCorpus(
  inputOptions: TeacherCorpusOptions,
  search: TeacherSearchRunner,
  hooks: TeacherCorpusHooks = {}
): TeacherCorpus {
  const options = normalize(inputOptions);
  const discarded = { repetition: 0, actionLimit: 0, invalid: 0, insufficientDepth: 0 };
  const samples: TeacherCorpusSample[] = [];
  let completedGames = 0;
  let gamesAttempted = 0;

  for (
    let gameIndex = options.shardIndex;
    gameIndex < options.games;
    gameIndex += options.shardCount
  ) {
    gamesAttempted += 1;
    try {
      hooks.beforeGame?.(gameIndex);
    } catch {
      discarded.invalid += 1;
      continue;
    }
    let state = reachableOpening(
      options.startingStates[gameIndex % options.startingStates.length],
      options.openingPlies + (gameIndex % 3),
      mixSeed(options.seed, gameIndex, 0x0b3a)
    );
    const repetitions = new Map<string, number>([[boardKey(state), 1]]);
    const trace: TracePosition[] = [];
    let rejection: keyof typeof discarded | null = null;
    let terminalOutcome: "sun" | "moon" | "draw" | null = null;

    for (let actionNumber = 0; actionNumber < options.maxActions; actionNumber += 1) {
      let observation: TeacherSearchObservation;
      try {
        observation = search(state, options.search);
      } catch {
        rejection = "invalid";
        break;
      }
      if (!observation.action || !Number.isFinite(observation.score)) {
        rejection = "invalid";
        break;
      }
      if (observation.completedDepth < options.minCompletedDepth) {
        rejection = "insufficientDepth";
        break;
      }
      const next = applyAction(state, observation.action);
      if (!next) {
        rejection = "invalid";
        break;
      }
      try {
        if (hooks.afterAction?.(state, observation.action) === false) {
          rejection = "invalid";
          break;
        }
      } catch {
        rejection = "invalid";
        break;
      }
      trace.push({ state: cloneState(state), observation });
      if (next.outcome) {
        terminalOutcome = next.outcome;
        break;
      }
      state = { board: next.board, current: next.current, movesLeft: next.movesLeft };
      const key = boardKey(state);
      const seen = (repetitions.get(key) ?? 0) + 1;
      repetitions.set(key, seen);
      if (seen >= options.repetitionCount) {
        rejection = "repetition";
        break;
      }
    }

    if (!terminalOutcome) {
      if (!rejection) rejection = "actionLimit";
      discarded[rejection] += 1;
      continue;
    }
    completedGames += 1;
    samples.push(...samplesFromTerminalTrace(trace, terminalOutcome, gameIndex, options));
  }

  samples.sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
  return {
    format: "linith-evaluation-corpus",
    version: 1,
    generator: options.generator,
    artifactFingerprint: options.artifactFingerprint,
    config: {
      seed: options.seed,
      games: options.games,
      shardIndex: options.shardIndex,
      shardCount: options.shardCount,
      openingPlies: options.openingPlies,
      maxActions: options.maxActions,
      repetitionCount: options.repetitionCount,
      sampleStride: options.sampleStride,
      samplesPerGame: options.samplesPerGame,
      minCompletedDepth: options.minCompletedDepth,
      startingStatesFingerprint: options.startingStatesFingerprint,
      search: { ...options.search }
    },
    gamesAttempted,
    completedGames,
    discarded,
    samples
  };
}

function corpusCompatibilityKey(corpus: TeacherCorpus): string {
  const { shardIndex: _shardIndex, shardCount: _shardCount, ...config } = corpus.config;
  return JSON.stringify({
    format: corpus.format,
    version: corpus.version,
    generator: corpus.generator,
    artifactFingerprint: corpus.artifactFingerprint,
    config
  });
}

function expectedGamesInShard(games: number, shardIndex: number, shardCount: number): number {
  if (shardIndex >= games) return 0;
  return Math.floor((games - 1 - shardIndex) / shardCount) + 1;
}

/**
 * Merge a complete modulo-sharded run into the exact shape emitted by a
 * one-process run.  Mixing engine artifacts, seeds, search settings, partial
 * shard sets, or duplicate sample ids is rejected rather than silently
 * contaminating training data.
 */
export function mergeTeacherCorpusShards(shards: readonly TeacherCorpus[]): TeacherCorpus {
  if (shards.length === 0) throw new Error("At least one teacher-corpus shard is required.");
  const ordered = [...shards].sort((left, right) => left.config.shardIndex - right.config.shardIndex);
  const first = ordered[0];
  const shardCount = first.config.shardCount;
  if (shardCount !== ordered.length) {
    throw new Error(`Expected all ${shardCount} teacher shards, received ${ordered.length}.`);
  }
  const compatibilityKey = corpusCompatibilityKey(first);
  const seenShardIndices = new Set<number>();
  const seenSampleIds = new Set<string>();
  for (const corpus of ordered) {
    if (corpusCompatibilityKey(corpus) !== compatibilityKey
      || corpus.config.shardCount !== shardCount) {
      throw new Error("Teacher shards have incompatible engine provenance or generation settings.");
    }
    const shardIndex = corpus.config.shardIndex;
    if (shardIndex < 0 || shardIndex >= shardCount || seenShardIndices.has(shardIndex)) {
      throw new Error(`Teacher shard index ${shardIndex} is invalid or duplicated.`);
    }
    seenShardIndices.add(shardIndex);
    const expectedGames = expectedGamesInShard(corpus.config.games, shardIndex, shardCount);
    if (corpus.gamesAttempted !== expectedGames) {
      throw new Error(
        `Teacher shard ${shardIndex} attempted ${corpus.gamesAttempted} games; expected ${expectedGames}.`
      );
    }
    const classifiedGames = corpus.completedGames + Object.values(corpus.discarded)
      .reduce((sum, count) => sum + count, 0);
    if (classifiedGames !== corpus.gamesAttempted) {
      throw new Error(`Teacher shard ${shardIndex} has inconsistent terminal/discard counts.`);
    }
    for (const sample of corpus.samples) {
      if (seenSampleIds.has(sample.id)) throw new Error(`Duplicate teacher sample id ${sample.id}.`);
      seenSampleIds.add(sample.id);
    }
  }
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    if (!seenShardIndices.has(shardIndex)) throw new Error(`Teacher shard ${shardIndex} is missing.`);
  }

  const samples = ordered.flatMap((corpus) => corpus.samples)
    .sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
  const discarded = {
    repetition: ordered.reduce((sum, corpus) => sum + corpus.discarded.repetition, 0),
    actionLimit: ordered.reduce((sum, corpus) => sum + corpus.discarded.actionLimit, 0),
    invalid: ordered.reduce((sum, corpus) => sum + corpus.discarded.invalid, 0),
    insufficientDepth: ordered.reduce((sum, corpus) => sum + corpus.discarded.insufficientDepth, 0)
  };
  return {
    format: first.format,
    version: first.version,
    generator: first.generator,
    artifactFingerprint: first.artifactFingerprint,
    config: { ...first.config, shardIndex: 0, shardCount: 1 },
    gamesAttempted: first.config.games,
    completedGames: ordered.reduce((sum, corpus) => sum + corpus.completedGames, 0),
    discarded,
    samples
  };
}

export function serializeTeacherCorpus(corpus: TeacherCorpus): string {
  return `${JSON.stringify(corpus, null, 2)}\n`;
}
