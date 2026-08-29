import { availableParallelism } from "node:os";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AI_STYLES } from "../src/renderer/game/aiStyles";

type Opponent = "easy" | "medium" | "hard";
type Classification =
  | "win"
  | "loss"
  | "terminal-draw"
  | "repetition-unresolved"
  | "action-limit-unresolved"
  | "invalid"
  | "crash";

export interface LadderGame {
  id: string;
  pairId: string;
  openingId: string;
  challengerSide: 1 | 2;
  challengerStyle: string;
  opponentStyle: string;
  classification: Classification;
  winner: 1 | 2 | null;
  actions: number;
  illegalActions: number;
  nullActions: number;
  crashes: number;
}

export interface NativeArtifactRecord {
  kind: "wasm";
  sha256: string;
  bytes: number;
  openingBookSha256: string;
  openingBookBytes: number;
}

interface ArenaReport {
  kind: "linith-ai-arena";
  manifestId: string;
  config: { opponent: Opponent; veryHardEngine?: "native" | "typescript"; [key: string]: unknown };
  games: LadderGame[];
  veryHardArtifact: NativeArtifactRecord | null;
  search?: {
    searches: number;
    bookHits: number;
    continuationHits: number;
    nodes: number;
    engineElapsedMs: number;
    elapsedMs: number;
    completedDepth: number;
    fallbackSearches: number;
    stops: Record<string, number>;
  };
}

interface TaggedGame extends LadderGame {
  shardId: string;
  manifestId: string;
  opponent: Opponent;
}

interface LadderJob {
  id: string;
  opponent: Opponent;
  games: number;
  seed: number;
  reportPath: string;
  arguments: string[];
  reuseExisting: boolean;
  expectedConfig: Record<string, unknown>;
  expectedArtifact: NativeArtifactRecord | null;
}

interface GateTarget {
  score: number;
  lower: number;
}

interface LadderConfig {
  seed: number;
  opponents: Opponent[];
  gamesPerOpponent: number;
  shardsPerOpponent: number;
  workers: number;
  output: string;
  shardDirectory: string;
  maxActions: number;
  repetition: number;
  openingMode: "curated" | "generated" | "mixed";
  openingCount: number;
  openingPlies: number;
  timing: "shipping" | "fixed";
  platform: "desktop" | "browser" | "android";
  budgetMs: number;
  nodeBudget: number;
  depth: number;
  engine: "native" | "typescript";
  floor: "reply" | "depth1" | "never";
  allStyles: boolean;
  challengerStyles: string | null;
  opponentStyles: string | null;
  styleOffset: number;
  gates: boolean;
  maxUnresolvedRate: number;
  maxOperationalFailures: number;
  minSubgroupScore: number;
  searchTrace: boolean;
  resume: boolean;
}

const GATE_TARGETS: Record<Opponent, GateTarget> = {
  hard: { score: 0.70, lower: 0.65 },
  medium: { score: 0.80, lower: 0.75 },
  easy: { score: 0.90, lower: 0.85 }
};

export function aggregateLadderReports(
  reports: Array<{ shardId: string; report: ArenaReport }>,
  alpha = 0.05
): Record<string, unknown> {
  const games: TaggedGame[] = reports.flatMap(({ shardId, report }) => {
    if (report.kind !== "linith-ai-arena" || !Array.isArray(report.games)) {
      throw new Error(`Shard ${shardId} is not a full Linith arena report.`);
    }
    return report.games.map((game) => ({
      ...game,
      shardId,
      manifestId: report.manifestId,
      opponent: report.config.opponent
    }));
  });
  const opponents = [...new Set(games.map((game) => game.opponent))];
  const artifacts = [...new Map(
    reports
      .map(({ report }) => report.veryHardArtifact)
      .filter((artifact): artifact is NativeArtifactRecord => isNativeArtifactRecord(artifact))
      .map((artifact) => [`${artifact.sha256}:${artifact.bytes}`, artifact])
  ).values()];
  const artifactIntegrityFailures = artifactIntegrityFailureCount(reports.map(({ report }) => report));
  return {
    kind: "linith-ai-ladder",
    formatVersion: 1,
    confidence: {
      alpha,
      method: "distribution-free Hoeffding lower bound over color-swapped opening pairs",
      unresolvedPolicy: "repetition/action-limit/invalid/crash score zero"
    },
    games: games.length,
    pairs: games.length / 2,
    veryHardArtifacts: artifacts,
    artifactIntegrityFailures,
    search: summarizeSearchReports(reports.map(({ report }) => report)),
    opponents: opponents.map((opponent) => ({
      ...summarizeOpponent(games.filter((game) => game.opponent === opponent), alpha),
      search: summarizeSearchReports(
        reports.map(({ report }) => report).filter((report) => report.config.opponent === opponent)
      )
    })),
    shards: reports.map(({ shardId, report }) => ({
      shardId,
      manifestId: report.manifestId,
      opponent: report.config.opponent,
      games: report.games.length,
      veryHardArtifact: report.veryHardArtifact ?? null
    }))
  };
}

function isNativeArtifactRecord(value: unknown): value is NativeArtifactRecord {
  if (!value || typeof value !== "object") return false;
  const artifact = value as Partial<NativeArtifactRecord>;
  return artifact.kind === "wasm" &&
    typeof artifact.sha256 === "string" && /^[0-9a-f]{64}$/i.test(artifact.sha256) &&
    Number.isSafeInteger(artifact.bytes) && (artifact.bytes ?? 0) > 0 &&
    typeof artifact.openingBookSha256 === "string" && /^[0-9a-f]{64}$/i.test(artifact.openingBookSha256) &&
    Number.isSafeInteger(artifact.openingBookBytes) && (artifact.openingBookBytes ?? 0) > 0;
}

function artifactIntegrityFailureCount(reports: ArenaReport[]): number {
  let failures = 0;
  let nativeReports = 0;
  let typescriptReports = 0;
  const nativeArtifacts = new Set<string>();
  for (const report of reports) {
    if (report.config?.veryHardEngine === "native") {
      nativeReports += 1;
      if (!isNativeArtifactRecord(report.veryHardArtifact)) {
        failures += 1;
      } else {
        nativeArtifacts.add(
          `${report.veryHardArtifact.sha256}:${report.veryHardArtifact.bytes}:` +
          `${report.veryHardArtifact.openingBookSha256}:${report.veryHardArtifact.openingBookBytes}`
        );
      }
      continue;
    }
    if (report.config?.veryHardEngine === "typescript") {
      typescriptReports += 1;
      if (report.veryHardArtifact !== null) failures += 1;
      continue;
    }
    failures += 1;
  }
  if (nativeReports > 0 && typescriptReports > 0) failures += 1;
  failures += Math.max(0, nativeArtifacts.size - 1);
  return failures;
}

export function describeCurrentNativeArtifact(): NativeArtifactRecord {
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

export function validateResumeReport(
  shardId: string,
  value: unknown,
  expectedConfig: Record<string, unknown>,
  expectedArtifact: NativeArtifactRecord | null
): ArenaReport {
  if (!value || typeof value !== "object") throw new Error(`Shard ${shardId} is not a report object.`);
  const report = value as Partial<ArenaReport>;
  if (report.kind !== "linith-ai-arena") throw new Error(`Shard ${shardId} has an invalid report kind.`);
  if (!report.config || typeof report.config !== "object") throw new Error(`Shard ${shardId} has no arena config.`);
  if (!Array.isArray(report.games)) throw new Error(`Shard ${shardId} has no game records.`);
  if (typeof report.manifestId !== "string" || report.manifestId.length === 0) {
    throw new Error(`Shard ${shardId} has no manifest identity.`);
  }
  for (const [field, expected] of Object.entries(expectedConfig)) {
    const actual = report.config[field];
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(
        `Shard ${shardId} config ${field} does not match the requested run ` +
        `(found ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}).`
      );
    }
  }
  if (report.games.length !== expectedConfig.games) {
    throw new Error(`Shard ${shardId} game records do not match the requested game count.`);
  }

  if (expectedArtifact === null) {
    if (report.veryHardArtifact !== null) {
      throw new Error(`Shard ${shardId} must record a null artifact for the TypeScript engine.`);
    }
  } else {
    if (!isNativeArtifactRecord(report.veryHardArtifact)) {
      throw new Error(`Shard ${shardId} is missing a valid native artifact record.`);
    }
    if (report.veryHardArtifact.sha256 !== expectedArtifact.sha256 ||
        report.veryHardArtifact.bytes !== expectedArtifact.bytes ||
        report.veryHardArtifact.openingBookSha256 !== expectedArtifact.openingBookSha256 ||
        report.veryHardArtifact.openingBookBytes !== expectedArtifact.openingBookBytes) {
      throw new Error(`Shard ${shardId} native artifact does not match the current WASM artifact.`);
    }
  }

  validateReportPairs(shardId, report.games);
  return report as ArenaReport;
}

function validateReportPairs(shardId: string, games: LadderGame[]): void {
  const ids = new Set<string>();
  const pairs = new Map<string, LadderGame[]>();
  for (const game of games) {
    if (!game || typeof game !== "object" || typeof game.id !== "string" || game.id.length === 0) {
      throw new Error(`Shard ${shardId} contains a malformed game record.`);
    }
    if (ids.has(game.id)) throw new Error(`Shard ${shardId} contains duplicate game id ${game.id}.`);
    ids.add(game.id);
    if (typeof game.pairId !== "string" || game.pairId.length === 0) {
      throw new Error(`Shard ${shardId} game ${game.id} has no pair id.`);
    }
    const pair = pairs.get(game.pairId) ?? [];
    pair.push(game);
    pairs.set(game.pairId, pair);
  }
  for (const [pairId, pair] of pairs) {
    if (pair.length !== 2 || pair[0].challengerSide === pair[1].challengerSide) {
      throw new Error(`Shard ${shardId} pair ${pairId} is not a color-swapped two-game pair.`);
    }
    if (pair[0].openingId !== pair[1].openingId ||
        pair[0].challengerStyle !== pair[1].challengerStyle ||
        pair[0].opponentStyle !== pair[1].opponentStyle) {
      throw new Error(`Shard ${shardId} pair ${pairId} does not share its opening and styles.`);
    }
  }
}

function summarizeSearchReports(reports: ArenaReport[]): Record<string, unknown> {
  const total = {
    searches: 0,
    bookHits: 0,
    continuationHits: 0,
    nodes: 0,
    engineElapsedMs: 0,
    elapsedMs: 0,
    completedDepth: 0,
    fallbackSearches: 0,
    stops: {} as Record<string, number>
  };
  for (const report of reports) {
    if (!report.search) continue;
    total.searches += report.search.searches;
    total.bookHits += report.search.bookHits;
    total.continuationHits += report.search.continuationHits;
    total.nodes += report.search.nodes;
    total.engineElapsedMs += report.search.engineElapsedMs;
    total.elapsedMs += report.search.elapsedMs;
    total.completedDepth += report.search.completedDepth;
    total.fallbackSearches += report.search.fallbackSearches;
    for (const [reason, count] of Object.entries(report.search.stops)) {
      total.stops[reason] = (total.stops[reason] ?? 0) + count;
    }
  }
  return {
    ...total,
    averageCompletedDepth: total.searches > 0 ? total.completedDepth / total.searches : null,
    averageEngineElapsedMs: total.searches > 0 ? total.engineElapsedMs / total.searches : null,
    averageSelectorElapsedMs: total.searches > 0 ? total.elapsedMs / total.searches : null,
    selectorOverheadMs: total.searches > 0 ? (total.elapsedMs - total.engineElapsedMs) / total.searches : null,
    nodesPerSecond: total.engineElapsedMs > 0 ? Math.round(total.nodes / (total.engineElapsedMs / 1000)) : null,
    fallbackRate: total.searches > 0 ? total.fallbackSearches / total.searches : null,
    continuationShare: total.searches + total.continuationHits > 0
      ? total.continuationHits / (total.searches + total.continuationHits)
      : null
  };
}

function summarizeOpponent(games: TaggedGame[], alpha: number): Record<string, unknown> {
  const byPair = new Map<string, TaggedGame[]>();
  for (const game of games) {
    const key = `${game.shardId}\u0000${game.pairId}`;
    const pair = byPair.get(key) ?? [];
    pair.push(game);
    byPair.set(key, pair);
  }

  let pairIntegrityFailures = 0;
  const pairScores: number[] = [];
  const pairScoreCounts: Record<string, number> = {};
  for (const pair of byPair.values()) {
    const colors = new Set(pair.map((game) => game.challengerSide));
    if (pair.length !== 2 || colors.size !== 2) {
      pairIntegrityFailures += 1;
      continue;
    }
    const score = pair.reduce((total, game) => total + gamePoints(game), 0) / 2;
    pairScores.push(score);
    const label = score.toFixed(2);
    pairScoreCounts[label] = (pairScoreCounts[label] ?? 0) + 1;
  }

  const mean = average(pairScores);
  const lower = hoeffdingLower(mean, pairScores.length, alpha);
  const classifications = classificationCounts(games);
  const operationalFailures = classifications.invalid + classifications.crash;
  const unresolved = classifications["repetition-unresolved"] + classifications["action-limit-unresolved"];
  const subgroup = (label: string, subset: TaggedGame[]) => ({
    label,
    games: subset.length,
    pessimisticScoreRate: subset.length > 0 ? average(subset.map(gamePoints)) : null
  });
  const styleKeys = [...new Set(games.map((game) => `${game.challengerStyle}\u0000${game.opponentStyle}`))];
  const challengerStyles = [...new Set(games.map((game) => game.challengerStyle))];
  const opponentStyles = [...new Set(games.map((game) => game.opponentStyle))];
  const openingSources = ["curated", "generated"].map((source) => subgroup(
    source,
    games.filter((game) => game.openingId.startsWith(source))
  ));

  return {
    opponent: games[0]?.opponent ?? null,
    games: games.length,
    pairs: pairScores.length,
    pairIntegrityFailures,
    classifications,
    wins: classifications.win,
    losses: classifications.loss,
    terminalDraws: classifications["terminal-draw"],
    unresolved,
    unresolvedRate: games.length > 0 ? unresolved / games.length : null,
    operationalFailures,
    pessimisticScoreRate: games.length > 0 ? average(games.map(gamePoints)) : null,
    pairMeanScore: mean,
    pairHoeffdingOneSided95Lower: lower,
    pairScoreCounts,
    byColor: [
      subgroup("sun", games.filter((game) => game.challengerSide === 1)),
      subgroup("moon", games.filter((game) => game.challengerSide === 2))
    ],
    byOpeningSource: openingSources,
    byChallengerStyle: challengerStyles.map((style) => subgroup(
      style,
      games.filter((game) => game.challengerStyle === style)
    )),
    byOpponentStyle: opponentStyles.map((style) => subgroup(
      style,
      games.filter((game) => game.opponentStyle === style)
    )),
    byStyleMatchup: styleKeys.map((key) => {
      const [challengerStyle, opponentStyle] = key.split("\u0000");
      return subgroup(
        `${challengerStyle} vs ${opponentStyle}`,
        games.filter((game) => game.challengerStyle === challengerStyle && game.opponentStyle === opponentStyle)
      );
    })
  };
}

function classificationCounts(games: TaggedGame[]): Record<Classification, number> {
  const counts: Record<Classification, number> = {
    win: 0,
    loss: 0,
    "terminal-draw": 0,
    "repetition-unresolved": 0,
    "action-limit-unresolved": 0,
    invalid: 0,
    crash: 0
  };
  for (const game of games) counts[game.classification] += 1;
  return counts;
}

function gamePoints(game: LadderGame): number {
  if (game.classification === "win") return 1;
  if (game.classification === "terminal-draw") return 0.5;
  return 0;
}

function average(values: number[]): number | null {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function hoeffdingLower(mean: number | null, clusters: number, alpha = 0.05): number | null {
  if (mean === null || clusters < 1) return null;
  return Math.max(0, mean - Math.sqrt(Math.log(1 / alpha) / (2 * clusters)));
}

function readConfig(arguments_: string[]): LadderConfig {
  const raw = new Map<string, string>();
  for (const argument of arguments_) {
    if (!argument.startsWith("--")) throw new Error(`Unexpected argument: ${argument}`);
    const equals = argument.indexOf("=");
    raw.set(equals < 0 ? argument.slice(2) : argument.slice(2, equals), equals < 0 ? "true" : argument.slice(equals + 1));
  }
  const opponents = list(raw, "opponents", ["hard", "medium", "easy"]) as Opponent[];
  for (const opponent of opponents) {
    if (opponent !== "easy" && opponent !== "medium" && opponent !== "hard") {
      throw new Error(`Unsupported ladder opponent: ${opponent}`);
    }
  }
  const workers = integer(raw, "workers", Math.min(16, availableParallelism()), 1);
  const gamesPerOpponent = evenInteger(raw, "games", 1_000, 2);
  const shardsPerOpponent = integer(raw, "shards", workers, 1);
  if (shardsPerOpponent > gamesPerOpponent / 2) {
    throw new Error("--shards cannot exceed the number of color-swapped pairs");
  }
  return {
    seed: integer(raw, "seed", 0x51a700, 0),
    opponents,
    gamesPerOpponent,
    shardsPerOpponent,
    workers,
    output: text(raw, "output", "native/very-hard/ladder-report.json"),
    shardDirectory: text(raw, "shard-dir", "native/very-hard/ladder-shards"),
    maxActions: integer(raw, "max-actions", 400, 1),
    repetition: integer(raw, "repetition", 3, 2),
    openingMode: oneOf(raw, "opening-mode", "mixed", ["curated", "generated", "mixed"]),
    openingCount: integer(raw, "opening-count", 8, 1),
    openingPlies: integer(raw, "opening-plies", 8, 1),
    timing: oneOf(raw, "timing", "shipping", ["shipping", "fixed"]),
    platform: oneOf(raw, "platform", "desktop", ["desktop", "browser", "android"]),
    budgetMs: number(raw, "budget-ms", 500, 1),
    nodeBudget: integer(raw, "node-budget", 2_000_000, 1),
    depth: integer(raw, "depth", 6, 1),
    engine: oneOf(raw, "very-hard-engine", "native", ["native", "typescript"]),
    floor: oneOf(raw, "floor", "never", ["reply", "depth1", "never"]),
    allStyles: boolean(raw, "all-styles", true),
    challengerStyles: raw.get("challenger-styles") ?? null,
    opponentStyles: raw.get("opponent-styles") ?? null,
    styleOffset: integer(raw, "style-offset", 0, 0),
    gates: boolean(raw, "gates", true),
    maxUnresolvedRate: number(raw, "max-unresolved-rate", 0.005, 0, 1),
    maxOperationalFailures: integer(raw, "max-operational-failures", 0, 0),
    minSubgroupScore: number(raw, "min-subgroup-score", 0.5, 0, 1),
    searchTrace: boolean(raw, "search-trace", false),
    resume: boolean(raw, "resume", false)
  };
}

function buildJobs(config: LadderConfig): LadderJob[] {
  const jobs: LadderJob[] = [];
  const challengerStyles = configuredStyles(config.allStyles, config.challengerStyles);
  const opponentStyles = configuredStyles(config.allStyles, config.opponentStyles);
  const expectedArtifact = config.engine === "native" ? describeCurrentNativeArtifact() : null;
  config.opponents.forEach((opponent, opponentIndex) => {
    const totalPairs = config.gamesPerOpponent / 2;
    const basePairs = Math.floor(totalPairs / config.shardsPerOpponent);
    const extraPairs = totalPairs % config.shardsPerOpponent;
    const challengerStyleCount = challengerStyles.length;
    const opponentStyleCount = opponentStyles.length;
    const stylePairCount = challengerStyleCount * opponentStyleCount;
    let styleBlockOffset = config.styleOffset % stylePairCount;
    for (let shard = 0; shard < config.shardsPerOpponent; shard += 1) {
      const games = 2 * (basePairs + Number(shard < extraPairs));
      const id = `${opponent}-${String(shard + 1).padStart(3, "0")}`;
      const reportPath = resolve(config.shardDirectory, `${id}.json`);
      const seed = mixSeed(config.seed, opponentIndex, shard);
      const shardOpeningCount = Math.min(config.openingCount, games / 2);
      const args = [
        `--challenger=very-hard`,
        `--opponent=${opponent}`,
        `--games=${games}`,
        `--seed=${seed}`,
        `--max-actions=${config.maxActions}`,
        `--repetition=${config.repetition}`,
        `--opening-mode=${config.openingMode}`,
        `--opening-count=${shardOpeningCount}`,
        `--opening-plies=${config.openingPlies}`,
        `--timing=${config.timing}`,
        `--platform=${config.platform}`,
        `--budget-ms=${config.budgetMs}`,
        `--node-budget=${config.nodeBudget}`,
        `--depth=${config.depth}`,
        `--very-hard-engine=${config.engine}`,
        `--floor=${config.floor}`,
        `--all-styles=${config.allStyles}`,
        `--style-offset=${styleBlockOffset}`,
        `--summary-only=false`,
        `--search-trace=${config.searchTrace}`,
        `--output=${reportPath}`
      ];
      if (config.challengerStyles) args.push(`--challenger-styles=${config.challengerStyles}`);
      if (config.opponentStyles) args.push(`--opponent-styles=${config.opponentStyles}`);
      jobs.push({
        id,
        opponent,
        games,
        seed,
        reportPath,
        arguments: args,
        reuseExisting: config.resume,
        expectedArtifact,
        expectedConfig: {
          challenger: "very-hard",
          opponent,
          games,
          seed,
          maxActions: config.maxActions,
          repetitionCount: config.repetition,
          openingMode: config.openingMode,
          openingCount: shardOpeningCount,
          openingPlies: config.openingPlies,
          challengerStyles,
          opponentStyles,
          styleOffset: styleBlockOffset,
          veryHardBudgetMs: config.budgetMs,
          veryHardNodeBudget: config.nodeBudget,
          veryHardMaxDepth: config.depth,
          veryHardEngine: config.engine,
          veryHardTiming: config.timing,
          veryHardPlatform: config.platform,
          veryHardFloor: config.floor,
          searchTrace: config.searchTrace
        }
      });
      styleBlockOffset = (
        styleBlockOffset + Math.ceil((games / 2) / shardOpeningCount)
      ) % stylePairCount;
    }
  });
  return jobs;
}

async function runJob(job: LadderJob): Promise<{ shardId: string; report: ArenaReport }> {
  if (job.reuseExisting && existsSync(job.reportPath)) {
    return { shardId: job.id, report: readAndValidateJobReport(job) };
  }
  const arenaPath = resolve("scripts/ai-arena.ts");
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(process.execPath, ["--import", "tsx", arenaPath, ...job.arguments], {
      stdio: ["ignore", "ignore", "pipe"]
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => { stderr = `${stderr}${chunk}`.slice(-16_384); });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`Arena shard ${job.id} failed (${signal ?? code}): ${stderr.trim()}`));
    });
  });
  return { shardId: job.id, report: readAndValidateJobReport(job) };
}

function readAndValidateJobReport(job: LadderJob): ArenaReport {
  const report = JSON.parse(readFileSync(job.reportPath, "utf8")) as unknown;
  return validateResumeReport(job.id, report, job.expectedConfig, job.expectedArtifact);
}

async function runPool<T, R>(items: T[], concurrency: number, run: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  let completed = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;
      results[index] = await run(items[index]);
      completed += 1;
      process.stderr.write(`Linith ladder: ${completed}/${items.length} shards complete\n`);
    }
  });
  await Promise.all(workers);
  return results;
}

function applyGates(report: Record<string, unknown>, config: LadderConfig): void {
  if (!config.gates) return;
  const failures: string[] = [];
  if ((report.artifactIntegrityFailures as number) !== 0) {
    failures.push("arena shards used different Very Hard artifacts");
  }
  const opponents = report.opponents as Array<Record<string, unknown>>;
  for (const summary of opponents) {
    const opponent = summary.opponent as Opponent;
    const target = GATE_TARGETS[opponent];
    const score = summary.pessimisticScoreRate as number | null;
    const lower = summary.pairHoeffdingOneSided95Lower as number | null;
    const unresolvedRate = summary.unresolvedRate as number | null;
    if (score === null || score < target.score) failures.push(`${opponent} score ${format(score)} < ${target.score}`);
    if (lower === null || lower < target.lower) failures.push(`${opponent} paired lower ${format(lower)} < ${target.lower}`);
    if (unresolvedRate === null || unresolvedRate > config.maxUnresolvedRate) {
      failures.push(`${opponent} unresolved ${format(unresolvedRate)} > ${config.maxUnresolvedRate}`);
    }
    if ((summary.operationalFailures as number) > config.maxOperationalFailures) {
      failures.push(`${opponent} operational failures ${summary.operationalFailures} > ${config.maxOperationalFailures}`);
    }
    if ((summary.pairIntegrityFailures as number) !== 0) failures.push(`${opponent} has malformed color pairs`);
    for (const dimension of ["byColor", "byOpeningSource", "byChallengerStyle"] as const) {
      for (const subgroup of summary[dimension] as Array<{ label: string; games: number; pessimisticScoreRate: number | null }>) {
        if (subgroup.games > 0 && (subgroup.pessimisticScoreRate ?? -1) <= config.minSubgroupScore) {
          failures.push(`${opponent} ${dimension} ${subgroup.label} is not positive (${format(subgroup.pessimisticScoreRate)})`);
        }
      }
    }
  }
  if (failures.length > 0) {
    process.stderr.write(`Ladder gate failed:\n- ${failures.join("\n- ")}\n`);
    process.exitCode = 1;
  }
}

async function main(): Promise<void> {
  const config = readConfig(process.argv.slice(2));
  mkdirSync(resolve(config.shardDirectory), { recursive: true });
  mkdirSync(dirname(resolve(config.output)), { recursive: true });
  const jobs = buildJobs(config);
  const reports = await runPool(jobs, config.workers, runJob);
  const aggregate = {
    ...aggregateLadderReports(reports),
    config: {
      ...config,
      output: portableRelativePath(config.output),
      shardDirectory: portableRelativePath(config.shardDirectory)
    }
  };
  writeFileSync(resolve(config.output), `${JSON.stringify(aggregate, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(aggregate, null, 2));
  applyGates(aggregate, config);
}

function portableRelativePath(path: string): string {
  return relative(process.cwd(), resolve(path)).replaceAll("\\", "/");
}

function text(raw: Map<string, string>, key: string, fallback: string): string {
  return raw.get(key) ?? fallback;
}

function list(raw: Map<string, string>, key: string, fallback: string[]): string[] {
  return (raw.get(key)?.split(",") ?? fallback).map((value) => value.trim()).filter(Boolean);
}

function configuredStyles(allStyles: boolean, configured: string | null): string[] {
  const styles = allStyles
    ? Object.keys(AI_STYLES)
    : (configured?.split(",").map((style) => style.trim()).filter(Boolean) ?? ["doctrinal"]);
  if (styles.length === 0) throw new Error("Ladder style lists cannot be empty.");
  for (const style of styles) {
    if (!(style in AI_STYLES)) throw new Error(`Unknown ladder AI style: ${style}`);
  }
  return styles;
}

function integer(raw: Map<string, string>, key: string, fallback: number, minimum: number): number {
  const value = Number(raw.get(key) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${key} must be an integer >= ${minimum}`);
  return value;
}

function evenInteger(raw: Map<string, string>, key: string, fallback: number, minimum: number): number {
  const value = integer(raw, key, fallback, minimum);
  if (value % 2 !== 0) throw new Error(`--${key} must be even`);
  return value;
}

function number(raw: Map<string, string>, key: string, fallback: number, minimum: number, maximum = Number.POSITIVE_INFINITY): number {
  const value = Number(raw.get(key) ?? fallback);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`--${key} must be between ${minimum} and ${maximum}`);
  }
  return value;
}

function boolean(raw: Map<string, string>, key: string, fallback: boolean): boolean {
  const value = raw.get(key);
  if (value === undefined) return fallback;
  if (value === "true" || value === "1") return true;
  if (value === "false" || value === "0") return false;
  throw new Error(`--${key} must be true or false`);
}

function oneOf<T extends string>(raw: Map<string, string>, key: string, fallback: T, values: readonly T[]): T {
  const value = text(raw, key, fallback);
  if (!values.includes(value as T)) throw new Error(`--${key} must be one of ${values.join(", ")}`);
  return value as T;
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

function format(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(4);
}

const isMain = process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
