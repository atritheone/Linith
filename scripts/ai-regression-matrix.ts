import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { aggregateLadderReports } from "./ai-ladder";

interface MatrixConfig {
  manifestDirectories: string[];
  baselineDirectory: string;
  outputDirectory: string;
  output: string;
  workers: number;
  floor: "reply" | "depth1" | "never";
  expectedArtifact: string | null;
  resume: boolean;
}

interface Manifest {
  seed: number;
  challenger: string;
  opponent: string;
  maxActions: number;
  repetitionCount: number;
  games: Array<{ id: string; pairId: string }>;
}

interface ArenaGame {
  id: string;
  pairId: string;
  classification: string;
  challengerSide: 1 | 2;
  actions: number;
}

interface ArenaReport {
  kind: "linith-ai-arena";
  manifestId: string;
  config: { opponent: "hard"; [key: string]: unknown };
  veryHardArtifact: { sha256: string; bytes: number } | null;
  games: ArenaGame[];
  search: Record<string, unknown>;
}

interface MatrixJob {
  id: string;
  manifestPath: string;
  manifestHash: string;
  baselineReportPath: string;
  reportPath: string;
}

async function main(): Promise<void> {
const config = readConfig(process.argv.slice(2));
mkdirSync(resolve(config.outputDirectory), { recursive: true });
mkdirSync(dirname(resolve(config.output)), { recursive: true });
const jobs = discoverJobs(config);
const reports = await runPool(jobs, config.workers, (job) => runJob(job, config));
const artifacts = [...new Set(reports.map(({ report }) => report.veryHardArtifact?.sha256).filter(Boolean))];
if (artifacts.length !== 1) throw new Error(`Regression matrix used ${artifacts.length} native artifacts.`);
if (config.expectedArtifact && artifacts[0] !== config.expectedArtifact.toLowerCase()) {
  throw new Error(`Expected artifact ${config.expectedArtifact}, received ${artifacts[0]}.`);
}

const comparisons = reports.map(({ job, report }) => {
  const baseline = JSON.parse(readFileSync(job.baselineReportPath, "utf8")) as ArenaReport;
  const pairId = (JSON.parse(readFileSync(job.manifestPath, "utf8")) as Manifest).games[0].pairId;
  const baselineGames = baseline.games.filter((game) => game.pairId === pairId);
  return {
    id: job.id,
    manifest: job.manifestPath,
    manifestHash: job.manifestHash,
    baselineReport: job.baselineReportPath,
    report: job.reportPath,
    baseline: summarizeGames(baselineGames),
    current: summarizeGames(report.games),
    transitions: report.games.map((game) => ({
      gameId: game.id,
      challengerSide: game.challengerSide === 1 ? "sun" : "moon",
      baseline: baselineGames.find((candidate) => candidate.id === game.id)?.classification ?? null,
      current: game.classification,
      actions: game.actions
    }))
  };
});

const aggregate = aggregateLadderReports(reports.map(({ job, report }) => ({
  shardId: job.id,
  report: report as never
})));
const output = {
  kind: "linith-ai-regression-matrix",
  formatVersion: 1,
  config,
  artifact: reports[0].report.veryHardArtifact,
  pairs: jobs.length,
  aggregate,
  comparisons
};
writeFileSync(resolve(config.output), `${JSON.stringify(output, null, 2)}\n`, "utf8");
console.log(JSON.stringify({
  output: resolve(config.output),
  artifact: output.artifact,
  pairs: output.pairs,
  aggregate: (aggregate.opponents as unknown[])[0]
}, null, 2));
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});

function discoverJobs(settings: MatrixConfig): MatrixJob[] {
  const byHash = new Map<string, MatrixJob>();
  for (const directory of settings.manifestDirectories) {
    for (const name of readdirSync(resolve(directory)).filter((entry) => entry.endsWith(".manifest.json")).sort()) {
      const manifestPath = resolve(directory, name);
      const contents = readFileSync(manifestPath);
      const manifestHash = createHash("sha256").update(contents).digest("hex");
      if (byHash.has(manifestHash)) continue;
      const match = /^(hard-\d+)-(p\d+)\.manifest\.json$/.exec(name);
      if (!match) throw new Error(`Cannot infer baseline shard/pair from ${name}.`);
      const id = `${match[1]}-${match[2]}`;
      byHash.set(manifestHash, {
        id,
        manifestPath,
        manifestHash,
        baselineReportPath: resolve(settings.baselineDirectory, `${match[1]}.json`),
        reportPath: resolve(settings.outputDirectory, `${id}.json`)
      });
    }
  }
  return [...byHash.values()].sort((left, right) => left.id.localeCompare(right.id));
}

async function runJob(job: MatrixJob, settings: MatrixConfig): Promise<{ job: MatrixJob; report: ArenaReport }> {
  if (!settings.resume || !existsSync(job.reportPath)) {
    const manifest = JSON.parse(readFileSync(job.manifestPath, "utf8")) as Manifest;
    const arena = resolve("scripts/ai-arena.ts");
    const arguments_ = [
      "--import", "tsx", arena,
      `--manifest-in=${job.manifestPath}`,
      `--games=${manifest.games.length}`,
      `--challenger=${manifest.challenger}`,
      `--opponent=${manifest.opponent}`,
      `--seed=${manifest.seed}`,
      `--max-actions=${manifest.maxActions}`,
      `--repetition=${manifest.repetitionCount}`,
      "--very-hard-engine=native",
      "--timing=shipping",
      "--platform=desktop",
      `--floor=${settings.floor}`,
      "--node-budget=2000000",
      "--depth=6",
      "--search-trace=true",
      `--output=${job.reportPath}`
    ];
    await spawnArena(job.id, arguments_);
  }
  const report = JSON.parse(readFileSync(job.reportPath, "utf8")) as ArenaReport;
  if (report.kind !== "linith-ai-arena" || report.games.length !== 2) {
    throw new Error(`${job.id} did not produce a two-game arena report.`);
  }
  if (report.config.veryHardFloor !== settings.floor) {
    throw new Error(`${job.id} was run with floor ${String(report.config.veryHardFloor)}, expected ${settings.floor}.`);
  }
  return { job, report };
}

async function spawnArena(id: string, arguments_: string[]): Promise<void> {
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(process.execPath, arguments_, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => { stderr = `${stderr}${chunk}`.slice(-16_384); });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${id} failed (${signal ?? code}): ${stderr.trim()}`));
    });
  });
}

async function runPool<T, R>(items: T[], concurrency: number, run: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  let completed = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      results[index] = await run(items[index]);
      completed += 1;
      process.stderr.write(`Linith regression matrix: ${completed}/${items.length} pairs complete\n`);
    }
  }));
  return results;
}

function summarizeGames(games: ArenaGame[]): Record<string, number> {
  return {
    games: games.length,
    wins: games.filter((game) => game.classification === "win").length,
    losses: games.filter((game) => game.classification === "loss").length,
    terminalDraws: games.filter((game) => game.classification === "terminal-draw").length,
    repetitionUnresolved: games.filter((game) => game.classification === "repetition-unresolved").length,
    actionLimitUnresolved: games.filter((game) => game.classification === "action-limit-unresolved").length,
    operationalFailures: games.filter((game) => game.classification === "invalid" || game.classification === "crash").length,
    pessimisticPoints: games.reduce((points, game) => (
      points + (game.classification === "win" ? 1 : game.classification === "terminal-draw" ? 0.5 : 0)
    ), 0)
  };
}

function readConfig(arguments_: string[]): MatrixConfig {
  const raw = new Map<string, string>();
  for (const argument of arguments_) {
    const equals = argument.indexOf("=");
    if (!argument.startsWith("--") || equals < 0) throw new Error(`Expected --name=value, received ${argument}`);
    raw.set(argument.slice(2, equals), argument.slice(equals + 1));
  }
  const floor = raw.get("floor") ?? "reply";
  if (floor !== "reply" && floor !== "depth1" && floor !== "never") throw new Error("--floor must be reply, depth1, or never");
  return {
    manifestDirectories: (raw.get("manifest-dirs") ?? "native/very-hard/stage2-failures/loss,native/very-hard/stage2-failures/repetition").split(","),
    baselineDirectory: raw.get("baseline-dir") ?? "native/very-hard/ladder-stage2-32-shards",
    outputDirectory: raw.get("report-dir") ?? `native/very-hard/stage2-regression-${floor}`,
    output: raw.get("output") ?? `native/very-hard/stage2-regression-${floor}.json`,
    workers: integer(raw, "workers", 8, 1),
    floor,
    expectedArtifact: raw.get("artifact")?.toLowerCase() ?? null,
    resume: boolean(raw, "resume", false)
  };
}

function integer(raw: Map<string, string>, key: string, fallback: number, minimum: number): number {
  const value = Number(raw.get(key) ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${key} must be an integer >= ${minimum}`);
  return value;
}

function boolean(raw: Map<string, string>, key: string, fallback: boolean): boolean {
  const value = raw.get(key);
  if (value === undefined) return fallback;
  if (value === "true" || value === "1") return true;
  if (value === "false" || value === "0") return false;
  throw new Error(`--${key} must be true or false`);
}
