import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  TEACHER_CORPUS_GENERATOR_VERSION,
  generateTeacherCorpus,
  mergeTeacherCorpusShards,
  serializeTeacherCorpus,
  type TeacherCorpus,
  type TeacherSearchRunner
} from "../src/renderer/game/veryHard/teacherCorpus";
import {
  createOfflineVeryHardEngine,
  parseOfflineVeryHardEngine
} from "./lib/very-hard-offline-engine";

function argument(name: string): string | null {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? null;
}

function integer(name: string, fallback: number, minimum = 1): number {
  const raw = argument(name);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed < minimum) {
    throw new Error(`--${name} must be an integer >= ${minimum}.`);
  }
  return parsed;
}

function runProcess(command: string, args: string[]): Promise<void> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd: process.cwd(), stdio: "inherit", shell: false });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`Teacher shard exited with ${signal ?? `code ${code ?? "unknown"}`}.`));
    });
  });
}

function forwardedArguments(): string[] {
  const removed = ["output", "workers", "shard-index", "shard-count"];
  return process.argv.slice(2).filter((value) => (
    !removed.some((name) => value.startsWith(`--${name}=`))
  ));
}

async function generateParallel(outputPath: string, workers: number): Promise<void> {
  const temporary = await mkdtemp(join(tmpdir(), "linith-teacher-"));
  try {
    const scriptPath = fileURLToPath(import.meta.url);
    const shardPaths = Array.from(
      { length: workers },
      (_unused, index) => join(temporary, `shard-${String(index).padStart(3, "0")}.json`)
    );
    const common = forwardedArguments();
    const outcomes = await Promise.allSettled(shardPaths.map((shardPath, shardIndex) => runProcess(process.execPath, [
      "--import",
      "tsx",
      scriptPath,
      ...common,
      `--workers=1`,
      `--shard-index=${shardIndex}`,
      `--shard-count=${workers}`,
      `--output=${shardPath}`
    ])));
    const failures = outcomes.filter((outcome): outcome is PromiseRejectedResult => outcome.status === "rejected");
    if (failures.length > 0) {
      throw new AggregateError(failures.map((failure) => failure.reason), "Teacher shard generation failed.");
    }
    const shards = await Promise.all(shardPaths.map(async (path) => (
      JSON.parse(await readFile(path, "utf8")) as TeacherCorpus
    )));
    const corpus = mergeTeacherCorpusShards(shards);
    await writeFile(resolve(outputPath), serializeTeacherCorpus(corpus), "utf8");
    process.stdout.write(
      `Merged ${workers} deterministic shards: ${corpus.samples.length} samples from `
      + `${corpus.completedGames}/${corpus.config.games} terminal games.\n`
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
const outputPath = argument("output");
if (!outputPath) {
  throw new Error(
    "Usage: node --import tsx scripts/generate-very-hard-teacher-corpus.ts "
    + "--output=teacher-corpus.json [--engine=native|typescript] "
    + "[--games=128] [--workers=4] [--node-budget=1000000] [--depth=7]"
  );
}

const workers = integer("workers", 1);
const explicitShard = argument("shard-index") !== null || argument("shard-count") !== null;
if (workers > 1) {
  if (explicitShard) throw new Error("--workers cannot be combined with explicit shard arguments.");
  await generateParallel(outputPath, workers);
} else {
  const engineName = parseOfflineVeryHardEngine(argument("engine"));
  const engine = createOfflineVeryHardEngine(engineName);
  const runner: TeacherSearchRunner = (state, config) => {
    const result = engine.search(state, {
      nodeBudget: config.nodeBudget,
      maxDepth: config.maxDepth,
      tacticalDepth: config.tacticalDepth,
      exactDepth: config.exactDepth,
      style: config.style
    });
    return {
      action: result.action,
      score: result.score,
      completedDepth: result.completedDepth,
      attemptedDepth: result.attemptedDepth,
      nodes: result.nodes,
      stopReason: result.stopReason,
      exactSolved: result.exactSolved
    };
  };

  const corpus = generateTeacherCorpus({
    generator: `${TEACHER_CORPUS_GENERATOR_VERSION}:${engineName}:${engine.fingerprint}`,
    artifactFingerprint: engine.fingerprint,
    seed: integer("seed", 0x7ea4c3, 0),
    games: integer("games", 8),
    shardIndex: integer("shard-index", 0, 0),
    shardCount: integer("shard-count", 1),
    openingPlies: integer("opening-plies", 8, 0),
    maxActions: integer("max-actions", 240),
    sampleStride: integer("sample-stride", 3),
    samplesPerGame: integer("samples-per-game", 24),
    minCompletedDepth: integer("min-depth", 3),
    search: {
      nodeBudget: integer("node-budget", 1_000_000, 100_000),
      maxDepth: integer("depth", 7, 4),
      tacticalDepth: integer("tactical-depth", 3),
      exactDepth: integer("exact-depth", 12),
      style: argument("style") ?? "doctrinal"
    }
  }, runner, {
    // Every game's result is independent of process scheduling/shard count,
    // while accepted actions within that game still feed native repetition
    // history exactly as they do in the shipping worker.
    beforeGame: () => engine.clearCache(),
    afterAction: (state, action) => engine.commitAction(state, action)
  });
  await writeFile(resolve(outputPath), serializeTeacherCorpus(corpus), "utf8");
  process.stdout.write(
    `${engineName} teacher shard ${corpus.config.shardIndex + 1}/${corpus.config.shardCount}: `
    + `${corpus.samples.length} samples from ${corpus.completedGames}/${corpus.gamesAttempted} `
    + `terminal games; discarded ${JSON.stringify(corpus.discarded)}.\n`
  );
}
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
