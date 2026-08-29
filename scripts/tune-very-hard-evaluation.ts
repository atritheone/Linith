import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { explainVeryHardPosition } from "../src/renderer/game/veryHard/evaluate";
import {
  TUNABLE_EVALUATION_FEATURES,
  assessEvaluationTuning,
  serializeEvaluationTuningResult,
  splitEvaluationTuningSamples,
  tuneEvaluationWeights,
  type EvaluationFeatureVector,
  type EvaluationTuningSample
} from "../src/renderer/game/veryHard/evaluationTuner";
import type { TeacherCorpus } from "../src/renderer/game/veryHard/teacherCorpus";
import {
  createOfflineVeryHardEngine,
  parseOfflineVeryHardEngine
} from "./lib/very-hard-offline-engine";

function argument(name: string): string | null {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? null;
}

function numberArgument(name: string, fallback: number): number {
  const raw = argument(name);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) throw new Error(`--${name} must be a finite number.`);
  return parsed;
}

function booleanArgument(name: string, fallback: boolean): boolean {
  const raw = argument(name);
  if (raw === null) return fallback;
  if (raw === "true") return true;
  if (raw === "false") return false;
  throw new Error(`--${name} must be true or false.`);
}

function validateTerminalCorpus(corpus: TeacherCorpus): void {
  if (corpus.config.shardCount !== 1 || corpus.config.shardIndex !== 0) {
    throw new Error("Teacher shards must be merged before tuning.");
  }
  if (corpus.gamesAttempted !== corpus.config.games) {
    throw new Error("Teacher corpus does not cover its declared logical game count.");
  }
  const classifiedGames = corpus.completedGames + Object.values(corpus.discarded)
    .reduce((sum, count) => sum + count, 0);
  if (classifiedGames !== corpus.gamesAttempted) {
    throw new Error("Teacher corpus has inconsistent terminal/discard counts.");
  }
  for (const sample of corpus.samples) {
    if (!sample.groupId) throw new Error(`Teacher sample ${sample.id} has no game grouping key.`);
    if (sample.terminalOutcome !== "sun"
      && sample.terminalOutcome !== "moon"
      && sample.terminalOutcome !== "draw") {
      throw new Error(`Teacher sample ${sample.id} has no executor-confirmed terminal outcome.`);
    }
    const expected = sample.terminalOutcome === "draw"
      ? 0
      : (sample.terminalOutcome === "sun" ? 1 : 2) === sample.perspective ? 1 : -1;
    if (sample.target !== expected) {
      throw new Error(`Teacher sample ${sample.id} target disagrees with its terminal outcome.`);
    }
  }
}

async function main(): Promise<void> {
const inputPath = argument("input");
const outputPath = argument("output");
if (!inputPath || !outputPath) {
  throw new Error(
    "Usage: node --import tsx scripts/tune-very-hard-evaluation.ts "
    + "--input=teacher-corpus.json --output=weights.json "
    + "[--engine=native|typescript] [--algorithm=coordinate|spsa] "
    + "[--iterations=160] [--seed=5351196]"
  );
}

const corpus = JSON.parse(await readFile(resolve(inputPath), "utf8")) as TeacherCorpus;
if (corpus.format !== "linith-evaluation-corpus" || corpus.version !== 1 || !Array.isArray(corpus.samples)) {
  throw new Error("Unsupported evaluation corpus. Expected linith-evaluation-corpus version 1.");
}
validateTerminalCorpus(corpus);
const engineName = parseOfflineVeryHardEngine(argument("engine"));
const engine = createOfflineVeryHardEngine(engineName);
const expectedProvenance = `:${engineName}:${engine.fingerprint}`;
if (corpus.artifactFingerprint !== engine.fingerprint || !corpus.generator.endsWith(expectedProvenance)) {
  throw new Error(
    `Teacher corpus does not match the current ${engineName} engine artifact (${engine.fingerprint}). `
    + "Regenerate it before tuning."
  );
}

const samples: EvaluationTuningSample[] = corpus.samples.map((sample) => {
  const breakdown = explainVeryHardPosition(sample.state, sample.perspective, "doctrinal");
  const features = Object.fromEntries(
    TUNABLE_EVALUATION_FEATURES.map((feature) => [feature, breakdown[feature]])
  ) as unknown as EvaluationFeatureVector;
  return {
    id: sample.id,
    groupId: sample.groupId,
    features,
    target: sample.target,
    weight: sample.weight
  };
});

const requestedAlgorithm = argument("algorithm") ?? "coordinate";
if (requestedAlgorithm !== "coordinate" && requestedAlgorithm !== "spsa") {
  throw new Error("--algorithm must be coordinate or spsa.");
}
const algorithm = requestedAlgorithm;
const split = splitEvaluationTuningSamples(samples, {
  seed: Math.floor(numberArgument("split-seed", 0x4f1d0a7)),
  holdoutFraction: numberArgument("holdout", 0.2)
});
const candidate = tuneEvaluationWeights(split.training, {
  algorithm,
  iterations: Math.floor(numberArgument("iterations", 160)),
  seed: Math.floor(numberArgument("seed", 0x51a71c)),
  scoreScale: numberArgument("score-scale", 20_000),
  teacherProvenance: corpus.generator,
  engineFingerprint: corpus.artifactFingerprint
});
const result = assessEvaluationTuning(
  candidate,
  split,
  numberArgument("minimum-holdout-improvement", 0)
);
await writeFile(resolve(outputPath), serializeEvaluationTuningResult(result), "utf8");
process.stdout.write(
  `${algorithm} tuned ${result.sampleCount} training samples: `
  + `${result.initialLoss.toFixed(8)} -> ${result.finalLoss.toFixed(8)}; `
  + `held-out ${result.validation!.heldOutInitialLoss.toFixed(8)} -> `
  + `${result.validation!.heldOutFinalLoss.toFixed(8)} `
  + `[${result.validation!.accepted ? "ACCEPTED" : "REJECTED"}].\n`
);
if (!result.validation!.accepted && booleanArgument("fail-on-rejection", true)) process.exitCode = 2;
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
