import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { EMPTY } from "../src/renderer/game/encirclement";
import type { SearchState } from "../src/renderer/game/rulesEngine";
import { canonicalizeOpeningPosition } from "../src/renderer/game/veryHard/openingBook";
import {
  OPENING_BOOK_GENERATOR_VERSION,
  DEFAULT_OPENING_BOOK_SEARCH_PLANS,
  generateVerifiedOpeningBook,
  serializeGeneratedOpeningBook,
  serializeShippingOpeningBookData,
  type OpeningBookSearchPlan,
  type OpeningBookSearchRunner
} from "../src/renderer/game/veryHard/openingBookBuilder";
import {
  createOfflineVeryHardEngine,
  parseOfflineVeryHardEngine
} from "./lib/very-hard-offline-engine";

function argument(name: string): string | null {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? null;
}

function positiveInteger(name: string, fallback: number, minimum = 1): number {
  const raw = argument(name);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed < minimum) {
    throw new Error(`--${name} must be an integer >= ${minimum}.`);
  }
  return parsed;
}

function statesFromInput(value: unknown): SearchState[] {
  let candidates: unknown[];
  if (Array.isArray(value)) candidates = value;
  else if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record.format === "linith-ai-arena-manifest" && Array.isArray(record.games)) {
      candidates = record.games.map((game) => (
        game && typeof game === "object" ? (game as Record<string, unknown>).opening : null
      ));
    } else if (record.format === "linith-evaluation-corpus" && Array.isArray(record.samples)) {
      candidates = record.samples.map((sample) => (
        sample && typeof sample === "object" ? (sample as Record<string, unknown>).state : null
      ));
    } else if (Array.isArray(record.positions)) candidates = record.positions;
    else throw new Error("Opening-book input is not a positions array, arena manifest, or teacher corpus.");
  } else {
    throw new Error("Opening-book input must be a JSON object or array.");
  }

  return candidates.map((candidate, index) => {
    if (!candidate || typeof candidate !== "object") {
      throw new Error(`Opening-book position ${index} is not an object.`);
    }
    const state = candidate as Partial<SearchState>;
    if (!Array.isArray(state.board)
      || state.board.length !== 10
      || state.board.some((row) => !Array.isArray(row) || row.length !== 10)
      || (state.current !== 1 && state.current !== 2)
      || !Number.isSafeInteger(state.movesLeft)
      || state.movesLeft! < 1) {
      throw new Error(`Opening-book position ${index} is not a valid 10x10 SearchState.`);
    }
    return state as SearchState;
  });
}

async function main(): Promise<void> {
const inputPath = argument("input");
const outputPath = argument("output");
if (!inputPath || !outputPath) {
  throw new Error(
    "Usage: node --import tsx scripts/generate-very-hard-book.ts "
    + "--input=positions.json --output=verified-book.json "
    + "[--shipping-output=src/renderer/game/veryHard/openingBookData.ts] "
    + "[--engine=native|typescript] [--min-depth=4]"
  );
}

const maximumOccupied = positiveInteger("max-occupied", 100);
const maximumPositions = positiveInteger("max-positions", Number.MAX_SAFE_INTEGER);
const positions = statesFromInput(JSON.parse(await readFile(resolve(inputPath), "utf8")))
  .filter((state) => state.board.flat().filter((tile) => tile !== EMPTY).length <= maximumOccupied)
  .sort((left, right) => {
    const leftKey = canonicalizeOpeningPosition(left).key;
    const rightKey = canonicalizeOpeningPosition(right).key;
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  })
  .filter((state, index, sorted) => index === 0
    || canonicalizeOpeningPosition(state).key !== canonicalizeOpeningPosition(sorted[index - 1]).key)
  .slice(0, maximumPositions);
if (positions.length === 0) {
  throw new Error("No opening-book positions remain after filtering and canonical deduplication.");
}
const engineName = parseOfflineVeryHardEngine(argument("engine"));
const engine = createOfflineVeryHardEngine(engineName);
const plans = DEFAULT_OPENING_BOOK_SEARCH_PLANS.map((plan) => {
  const prefix = plan.name === "primary" ? "primary"
    : plan.name === "corroborator" ? "corroborator"
      : "tactical";
  return {
    ...plan,
    nodeBudget: positiveInteger(`${prefix}-nodes`, plan.nodeBudget, 100_000),
    maxDepth: positiveInteger(`${prefix}-depth`, plan.maxDepth, 4),
    tacticalDepth: positiveInteger(`${prefix}-tactical-depth`, plan.tacticalDepth),
    exactDepth: positiveInteger(`${prefix}-exact-depth`, plan.exactDepth)
  };
});

const search: OpeningBookSearchRunner = (state: SearchState, plan: OpeningBookSearchPlan) => {
  // Agreement must come from independent runs, not a warm transposition hit
  // left by the preceding transformed verification.
  engine.clearCache();
  const result = engine.search(state, {
    nodeBudget: plan.nodeBudget,
    maxDepth: plan.maxDepth,
    tacticalDepth: plan.tacticalDepth,
    exactDepth: plan.exactDepth,
    style: "doctrinal"
  });
  return {
    action: result.action,
    score: result.score,
    completedDepth: result.completedDepth,
    nodes: result.nodes,
    stopped: result.stopReason !== "max-depth"
  };
};

const book = generateVerifiedOpeningBook(positions, search, {
  minCompletedDepth: positiveInteger("min-depth", 4),
  plans,
  generator: `${OPENING_BOOK_GENERATOR_VERSION}:${engineName}:${engine.fingerprint}`,
  artifactFingerprint: engine.fingerprint
});
await writeFile(resolve(outputPath), serializeGeneratedOpeningBook(book), "utf8");
const shippingOutputPath = argument("shipping-output");
if (shippingOutputPath) {
  if (book.entries.length === 0) {
    throw new Error("Refusing to replace the shipping opening book with zero verified entries.");
  }
  await writeFile(resolve(shippingOutputPath), serializeShippingOpeningBookData(book), "utf8");
}
process.stdout.write(
  `${engineName} verified ${book.entries.length}/${book.positionsConsidered} canonical positions; `
  + `${book.rejections.length} rejected; artifact ${engine.fingerprint}.\n`
);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
