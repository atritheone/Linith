import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadNativeVeryHardCoreSync } from "../../native/very-hard/node-adapter";
import { createVeryHardSearchEngine } from "../../src/renderer/game/veryHard/search";
import type { LinithAction, SearchState } from "../../src/renderer/game/rulesEngine";

export type OfflineVeryHardEngineName = "native" | "typescript";

export interface OfflineVeryHardSearchOptions {
  nodeBudget: number;
  maxDepth: number;
  tacticalDepth: number;
  exactDepth?: number;
  style: string;
}

export interface OfflineVeryHardSearchResult {
  action: LinithAction | null;
  score: number;
  completedDepth: number;
  attemptedDepth: number;
  nodes: number;
  stopReason: string;
  exactSolved: boolean;
}

export interface OfflineVeryHardEngine {
  readonly name: OfflineVeryHardEngineName;
  /** Hash of the exact WASM artifact or TypeScript rule/search sources. */
  readonly fingerprint: string;
  search(state: SearchState, options: OfflineVeryHardSearchOptions): OfflineVeryHardSearchResult;
  commitAction(state: SearchState, action: LinithAction): boolean;
  clearCache(): void;
}

/**
 * Build one persistent engine for an entire offline run.  Native is the exact
 * checked-in WASM artifact loaded by the shipping worker; TypeScript exists
 * only for parity investigations and explicit fallback comparisons.
 */
export function createOfflineVeryHardEngine(name: OfflineVeryHardEngineName): OfflineVeryHardEngine {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
  if (name === "native") {
    const native = loadNativeVeryHardCoreSync();
    const bytes = readFileSync(resolve(root, "src/renderer/game/veryHard/native/linith-core.wasm"));
    const fingerprint = `sha256-${createHash("sha256").update(bytes).digest("hex")}`;
    return {
      name,
      fingerprint,
      clearCache() { native.clearCache(); },
      commitAction(state, action) { return native.commitAction(state, action); },
      search(state, options) {
        const result = native.search(state, {
          maxTurnDepth: options.maxDepth,
          nodeBudget: options.nodeBudget,
          // Zero disables the WASM wall clock, making output node-deterministic.
          budgetMs: 0,
          style: options.style,
          tacticalDepth: options.tacticalDepth,
          exactDepth: options.exactDepth
        });
        return {
          action: result.action,
          score: result.score,
          completedDepth: result.completedTurnDepth,
          attemptedDepth: result.attemptedTurnDepth,
          nodes: result.nodes,
          stopReason: result.stopReason,
          // This is the native proof bit, not an inference from a mate-like
          // score.  Unfinished searches therefore remain conservatively false.
          exactSolved: result.exactSolved
        };
      }
    };
  }

  const typescript = createVeryHardSearchEngine();
  const sourceHash = createHash("sha256");
  for (const path of [
    "src/renderer/game/encirclement.ts",
    "src/renderer/game/aiStyles.ts",
    "src/renderer/game/rulesEngine.ts",
    "src/renderer/game/veryHard/search.ts",
    "src/renderer/game/veryHard/evaluate.ts",
    "src/renderer/game/veryHard/openingBook.ts",
    "src/renderer/game/veryHard/openingBookData.ts"
  ]) {
    sourceHash.update(path);
    sourceHash.update("\0");
    sourceHash.update(readFileSync(resolve(root, path)));
    sourceHash.update("\0");
  }
  return {
    name,
    fingerprint: `sha256-${sourceHash.digest("hex")}`,
    clearCache() { typescript.clear(); },
    // TypeScript search tracks repetitions inside each candidate line but has
    // no cross-call game-history ABI.  The external generator still rejects
    // actual repeated states with the live board key.
    commitAction() { return true; },
    search(state, options) {
      const result = typescript.search(state, {
        budgetMs: Infinity,
        nodeBudget: options.nodeBudget,
        maxDepth: options.maxDepth,
        tacticalDepth: options.tacticalDepth,
        exactDepth: options.exactDepth,
        style: options.style
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
    }
  };
}

export function parseOfflineVeryHardEngine(value: string | null): OfflineVeryHardEngineName {
  const normalized = value ?? "native";
  if (normalized !== "native" && normalized !== "typescript") {
    throw new Error("--engine must be native or typescript.");
  }
  return normalized;
}
