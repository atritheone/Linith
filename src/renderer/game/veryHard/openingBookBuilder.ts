import { actionKey, applyAction, type LinithAction, type SearchState } from "../rulesEngine";
import {
  D4_SYMMETRIES,
  OPENING_BOOK_SCHEMA_VERSION,
  canonicalizeOpeningPosition,
  inverseSymmetry,
  isWellFormedOpeningBookEntry,
  transformAction,
  transformState,
  type D4Symmetry,
  type OpeningBookEntry
} from "./openingBook";

export const OPENING_BOOK_GENERATOR_VERSION = "linith-classical-book-1";

export interface OpeningBookSearchPlan {
  name: "primary" | "corroborator" | "tactical";
  symmetry: D4Symmetry;
  nodeBudget: number;
  maxDepth: number;
  tacticalDepth: number;
  exactDepth: number;
}

export interface OpeningBookSearchObservation {
  action: LinithAction | null;
  score: number;
  completedDepth: number;
  nodes: number;
  stopped: boolean;
}

export type OpeningBookSearchRunner = (
  state: SearchState,
  plan: OpeningBookSearchPlan
) => OpeningBookSearchObservation;

export interface OpeningBookGenerationOptions {
  plans?: readonly OpeningBookSearchPlan[];
  minCompletedDepth?: number;
  generator?: string;
  /** Exact artifact fingerprint, recorded separately for machine validation. */
  artifactFingerprint?: string;
}

export type OpeningBookRejectionReason =
  | "invalid-plan"
  | "missing-action"
  | "insufficient-depth"
  | "illegal-action"
  | "search-disagreement"
  | "tactical-verification-failed";

export interface OpeningBookVerificationResult {
  entry: OpeningBookEntry | null;
  reason: OpeningBookRejectionReason | null;
}

export interface GeneratedOpeningBook {
  schema: typeof OPENING_BOOK_SCHEMA_VERSION;
  generator: string;
  artifactFingerprint: string | null;
  config: {
    minCompletedDepth: number;
    plans: readonly OpeningBookSearchPlan[];
  };
  positionsConsidered: number;
  entries: OpeningBookEntry[];
  rejections: Array<{ key: string; reason: OpeningBookRejectionReason }>;
}

export const DEFAULT_OPENING_BOOK_SEARCH_PLANS: readonly OpeningBookSearchPlan[] = Object.freeze([
  Object.freeze({
    name: "primary",
    symmetry: "identity",
    nodeBudget: 1_000_000,
    maxDepth: 6,
    tacticalDepth: 2,
    exactDepth: 12
  }),
  Object.freeze({
    name: "corroborator",
    symmetry: "rotate90",
    nodeBudget: 1_500_000,
    maxDepth: 7,
    tacticalDepth: 1,
    exactDepth: 12
  }),
  Object.freeze({
    name: "tactical",
    symmetry: "mirrorRotate270",
    nodeBudget: 1_500_000,
    maxDepth: 5,
    // Three is the forcing-depth ceiling of the shipping native core.
    tacticalDepth: 3,
    exactDepth: 16
  })
]);

function normalizeStoredAction(action: LinithAction): LinithAction {
  if (action.type === "stone" || action.type === "swan") {
    return { type: action.type, r: action.r, c: action.c };
  }
  return {
    type: action.type,
    swans: [...action.swans]
      .map(({ r, c }) => ({ r, c }))
      .sort((left, right) => left.r - right.r || left.c - right.c),
    dir: [action.dir[0], action.dir[1]]
  };
}

function validPlan(plan: OpeningBookSearchPlan): boolean {
  return D4_SYMMETRIES.includes(plan.symmetry)
    && Number.isSafeInteger(plan.nodeBudget)
    && plan.nodeBudget >= 100_000
    && Number.isSafeInteger(plan.maxDepth)
    && plan.maxDepth >= 4
    && Number.isSafeInteger(plan.tacticalDepth)
    && plan.tacticalDepth > 0
    && Number.isSafeInteger(plan.exactDepth)
    && plan.exactDepth > 0;
}

function confidenceFor(minimumDepth: number, runCount: number): number {
  // This is deliberately capped below one: agreement between deterministic
  // searches is useful evidence, not a proof that the move is game-theoretic.
  return Math.min(0.99, 0.9 + Math.min(5, minimumDepth) * 0.01 + Math.min(4, runCount) * 0.01);
}

export function verifyOpeningBookCandidate(
  state: SearchState,
  search: OpeningBookSearchRunner,
  options: OpeningBookGenerationOptions = {}
): OpeningBookVerificationResult {
  const canonical = canonicalizeOpeningPosition(state);
  const plans = options.plans ?? DEFAULT_OPENING_BOOK_SEARCH_PLANS;
  const minCompletedDepth = Math.max(1, Math.floor(options.minCompletedDepth ?? 4));
  const generator = options.generator ?? OPENING_BOOK_GENERATOR_VERSION;
  if (plans.length < 3
    || !plans.some((plan) => plan.name === "primary")
    || !plans.some((plan) => plan.name === "corroborator")
    || !plans.some((plan) => plan.name === "tactical")
    || new Set(plans.map((plan) => plan.symmetry)).size !== plans.length
    || plans.some((plan) => !validPlan(plan))) {
    return { entry: null, reason: "invalid-plan" };
  }

  const observations: Array<{
    plan: OpeningBookSearchPlan;
    observation: OpeningBookSearchObservation;
    canonicalAction: LinithAction;
  }> = [];

  for (const plan of plans) {
    const runState = transformState(canonical.state, plan.symmetry);
    const observation = search(runState, plan);
    if (!observation.action) return { entry: null, reason: "missing-action" };
    if (!Number.isSafeInteger(observation.completedDepth)
      || observation.completedDepth < minCompletedDepth) {
      return { entry: null, reason: "insufficient-depth" };
    }
    if (!applyAction(runState, observation.action)) {
      return { entry: null, reason: "illegal-action" };
    }
    const canonicalAction = normalizeStoredAction(transformAction(
      observation.action,
      inverseSymmetry(plan.symmetry),
      canonical.state.board.length
    ));
    if (!applyAction(canonical.state, canonicalAction)) {
      return { entry: null, reason: "illegal-action" };
    }
    observations.push({ plan, observation, canonicalAction });
  }

  const agreedKey = actionKey(observations[0].canonicalAction);
  const disagreements = observations.filter(({ canonicalAction }) => actionKey(canonicalAction) !== agreedKey);
  if (disagreements.length > 0) {
    const tacticalDisagreed = disagreements.some(({ plan }) => plan.name === "tactical")
      || actionKey(observations.find(({ plan }) => plan.name === "tactical")!.canonicalAction) !== agreedKey;
    return {
      entry: null,
      reason: tacticalDisagreed ? "tactical-verification-failed" : "search-disagreement"
    };
  }

  const completedDepth = Math.min(...observations.map(({ observation }) => observation.completedDepth));
  const score = Math.min(...observations.map(({ observation }) => observation.score));
  const minNodeBudget = Math.min(...plans.map((plan) => plan.nodeBudget));
  const tacticalDepth = Math.max(
    ...plans.filter((plan) => plan.name === "tactical").map((plan) => plan.tacticalDepth)
  );
  const entry: OpeningBookEntry = {
    schema: OPENING_BOOK_SCHEMA_VERSION,
    key: canonical.key,
    action: observations[0].canonicalAction,
    score,
    confidence: confidenceFor(completedDepth, observations.length),
    verification: {
      agreements: observations.length,
      independentRuns: observations.length,
      minCompletedDepth: completedDepth,
      tacticalDepth,
      minNodeBudget
    },
    generator
  };
  return { entry, reason: null };
}

function compareStates(left: SearchState, right: SearchState): number {
  const leftKey = canonicalizeOpeningPosition(left).key;
  const rightKey = canonicalizeOpeningPosition(right).key;
  return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
}

export function generateVerifiedOpeningBook(
  positions: readonly SearchState[],
  search: OpeningBookSearchRunner,
  options: OpeningBookGenerationOptions = {}
): GeneratedOpeningBook {
  const generator = options.generator ?? OPENING_BOOK_GENERATOR_VERSION;
  const plans = options.plans ?? DEFAULT_OPENING_BOOK_SEARCH_PLANS;
  const minCompletedDepth = Math.max(1, Math.floor(options.minCompletedDepth ?? 4));
  const sorted = [...positions].sort(compareStates);
  const unique = sorted.filter((state, index) => {
    if (index === 0) return true;
    return canonicalizeOpeningPosition(state).key !== canonicalizeOpeningPosition(sorted[index - 1]).key;
  });
  const entries: OpeningBookEntry[] = [];
  const rejections: GeneratedOpeningBook["rejections"] = [];

  for (const state of unique) {
    const canonical = canonicalizeOpeningPosition(state);
    const result = verifyOpeningBookCandidate(state, search, {
      plans,
      minCompletedDepth,
      generator
    });
    if (result.entry) entries.push(result.entry);
    else rejections.push({ key: canonical.key, reason: result.reason ?? "missing-action" });
  }

  entries.sort((left, right) => left.key < right.key ? -1 : left.key > right.key ? 1 : 0);
  rejections.sort((left, right) => left.key < right.key ? -1 : left.key > right.key ? 1 : 0);
  return {
    schema: OPENING_BOOK_SCHEMA_VERSION,
    generator,
    artifactFingerprint: options.artifactFingerprint?.trim() || null,
    config: {
      minCompletedDepth,
      plans: plans.map((plan) => ({ ...plan }))
    },
    positionsConsidered: unique.length,
    entries,
    rejections
  };
}

/** Stable byte-for-byte serialization used by CI and reviewable book updates. */
export function serializeGeneratedOpeningBook(book: GeneratedOpeningBook): string {
  return `${JSON.stringify(book, null, 2)}\n`;
}

/**
 * Emit the reviewed entries in the exact TypeScript module consumed by the
 * shipping worker.  The audit JSON remains the source of verification
 * metadata; this serializer deliberately copies only runtime entries and a
 * provenance banner, keeping the installed payload tiny.
 */
export function serializeShippingOpeningBookData(book: GeneratedOpeningBook): string {
  if (!book.artifactFingerprint?.startsWith("sha256-")) {
    throw new Error("Shipping opening-book data requires an explicit SHA-256 engine fingerprint.");
  }
  if (book.entries.length === 0) {
    throw new Error("Shipping opening-book data requires at least one verified entry.");
  }
  const keys = new Set<string>();
  for (const entry of book.entries) {
    if (!isWellFormedOpeningBookEntry(entry)
      || entry.generator !== book.generator
      || keys.has(entry.key)) {
      throw new Error(`Opening-book entry ${entry.key || "<missing>"} is malformed, duplicated, or stale.`);
    }
    keys.add(entry.key);
  }
  const entries = JSON.stringify(book.entries, null, 2);
  const provenance = JSON.stringify({
    generator: book.generator,
    artifactFingerprint: book.artifactFingerprint,
    positionsConsidered: book.positionsConsidered,
    acceptedEntries: book.entries.length,
    rejectedEntries: book.rejections.length
  });
  return [
    'import type { OpeningBookEntry } from "./openingBook";',
    "",
    `// Generated by scripts/generate-very-hard-book.ts; provenance ${provenance}`,
    `const GENERATED_VERY_HARD_OPENING_BOOK: readonly OpeningBookEntry[] = ${entries};`,
    "",
    "export const VERY_HARD_OPENING_BOOK: readonly OpeningBookEntry[] =",
    "  Object.freeze(GENERATED_VERY_HARD_OPENING_BOOK);",
    ""
  ].join("\n");
}
