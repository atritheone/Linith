import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board,
  type Player,
  type Tile
} from "../encirclement";
import {
  actionKey,
  applyAction,
  type ActionCoordinate,
  type Direction,
  type LinithAction,
  type SearchState
} from "../rulesEngine";
import { VERY_HARD_OPENING_BOOK } from "./openingBookData";

/**
 * The eight symmetries of a square.  Their order is part of the canonical
 * position format: when a symmetric position has several identical minimum
 * encodings, the first transform wins.
 */
export const D4_SYMMETRIES = [
  "identity",
  "rotate90",
  "rotate180",
  "rotate270",
  "mirror",
  "mirrorRotate90",
  "mirrorRotate180",
  "mirrorRotate270"
] as const;

export type D4Symmetry = (typeof D4_SYMMETRIES)[number];

export const OPENING_BOOK_SCHEMA_VERSION = 1 as const;
export const DEFAULT_BOOK_MIN_CONFIDENCE = 0.9;

export interface OpeningBookVerification {
  /** Independent full-search runs which selected this exact action. */
  agreements: number;
  /** Total independent full-search runs used by the generator. */
  independentRuns: number;
  /** Lowest fully completed turn depth among those runs. */
  minCompletedDepth: number;
  /** Dedicated forcing depth used by the tactical verifier. */
  tacticalDepth: number;
  /** Smallest node allowance among the agreeing runs. */
  minNodeBudget: number;
}

export interface OpeningBookEntry {
  schema: typeof OPENING_BOOK_SCHEMA_VERSION;
  /** Exact canonical position key, including side to move and movesLeft. */
  key: string;
  /** Action expressed in this entry's canonical geometry. */
  action: LinithAction;
  /** Root score reported by the weakest agreeing verification run. */
  score: number;
  /** Conservative, bounded generator confidence in [0, 1]. */
  confidence: number;
  verification: OpeningBookVerification;
  /** Versioned description of the engine/configuration that made the entry. */
  generator: string;
}

export interface CanonicalOpeningPosition {
  key: string;
  state: SearchState;
  /** Geometry transform from the caller's state to `state`. */
  toCanonical: D4Symmetry;
  /** Geometry transform from `state` back to the caller's state. */
  fromCanonical: D4Symmetry;
  /** Whether Sun/Moon tile identities were exchanged. */
  colorsSwapped: boolean;
}

export interface OpeningBookHit {
  action: LinithAction;
  confidence: number;
  score: number;
  generator: string;
  canonicalKey: string;
}

export interface OpeningBookLookupOptions {
  entries?: readonly OpeningBookEntry[];
  minConfidence?: number;
}

function sizeOf(board: Board): number {
  if (board.length === 0 || board.some((row) => row.length !== board.length)) {
    throw new Error("Opening-book canonicalization requires a non-empty square board.");
  }
  return board.length;
}

export function transformCoordinate(
  coordinate: ActionCoordinate,
  symmetry: D4Symmetry,
  size = 10
): ActionCoordinate {
  const { r, c } = coordinate;
  const last = size - 1;
  switch (symmetry) {
    case "identity": return { r, c };
    case "rotate90": return { r: c, c: last - r };
    case "rotate180": return { r: last - r, c: last - c };
    case "rotate270": return { r: last - c, c: r };
    case "mirror": return { r, c: last - c };
    case "mirrorRotate90": return { r: last - c, c: last - r };
    case "mirrorRotate180": return { r: last - r, c };
    case "mirrorRotate270": return { r: c, c: r };
  }
}

export function transformDirection(direction: Direction, symmetry: D4Symmetry): Direction {
  const [dr, dc] = direction;
  switch (symmetry) {
    case "identity": return [dr, dc];
    case "rotate90": return [dc, -dr];
    case "rotate180": return [-dr, -dc];
    case "rotate270": return [-dc, dr];
    case "mirror": return [dr, -dc];
    case "mirrorRotate90": return [-dc, -dr];
    case "mirrorRotate180": return [-dr, dc];
    case "mirrorRotate270": return [dc, dr];
  }
}

export function inverseSymmetry(symmetry: D4Symmetry): D4Symmetry {
  switch (symmetry) {
    case "rotate90": return "rotate270";
    case "rotate270": return "rotate90";
    default: return symmetry;
  }
}

export function transformAction(
  action: LinithAction,
  symmetry: D4Symmetry,
  size = 10
): LinithAction {
  if (action.type === "stone" || action.type === "swan") {
    return { type: action.type, ...transformCoordinate(action, symmetry, size) };
  }
  return {
    type: action.type,
    swans: action.swans.map((coordinate) => transformCoordinate(coordinate, symmetry, size)),
    dir: transformDirection(action.dir, symmetry)
  };
}

export function transformBoard(board: Board, symmetry: D4Symmetry): Board {
  const size = sizeOf(board);
  const transformed = Array.from(
    { length: size },
    () => Array<Tile>(size).fill(EMPTY)
  );
  for (let r = 0; r < size; r += 1) {
    for (let c = 0; c < size; c += 1) {
      const destination = transformCoordinate({ r, c }, symmetry, size);
      transformed[destination.r][destination.c] = board[r][c];
    }
  }
  return transformed;
}

export function transformState(state: SearchState, symmetry: D4Symmetry): SearchState {
  return {
    board: transformBoard(state.board, symmetry),
    current: state.current,
    movesLeft: state.movesLeft
  };
}

function swapTileColor(tile: Tile): Tile {
  switch (tile) {
    case SWAN_SUN: return SWAN_MOON;
    case SWAN_MOON: return SWAN_SUN;
    case FROZEN_SUN: return FROZEN_MOON;
    case FROZEN_MOON: return FROZEN_SUN;
    case EMPTY: return EMPTY;
    case STONE: return STONE;
  }
}

export function swapStateColors(state: SearchState): SearchState {
  return {
    board: state.board.map((row) => row.map(swapTileColor)),
    current: (state.current === 1 ? 2 : 1) as Player,
    movesLeft: state.movesLeft
  };
}

function encodedBoard(board: Board): string {
  return board.map((row) => row.join("")).join("");
}

/**
 * Normalize the moving side to Sun, then choose the lexicographically smallest
 * D4 representation.  The exact turn fragment (`movesLeft`) remains in the
 * key, so a book move can never be reused in the wrong half of a turn.
 */
export function canonicalizeOpeningPosition(state: SearchState): CanonicalOpeningPosition {
  const colorsSwapped = state.current === 2;
  const colorNormalized = colorsSwapped ? swapStateColors(state) : {
    board: state.board.map((row) => [...row]),
    current: state.current,
    movesLeft: state.movesLeft
  };

  let bestSymmetry: D4Symmetry = "identity";
  let bestBoard = transformBoard(colorNormalized.board, bestSymmetry);
  let bestEncoding = encodedBoard(bestBoard);
  for (const symmetry of D4_SYMMETRIES.slice(1)) {
    const candidate = transformBoard(colorNormalized.board, symmetry);
    const encoding = encodedBoard(candidate);
    if (encoding < bestEncoding) {
      bestEncoding = encoding;
      bestBoard = candidate;
      bestSymmetry = symmetry;
    }
  }

  const canonicalState: SearchState = {
    board: bestBoard,
    current: 1,
    movesLeft: state.movesLeft
  };
  return {
    key: `linith-book-v${OPENING_BOOK_SCHEMA_VERSION}:1:${state.movesLeft}:${bestEncoding}`,
    state: canonicalState,
    toCanonical: bestSymmetry,
    fromCanonical: inverseSymmetry(bestSymmetry),
    colorsSwapped
  };
}

function boundedInteger(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === "number"
    && Number.isSafeInteger(value)
    && value >= minimum
    && value <= maximum;
}

function wellFormedCoordinate(value: unknown): value is ActionCoordinate {
  if (!value || typeof value !== "object") return false;
  const coordinate = value as Partial<ActionCoordinate>;
  return boundedInteger(coordinate.r, 0, 9) && boundedInteger(coordinate.c, 0, 9);
}

function wellFormedAction(value: unknown): value is LinithAction {
  if (!value || typeof value !== "object") return false;
  const action = value as Partial<LinithAction>;
  if (action.type === "stone" || action.type === "swan") return wellFormedCoordinate(action);
  if (action.type !== "move" && action.type !== "push") return false;
  if (!Array.isArray(action.swans)
    || action.swans.length < 1
    || action.swans.length > 6
    || action.swans.some((coordinate) => !wellFormedCoordinate(coordinate))) return false;
  const unique = new Set(action.swans.map(({ r, c }) => `${r},${c}`));
  if (unique.size !== action.swans.length) return false;
  if (!Array.isArray(action.dir) || action.dir.length !== 2) return false;
  const [dr, dc] = action.dir;
  return Number.isInteger(dr)
    && Number.isInteger(dc)
    && dr >= -1
    && dr <= 1
    && dc >= -1
    && dc <= 1
    && (dr !== 0 || dc !== 0);
}

/** Reject corrupt, hand-edited, obsolete, or implausibly confident records. */
export function isWellFormedOpeningBookEntry(entry: OpeningBookEntry): boolean {
  const verification = entry?.verification;
  return entry?.schema === OPENING_BOOK_SCHEMA_VERSION
    && typeof entry.key === "string"
    && entry.key.startsWith(`linith-book-v${OPENING_BOOK_SCHEMA_VERSION}:`)
    && typeof entry.generator === "string"
    && entry.generator.length > 0
    && Number.isFinite(entry.score)
    && Math.abs(entry.score) <= 1_000_000_000
    && Number.isFinite(entry.confidence)
    && entry.confidence >= 0
    && entry.confidence <= 1
    && !!verification
    && boundedInteger(verification.agreements, 2, 16)
    && boundedInteger(verification.independentRuns, 2, 16)
    && verification.agreements <= verification.independentRuns
    && boundedInteger(verification.minCompletedDepth, 1, 32)
    && boundedInteger(verification.tacticalDepth, 1, 16)
    && boundedInteger(verification.minNodeBudget, 1, 100_000_000)
    && wellFormedAction(entry.action);
}

/**
 * Return a book hint only after exact-state, confidence, round-trip and live
 * legality checks.  A malformed shipping record therefore degrades to search,
 * never to an illegal move.
 */
export function lookupOpeningBookAction(
  state: SearchState,
  options: OpeningBookLookupOptions = {}
): OpeningBookHit | null {
  const canonical = canonicalizeOpeningPosition(state);
  const entries = options.entries ?? VERY_HARD_OPENING_BOOK;
  const requestedConfidence = options.minConfidence ?? DEFAULT_BOOK_MIN_CONFIDENCE;
  const minConfidence = Number.isFinite(requestedConfidence)
    ? Math.max(0, Math.min(1, requestedConfidence))
    : DEFAULT_BOOK_MIN_CONFIDENCE;
  const entry = entries.find((candidate) => candidate.key === canonical.key);
  if (!entry || !isWellFormedOpeningBookEntry(entry) || entry.confidence < minConfidence) return null;

  const action = transformAction(entry.action, canonical.fromCanonical, state.board.length);
  const canonicalRoundTrip = transformAction(action, canonical.toCanonical, state.board.length);
  if (actionKey(canonicalRoundTrip) !== actionKey(entry.action)) return null;
  if (!applyAction(state, action)) return null;
  return {
    action,
    confidence: entry.confidence,
    score: entry.score,
    generator: entry.generator,
    canonicalKey: canonical.key
  };
}
