import {
  EMPTY,
  SWAN_MOON,
  SWAN_SUN,
  type Board,
  type Player
} from "../encirclement";
import {
  DIRECTIONS,
  countTotalSwans,
  inBounds,
  isActiveSwan,
  type SearchState
} from "../rulesEngine";

export type VeryHardPlatform = "desktop" | "browser" | "android";
export type VeryHardComplexity = "opening" | "quiet" | "complex" | "critical";

export interface VeryHardTimeBudget {
  budgetMs: number;
  hardLimitMs: number;
  complexity: VeryHardComplexity;
  reasons: string[];
}

interface PositionSignals {
  occupied: number;
  activeSwans: number;
  minimumLiberties: number;
  enemyContacts: number;
  bothAtSix: boolean;
}

const PLATFORM_BUDGETS: Record<
  VeryHardPlatform,
  Record<VeryHardComplexity, readonly [number, number]>
> = {
  // Each pair is [normal search allowance, watchdog hard limit]. The search
  // can still finish earlier after a forced move or a stable iteration.
  desktop: {
    opening: [160, 300],
    quiet: [300, 500],
    complex: [475, 650],
    critical: [700, 750]
  },
  browser: {
    opening: [160, 300],
    quiet: [280, 500],
    complex: [450, 650],
    critical: [675, 750]
  },
  android: {
    opening: [220, 400],
    quiet: [380, 650],
    complex: [650, 850],
    critical: [900, 1_000]
  }
};

/**
 * Chooses a small, deterministic time allowance from cheap board signals.
 * This runs on the renderer thread, so it intentionally does not enumerate
 * the (potentially hundreds of) legal actions.
 */
export function chooseVeryHardTimeBudget(
  state: SearchState,
  platform: VeryHardPlatform
): VeryHardTimeBudget {
  const signals = inspectPosition(state.board);
  const reasons: string[] = [];
  let complexity: VeryHardComplexity;

  if (signals.occupied <= 4 && signals.activeSwans <= 4 && state.movesLeft === 1) {
    complexity = "opening";
    reasons.push("sparse opening");
  } else if (signals.minimumLiberties <= 2 || state.movesLeft >= 3) {
    complexity = "critical";
    if (signals.minimumLiberties <= 2) reasons.push("group near encirclement");
    if (state.movesLeft >= 3) reasons.push("freeze-bonus chain");
  } else if (
    signals.bothAtSix ||
    state.movesLeft > 1 ||
    signals.activeSwans >= 8 ||
    signals.enemyContacts >= 3
  ) {
    complexity = "complex";
    if (signals.bothAtSix) reasons.push("two-action phase");
    if (state.movesLeft > 1) reasons.push("same-turn continuation");
    if (signals.activeSwans >= 8) reasons.push("many active Swans");
    if (signals.enemyContacts >= 3) reasons.push("contact position");
  } else {
    complexity = "quiet";
    reasons.push("ordinary position");
  }

  const [budgetMs, hardLimitMs] = PLATFORM_BUDGETS[platform][complexity];
  return { budgetMs, hardLimitMs, complexity, reasons };
}

export function detectVeryHardPlatform(
  userAgent: string,
  desktopBridgePresent: boolean
): VeryHardPlatform {
  if (/Android/i.test(userAgent)) return "android";
  return desktopBridgePresent ? "desktop" : "browser";
}

function inspectPosition(board: Board): PositionSignals {
  let occupied = 0;
  let activeSwans = 0;
  let minimumLiberties = Number.POSITIVE_INFINITY;
  let enemyContacts = 0;
  const seen = new Set<number>();

  for (let r = 0; r < board.length; r += 1) {
    for (let c = 0; c < board[r].length; c += 1) {
      const tile = board[r][c];
      if (tile !== EMPTY) occupied += 1;
      if (tile !== SWAN_SUN && tile !== SWAN_MOON) continue;
      activeSwans += 1;

      const player = tile as Player;
      for (const [dr, dc] of DIRECTIONS) {
        const nr = r + dr;
        const nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const neighbour = board[nr][nc];
        if ((player === 1 && neighbour === SWAN_MOON) || (player === 2 && neighbour === SWAN_SUN)) {
          // Count each opposing pair only once.
          if (nr > r || (nr === r && nc > c)) enemyContacts += 1;
        }
      }

      const start = r * 10 + c;
      if (seen.has(start)) continue;
      const queue = [start];
      const liberties = new Set<number>();
      seen.add(start);
      while (queue.length > 0) {
        const index = queue.pop()!;
        const row = Math.floor(index / 10);
        const column = index % 10;
        for (const [dr, dc] of DIRECTIONS) {
          const nr = row + dr;
          const nc = column + dc;
          if (!inBounds(nr, nc)) continue;
          if (board[nr][nc] === EMPTY) {
            liberties.add(nr * 10 + nc);
          } else if (isActiveSwan(board[nr][nc], player)) {
            const neighbourIndex = nr * 10 + nc;
            if (!seen.has(neighbourIndex)) {
              seen.add(neighbourIndex);
              queue.push(neighbourIndex);
            }
          }
        }
      }
      minimumLiberties = Math.min(minimumLiberties, liberties.size);
    }
  }

  return {
    occupied,
    activeSwans,
    minimumLiberties,
    enemyContacts,
    bothAtSix: countTotalSwans(board, 1) >= 6 && countTotalSwans(board, 2) >= 6
  };
}
