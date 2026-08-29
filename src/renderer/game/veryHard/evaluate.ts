import { AI_STYLES, type AiStyleWeights } from "../aiStyles";
import {
  BOARD_SIZE,
  DIRECTIONS,
  countTotalSwans,
  inBounds,
  isActiveSwan,
  opponentOf,
  type SearchState
} from "../rulesEngine";
import {
  EMPTY,
  FROZEN_MOON,
  FROZEN_SUN,
  STONE,
  type Board,
  type Player
} from "../encirclement";

export const VERY_HARD_MATE_SCORE = 1_000_000_000;
export const VERY_HARD_STYLE_TIE_BREAK_LIMIT = 300;

/**
 * Explicit, integer weights make the hand-tuned evaluator reproducible and
 * give offline arena tuning one stable surface to optimise.
 */
export const VERY_HARD_EVALUATION_WEIGHTS = Object.freeze({
  enemyFrozen: 18_000,
  ownFrozen: 19_500,
  activeSwan: 4_800,
  liberty: 92,
  tightness: 42,
  stoneContact: 210,
  territory: 18,
  development: 260,
  centre: 6,
  mobility: 115,
  tempoAction: 70
});

interface Coordinate {
  r: number;
  c: number;
}

interface GroupShape {
  size: number;
  liberties: number;
}

export interface VeryHardEvaluationBreakdown {
  frozen: number;
  activity: number;
  liberties: number;
  pressure: number;
  territory: number;
  development: number;
  centre: number;
  mobility: number;
  urgency: number;
  phase: number;
  tempo: number;
  style: number;
  total: number;
}

function activeCoordinates(board: Board, player: Player): Coordinate[] {
  const result: Coordinate[] = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (isActiveSwan(board[r][c], player)) result.push({ r, c });
    }
  }
  return result;
}

function key(r: number, c: number): number {
  return r * BOARD_SIZE + c;
}

function liberties(board: Board, swans: readonly Coordinate[]): number {
  const empty = new Set<number>();
  for (const { r, c } of swans) {
    for (const [dr, dc] of DIRECTIONS) {
      const nr = r + dr;
      const nc = c + dc;
      if (inBounds(nr, nc) && board[nr][nc] === EMPTY) empty.add(key(nr, nc));
    }
  }
  return empty.size;
}

function activeGroups(board: Board, player: Player): GroupShape[] {
  const seen = new Set<number>();
  const result: GroupShape[] = [];

  for (const start of activeCoordinates(board, player)) {
    const startKey = key(start.r, start.c);
    if (seen.has(startKey)) continue;

    const queue = [start];
    const rim = new Set<number>();
    let size = 0;
    seen.add(startKey);

    while (queue.length > 0) {
      const current = queue.pop()!;
      size += 1;
      for (const [dr, dc] of DIRECTIONS) {
        const nr = current.r + dr;
        const nc = current.c + dc;
        if (!inBounds(nr, nc)) continue;
        if (board[nr][nc] === EMPTY) {
          rim.add(key(nr, nc));
          continue;
        }
        if (!isActiveSwan(board[nr][nc], player)) continue;
        const neighbourKey = key(nr, nc);
        if (!seen.has(neighbourKey)) {
          seen.add(neighbourKey);
          queue.push({ r: nr, c: nc });
        }
      }
    }
    result.push({ size, liberties: rim.size });
  }
  return result;
}

function frozenCount(board: Board, player: Player): number {
  const target = player === 1 ? FROZEN_SUN : FROZEN_MOON;
  let count = 0;
  for (const row of board) for (const tile of row) if (tile === target) count += 1;
  return count;
}

function tightness(groups: readonly GroupShape[]): number {
  let result = 0;
  for (const group of groups) {
    // A nearly closed group is urgent. Squaring makes the final liberties much
    // more important than broad, harmless reductions elsewhere on the board.
    const missing = Math.max(0, 8 + group.size * 2 - group.liberties);
    result += missing * missing + (group.liberties <= 2 ? 80 : 0);
  }
  return result;
}

function groupUrgency(groups: readonly GroupShape[]): number {
  let score = 0;
  for (const group of groups) {
    if (group.liberties <= 1) score += 14_000 + group.size * 2_500;
    else if (group.liberties === 2) score += 5_000 + group.size * 900;
    else if (group.liberties === 3) score += 1_500 + group.size * 250;
  }
  return score;
}

function territoryBalance(
  board: Board,
  mine: readonly Coordinate[],
  theirs: readonly Coordinate[]
): number {
  if (mine.length === 0 || theirs.length === 0) return 0;
  let score = 0;
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (board[r][c] !== EMPTY) continue;
      let myDistance = BOARD_SIZE;
      let theirDistance = BOARD_SIZE;
      for (const swan of mine) {
        myDistance = Math.min(myDistance, Math.max(Math.abs(swan.r - r), Math.abs(swan.c - c)));
      }
      for (const swan of theirs) {
        theirDistance = Math.min(theirDistance, Math.max(Math.abs(swan.r - r), Math.abs(swan.c - c)));
      }
      if (myDistance < theirDistance) score += Math.min(4, theirDistance - myDistance);
      else if (theirDistance < myDistance) score -= Math.min(4, myDistance - theirDistance);
    }
  }
  return score;
}

function centreControl(swans: readonly Coordinate[]): number {
  let score = 0;
  for (const { r, c } of swans) {
    // Doubled coordinates avoid fractional arithmetic around the 4.5/4.5 centre.
    score += 18 - Math.abs(9 - r * 2) - Math.abs(9 - c * 2);
  }
  return score;
}

function stoneContact(board: Board, swans: readonly Coordinate[]): number {
  let score = 0;
  for (const { r, c } of swans) {
    for (const [dr, dc] of DIRECTIONS) {
      const nr = r + dr;
      const nc = c + dc;
      if (inBounds(nr, nc) && board[nr][nc] === STONE) score += 1;
    }
  }
  return score;
}

function localMobility(board: Board, swans: readonly Coordinate[]): number {
  let score = 0;
  for (const { r, c } of swans) {
    for (const [dr, dc] of DIRECTIONS) {
      const nr = r + dr;
      const nc = c + dc;
      if (!inBounds(nr, nc)) continue;
      if (board[nr][nc] === EMPTY) score += 2;
      else if (board[nr][nc] === STONE) score += 1;
    }
  }
  return score;
}

function profile(style: string): AiStyleWeights {
  return AI_STYLES[style] ?? AI_STYLES.doctrinal;
}

export function explainVeryHardPosition(
  state: SearchState,
  perspective: Player,
  style = "doctrinal"
): VeryHardEvaluationBreakdown {
  const opponent = opponentOf(perspective);
  const mine = activeCoordinates(state.board, perspective);
  const theirs = activeCoordinates(state.board, opponent);
  const myGroups = activeGroups(state.board, perspective);
  const theirGroups = activeGroups(state.board, opponent);
  const personality = profile(style);
  const weights = VERY_HARD_EVALUATION_WEIGHTS;

  const myFrozen = frozenCount(state.board, perspective);
  const theirFrozen = frozenCount(state.board, opponent);
  const frozen = theirFrozen * weights.enemyFrozen - myFrozen * weights.ownFrozen;

  const activity = (mine.length - theirs.length) * weights.activeSwan;
  const myLiberties = liberties(state.board, mine);
  const theirLiberties = liberties(state.board, theirs);
  const libertyScore = (myLiberties - theirLiberties) * weights.liberty;

  const contactDifference = stoneContact(state.board, theirs) - stoneContact(state.board, mine);
  const pressure = (tightness(theirGroups) - tightness(myGroups)) * weights.tightness
    + contactDifference * weights.stoneContact;
  const urgency = groupUrgency(theirGroups) - groupUrgency(myGroups);

  const territory = territoryBalance(state.board, mine, theirs) * weights.territory;
  const totalMine = countTotalSwans(state.board, perspective);
  const totalTheirs = countTotalSwans(state.board, opponent);
  const development = (totalMine - totalTheirs) * weights.development;
  const centre = (centreControl(mine) - centreControl(theirs)) * weights.centre;
  const mobility = (localMobility(state.board, mine) - localMobility(state.board, theirs)) * weights.mobility;
  const deployedPhase = Math.min(1, (totalMine + totalTheirs) / 12);
  const phase = Math.round(
    (1 - deployedPhase) * (development + centre) * 0.2
    + deployedPhase * (pressure + urgency) * 0.08
  );
  const tempo = (state.current === perspective ? 1 : -1)
    * Math.max(1, state.movesLeft)
    * weights.tempoAction;

  // Personalities are deliberately a tie-break, never a second evaluator.
  // Every term below is antisymmetric between players and the final delta is
  // tightly bounded, so a style cannot sacrifice tactical strength.
  const styleRaw =
    ((personality.wFreeze ?? 500) - 500) * (theirFrozen - myFrozen) * 0.08
    + ((personality.wMyLib ?? 5) - 5) * (myLiberties - theirLiberties) * 3
    + ((personality.wOpLib ?? -9) + 9) * (myLiberties - theirLiberties) * 2
    + (personality.wRing ?? 0) * Math.sign(pressure) * Math.min(80, Math.abs(pressure) / 100)
    + ((personality.wSpace ?? 0) - 1) * Math.sign(territory) * Math.min(60, Math.abs(territory) / 20);
  const styleTieBreak = Math.round(Math.max(
    -VERY_HARD_STYLE_TIE_BREAK_LIMIT,
    Math.min(VERY_HARD_STYLE_TIE_BREAK_LIMIT, styleRaw)
  ));
  const total = frozen + activity + libertyScore + pressure + territory + development
    + centre + mobility + urgency + phase + tempo + styleTieBreak;

  return {
    frozen,
    activity,
    liberties: libertyScore,
    pressure,
    territory,
    development,
    centre,
    mobility,
    urgency,
    phase,
    tempo,
    style: styleTieBreak,
    // Heuristic values must never overlap the reserved terminal-score band.
    total: Math.max(-VERY_HARD_MATE_SCORE / 4, Math.min(VERY_HARD_MATE_SCORE / 4, total))
  };
}

export function evaluateVeryHardPosition(
  state: SearchState,
  perspective: Player,
  style = "doctrinal"
): number {
  return explainVeryHardPosition(state, perspective, style).total;
}
