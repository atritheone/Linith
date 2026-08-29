import { aiPersonality, type AiPersonalityProfile } from "../aiStyles";
import {
  BOARD_SIZE,
  DIRECTIONS,
  countTotalSwans,
  inBounds,
  isActiveSwan,
  opponentOf,
  type AppliedAction,
  type LinithAction,
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
export const VERY_HARD_STYLE_TIE_BREAK_LIMIT = 900;
/**
 * A personality may only break close root choices. Keeping the per-action
 * adjustment to half the positional style band limits the maximum objective
 * regret between any two personality-preferred actions to 900 points.
 */
export const VERY_HARD_ROOT_PERSONALITY_LIMIT = 450;

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

function profile(style: string): AiPersonalityProfile {
  return aiPersonality(style);
}

function bounded(value: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, value));
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
  const traits = personality.traits;
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
  const mobilityDifference = localMobility(state.board, mine) - localMobility(state.board, theirs);
  const mobility = mobilityDifference * weights.mobility;
  const deployedPhase = Math.min(1, (totalMine + totalTheirs) / 12);
  const phase = Math.round(
    (1 - deployedPhase) * (development + centre) * 0.2
    + deployedPhase * (pressure + urgency) * 0.08
  );
  const tempo = (state.current === perspective ? 1 : -1)
    * Math.max(1, state.movesLeft)
    * weights.tempoAction;

  // Personalities are deliberately a bounded tie-break, never a second
  // evaluator. Every feature is a mine-minus-theirs difference, so the style
  // remains antisymmetric between players. Doctrinal has no traits and is
  // therefore an exact zero-bias baseline.
  const frozenBalance = theirFrozen - myFrozen;
  const libertyBalance = myLiberties - theirLiberties;
  const pressureSignal = bounded(pressure / 100, 80);
  const territorySignal = bounded(territory / 20, 60);
  const urgencySignal = bounded(urgency / 500, 60);
  const fragmentationBalance = theirGroups.length - myGroups.length;
  const developmentBalance = totalMine - totalTheirs;
  const styleRaw = 3 * (
    (traits.freezeUrgency ?? 0) * frozenBalance * 35
    + (traits.selfPreservation ?? 0) * urgencySignal
    + (traits.libertyBalance ?? 0) * libertyBalance * 4
    + (traits.containment ?? 0) * pressureSignal
    + (traits.territory ?? 0) * territorySignal
    + (traits.fragmentation ?? 0) * fragmentationBalance * 25
    + (traits.development ?? 0) * developmentBalance * 20
    + (traits.structure ?? 0) * contactDifference * 3
    + (traits.mobility ?? 0) * mobilityDifference * 1.5
  );
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

/**
 * Score the character expressed by one concrete root transition. This is kept
 * separate from the recursive evaluator so personality cannot compound with
 * depth or alter forced tactical results. Doctrinal has no traits, making this
 * function an exact zero for the canonical AI.
 */
export function evaluateVeryHardRootPersonality(
  before: SearchState,
  action: LinithAction,
  after: AppliedAction,
  perspective: Player,
  style = "doctrinal"
): number {
  if (after.outcome !== null) return 0;
  const traits = profile(style).traits;
  if (Object.keys(traits).length === 0) return 0;

  const beforeStyle = explainVeryHardPosition(before, perspective, style).style;
  const afterStyle = explainVeryHardPosition(after, perspective, style).style;
  const undeployed = Math.max(0, 6 - countTotalSwans(before.board, perspective));
  const placementIntent = action.type === "swan"
    ? (traits.development ?? 0) * undeployed * 40
    : action.type === "stone"
      ? (traits.earlyStone ?? 0) * undeployed * 40
      : 0;
  const concreteFreezeIntent = (traits.freezeUrgency ?? 0) * after.opponentLoss * 160;
  const ownFrozenBefore = frozenCount(before.board, perspective);
  const ownFrozenAfter = frozenCount(after.board, perspective);
  const concreteSacrifice = Math.min(
    after.opponentLoss,
    Math.max(0, ownFrozenAfter - ownFrozenBefore)
  );
  const sacrificeIntent = (traits.sacrificeTolerance ?? 0) * concreteSacrifice * 120;
  const raw = (afterStyle - beforeStyle)
    + placementIntent
    + concreteFreezeIntent
    + sacrificeIntent;
  return Math.round(bounded(raw, VERY_HARD_ROOT_PERSONALITY_LIMIT));
}
