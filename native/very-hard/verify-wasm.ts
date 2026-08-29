import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import {
  EMPTY,
  FROZEN_SUN,
  STONE,
  SWAN_MOON,
  SWAN_SUN,
  type Board,
  type Player
} from "../../src/renderer/game/encirclement";
import {
  actionKey,
  applyAction,
  generateLegalActions,
  type GameOutcome,
  type LinithAction,
  type SearchState
} from "../../src/renderer/game/rulesEngine";

interface CoreExports {
  reset_state(player: number, movesLeft: number): void;
  set_cell(cell: number, tile: number): void;
  get_cell(cell: number): number;
  get_current_player(): number;
  get_actions_left(): number;
  get_outcome(): number;
  generate_actions(): number;
  get_action_type(index: number): number;
  get_action_cell(index: number): number;
  get_action_direction(index: number): number;
  get_action_mask_low(index: number): bigint;
  get_action_mask_high(index: number): bigint;
  apply_generated_action(index: number): number;
  undo_position(): number;
  position_hash(): bigint;
  perft(depth: number): bigint;
  memory: WebAssembly.Memory;
}

const DIRECTIONS = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [-1, 1], [1, -1], [1, 1]
] as const;

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, "../../src/renderer/game/veryHard/native/linith-core.wasm");
const bytes = readFileSync(wasmPath);
const compiled = new WebAssembly.Module(bytes);
const instance = new WebAssembly.Instance(compiled, {
  env: {
    abort(_message: number, _file: number, line: number, column: number): never {
      throw new Error(`AssemblyScript abort at ${line}:${column}`);
    },
    now: () => performance.now()
  }
});
const core = instance.exports as unknown as CoreExports;

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function cloneState(state: SearchState): SearchState {
  return { board: state.board.map((row) => [...row]), current: state.current, movesLeft: state.movesLeft };
}

function load(state: SearchState): void {
  core.reset_state(state.current, state.movesLeft);
  const flat = state.board.flat();
  for (let cell = 0; cell < 100; cell += 1) core.set_cell(cell, flat[cell]);
}

function maskCells(lowValue: bigint, highValue: bigint): Array<{ r: number; c: number }> {
  const low = BigInt.asUintN(64, lowValue);
  const high = BigInt.asUintN(64, highValue);
  const cells: Array<{ r: number; c: number }> = [];
  for (let cell = 0; cell < 100; cell += 1) {
    const present = cell < 64
      ? (low & (1n << BigInt(cell))) !== 0n
      : (high & (1n << BigInt(cell - 64))) !== 0n;
    if (present) cells.push({ r: Math.floor(cell / 10), c: cell % 10 });
  }
  return cells;
}

function wasmAction(index: number): LinithAction {
  const type = core.get_action_type(index);
  if (type === 0 || type === 1) {
    const cell = core.get_action_cell(index);
    return { type: type === 0 ? "stone" : "swan", r: Math.floor(cell / 10), c: cell % 10 };
  }
  const swans = maskCells(core.get_action_mask_low(index), core.get_action_mask_high(index));
  const dir = DIRECTIONS[core.get_action_direction(index)];
  return { type: type === 2 ? "move" : "push", swans, dir };
}

function wasmBoard(): Board {
  const next = emptyBoard();
  for (let cell = 0; cell < 100; cell += 1) next[Math.floor(cell / 10)][cell % 10] = core.get_cell(cell) as Board[number][number];
  return next;
}

function wasmOutcome(): GameOutcome {
  const value = core.get_outcome();
  return value === 1 ? "sun" : value === 2 ? "moon" : value === 3 ? "draw" : null;
}

let comparedPositions = 0;
let comparedTransitions = 0;

function verifyPosition(state: SearchState, label: string, exhaustive = true): void {
  load(state);
  const expectedActions = generateLegalActions(state);
  const count = core.generate_actions();
  const actualActions = Array.from({ length: count }, (_, index) => wasmAction(index));
  assert.deepEqual(
    actualActions.map(actionKey),
    expectedActions.map(actionKey),
    `${label}: legal action corpus differs`
  );

  const indexes = exhaustive
    ? expectedActions.map((_, index) => index)
    : [...new Set([0, Math.floor(count / 4), Math.floor(count / 2), Math.floor(3 * count / 4), count - 1])]
        .filter((index) => index >= 0 && index < count);

  for (const index of indexes) {
    const expected = applyAction(state, expectedActions[index]);
    assert.ok(expected, `${label}: TS rejected its generated action ${actionKey(expectedActions[index])}`);
    assert.equal(core.apply_generated_action(index), 1, `${label}: Wasm rejected action ${index}`);
    assert.deepEqual(wasmBoard(), expected.board, `${label}: board mismatch after ${actionKey(expectedActions[index])}`);
    assert.equal(core.get_current_player(), expected.current, `${label}: current-player mismatch`);
    assert.equal(core.get_actions_left(), expected.movesLeft, `${label}: actions-left mismatch`);
    assert.equal(wasmOutcome(), expected.outcome, `${label}: outcome mismatch`);
    assert.equal(core.undo_position(), 1, `${label}: undo failed`);
    assert.deepEqual(wasmBoard(), state.board, `${label}: undo did not restore board`);
    assert.equal(core.get_current_player(), state.current, `${label}: undo did not restore player`);
    assert.equal(core.get_actions_left(), state.movesLeft, `${label}: undo did not restore actions`);
    comparedTransitions += 1;
  }
  comparedPositions += 1;
}

function fixtures(): Array<[string, SearchState]> {
  const initial = emptyBoard();
  initial[4][4] = SWAN_SUN;
  initial[6][6] = SWAN_MOON;

  const development = initial.map((row) => [...row]);
  development[4][5] = STONE;

  const followingStone = emptyBoard();
  followingStone[5][4] = SWAN_SUN;
  followingStone[5][5] = STONE;
  followingStone[4][5] = FROZEN_SUN;
  followingStone[8][8] = SWAN_MOON;

  const push = emptyBoard();
  push[5][4] = SWAN_SUN;
  push[5][5] = SWAN_MOON;
  push[5][6] = STONE;

  const simultaneous = emptyBoard();
  simultaneous[0][0] = SWAN_SUN;
  simultaneous[9][9] = SWAN_MOON;
  for (const [r, c] of [[0, 1], [1, 0], [8, 8], [8, 9], [9, 8]] as const) simultaneous[r][c] = STONE;

  const saturation = Array.from({ length: 10 }, () => Array(10).fill(STONE) as Board[number]);
  saturation[0][0] = SWAN_SUN;
  saturation[0][2] = SWAN_MOON;
  saturation[1][1] = EMPTY;

  return [
    ["initial", { board: initial, current: 2, movesLeft: 1 }],
    ["development", { board: development, current: 1, movesLeft: 1 }],
    ["two-action development", { board: development, current: 1, movesLeft: 2 }],
    ["frozen follower", { board: followingStone, current: 1, movesLeft: 1 }],
    ["push follower", { board: push, current: 1, movesLeft: 1 }],
    ["simultaneous final encirclement", { board: simultaneous, current: 1, movesLeft: 1 }],
    ["saturation", { board: saturation, current: 1, movesLeft: 1 }]
  ];
}

for (const [label, state] of fixtures()) verifyPosition(state, label);

// A deterministic reachable corpus catches combinations that hand-authored
// tactical fixtures rarely cover: multi-Swan groups, pushes, and bonus chains.
let corpusState = cloneState(fixtures()[1][1]);
let seed = 0x7a11ce55;
for (let sample = 0; sample < 48; sample += 1) {
  verifyPosition(corpusState, `reachable-${sample}`, sample < 16);
  const actions = generateLegalActions(corpusState);
  seed = (Math.imul(seed, 1_664_525) + 1_013_904_223) >>> 0;
  const action = actions[seed % actions.length];
  const next = applyAction(corpusState, action);
  assert.ok(next);
  corpusState = next.outcome ? cloneState(fixtures()[1][1]) : {
    board: next.board,
    current: next.current,
    movesLeft: next.movesLeft
  };
}

// Perft stays entirely inside Wasm, which is the relevant integration shape
// for the final engine. It also verifies that recursive apply/undo is stable.
const benchmarkState = fixtures()[1][1];
load(benchmarkState);
const beforeHash = core.position_hash();
const startPerft = performance.now();
const nodes = core.perft(2);
const perftMs = performance.now() - startPerft;
assert.equal(core.position_hash(), beforeHash, "perft did not restore the root position");

load(benchmarkState);
const generationIterations = 2_000;
const startGeneration = performance.now();
let generated = 0;
for (let iteration = 0; iteration < generationIterations; iteration += 1) generated += core.generate_actions();
const generationMs = performance.now() - startGeneration;

console.log(JSON.stringify({
  wasmBytes: bytes.byteLength,
  comparedPositions,
  comparedTransitions,
  perftDepth: 2,
  perftNodes: nodes.toString(),
  perftMs: Number(perftMs.toFixed(3)),
  perftNodesPerSecond: Math.round(Number(nodes) * 1_000 / perftMs),
  generationIterations,
  generatedActions: generated,
  generationsPerSecond: Math.round(generationIterations * 1_000 / generationMs)
}, null, 2));
