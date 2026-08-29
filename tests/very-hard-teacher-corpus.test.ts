import test from "node:test";
import assert from "node:assert/strict";
import { EMPTY, STONE, SWAN_MOON, SWAN_SUN, type Board } from "../src/renderer/game/encirclement";
import { generateLegalActions, type LinithAction, type SearchState } from "../src/renderer/game/rulesEngine";
import {
  generateTeacherCorpus,
  mergeTeacherCorpusShards,
  serializeTeacherCorpus,
  type TeacherSearchRunner
} from "../src/renderer/game/veryHard/teacherCorpus";

test("teacher corpus labels only executor-confirmed terminal games", () => {
  const winningAction: LinithAction = { type: "stone", r: 1, c: 1 };
  const runner: TeacherSearchRunner = () => ({
    action: winningAction,
    score: 900_000_000,
    completedDepth: 5,
    attemptedDepth: 6,
    nodes: 250_000,
    stopReason: "node-budget",
    exactSolved: true
  });
  const options = {
    seed: 17,
    games: 1,
    openingPlies: 0,
    maxActions: 4,
    minCompletedDepth: 4,
    startingStates: [immediateWinState()]
  };
  const first = generateTeacherCorpus(options, runner);
  const second = generateTeacherCorpus(options, runner);

  assert.equal(first.completedGames, 1);
  assert.equal(first.samples.length, 2, "the terminal trace is labelled from both perspectives");
  assert.deepEqual(first.samples.map((sample) => sample.target), [1, -1]);
  assert.deepEqual(first.samples.map((sample) => sample.terminalOutcome), ["sun", "sun"]);
  assert.deepEqual(first.samples.map((sample) => sample.teacher.score), [900_000_000, -900_000_000]);
  assert.equal(first.samples[0].groupId, first.samples[1].groupId);
  assert.equal(serializeTeacherCorpus(first), serializeTeacherCorpus(second));
});

test("modulo shards merge to the byte-identical single-process corpus", () => {
  const runner: TeacherSearchRunner = () => ({
    action: { type: "stone", r: 1, c: 1 },
    score: 900_000_000,
    completedDepth: 5,
    attemptedDepth: 6,
    nodes: 250_000,
    stopReason: "node-budget",
    exactSolved: true
  });
  const base = {
    seed: 0,
    games: 6,
    openingPlies: 0,
    maxActions: 4,
    minCompletedDepth: 4,
    artifactFingerprint: "sha256-test",
    generator: "test:sha256-test",
    startingStates: [immediateWinState()]
  };
  const unsharded = generateTeacherCorpus(base, runner);
  const shards = [0, 1, 2].map((shardIndex) => generateTeacherCorpus({
    ...base,
    shardIndex,
    shardCount: 3
  }, runner));
  const merged = mergeTeacherCorpusShards([...shards].reverse());

  assert.equal(serializeTeacherCorpus(merged), serializeTeacherCorpus(unsharded));
  assert.equal(merged.gamesAttempted, 6);
  assert.throws(() => mergeTeacherCorpusShards([
    shards[0],
    { ...shards[1], artifactFingerprint: "sha256-wrong" },
    shards[2]
  ]), /incompatible/);
});

test("unresolved action-limit games never create speculative labels", () => {
  const runner: TeacherSearchRunner = (state) => ({
    action: generateLegalActions(state)[0] ?? null,
    score: 10,
    completedDepth: 5,
    attemptedDepth: 6,
    nodes: 200_000,
    stopReason: "node-budget",
    exactSolved: false
  });
  const corpus = generateTeacherCorpus({
    games: 1,
    openingPlies: 0,
    maxActions: 1,
    startingStates: [ordinaryState()]
  }, runner);
  assert.equal(corpus.completedGames, 0);
  assert.equal(corpus.samples.length, 0);
  assert.equal(corpus.discarded.actionLimit, 1);
});

test("positions below the configured teacher depth are discarded", () => {
  const runner: TeacherSearchRunner = () => ({
    action: { type: "stone", r: 1, c: 1 },
    score: 0,
    completedDepth: 2,
    attemptedDepth: 3,
    nodes: 100_000,
    stopReason: "node-budget",
    exactSolved: false
  });
  const corpus = generateTeacherCorpus({
    games: 1,
    openingPlies: 0,
    minCompletedDepth: 3,
    startingStates: [immediateWinState()]
  }, runner);
  assert.equal(corpus.samples.length, 0);
  assert.equal(corpus.discarded.insufficientDepth, 1);
});

function emptyBoard(): Board {
  return Array.from({ length: 10 }, () => Array(10).fill(EMPTY) as Board[number]);
}

function immediateWinState(): SearchState {
  const board = emptyBoard();
  board[5][5] = SWAN_SUN;
  board[0][0] = SWAN_MOON;
  board[0][1] = STONE;
  board[1][0] = STONE;
  return { board, current: 1, movesLeft: 1 };
}

function ordinaryState(): SearchState {
  const board = emptyBoard();
  board[4][4] = SWAN_SUN;
  board[6][6] = SWAN_MOON;
  return { board, current: 2, movesLeft: 1 };
}
