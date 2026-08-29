import test from "node:test";
import assert from "node:assert/strict";
import {
  aggregateLadderReports,
  hoeffdingLower,
  validateResumeReport,
  type LadderGame,
  type NativeArtifactRecord
} from "../scripts/ai-ladder";

function game(
  id: string,
  side: 1 | 2,
  classification: LadderGame["classification"],
  pairId = "p00001"
): LadderGame {
  return {
    id,
    pairId,
    openingId: "generated-001",
    challengerSide: side,
    challengerStyle: "doctrinal",
    opponentStyle: "doctrinal",
    classification,
    winner: classification === "win" ? side : classification === "loss" ? (side === 1 ? 2 : 1) : null,
    actions: 12,
    illegalActions: 0,
    nullActions: 0,
    crashes: 0
  };
}

function report(games: LadderGame[]) {
  return {
    kind: "linith-ai-arena" as const,
    manifestId: "manifest",
    config: { opponent: "hard" as const, veryHardEngine: "typescript" as const },
    veryHardArtifact: null,
    games
  };
}

test("ladder aggregation clusters swapped-color games and keeps shard pair IDs distinct", () => {
  const aggregate = aggregateLadderReports([
    { shardId: "hard-001", report: report([game("a", 1, "win"), game("b", 2, "loss")]) },
    { shardId: "hard-002", report: report([game("c", 1, "win"), game("d", 2, "win")]) }
  ]);
  const hard = (aggregate.opponents as Array<Record<string, unknown>>)[0];
  assert.equal(hard.games, 4);
  assert.equal(hard.pairs, 2);
  assert.equal(hard.pairIntegrityFailures, 0);
  assert.equal(hard.pessimisticScoreRate, 0.75);
  assert.equal(hard.pairMeanScore, 0.75);
  assert.deepEqual(hard.pairScoreCounts, { "0.50": 1, "1.00": 1 });
  assert.equal(aggregate.artifactIntegrityFailures, 0);
});

test("unresolved and operational outcomes receive zero pessimistic points", () => {
  const aggregate = aggregateLadderReports([{ shardId: "hard-001", report: report([
    game("a", 1, "terminal-draw"),
    game("b", 2, "repetition-unresolved")
  ]) }]);
  const hard = (aggregate.opponents as Array<Record<string, unknown>>)[0];
  assert.equal(hard.pessimisticScoreRate, 0.25);
  assert.equal(hard.unresolved, 1);
  assert.equal(hard.unresolvedRate, 0.5);
});

test("confidence lower bound uses color-pairs as the independent sample count", () => {
  const expected = Math.max(0, 0.8 - Math.sqrt(Math.log(20) / (2 * 500)));
  assert.equal(hoeffdingLower(0.8, 500), expected);
  assert.equal(hoeffdingLower(null, 500), null);
  assert.equal(hoeffdingLower(0.8, 0), null);
});

test("resume validation rejects configuration drift and stale native artifacts", () => {
  const artifact: NativeArtifactRecord = {
    kind: "wasm", sha256: "a".repeat(64), bytes: 1234,
    openingBookSha256: "c".repeat(64), openingBookBytes: 4321
  };
  const base = {
    kind: "linith-ai-arena" as const,
    manifestId: "manifest",
    config: {
      opponent: "hard" as const,
      veryHardEngine: "native" as const,
      games: 2,
      veryHardFloor: "never",
      veryHardNodeBudget: 1000
    },
    veryHardArtifact: artifact,
    games: [game("a", 1, "win"), game("b", 2, "loss")]
  };
  const expected = { ...base.config };
  assert.equal(validateResumeReport("hard-001", base, expected, artifact), base);
  assert.throws(
    () => validateResumeReport("hard-001", {
      ...base,
      config: { ...base.config, veryHardFloor: "reply" }
    }, expected, artifact),
    /veryHardFloor/
  );
  assert.throws(
    () => validateResumeReport("hard-001", {
      ...base,
      veryHardArtifact: { ...artifact, sha256: "b".repeat(64) }
    }, expected, artifact),
    /does not match the current WASM artifact/
  );
  assert.throws(
    () => validateResumeReport("hard-001", {
      ...base,
      veryHardArtifact: { ...artifact, openingBookSha256: "d".repeat(64) }
    }, expected, artifact),
    /does not match the current WASM artifact/
  );
});

test("ladder artifact integrity rejects missing, mixed, and unexpected artifacts", () => {
  const artifactA: NativeArtifactRecord = {
    kind: "wasm", sha256: "a".repeat(64), bytes: 1234,
    openingBookSha256: "c".repeat(64), openingBookBytes: 4321
  };
  const artifactB: NativeArtifactRecord = {
    kind: "wasm", sha256: "b".repeat(64), bytes: 1234,
    openingBookSha256: "c".repeat(64), openingBookBytes: 4321
  };
  const native = (artifact: NativeArtifactRecord | null) => ({
    ...report([game("a", 1, "win"), game("b", 2, "win")]),
    config: { opponent: "hard" as const, veryHardEngine: "native" as const },
    veryHardArtifact: artifact
  });
  assert.equal(aggregateLadderReports([{ shardId: "n1", report: native(null) }]).artifactIntegrityFailures, 1);
  assert.equal(aggregateLadderReports([
    { shardId: "n1", report: native(artifactA) },
    { shardId: "n2", report: native(artifactB) }
  ]).artifactIntegrityFailures, 1);
  assert.equal(aggregateLadderReports([
    { shardId: "n1", report: native(artifactA) },
    { shardId: "ts", report: report([game("c", 1, "win"), game("d", 2, "win")]) }
  ]).artifactIntegrityFailures, 1);
  assert.equal(aggregateLadderReports([
    { shardId: "ts1", report: report([game("a", 1, "win"), game("b", 2, "win")]) },
    { shardId: "ts2", report: report([game("c", 1, "win"), game("d", 2, "win")]) }
  ]).artifactIntegrityFailures, 0);
});
