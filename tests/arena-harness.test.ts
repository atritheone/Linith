import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const script = resolve("scripts/ai-arena.ts");
const smokeArguments = [
  "--import", "tsx", script,
  "--challenger=easy",
  "--opponent=medium",
  "--games=2",
  "--max-actions=1",
  "--opening-mode=generated",
  "--opening-count=1",
  "--opening-plies=2",
  "--seed=424242",
  "--summary-only=true"
];

test("arena manifests are reproducible and unresolved games receive no score", () => {
  const first = runArena(smokeArguments);
  const second = runArena(smokeArguments);
  assert.equal(first.status, 0, first.stderr);
  assert.equal(second.status, 0, second.stderr);

  const firstReport = JSON.parse(first.stdout);
  const secondReport = JSON.parse(second.stdout);
  assert.equal(firstReport.manifestId, secondReport.manifestId);
  assert.equal(firstReport.summary.games, 2);
  assert.equal(firstReport.summary.actionLimitUnresolved, 2);
  assert.equal(firstReport.summary.resolved, 0);
  assert.equal(firstReport.summary.pessimisticScoreRate, 0);
  assert.equal(firstReport.summary.resolvedScoreRate, null);
  assert.equal(firstReport.summary.invalid, 0);
  assert.equal(firstReport.summary.crashes, 0);
  assert.equal(firstReport.config.veryHardFloor, "never");
});

test("arena confidence gates fail the process without rewriting unresolved games as draws", () => {
  const result = runArena([...smokeArguments, "--min-pessimistic-score=0.5"]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Arena gate failed/);
  const report = JSON.parse(result.stdout);
  assert.equal(report.summary.terminalDraws, 0);
  assert.equal(report.summary.unresolved, 2);
});

test("optional search trace serializes compact per-decision diagnostics", () => {
  const result = runArena([
    "--import", "tsx", script,
    "--challenger=very-hard",
    "--opponent=easy",
    "--games=2",
    "--max-actions=1",
    "--opening-mode=generated",
    "--opening-count=1",
    "--opening-plies=3",
    "--seed=8675309",
    "--very-hard-engine=typescript",
    "--timing=fixed",
    "--budget-ms=10",
    "--node-budget=10000",
    "--depth=2",
    "--floor=never",
    "--search-trace=true"
  ]);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.deepEqual(report.games[0].decisionTrace, []);
  assert.equal(report.games[1].decisionTrace.length, 1);
  assert.deepEqual(
    Object.keys(report.games[1].decisionTrace[0]).sort(),
    [
      "action", "actionNumber", "attemptedDepth", "budgetMs", "completedDepth",
      "elapsedMs", "engineElapsedMs", "nodes", "player", "score", "source", "stopReason"
    ]
  );
  assert.equal(report.games[1].decisionTrace[0].source, "typescript");
  assert.equal(report.games[1].decisionTrace[0].actionNumber, 1);
});

test("style offsets let parallel shards cover the full matchup matrix", () => {
  const result = runArena([
    "--import", "tsx", script,
    "--challenger=easy",
    "--opponent=medium",
    "--games=2",
    "--max-actions=1",
    "--opening-mode=curated",
    "--opening-count=1",
    "--all-styles=true",
    "--style-offset=8"
  ]);
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.games[0].challengerStyle, "constrictor");
  assert.equal(report.games[0].opponentStyle, "constrictor");
  assert.equal(report.config.styleOffset, 8);
});

test("arena reuses only an exact-state native same-turn continuation", () => {
  const directory = mkdtempSync(resolve(tmpdir(), "linith-arena-continuation-"));
  const manifestPath = resolve(directory, "manifest.json");
  const board = Array.from({ length: 10 }, () => Array(10).fill(0));
  board[5][5] = 1;
  board[0][0] = 2;
  board[1][1] = 3;
  const opening = { board, current: 1, movesLeft: 2 };
  const baseGame = {
    pairId: "p00001",
    openingId: "continuation",
    opening,
    challengerSide: 1,
    challengerStyle: "doctrinal",
    opponentStyle: "doctrinal",
    decisionSeed: 12345
  };
  writeFileSync(manifestPath, JSON.stringify({
    format: "linith-ai-arena-manifest",
    version: 1,
    seed: 12345,
    challenger: "very-hard",
    opponent: "hard",
    maxActions: 2,
    repetitionCount: 3,
    games: [
      { ...baseGame, id: "g00001" },
      { ...baseGame, id: "g00002", challengerSide: 2 }
    ]
  }));
  try {
    const result = runArena([
      "--import", "tsx", script,
      `--manifest-in=${manifestPath}`,
      "--games=2",
      "--seed=12345",
      "--max-actions=2",
      "--repetition=3",
      "--very-hard-engine=native",
      "--timing=fixed",
      "--budget-ms=2000",
      "--node-budget=250000",
      "--depth=1",
      "--floor=never",
      "--search-trace=true"
    ]);
    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.deepEqual(
      report.games[0].decisionTrace.map((entry: { source: string }) => entry.source),
      ["native", "continuation"]
    );
    assert.equal(report.games[0].decisionTrace[1].engineElapsedMs, 0);
    assert.deepEqual(report.games[1].decisionTrace, []);
    assert.equal(report.search.searches, 1);
    assert.equal(report.search.continuationHits, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("manifest replay rejects CLI identity drift", () => {
  const directory = mkdtempSync(resolve(tmpdir(), "linith-arena-manifest-identity-"));
  const manifestPath = resolve(directory, "manifest.json");
  writeFileSync(manifestPath, JSON.stringify(validPairedManifest()));
  const base = manifestReplayArguments(manifestPath);
  const cases: Array<[string, string, RegExp]> = [
    ["challenger", "--challenger=hard", /manifest challenger/],
    ["opponent", "--opponent=easy", /manifest opponent/],
    ["seed", "--seed=54321", /manifest seed/],
    ["maxActions", "--max-actions=3", /manifest maxActions/],
    ["repetitionCount", "--repetition=4", /manifest repetitionCount/],
    ["games", "--games=4", /manifest games/]
  ];
  try {
    for (const [label, override, pattern] of cases) {
      const result = runArena([...base, override]);
      assert.notEqual(result.status, 0, `${label} drift was accepted`);
      assert.match(result.stderr, pattern);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("manifest replay rejects duplicate ids and malformed color-pair membership", () => {
  const directory = mkdtempSync(resolve(tmpdir(), "linith-arena-manifest-pairs-"));
  const manifestPath = resolve(directory, "manifest.json");
  const cases: Array<[string, (manifest: ReturnType<typeof validPairedManifest>) => void, RegExp]> = [
    ["duplicate game id", (manifest) => { manifest.games[1].id = manifest.games[0].id; }, /duplicate game id/],
    ["split pair", (manifest) => { manifest.games[1].pairId = "p00002"; }, /exactly two games/],
    ["same color", (manifest) => { manifest.games[1].challengerSide = 1; }, /swap challenger colors/],
    ["opening", (manifest) => { manifest.games[1].opening.board[9][9] = 3; }, /exact same opening/],
    ["styles", (manifest) => { manifest.games[1].challengerStyle = "constrictor"; }, /exact same styles/],
    ["decision seed", (manifest) => { manifest.games[1].decisionSeed += 1; }, /exact same decision seed/]
  ];
  try {
    for (const [label, mutate, pattern] of cases) {
      const manifest = validPairedManifest();
      mutate(manifest);
      writeFileSync(manifestPath, JSON.stringify(manifest));
      const result = runArena(manifestReplayArguments(manifestPath));
      assert.notEqual(result.status, 0, `${label} corruption was accepted`);
      assert.match(result.stderr, pattern);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

function validPairedManifest() {
  const board = Array.from({ length: 10 }, () => Array(10).fill(0));
  board[5][5] = 1;
  board[0][0] = 2;
  const opening = { board, current: 1 as const, movesLeft: 1 };
  const common = {
    pairId: "p00001",
    openingId: "curated-001",
    opening,
    challengerStyle: "doctrinal",
    opponentStyle: "doctrinal",
    decisionSeed: 98765
  };
  return {
    format: "linith-ai-arena-manifest" as const,
    version: 1 as const,
    seed: 12345,
    challenger: "very-hard" as const,
    opponent: "hard" as const,
    maxActions: 2,
    repetitionCount: 3,
    games: [
      { ...common, id: "g00001", challengerSide: 1 as const },
      { ...common, opening: { ...opening, board: board.map((row) => [...row]) }, id: "g00002", challengerSide: 2 as const }
    ]
  };
}

function manifestReplayArguments(manifestPath: string): string[] {
  return [
    "--import", "tsx", script,
    `--manifest-in=${manifestPath}`,
    "--games=2",
    "--challenger=very-hard",
    "--opponent=hard",
    "--seed=12345",
    "--max-actions=2",
    "--repetition=3",
    "--very-hard-engine=typescript",
    "--summary-only=true"
  ];
}

function runArena(arguments_: string[]): { status: number | null; stdout: string; stderr: string } {
  const result = spawnSync(process.execPath, arguments_, { encoding: "utf8" });
  return { status: result.status, stdout: result.stdout.trim(), stderr: result.stderr.trim() };
}
