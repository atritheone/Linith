import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { linithAI } from "../src/renderer/game/ai";
import type { Board } from "../src/renderer/game/encirclement";
import { actionKey, applyActionToBoard } from "../src/renderer/game/rulesEngine";

const htmlSource = readFileSync(new URL("../src/renderer/index.html", import.meta.url), "utf8");
const gameSource = readFileSync(new URL("../src/renderer/game/game.ts", import.meta.url), "utf8");
const workerSource = readFileSync(
  new URL("../src/renderer/game/veryHard/worker.ts", import.meta.url),
  "utf8"
);
const workerRuntimeSource = readFileSync(
  new URL("../src/renderer/game/veryHard/workerRuntime.ts", import.meta.url),
  "utf8"
);

test("the difficulty selector exposes the stable very_hard persistence value", () => {
  assert.match(htmlSource, /<option\s+value=["']very_hard["']>Very Hard<\/option>/);
  assert.match(gameSource, /localStorage\.getItem\(['"]linith_ai_difficulty['"]\)/);
  assert.match(gameSource, /localStorage\.setItem\(['"]linith_ai_difficulty['"],\s*aiDifficulty\)/);
});

test("only Very Hard is routed to the isolated worker selector", () => {
  assert.match(gameSource, /import VeryHardWorker from ["']\.\/veryHard\/worker\?worker&inline["']/);
  assert.match(
    gameSource,
    /if \(difficulty === ['"]very_hard['"]\) \{\s*startVeryHardSearch\(\);\s*return;\s*\}\s*const act = linithAI\(board, current, difficulty\)/,
    "Easy, Medium, and Hard must continue through the frozen legacy selector"
  );
});

test("the Very Hard integration leaves representative legacy decisions identical", () => {
  const board: Board = [
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,2,0],
    [0,0,0,0,1,0,2,0,0,0], [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]
  ];
  const expected = {
    easy: "swan:2,4",
    medium: "move:3,4:-1,-1",
    hard: "move:3,4:-1,-1"
  } as const;
  const originalWindow = globalThis.window;
  const originalRandom = Math.random;
  globalThis.window = { linithGetStyle: () => "doctrinal" } as Window & typeof globalThis;
  Math.random = () => 0.9;

  try {
    for (const difficulty of ["easy", "medium", "hard"] as const) {
      const action = linithAI(board, 1, difficulty);
      assert.ok(action);
      assert.equal(actionKey(action), expected[difficulty]);
      assert.ok(applyActionToBoard(board, 1, action));
    }
  } finally {
    Math.random = originalRandom;
    globalThis.window = originalWindow;
  }
});

test("worker requests include the complete turn state and stale-result identity", () => {
  assert.match(
    gameSource,
    /const state = \{\s*board: board\.map\(\(row\) => row\.slice\(\)\),\s*current,\s*movesLeft\s*\}/
  );
  assert.match(
    gameSource,
    /worker\.postMessage\(\{\s*type: ['"]search['"],\s*requestId,\s*sessionId,\s*fingerprint,\s*state,\s*style,\s*budgetMs\s*\}\)/
  );
  assert.match(
    gameSource,
    /message\.requestId === requestId[\s\S]*message\.sessionId === sessionId[\s\S]*sessionId === veryHardSessionSequence[\s\S]*message\.fingerprint === fingerprint[\s\S]*isCurrentVeryHardTurn\(fingerprint, sessionId\)/
  );
});

test("Very Hard caches and pending work cannot cross lifecycle boundaries", () => {
  assert.match(
    gameSource,
    /function resetVeryHardSession\(\) \{\s*const wasThinking = cancelAiMove\(\);\s*terminateVeryHardWorker\(\);\s*veryHardSessionSequence\+\+;/
  );

  const lifecycleBoundaries = [
    /function reset\(\)\{\s*resetVeryHardSession\(\);/,
    /function startgame\(\)\{\s*resetVeryHardSession\(\);/,
    /function finishgame\(message, logLine\)\{\s*stopMasterLoop\(\);\s*resetVeryHardSession\(\);/,
    /function enterrecital\(payload, fileName\)\{[\s\S]*?validateReplayFrames\(payload\?\.replay \|\| \[\]\);[\s\S]*?resetVeryHardSession\(\);/,
    /if \(history\.length <= 1\) return;[^\n]*\n\s*resetVeryHardSession\(\);/,
    /function reviewApply\(idx\)\{\s*resetVeryHardSession\(\);/,
    /function reviewBack\(fromAuto = false\)\{\s*if \(appMode === ['"]menu['"]\) return;\s*resetVeryHardSession\(\);/,
    /function restartAiAfterConfigurationChange\(\) \{[\s\S]*?const wasThinking = resetVeryHardSession\(\);/
  ];
  for (const boundary of lifecycleBoundaries) assert.match(gameSource, boundary);
});

test("worker responses echo the renderer session identity", () => {
  assert.match(workerRuntimeSource, /type: ["']result["'][\s\S]*?sessionId: request\.sessionId/);
  assert.match(workerRuntimeSource, /type: ["']error["'][\s\S]*?sessionId: request\.sessionId/);
});

test("a rejected stale response can only schedule a clean request", () => {
  const staleBranch = gameSource.match(
    /if \(!identityMatches \|\| !positionMatches\) \{([\s\S]*?)\n\s*\}\n\n\s*if \(message\.type === ['"]error['"]\)/
  )?.[1] ?? "";
  assert.match(staleBranch, /disposeVeryHardRequest\(request, true, true\)/);
  assert.match(staleBranch, /aiturn\(\)/);
  assert.doesNotMatch(staleBranch, /performAiAction|performHardFallback/);
});

test("a timer callback queued before a lifecycle reset is inert", () => {
  assert.match(
    gameSource,
    /const scheduledSessionId = veryHardSessionSequence;[\s\S]*?requestAnimationFrame\(\(\) => \{[\s\S]*?if \(scheduledSessionId !== veryHardSessionSequence\) return;/
  );
});

test("the shipping worker prefers native search and retains its engine between successful actions", () => {
  assert.match(workerSource, /tryLoadNativeVeryHardCore/);
  assert.match(workerSource, /nativeCore: tryLoadNativeVeryHardCore\(\)/);
  assert.match(workerSource, /engine: createVeryHardSearchEngine\(\)/);
  assert.match(workerRuntimeSource, /tryCachedContinuation/);
  assert.match(workerRuntimeSource, /core\.positionHash\(state\) === next\.inputHash/);
  assert.match(gameSource, /let veryHardWorker = null/);
  assert.match(gameSource, /let worker = veryHardWorker;\s*if \(!worker\)/);
  assert.match(gameSource, /disposeVeryHardRequest\(request, true\);\s*render\(\)/);
});

test("a legal shallow search result is never overridden by legacy Hard", () => {
  assert.doesNotMatch(workerRuntimeSource, /floor-request|floor-response|needsFloor|pendingFloor|hard-floor/);
  assert.match(
    workerRuntimeSource,
    /if \(!action \|\| !applyAction\(state, action\)\)[\s\S]*?postFinalResult\(request, state, null[\s\S]*?postFinalResult\(request, state, action, proposedEngine, completedDepth, core, continuation\)/
  );
  assert.doesNotMatch(gameSource, /floor-request|floor-response|Hard floor/);
});

test("Hard is reserved for operational fallback and not computed for a successful search", () => {
  const startSearch = gameSource.match(
    /function startVeryHardSearch\(\) \{([\s\S]*?)\n\s*function cancelAiMove\(\)/
  )?.[1] ?? "";
  const eagerSetup = startSearch.split("aiThinking = true")[0] ?? startSearch;
  assert.equal(
    [...gameSource.matchAll(/linithAI\(board, current, ['"]hard['"]\)/g)].length,
    2,
    "Hard may only appear in the legacy null-action safety and the Very Hard operational fallback helper"
  );
  assert.doesNotMatch(eagerSetup, /linithAI\(board, current, ['"]hard['"]\)/);
  assert.doesNotMatch(eagerSetup, /fallbackAction/);
  assert.doesNotMatch(startSearch, /floor-request|floor-response|hard-floor/);
  assert.match(startSearch, /message\.type !== ['"]result['"] \|\| !isWorkerAction\(resultAction\)[\s\S]*?finishVeryHardFailure/);
  assert.match(startSearch, /worker\.onerror = [\s\S]*?finishVeryHardFailure/);
  assert.match(startSearch, /worker exceeded its response deadline/);
  assert.match(gameSource, /function acknowledgeVeryHardAction[\s\S]*?type: ['"]commit['"]/);
  assert.match(
    startSearch,
    /const actualFallback = performHardFallback\(beforeAction, sessionId\);\s*if \(actualFallback\) acknowledgeVeryHardAction\(request, actualFallback\)/
  );
});

test("worker failure falls back to Hard without relabeling the stored difficulty", () => {
  assert.match(gameSource, /const fallback = linithAI\(board, current, ['"]hard['"]\)/);
  assert.doesNotMatch(
    gameSource,
    /performHardFallback[\s\S]{0,500}localStorage\.setItem\(['"]linith_ai_difficulty['"],\s*['"]hard['"]\)/
  );
});
