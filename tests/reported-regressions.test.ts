import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (path: string): string =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const game = read("../src/renderer/game/game.ts");
const html = read("../src/renderer/index.html");
const sound = read("../src/renderer/sound.ts");
const settings = read("../src/renderer/ui/settings.ts");
const androidTest = read(
  "../platforms/android-wrapper/android/app/src/androidTest/java/com/getcapacitor/myapp/ExampleInstrumentedTest.java"
);

test("replay-controlled text is rendered as text under a script-safe CSP", () => {
  assert.match(
    game,
    /function log\(msg\)\{\s*const row = document\.createElement\(['"]div['"]\);\s*row\.textContent = String\(msg \?\? ['"]{2}\);/
  );
  assert.match(game, /d\.textContent = String\(m \?\? ['"]{2}\);/);
  assert.doesNotMatch(html, /script-src[^;]*['"]unsafe-inline['"]/);
});

test("replay import validates before crossing a lifecycle boundary", () => {
  assert.match(
    game,
    /function enterrecital\(payload, fileName\)\{[\s\S]*?const frames = validateReplayFrames\(payload\?\.replay \|\| \[\]\);[\s\S]*?resetVeryHardSession\(\);/
  );
  assert.match(game, /lastOutcomeShort = typeof payload\?\.outcome\?\.short === ['"]string['"]/);
});

test("review controls preserve the intended save and hint behavior", () => {
  assert.match(game, /simulateRulesPush\([\s\S]*?pushScored\(['"]push['"]/);
  assert.match(
    game,
    /setMuted\(btnSave, !\(appMode === ['"]review['"] && \(gameOver \|\| recite\)\)\);/
  );
  assert.match(game, /if \(appMode !== ['"]review['"] \|\| \(!gameOver && !recite\)\) return;/);
  assert.match(game, /setMuted\(btnHint, aiThinking \|\| !inGame \|\| history\.length === 0/);
});

test("new games reset outcomes and transient surrender UI", () => {
  const resetBlock = game.match(/function reset\(\)\{([\s\S]*?)\n\s*function startgame\(\)\{/u)?.[1] ?? "";
  const startBlock = game.match(/function startgame\(\)\{([\s\S]*?)\n\s*function finishgame/u)?.[1] ?? "";
  for (const block of [resetBlock, startBlock]) {
    assert.match(block, /lastOutcomeShort = null;/);
    assert.match(block, /lastOutcomeDetailed = null;/);
    assert.match(block, /setVisible\(confirmBox, false\);/);
  }
});

test("reported settings, timer, replay-tip, wording, and Android regressions stay fixed", () => {
  assert.match(sound, /const DEFAULT_VOLUME = 0\.4;/);
  assert.match(settings, /localStorage\.setItem\(storageKey, String\(value\)\)/);
  assert.match(settings, /stored === null \? Number\.NaN : Number\(stored\)/);
  assert.match(game, /if \(clockMode === ['"]stopwatch['"]\) timerStart\(\);/);
  assert.match(game, /const maximumSteps = Array\.isArray\(replay\) \? replay\.length : 0;/);
  assert.match(game, /Moon placed their first Swan/);
  assert.match(androidTest, /assertEquals\(['"]com\.linith\.app['"]/);
});
