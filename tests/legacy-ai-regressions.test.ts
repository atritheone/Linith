import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const aiSource = readFileSync(new URL("../src/renderer/game/ai.ts", import.meta.url), "utf8");
const gameSource = readFileSync(new URL("../src/renderer/game/game.ts", import.meta.url), "utf8");

test("legacy tactical simulation routes moves and pushes through the corrected successor model", () => {
  assert.match(aiSource, /a\.type === 'push'[\s\S]*simulateCorrectPushSubset/);
  assert.match(aiSource, /simulateSwanMoveOn/);
  assert.match(aiSource, /simulatePushOn/);
});

test("opponent replies are evaluated from the replying player's perspective", () => {
  assert.match(aiSource, /evaluateStyled\(b, b2, null, null, player\)/);
  assert.match(aiSource, /a\.score - oppBest\.score/);
});

test("mutual final encirclement is not promoted as a tactical win", () => {
  assert.match(aiSource, /sealedSun > 0 && res\.sealedMoon > 0\) return 0/);
  assert.match(aiSource, /!\(enemySealed > 0 && selfSealed > 0\)/);
});

test("opponent placement cap counts active and frozen Swans", () => {
  assert.match(aiSource, /countTotalSwansOn\(b, player\)<6/);
});

test("ring pressure writes its computed cache entry", () => {
  assert.match(aiSource, /entry\[player\] = score/);
});

test("unknown difficulty no longer silently uses Easy", () => {
  assert.match(aiSource, /includes\(difficulty\) \? difficulty : 'medium'/);
});

test("push replay metadata records every earned bonus action", () => {
  assert.match(gameSource, /nextMovesLeft \+= opponentLoss/);
  assert.doesNotMatch(gameSource, /if \(opponentLoss > 0\) nextMovesLeft\+\+/);
});

test("live placement and pushed-Stone sharing match the documented rule model", () => {
  assert.match(gameSource, /for \(const \[nr,nc\] of neighbours8\(r,c\)\)[\s\S]*enemySwan\(v,p\)/);
  assert.match(gameSource, /Frozen friendly Swans are stationary and share\/anchor the Stone too/);
});
