import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  instantiateNativeVeryHardCore,
  type NativeVeryHardCore
} from "../../src/renderer/game/veryHard/native/core";

let singleton: NativeVeryHardCore | undefined;

export function createNativeVeryHardCoreSync(): NativeVeryHardCore {
  const here = dirname(fileURLToPath(import.meta.url));
  const wasm = readFileSync(resolve(here, "../../src/renderer/game/veryHard/native/linith-core.wasm"));
  return instantiateNativeVeryHardCore(wasm);
}

/**
 * Synchronous adapter for deterministic arenas and tests. It instantiates the
 * exact `.wasm` artifact loaded by the shipping browser Worker.
 */
export function loadNativeVeryHardCoreSync(): NativeVeryHardCore {
  if (singleton) return singleton;
  singleton = createNativeVeryHardCoreSync();
  return singleton;
}
