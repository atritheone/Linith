import wasmDataUrl from "./linith-core.wasm?url&inline";
import {
  instantiateNativeVeryHardCore,
  type NativeVeryHardCore
} from "./core";

let browserCore: Promise<NativeVeryHardCore> | undefined;

/** Cached asynchronous loader used by the persistent browser Worker. */
export function loadNativeVeryHardCore(): Promise<NativeVeryHardCore> {
  browserCore ??= (async () => {
    const response = await fetch(wasmDataUrl);
    if (!response.ok) throw new Error(`Unable to load Very Hard native core (${response.status}).`);
    return instantiateNativeVeryHardCore(await response.arrayBuffer());
  })();
  return browserCore;
}

/** Graceful feature fallback for browsers that cannot compile the module. */
export async function tryLoadNativeVeryHardCore(): Promise<NativeVeryHardCore | null> {
  try {
    return await loadNativeVeryHardCore();
  } catch {
    browserCore = undefined;
    return null;
  }
}
