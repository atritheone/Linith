import { createVeryHardSearchEngine } from "./search";
import { lookupOpeningBookAction } from "./openingBook";
import { tryLoadNativeVeryHardCore } from "./native/browser";
import { createVeryHardWorkerRuntime } from "./workerRuntime";
import type {
  VeryHardWorkerInboundMessage,
  VeryHardWorkerOutboundMessage
} from "./types";

interface LinithWorkerScope {
  onmessage: ((event: MessageEvent<VeryHardWorkerInboundMessage>) => void) | null;
  postMessage(message: VeryHardWorkerOutboundMessage): void;
}

const workerScope = globalThis as unknown as LinithWorkerScope;
// The renderer keeps this worker alive within one live match. The runtime
// serializes search/commit messages so native history records only moves
// accepted by the live executor and same-turn PV continuations remain exact.
const runtime = createVeryHardWorkerRuntime({
  engine: createVeryHardSearchEngine(),
  nativeCore: tryLoadNativeVeryHardCore(),
  lookupBook: lookupOpeningBookAction,
  postMessage: (message) => workerScope.postMessage(message)
});

workerScope.onmessage = (event): void => {
  void runtime.handleMessage(event.data);
};
