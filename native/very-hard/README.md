# Linith Very Hard native search core

This directory contains a compact, deterministic AssemblyScript/WebAssembly
port of the live TypeScript rules. It intentionally does not reuse the old
`epsilon/cport` implementation, which encodes a historical rules variant.

Build from the repository root:

```powershell
node native/very-hard/build-wasm.mjs
```

The build pins AssemblyScript 0.28.20 and emits
`src/renderer/game/veryHard/native/linith-core.wasm`. The compiler is fetched
through npm's executable cache, so the application has no runtime dependency on
AssemblyScript and `package.json` is not modified.

The release Wasm is checked in because it is the exact artifact loaded by every
shipping target and identified by qualification reports. AssemblyScript's
optional generated JavaScript/declaration bindings and local `build/` checks are
ignored; Linith uses the maintained TypeScript adapter in
`src/renderer/game/veryHard/native/core.ts`.

The Wasm module owns the complete Very Hard search, rather than crossing the
JavaScript/Wasm boundary once per node. Its ABI supports loading a 100-cell
board, deterministic legal-action generation, exact apply/unapply (including
pushes, following Stones, freezes, bonus actions, simultaneous encirclement,
and saturation), position hashing, recursive `perft`, evaluation, and one-call
completed-turn search.

Search uses iterative PVS, a persistent transposition table, root-history and
path repetition handling, exhaustive depth-one root coverage, selective deeper
widening ordered by the completed depth-one scores, tactical/only-defence
exemptions, quiescence at volatile leaves, and a bounded exact extension for
small late positions. A depth unit is a completed turn: scheduled second actions
and bonus-action chains stay on the same side and do not consume another turn.
The result carries the first action plus a hash-guarded same-player continuation,
so a two-action plan does not require a second search when the expected state is
actually reached.

Search itself does not alter persistent game history. The caller must acknowledge
the action that was really applied through `commit_root_action`; this prevents a
fallback-selected action from creating phantom repetition history. Both the Node
arena adapter and browser Worker use this same artifact. The browser loader
embeds it as a data URL, including standalone builds, and falls back to the
TypeScript engine if native initialization fails.

Run the native search tests plus rule-parity and performance checks after
building:

```powershell
node --import tsx --test tests/very-hard-native.test.ts
node --import tsx native/very-hard/verify-wasm.ts
```

`verify-wasm.ts` compares generated actions and every applied transition with the
live TypeScript rules over a deterministic corpus before reporting perft and
generation throughput. Evaluator parity, tactical defence, continuation,
history-commit, and exact-proof fixtures live in `tests/very-hard-native.test.ts`.
