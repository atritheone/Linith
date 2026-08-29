<p align="center">
  <img src="build/icon.png" alt="Linith logo" width="300" />
</p>

<h1 align="center">Linith</h1>

<p align="center">
  A two-player abstract strategy game of movement, pressure, and encirclement.
</p>

<p align="center">
  <a href="https://atritheone.com/linith">Website</a>
  ·
  <a href="#how-to-play">How to play</a>
  ·
  <a href="#development">Development</a>
  ·
  <a href="#builds">Builds</a>
  ·
  <a href="#epsilon">Epsilon</a>
  ·
  <a href="LICENSE.md">MIT license</a>
</p>

## About the game

Played on a 10×10 board, **Sun** and **Moon** place Swans and Stones, move
formations, push opposing Swans, and gradually restrict the opponent's
available space.

Swans are defeated by being completely encircled and frozen.

This repository contains Linith 0.232, its browser and native builds, and
**Epsilon**: the Linith-specific AlphaZero-style self-play and neural-network
training system.

## How to play

Sun places the first Swan. Moon places the second on a non-adjacent square,
then takes the first turn.

On a turn, players can:

- Place a Swan.
- Place a Stone.
- Move one or more Swans one square.
- Push adjacent enemy Swans.

Players may have up to **six Swans**. Once both sides have six, turns begin
with **two actions**.

Stones adjacent to moving Swans may move with them, allowing the board itself
to be reshaped.

### Encirclement

A Swan group freezes when it has no adjacent empty squares. Frozen Swans remain
on the board but cannot move or spawn.

Freezing an enemy Swan grants an additional action. A player loses when their
final active Swan is encircled. If both players' final active Swans are
encircled during the same action, the game is a draw.

The complete rules are in [`rules.txt`](rules.txt).

## Game modes

- Local two-player
- AI plays Sun or Moon
- Easy, Medium, Hard, and Very Hard difficulties
- Doctrinal, Constrictor, Rupture, Blizzard, Librarian, Swarm, and Fortress AI styles

AI style notes are available in [`aipersons.txt`](aipersons.txt).

Very Hard is a separate, bounded iterative-deepening search built on the
corrected pure rule model kept in parity with live play. Its classical
alpha-beta/PVS engine uses completed-turn depth, tactical extensions,
transposition and played-position history, exact small-endgame extensions, and
a compact independently verified opening book. It understands scheduled
two-action turns and freeze-bonus continuations, runs as a 20 KB WebAssembly
kernel outside the renderer, and falls back to Hard only after a genuine
worker/null/illegal-result failure. A legal shallow result is never replaced by
Hard. Easy, Medium, and Hard retain their frozen corrected compatibility
behavior: the captured unaffected decision corpus remains exact, while a
global legal-Stone fallback prevents the inherited selector from returning no
action when its local candidate filters are exhausted.

Very Hard has no neural-network or Epsilon runtime dependency. The historical
Epsilon project remains available for research, but is not used to train or
ship this difficulty. The full standalone build remains about 3 MB, far below
the 4 GB product limit.

## Features

- 10×10 board
- Multi-Swan movement
- Enemy-Swan pushing
- Dynamic Stone movement
- Encirclement and freezing
- Hint system
- Undo and move-history navigation
- Surrender
- Stopwatch and chess clocks
- Save/load
- Game review and replay
- Sound effects
- Configurable board appearance
- Standalone browser, Windows, macOS, Linux, and Android targets
- AlphaZero-style training and evaluation through Epsilon

## Architecture

Linith has one authored TypeScript renderer shared by every target:

```text
src/renderer                     authored game and interface
        │
        ├── Vite ─────────────── dist/renderer
        │                            ├── Electron (Windows/macOS/Linux)
        │                            └── Capacitor copy + Android shell
        │
        └── standalone builder ─ dist/linith-standalone.html
```

Generated HTML is never edited per operating system. Desktop packages load the Vite renderer directly. The Android wrapper copies that renderer and adds its portrait CSS/JavaScript from `platforms/android-wrapper/android-assets`. The standalone builder inlines the compiled CSS, JavaScript, favicon, and sounds into one portable HTML file.

The renderer is divided into game orchestration, AI selection, encirclement rules, sound, layout, and settings modules. New infrastructure and pure rules are strictly typed, encirclement behavior has automated tests, and TypeScript validation runs before every production build. The large inherited game and AI engines remain explicit incremental-migration boundaries instead of leaking into the build and platform tooling.

## Development

Install Node.js, then from the repository root run:

```powershell
npm ci
npm test
npm run dev
```

`npm run dev` opens the Electron development build with Vite hot reload.

Create a checked production renderer with:

```powershell
npm run build
```

Build and verify the exact native artifact used by the browser worker with:

```powershell
npm run build:very-hard-native
npm run verify:very-hard-native
```

Run a side-swapped arena against Hard with:

```powershell
npm run arena:very-hard
```

The harness accepts `--node-budget`, `--budget-ms`, `--depth`, `--games`,
`--max-actions`, and style overrides. It reports legality, artifact identity,
search throughput and depth, stop reasons, unresolved causes, and arena
outcomes as JSON. Games are paired on the exact same opening and style matchup
with Very Hard playing each colour. Unresolved and operational outcomes score
zero in the pessimistic release metric. For a release-strength gate, use the
ladder runner with a suitably large even sample:

```powershell
npm run ladder:very-hard -- --opponents=hard --games=1000 --workers=4
```

Resume validation rejects configuration drift, malformed color pairs, mixed
WASM hashes, and stale opening-book hashes. The deterministic offline tools are
available as `npm run ai:teacher`, `npm run ai:tune`, and `npm run ai:book`;
tuned weights or book entries must pass held-out/agreement checks before they
are copied into shipping code.

The frozen release bundle (`602974…628c9` WASM, `7ea553…74dcc` book) scored
89.8% against Hard over 1,000 games / 500 color-swapped pairs under a fixed
10,000-node qualification: 896 wins, 99 losses, 4 terminal draws, and 1
repetition-unresolved game. The distribution-free one-sided 95% lower bound
was 84.33%; both colours and all seven Very Hard personalities were positive,
with zero illegal actions, crashes, artifact mismatches, or Hard fallbacks.
This bulk fixed-work run is a strength test, not a latency claim. A separate
32-game shipping-clock screen averaged 550 ms per fresh search and scored
25–7 against Hard with no unresolved or operational outcomes.

The same frozen bundle scored 196–4 against Medium and 200–0 against Easy in
separate 200-game / 100 color-swapped-pair qualifications. Their
distribution-free one-sided 95% lower bounds were 85.76% and 87.76%,
respectively. All 400 games resolved normally, with zero invalid actions,
crashes, fallbacks, repetitions, action-limit outcomes, artifact mismatches,
or pair-integrity failures.

The compact release evidence is checked in as the
[`32-game shipping-clock screen`](native/very-hard/final-hard-32-book.json),
[`1,000-game Hard qualification`](native/very-hard/final-hard-1000.json), and
[`400-game Medium/Easy qualification`](native/very-hard/final-medium-easy-400-corrected.json).
Raw game shards, diagnostic traces, and intermediate regression reports are
local generated output and are intentionally ignored by Git.

## Builds

### Standalone browser file

```powershell
npm run build:standalone
```

The self-contained result is `dist/linith-standalone.html`. It can be opened directly without a web server.

### Desktop

Electron owns all three desktop targets:

```powershell
npm run dist:win
npm run dist:linux
npm run dist:mac
```

Run each packaging command on its native operating system. Packages are written to `release/`.

### Android

With JDK 21 and Android SDK 36 installed, build the shared renderer first, then sync it into Capacitor:

```powershell
npm run build
cd platforms/android-wrapper
npm ci
npm run sync
npm run open
```

The `www/` directory and Capacitor's copied Android web assets are generated. Game source must remain in the root `src/` tree. See [`platforms/android-wrapper/README.md`](platforms/android-wrapper/README.md) for Android-specific commands.

## Epsilon

`epsilon/` contains Linith's AlphaZero-style policy/value networks, Monte Carlo tree search, self-play, evaluation, model promotion, and optional Pybind11/C++ acceleration.

Create an environment and install its dependencies:

```sh
cd epsilon
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install the appropriate CUDA-enabled PyTorch build when training on a GPU. Build the accelerated module from `epsilon/cport` with:

```sh
python setup.py build_ext --inplace
```

Model checkpoints, replay data, logs, compiled extensions, and environments are generated artifacts and are not committed.

## Project structure

| Path | Contents |
| --- | --- |
| `src/main/` | Electron main process |
| `src/preload/` | Secure desktop preload bridge |
| `src/renderer/` | Shared TypeScript game, UI, styles, and HTML shell |
| `scripts/` | Standalone build tooling |
| `build/` | Authored icons and sound assets |
| `platforms/android-wrapper/` | Capacitor project and Android-only shell assets |
| `tests/` | TypeScript rule tests |
| `epsilon/` | Self-play, MCTS, neural-network training, and C++ acceleration |
| `pressure/` | *Pressure*, the Linith strategy book and source chapters |
| `flash/` | Additional artwork and historical media sources |

## Technology

Linith uses TypeScript, Vite, Electron, Capacitor, the Web Audio API, browser local storage, Node's test runner, Python, PyTorch, NumPy, Pybind11, and C++.

## License

Linith is released under the [MIT License](LICENSE.md).
