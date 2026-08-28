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
  <a href="#development">Development</a>
  ·
  <a href="#builds">Builds</a>
  ·
  <a href="#epsilon">Epsilon</a>
  ·
  <a href="LICENSE">MIT license</a>
</p>

## About the game

Played on a 10×10 board, Sun and Moon place Swans and Stones, move formations, push opposing Swans, and gradually restrict the opponent's available space. A Swan group freezes when it has no adjacent empty squares; a player loses when their final active Swan is encircled.

The complete rules are in [`rules.txt`](rules.txt), and AI style notes are in [`aipersons.txt`](aipersons.txt).

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

## Features

- Local two-player and AI games
- Easy, Medium, and Hard difficulty
- Seven AI playing styles
- Multi-Swan movement and enemy-Swan pushing
- Dynamic Stone movement, encirclement, and freezing
- Hint, undo, move-history, replay, save/load, and surrender
- Stopwatch and chess clocks
- Sound and configurable board appearance
- Standalone browser, Windows, macOS, Linux, and Android targets

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
