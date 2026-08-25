<p align="center">
  <img src="flash/linith_logo_5.png" alt="Linith logo" width="520" />
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
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#native-wrappers">Builds</a>
  ·
  <a href="#epsilon">Epsilon</a>
  ·
  <a href="LICENSE">MIT license</a>
</p>

## About the game

Played on a 10×10 board, **Sun** and **Moon** place Swans and Stones, move
formations, push opposing Swans, and gradually restrict the opponent's
available space.

Swans are defeated by being completely encircled and frozen.

This repository contains Linith 0.232, its browser and native wrappers, and
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
- Easy, Medium, and Hard difficulty
- Doctrinal, Constrictor, Rupture, Blizzard, Librarian, Swarm, and Fortress AI styles

AI style notes are available in [`aipersons.txt`](aipersons.txt).

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
- Browser, Windows, macOS, and Android targets
- AlphaZero-style training and evaluation through Epsilon

## Quick start

Open [`web/current build/linith_0.232_web.html`](web/current%20build/linith_0.232_web.html)
in a modern browser. The game is a self-contained HTML file and does not need a
web server.

The standalone variant used by native wrappers is
[`web/current build/linith_0.232.html`](web/current%20build/linith_0.232.html).

## Native wrappers

### Windows

Install the .NET 8 SDK and the Microsoft Edge WebView2 Runtime, then run:

```powershell
dotnet run --project windows/Linith/Linith.csproj
```

The project uses a bundled fixed WebView2 runtime when one is present locally
and otherwise falls back to the installed system runtime. The large fixed
runtime is not stored in Git.

### macOS

Install Node.js, then run:

```sh
cd mac
npm ci
npm start
```

Create a distributable package with `npm run dist:mac` on macOS.

### Android

Install Node.js, a supported JDK, and the Android SDK, then run:

```sh
cd android/lindroid
npm ci
npx cap sync android
cd android
./gradlew assembleDebug
```

On Windows, use `gradlew.bat assembleDebug` for the final command.

## Epsilon

Epsilon implements Linith's game state and action space, policy/value networks,
Monte Carlo tree search, self-play, evaluation, model promotion, and optional
Pybind11/C++ acceleration.

Create an environment and install the Python dependencies:

```sh
cd epsilon
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install the appropriate
CUDA-enabled PyTorch build when training on a GPU. Build the accelerated module
from `epsilon/cport` with:

```sh
python setup.py build_ext --inplace
```

Model checkpoints (`.pt`), replay/training arrays (`.npz`), logs, compiled
extensions, and virtual environments are generated artifacts. They remain local
and should be published through release or model storage with checksums when
they need to be shared.

## Project structure

| Path | Contents |
| --- | --- |
| `web/` | Current standalone browser builds |
| `windows/` | .NET 8 WinForms/WebView2 wrapper |
| `mac/` | Electron wrapper and packaged web source |
| `android/` | Capacitor Android project |
| `epsilon/` | Rules, self-play, MCTS, neural-network training, and C++ acceleration |
| `flash/` | Logos, icons, and sound assets |
| `pressure/` | *Pressure*, the Linith strategy book and source chapters |

Historical exported HTML files, compiled applications, dependencies, IDE state,
and local saves are intentionally not versioned. Git history is the version
archive from this point forward.

## Technology

Linith is built with:

- HTML, CSS, and JavaScript
- Web Audio and browser local storage
- .NET 8, WinForms, and WebView2
- Electron and Node.js
- Capacitor and Android
- Python, PyTorch, NumPy, Pybind11, and C++

## License

Linith is released under the [MIT License](LICENSE).
