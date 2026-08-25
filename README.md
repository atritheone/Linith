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
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#native-wrappers">Builds</a>
  ·
  <a href="#epsilon">Epsilon</a>
  ·
  <a href="LICENSE">MIT license</a>
</p>

## About the game

On a 10×10 board, Sun and Moon place Swans and Stones, move formations, push
opposing Swans, and gradually restrict the opponent's space.

A Swan becomes frozen when its group is completely encircled, and a player
loses when their final active Swan is frozen.

This repository contains Linith 0.232, its browser and native wrappers, and
**Epsilon**: the Linith-specific AlphaZero-style self-play and neural-network
training system.

## Quick start

Open [`web/current build/linith_0.232_web.html`](web/current%20build/linith_0.232_web.html)
in a modern browser. The game is a self-contained HTML file and does not need a
web server.

The standalone variant used by native wrappers is
[`web/current build/linith_0.232.html`](web/current%20build/linith_0.232.html).

## Game modes

- Local two-player
- AI plays Sun or Moon
- Easy, Medium, and Hard difficulty
- Doctrinal, Constrictor, Rupture, Blizzard, Librarian, Swarm, and Fortress AI styles

The complete rules are in [`rules.txt`](rules.txt), with AI style notes in
[`aipersons.txt`](aipersons.txt).

## Repository layout

| Path | Contents |
| --- | --- |
| `web/` | Current standalone browser builds |
| `windows/` | .NET 8 WinForms/WebView2 wrapper |
| `mac/` | Electron wrapper and packaged web source |
| `android/` | Capacitor Android project |
| `epsilon/` | AlphaZero-style rules, self-play, MCTS, training, and C++ acceleration |
| `flash/` | Logos, icons, and sound assets |
| `pressure/` | *Pressure*, the Linith strategy book and source chapters |

Historical exported HTML files, compiled applications, dependencies, IDE state,
and local saves are intentionally not versioned. Git history is the version
archive from this point forward.

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

See [`epsilon/setup.txt`](epsilon/setup.txt) for the GPU-host workflow.

Model checkpoints (`.pt`), replay/training arrays (`.npz`), logs, compiled
extensions, and virtual environments are generated artifacts. They remain local
and should be published through release or model storage with checksums when
they need to be shared.

## License

Linith is released under the [MIT License](LICENSE).
