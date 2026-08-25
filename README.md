# Linith

**Linith** is a two-player abstract strategy game of movement, pressure and encirclement.

Played on a 10×10 board, **Sun** and **Moon** place Swans and Stones, move formations, push opposing Swans and gradually restrict the opponent's available space.

Swans are defeated by being completely encircled and frozen.

This repository contains the browser version of Linith, built with HTML, CSS and JavaScript.

## How to Play

Sun places the first Swan. Moon places the second on a non-adjacent square, then takes the first turn.

On a turn, players can:

* Place a Swan.
* Place a Stone.
* Move one or more Swans one square.
* Push adjacent enemy Swans.

Players may have up to **six Swans**. Once both sides have six, turns begin with **two actions**.

Stones adjacent to moving Swans may move with them, allowing the board itself to be reshaped.

## Encirclement

A Swan group freezes when it has no adjacent empty squares.

Frozen Swans remain on the board but cannot move or spawn.

Freezing an enemy Swan grants an additional action.

A player loses when their final active Swan is encircled. If both players' final active Swans are encircled during the same action, the game is a draw.

## Game Modes

* Local
* AI Plays Sun
* AI Plays Moon

AI difficulty:

* Easy
* Medium
* Hard

AI styles:

* Doctrinal
* Constrictor
* Rupture
* Blizzard
* Librarian
* Swarm
* Fortress

## Features

* 10×10 board
* Multi-Swan movement
* Enemy-Swan pushing
* Dynamic Stone movement
* Encirclement and freezing
* Hint system
* Undo and move-history navigation
* Surrender
* Stopwatch and chess clocks
* Save/load
* Game review and replay
* Sound effects
* Configurable board appearance

## Running

Linith can be built for several targets:

Standalone HTML
Windows
macOS
Linux
Android

Build the desired version from the source, then run the resulting HTML file or platform-specific application.

## Technology

Linith is built with:

* HTML
* CSS
* JavaScript
* Web Audio
* Browser local storage
