// This module retains the original game orchestration while the pure AI and
// encirclement rules live in independently testable TypeScript modules.
// @ts-nocheck -- incremental migration boundary for the original UI/game coordinator.
import { linithAI } from "./ai";
import { computeFreezesOn } from "./encirclement";
import VeryHardWorker from "./veryHard/worker?worker&inline";
import { chooseVeryHardTimeBudget, detectVeryHardPlatform } from "./veryHard/timeManager";
import { playReady, sfxReady } from "../sound";

export function initGame(): void {
  /* -------------------- constants & enums -------------------- */
    const SIZE = 10;                                        // board dimension
    const EMPTY = 0, SWAN_SUN = 1, SWAN_MOON = 2,           // tile codes
          STONE = 3, FROZEN_SUN = 4, FROZEN_MOON = 5;
    const SUN = 1, MOON = 2;                                // players
    const DIRS8 = [                                         // 8-neighborhood
      [-1,-1],[-1,0],[-1,1],
      [ 0,-1],       [ 0,1],
      [ 1,-1],[ 1,0],[ 1,1]
    ];
    const DIRS4 = [[-1,0],[1,0],[0,-1],[0,1]];              // orthogonal

    // svgs for piece icons (styled via currentColor)
    const SVG_SUN = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="4" fill="currentColor"/>
        <g stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <line x1="12" y1="1.5"  x2="12" y2="5"/>
          <line x1="12" y1="19"   x2="12" y2="22.5"/>
          <line x1="1.5" y1="12"  x2="5"  y2="12"/>
          <line x1="19" y1="12"   x2="22.5" y2="12"/>
          <line x1="4.2" y1="4.2"   x2="6.8"  y2="6.8"/>
          <line x1="17.2" y1="17.2" x2="19.8" y2="19.8"/>
          <line x1="17.2" y1="6.8"  x2="19.8" y2="4.2"/>
          <line x1="4.2"  y1="19.8" x2="6.8"  y2="17.2"/>
        </g>
      </svg>`;
    const SVG_MOON = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path fill="currentColor" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
      </svg>`;

    // momentum / actions-per-turn (1 until both sides reach six total swans, then 2)
    let movesLeft = 1;

    /* -------------------- utility - counts & phases -------------------- */
    function bothAtSix(){
      return countTotalSwans(SUN) >= 6 && countTotalSwans(MOON) >= 6;
    }

    function beginTurn(){
      enableBoardInteraction();
      movesLeft = bothAtSix() ? 2 : 1;

      // chess mark active side for highlight only
      if (clockMode && clockMode.startsWith('chess')) {
        activeSide = (current === SUN ? 'SUN' : 'MOON');
        markActive(activeSide);
      }

      // stopwatch ensure it’s ticking when that mode is active
      if (clockMode === 'stopwatch') {
        timerStart();
      }

      // keep the faces in sync (in case side/mode changed)
      updateClockFacesVisibility();
    }

    /* -------------------- state -------------------- */
    let board   = Array.from({length: SIZE}, ()=>Array(SIZE).fill(EMPTY)); // 10x10 matrix
    let turn    = 'setup';            // 'setup' or 'play'
    let toPlace = 'sun';              // which setup step: 'sun' or 'moon'
    let current = MOON;               // current player (moon goes first after setup)
    let selected = new Set();         // selected swans for movement (indices)
    let splitIdx = null;              // cell index showing “swan/stone” chooser
    let anchorIdx = null;             // anchor swan index for move hints
    let action = null;                // null | 'placeSwan' | 'placeStone' | 'moveSwans'
    let history = [];                 // json snapshots for undo
    let aiSide = null;                // null | sun | moon
    let gameOver = false;             // game over
    let aiMoveTimer = null;   // pending AI timeout id, or null when none
    let veryHardRequestSequence = 0;
    let veryHardSessionSequence = 0;
    let veryHardWorker = null;
    let activeVeryHardRequest = null;
    let aiThinking = false;

    let aiDifficulty = (localStorage.getItem('linith_ai_difficulty') || 'medium');   // default ai difficulty
    // difficulty bridge (shared via localstorage)
    window.linithGetDifficulty = ()=> localStorage.getItem('linith_ai_difficulty') || 'medium';
    window.linithSetDifficulty = (d)=> {
      const nextDifficulty = d || 'medium';
      const changed = nextDifficulty !== aiDifficulty;
      aiDifficulty = nextDifficulty;
      localStorage.setItem('linith_ai_difficulty', aiDifficulty);
      if (changed) restartAiAfterConfigurationChange();
    };

    // style bridge (shared via localstorage)
    let aiStyle = (localStorage.getItem('linith_ai_style') || 'doctrinal');
    window.linithGetStyle = ()=> localStorage.getItem('linith_ai_style') || 'doctrinal';
    window.linithSetStyle = (s)=> {
      const nextStyle = s || 'doctrinal';
      const changed = nextStyle !== aiStyle;
      aiStyle = nextStyle;
      localStorage.setItem('linith_ai_style', aiStyle);
      if (changed) restartAiAfterConfigurationChange();
    };
    // ensure our in-iife copy matches stored value on load
    aiStyle = window.linithGetStyle();
    aiDifficulty = window.linithGetDifficulty();
    let appMode = 'menu';             // 'menu' | 'playing' | 'review'
    let boardLocked = false;          // board locking
    let moveHighlightsOn = true;      // ui toggle; default on

    // review-mode sfx control - suppress during multi-step jumps (e.g., goToTip)
    // and allow a one-time forced play on the final frame
    let suppressReviewSFX = false;
    let forceReviewSFXOnce = false;

    // transient recent move highlights (origin/destination)
    let recentFromCells = new Set();
    let recentToCells   = new Set();
    let recentTimer     = null;

    function flashRecent(fromIdxs = [], toIdxs = [], ms = 800) {
      try { if (recentTimer) clearTimeout(recentTimer); } catch {}

      // respect user setting
      if (!moveHighlightsOn) {
        // ensure any old highlight is cleared
        recentFromCells.clear();
        recentToCells.clear();
        render();
        return;
      }

      recentFromCells = new Set(fromIdxs);
      recentToCells   = new Set(toIdxs);
      render();
      recentTimer = setTimeout(() => {
        recentFromCells.clear();
        recentToCells.clear();
        render();
      }, ms);
    }

    // In local vs local show all highlights. In vs AI show only AI's.
    // actor can be 'ai' | 'human' | null (infer via current and aiSide)
    function flashRecentIfAIOnly(fromIdxs = [], toIdxs = [], ms = 800, actor = null) {
      const aiGame = (aiSide !== null);
      let isAIActor;
      if (actor == null) {
        isAIActor = aiGame && (current === aiSide);
      } else {
        isAIActor = (actor === 'ai');
      }
      if (!aiGame || isAIActor) {
        return flashRecent(fromIdxs, toIdxs, ms);
      }
      // In AI game and human action: suppress highlight
      return;
    }

    // ----- stopwatch runtime (mm:ss.cc) -----
    let tInterval = null;
    let tStartPerf = 0;   // performance.now() at start
    let tAccUs    = 0;    // accumulated microseconds

    function fmtMMSScc(us) {
      const totalSec = Math.floor(Math.max(0, us) / 1_000_000);
      const mm = Math.floor(totalSec / 60);
      const ss = totalSec % 60;
      const cs = Math.floor((us % 1_000_000) / 10_000); // 00..99
      const pad = (n, l=2) => String(n).padStart(l, '0');
      return `${pad(mm)}:${pad(ss)}.${pad(cs)}`;
    }
    function renderTimer() {
      if (!elTimer) return;
      let elapsedUs = tAccUs;
      if (tStartPerf) elapsedUs += (performance.now() - tStartPerf) * 1000;
      elTimer.textContent = fmtMMSScc(elapsedUs);
    }
    function timerStart() {
      if (tInterval) return;
      tStartPerf = performance.now();
      tInterval = setInterval(renderTimer, 50);
    }
    function timerStop() {
      if (!tInterval) return;
      clearInterval(tInterval);
      tInterval = null;
      if (tStartPerf) {
        tAccUs += (performance.now() - tStartPerf) * 1000;
        tStartPerf = 0;
      }
      renderTimer();
    }
    function timerReset() {
      if (tInterval) { clearInterval(tInterval); tInterval = null; }
      tStartPerf = 0; tAccUs = 0; renderTimer();
    }

    const CLOCK_KEY = 'linith_clock_mode'; // 'off' | 'stopwatch' | 'chess-10' | 'chess-5' | 'chess-3'
    let clockMode = localStorage.getItem(CLOCK_KEY) || 'off';
    window.linithGetClockMode = () => clockMode;
    window.linithSetClockMode = (m) => {
      clockMode = m || 'off';
      localStorage.setItem(CLOCK_KEY, clockMode);
      applyClockVisibility();
      updateClockFacesVisibility();
      resetClocksForNewGame();

        // stopwatch lifecycle - do not run before play starts.
        if (clockMode === 'stopwatch') {
          // show 00:00.00 until the game actually begins.
          timerStop();
          timerReset();

          // if we are already in an active game (post-setup) begin ticking.
          if (appMode === 'playing' && turn === 'play') {
            timerStart();
          }
        } else {
          // leaving stopwatch mode
          timerStop();
          timerReset();
        }
    };

    // move highlight preference
    const HIGHLIGHT_KEY = 'linith_move_highlights';  // 'on' | 'off'
    window.linithGetMoveHighlights = () => moveHighlightsOn;
    window.linithSetMoveHighlights = (m) => {
      const mode = (m === 'off') ? 'off' : 'on';
      moveHighlightsOn = (mode === 'on');
      localStorage.setItem(HIGHLIGHT_KEY, mode);

      // if turning off while a highlight is active, clear it immediately
      if (!moveHighlightsOn) {
        try { if (recentTimer) clearTimeout(recentTimer); } catch {}
        recentFromCells.clear();
        recentToCells.clear();
        render();
      }
    };

    // -------------------- chess clock state --------------------
    const CHESS_PRESETS_MIN = { 'chess-10':10, 'chess-5':5, 'chess-3':3 };
    let chessSunUs  = 0;   // remaining microseconds for Sun
    let chessMoonUs = 0;   // remaining microseconds for Moon
    let activeSide  = null; // 'SUN' | 'MOON' | null
    let lastPerf    = 0;   // last high-res sample while running

    function isLocalOnly(){ return aiSide === null; }
    function humanSide(){
      if (aiSide === SUN)  return 'MOON'; // human plays Moon
      if (aiSide === MOON) return 'SUN';  // human plays Sun
      return null; // local
    }

    // format mm:ss for chess faces
    function fmtMMSS_fromUs(us) {
      const totalSec = Math.max(0, Math.floor(us / 1_000_000));
      const mm = Math.floor(totalSec / 60);
      const ss = totalSec % 60;
      const pad = (n)=> (n<10?'0':'')+n;
      return `${pad(mm)}:${pad(ss)}`;
    }

    function setChessPreset(presetKey) {
      const mins = CHESS_PRESETS_MIN[presetKey] || 10;
      chessSunUs  = mins * 60 * 1_000_000;
      chessMoonUs = mins * 60 * 1_000_000;
      renderChessClocks();
    }

    function renderChessClocks() {
      if (elClockSun)  elClockSun.textContent  = fmtMMSS_fromUs(chessSunUs);
      if (elClockMoon) elClockMoon.textContent = fmtMMSS_fromUs(chessMoonUs);
    }

    function applyClockVisibility() {
      // stopwatch visibility
      if (clockMode === 'stopwatch') {
        elTimer?.classList.remove('hidden');
      } else {
        elTimer?.classList.add('hidden');
      }

      // chess clocks group visibility
      const chessOn = clockMode.startsWith('chess');
      if (chessOn) elClocks?.classList.remove('hidden');
      else         elClocks?.classList.add('hidden');

      // ensure the correct faces are shown/hidden (AI vs human)
      updateClockFacesVisibility();
    }

    function updateClockFacesVisibility() {
      if (!elClockSun || !elClockMoon) return;

      // if not in a chess mode, show both spans (the group is hidden elsewhere)
      if (!clockMode.startsWith('chess')) {
        elClockSun.style.display  = '';
        elClockMoon.style.display = '';
        return;
      }

      // local (human vs human) - show both faces
      if (aiSide === null) {
        elClockSun.style.display  = '';
        elClockMoon.style.display = '';
        return;
      }

      // ai game - show ONLY the human's face
      const h = humanSide(); // already defined above
      elClockSun.style.display  = (h === 'SUN')  ? '' : 'none';
      elClockMoon.style.display = (h === 'MOON') ? '' : 'none';
    }

    function markActive(side) {
      elClockSun?.classList.toggle('active', side==='SUN');
      elClockMoon?.classList.toggle('active', side==='MOON');
    }

    // Ensure chess clocks follow the present (tip) turn only, regardless of replay view
    function markActiveFromTip() {
      try {
        // Prefer the last entry of replay (the tip). Fall back to history if needed.
        let tipSnap = null;
        if (replay && replay.length) {
          try { tipSnap = JSON.parse(replay[replay.length - 1]); } catch {}
        }
        if (!tipSnap && history && history.length) {
          try { tipSnap = JSON.parse(history[history.length - 1]); } catch {}
        }

        if (tipSnap && tipSnap.turn === 'play') {
          const side = (tipSnap.current === SUN ? 'SUN' : 'MOON');
          activeSide = side;
          markActive(side);
        } else {
          activeSide = null;
          markActive(null);
        }
      } catch {
        // On any parsing error, do not alter current activeSide
      }
    }

    function resetClocksForNewGame() {
      if (clockMode.startsWith('chess')) {
        setChessPreset(clockMode);
        activeSide = null;
        markActive(null);
      }
      // keep stopwatch reset in your timerreset()
    }

    // master tick (shared loop) - drives stopwatch and chess clocks
    let masterInterval = null;
    function startMasterLoop() {
      if (masterInterval) return;
      lastPerf = performance.now();
      masterInterval = setInterval(tickMaster, 50);
    }
    function stopMasterLoop() {
      if (!masterInterval) return;
      clearInterval(masterInterval);
      masterInterval = null;
    }

    function tickMaster() {
      const now = performance.now();
      const dtUs = (now - lastPerf) * 1000;
      lastPerf = now;

      // stopwatch (only if enabled and running by your existing flags)
      if (tStartPerf || tAccUs) renderTimer(); // your existing render handles the right display

      // chess - decrement only the active side
      if (clockMode.startsWith('chess') && activeSide) {
        if (isLocalOnly()) {
          if (activeSide === 'SUN')  chessSunUs  = Math.max(0, chessSunUs  - dtUs);
          if (activeSide === 'MOON') chessMoonUs = Math.max(0, chessMoonUs - dtUs);
        } else {
          // ai game - human-only countdown
          const h = humanSide();
          if (h === activeSide) {
            if (h === 'SUN')  chessSunUs  = Math.max(0, chessSunUs  - dtUs);
            if (h === 'MOON') chessMoonUs = Math.max(0, chessMoonUs - dtUs);
          }
        }
        renderChessClocks();

        if (chessSunUs <= 0 && chessMoonUs <= 0) {
          playReady('draw', { gain: 0.5, rate: 1.0 });
          timerStop();
          stopMasterLoop();
          finishgame('Draw by time as both clocks expired.', 'Draw.');
          return;
        }

        if (chessSunUs <= 0 || chessMoonUs <= 0) {
          const loserSide = (chessSunUs <= 0) ? 'Sun' : 'Moon';
          const winnerSide = (loserSide === 'Sun') ? 'Moon' : 'Sun';
          const winnerIcon = (winnerSide === 'Sun') ? '☼' : '☾';

          const h = humanSide(); // 'SUN' | 'MOON' | null
          if (h) {
            const humanWins =
              (winnerSide === 'Sun'  && h === 'SUN') ||
              (winnerSide === 'Moon' && h === 'MOON');
            playReady(humanWins ? 'win' : 'loss', { gain: 0.5, rate: 1.0 });
          } else {
            // local (human vs human) - play the losing side’s loss
            playReady('loss', { gain: 0.5, rate: 1.0 });
          }

          timerStop();
          stopMasterLoop();

          // on-screen message first argument, log message second argument
          finishgame(
            `${winnerSide} ${winnerIcon} wins by time as ${loserSide}’s clock expired.`,
            `${winnerSide} wins.`
          );
          return;
        }
      }
    }

    // dom references
    const elBoard      = document.getElementById('board');
    const elLog        = document.getElementById('log');
    const elStartMenu  = document.getElementById('startMenu');
    const elTurn       = document.getElementById('turnInfo');
    const elTimer      = document.getElementById('timer');
    const elClocks     = document.getElementById('chessClocks');
    const elClockSun   = document.getElementById('clockSun');
    const elClockMoon  = document.getElementById('clockMoon');
    const btnUndo      = document.getElementById('actUndo');
    const btnReset     = document.getElementById('actReset');
    const btnLocal     = document.getElementById('btnLocal');
    const btnAiSun     = document.getElementById('btnAiSun');
    const btnAiMoon    = document.getElementById('btnAiMoon');
    const btnHint      = document.getElementById('actHint');
    const btnSurrender = document.getElementById('actSurrender');
    const confirmBox   = document.getElementById('surrenderConfirm');
    const panelRoot    = document.getElementById('controls') || document;
    const btnBack      = document.getElementById('actBack');
    const btnForward   = document.getElementById('actForward');
    const btnSave      = document.getElementById('actSave');
    const btnLoad      = document.getElementById('actLoad');
    // replay controls (under the log)
    const elReplayControls = document.getElementById('replayControls');
    const btnReplayPP      = document.getElementById('replayPlayPause');
    const inputReplaySpeed = document.getElementById('replaySpeed');
    const elReplaySpeedVal = document.getElementById('replaySpeedVal');

    // Track final outcome texts to embed in saved game records
    let lastOutcomeShort = null;    // e.g. "Sun wins.", "Draw by saturation."
    let lastOutcomeDetailed = null; // e.g. on-screen popup text with details

    sfxReady.finally(() => startMasterLoop());

    // cancel empty-cell chooser if the user clicks outside the board
    document.addEventListener('pointerdown', (e) => {
      if (elBoard && !elBoard.contains(e.target)) {
        let changed = false;
        // If the split chooser is open, close it
        if (splitIdx !== null) { splitIdx = null; changed = true; }
        // If any swans are selected or we're in move mode, clear that selection/state
        if ((selected && selected.size > 0) || action === 'moveSwans') {
          selected.clear();
          action = null;
          anchorIdx = null;
          changed = true;
        }
        if (changed) render();
      }
    }, { capture: true });

    /* -------------------- serialisation / history -------------------- */

    // running count of full moves in this game (for replay/log alignment)
    let moveNumber = 0;

    // -------- imported replay auto-play state --------
    let isAutoPlaying = false;
    let replayTimer = null;
    let replaySpeedX = 1.0;       // 0.25x .. 4x
    const baseDelayMs = 1000;     // 1x = 1 move/sec

    function setReplayControlsVisible(v){
      if (!elReplayControls) return;
      elReplayControls.style.display = v ? 'flex' : 'none';
    }

    function updateReplayButtonIcon(){
      if (!btnReplayPP) return;
      btnReplayPP.textContent = isAutoPlaying ? '⏸' : '▶';
      btnReplayPP.setAttribute('aria-label', isAutoPlaying ? 'Pause' : 'Play');
      btnReplayPP.title = isAutoPlaying ? 'Pause' : 'Play';
    }

    function setReplaySpeed(x){
      replaySpeedX = Math.max(0.25, Math.min(4, Number(x)||1));
      if (elReplaySpeedVal) elReplaySpeedVal.textContent = `${replaySpeedX.toFixed(2).replace(/\.00$/,'.0')}×`;
      // if currently running, restart timer to apply new delay
      if (isAutoPlaying) {
        stopReplayAuto();
        startReplayAuto();
      }
    }

    function tickReplayOnce(){
      // advance one step; if at the end, stop
      const wasAtTip = isAtTip();
      const R = replay ? replay.length : 0;
      if (!R) { stopReplayAuto(); return; }
      if (replayIdx >= R - 1 && wasAtTip) { stopReplayAuto(); return; }
      // advance using non-manual path so we don't auto-pause ourselves
      reviewForward(true);
      // if we just reached the end, stop
      if (replayIdx >= (replay?.length||0) - 1 && isAtTip()) {
        stopReplayAuto();
      }
    }

    function startReplayAuto(){
      if (isAutoPlaying) return;
      if (!replay || !replay.length) return;
      // don't auto-play from menu or normal play
      if (appMode !== 'review') return;
      isAutoPlaying = true;
      updateReplayButtonIcon();
      const delay = Math.max(60, Math.round(baseDelayMs / replaySpeedX));
      replayTimer = setInterval(tickReplayOnce, delay);
    }

    function stopReplayAuto(){
      if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
      if (isAutoPlaying) { isAutoPlaying = false; updateReplayButtonIcon(); }
    }

    function toggleReplayAuto(){
      if (isAutoPlaying) stopReplayAuto(); else startReplayAuto();
    }

    function cloneState(meta){
      const snap = {
        board,
        turn,
        toPlace,
        current,
        movesLeft
      };

      // capture clock timing at this snapshot for per-move replay
      try {
        if (typeof clockMode === 'string' && clockMode && clockMode !== 'off') {
          if (clockMode === 'stopwatch') {
            // elapsed microseconds up to now (accum + live delta if running)
            let us = 0;
            try {
              us = (typeof tAccUs === 'number' ? tAccUs : 0);
              if (typeof tStartPerf === 'number' && tStartPerf) {
                us += (performance.now() - tStartPerf) * 1000;
              }
            } catch {}
            snap.clockAt = { mode: 'stopwatch', us: Math.max(0, Math.floor(us)) };
          } else if (clockMode.startsWith('chess')) {
            // Remaining microseconds for sun/moon at this moment.
            // If vs AI, only record the human side per requirement.
            const h = (typeof humanSide === 'function') ? humanSide() : null; // 'SUN' | 'MOON' | null
            const recordBoth = (h === null); // local game: record both
            const ca = { mode: 'chess', preset: clockMode };

            // Derive preset base time in case clocks haven't been initialised yet (e.g., initial snapshot)
            let baseUs = null;
            try {
              const mins = CHESS_PRESETS_MIN && CHESS_PRESETS_MIN[clockMode];
              if (typeof mins === 'number') baseUs = Math.max(0, Math.floor(mins * 60 * 1_000_000));
            } catch {}

            const sunNow  = (typeof chessSunUs  === 'number') ? Math.max(0, Math.floor(chessSunUs))  : null;
            const moonNow = (typeof chessMoonUs === 'number') ? Math.max(0, Math.floor(chessMoonUs)) : null;
            const bothZero = (sunNow === 0 && moonNow === 0);

            if (recordBoth || h === 'SUN') {
              if (sunNow !== null) ca.sunUs = (bothZero && baseUs !== null) ? baseUs : sunNow;
            }
            if (recordBoth || h === 'MOON') {
              if (moonNow !== null) ca.moonUs = (bothZero && baseUs !== null) ? baseUs : moonNow;
            }
            // Optional hint of who is recorded
            ca.who = recordBoth ? 'both' : h;
            snap.clockAt = ca;
          }
        }
      } catch {}

      // allow extra metadata (labels, move number, etc.)
      if (meta && typeof meta === 'object'){
        for (const k in meta) {
          snap[k] = meta[k];
        }
      }

      return JSON.stringify(snap);
    }

    function restoreState(s){
      const v = JSON.parse(s);
      board   = v.board;
      turn    = v.turn;
      toPlace = v.toPlace;
      current = v.current;
      movesLeft = (typeof v.movesLeft === 'number') ? v.movesLeft : movesLeft;
    }

    function pushHistory(meta){
      // if we time-travel and then make a new move, discard the future branch
      if (replay && replay.length && replayIdx < replay.length - 1){
        history = history.slice(0, replayIdx + 1);
      }

      // advance move counter only when this snapshot represents an actual move
      if (meta && meta.isMove){
        moveNumber++;
      }

      const enrichedMeta = Object.assign({}, meta || {}, { moveNumber });

      history.push(cloneState(enrichedMeta));
      // no pruning: keep a perfect record of the whole game
      // (if you *really* want a cap, make it very generous)

      rebuildReplay();
      replayIdx = replay.length - 1;
      updateActionButtonsEnabled();
      updateLogActive();
    }

    // -------- review navigation state --------
    let replay = null;
    let replayIdx = 0;
    // flag: true only when we imported a saved file and are in pure replay mode
    let recite = false;
    // when true, rebuildReplay will avoid appending a transient live tip
    // (used between pushing a move snapshot and endTurn() normalising live state)
    let suppressTipOnce = false;

    function isAtTip(){
      return !!(replay && replay.length && replayIdx === replay.length - 1);
    }

    // build replay from the current game stack (call when entering review)
    function rebuildReplay(){
      // history is the canonical record
      replay = history.slice();

      // Decide whether we need an extra live "tip" snapshot.
      // Compare only core state (ignore metadata like isMove/moveNumber) so
      // we don't create duplicate adjacent snapshots around the current turn.
      const last = replay.length ? JSON.parse(replay[replay.length - 1]) : null;
      const sameCore = !!(last && last.board &&
        JSON.stringify(last.board) === JSON.stringify(board) &&
        last.turn === turn && last.toPlace === toPlace &&
        last.current === current &&
        (typeof last.movesLeft === 'number' ? last.movesLeft : movesLeft) === movesLeft);

      if (!sameCore){
        // Preserve current moveNumber on the live tip for log alignment
        if (!suppressTipOnce) replay.push(cloneState({ moveNumber }));
      }

      // keep the cursor at the tip
      replayIdx = replay.length - 1;
      updateReviewButtons();

      syncInteractionLock();
      // tip suppression applies only for this rebuild
      suppressTipOnce = false;

      // Ensure clock highlight/ticking follow the real present turn
      markActiveFromTip();
    }

    // highlight the board changes between replay frames (review + recite)
    function highlightReplayFrame(frameIdx) {
      // only in timeline views - pure recite or in-review scrub
      // Also allow highlighting when we've landed on the real tip (last move)
      if (!recite && appMode !== 'review' && !isAtTip()) return false;

      if (!replay || !Array.isArray(replay) || replay.length === 0) return false;

      // Determine which frame to use for diffing. If the given frame isn't a move
      // (e.g., it's the live tip snapshot), walk backward to the most recent move.
      let i = Math.min(Math.max(frameIdx, 0), replay.length - 1);
      let snap, prev;
      try {
        let s = JSON.parse(replay[i]);
        if (!(s && s.isMove)) {
          for (let j = i; j >= 0; j--) {
            let sj;
            try { sj = JSON.parse(replay[j]); } catch { continue; }
            if (sj && sj.isMove) { i = j; s = sj; break; }
          }
        }
        // If we couldn't find a move or it's the very first entry, we cannot diff
        if (!s || !s.isMove || i <= 0) return false;
        snap = s;
        prev = JSON.parse(replay[i - 1]);
      } catch {
        return false;
      }

      const currBoard = snap && snap.board;
      const prevBoard = prev && prev.board;
      if (!Array.isArray(currBoard) || !Array.isArray(prevBoard)) return false;

      const fromIdxs = [];
      const toIdxs   = [];

      for (let r = 0; r < SIZE; r++) {
        const pr = prevBoard[r] || [];
        const cr = currBoard[r] || [];
        for (let c = 0; c < SIZE; c++) {
          const before = pr[c];
          const after  = cr[c];
          if (before === after) continue;

          const wasEmpty = (before === EMPTY || before === null || before === undefined);
          const nowEmpty = (after  === EMPTY || after  === null || after  === undefined);

          if (!wasEmpty) fromIdxs.push(idx(r, c)); // origin(s)
          if (!nowEmpty) toIdxs.push(idx(r, c));   // destination(s)
        }
      }

      if (!fromIdxs.length && !toIdxs.length) return false;

      // Uses the existing transient highlight system and respects moveHighlightsOn
      flashRecent(fromIdxs, toIdxs, 900);
      return true;
    }

    function syncInteractionLock(){
      const atTip = (typeof isAtTip === 'function') ? isAtTip() : false;

      if (recite || gameOver) {
        // imported or finished game: always non-interactive
        boardLocked = true;
        if (appMode !== 'menu') appMode = 'review';
      } else {
        // live game: locked when rewound, unlocked at the real tip
        boardLocked = !atTip;
        if (atTip && appMode !== 'menu') {
          appMode = 'playing';
        }
      }

      updateActionButtonsEnabled();
    }

    // apply a replay snapshot and re-render
    function reviewApply(idx){
      resetVeryHardSession();
      restoreState(replay[idx]);
      selected.clear();
      anchorIdx = null; action = null; splitIdx = null;

      // while reviewing, clocks must reflect the present tip turn, not this snapshot
      markActiveFromTip();

      // during review/recite, apply the saved clock value for this frame if present
      try {
        if (appMode === 'review' && recite === true) {
          const snap = JSON.parse(replay[idx]);
          const ca = snap && snap.clockAt;
          if (ca && typeof ca === 'object') {
            if (ca.mode === 'stopwatch') {
              try {
                timerStop?.();
                if (typeof tStartPerf !== 'undefined') tStartPerf = 0;
                if (typeof tAccUs !== 'undefined') tAccUs = Math.max(0, Math.floor(ca.us || 0));
                renderTimer?.();
              } catch {}
            } else if (ca.mode === 'chess') {
              try {
                // Only set the sides present in the snapshot (AI games may have one)
                if (Object.prototype.hasOwnProperty.call(ca,'sunUs') && typeof chessSunUs !== 'undefined') {
                  chessSunUs = Math.max(0, Math.floor(ca.sunUs || 0));
                }
                if (Object.prototype.hasOwnProperty.call(ca,'moonUs') && typeof chessMoonUs !== 'undefined') {
                  chessMoonUs = Math.max(0, Math.floor(ca.moonUs || 0));
                }
                renderChessClocks?.();
                // Do not highlight any side while reviewing
                markActive?.(null);
              } catch {}
            }
            applyClockVisibility?.();
            updateClockFacesVisibility?.();
          }
        }
      } catch {}

      render();
      syncInteractionLock();
      updateActionButtonsEnabled();
      updateLogActive();

      highlightReplayFrame(idx);

      try {
        const atTipNow =
          (typeof isAtTip === 'function') ? isAtTip() : false;

        if (appMode === 'review' || atTipNow || forceReviewSFXOnce) {
          const allowSFX = (forceReviewSFXOnce || !suppressReviewSFX);
          const snap = JSON.parse(replay[idx]);
          if (allowSFX) {
            let sfxPlayed = false;
            if (snap && snap.isMove) {
              if (snap.tag === 'placeSwan' || snap.tag === 'placeStone') {
                playReady('place', { rate: 1.0 });
                sfxPlayed = true;
              } else if (snap.tag === 'moveSwans') {
                const count = Array.isArray(snap.movedFrom)
                  ? snap.movedFrom.length
                  : 1;
                playReady(
                  count > 1 ? 'moveMany' : 'move1',
                  { gain: 0.75, rate: count > 1 ? 0.97 : 1.00 }
                );
                sfxPlayed = true;
              } else if (snap.tag === 'pushSwans') {
                // mirror live play: use pushedFrom length to choose move1 vs moveMany
                const count = Array.isArray(snap.pushedFrom)
                  ? snap.pushedFrom.length
                  : 1;
                playReady(
                  count > 1 ? 'moveMany' : 'move1',
                  { gain: 0.75, rate: count > 1 ? 0.97 : 1.00 }
                );
                sfxPlayed = true;
              }
            }
            if (!sfxPlayed) {
              // fallback - older save frames may miss tags or isMove on setup placements
              // detect a single Swan added during setup by diffing with the previous frame
              try {
                if (idx > 0) {
                  const prev = JSON.parse(replay[idx - 1]);
                  const prevBoard = prev?.board, currBoard = snap?.board;
                  const N = Math.max(prevBoard?.length||0, currBoard?.length||0);
                  let added = 0, addedCode = 0;
                  for (let r = 0; r < N; r++) {
                    const pr = prevBoard?.[r] || [];
                    const cr = currBoard?.[r] || [];
                    const M = Math.max(pr.length, cr.length);
                    for (let c = 0; c < M; c++) {
                      const a = pr?.[c];
                      const b = cr?.[c];
                      if ((a === 0 || a === undefined || a === null) && (b === 1 || b === 2)) {
                        added++;
                        addedCode = b;
                        if (added > 1) break;
                      }
                    }
                    if (added > 1) break;
                  }
                  const wasSetup = (prev?.turn === 'setup') || (snap?.turn === 'setup');
                  if (wasSetup && added === 1 && (addedCode === 1 || addedCode === 2)) {
                    playReady('place', { rate: 1.0 });
                  }
                }
              } catch {}
            }
          }
          if (forceReviewSFXOnce) { forceReviewSFXOnce = false; }
        }
      } catch {}
    }

    // button muting logic for arrows
    function updateReviewButtons(){
      // enable navigation in any non-menu state
      const canNavNow = (appMode !== 'menu');
      if (!canNavNow){
        setMuted(btnBack, true);
        setMuted(btnForward, true);
        return;
      }
      // ensure replay exists and is aligned to current history+state
      if (!replay || !replay.length) rebuildReplay();

      setMuted(btnBack,    replayIdx <= 0);
      setMuted(btnForward, replayIdx >= (replay.length - 1));
    }

    // click handlers
    function reviewBack(fromAuto = false){
      if (appMode === 'menu') return;
      resetVeryHardSession();
      // ensure we are in review mode so SFX and UI behave consistently
      appMode = (appMode === 'menu') ? 'playing' : appMode;
      appMode = 'review';
      // any manual nav should pause autoplay (but not auto-advances)
      if (!fromAuto) stopReplayAuto?.();

      if (!replay || !replay.length) rebuildReplay();
      if (!replay || !replay.length) return;

      // If we're already at the very beginning, nothing to do
      if (replayIdx <= 0) return;

      const R = replay.length;

      // Walk backwards to the previous snapshot that represents a move
      let targetIdx = -1;
      for (let i = replayIdx - 1; i >= 0; i--){
        let snap;
        try {
          snap = JSON.parse(replay[i]);
        } catch(e) {
          continue;
        }
        if (snap && snap.isMove){
          targetIdx = i;
          break;
        }
      }

      // If there is no earlier move snapshot, go to the very first snapshot
      if (targetIdx < 0){
        targetIdx = 0;
      }

      // If we're already at that snapshot, nothing to do
      if (targetIdx === replayIdx) return;

      replayIdx = targetIdx;
      reviewApply(replayIdx);
      updateReviewButtons();
      updateActionButtonsEnabled();
    }

    function reviewForward(fromAuto = false){
      if (appMode === 'menu') return;

      // any manual nav should pause autoplay (but not auto-advances)
      if (!fromAuto) stopReplayAuto?.();

      if (!replay || !replay.length) rebuildReplay();
      if (!replay || !replay.length) return;

      const R = replay.length;

      // If we're already at the tip, nothing to do – and *don’t* flip modes
      if (replayIdx >= R - 1 && isAtTip()) {
        // if you have a central lock/mode helper, call it here so UI stays in sync:
        // syncInteractionLock?.();
        return;
      }

      // Look for the next move snapshot after the current index
      let targetIdx = -1;
      for (let i = replayIdx + 1; i < R; i++){
        let snap;
        try {
          snap = JSON.parse(replay[i]);
        } catch(e) {
          continue;
        }
        if (snap && snap.isMove){
          targetIdx = i;
          break;
        }
      }

      // If there is no later move snapshot, jump to the real tip
      if (targetIdx < 0){
        targetIdx = R - 1;
      }

      if (targetIdx === replayIdx) return;

      // at this point we know we’re actually moving → now go into review mode
      appMode = (appMode === 'menu') ? 'playing' : appMode;
      appMode = 'review';

      replayIdx = targetIdx;
      reviewApply(replayIdx);
      updateReviewButtons();
      updateActionButtonsEnabled();
    }

    /* -------------------- logging -------------------- */
    function log(msg){
      elLog.innerHTML += `<div>${msg}</div>`;
      elLog.scrollTop = elLog.scrollHeight;
      updateLogActive();
    }

    // Insert one or more hint messages near the currently selected move while in recital mode
    // Ensures any end-state (win/lose/draw) line remains the last entry in the log
    function insertHintLogsRecital(msgs){
      try {
        if (!elLog || !Array.isArray(msgs) || !msgs.length) return;

        // collect rows and helpers
        const rows = Array.from(elLog.children);
        const norm = (t)=> (t||'').replace(/\s+/g,' ').trim();

        // detect if the very last row is an end-state line (short form), e.g. "Sun wins.", "Moon wins.", "Draw."
        let outcomeIdx = -1;
        if (rows.length){
          const lastTxt = norm(rows[rows.length-1].textContent).toLowerCase();
          const known = (typeof lastOutcomeShort === 'string') ? lastOutcomeShort.toLowerCase() : null;
          if ((known && lastTxt === known) || /(wins\.|draw\.)$/.test(lastTxt)){
            outcomeIdx = rows.length - 1;
          }
        }

        // find the active move row (highlight added by updateLogActive)
        let activeRow = elLog.querySelector('.is-active');
        if (!activeRow){
          // fallback: pick the last row that looks like a move
          for (let i = rows.length - 1; i >= 0; i--){
            const t = norm(rows[i].textContent);
            if (isMoveRowText(t)) { activeRow = rows[i]; break; }
          }
        }

        // compute insertion reference node (insert before this node)
        // default to appending at end (or before outcome if present)
        let refNode = null; // null => append at end
        if (activeRow){
          // place immediately after the active move row, but before outcome if it exists
          const idx = rows.indexOf(activeRow);
          const afterIdx = (idx >= 0) ? (idx + 1) : rows.length;
          const capIdx = (outcomeIdx >= 0) ? outcomeIdx : rows.length;
          const insertIdx = Math.min(afterIdx, capIdx);
          refNode = (insertIdx < rows.length) ? rows[insertIdx] : null;
        } else if (outcomeIdx >= 0){
          // no active row found; insert before outcome line
          refNode = rows[outcomeIdx];
        }

        // create and insert nodes
        const frag = document.createDocumentFragment();
        for (const m of msgs){
          const d = document.createElement('div');
          d.innerHTML = m;
          frag.appendChild(d);
        }
        elLog.insertBefore(frag, refNode);

        // keep highlight correct
        updateLogActive();
      } catch (e) {
        // fallback: just append normally
        for (const m of msgs) log(m);
      }
    }

    // Build human-readable lines describing newly frozen Swan groups
    function composeFreezeNotes(encRes, actorName){
      try {
        const groups = Array.isArray(encRes?.frozenGroups) ? encRes.frozenGroups : [];
        if (!groups.length) return [];
        const nameOf = (owner)=> owner===SUN ? 'Sun' : 'Moon';
        const notes = [];
        for (const g of groups){
          const owner = g?.owner;
          const tiles = Array.isArray(g?.tiles) ? g.tiles.slice() : [];
          if (owner!==SUN && owner!==MOON) continue;
          if (!tiles.length) continue;
          // sort coords for stable output (by col then row, descending row for readability)
          tiles.sort((a,b)=> (a[1]-b[1]) || (b[0]-a[0]));
          const coords = tiles.map(([r,c])=> tileAlg(r,c)).join(', ');
          const ownerName = nameOf(owner);
          const isGroup = tiles.length > 1;
          const line = `${actorName} froze ${ownerName}'s Swan${isGroup?' group':''} at ${coords}.`;
          notes.push(line);
        }
        return notes;
      } catch { return []; }
    }

    function isMoveRowText(txt){
      // trim & normalize spaces
      txt = (txt || "").replace(/\s+/g, " ").trim();

      if (!txt) return false;
      if (/^\[sfx\]/i.test(txt)) return false;
      if (/^(new game|undo\.)$/i.test(txt)) return false;
      if (/^(hint|move blocked|moon cannot place|swan placement must|swan cannot be placed)/i.test(txt)) return false;
      if (/gains \+1 action/i.test(txt)) return false; // meta note, not the move itself

      // positive match for actual moves
      if (/^(sun|moon)\s+(placed|moved|pushed)\b/i.test(txt)) return true;
      if (/placed\s+(first|second)\s+swan/i.test(txt)) return true;

      return false;
    }

    function updateLogActive() {
      if (!elLog) return;

      const allRows = Array.from(elLog.children);
      if (!allRows.length) return;

      const moveRows = allRows.filter(r => isMoveRowText(r.textContent));

      // clear existing highlight
      allRows.forEach(r => r.classList.remove('is-active'));

      const L = moveRows.length;
      const R = replay ? replay.length : 0;

      // helper: pick the correct setup row to highlight based on where we are in replay
      function highlightSetupRow() {
        const norm = txt => (txt || "").replace(/\s+/g, " ").trim().toLowerCase();

        // helpers to find a row starting with a phrase
        function findRowStartsWith(prefix){
          const p = prefix.toLowerCase();
          for (let i = allRows.length - 1; i >= 0; i--){
            const t = norm(allRows[i].textContent);
            if (t.startsWith(p)) return allRows[i];
          }
          return null;
        }

        // Figure out our position in the timeline (if replay exists)
        const R = replay ? replay.length : 0;
        let snapIdx = -1;
        if (R > 0){
          snapIdx = isAtTip() ? (R - 1) : Math.min(Math.max(replayIdx, 0), R - 1);
        }

        // Find indices of the first two move snapshots (Sun first, Moon second)
        let firstMoveIdx = null, secondMoveIdx = null;
        if (R > 0){
          for (let i = 0; i < R; i++){
            let s; try { s = JSON.parse(replay[i]); } catch(e){ continue; }
            if (s && s.isMove){
              if (firstMoveIdx === null){ firstMoveIdx = i; }
              else if (secondMoveIdx === null){ secondMoveIdx = i; break; }
            }
          }
        }

        // Decide which setup row corresponds to this snapshot
        let active = null;
        if (snapIdx < 0 || firstMoveIdx === null || snapIdx < firstMoveIdx){
          // before any setup move → "New Game"
          active = findRowStartsWith('New Game');
        } else if (secondMoveIdx === null || snapIdx < secondMoveIdx){
          // after Sun’s first, before Moon’s second
          active = findRowStartsWith('Sun placed their first Swan');
          // fallback if that line was trimmed for some reason
          if (!active) active = findRowStartsWith('New Game');
        } else {
          // after Moon’s second
          active = findRowStartsWith('Moon placed their first Swan') ||
                   findRowStartsWith('Sun placed their first Swan') ||
                   findRowStartsWith('New Game');
        }

        if (!active) return;

        active.classList.add('is-active');

        // scroll into view
        function getScroller(node) {
          let n = node.parentElement;
          while (n) {
            const s  = getComputedStyle(n);
            const oy = s.overflowY;
            if ((oy === 'auto' || oy === 'scroll') && n.scrollHeight > n.clientHeight) {
              return n;
            }
            n = n.parentElement;
          }
          return elLog; // fallback
        }

        const padding  = 8;
        const scroller = getScroller(active);

        requestAnimationFrame(() => {
          const cRect = scroller.getBoundingClientRect();
          const rRect = active.getBoundingClientRect();

          const rowTop    = (rRect.top    - cRect.top) + scroller.scrollTop;
          const rowBottom = (rRect.bottom - cRect.top) + scroller.scrollTop;

          const viewTop = scroller.scrollTop;
          const viewBot = viewTop + scroller.clientHeight;

          if (rowTop < viewTop + padding) {
            scroller.scrollTop = Math.max(0, rowTop - padding);
          } else if (rowBottom > viewBot - padding) {
            scroller.scrollTop = rowBottom - scroller.clientHeight + padding;
          }
        });

        // Recital UX: when we've progressed to the real tip and play is still active,
        // push the log to the absolute bottom so the latest context is visible.
        try {
          // Also scroll when a playing game ends, so the final log lines are visible.
          if (typeof isAtTip === 'function' && isAtTip() && (recite === true || gameOver === true)) {
            requestAnimationFrame(() => {
              const s = getScroller(active);
              s.scrollTop = s.scrollHeight;
            });
          }
        } catch {}
      }

      // If we have no move-rows yet or no replay snapshots, just highlight setup.
      if (!L || !R){
        highlightSetupRow();
        return;
      }

      // 1) build a list of replay snapshots that actually represent moves
      const moveSnaps = [];
      for (let i = 0; i < R; i++) {
        let snap;
        try {
          snap = JSON.parse(replay[i]);
        } catch (e) {
          continue;
        }
        if (snap && snap.isMove) {
          const mNum = (typeof snap.moveNumber === 'number') ? snap.moveNumber : null;
          moveSnaps.push({ idx: i, moveNumber: mNum });
        }
      }

      if (!moveSnaps.length){
        highlightSetupRow();
        return;
      }

      // 2) work out which snapshot we're currently showing:
      //    - at tip  → last snapshot
      //    - not tip → current replayIdx
      let snapIdx;
      if (isAtTip()){
        snapIdx = R - 1;
      } else {
        snapIdx = Math.min(Math.max(replayIdx, 0), R - 1);
      }

      // 3) find the latest move snapshot at or before that snapshot
      let moveSnap = null;
      for (let i = moveSnaps.length - 1; i >= 0; i--) {
        if (moveSnaps[i].idx <= snapIdx) {
          moveSnap = moveSnaps[i];
          break;
        }
      }

      if (!moveSnap){
        // we're before the first move; stick with setup rows
        highlightSetupRow();
        return;
      }

      // 4) map that move snapshot onto a log row
      let rowIdx;
      if (typeof moveSnap.moveNumber === 'number') {
        // moveNumber is 1-based; moveRows array is 0-based
        rowIdx = moveSnap.moveNumber - 1;
      } else {
        // fallback: position in the moveSnaps array
        rowIdx = moveSnaps.findIndex(ms => ms === moveSnap);
      }

      if (rowIdx < 0) rowIdx = 0;
      if (rowIdx > L - 1) rowIdx = L - 1;

      const active = moveRows[rowIdx];
      if (!active) return;

      active.classList.add('is-active');

      // 5) keep the active row in view (same behaviour as before)
      function getScroller(node) {
        let n = node.parentElement;
        while (n) {
          const s  = getComputedStyle(n);
          const oy = s.overflowY;
          if ((oy === 'auto' || oy === 'scroll') && n.scrollHeight > n.clientHeight) {
            return n;
          }
          n = n.parentElement;
        }
        return elLog; // fallback
      }

      const padding  = 8;
      const scroller = getScroller(active);

      requestAnimationFrame(() => {
        const cRect = scroller.getBoundingClientRect();
        const rRect = active.getBoundingClientRect();

        const rowTop    = (rRect.top    - cRect.top) + scroller.scrollTop;
        const rowBottom = (rRect.bottom - cRect.top) + scroller.scrollTop;

        const viewTop = scroller.scrollTop;
        const viewBot = viewTop + scroller.clientHeight;

        if (rowTop < viewTop + padding) {
          scroller.scrollTop = Math.max(0, rowTop - padding);
        } else if (rowBottom > viewBot - padding) {
          scroller.scrollTop = rowBottom - scroller.clientHeight + padding;
        }
      });

      // Recital UX: when we've advanced to the real tip and play remains active,
      // auto-scroll the log all the way to the bottom.
      try {
        // Also scroll when a playing game ends, so the final log lines are visible.
        if (typeof isAtTip === 'function' && isAtTip() && (recite === true || gameOver === true)) {
          requestAnimationFrame(() => {
            const s = getScroller(active);
            s.scrollTop = s.scrollHeight;
          });
        }
      } catch {}
    }

    /* -------------------- small helpers -------------------- */
    function idx(r,c){ return r*SIZE+c; }                                      // 2d -> 1d
    function inb(r,c){ return r>=0 && c>=0 && r<SIZE && c<SIZE; }              // in-bounds
    function cell(r,c){ return board[r][c]; }                                   // read cell
    function setcell(r,c,v){ board[r][c]=v; }                                   // write cell
    function isEmpty(r,c){ return cell(r,c)===EMPTY; }                          // empty?
    function isSwan(v){ return v===SWAN_SUN||v===SWAN_MOON||v===FROZEN_SUN||v===FROZEN_MOON; }
    function isActiveSwan(v){ return v===SWAN_SUN||v===SWAN_MOON; }
    function playerOfSwan(v){ return (v===SWAN_SUN||v===FROZEN_SUN)?SUN:MOON; } // owner
    function neighbours8(r,c){ return DIRS8.map(([dr,dc])=>[r+dr,c+dc]).filter(([nr,nc])=>inb(nr,nc)); }
    function neighbours4(r,c){ return DIRS4.map(([dr,dc])=>[r+dr,c+dc]).filter(([nr,nc])=>inb(nr,nc)); }
    const FILES = 'ABCDEFGHIJ';                                                 // file labels
    function tileAlg(r, c) { return `${FILES[c]}${r+1}`; }                      // A1..J10
    function disableBoardInteraction() {boardLocked = true;}
    function enableBoardInteraction() {boardLocked = false;}

    /* -------------------- hint state -------------------- */
    let hintBestCells = new Set();   // indices (r*size+c) to outline
    let hintWorstCells = new Set();  // indices to outline
    let hintTimer = null;

    /* small helpers used by the analyser */
    const deepClone = (b) => {
      const nb = new Array(SIZE);
      for (let r = 0; r < SIZE; r++) {
        const row = b[r];
        const copy = new Array(SIZE);
        for (let c = 0; c < SIZE; c++) copy[c] = row[c];
        nb[r] = copy;
      }
      return nb;
    };

    function countActiveSwansOn(b,p){
      let n=0; for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
        const v=b[r][c]; if((p===SUN && v===SWAN_SUN)||(p===MOON && v===SWAN_MOON)) n++;
      } return n;
    }

    function silversForPlayerOn(p,b){
      const DIRS8L=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
      const seen=new Set();
      for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
        const v=b[r][c];
        if(!((p===SUN && v===SWAN_SUN)||(p===MOON && v===SWAN_MOON))) continue;
        for(const [dr,dc] of DIRS8L){
          const nr=r+dr, nc=c+dc;
          if(nr<0||nc<0||nr>=SIZE||nc>=SIZE) continue;
          if(b[nr][nc]===EMPTY) seen.add(nr*SIZE+nc);
        }
      }
      return seen.size;
    }

    /* ===============================================================
       shared freeze detector
       - pure (no DOM, no logs)
       - returns {nb, frozeSun, frozeMoon, sealedSun, sealedMoon, frozenGroups}
       =============================================================== */
    function simulateAndFreeze(b){
      return computeFreezesOn(b);
    }

    /* list all legal actions with a score and explanation */
    function analyzeCandidates(b, player){
      const OPP = (player===SUN)?MOON:SUN;
      const cand = [];

      const isEmptyAt=(r,c)=> b[r][c]===EMPTY;
      const clone=()=>deepClone(b);

      // legal stone placements (same heuristic as ai)
      const frontier=[], all=[];
      for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
        if(!isEmptyAt(r,c)) continue;
        all.push([r,c]);
        for(const [dr,dc] of DIRS8){
          const nr=r+dr, nc=c+dc;
          if(nr<0||nc<0||nr>=SIZE||nc>=SIZE) continue;
          if(b[nr][nc]!==EMPTY){ frontier.push([r,c]); break; }
        }
      }
      const stonePool = frontier.length? frontier : all;

      function pushScored(type, payload, b2, changedCells){
        const myLib0 = silversForPlayerOn(player, b);
        const opLib0 = silversForPlayerOn(OPP,    b);
        const { nb, frozeSun, frozeMoon, sealedSun, sealedMoon } = simulateAndFreeze(b2);
        const myLib1 = silversForPlayerOn(player, nb);
        const opLib1 = silversForPlayerOn(OPP,    nb);

        const myΔ = myLib1 - myLib0;
        const opΔ = opLib1 - opLib0;

        const frozeEnemy   = (player===SUN)? (frozeMoon+sealedMoon) : (frozeSun+sealedSun);
        const frozeSelf    = (player===SUN)? (frozeSun +sealedSun ) : (frozeMoon+sealedMoon);
        const sealedEnemy  = (player===SUN)?  sealedMoon           :  sealedSun;
        const sealedSelf   = (player===SUN)?  sealedSun            :  sealedMoon;

        // base heuristic
        let score = (frozeEnemy*500) + (frozeSelf*-600) + (myΔ*5) + (opΔ*-9);

        // --- hard overrides for terminal outcomes in the sim board ---
        // WIN now
        if (sealedEnemy > 0 && sealedSelf === 0) score = 1e9;
        // LOSS now
        if (sealedSelf  > 0 && sealedEnemy === 0) score = -1e9;
        // DRAW now (both sides sealed)
        if (sealedSelf  > 0 && sealedEnemy  > 0) score = 1e6;

        const why = [];
        if (sealedEnemy > 0 && sealedSelf === 0) {
          why.push("wins the game");
        } else if (sealedSelf > 0 && sealedEnemy === 0) {
          why.push("loses the game");
        } else if (sealedSelf > 0 && sealedEnemy > 0) {
          why.push("draws the game");
        } else {
          if (frozeEnemy > 0) why.push(`freezes ${frozeEnemy} enemy swan${frozeEnemy>1?'s':''}`);
          if (frozeSelf  > 0) why.push(`risks freezing ${frozeSelf} of yours`);
          if (opΔ < 0)       why.push(`chokes ${-opΔ} enemy silvers`);
          if (myΔ > 0)       why.push(`improves your silvers by ${myΔ}`);
          if (!why.length)   why.push(`neutral positioning`);
          // If overall score is negative (and no terminal outcomes), mark that this worsens position
          if (score < 0)     why.push(`overall worsens your position`);
        }

        cand.push({ type, payload, score, why: why.join('; '), changedCells });
      }

      // stones
      for(const [r,c] of stonePool){
        const b2=clone(); b2[r][c]=STONE;
        pushScored('stone', {r,c}, b2, [[r,c]]);
      }

      // swan placements (under 6 total)
      const totalMine = (()=>{let n=0; for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){
        const v=b[r][c]; if(player===SUN?(v===SWAN_SUN||v===FROZEN_SUN):(v===SWAN_MOON||v===FROZEN_MOON)) n++;
      } return n;})();
      function isMine(v){ return player===SUN?(v===SWAN_SUN||v===FROZEN_SUN):(v===SWAN_MOON||v===FROZEN_MOON); }
      function isEnemy(v){ return player===SUN?(v===SWAN_MOON||v===FROZEN_MOON):(v===SWAN_SUN||v===FROZEN_SUN); }
      if(totalMine<6){
        for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
          if(!isEmptyAt(r,c)) continue;
          let adjMine=false, adjEnemy=false;
          for(const [dr,dc] of DIRS4){ const nr=r+dr,nc=c+dc; if(nr<0||nc<0||nr>=SIZE||nc>=SIZE) continue;
            const v=b[nr][nc]; if(isMine(v)) adjMine=true; }
          for(const [dr,dc] of DIRS8){ const nr=r+dr,nc=c+dc; if(nr<0||nc<0||nr>=SIZE||nc>=SIZE) continue;
            const v=b[nr][nc]; if(isEnemy(v)) adjEnemy=true; }
          if(adjMine && !adjEnemy){
            const b2=clone(); b2[r][c]=(player===SUN?SWAN_SUN:SWAN_MOON);
            pushScored('swan', {r,c}, b2, [[r,c]]);
          }
        }
      }

      // moves - try all non-empty subsets by 8 dirs (keep to small subsets for speed)
      const coords=[]; for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
        const v=b[r][c]; if((player===SUN && v===SWAN_SUN)||(player===MOON && v===SWAN_MOON)) coords.push([r,c]);
      }
      // generate all swan subsets up to 6 swans
      function* kCombinations(list, k, start=0, prefix=[]){
        if (prefix.length === k){ yield prefix.slice(); return; }
        for (let i = start; i <= list.length - (k - prefix.length); i++){
          prefix.push(list[i]);
          yield* kCombinations(list, k, i + 1, prefix);
          prefix.pop();
        }
      }

      const subsets = [];
      const maxK = Math.min(6, coords.length);
      for (let k = 1; k <= maxK; k++){
        for (const combo of kCombinations(coords, k)){
          subsets.push(combo);
        }
      }

      function legalMoveSubsetLocal(subset, dir){
        const [dr,dc]=dir;
        const inb=(r,c)=>r>=0&&c>=0&&r<SIZE&&c<SIZE;
        const isSwan=v=> v===SWAN_SUN||v===SWAN_MOON||v===FROZEN_SUN||v===FROZEN_MOON;
        const activeSwanOf=(p,v)=> (p===SUN && v===SWAN_SUN) || (p===MOON && v===SWAN_MOON);
        const samePlayerSwan=(v,p)=> (p===SUN && (v===SWAN_SUN||v===FROZEN_SUN)) || (p===MOON && (v===SWAN_MOON||v===FROZEN_MOON));
        const enemySwan=(v,p)=> (p===SUN && (v===SWAN_MOON||v===FROZEN_MOON)) || (p===MOON && (v===SWAN_SUN||v===FROZEN_SUN));
        const stoneKey=(r,c)=>`s:${r},${c}`;

        const moving = new Set(subset.map(([r,c])=>r*SIZE+c));
        const stonesFrom = new Set();
        const stonesTo   = new Map();

        // collect stones that would follow (not shared)
        for(const [r,c] of subset){
          for(const [er,ec] of DIRS8){
            const sr=r+er, sc=c+ec;
            if(!inb(sr,sc) || b[sr][sc]!==STONE) continue;

            let shared=false;
            for(const [ar,ac] of DIRS8){
              const xr=sr+ar, xc=sc+ac;
              if(!inb(xr,xc)) continue;
              const vv=b[xr][xc];
              if(!isSwan(vv)) continue;
              if(enemySwan(vv,player)) { shared=true; break; }
              if(samePlayerSwan(vv,player) && activeSwanOf(player,vv) && !moving.has(xr*SIZE+xc)){ shared=true; break; }
            }
            if(shared) continue;

            const tr=sr+dr, tc=sc+dc;
            if(!inb(tr,tc)) return null;
            const sk=stoneKey(sr,sc);
            stonesFrom.add(sk);
            stonesTo.set(sk,[tr,tc]);
          }
        }

        const isVacantAfterMove=(r,c)=>{
          if(!inb(r,c)) return false;
          if(b[r][c]===EMPTY) return true;
          if(moving.has(r*SIZE+c)) return true;
          if(stonesFrom.has(stoneKey(r,c))) return true;
          return false;
        };

        // --- NEW HELPERS: naked enemy Swan & its 8-tile zone ---

        // enemy Swan with no Stones in any of the 8 surrounding tiles
        function isEnemySwanNakedLocal(sr, sc){
          const v = b[sr][sc];
          if (!enemySwan(v, player)) return false;

          // any Stone in the 8 neighbours means it is NOT naked
          for (const [dr8, dc8] of DIRS8){
            const nr = sr + dr8, nc = sc + dc8;
            if (!inb(nr, nc)) continue;
            if (b[nr][nc] === STONE) return false;
          }
          // no Stones adjacent in any of the 8 directions
          return true;
        }

        // is (r,c) inside the 8-neighbourhood of a naked enemy Swan?
        function isInNakedEnemyZoneLocal(r, c){
          for (const [dr8, dc8] of DIRS8){
            const er = r + dr8, ec = c + dc8;
            if (!inb(er, ec)) continue;
            if (isEnemySwanNakedLocal(er, ec)) return true;
          }
          return false;
        }

        // validate swan targets
        for(const [r,c] of subset){
          const nr=r+dr, nc=c+dc;
          if(!inb(nr,nc)) return null;

          // NEW: cannot move into the 8 tiles around a naked enemy Swan
          if (isInNakedEnemyZoneLocal(nr, nc)) return null;

          const occ=b[nr][nc];
          if(occ===EMPTY) continue;
          if(isSwan(occ)){
            if(!moving.has(nr*SIZE+nc)) return null;
          }else if(occ===STONE){
            const sk=stoneKey(nr,nc);
            if(!stonesTo.has(sk)) return null;
            const [tr,tc]=stonesTo.get(sk);
            if(!isVacantAfterMove(tr,tc)) return null;
          }else return null;
        }

        // validate stone targets (bounds, vacancy, no collisions)
        const seen=new Set();
        for(const [_,[tr,tc]] of stonesTo){
          if(!isVacantAfterMove(tr,tc)) return null;
          const k=`${tr},${tc}`; if(seen.has(k)) return null; seen.add(k);
        }
        return {stonesFrom, stonesTo};
      }

      function simulateMoveSubsetLocal(subset, dir){
        const [dr,dc]=dir;

        // pure legality check for this subset+dir against local board 'b'
        const legal = legalMoveSubsetLocal(subset, [dr,dc]);
        if(!legal) return null;

        const {stonesFrom, stonesTo} = legal;
        const moving = new Set(subset.map(([r,c])=>r*SIZE+c));
        const nb = deepClone(b);

        // clear swans and stones
        for(const [r,c] of subset) nb[r][c]=EMPTY;
        for(const sk of Array.from(stonesFrom)){
          const [sr,sc]=sk.slice(2).split(',').map(Number);
          nb[sr][sc]=EMPTY;
        }

        // place stones
        for(const [_,[tr,tc]] of stonesTo) nb[tr][tc]=STONE;

        // place swans
        for(const [r,c] of subset) nb[r+dr][c+dc] = (player===SUN?SWAN_SUN:SWAN_MOON);

        const changed = subset.map(([r,c])=>[r+dr,c+dc]).concat(Array.from(stonesTo.values()));
        return { nb, changed };
      }

      const DIRS8L=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
      for(const subset of subsets){
        for(const dir of DIRS8L){
          const sim = simulateMoveSubsetLocal(subset, dir);
          if(!sim) continue;
          pushScored('move', {swans: subset, dir}, sim.nb, sim.changed);
        }
      }

      cand.sort((a,b)=> b.score - a.score);
      return cand;
    }

    /* show best & worst and outline them */
    function showHint(){

      // cancel any transient UI so hint overlays are unambiguous
      action = null;
      selected.clear();
      anchorIdx = null;
      splitIdx = null;

      // compute candidates for the current player
      const cands = analyzeCandidates(board, current);
      if(!cands.length){ log('No legal actions to hint.'); return; }

      const best = cands[0];
      const worst = cands[cands.length-1];

      // mark overlays
      function hintCellsFor(a){
        if (a.type === 'move') {
          // only the swan destinations (first N entries)
          return a.changedCells.slice(0, a.payload.swans.length);
        }
        if (a.type === 'push') {
          // pushed swans' destinations
          return a.changedCells.slice(0, a.payload.swans.length);
        }
        if (a.type === 'swan')  return [[a.payload.r, a.payload.c]];
        if (a.type === 'stone') return [[a.payload.r, a.payload.c]];
        return [];
      }

      hintBestCells.clear(); hintWorstCells.clear();
      for (const [r,c] of hintCellsFor(best))  hintBestCells.add(r*SIZE + c);
      for (const [r,c] of hintCellsFor(worst)) hintWorstCells.add(r*SIZE + c);

      // explain moves
      function fmtAct(a){
        if(a.type==='stone') return `Place Stone at ${tileAlg(a.payload.r, a.payload.c)}`;
        if(a.type==='swan')  return `Place Swan at ${tileAlg(a.payload.r, a.payload.c)}`;
        if(a.type==='move'){
          const [dr,dc]=a.payload.dir;
          const from = a.payload.swans.map(([r,c])=>tileAlg(r,c)).join(', ');
          const to   = a.changedCells.slice(0, a.payload.swans.length).map(([r,c])=>tileAlg(r,c)).join(', ');
          return `Move ${a.payload.swans.length>1?'Swans':'Swan'} ${from} → ${to}`;
        }
        if(a.type==='push'){
          const [dr,dc]=a.payload.dir;
          const from = a.payload.swans.map(([r,c])=>tileAlg(r,c)).join(', ');
          const to   = a.changedCells.slice(0, a.payload.swans.length).map(([r,c])=>tileAlg(r,c)).join(', ');
          return `Push ${a.payload.swans.length>1?'Swans':'Swan'} ${from} → ${to}`;
        }
        return 'Action';
      }

      if (recite === true){
        insertHintLogsRecital([
          `<b>Best</b>: ${fmtAct(best)} — ${best.why}`,
          `<b>Worst</b>: ${fmtAct(worst)} — ${worst.why}`,
        ]);
      } else {
        log(`<b>Best</b>: ${fmtAct(best)} — ${best.why}`);
        log(`<b>Worst</b>: ${fmtAct(worst)} — ${worst.why}`);
      }

      // paint and auto-clear highlights after a short delay
      render();
      clearTimeout(hintTimer);
      hintTimer = setTimeout(()=>{ hintBestCells.clear(); hintWorstCells.clear(); render(); }, 4500);
    }

    /* ===============================================================
       ai (pure action selector) + ai execution glue
       - linithAI(board, current) returns best action
       - performAiAction applies that action to current ui/functions
       =============================================================== */

    // apply ai action using the existing play functions
    function performAiAction(act){
      // hard safety - if the caller ever passes null, re-ask ai in 'hard' mode
      if (!act) {
        const retry = linithAI(board, current, 'hard');
        if (!retry) return;           // leaves board interactive only if truly no legal move exists
        act = retry;
      }
      if(act.type === 'stone'){ return doPlaceStone(act.r, act.c); }
      if(act.type === 'swan'){  return doPlaceSwan(act.r, act.c); }
      if(act.type === 'move'){
        selected.clear();
        for (const {r,c} of act.swans) selected.add(r*SIZE + c);
        action = 'moveSwans';
        const [dr, dc] = act.dir;
        return tryMoveSelected(`${dr},${dc}`);
      }
      if(act.type === 'push'){
        // select enemy swans to be pushed, then call the existing push executor
        selected.clear();
        for (const {r,c} of act.swans) selected.add(r*SIZE + c);
        action = 'pushSwans';
        const [dr, dc] = act.dir;
        return tryPushSelected(`${dr},${dc}`);
      }
    }

    // remove the latest move line from the on-screen log (keeps other notes)
    function removeLastMoveRowFromLog(){
      if (!elLog) return;
      const rows = Array.from(elLog.children);
      for (let i = rows.length - 1; i >= 0; i--) {
        const row = rows[i];
        const txt = row?.textContent || '';
        if (isMoveRowText(txt)) { row.remove(); break; }
      }
    }

    function veryHardTiming() {
      const platform = detectVeryHardPlatform(
        navigator.userAgent || '',
        Boolean(window.linithDesktop)
      );
      return chooseVeryHardTimeBudget({ board, current, movesLeft }, platform);
    }

    // Exact deterministic identity for the position handed to a worker. The
    // board is only 100 cells, so retaining every tile avoids accepting the
    // wrong position through a lossy hash collision.
    function positionFingerprint(b = board, player = current, actionsLeft = movesLeft) {
      return `${player}:${actionsLeft}:${b.map((row) => row.join('')).join('/')}`;
    }

    function isCurrentVeryHardTurn(fingerprint, sessionId = veryHardSessionSequence) {
      return sessionId === veryHardSessionSequence &&
        appMode === 'playing' &&
        !gameOver &&
        turn === 'play' &&
        aiSide !== null &&
        aiSide === current &&
        (window.linithGetDifficulty?.() || aiDifficulty) === 'very_hard' &&
        positionFingerprint() === fingerprint;
    }

    function unlockAfterAiThinking() {
      if (appMode === 'playing' && turn === 'play' && isAtTip()) {
        enableBoardInteraction();
      }
    }

    function terminateVeryHardWorker(worker = veryHardWorker) {
      if (!worker) return;
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
      if (veryHardWorker === worker) veryHardWorker = null;
    }

    function disposeVeryHardRequest(request, unlock = true, terminateWorker = false) {
      if (!request || activeVeryHardRequest !== request) return false;
      if (request.timeout !== null) clearTimeout(request.timeout);
      request.worker.onmessage = null;
      request.worker.onerror = null;
      if (terminateWorker) terminateVeryHardWorker(request.worker);
      activeVeryHardRequest = null;
      aiThinking = false;
      if (unlock) unlockAfterAiThinking();
      return true;
    }

    function cancelVeryHardSearch() {
      const wasThinking = aiThinking;
      const request = activeVeryHardRequest;
      if (request) {
        // A worker cannot process a cancellation message while synchronous
        // search code is running, so cancellation deliberately discards it.
        // Successful searches keep the worker and its bounded caches alive.
        disposeVeryHardRequest(request, true, true);
      } else if (wasThinking) {
        aiThinking = false;
        unlockAfterAiThinking();
      }
      return wasThinking || request !== null;
    }

    function performHardFallback(fingerprint, sessionId = veryHardSessionSequence) {
      if (!isCurrentVeryHardTurn(fingerprint, sessionId)) return null;
      selected.clear();
      action = null;
      anchorIdx = null;
      const fallback = linithAI(board, current, 'hard');
      if (!fallback) return null;
      const beforeAction = positionFingerprint();
      performAiAction(fallback);
      return positionFingerprint() === beforeAction ? null : fallback;
    }

    function finishVeryHardFailure(request, reason) {
      if (!disposeVeryHardRequest(request, true, true)) return;
      if (reason) console.warn(`Very Hard AI unavailable; using Hard for this action: ${reason}`);
      render();
      performHardFallback(request.fingerprint, request.sessionId);
    }

    function isWorkerAction(actionValue) {
      if (!actionValue || typeof actionValue !== 'object') return false;
      if (actionValue.type === 'stone' || actionValue.type === 'swan') {
        return Number.isInteger(actionValue.r) && Number.isInteger(actionValue.c);
      }
      if (actionValue.type === 'move' || actionValue.type === 'push') {
        return Array.isArray(actionValue.dir) && actionValue.dir.length === 2 &&
          actionValue.dir.every(Number.isInteger) &&
          Array.isArray(actionValue.swans) && actionValue.swans.length > 0 &&
          actionValue.swans.every(({r, c}) => Number.isInteger(r) && Number.isInteger(c));
      }
      return false;
    }

    function acknowledgeVeryHardAction(request, actualAction) {
      if (!request || !actualAction ||
          request.sessionId !== veryHardSessionSequence ||
          request.worker !== veryHardWorker) return;
      try {
        request.worker.postMessage({
          type: 'commit',
          requestId: request.requestId,
          sessionId: request.sessionId,
          fingerprint: request.fingerprint,
          preActionState: request.state,
          action: actualAction
        });
      } catch {
        // A worker that missed an accepted action must not retain incomplete
        // history or a continuation derived from a different line.
        terminateVeryHardWorker(request.worker);
      }
    }

    function startVeryHardSearch() {
      const fingerprint = positionFingerprint();
      const style = window.linithGetStyle?.() || aiStyle || 'doctrinal';
      const timing = veryHardTiming();
      const budgetMs = timing.budgetMs;
      const requestId = ++veryHardRequestSequence;
      const sessionId = veryHardSessionSequence;
      const state = {
        board: board.map((row) => row.slice()),
        current,
        movesLeft
      };

      aiThinking = true;
      disableBoardInteraction();
      render();

      let worker = veryHardWorker;
      if (!worker) {
        try {
          worker = new VeryHardWorker();
          veryHardWorker = worker;
        } catch (error) {
          aiThinking = false;
          unlockAfterAiThinking();
          render();
          performHardFallback(fingerprint);
          return;
        }
      }

      const request = {
        requestId,
        sessionId,
        fingerprint,
        state,
        worker,
        timeout: null
      };
      activeVeryHardRequest = request;

      worker.onmessage = (event) => {
        if (activeVeryHardRequest !== request) return;
        const message = event.data || {};
        const identityMatches = message.requestId === requestId &&
          message.sessionId === sessionId &&
          sessionId === veryHardSessionSequence &&
          message.fingerprint === fingerprint;
        const positionMatches = isCurrentVeryHardTurn(fingerprint, sessionId);

        // Never let a superseded or position-mismatched response influence the
        // live match. Destroy that worker and schedule a clean request against
        // the current state instead of turning stale traffic into a fallback
        // move on a possibly new game.
        if (!identityMatches || !positionMatches) {
          disposeVeryHardRequest(request, true, true);
          render();
          if (appMode === 'playing' && turn === 'play' && aiSide === current) {
            aiturn();
          }
          return;
        }

        if (message.type === 'error') {
          finishVeryHardFailure(request, message.message || 'worker search failed');
          return;
        }

        const resultAction = message.action;
        if (message.type !== 'result' || !isWorkerAction(resultAction)) {
          finishVeryHardFailure(request, 'worker returned no valid action');
          return;
        }

        disposeVeryHardRequest(request, true);
        render();
        const beforeAction = positionFingerprint();
        performAiAction(resultAction);
        if (positionFingerprint() === beforeAction) {
          const actualFallback = performHardFallback(beforeAction, sessionId);
          if (actualFallback) acknowledgeVeryHardAction(request, actualFallback);
        } else {
          acknowledgeVeryHardAction(request, resultAction);
        }
      };

      worker.onerror = (event) => {
        event.preventDefault?.();
        finishVeryHardFailure(request, event.message || 'worker crashed');
      };

      // The search itself observes budgetMs. This watchdog only handles a
      // wedged worker or a response lost during platform shutdown.
      request.timeout = setTimeout(() => {
        finishVeryHardFailure(request, 'worker exceeded its response deadline');
      }, timing.hardLimitMs + 100);

      try {
        worker.postMessage({
          type: 'search',
          requestId,
          sessionId,
          fingerprint,
          state,
          style,
          budgetMs
        });
      } catch (error) {
        finishVeryHardFailure(request, error instanceof Error ? error.message : 'worker request failed');
      }
    }

    function cancelAiMove() {
      let cancelled = false;
      if (aiMoveTimer !== null) {
        clearTimeout(aiMoveTimer);
        aiMoveTimer = null;
        cancelled = true;
      }
      return cancelVeryHardSearch() || cancelled;
    }

    // A worker's bounded caches are valuable within one live match, but must
    // never cross a match/timeline/configuration boundary. Incrementing the
    // session before any later request also makes a delayed old response
    // ineligible even if its board fingerprint happens to recur.
    function resetVeryHardSession() {
      const wasThinking = cancelAiMove();
      terminateVeryHardWorker();
      veryHardSessionSequence++;
      return wasThinking;
    }

    function restartAiAfterConfigurationChange() {
      // Evaluation style is part of cached search values. Drop an idle worker
      // as well as an active one whenever difficulty or style changes.
      const wasThinking = resetVeryHardSession();
      const shouldRestart = appMode === 'playing' &&
        turn === 'play' &&
        aiSide !== null &&
        aiSide === current;
      if (wasThinking || shouldRestart) render();
      if (shouldRestart) aiturn();
    }

    function scheduleAiMove(delayMs = 120) {
      // avoid double-scheduling
      if (aiMoveTimer !== null || activeVeryHardRequest !== null) return;

      // basic guard: only in a real AI turn
      if (!aiSide) return;
      if (appMode !== 'playing') return;
      if (turn !== 'play') return;
      if (aiSide !== current) return;

      const scheduledDifficulty = (window.linithGetDifficulty?.() || aiDifficulty);
      const scheduledSessionId = veryHardSessionSequence;
      if (scheduledDifficulty === 'very_hard') {
        aiThinking = true;
        disableBoardInteraction();
        render();
      }

      aiMoveTimer = setTimeout(() => {
        aiMoveTimer = null;

        // give the UI one paint frame to settle, *then* re-check
        requestAnimationFrame(() => {
          // The timeout may have fired just before a reset/configuration
          // change, leaving this animation-frame callback beyond the reach of
          // clearTimeout. A session mismatch makes that callback inert.
          if (scheduledSessionId !== veryHardSessionSequence) return;
          if (appMode !== 'playing' || turn !== 'play' || !aiSide || aiSide !== current) {
            if (aiThinking) {
              cancelVeryHardSearch();
              render();
            }
            return;
          }

          const difficulty = (window.linithGetDifficulty?.() || aiDifficulty);
          if (aiThinking && difficulty !== 'very_hard') {
            aiThinking = false;
            unlockAfterAiThinking();
            render();
          }
          if (difficulty === 'very_hard') {
            startVeryHardSearch();
            return;
          }
          const act = linithAI(board, current, difficulty);
          performAiAction(act);
        });
      }, delayMs);
    }

    function aiturn() {
      // keep the original "not playing" safety
      if (appMode !== 'playing') {
        activeSide = null;
        markActive(null);
        cancelAiMove();
        return;
      }
      // delegate to the scheduler – it will handle all the checks
      scheduleAiMove();
    }

    /* ===============================================================
       setup helpers (initial swans)
       =============================================================== */

    function firstEmptyNearCenter(){
      const centers = [[5,5],[4,5],[5,4],[4,4],[5,6],[6,5],[4,6],[6,4]];
      for (const [r,c] of centers) if (isEmpty(r,c)) return [r,c];
      for (let r=0;r<SIZE;r++) for (let c=0;c<SIZE;c++) if (isEmpty(r,c)) return [r,c];
      return null;
    }

    function findSunInitial(){
      for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++) if (cell(r,c)===SWAN_SUN) return [r,c];
      return null;
    }

    function nonAdjacentEmptiesTo(r0,c0){
      const out=[]; for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){
        if (!isEmpty(r,c)) continue;
        let adj=false; for (const [nr,nc] of neighbours8(r0,c0)){ if (nr===r && nc===c){ adj=true; break; } }
        if (!adj) out.push([r,c]);
      }
      return out;
    }

    function aisetup(){
      if (turn !== 'setup') return;

      // ai as sun, place first near center
      if (toPlace === 'sun' && aiSide === SUN){
        const pos = firstEmptyNearCenter();
        if (!pos) return;
        const [r,c] = pos;
        setcell(r,c,SWAN_SUN);
        // highlight AI placement (only in vs-AI)
        flashRecentIfAIOnly([], [idx(r,c)], 800, 'ai');
        playReady('place', { rate: 1.0 });
        const msgSun = `Sun placed their first Swan at ${tileAlg(r,c)}.`;
        log(msgSun);
        toPlace = 'moon';
        // label setup placement so review-mode SFX can trigger
        pushHistory({
          tag: 'placeSwan',
          actor: 'sun',
          tile: tileAlg(r,c),
          isMove: true,
          log: msgSun
        });
        render();
      }

      // ai as moon, place not adjacent to sun's initial, then start play
      if (toPlace === 'moon' && aiSide === MOON){
        const sun0 = findSunInitial(); if (!sun0) return;
        const choices = nonAdjacentEmptiesTo(sun0[0], sun0[1]); if (!choices.length) return;

        // score candidates (silvers, center preference, avoid edge)
        function scoreMoonSetup(r, c){
          let libs = 0; for (const [nr, nc] of neighbours8(r, c)) if (isEmpty(nr, nc)) libs++;
          const cr = (SIZE - 1) / 2, cc = (SIZE - 1) / 2;
          const dist2Center = (r - cr)*(r - cr) + (c - cc)*(c - cc);
          const edgeMin = Math.min(r, c, SIZE-1-r, SIZE-1-c);
          const edgePenalty = edgeMin === 0 ? 2 : (edgeMin === 1 ? 1 : 0);
          return libs*10 - dist2Center*0.1 - edgePenalty + Math.random()*0.01;
        }

        let best = choices[0], bestScore = -Infinity;
        for (const [r, c] of choices){ const s = scoreMoonSetup(r, c); if (s > bestScore){ bestScore = s; best = [r, c]; } }

        const [r, c] = best;
        setcell(r, c, SWAN_MOON);
        // highlight AI placement (only in vs-AI)
        flashRecentIfAIOnly([], [idx(r,c)], 800, 'ai');
        playReady('place', { rate: 1.0 });
        const msgMoon = `Moon placed their first Swan at ${tileAlg(r,c)}. Moon begins.`;
        log(msgMoon);
        turn = 'play';
        current = MOON;
        beginTurn();
        // label setup placement so review-mode SFX can trigger
        pushHistory({
          tag: 'placeSwan',
          actor: 'moon',
          tile: tileAlg(r,c),
          isMove: true,
          log: msgMoon
        });
        timerStart();
        render()
        aiturn();
      }
    }

    /* ===============================================================
       rendering (board, glyphs, overlays, labels, turn info)
       =============================================================== */
    function renderAxisLabels() {
      const y = document.getElementById('yLabels');
      const x = document.getElementById('xLabels');
      if (!y || !x) return;

      y.innerHTML = ''; for (let i = 1; i <= SIZE; i++){ const div = document.createElement('div'); div.textContent = i; y.appendChild(div); }
      x.innerHTML = ''; for (let i = 0; i < SIZE; i++){ const div = document.createElement('div'); div.textContent = FILES[i]; x.appendChild(div); }
    }

    function render(){
      // board tiles
      elBoard.innerHTML = '';
      for(let r=0;r<SIZE;r++){
        for(let c=0;c<SIZE;c++){
          const v = cell(r,c);

          // shell
          const d = document.createElement('div');  d.className = 'cell';
          const inner = document.createElement('div'); inner.className = 'inner';

          // cell background classes
          if(v===SWAN_SUN||v===FROZEN_SUN) inner.classList.add('swan');
          if(v===SWAN_MOON||v===FROZEN_MOON) inner.classList.add('swan');
          if(v===STONE) inner.classList.add('stone');
          if(v===FROZEN_SUN||v===FROZEN_MOON) inner.classList.add('frozen');

          // glyph host
          const glyph = document.createElement('div'); glyph.className = 'glyph';

          // split chooser ui (only on empty cell that is toggled)
          if (v===EMPTY && splitIdx === idx(r,c)) {
            const chooser = document.createElement('div'); chooser.className = 'chooser';
            const left = document.createElement('div');  left.className = 'half left';
            const right = document.createElement('div'); right.className = 'half right';

            left.addEventListener('click', (ev) => {
              ev.stopPropagation();
              if (turn === 'setup') return;
              const ok = doPlaceSwan(r,c);
              if (ok) splitIdx = null;
              render();
            });

            right.addEventListener('click', (ev) => {
              ev.stopPropagation();
              if (turn === 'setup') return;
              const ok = doPlaceStone(r,c);
              if (ok) splitIdx = null;
              render();
            });

            chooser.appendChild(left); chooser.appendChild(right);
            inner.appendChild(chooser);
          }

          // piece glyphs
          if (v===SWAN_SUN || v===FROZEN_SUN) { glyph.classList.add('pieceSun');  glyph.innerHTML = SVG_SUN; }
          if (v===SWAN_MOON || v===FROZEN_MOON){ glyph.classList.add('pieceMoon'); glyph.innerHTML = SVG_MOON; }

          // highlight selected swans during move / push actions
          if (isActiveSwan(v) && action==='moveSwans' && selected.has(idx(r,c))) {
            inner.classList.add('selectedMove');
          }
          if (isSwan(v) && action==='pushSwans' && selected.has(idx(r,c))) {
            inner.classList.add('selectedPush');
          }

          inner.appendChild(glyph);

          // move hints around anchor swan (chebyshev distance 1)
          if (action==='moveSwans' && selected.size>0 && anchorIdx!==null) {
            const ar = Math.floor(anchorIdx / SIZE), ac = anchorIdx % SIZE;
            const dr = r - ar, dc = c - ac;
            if (Math.max(Math.abs(dr), Math.abs(dc)) === 1) {    // only neighbours
              const dir = `${dr},${dc}`;
              const ok = canMoveSelected(dir);
              const kHere = idx(r,c);
              const vHere = cell(r,c);
              const isSelFriendly = isSwan(vHere) && activeSwanOf(current, vHere) && selected.has(kHere);

              if (isSelFriendly) {
                // split overlay over friendly selected; left = move, right = deselect
                const split = document.createElement('div'); split.className = 'moveSplit';

                const left = document.createElement('div');  left.className = 'moveHalf left ' + (ok ? 'green' : 'red');
                left.addEventListener('click', (ev)=>{ ev.stopPropagation();
                tryMoveSelected(dir); });

                const right = document.createElement('div'); right.className = 'moveHalf right sel';
                right.addEventListener('click', (ev)=>{ ev.stopPropagation();
                toggleSelectSwan(r,c); });

                split.appendChild(left); split.appendChild(right);
                inner.appendChild(split);
              } else {
                // full tile overlay hint
                const overlay = document.createElement('div'); overlay.className = 'moveHint ' + (ok ? 'green' : 'red');
                overlay.addEventListener('click', (ev)=>{
                  ev.stopPropagation();
                  const vHere2 = cell(r,c);
                  if (isSwan(vHere2) && activeSwanOf(current, vHere2)) { toggleSelectSwan(r,c);
                  return; }
                  tryMoveSelected(dir);
                });
                inner.appendChild(overlay);
              }
            }
          }

          // push hints around anchor enemy swan (same pattern)
          if (action==='pushSwans' && selected.size>0 && anchorIdx!==null) {
            const ar = Math.floor(anchorIdx / SIZE), ac = anchorIdx % SIZE;
            const dr = r - ar, dc = c - ac;
            if (Math.max(Math.abs(dr), Math.abs(dc)) === 1) {    // only neighbours
              const dir = `${dr},${dc}`;
              const ok = canPushSelected(dir);
              const kHere = idx(r,c);
              const vHere = cell(r,c);
              const isSelEnemy = isSwan(vHere) && enemySwan(vHere, current) && selected.has(kHere);

              if (isSelEnemy) {
                // split overlay over selected enemy; left = push, right = deselect
                const split = document.createElement('div'); split.className = 'moveSplit';

                const left = document.createElement('div');  left.className = 'moveHalf left ' + (ok ? 'green' : 'red');
                left.addEventListener('click', (ev)=>{ ev.stopPropagation();
                tryPushSelected(dir); });

                const right = document.createElement('div'); right.className = 'moveHalf right sel selPush';
                right.addEventListener('click', (ev)=>{ ev.stopPropagation();
                toggleSelectPushSwan(r,c); });

                split.appendChild(left); split.appendChild(right);
                inner.appendChild(split);
              } else {
                // full tile overlay hint
                const overlay = document.createElement('div'); overlay.className = 'moveHint ' + (ok ? 'green' : 'red');
                overlay.addEventListener('click', (ev)=>{
                  ev.stopPropagation();
                  const vHere2 = cell(r,c);
                  if (isSwan(vHere2) && enemySwan(vHere2, current)) {
                    toggleSelectPushSwan(r,c);
                    return;
                  }
                  tryPushSelected(dir);
                });
                inner.appendChild(overlay);
              }
            }
          }

          // draw hint overlays (best/worst)
          const k = idx(r,c);
          if(hintBestCells.has(k)){
            const o = document.createElement('div');
            o.className = 'hintMark best';
            inner.appendChild(o);
          }
          if(hintWorstCells.has(k)){
            const o = document.createElement('div');
            o.className = 'hintMark worst';
            inner.appendChild(o);
          }
          // draw recent move/placement highlights
          if (recentFromCells.has(k)){
            const o = document.createElement('div');
            o.className = 'recentMark from';
            inner.appendChild(o);
          }
          if (recentToCells.has(k)){
            const o = document.createElement('div');
            o.className = 'recentMark to';
            inner.appendChild(o);
          }
          d.appendChild(inner);
          d.addEventListener('click',()=>onCell(r,c)); // main click handler
          elBoard.appendChild(d);
        }
      }

      // turn pill
      const isGameActive = (appMode === 'playing' || appMode === 'review');
      if (!isGameActive) { elTurn.innerHTML = ''; return; }

      let phtml='';
      if(turn==='setup'){
        phtml = `<span class="pill ${toPlace==='sun'?'sun':'moon'}">Setup: ${(toPlace==='sun')? 'Sun place first Swan' : 'Moon place second Swan (not adjacent)'}</span>`;
      } else {
        const who = current===SUN? 'Sun ☼' : 'Moon ☾';
        const acts = `${movesLeft} action${movesLeft>1?'s':''}`;
        const thinking = aiThinking && aiSide === current ? ' • AI thinking…' : '';
        phtml = `<span class="pill ${current===SUN?'sun':'moon'}">Turn: ${who}${movesLeft>1 ? ` • ${acts}` : ''}${thinking}</span>`;
      }
      elTurn.innerHTML = phtml;

      updateActionButtonsEnabled();
    }

    /* ===============================================================
       game control (reset, start game, click handling)
       =============================================================== */
    function reset(){

      resetVeryHardSession();
      // re-enable board
      enableBoardInteraction();
      // reset transient ui
      splitIdx = null;
      anchorIdx = null;
      selected.clear();
      action = null;
      history = [];
      movesLeft = 1;
      gameOver = false;
      // ensure no lingering AI side from imported recite
      aiSide = null;

      // reset board state and phase
      board = Array.from({length: SIZE},
      ()=>Array(SIZE).fill(EMPTY));
      turn = 'setup';
      toPlace = 'sun';
      current = MOON;

      // back to menu
      appMode = 'menu';
      recite = false;
      replay = null;
      replayIdx = 0;
      updateReviewButtons();
      if (elStartMenu) elStartMenu.style.display = 'grid';
      if (elLog) { elLog.style.display = 'none';
      elLog.innerHTML = ''; }

      // Restore user preference for clock mode (avoid lingering recite mode)
      try {
        const pref = localStorage.getItem(CLOCK_KEY) || 'off';
        clockMode = pref;
      } catch {}

      timerStop();
      timerReset();
      stopMasterLoop();
      startMasterLoop();       // restart cleanly for the next run
      resetClocksForNewGame();
      applyClockVisibility();
      updateClockFacesVisibility();
      // clear transient recent highlights
      try { if (recentTimer) clearTimeout(recentTimer); } catch {}
      recentFromCells.clear();
      recentToCells.clear();
      render();
      renderAxisLabels();
      updateActionButtonsEnabled();
      // hide replay controls and stop auto on reset
      stopReplayAuto?.();
      setReplayControlsVisible?.(false);
    }

    function startgame(){
      resetVeryHardSession();
      appMode = 'playing';
      recite = false;
      gameOver = false;
      replay = null;
      replayIdx = 0;
      updateReviewButtons();
      splitIdx = null;
      anchorIdx = null;

      board = Array.from({length: SIZE}, ()=>Array(SIZE).fill(EMPTY));
      turn = 'setup';
      toPlace = 'sun';
      current = MOON;
      selected.clear();
      action = null;
      history = [];
      movesLeft = 1;
      moveNumber = 0;
      if (elLog) elLog.innerHTML = '';
      if (elStartMenu) elStartMenu.style.display = 'none';
      if (elLog) elLog.style.display = 'block';
      updateActionButtonsEnabled();

      try {
        const pref = localStorage.getItem(CLOCK_KEY) || 'off';
        clockMode = pref;
      } catch {}

      startMasterLoop();
      timerStop();
      timerReset();
      resetClocksForNewGame();
      applyClockVisibility();
      updateClockFacesVisibility();
      // clear transient recent highlights
      try { if (recentTimer) clearTimeout(recentTimer); } catch {}
      recentFromCells.clear();
      recentToCells.clear();
      // after clocks are fully reset/visible, capture the initial snapshot
      history.push(cloneState({ tag: 'initial' }));
      // ensure the freshly reset clocks are what we render first
      render();
      log('New Game.');
      renderAxisLabels();
      aisetup();
      // hide replay controls and stop auto when starting a game
      stopReplayAuto?.();
      setReplayControlsVisible?.(false);
    }

    // main cell click router
    function onCell(r, c) {
      if (appMode === 'menu') return; // ignore before start

      // --- replay / rewound: allow Swan selection to preview moves, then stop
      if (appMode === 'review' || !isAtTip()) {
        const vClick = cell(r, c);
        // In replay, don't filter by `current` (it may be opponent at this snapshot)
        if (isSwan(vClick)) {
          toggleSelectSwan(r, c);
        }
        return; // IMPORTANT: prevent any normal-play handlers/commits
      }

      // --- normal play path (only when at the tip)
      if (boardLocked) return;

      // if moving clicking friendly swan toggles selection (multi-select)
      if (action==='moveSwans') {
        const vClick = cell(r,c);
        if (isSwan(vClick) && activeSwanOf(current, vClick)) {
          toggleSelectSwan(r,c);
          return;
        }
      }

      // if pushing, clicking enemy Swan toggles selection
      if (action==='pushSwans') {
        const vClick = cell(r,c);
        if (isSwan(vClick) && enemySwan(vClick, current) && isActiveSwan(vClick)) {
          toggleSelectPushSwan(r,c);
          return;
        }
      }

      // during setup only place initial swans
      if(turn==='setup') return setupClick(r,c);

      // no action selected
      if(!action){
        const v0 = cell(r,c);

        // click your swan, start move mode
        if (activeSwanOf(current, v0)){
          action = 'moveSwans';
          selected.clear();
          anchorIdx = null;
          toggleSelectSwan(r,c);
          return;
        }

        // click enemy swan adjacent to any of your active swans, start push mode
        if (isSwan(v0) && enemySwan(v0, current) && isActiveSwan(v0)) {
          const hasPusher = neighbours8(r,c).some(([nr,nc])=>{
            const vN = cell(nr,nc);
            return isSwan(vN) && samePlayerSwan(vN, current) && isActiveSwan(vN);
          });
          if (hasPusher){
            action = 'pushSwans';
            selected.clear();
            anchorIdx = null;
            toggleSelectPushSwan(r,c);
            return;
          }
        }

        // click empty, open split chooser (place swan/stone)
        if (isEmpty(r,c)){
          const k = idx(r,c);
          splitIdx = (splitIdx === k) ? null : k;
          render();
          return;
        }
        return;
      }

      // action selected
      if(action==='placeSwan')  return doPlaceSwan(r,c);
      if(action==='placeStone') return doPlaceStone(r,c);
      if(action==='moveSwans')  return toggleSelectSwan(r,c);
      if(action==='pushSwans')  return toggleSelectPushSwan(r,c);
    }

    // setup clicks for initial two swans
    function setupClick(r,c){
      if (boardLocked) return;
      if (!isEmpty(r,c)) return;

      // sun places first anywhere
      if (toPlace === 'sun') {
        setcell(r,c,SWAN_SUN);
        flashRecentIfAIOnly([], [idx(r,c)], 1200, 'human');
        playReady('place',{ rate:1.0 });
        toPlace = 'moon';
        const msgSun = `Sun placed their first Swan at ${tileAlg(r,c)}.`;
        pushHistory({
          tag: 'placeSwan',
          actor: 'sun',
          tile: tileAlg(r,c),
          isMove: true,
          log: msgSun
        });
        log(msgSun);
        render();
        aisetup();
        return;
      }

      // moon cannot be adjacent to sun's initial
      const adjSun = neighbours8(r,c).some(([nr,nc]) => cell(nr,nc) === SWAN_SUN);
      if (adjSun) {
        log('Moon cannot place adjacent to Sun\'s initial Swan.');
        return;
      }

      setcell(r,c,SWAN_MOON);
      flashRecentIfAIOnly([], [idx(r,c)], 1200, 'human');

      playReady('place', { rate: 1.0 });

      // move into real play *before* snapshot
      turn = 'play';
      current = MOON;
      activeSide = (current === SUN ? 'SUN' : 'MOON');
      markActive(activeSide);

      const msgMoon = `Moon placed their first Swan at ${tileAlg(r,c)}. Moon begins.`;
      log(msgMoon);

      // snapshot now sees turn === 'play'
      pushHistory({
        tag: 'placeSwan',
        actor: 'moon',
        tile: tileAlg(r,c),
        isMove: true,
        log: msgMoon
      });

      if (clockMode === 'stopwatch') {
        timerStart();
      }
      beginTurn();
      render();
      aiturn();
    }

    /* ===============================================================
       rules helpers (ownership checks & counts)
       =============================================================== */
    function samePlayerSwan(v,p){ return (p===SUN && (v===SWAN_SUN||v===FROZEN_SUN)) || (p===MOON && (v===SWAN_MOON||v===FROZEN_MOON)); }
    function activeSwanOf(p,v){ return (p===SUN && v===SWAN_SUN) || (p===MOON && v===SWAN_MOON); }
    function enemySwan(v,p){ return (p===SUN && (v===SWAN_MOON||v===FROZEN_MOON)) || (p===MOON && (v===SWAN_SUN||v===FROZEN_SUN)); }

    // total swans (active and frozen) for momentum cap
    function countTotalSwans(p){
      let n = 0;
      for (let r=0;r<SIZE;r++) for (let c=0;c<SIZE;c++){
        const v = cell(r,c);
        if ((p===SUN  && (v===SWAN_SUN || v===FROZEN_SUN)) ||
            (p===MOON && (v===SWAN_MOON|| v===FROZEN_MOON))) n++;
      }
      return n;
    }
    function countActiveSwans(p){
      let n=0; for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){ const v=cell(r,c); if(activeSwanOf(p,v)) n++; } return n;
    }
    function anyEmpty(){
      for (let r=0;r<SIZE;r++) for (let c=0;c<SIZE;c++) if (cell(r,c) === EMPTY) return true;
      return false;
    }

    // convert 1..6 to first/s/third/... specifically for Swan count lines
    function swanOrdinalWord(n){
      switch(n){
        case 1: return 'first';
        case 2: return 'second';
        case 3: return 'third';
        case 4: return 'fourth';
        case 5: return 'fifth';
        case 6: return 'sixth';
        default: return `${n}th`;
      }
    }

    /* ===============================================================
       actions - placement (swan/stone)
       =============================================================== */
    function doPlaceSwan(r,c){
      if (boardLocked) return false;
      const p = current;
      if (countTotalSwans(p) >= 6) { log('You already have six Swans on the board.'); return false; }
      if (!isEmpty(r,c)) return false;

      let hasAdjMine=false, adjEnemy=false;
      for (const [nr,nc] of neighbours4(r,c)) {
        const v = cell(nr,nc);
        if (samePlayerSwan(v,p)) hasAdjMine=true;
      }
      for (const [nr,nc] of neighbours8(r,c)) {
        const v = cell(nr,nc);
        if (enemySwan(v,p)) { adjEnemy=true; break; }
      }
      if (!hasAdjMine) { log('Swan placement must be adjacent to one of your Swans.'); return false; }
      if (adjEnemy)    { log('Swan cannot be placed adjacent to an opponent Swan.'); return false; }

      // apply placement to the live board first
      setcell(r,c, p===SUN ? SWAN_SUN : SWAN_MOON);
      // transient highlight for placement destination (AI-only in vs-AI)
      flashRecentIfAIOnly([], [idx(r,c)]);

      // predict freezes (pure), then mutate board to the frozen state for consistency with replay
      const encPred = computeFreezesOn(board);
      board = encPred.nb; // bring live board to the post-freeze state
      render();

      // compute who moves next and remaining actions after this placement
      const opponentLoss = (current===SUN) ? (encPred.frozeMoon + encPred.sealedMoon)
                                           : (encPred.frozeSun  + encPred.sealedSun);
      let nextMovesLeft = movesLeft;
      nextMovesLeft += opponentLoss;      // gain 1 action per enemy Swan frozen/sealed
      nextMovesLeft--;                    // spend this action
      let nextCurrent, nextMoves;
      if (nextMovesLeft > 0) {
        nextCurrent = current;
        nextMoves   = nextMovesLeft;
      } else {
        nextCurrent = (current===SUN) ? MOON : SUN;
        nextMoves   = bothAtSix() ? 2 : 1;
      }

      const actorName = (p===SUN ? 'Sun' : 'Moon');
      // count AFTER placement (board already updated), include frozen
      const placedCount = countTotalSwans(p);
      function swanOrdinalWord(n){
        switch(n){
          case 1: return 'first';
          case 2: return 'second';
          case 3: return 'third';
          case 4: return 'fourth';
          case 5: return 'fifth';
          case 6: return 'sixth';
          default: return `${n}th`;
        }
      }
      const ord = swanOrdinalWord(placedCount);
      const msg = `${actorName} placed their ${ord} Swan at ${tileAlg(r,c)}.`;
      const freezeNotes = composeFreezeNotes(encPred, actorName);
      // record snapshot reflecting the correct side-to-move after this action
      suppressTipOnce = true;
      pushHistory({
        tag: 'placeSwan',
        actor: (p === SUN ? 'sun' : 'moon'),
        tile: tileAlg(r,c),
        log: msg,
        freezeNotes,
        isMove: true,
        current: nextCurrent,
        movesLeft: nextMoves,
        enc: { frozeSun: encPred.frozeSun, frozeMoon: encPred.frozeMoon, sealedSun: encPred.sealedSun, sealedMoon: encPred.sealedMoon }
      });

      playReady('place', { rate: 1.02 });
      log(msg);
      // Add freeze notes (non-traversable log entries)
      for (const m of freezeNotes) log(m);
      // pass encPred to endTurn so freezes are not recomputed unnecessarily
      endTurn(encPred);
      return true;
    }

    function doPlaceStone(r, c) {
      if (boardLocked) return false;
      if (!isEmpty(r, c)) return false;

      // apply the placement
      setcell(r, c, STONE);
      // transient highlight for placement destination (AI-only in vs-AI)
      flashRecentIfAIOnly([], [idx(r,c)]);

      // predict freezes (pure), then mutate board to the frozen state for consistency
      const encPred = computeFreezesOn(board);
      board = encPred.nb; // bring live board to the post-freeze state
      render();

      // compute who moves next and remaining actions after this placement
      const opponentLoss = (current===SUN) ? (encPred.frozeMoon + encPred.sealedMoon)
                                           : (encPred.frozeSun  + encPred.sealedSun);
      let nextMovesLeft = movesLeft;
      nextMovesLeft += opponentLoss;      // gain 1 action per enemy Swan frozen/sealed
      nextMovesLeft--;                    // spend this action
      let nextCurrent, nextMoves;
      if (nextMovesLeft > 0) {
        nextCurrent = current;
        nextMoves   = nextMovesLeft;
      } else {
        nextCurrent = (current===SUN) ? MOON : SUN;
        nextMoves   = bothAtSix() ? 2 : 1;
      }

      const actor = (current === SUN ? 'sun' : 'moon');
      const actorName = (current === SUN ? 'Sun' : 'Moon');
      const msg = `${actorName} placed a Stone at ${tileAlg(r, c)}.`;
      const freezeNotes = composeFreezeNotes(encPred, actorName);

      // save a labelled snapshot that already encodes the correct next player
      suppressTipOnce = true;
      pushHistory({
        tag: 'placeStone',
        actor,
        tile: tileAlg(r, c),
        row: r,
        col: c,
        log: msg,
        freezeNotes,
        isMove: true,
        current: nextCurrent,
        movesLeft: nextMoves,
        enc: { frozeSun: encPred.frozeSun, frozeMoon: encPred.frozeMoon, sealedSun: encPred.sealedSun, sealedMoon: encPred.sealedMoon }
      });

      playReady('place', { rate: 0.98 });

      log(msg);
      for (const m of freezeNotes) log(m);

      endTurn(encPred);
      return true;
    }

    /* ===============================================================
       actions - movement (multi-select swans + stones follow)
       =============================================================== */
    function toggleSelectSwan(r,c){
      // allow selection during replay/review; commits remain blocked in trymoveselected/doplace
      if (boardLocked && !(appMode === 'review' || !isAtTip())) return;

      const v = cell(r,c);

      // during replay we allow probing any swan, regardless of `current`
      if (appMode === 'review' || !isAtTip()) {
        if (!isSwan(v)) return;
      } else {
        if (!activeSwanOf(current, v)) return; // normal at-tip owner check
      }

      const k = idx(r,c);

      if (selected.has(k)) {
        selected.delete(k);
        if (anchorIdx === k) {
          const arr = Array.from(selected);
          anchorIdx = arr.length ? arr[0] : null;
        }
      } else {
        selected.add(k);
        if (anchorIdx === null) anchorIdx = k;
      }

      // ensure probe mode during review/rewind
      if (appMode === 'review' || !isAtTip()) {
        action = selected.size ? 'moveSwans' : null;
      }

      if (selected.size === 0) { action = null; anchorIdx = null; }
      render();
    }

    // selection for enemy Swans to be pushed
    function toggleSelectPushSwan(r,c){
      if (boardLocked && !(appMode === 'review' || !isAtTip())) return;

      const v = cell(r,c);
      // For now, push selection only makes sense in normal play at the tip
      if (appMode === 'review' || !isAtTip()) return;

      // only active enemy Swans can be pushed
      if (!isSwan(v) || !enemySwan(v, current) || !isActiveSwan(v)) return;

      const k = idx(r,c);

      if (selected.has(k)) {
        selected.delete(k);
        if (anchorIdx === k) {
          const arr = Array.from(selected);
          anchorIdx = arr.length ? arr[0] : null;
        }
      } else {
        selected.add(k);
        if (anchorIdx === null) anchorIdx = k;
      }

      if (selected.size === 0) { action = null; anchorIdx = null; }

      render();
    }

    function canMoveSelected(dir){
      if(selected.size===0) return false;
      const [dr,dc] = dir.split(',').map(Number);
      const movingSwans = new Set(selected);
      const stoneKey = (r,c)=>`s:${r},${c}`;

      // find stones dragged by moving swans (exclude shared)
      const stonesFrom = new Set();
      const stonesTo   = new Map();

      for(const k of movingSwans){
        const r=Math.floor(k/SIZE), c=k%SIZE;
        for(const [nr,nc] of neighbours8(r,c)){
          if(!inb(nr,nc)) continue;
          if(cell(nr,nc)!==STONE) continue;

          let adjEnemy=false, adjUnmovedFriendly=false;
          for(const [ar,ac] of neighbours8(nr,nc)){
            if(!inb(ar,ac)) continue;
            const v=cell(ar,ac);
            if(isSwan(v) && enemySwan(v,current)){ adjEnemy=true; break; }
          }
          if(!adjEnemy){
            for(const [ar,ac] of neighbours8(nr,nc)){
              if(!inb(ar,ac)) continue;
              const v=cell(ar,ac);
              const friendly = samePlayerSwan(v,current);
              const movingFriendly = friendly && activeSwanOf(current,v) && movingSwans.has(idx(ar,ac));
              if(friendly && !movingFriendly){ adjUnmovedFriendly=true; break; }
            }
          }
          if(adjEnemy || adjUnmovedFriendly) continue;

          const tr = nr + dr, tc = nc + dc;
          if(!inb(tr,tc)) return false;
          stonesFrom.add(stoneKey(nr,nc));
          stonesTo.set(stoneKey(nr,nc), [tr,tc]);
        }
      }

      const isVacantAfterMove = (r,c)=>{
        if(!inb(r,c)) return false;
        if(isEmpty(r,c)) return true;
        if(movingSwans.has(idx(r,c))) return true;
        if(stonesFrom.has(stoneKey(r,c))) return true;
        return false;
      };

      // validate swan destinations
      // helper: enemy Swan with no Stones in any of the 8 surrounding tiles
      function isEnemySwanNaked(sr, sc){
        const v = cell(sr, sc);
        if (!isSwan(v)) return false;
        if (!enemySwan(v, current)) return false;

        // any Stone in the 8 neighbours means it is NOT naked
        for (const [nr, nc] of neighbours8(sr, sc)) {
          if (!inb(nr, nc)) continue;        // neighbours8 is already in-bounds, but safe
          if (cell(nr, nc) === STONE) {
            return false;
          }
        }
        // no Stones adjacent in any of the 8 directions
        return true;
      }

      // helper: is (r,c) inside the 8-neighbourhood of a naked enemy Swan?
      function isInNakedEnemyZone(r, c){
        for (const [er, ec] of neighbours8(r, c)) {
          if (!inb(er, ec)) continue;
          const v = cell(er, ec);
          if (!isSwan(v)) continue;
          if (!enemySwan(v, current)) continue;
          if (isEnemySwanNaked(er, ec)) return true;
        }
        return false;
      }

      // validate swan destinations
      for (const k of movingSwans) {
        const r = Math.floor(k / SIZE), c = k % SIZE;
        const nr = r + dr, nc = c + dc;
        if (!inb(nr, nc)) return false;

        // NEW: cannot move into the 8 tiles around a naked enemy Swan
        if (isInNakedEnemyZone(nr, nc)) return false;

        const occ = cell(nr, nc);

        if (occ === EMPTY) continue;
        if (isSwan(occ)) {
          if (movingSwans.has(idx(nr, nc))) continue;
          return false;
        }
        if (occ === STONE) {
          const kStone = stoneKey(nr, nc);
          if (!stonesTo.has(kStone)) return false;
          const [tr, tc] = stonesTo.get(kStone);
          if (!isVacantAfterMove(tr, tc)) return false;
        }
      }

      // validate stones map (no collisions, valid targets)
      const seenDest = new Set()

      for(const [_, [tr,tc]] of stonesTo){
        if(!isVacantAfterMove(tr,tc)) return false;
        const tkey = `${tr},${tc}`; if(seenDest.has(tkey)) return false; seenDest.add(tkey);
      }
      return true;
    }

    function tryMoveSelected(dir){
      if (boardLocked) return;
      if (selected.size === 0) {
        log('Select one or more of your Swans to move.');
        return;
      }

      const [dr, dc] = dir.split(',').map(Number);
      const movingSwans = new Set(selected);
      const stoneKey = (r,c)=>`s:${r},${c}`;
      const keyToRC = k => k.substring(2).split(',').map(Number);

      // collect stones that follow
      const stonesFrom = new Set();
      const stonesTo   = new Map();

      for (const k of movingSwans){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        for (const [nr, nc] of neighbours8(r, c)){
          if (!inb(nr, nc) || cell(nr, nc) !== STONE) continue;

          let adjEnemy = false, adjUnmovedFriendly = false;

          // enemy adjacency
          for (const [ar, ac] of neighbours8(nr, nc)){
            if (!inb(ar, ac)) continue;
            const v = cell(ar, ac);
            if (isSwan(v) && enemySwan(v, current)) { adjEnemy = true; break; }
          }

          // unmoved friendly adjacency
          if (!adjEnemy){
            for (const [ar, ac] of neighbours8(nr, nc)){
              if (!inb(ar, ac)) continue;
              const v = cell(ar, ac);
              const friendly = samePlayerSwan(v, current);
              const movingFriendly = friendly && activeSwanOf(current, v) && movingSwans.has(idx(ar, ac));
              if (friendly && !movingFriendly){ adjUnmovedFriendly = true; break; }
            }
          }

          if (adjEnemy || adjUnmovedFriendly) continue;

          const sk = stoneKey(nr, nc);
          stonesFrom.add(sk);
          stonesTo.set(sk, [nr + dr, nc + dc]);
        }
      }

      const isVacantAfterMove = (r,c)=>{
        if (!inb(r,c)) return false;
        if (isEmpty(r,c)) return true;
        if (movingSwans.has(idx(r,c))) return true;
        if (stonesFrom.has(stoneKey(r,c))) return true;
        return false;
      };

      // validate swan targets
      // Pre-check: cannot move into the 8 tiles around an enemy Swan with no adjacent Stones
      {
        const violates = Array.from(movingSwans).some(k=>{
          const r = Math.floor(k / SIZE), c = k % SIZE;
          const nr = r + dr, nc = c + dc;
          if (!inb(nr, nc)) return false; // handled below
          // local helper
          const isEnemySwanNaked = (sr, sc) => {
            const v = cell(sr, sc);
            if (!isSwan(v) || !enemySwan(v, current)) return false;
            for (const [ar, ac] of neighbours8(sr, sc)){
              if (!inb(ar, ac)) continue;
              if (cell(ar, ac) === STONE) return false;
            }
            return true;
          };
          for (const [er, ec] of neighbours8(nr, nc)){
            if (!inb(er, ec)) continue;
            if (isEnemySwanNaked(er, ec)) return true;
          }
          return false;
        });
        if (violates){
          log('Move blocked - you cannot move into the tiles around a free enemy Swan.');
          return;
        }
      }

      for (const k of movingSwans) {
        const r = Math.floor(k / SIZE), c = k % SIZE;
        const nr = r + dr, nc = c + dc;

        if (!inb(nr, nc)) { log('Move blocked - target out of bounds or occupied.'); return; }

        const occ = cell(nr, nc);
        if (occ === EMPTY) {
          // ok
        } else if (isSwan(occ)) {
          if (!movingSwans.has(idx(nr, nc))) {
            log('Move blocked - target occupied by a stationary Swan.');
            return;
          }
        } else if (occ === STONE) {
          const sk = stoneKey(nr, nc);
          if (stonesFrom.has(sk)) {
            const [tr, tc] = stonesTo.get(sk);
            if (!(inb(tr, tc) && isVacantAfterMove(tr, tc))) {
              log('Move blocked - stone cannot be displaced into an occupied square.');
              return;
            }
          } else {
            log('Move blocked - stone cannot be displaced into an occupied square.');
            return;
          }
        } else {
          log('Move blocked - target out of bounds or occupied.');
          return;
        }
      }

      // validate stone targets
      const seenDest = new Set();
      for (const [sk,[tr,tc]] of stonesTo){
        if (!inb(tr,tc)) { log('Move blocked during stone transfer.'); return; }
        if (!isVacantAfterMove(tr,tc)) { log('Move blocked during stone transfer.'); return; }
        const tkey = `${tr},${tc}`;
        if (seenDest.has(tkey)) {
          log('Move blocked - two stones conflict on the same destination.');
          return;
        }
        seenDest.add(tkey);
      }

      // precompute recent highlight indices (from -> to) for swans and stones
      const recentFrom = [];
      const recentTo = [];
      for (const k of movingSwans){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        recentFrom.push(idx(r,c));
        recentTo.push(idx(r + dr, c + dc));
      }
      for (const [sk, [tr, tc]] of stonesTo){
        const [sr, sc] = keyToRC(sk);
        recentFrom.push(idx(sr, sc));
        recentTo.push(idx(tr, tc));
      }

      // ---- apply move ----

      // clear swans
      for (const k of movingSwans){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        setcell(r, c, EMPTY);
      }

      // clear stones
      for (const sk of stonesFrom){
        const [r, c] = keyToRC(sk);
        setcell(r, c, EMPTY);
      }

      // place stones
      for (const [sk, [tr, tc]] of stonesTo){
        setcell(tr, tc, STONE);
      }

      // place swans in consistent order
      for (const k of Array.from(movingSwans).sort((a,b)=>a-b)){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        setcell(r + dr, c + dc, (current === SUN ? SWAN_SUN : SWAN_MOON));
      }

      // apply freezes/seals as part of the visible move result
      const enc = resolveEncirclements(); // mutates board + render
      // flash recent move origins/destinations briefly (AI-only in vs-AI)
      flashRecentIfAIOnly(recentFrom, recentTo);

      // summary for log and record
      const movedFrom = [], movedTo = [];
      for (const k of movingSwans) {
        const r = Math.floor(k / SIZE), c = k % SIZE;
        const nr = r + dr, nc = c + dc;
        movedFrom.push(tileAlg(r, c));
        movedTo.push(tileAlg(nr, nc));
      }

      const actorName = (current === SUN ? 'Sun' : 'Moon');
      const actor = (current === SUN ? 'sun' : 'moon');

      // small helper for 1–6 swans
      const numberToWordForLog = (n) => {
        const map = {
          0: 'no',
          1: 'one',
          2: 'two',
          3: 'three',
          4: 'four',
          5: 'five',
          6: 'six'
        };
        return map[n] || String(n);
      };

      const movedCount = movedFrom.length;
      const msg = movedCount
        ? `${actorName} moved ${numberToWordForLog(movedCount)} of their Swan${movedCount === 1 ? '' : 's'} from ${movedFrom.join(', ')} to ${movedTo.join(', ')}.`
        : `${actorName} moved.`;

      // compute correct next side-to-move for the snapshot
      const opponentLoss = (current===SUN) ? (enc.frozeMoon + enc.sealedMoon)
                                           : (enc.frozeSun  + enc.sealedSun);
      let nextMovesLeft = movesLeft;
      nextMovesLeft += opponentLoss;      // gain 1 action per enemy Swan frozen/sealed
      nextMovesLeft--;                    // spend this action
      let nextCurrent, nextMoves;
      if (nextMovesLeft > 0) {
        nextCurrent = current;
        nextMoves   = nextMovesLeft;
      } else {
        nextCurrent = (current===SUN) ? MOON : SUN;
        nextMoves   = bothAtSix() ? 2 : 1;
      }

      // record a full, labelled snapshot of this move's final board state
      suppressTipOnce = true;
      const freezeNotes = composeFreezeNotes(enc, actorName);
      pushHistory({
        tag: 'moveSwans',
        actor,
        dir,
        movedFrom,
        movedTo,
        enc,
        freezeNotes,
        isMove: true,
        log: msg,
        current: nextCurrent,
        movesLeft: nextMoves
      });

      // sound and visible log + turn advance
      const count = selected.size;
      playReady(count > 1 ? 'moveMany' : 'move1', { gain: 0.75, rate: count > 1 ? 0.97 : 1.00 });

      log(msg);
      for (const m of freezeNotes) log(m);

      endTurn(enc);
    }

    // ---------- push rules (enemy swans) ----------

    // Compute which Stones would follow a set of moving Swans in a direction,
    // using the exact same rules as normal Swan movement. The moving side is
    // given by 'playerMoving' (SUN|MOON) to evaluate shared-adjacency rules
    // correctly for Stones.
    function computeFollowingStonesForSubset(subsetCoords /* [[r,c],...] */, dr, dc, playerMoving){
      const stonesFrom = new Set();           // keys "s:r,c" of origin stones
      const stonesTo   = new Map();           // key -> [tr,tc]

      const moving = new Set(subsetCoords.map(([r,c])=> r*SIZE + c));

      const stoneKey=(r,c)=>`s:${r},${c}`;

      // helper mirrors the analyzer's logic
      function isVacantAfterMove(r,c){
        if (!inb(r,c)) return false;
        const v = cell(r,c);
        if (v === EMPTY) return true;
        if (moving.has(r*SIZE + c)) return true; // a moving swan vacates it
        if (stonesFrom.has(stoneKey(r,c))) return true; // a moving stone vacates it
        return false;
      }

      // gather Stones adjacent to any moving Swan that will follow
      for (const [r,c] of subsetCoords){
        for (const [er,ec] of DIRS8){
          const sr = r+er, sc = c+ec;
          if (!inb(sr,sc) || cell(sr,sc)!==STONE) continue;

          // shared with any non-moving neighbour according to rules?
          let shared = false;
          for (const [ar,ac] of DIRS8){
            const xr = sr+ar, xc = sc+ac;
            if (!inb(xr,xc)) continue;
            const vv = cell(xr,xc);
            if (!isSwan(vv)) continue;
            // If adjacent to an enemy swan (relative to the moving side), it's shared
            if (enemySwan(vv, playerMoving)) { shared = true; break; }
            // Frozen friendly Swans are stationary and share/anchor the Stone too.
            const movingFriendly = samePlayerSwan(vv, playerMoving) && isActiveSwan(vv) && moving.has(xr*SIZE + xc);
            if (samePlayerSwan(vv, playerMoving) && !movingFriendly) { shared = true; break; }
          }
          if (shared) continue;

          const tr = sr + dr, tc = sc + dc;
          if (!inb(tr,tc)) return null; // stone would go off-board -> illegal overall
          const sk = stoneKey(sr,sc);
          stonesFrom.add(sk);
          stonesTo.set(sk, [tr,tc]);
        }
      }

      // Validate stone targets are vacant/unique wrt this move
      const seen = new Set();
      for (const [_, [tr,tc]] of stonesTo){
        if (!isVacantAfterMove(tr,tc)) return null;
        const k = `${tr},${tc}`;
        if (seen.has(k)) return null; // two stones colliding
        seen.add(k);
      }

      return { stonesFrom, stonesTo, isVacantAfterMove };
    }

    function canPushSelected(dir){
      if (selected.size === 0) return false;
      const [dr, dc] = dir.split(',').map(Number);
      const pushingSwans = new Set(selected);

      // helper: each enemy Swan must have at least one adjacent friendly active Swan
      function hasFriendlyPusher(r,c){
        for (const [nr,nc] of neighbours8(r,c)){
          if (!inb(nr,nc)) continue;
          const v = cell(nr,nc);
          if (isSwan(v) && samePlayerSwan(v, current) && isActiveSwan(v)) return true;
        }
        return false;
      }

      // helper: enemy Swan at (sr,sc) with no adjacent stones
      function isEnemySwanNaked(sr, sc){
        const v = cell(sr, sc);
        if (!isSwan(v) || !enemySwan(v, current)) return false;
        for (const [ar, ac] of neighbours8(sr, sc)){
          if (!inb(ar, ac)) continue;
          if (cell(ar, ac) === STONE) return false;
        }
        return true;
      }
      // helper: is destination inside 8-neighbourhood of a naked enemy Swan
      function isInNakedEnemyZone(r, c){
        for (const [er, ec] of neighbours8(r, c)){
          if (!inb(er, ec)) continue;
          const v = cell(er, ec);
          if (!isSwan(v) || !enemySwan(v, current)) continue;
          if (isEnemySwanNaked(er, ec)) return true;
        }
        return false;
      }

      // Build subset coordinates for the selected enemy swans being pushed
      const subset = [];
      for (const k of pushingSwans){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        subset.push([r,c]);
      }

      // From the perspective of stones-follow rules, the moving side is the opponent
      const playerMoving = (current === SUN ? MOON : SUN);

      // Compute stones that will follow the pushed enemy swans (may be none)
      const stonesPack = computeFollowingStonesForSubset(subset, dr, dc, playerMoving);
      if (stonesPack === null) return false;
      const { stonesFrom, stonesTo, isVacantAfterMove } = stonesPack;

      const destSet = new Set(); // track swan destination uniqueness

      for (const k of pushingSwans){
        const r = Math.floor(k / SIZE), c = k % SIZE;
        const v = cell(r,c);

        // must still be an active enemy Swan
        if (!isSwan(v) || !enemySwan(v, current) || !isActiveSwan(v)) return false;
        if (!hasFriendlyPusher(r,c)) return false;

        const nr = r + dr, nc = c + dc;
        if (!inb(nr,nc)) return false;

        const occ = cell(nr,nc);
        const destKey = `${nr},${nc}`;

        if (occ === EMPTY) {
          if (destSet.has(destKey)) return false;
          destSet.add(destKey);
          continue;
        }

        if (isSwan(occ)) {
          // allowed only if it is also one of the swans being pushed from this origin square
          if (!pushingSwans.has(idx(nr,nc))) return false;
          if (destSet.has(destKey)) return false;
          destSet.add(destKey);
          continue;
        }

        if (occ === STONE) {
          // Allowed only if that Stone is also moving out due to following rules
          const sk = `s:${nr},${nc}`;
          if (!stonesTo.has(sk)) return false;
          const [tr, tc] = stonesTo.get(sk);
          if (!isVacantAfterMove(tr, tc)) return false;
          if (destSet.has(destKey)) return false; // another swan trying to land here concurrently
          destSet.add(destKey);
          continue;
        }

        // cannot push into anything else
        return false;
      }

      return true;
    }

    function tryPushSelected(dir) {
        if (boardLocked) return;
        if (selected.size === 0) {
            log('Select one or more enemy Swans to push.');
            return;
        }

        if (!canPushSelected(dir)) {
            log('Push is not legal in that direction.');
            return;
        }

        const [dr, dc] = dir.split(',').map(Number);
        const pushingSwans = new Set(selected);

        // Prepare subset and compute stones that will follow (same as in canPushSelected)
        const subset = [];
        for (const k of pushingSwans){
          const r = Math.floor(k / SIZE), c = k % SIZE;
          subset.push([r,c]);
        }
        const playerMoving = (current === SUN ? MOON : SUN);
        const stonesPack = computeFollowingStonesForSubset(subset, dr, dc, playerMoving);
        // canPushSelected already validated but guard anyway
        const stonesFrom = stonesPack ? stonesPack.stonesFrom : new Set();
        const stonesTo   = stonesPack ? stonesPack.stonesTo   : new Map();

        // snapshot before move
        const movedFrom = [];
        const movedTo = [];
        const movedRC = []; // for applying

        for (const k of pushingSwans) {
            const r = Math.floor(k / SIZE), c = k % SIZE;
            const v = cell(r, c);

            const nr = r + dr, nc = c + dc;

            movedFrom.push(tileAlg(r, c));
            movedTo.push(tileAlg(nr, nc));
            movedRC.push({r, c, nr, nc, v});
        }

        // apply push to board
        // 1) clear original swans
        for (const {r, c} of movedRC) setcell(r, c, EMPTY);
        // 2) clear Stones that will follow
        for (const sk of Array.from(stonesFrom)){
          const [sr, sc] = sk.slice(2).split(',').map(Number);
          setcell(sr, sc, EMPTY);
        }
        // 3) place Stones at their targets
        for (const [_, [tr, tc]] of stonesTo){
          setcell(tr, tc, STONE);
        }
        // 4) place enemy Swans at their pushed destinations, preserving owner
        for (const {nr, nc, v} of movedRC) setcell(nr, nc, v);

        // precompute recent highlight indices (from -> to) for pushed swans
        const recentFrom = [];
        const recentTo = [];
        for (const {r, c, nr, nc} of movedRC) {
            recentFrom.push(idx(r, c));
            recentTo.push(idx(nr, nc));
        }

        // resolve encirclements and momentum exactly like a move
        const enc = resolveEncirclements(); // mutates board + render

        // flash recent push origins/destinations briefly (AI-only in vs-AI)
        flashRecentIfAIOnly(recentFrom, recentTo);

        const actorName = current === SUN ? 'Sun' : 'Moon';
        const actor      = current === SUN ? 'sun' : 'moon';

        // opponent is the opposite lowercase identifier
        const opponent      = actor === 'sun' ? 'moon' : 'sun';
        const opponentName  = opponent === 'sun' ? 'Sun' : 'Moon';

        const msg = movedFrom.length
          ? `${actorName} pushed ${
              movedFrom.length === 1
                ? `one of ${opponentName}'s Swans`
                : `${((n) => {
                     const words = ['', 'one', 'two', 'three', 'four', 'five', 'six'];
                     return words[n] || n;
                   })(movedFrom.length)} of ${opponentName}'s Swans`
            } from ${movedFrom.join(', ')} to ${movedTo.join(', ')}.`
          : `${actorName} pushed.`;

        const opponentLoss = (current === SUN) ? (enc.frozeMoon + enc.sealedMoon)
            : (enc.frozeSun + enc.sealedSun);
        let nextMovesLeft = movesLeft;
        nextMovesLeft += opponentLoss;
        nextMovesLeft--; // spend this action
        let nextCurrent, nextMoves;
        if (nextMovesLeft > 0) {
            nextCurrent = current;
            nextMoves = nextMovesLeft;
        } else {
            nextCurrent = (current === SUN) ? MOON : SUN;
            nextMoves = bothAtSix() ? 2 : 1;
        }

        suppressTipOnce = true;
        const freezeNotes = composeFreezeNotes(enc, actorName);
        pushHistory({
            tag: 'pushSwans',
            actor,
            dir,
            pushedFrom: movedFrom,
            pushedTo: movedTo,
            enc,
            freezeNotes,
            isMove: true,
            log: msg,
            current: nextCurrent,
            movesLeft: nextMoves
        });

        const count = selected.size;
        playReady(count > 1 ? 'moveMany' : 'move1', {gain: 0.75, rate: count > 1 ? 0.97 : 1.00});

        log(msg);
        for (const m of freezeNotes) log(m);

        endTurn(enc);
    }

    // optional confirm hook (kept identical to original)
    function confirmMove(){ const enc = resolveEncirclements(); endTurn(enc); }

      function resolveEncirclements(){
        const {nb,frozeSun,frozeMoon,sealedSun,sealedMoon,frozenGroups} = computeFreezesOn(board);
        board = nb; // mutate live board
        render();
        return {frozeSun,frozeMoon,sealedSun,sealedMoon,frozenGroups};
      }

    /* ===============================================================
       turn / momentum / ending (win/draw/extra action)
       =============================================================== */

      function showMessage(msg) {
        const board = document.querySelector('#board');
        if (!board) {
          console.warn('Board element not found; defaulting to window center.');
          return showMessageCenteredOnScreen(msg);
        }

        // create message box
        const box = document.createElement('div');
        box.textContent = msg;

        Object.assign(box.style, {
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '1.2rem 2rem',
          background: 'var(--panel)',
          color: 'var(--text)',
          border: '2px solid var(--black)',
          borderRadius: '12px',
          boxShadow: '0 6px 20px rgba(0, 0, 0, 0.4)',
          fontSize: '1.1rem',
          fontFamily: 'system-ui, sans-serif',
          zIndex: 9999,
          pointerEvents: 'none',
          opacity: '0',
          transition: 'opacity 0.3s ease'
        });

        board.appendChild(box);

        // fade-in
        requestAnimationFrame(() => { box.style.opacity = '1'; });

        // fade-out after delay
        setTimeout(() => {
          box.style.opacity = '0';
          setTimeout(() => box.remove(), 300);
        }, 5000);
      }

      // fallback if board not found
      function showMessageCenteredOnScreen(msg) {
        const box = document.createElement('div');
        box.textContent = msg;
        Object.assign(box.style, {
          position: 'fixed',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '1.2rem 2rem',
          background: 'var(--panel)',
          color: 'var(--text)',
          border: '2px solid var(--black)',
          borderRadius: '12px',
          fontFamily: 'system-ui, sans-serif',
          zIndex: 9999
        });
        document.body.appendChild(box);
        setTimeout(() => box.remove(), 5000);
      }

      function afterPaint(fn){
        requestAnimationFrame(()=>requestAnimationFrame(fn));
      }

      function showEndPopup(message){
        afterPaint(()=>showMessage(message));
      }

      // centralised game finish: log → render → (next frame) popup
      function finishgame(message, logLine){
        stopMasterLoop();
        resetVeryHardSession();
        timerStop()
        gameOver = true;
        appMode = 'review';          // freeze play/ai
        rebuildReplay();
        activeSide = null;           // ← ensure clocks stop & UI un-highlights
        markActive(null);
        // capture final outcome texts for saving
        try {
          lastOutcomeShort = logLine || null;
          lastOutcomeDetailed = message || null;
        } catch {}
        log(logLine);                // make sure the final action is logged
        render();                    // draw the final board state
        showEndPopup(message);       // let the paint complete before the popup
        disableBoardInteraction();   // disable the board
        updateActionButtonsEnabled();
      }

      function playOutcomeForWinner(winner /* SUN|MOON */){
        const h = humanSide(); // 'SUN' | 'MOON' | null
        if (h){
          const humanWins = (winner === SUN && h === 'SUN') ||
                            (winner === MOON && h === 'MOON');
          playReady(humanWins ? 'win' : 'loss', { gain: 0.5, rate: 1.0 });
        } else {
          // local hotseat - celebrate the winner (no single human perspective)
          playReady('win', { gain: 0.5, rate: 1.0 });
        }
      }

      function endTurn(encResult=null){
        action = null; selected.clear(); anchorIdx=null;

        const res = encResult || resolveEncirclements();

        // draw by mutual final encirclement
        if (res.sealedSun > 0 && res.sealedMoon > 0){
          playReady('draw', { gain: 0.5 });
          finishgame("Draw as both players’ final Swans were encircled on the same turn.",
                     "Draw.");
          return;
        }

        // single-side final encirclement → opponent wins
        if (res.sealedSun > 0){
          playOutcomeForWinner(MOON);
          finishgame("Moon ☾ wins by encircling Sun’s final Swan.", "Moon wins.");
          return;
        }
        if (res.sealedMoon > 0){
          playOutcomeForWinner(SUN);
          finishgame("Sun ☼ wins by encircling Moon’s final Swan.", "Sun wins.");
          return;
        }

        // fallback safety checks (active swans count)
        const aSun = countActiveSwans(SUN);
        const aMoon = countActiveSwans(MOON);
        if (aSun === 0 && aMoon === 0){
          playReady('draw', { gain: 0.5 });
          finishgame("Draw by simultaneous final encirclement.",
                     "Draw.");
          return;
        }
        if (aMoon === 0){
          playOutcomeForWinner(SUN);
          finishgame("Sun ☼ wins by encircling the final Swan.", "Sun wins.");
          return;
        }
        if (aSun === 0){
          playOutcomeForWinner(MOON);
          finishgame("Moon ☾ wins by encircling the final Swan.", "Moon wins.");
          return;
        }

        // saturation draw (rare)
        if (!anyEmpty()){
          const s = countActiveSwans(SUN), m = countActiveSwans(MOON);
          if (s > 0 && m > 0){
            playReady('draw', { gain: 0.5 });
            finishgame("Draw as no legal moves or placements remain.", "Draw.");
            return;
          }
        }

        // freeze bonus -> +1 action (same turn)
        const opponentLoss = (current===SUN)
          ? (res.frozeMoon + res.sealedMoon)
          : (res.frozeSun  + res.sealedSun);

        if (opponentLoss > 0) {

          // convert small integers to words
          const numberWord = {
            1: 'one',
            2: 'two',
            3: 'three',
            4: 'four',
            5: 'five',
            6: 'six'
          }[opponentLoss] || opponentLoss.toString();  // fallback for larger amounts

          movesLeft += opponentLoss;

          const gainedText =
            opponentLoss === 1
              ? 'one extra action'
              : `${numberWord} extra actions`;

          log(`${current === SUN ? 'Sun' : 'Moon'} gained ${gainedText} for freezing.`);

          // ensure the log jumps to the bottom after this message
          requestAnimationFrame(() => {
            if (elLog) elLog.scrollTop = elLog.scrollHeight;
          });
        }

        // spend one action
        movesLeft--;

        // keep turn if actions remain
        if (movesLeft > 0){
          render();
          aiturn();
          return;
        }

        // pass turn
        current = (current===SUN) ? MOON : SUN;
        beginTurn();
        render();
        aiturn();
      }

      function onSurrender(){
        if (appMode !== 'playing' || turn !== 'play' || !isAtTip()) return;

        const loserIsSun = (current === SUN);
        const h = humanSide(); // 'SUN' | 'MOON' | null

        // pick clip from the human perspective when ai is present.
        if (h) {
          const humanLoses = (h === 'SUN') ? loserIsSun : !loserIsSun;
          playReady(humanLoses ? 'loss' : 'win', { gain: 0.5, rate: 1.0 });
        } else {
          // local (human vs human) - play the surrendering side’s loss locally
          playReady('loss', { gain: 0.5, rate: 1.0 });
        }

        const loser  = loserIsSun ? 'Sun'  : 'Moon';
        const winner = loserIsSun ? 'Moon' : 'Sun';
        const icon   = loserIsSun ? '☾'    : '☼';
        finishgame(`${winner} ${icon} wins by resignation.`, `${winner} wins.`);
      }

      function setMuted(btn, muted){
        if (!btn) return;
        btn.classList.toggle('muted', muted);
        btn.disabled = !!muted;
        btn.setAttribute('aria-disabled', muted ? 'true' : 'false');
        btn.toggleAttribute('inert', muted);
        if (muted) btn.setAttribute('tabindex', '-1');
        else btn.removeAttribute('tabindex');
      }

      function updateActionButtonsEnabled(){
        const inGame   = (appMode !== 'menu');
        const atTip    = isAtTip();
        const canPlay  = (appMode === 'playing' && turn === 'play' && atTip);

        // Pure replay: mute everything except Hint, Reset and the Back/Forward arrows
        if (recite) {
          setMuted(btnUndo, true);
          setMuted(btnSurrender, true);
          setMuted(btnHint, false);
          setMuted(btnSave, false);
          // Back/Forward handled below by updateReviewButtons()
          setMuted(btnReset, false);
          // Load should be muted while in-game/replay
          setMuted(btnLoad, true);
          updateReviewButtons();
          return;
        }

        // undo - only at tip, during a playable turn, with history
        setMuted(btnUndo, aiThinking || !inGame || !atTip || history.length === 0 || turn !== 'play' || gameOver);

        // surrender - only at tip during a playable turn
        setMuted(btnSurrender, aiThinking || !canPlay || gameOver);

        // hint - enabled during any in-game snapshot (playing or review), including rewound
        setMuted(btnHint, aiThinking || !inGame || !atTip || history.length === 0 || turn !== 'play' || gameOver);

        // save - only enabled in review mode (after a game finished, before a new one starts)
        setMuted(btnSave, appMode !== 'review');

        // load - only available from the main menu
        setMuted(btnLoad, appMode !== 'menu');

        updateReviewButtons();
      }

    /* ===============================================================
       ui wiring (buttons) & bootstrap
       =============================================================== */

    btnUndo.addEventListener('click', ()=>{
      // Only allow undo at the live tip during an active game
      if (!isAtTip() || appMode === 'menu' || turn !== 'play') return;

      if (history.length <= 1) return; // nothing to undo beyond initial state

      resetVeryHardSession();

      // Pop snapshots until we remove exactly one move snapshot
      let removedMove = false;
      while (history.length > 1) {
        let last;
        try { last = JSON.parse(history[history.length - 1]); } catch { last = null; }
        history.pop();
        if (last && last.isMove) { removedMove = true; break; }
      }

      if (!removedMove) return; // guard: no change

      // Keep moveNumber aligned with the remaining history
      moveNumber = Math.max(0, moveNumber - 1);

      // Restore to the new tip state (last snapshot remaining)
      const newTip = history[history.length - 1];
      if (newTip) restoreState(newTip);

      // Clear transient UI state
      selected.clear();
      anchorIdx = null;
      action = null;
      splitIdx = null;

      // Rebuild replay to reflect the truncated history and stay at tip
      rebuildReplay();
      replayIdx = replay.length - 1;
      boardLocked = false;

      // Remove the corresponding last move line from the visual log, then log Undo
      removeLastMoveRowFromLog();
      log('Undo.');

      // Refresh UI
      render();
      updateActionButtonsEnabled();
      updateLogActive();
    });

    // button listeners //
    btnReset.addEventListener('click', reset);
    btnHint?.addEventListener('click', showHint);
    btnSurrender?.addEventListener('click', () => {
      if (!confirmBox) return;
      confirmBox.style.display = (confirmBox.style.display === 'none' || !confirmBox.style.display) ? 'block' : 'none';
    });
    btnBack?.addEventListener('click', reviewBack);
    btnForward?.addEventListener('click', reviewForward);

    // save/load wiring
    function downloadBlobAsFile(data, filename, type='application/json'){
      try {
        const blob = new Blob([data], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          URL.revokeObjectURL(url);
          a.remove();
        }, 0);
      } catch(e){
        console.error('Download failed', e);
        showMessageCenteredOnScreen?.('Could not start download.');
      }
    }

    function readVersionText(){
      const vEl = document.getElementById('version');
      if (!vEl) return null;
      const t = (vEl.textContent || '').trim();
      // extract last number-like token
      const m = t.match(/([0-9]+(?:\.[0-9]+)*)\s*$/);
      return m ? m[1] : t;
    }

    function saveGameRecord(){
      // Guard: only in review mode
      if (appMode !== 'review') return;

      // Prefer the built replay (exists in review), else fall back to history
      let frames = null;
      try { frames = (Array.isArray(replay) && replay.length ? replay.slice() : history.slice()); } catch { frames = null; }

      // Helper to parse a stored snapshot (strings are JSON; objects are passed through)
      const parseSnap = (s) => {
        try {
          if (!s) return null;
          if (typeof s === 'string') return JSON.parse(s);
          if (typeof s === 'object') return s;
          return null;
        } catch { return null; }
      };

      // First (New Game) and final states for convenience in the save file
      const firstState = Array.isArray(frames) && frames.length ? parseSnap(frames[0]) : null;
      const lastState  = Array.isArray(frames) && frames.length ? parseSnap(frames[frames.length - 1]) : null;

      // minimal metadata to enable future load & replay
      const meta = {
        kind: 'linith-game',
        version: readVersionText() || 'unknown',
        timestamp: new Date().toISOString(),
        boardSize: Array.isArray(board) ? board.length : 10,
        ai: {
          difficulty: document.getElementById('aiDifficulty')?.value ?? null,
          style: document.getElementById('aiStyle')?.value ?? null
        },
        clock: document.getElementById('clockMode')?.value ?? null
      };

      // Attach optional clock details so recite can display accurate times
      try {
        const clockVal = meta.clock || '';
        let clockInfo = null;
        if (clockVal === 'stopwatch') {
          // total elapsed microseconds at save time
          let us = 0;
          try {
            us = (typeof tAccUs === 'number' ? tAccUs : 0);
            if (typeof tStartPerf === 'number' && tStartPerf) {
              us += (performance.now() - tStartPerf) * 1000;
            }
          } catch {}
          clockInfo = { mode: 'stopwatch', stopwatchUs: Math.max(0, Math.floor(us)) };
        } else if (typeof clockVal === 'string' && clockVal.startsWith('chess-')) {
          // remaining microseconds for both sides at save time
          clockInfo = {
            mode: 'chess',
            preset: clockVal,
            sunUs:  (typeof chessSunUs  === 'number' ? Math.max(0, Math.floor(chessSunUs))  : null),
            moonUs: (typeof chessMoonUs === 'number' ? Math.max(0, Math.floor(chessMoonUs)) : null)
          };
        }
        if (clockInfo) meta.clockInfo = clockInfo;
      } catch {}

      const payload = {
        meta,
        // Explicitly include the New Game log line and the initial state snapshot
        newGame: {
          log: 'New Game.',
          state: firstState
        },
        // Full replay timeline for future load/replay
        replay: Array.isArray(frames) ? frames : [],
        outcome: {
          short: lastOutcomeShort,
          detailed: lastOutcomeDetailed,
          // Include the final state snapshot at game end
          state: lastState
        }
      };

      const json = JSON.stringify(payload);

      // build descriptive filename per request
      const slug = (s)=> String(s||'')
        .toLowerCase()
        .replace(/\s+/g,'-')
        .replace(/[^a-z0-9_-]/g,'')
        .replace(/-+/g,'-')
        .replace(/^-|-$|_/g, (m)=> m === '_' ? '_' : '');

      const styleSel = document.getElementById('aiStyle');
      const diffSel  = document.getElementById('aiDifficulty');
      const styleVal = slug(styleSel?.value || '');
      const diffVal  = slug(diffSel?.value || '');

      // Determine if this was a local (no AI) game
      let modePart = 'local';
      try {
        if (!isLocalOnly || !isLocalOnly()) {
          // AI game
          modePart = `${styleVal}_${diffVal || 'unknown'}`;
        }
      } catch {
        // Fallback if helper missing; infer from presence of selections
        if (styleVal || diffVal) modePart = `${styleVal || 'style'}_${diffVal || 'unknown'}`;
      }

      const ts = new Date();
      const pad = (n)=>String(n).padStart(2,'0');

      // Compute optional clock suffix to append after difficulty but before timestamp
      const clockVal = document.getElementById('clockMode')?.value || '';
      let clockSuffix = '';
      if (clockVal && clockVal !== 'off') {
        if (clockVal === 'stopwatch') {
          clockSuffix = '_stopwatch';
        } else if (clockVal.startsWith('chess-')) {
          // chess-N maps to Nminutes
          const mins = clockVal.split('-')[1] || '';
          clockSuffix = mins ? `_${mins}minutes` : '';
        } else {
          // fallback: sanitized value
          clockSuffix = `_${slug(clockVal)}`;
        }
      }

      // Convert Gregorian year to Atreyan Era (AE): 2025 -> 5AE, 2026 -> 6AE, etc.
      const gregYear = ts.getFullYear();
      const aeYear = gregYear - 2020; // 2020 => 0AE
      const MM = pad(ts.getMonth()+1);
      const DD = pad(ts.getDate());
      const HH = pad(ts.getHours());
      const mm = pad(ts.getMinutes());
      const SS = pad(ts.getSeconds());
      const timestamp = `${aeYear}AE${MM}${DD}_${HH}${mm}${SS}`;

      const fname = `linithgame_${modePart}${clockSuffix}_${timestamp}.json`;
      downloadBlobAsFile(json, fname);
      log('Saved game record.');
    }

    btnSave?.addEventListener('click', ()=>{
      if (btnSave?.classList.contains('muted')) return;
      saveGameRecord();
    });

    // ----- Load: import a saved game and enter pure replay mode -----
    function normaliseFramesToStrings(frames){
      if (!Array.isArray(frames)) return [];
      return frames.map((f)=>{
        if (typeof f === 'string') return f;
        try { return JSON.stringify(f); } catch { return null; }
      }).filter(Boolean);
    }

    function clearLog(){ if (elLog) elLog.innerHTML = ''; }

    // helpers for reconstructing setup move text from two board snapshots
    function countTiles(board2d, pred){
      let n = 0; const N = board2d?.length||0; for(let r=0;r<N;r++){ const row = board2d[r]||[]; for(let c=0;c<row.length;c++){ if (pred(row[c])) n++; }} return n;
    }
    function findAddedAt(prev, curr, val){
      const N = Math.max(prev?.length||0, curr?.length||0);
      for(let r=0;r<N;r++){
        const pr = prev?.[r]||[]; const cr = curr?.[r]||[]; const M = Math.max(pr.length, cr.length);
        for(let c=0;c<M;c++){
          const a = pr?.[c]; const b = cr?.[c];
          if ((a===0 || a===undefined || a===null) && b===val){ return [r,c]; }
        }
      }
      return null;
    }
    function sideName(code){ return (code===1?'Sun':'Moon'); }

    function populateLogForImportedReplay(payload, fileName){
      try {
        clearLog();
        // Loaded filename header
        if (fileName && typeof fileName === 'string') {
          log(`Loaded ${fileName}.`);
        }
        // New Game line
        if (payload?.newGame?.log) log(payload.newGame.log);
        else log('New Game.');
        const frames = Array.isArray(payload?.replay) ? payload.replay : [];
        for (let i = 0; i < frames.length; i++){
          let snap, prev;
          try { snap = (typeof frames[i] === 'string') ? JSON.parse(frames[i]) : frames[i]; } catch { snap = null; }
          try { const p = (i>0?frames[i-1]:null); prev = (typeof p === 'string')? JSON.parse(p) : p; } catch { prev = null; }
          if (!snap || !snap.isMove) continue;

          // Prefer the exact in-game message if present
          if (snap.log && typeof snap.log === 'string') {
            log(snap.log);
            // Append any freeze notes that were recorded for this move
            if (Array.isArray(snap.freezeNotes)){
              for (const m of snap.freezeNotes){ if (m && typeof m === 'string') log(m); }
            }
            continue;
          }

          // Reconstruct setup placement lines (first two moves often lack `log`)
          const prevBoard = prev?.board, currBoard = snap.board;
          const addedSun  = findAddedAt(prevBoard, currBoard, 1);
          const addedMoon = findAddedAt(prevBoard, currBoard, 2);
          if (addedSun || addedMoon){
            const add = addedSun || addedMoon;
            const code = addedSun ? 1 : 2;
            const [r,c] = add;
            const prevCount = countTiles(prevBoard, v => v===code);
            if (code===1 && prevCount===0){
              log(`Sun placed their first Swan at ${tileAlg(r,c)}.`);
              if (Array.isArray(snap.freezeNotes)){
                for (const m of snap.freezeNotes){ if (m && typeof m === 'string') log(m); }
              }
              continue;
            }
            if (code===2 && prevCount===0){
              log(`Moon placed their second Swan at ${tileAlg(r,c)}.`);
              if (Array.isArray(snap.freezeNotes)){
                for (const m of snap.freezeNotes){ if (m && typeof m === 'string') log(m); }
              }
              continue;
            }
            // generic placement fallback with ordinal count
            try {
              const actorName = sideName(code);
              const placedCount = prevCount + 1; // count after this placement
              const ord = swanOrdinalWord(placedCount);
              log(`${actorName} placed their ${ord} Swan at ${tileAlg(r,c)}.`);
            } catch {
              // hard fallback if anything goes wrong
              log(`${sideName(code)} placed a Swan at ${tileAlg(r,c)}.`);
            }
            if (Array.isArray(snap.freezeNotes)){
              for (const m of snap.freezeNotes){ if (m && typeof m === 'string') log(m); }
            }
            continue;
          }

          // Final fallback if we cannot infer: keep a neutral line
          log('Move.');
          if (Array.isArray(snap.freezeNotes)){
            for (const m of snap.freezeNotes){ if (m && typeof m === 'string') log(m); }
          }
        }
        // Prefer the detailed outcome message if present; fall back to short
        if (payload?.outcome) {
          const outText = payload.outcome.detailed || payload.outcome.short;
          if (outText) log(outText);
        }
      } catch(e){
        console.warn('Populate log failed', e);
      }
    }

    function enterrecital(payload, fileName){
      resetVeryHardSession();
      try { stopMasterLoop(); } catch {}
      try { timerStop?.(); } catch {}

      // Build history/replay from payload
      const frames = normaliseFramesToStrings(payload?.replay || []);
      if (!frames.length){
        showMessageCenteredOnScreen?.('Load failed - no replay frames found.');
        return;
      }

      // Ensure first frame has a predetermined default clock value for timed modes
      try {
        const savedClock = payload?.meta?.clock || null;
        if (savedClock && frames[0]) {
          let obj0 = null;
          try { obj0 = JSON.parse(frames[0]); } catch { obj0 = null; }
          if (obj0 && (!obj0.clockAt || typeof obj0.clockAt !== 'object')) {
            if (savedClock === 'stopwatch') {
              obj0.clockAt = { mode: 'stopwatch', us: 0 };
            } else if (String(savedClock).startsWith('chess-')) {
              let baseUs = null;
              try {
                const mins = CHESS_PRESETS_MIN && CHESS_PRESETS_MIN[savedClock];
                if (typeof mins === 'number') baseUs = Math.max(0, Math.floor(mins * 60 * 1_000_000));
              } catch {}
              if (baseUs !== null) {
                obj0.clockAt = { mode: 'chess', preset: savedClock, sunUs: baseUs, moonUs: baseUs, who: 'both' };
              }
            }
            try { frames[0] = JSON.stringify(obj0); } catch {}
          }
        }
      } catch {}

      // Assign replay stacks
      history = frames.slice();
      replay = history.slice();
      replayIdx = 0;

      // Enter review-like mode without a current turn (needed so reviewApply applies per-frame clocks)
      appMode = 'review';
      recite = true;
      activeSide = null;
      markActive?.(null);
      disableBoardInteraction(); // prevent actual moves/placements

      // UI: hide start buttons and ensure the log is visible in replay mode
      if (elStartMenu) elStartMenu.style.display = 'none';
      if (elLog) elLog.style.display = 'block';

      // Apply saved clock mode transiently and infer AI side BEFORE applying the first frame
      try {
        const savedClock = payload?.meta?.clock || null;
        if (savedClock) {
          try {
            clockMode = savedClock; // transient for this session/replay
            applyClockVisibility?.();
            updateClockFacesVisibility?.();
          } catch {}
        }
        // Infer AI side from per-move clock data so we only show the human face in recite
        try {
          if (clockMode && String(clockMode).startsWith('chess')) {
            const framesArr = Array.isArray(replay) ? replay : [];
            let human = null; // 'SUN' | 'MOON'
            for (let i = 0; i < framesArr.length; i++) {
              let snap = null;
              try { snap = JSON.parse(framesArr[i]); } catch { snap = null; }
              const ca = snap && snap.clockAt;
              if (ca && ca.mode === 'chess' && ca.who && ca.who !== 'both') {
                if (ca.who === 'SUN' || ca.who === 'MOON') { human = ca.who; break; }
              }
            }
            if (human) {
              aiSide = (human === 'SUN') ? MOON : SUN; // AI is opposite of human
            } else {
              aiSide = null; // show both faces
            }
            updateClockFacesVisibility?.();
          } else {
            aiSide = null;
          }
        } catch {}
      } catch {}

      // Restore initial snapshot (move 0) AFTER mode/visibility so we can apply the injected default clockAt
      try { reviewApply(0); } catch {
        try { restoreState(replay[0]); render(); } catch {}
      }

      // Populate log with synthetic move lines and outcome (include filename header)
      populateLogForImportedReplay(payload, fileName);

      // Re-render to ensure turn pill/log/controls reflect replay state
      render();

      updateReviewButtons();
      updateActionButtonsEnabled();
      updateLogActive();

      // show replay controls and initialise defaults
      setReplayControlsVisible?.(true);
      setReplaySpeed?.(Number(inputReplaySpeed?.value)||1);
      updateReplayButtonIcon?.();
    }

    function promptAndLoadFile(){
      // Only allow loading from the main menu
      if (typeof appMode !== 'undefined' && appMode !== 'menu') {
        showMessageCenteredOnScreen?.('You can only load a game from the menu. Please Reset to return to the menu.');
        return;
      }

      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json,.linith.json,application/json';
      input.style.display = 'none';
      document.body.appendChild(input);
      input.addEventListener('change', async () => {
        const file = input.files && input.files[0];
        input.remove();
        if (!file) return;
        try {
          const text = await file.text();
          const data = JSON.parse(text);
          if (!data || (data.meta && data.meta.kind && data.meta.kind !== 'linith-game')){
            // allow even if kind missing; just warn when clearly different
            console.warn('Unexpected file kind; proceeding');
          }
          enterrecital(data, file.name);
        } catch(e){
          console.error('Load failed', e);
          showMessageCenteredOnScreen?.('Could not load that file.');
        }
      });
      input.click();
    }

    btnLoad?.addEventListener('click', promptAndLoadFile);

    // wire replay control events
    btnReplayPP?.addEventListener('click', ()=>{
      if (appMode !== 'review') return;
      toggleReplayAuto();
    });
    inputReplaySpeed?.addEventListener('input', (e)=>{
      const v = Number(e?.target?.value);
      setReplaySpeed(v);
    });
    // double-click to reset recital/replay speed to default (1.0x)
    inputReplaySpeed?.addEventListener('dblclick', ()=>{
      if (!inputReplaySpeed) return;
      inputReplaySpeed.value = '1';
      setReplaySpeed?.(1);
    });
    // init display text early
    if (typeof replaySpeedX !== 'undefined') {
      setReplaySpeed?.(Number(inputReplaySpeed?.value)||1);
    }


      // heuristic - find the board root once
      const boardRoot =
        document.getElementById('board') ||
        document.querySelector('[data-board],[data-role="board"],.board,.board-wrap,.grid');

      // helper - fast-forward to tip safely
      function goToTip() {
        // suppress intermediate SFX when walking forward
        suppressReviewSFX = true;
        let guard = 0;
        while (!isAtTip()) {
          reviewForward();            // existing forward stepper
          if (++guard > 512) break;   // safety guard for corrupted timelines
        }
        suppressReviewSFX = false;
        forceReviewSFXOnce = true;
        reviewApply(replayIdx);  // this already renders + syncs buttons/lock
        updateReviewButtons?.();
        updateActionButtonsEnabled?.();
        updateLogActive?.();
        render();                     // force visual sync
      }

      // treat any pointer down outside the board as exit review
            document.addEventListener('pointerdown', (e) => {
              // Only react to primary (left) button. Ignore right/middle clicks.
              if (e.button !== 0) return;
              // don’t trigger from menus, inputs, when already at tip, or in pure replay mode
              if (recite || appMode === 'menu' || isAtTip()) return;

              const t = e.target;
              // ignore clicks on inputs, buttons, selects, and popovers
              if (t.closest && t.closest('input, textarea, select, button, [role="button"], [data-popover], [data-dialog], #log, .log, .panel')) return;

              // if we can identify the board root, only act when the click is outside it.
              if (boardRoot) {
                if (t === boardRoot || boardRoot.contains(t)) return; // it was on the board; do nothing
                e.preventDefault();
                goToTip();
                return;
              }

              // fallback - if we couldn't detect the board element, any background click counts
              e.preventDefault();
              goToTip();
            }, true); // capture phase so overlays don't swallow it first


    // click-to-jump in the log: navigate replay to the clicked move
    elLog?.addEventListener('click', (e) => {
      try {
        // find the clicked row (only direct children of #log)
        const row = e.target?.closest?.('#log > div');
        if (!row || !elLog || !elLog.contains(row)) return;

        const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
        const txt = row.textContent || '';
        const t = norm(txt);

        // Build replay if missing so we can navigate
        if (!replay || !replay.length) rebuildReplay();
        if (!replay || !replay.length) return;

        // Special case: "New Game" goes to the very first snapshot
        if (t.startsWith('new game')) {
          appMode = (appMode === 'menu') ? 'playing' : appMode;
          appMode = 'review';
          replayIdx = 0;
          reviewApply(replayIdx);
          updateReviewButtons();
          updateActionButtonsEnabled();
          e.preventDefault();
          return;
        }

        // Only act on rows that represent actual moves
        if (!isMoveRowText(txt)) return;

        const allRows = Array.from(elLog.children);
        const moveRows = allRows.filter(r => isMoveRowText(r.textContent));
        const ordinal = moveRows.indexOf(row); // 0-based order among move rows
        if (ordinal < 0) return;

        let targetIdx = -1;

        // If this is the last move row, treat it as "jump to real tip"
        if (ordinal === moveRows.length - 1) {
          targetIdx = replay.length - 1;
        } else {
          // Map the Nth move row to the Nth snapshot with `isMove`
          let seen = -1;
          for (let i = 0; i < replay.length; i++){
            let snap;
            try { snap = JSON.parse(replay[i]); } catch { continue; }
            if (snap && snap.isMove){
              seen++;
              if (seen === ordinal){ targetIdx = i; break; }
            }
          }
        }

        if (targetIdx < 0) targetIdx = replay.length - 1; // fallback to tip

        appMode = (appMode === 'menu') ? 'playing' : appMode;
        appMode = 'review';
        replayIdx = targetIdx;
        reviewApply(replayIdx);   // will set appMode/boardLocked correctly
        updateReviewButtons();
        e.preventDefault();
      } catch {}
    });

    document.addEventListener('keydown', (e) => {
      // ignore repeat and typing fields
      if (e.repeat) return;
      const t = e.target;
      if (t && (t.closest?.('input, textarea, select, [contenteditable]'))) return;

      if (appMode === 'menu') return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        reviewBack();     // same as clicking the Back button
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        reviewForward();  // same as clicking the Forward button
      }
    });

    // delegate clicks for Yes/No so it works even if nodes render later
    panelRoot.addEventListener('click', (e) => {
      const t = e.target;
      if (!(t instanceof HTMLElement)) return;

      if (t.id === 'surrenderNo') {
        if (confirmBox) confirmBox.style.display = 'none';
        e.stopPropagation();
        e.preventDefault();
        return;
      }

      if (t.id === 'surrenderYes') {
        if (confirmBox) confirmBox.style.display = 'none';
        // call your existing surrender flow
        if (typeof onSurrender === 'function') onSurrender();
        e.stopPropagation();
        e.preventDefault();
      }
    });

    btnLocal?.addEventListener('click', ()=>{ aiSide = null;
    startgame();
    updateClockFacesVisibility();
    aisetup();
    });
    btnAiSun?.addEventListener('click', ()=>{ aiSide = SUN;
    startgame();
    updateClockFacesVisibility();
    aisetup();
    });
    btnAiMoon?.addEventListener('click', ()=>{ aiSide = MOON;
    startgame();
    updateClockFacesVisibility();
    aisetup();
    });

    updateActionButtonsEnabled();

    // initial page state menu visible, log hidden, render board & labels
    appMode = 'menu';
    if (elStartMenu) elStartMenu.style.display = 'grid';
    if (elLog) elLog.style.display = 'none';
    render();
    renderAxisLabels();
}
