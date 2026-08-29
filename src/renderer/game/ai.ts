// The action selector is isolated from the DOM so it can evolve independently
// from rendering and platform wrappers.
// @ts-nocheck -- incremental migration boundary for the inherited AI engine.
import { aiPersonality } from "./aiStyles";
import { computeFreezesOn } from "./encirclement";
import {
  countTotalSwans as countTotalSwansOn,
  simulatePush as simulatePushOn,
  simulateSwanMove as simulateSwanMoveOn
} from "./rulesEngine";

export function linithAI(board, current, difficulty = 'hard') {

      // Unknown persisted values historically fell into Easy. Normalize them at
      // the engine boundary so a typo or future value cannot silently weaken AI.
      difficulty = ['easy', 'medium', 'hard'].includes(difficulty) ? difficulty : 'medium';

      // ----- constants & helpers -----
      const SIZE = 10;
      const EMPTY = 0, SWAN_SUN = 1, SWAN_MOON = 2, STONE = 3, FROZEN_SUN = 4, FROZEN_MOON = 5;
      const SUN = 1, MOON = 2;
      const OPP = current === SUN ? MOON : SUN;
      const DIRS8 = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
      const DIRS4 = [[-1,0],[1,0],[0,-1],[0,1]];
      const PROFILE = aiPersonality(window.linithGetStyle?.() || 'doctrinal');
      const styleName = PROFILE.id;
      const DECISIVE_CHOKE = 3, DECISIVE_RING  = 3.0;

      const inb=(r,c)=>r>=0&&c>=0&&r<SIZE&&c<SIZE;
      const get=(b,r,c)=>b[r][c];
      const set=(b,r,c,v)=> (b[r][c]=v);
      const clone=b=>JSON.parse(JSON.stringify(b));
      const isEmpty=(b,r,c)=>get(b,r,c)===EMPTY;
      const isActiveSwan=(p,v)=> (p===SUN && v===SWAN_SUN) || (p===MOON && v===SWAN_MOON);
      const samePlayerSwan=(p,v)=> (p===SUN && (v===SWAN_SUN||v===FROZEN_SUN)) || (p===MOON && (v===SWAN_MOON||v===FROZEN_MOON));
      const enemySwan=(p,v)=> (p===SUN && (v===SWAN_MOON||v===FROZEN_MOON)) || (p===MOON && (v===SWAN_SUN||v===FROZEN_SUN));
      const neigh8=(r,c)=>DIRS8.map(([dr,dc])=>[r+dr,c+dc]).filter(([nr,nc])=>inb(nr,nc));
      const neigh4=(r,c)=>DIRS4.map(([dr,dc])=>[r+dr,c+dc]).filter(([nr,nc])=>inb(nr,nc));
      const shuffled = arr => arr.map(v=>[Math.random(),v]).sort((a,b)=>a[0]-b[0]).map(x=>x[1]);

      function countActiveSwans(p,b){
        let n=0;
        for(let r=0;
        r<SIZE;
        r++)for(let c=0;
        c<SIZE;
        c++) if(isActiveSwan(p,get(b,r,c))) n++;
      return n; }

      const libCache = new WeakMap();

      function libertiesFor(p, b){
        let entry = libCache.get(b);
        if (!entry) {
          entry = { [SUN]: null, [MOON]: null };
          libCache.set(b, entry);
        }
        const cached = entry[p];
        if (cached !== null) return cached;

        const seen = new Set();
        for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
          const v = get(b, r, c);
          if (!isActiveSwan(p, v)) continue;
          for (const [nr, nc] of neigh8(r, c)) {
            if (isEmpty(b, nr, nc)) seen.add(`${nr},${nc}`);
          }
        }
        const val = seen.size;
        entry[p] = val;
        return val;
      }

      const rootMyLib  = libertiesFor(current, board);
      const rootOppLib = libertiesFor(OPP,     board);

      function bothAtSix(b){
        const t=p=>{let n=0; for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++) if(samePlayerSwan(p,get(b,r,c))) n++; return n;};
        return t(SUN)>=6 && t(MOON)>=6;
      }

      // ----- encirclement, evaluation, decisive stone -----
      function collectActiveGroups(b){ /* … same as your current … */
        const seen=Array.from({length:SIZE},()=>Array(SIZE).fill(false));
        const groups = { [SUN]:[], [MOON]:[] };
        for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){
          const v=get(b,r,c); if(!isActiveSwan(SUN,v) && !isActiveSwan(MOON,v)) continue;
          if(seen[r][c]) continue;
          const owner = isActiveSwan(SUN,v)?SUN:MOON;
          const q=[[r,c]], comp=[]; seen[r][c]=true;
          while(q.length){
            const [x,y]=q.pop(); comp.push([x,y]);
            for(const [nx,ny] of neigh8(x,y)){
              if(seen[nx][ny]) continue;
              if(isActiveSwan(owner, get(b,nx,ny))){ seen[nx][ny]=true; q.push([nx,ny]); }
            }
          }
          groups[owner].push(comp);
        }
        return groups;
      }

      function groupEncircled(b, comp, owner){
        const inside=new Set(comp.map(([r,c])=>`${r},${c}`));
        for(const [r,c] of comp){
          for(const [nr,nc] of neigh8(r,c)){
            if(inside.has(`${nr},${nc}`)) continue;
            const v=get(b,nr,nc);
            if(v===EMPTY) return false;
            if(samePlayerSwan(owner,v) && isActiveSwan(owner,v)) return false;
          }
        }
        return true;
      }

      function freezeEncircled(b){
        const resolved = computeFreezesOn(b);
        for (let r=0; r<SIZE; r++) for (let c=0; c<SIZE; c++) b[r][c] = resolved.nb[r][c];
        return resolved;
      }

      const ringCache = new WeakMap();

      function enemyRingPressure(b, player){
        let entry = ringCache.get(b);
        if (!entry) {
          entry = { [SUN]: null, [MOON]: null };
          ringCache.set(b, entry);
        }
        const cached = entry[player];
        if (cached !== null) return cached;

        const foe = player===SUN?MOON:SUN;
        const isEnemyActive = v => (foe===SUN && v===SWAN_SUN) || (foe===MOON && v===SWAN_MOON);
        const seen = Array.from({length:SIZE},()=>Array(SIZE).fill(false));
        const groups=[];
        for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){
          const v=get(b,r,c); if(!isEnemyActive(v) || seen[r][c]) continue;
          const Q=[[r,c]]; seen[r][c]=true; const comp=[];
          while(Q.length){
            const [x,y]=Q.pop(); comp.push([x,y]);
            for(const [dr,dc] of DIRS8){
              const nx=x+dr, ny=y+dc;
              if(!inb(nx,ny) || seen[nx][ny]) continue;
              if(isEnemyActive(get(b,nx,ny))){ seen[nx][ny]=true; Q.push([nx,ny]); }
            }
          }
          groups.push(comp);
        }
        let score=0;
        for(const comp of groups){
          const rim = new Set();
          for(const [r,c] of comp){
            for(const [dr,dc] of DIRS8){
              const nr=r+dr, nc=c+dc;
              if(!inb(nr,nc)) continue;
              if(get(b,nr,nc)===EMPTY) rim.add(`${nr},${nc}`);
            }
          }
          const k=rim.size;
          if(k<=6) score += (6-k)*1.0;
          if(k<=3) score += 2.0;
          if(k<=1) score += 4.0;
        }
        entry[player] = score;
        return score;
      }

      function stoneAdvancesGame(bBefore, bAfter, player){
        const nb = clone(bAfter);
        const res = freezeEncircled(nb);
        const frozeEnemy = (player===SUN ? (res.frozeMoon+res.sealedMoon) : (res.frozeSun +res.sealedSun));
        if (frozeEnemy > 0) return true;
        const myLibBefore  = libertiesFor(player, bBefore);
        const oppLibBefore = libertiesFor(player===SUN?MOON:SUN, bBefore);
        const myLibAfter   = libertiesFor(player, nb);
        const oppLibAfter  = libertiesFor(player===SUN?MOON:SUN, nb);
        if (oppLibAfter <= oppLibBefore - 1) return true;
        if (myLibAfter  >= myLibBefore  + 1) return true;
        const prBefore = enemyRingPressure(bBefore, player);
        const prAfter  = enemyRingPressure(nb, player);
        if (prAfter >= prBefore + 1.0) return true;
        return false;
      }

       // ----- territory helpers (space evaluation) -----
      function isEnemySwanNakedGlobal(b, r, c, player){
        const v = get(b, r, c);
        if (!enemySwan(player, v)) return false;
        for (const [dr8, dc8] of DIRS8){
          const nr = r + dr8, nc = c + dc8;
          if (!inb(nr, nc)) continue;
          if (get(b, nr, nc) === STONE) return false;
        }
        return true;
      }

      function isInNakedEnemyZoneGlobal(b, r, c, player){
        for (const [dr8, dc8] of DIRS8){
          const er = r + dr8, ec = c + dc8;
          if (!inb(er, ec)) continue;
          if (isEnemySwanNakedGlobal(b, er, ec, player)) return true;
        }
        return false;
      }

      function territoryAdvantage(b, current){
        const OPP = (current===SUN ? MOON : SUN);
        const INF   = 99;
        const MAXD  = 6;    // depth horizon in Swan moves
        const EDGE  = 10;    // strong territory bonus per safe tile
        const SCALE = 2;  // distance differential scaling

        function bfs(player){
          const dist = Array.from({length: SIZE}, () => Array(SIZE).fill(INF));
          const q = [];

          // seed: empty neighbours of active Swans for this player
          for (let r=0;r<SIZE;r++) for (let c=0;c<SIZE;c++){
            const v = get(b, r, c);
            if (!isActiveSwan(player, v)) continue;

            for (const [dr,dc] of DIRS8){
              const nr = r+dr, nc = c+dc;
              if (!inb(nr,nc)) continue;
              if (!isEmpty(b,nr,nc)) continue;
              if (isInNakedEnemyZoneGlobal(b, nr, nc, player)) continue;
              if (dist[nr][nc] > 1){
                dist[nr][nc] = 1;
                q.push([nr,nc]);
              }
            }
          }

          // flood-fill over empty tiles, respecting naked enemy zones
          while (q.length){
            const [r,c] = q.shift();
            const d = dist[r][c];
            if (d >= MAXD) continue; // don't expand too far

            for (const [dr,dc] of DIRS8){
              const nr = r+dr, nc = c+dc;
              if (!inb(nr,nc)) continue;
              if (!isEmpty(b,nr,nc)) continue;
              if (isInNakedEnemyZoneGlobal(b, nr, nc, player)) continue;
              if (dist[nr][nc] > d+1){
                dist[nr][nc] = d+1;
                q.push([nr,nc]);
              }
            }
          }
          return dist;
        }

        const dMe  = bfs(current);
        const dOpp = bfs(OPP);

        let score = 0;

        for (let r=0;r<SIZE;r++) for (let c=0;c<SIZE;c++){
          if (!isEmpty(b,r,c)) continue;

          let dm = dMe[r][c];
          let do_ = dOpp[r][c];

          const meFar  = (dm > MAXD);
          const oppFar = (do_ > MAXD);

          if (meFar && oppFar) continue; // nobody reaches soon

          // treat unreachable as MAXD+1 so it's valuable but not explosive
          if (dm > MAXD) dm = MAXD + 1;
          if (do_ > MAXD) do_ = MAXD + 1;

          if (dm <= MAXD && do_ === MAXD+1){
            // strong safe territory for current
            score += EDGE;
          } else if (do_ <= MAXD && dm === MAXD+1){
            // strong safe territory for opponent
            score -= EDGE;
          } else {
            // both can reach within horizon: compare distances
            const diff = do_ - dm; // >0 means we're closer
            score += diff * SCALE;
          }
        }

        return score;
      }

      function tileCount(b, tile){
        let count = 0;
        for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++) if(get(b,r,c)===tile) count++;
        return count;
      }

      function stoneContactFor(p, b){
        let count = 0;
        for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
          if(!isActiveSwan(p,get(b,r,c))) continue;
          for(const [nr,nc] of neigh8(r,c)) if(get(b,nr,nc)===STONE) count++;
        }
        return count;
      }

      function localMobilityFor(p, b){
        let count = 0;
        for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++){
          if(!isActiveSwan(p,get(b,r,c))) continue;
          for(const [nr,nc] of neigh8(r,c)) if(isEmpty(b,nr,nc)) count++;
        }
        return count;
      }

      const personalityFeatureCache = new WeakMap();
      function personalityFeatures(b){
        const cached = personalityFeatureCache.get(b);
        if(cached) return cached;
        const features = {
          groups: collectActiveGroups(b),
          active: {
            [SUN]: countActiveSwans(SUN,b),
            [MOON]: countActiveSwans(MOON,b)
          },
          contact: {
            [SUN]: stoneContactFor(SUN,b),
            [MOON]: stoneContactFor(MOON,b)
          },
          mobility: {
            [SUN]: localMobilityFor(SUN,b),
            [MOON]: localMobilityFor(MOON,b)
          },
          stones: tileCount(b,STONE)
        };
        personalityFeatureCache.set(b,features);
        return features;
      }

      const territoryCache = new WeakMap();
      function cachedTerritory(b,p){
        let cached = territoryCache.get(b);
        if(!cached){ cached = { [SUN]:null,[MOON]:null }; territoryCache.set(b,cached); }
        if(cached[p]===null) cached[p]=territoryAdvantage(b,p);
        return cached[p];
      }

      function evaluateStyled(bBefore, bAfter, myLibBeforeOverride = null, oppLibBeforeOverride = null, perspective = current){
        const perspectiveOpp = perspective===SUN ? MOON : SUN;
        const nb = clone(bAfter);
        const res = freezeEncircled(nb);
        const myActiveAfter  = countActiveSwans(perspective, nb);
        const oppActiveAfter = countActiveSwans(perspectiveOpp, nb);
        if (oppActiveAfter===0 && myActiveAfter>0) return +1e9;
        if (myActiveAfter===0 && oppActiveAfter>0) return -1e9;
        if (myActiveAfter===0 && oppActiveAfter===0) return 0;

        const myLibBefore  = (myLibBeforeOverride  ?? libertiesFor(perspective,    bBefore));
        const oppLibBefore = (oppLibBeforeOverride ?? libertiesFor(perspectiveOpp, bBefore));
        const myLibAfter   = libertiesFor(perspective,    nb);
        const oppLibAfter  = libertiesFor(perspectiveOpp, nb);

        const myΔ  = myLibAfter  - myLibBefore;
        const oppΔ = oppLibAfter - oppLibBefore;

        const frozeGain = (perspective===SUN ? (res.frozeMoon+res.sealedMoon) : (res.frozeSun +res.sealedSun));
        const selfLoss  = (perspective===SUN ? (res.frozeSun +res.sealedSun ) : (res.frozeMoon+res.sealedMoon));
        const ringBefore = enemyRingPressure(bBefore, perspective);
        const ring      = enemyRingPressure(nb, perspective);
        const momentum  = bothAtSix(bBefore) ? 10 : 0;

        const totFrozen = (res.frozeSun+res.sealedSun) + (res.frozeMoon+res.sealedMoon);
        const phase = Math.max(0, Math.min(1, totFrozen/6));
        const freezeBoost = 1 + 0.12*phase;
        const doctrinal = frozeGain * (500*freezeBoost)
             + selfLoss  * -600
             + myΔ       * 5
             + oppΔ      * -9
             + momentum  * frozeGain;

        // Doctrinal remains the frozen compatibility baseline. Personalities
        // only break strategically close decisions and cannot erase the core
        // tactical value of a freeze or the cost of losing an active Swan.
        if(styleName==='doctrinal') return doctrinal;

        const traits = PROFILE.traits;
        const beforeFeatures = personalityFeatures(bBefore);
        const afterFeatures = personalityFeatures(nb);
        const fragmentationΔ =
          (afterFeatures.groups[perspectiveOpp].length - beforeFeatures.groups[perspectiveOpp].length)
          - (afterFeatures.groups[perspective].length - beforeFeatures.groups[perspective].length);
        const developmentΔ =
          (afterFeatures.active[perspective] - beforeFeatures.active[perspective])
          - (afterFeatures.active[perspectiveOpp] - beforeFeatures.active[perspectiveOpp]);
        const contactBefore = beforeFeatures.contact[perspectiveOpp] - beforeFeatures.contact[perspective];
        const contactAfter = afterFeatures.contact[perspectiveOpp] - afterFeatures.contact[perspective];
        const mobilityBefore = beforeFeatures.mobility[perspective] - beforeFeatures.mobility[perspectiveOpp];
        const mobilityAfter = afterFeatures.mobility[perspective] - afterFeatures.mobility[perspectiveOpp];
        const territoryΔ = (traits.territory ?? 0) === 0
          ? 0
          : cachedTerritory(nb,perspective) - cachedTerritory(bBefore,perspective);
        const earlyStone = countTotalSwansOn(bBefore,perspective)<6 && afterFeatures.stones>beforeFeatures.stones ? 1 : 0;
        const controlledConcession = (frozeGain>0 || fragmentationΔ>0) ? Math.max(0,-myΔ) : 0;

        const personalityRaw =
          (traits.freezeUrgency ?? 0) * frozeGain * 40
          - (traits.selfPreservation ?? 0) * selfLoss * 50
          + (traits.libertyBalance ?? 0) * (myΔ-oppΔ) * 3
          + (traits.containment ?? 0) * (ring-ringBefore) * 4
          + (traits.territory ?? 0) * territoryΔ * 0.15
          + (traits.fragmentation ?? 0) * fragmentationΔ * 14
          + (traits.development ?? 0) * developmentΔ * 18
          + (traits.earlyStone ?? 0) * earlyStone * 12
          + (traits.structure ?? 0) * (contactAfter-contactBefore) * 2.5
          + (traits.mobility ?? 0) * (mobilityAfter-mobilityBefore) * 1.5
          + (traits.sacrificeTolerance ?? 0) * Math.min(6,controlledConcession) * 6;
        const personality = Math.max(-80,Math.min(80,personalityRaw));
        return doctrinal + personality;
      }

      function decisiveStone(b, r, c, me){
        const b2 = clone(b); set(b2,r,c,STONE);
        const nb = clone(b2);
        const res = freezeEncircled(nb);
        const frozeEnemy = (me===SUN ? (res.frozeMoon+res.sealedMoon) : (res.frozeSun+res.sealedSun));
        if (frozeEnemy > 0) return true;
        const opp = (me===SUN?MOON:SUN);
        const oppLibBefore = libertiesFor(opp, b);
        const oppLibAfter  = libertiesFor(opp, nb);
        if (oppLibAfter <= oppLibBefore - DECISIVE_CHOKE) return true;
        const prBefore = enemyRingPressure(b, me);
        const prAfter  = enemyRingPressure(nb, me);
        if (prAfter >= prBefore + DECISIVE_RING) return true;
        return false;
      }

      // ----- legal placements & movement -----
      function legalSwanPlacements(b,p){
        const out=[];
        for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){
          if(!isEmpty(b,r,c)) continue;
          let hasAdjMine=false, adjEnemy=false;
          for(const [nr,nc] of neigh4(r,c)){ const v=get(b,nr,nc); if(samePlayerSwan(p,v)) hasAdjMine=true; }
          for(const [nr,nc] of neigh8(r,c)){ const v=get(b,nr,nc); if(enemySwan(p,v)) { adjEnemy=true; break; } }
          if(hasAdjMine && !adjEnemy) out.push([r,c]);
        }
        return out;
      }

      function legalStonePlacements(b, player){
        const adjEnemy = [];
        const frontier = [];
        const all = [];

        for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
          if (!isEmpty(b, r, c)) continue;
          all.push([r, c]);

          let nearAny = false;
          for (const [dr, dc] of DIRS8) {
            const nr = r + dr, nc = c + dc;
            if (!inb(nr, nc)) continue;
            const v = get(b, nr, nc);
            if (v !== EMPTY) nearAny = true;
            if (enemySwan(player, v)) {           // only enemy swans
              adjEnemy.push([r, c]);
              nearAny = true;
              break;
            }
          }
          if (nearAny) frontier.push([r, c]);
        }

        // preferred - squares next to an enemy swan
        if (adjEnemy.length) return adjEnemy;

        // fallback - original heuristic (near anything) then global
        return frontier.length ? frontier : all;
      }

      function activeSwansOf(b,p){
        const arr=[]; for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){ const v=get(b,r,c); if(isActiveSwan(p,v)) arr.push([r,c]); } return arr;
      }

      function simulateMoveSubset(b, p, subset, dir) {
        const [dr, dc] = dir;
        const moving = new Set(subset.map(([r, c]) => `${r},${c}`));

        const isEnemySwanNakedLocal = (sr, sc) => {
          const v = get(b, sr, sc);
          if (!enemySwan(p, v)) return false;

          // If any of the 8 neighbours around this Swan is a Stone → not naked
          for (const [dr8, dc8] of DIRS8) {
            const rr = sr + dr8, cc = sc + dc8;
            if (!inb(rr, cc)) continue;
            if (get(b, rr, cc) === STONE) return false;
          }
          return true; // No stones around this swan → naked
        };

        const isInNakedEnemyZoneLocal = (r, c) => {
          for (const [dr8, dc8] of DIRS8) {
            const er = r + dr8, ec = c + dc8;
            if (!inb(er, ec)) continue;
            if (isEnemySwanNakedLocal(er, ec)) return true;
          }
          return false;
        };

        // --- Validate swan targets (NEW naked-zone check added) ---
        for (const [r, c] of subset) {
          const nr = r + dr, nc = c + dc;

          if (!inb(nr, nc)) return null;

          // NEW RULE: cannot enter naked enemy Swan zone
          if (isInNakedEnemyZoneLocal(nr, nc)) return null;

          // OLD collision logic
          const v = get(b, nr, nc);
          if (v !== EMPTY && !moving.has(`${nr},${nc}`)) return null;
        }

        // --- Collect stones that follow ---
        const stonesFrom = [];
        const stoneSeen = new Set();

        for (const [r, c] of subset) {
          for (const [sr, sc] of neigh8(r, c)) {
            if (!inb(sr, sc) || get(b, sr, sc) !== STONE) continue;

            const key = `${sr},${sc}`;
            if (stoneSeen.has(key)) continue;

            let shared = false;
            for (const [ar, ac] of neigh8(sr, sc)) {
              if (!inb(ar, ac)) continue;
              const vv = get(b, ar, ac);

              // shared with enemy Swan
              if (enemySwan(p, vv)) { shared = true; break; }

              // shared with unmoved friendly active Swan
              if (samePlayerSwan(p, vv) && isActiveSwan(p, vv) && !moving.has(`${ar},${ac}`)) {
                shared = true; break;
              }
            }
            if (!shared) {
              stoneSeen.add(key);
              stonesFrom.push([sr, sc]);
            }
          }
        }

        // --- Compute stone destinations ---
        const stonesTo = stonesFrom.map(([sr, sc]) => [sr + dr, sc + dc]);

        const toSet = new Set();
        for (const [tr, tc] of stonesTo) {
          if (!inb(tr, tc)) return null;

          const occ = get(b, tr, tc);
          const vac =
            occ === EMPTY ||
            moving.has(`${tr},${tc}`) ||
            stonesFrom.some(([sr, sc]) => sr === tr && sc === tc);

          if (!vac) return null;

          const tkey = `${tr},${tc}`;
          if (toSet.has(tkey)) return null;
          toSet.add(tkey);
        }

        // --- Apply move to cloned board ---
        const nb = clone(b);

        // clear swans
        for (const [r, c] of subset) set(nb, r, c, EMPTY);

        // clear moved stones
        for (const [sr, sc] of stonesFrom) set(nb, sr, sc, EMPTY);

        // place stones
        for (const [tr, tc] of stonesTo) set(nb, tr, tc, STONE);

        // place moved swans
        for (const [r, c] of subset) set(nb, r + dr, c + dc, p === SUN ? SWAN_SUN : SWAN_MOON);

        return nb;
      }

      function* allSwanSubsets(coords){
        const n=coords.length, total=(1<<n);
        for(let mask=1; mask<total; mask++){
          const subset=[]; for(let i=0;i<n;i++) if(mask&(1<<i)) subset.push(coords[i]);
          yield subset;
        }
      }

      // Simulate pushing a subset of ENEMY active swans by dir [dr,dc].
      // Returns a new board if legal, else null.
      function simulatePushSubset(b, p, subset, dir){
        const [dr,dc] = dir;
        // helper: must have at least one adjacent friendly active Swan to push
        function hasFriendlyPusher(r,c){
          for(const [nr,nc] of neigh8(r,c)){
            const v = get(b,nr,nc);
            if (isActiveSwan(p, v)) return true;
          }
          return false;
        }

        // All targets must be enemy active swans and have a friendly pusher
        for(const [r,c] of subset){
          const v = get(b,r,c);
          if (!enemySwan(p, v) || v===FROZEN_SUN || v===FROZEN_MOON) return null;
          if (!hasFriendlyPusher(r,c)) return null;
        }

        // Determine stones that would follow based on the pushed subset.
        // In stone-follow rules, the "moving side" is the side of the swans that move.
        // Here the moving swans are the opponent's, so movingSide = OPP.
        const movingSet = new Set(subset.map(([r,c])=> r*SIZE + c));
        const stonesFrom = new Set(); // keys "s:r,c"
        const stonesTo   = new Map(); // key -> [tr,tc]
        const stoneKey = (r,c)=>`s:${r},${c}`;

        const isVacantAfterMove = (nb, r,c)=>{
          if(!inb(r,c)) return false;
          const v = get(nb,r,c);
          if (v===EMPTY) return true;
          if (movingSet.has(r*SIZE + c)) return true; // enemy swan vacates
          if (stonesFrom.has(stoneKey(r,c))) return true; // stone vacates
          return false;
        };

        // collect following stones
        for (const [r,c] of subset){
          for (const [er,ec] of DIRS8){
            const sr = r+er, sc = c+ec;
            if (!inb(sr,sc) || get(b,sr,sc)!==STONE) continue;
            // shared adjacency?
            let shared=false;
            for (const [ar,ac] of DIRS8){
              const xr = sr+ar, xc = sc+ac;
              if (!inb(xr,xc)) continue;
              const vv = get(b,xr,xc);
              if (!vv) continue;
              // If adjacent to an enemy of moving side (i.e., our swan), shared
              if (isActiveSwan(p, vv) || samePlayerSwan(p, vv)) { shared=true; break; }
              // If adjacent to an unmoving ally of moving side (opponent), also shared
              if (enemySwan(p, vv) && (vv!==FROZEN_SUN && vv!==FROZEN_MOON) && !movingSet.has(xr*SIZE+xc)) { shared=true; break; }
            }
            if (shared) continue;
            const tr = sr+dr, tc = sc+dc;
            if (!inb(tr,tc)) return null; // off-board stone
            const sk = stoneKey(sr,sc);
            stonesFrom.add(sk);
            stonesTo.set(sk, [tr,tc]);
          }
        }

        // Validate stone targets
        const seenTargets = new Set();
        for (const [_,[tr,tc]] of stonesTo){
          if (!isVacantAfterMove(b, tr, tc)) return null;
          const key = `${tr},${tc}`; if (seenTargets.has(key)) return null; seenTargets.add(key);
        }

        // Validate pushed swan destinations
        const destSet = new Set();
        for (const [r,c] of subset){
          const nr=r+dr, nc=c+dc; if (!inb(nr,nc)) return null;
          const occ = get(b,nr,nc); const dkey = `${nr},${nc}`;
          if (occ===EMPTY){ if (destSet.has(dkey)) return null; destSet.add(dkey); continue; }
          if (occ===STONE){ const sk = stoneKey(nr,nc); if (!stonesTo.has(sk)) return null; const [tr,tc]=stonesTo.get(sk); if(!isVacantAfterMove(b,tr,tc)) return null; if(destSet.has(dkey)) return null; destSet.add(dkey); continue; }
          // If another enemy swan, it must itself be in the pushed subset
          if (movingSet.has(nr*SIZE+nc)){ if (destSet.has(dkey)) return null; destSet.add(dkey); continue; }
          return null;
        }

        // Apply to a clone
        const nb = clone(b);
        // clear original enemy swans
        for (const [r,c] of subset) set(nb,r,c,EMPTY);
        // move stones
        for (const sk of Array.from(stonesFrom)){
          const [sr,sc] = sk.slice(2).split(',').map(Number); set(nb,sr,sc,EMPTY);
        }
        for (const [_,[tr,tc]] of stonesTo) set(nb,tr,tc,STONE);
        // place enemy swans at destination
        for (const [r,c] of subset){
          const v = get(b,r,c);
          set(nb, r+dr, c+dc, v);
        }
        return nb;
      }

      // Use the corrected pure successor model shared by AI regression tests. The
      // inherited simulators above remain temporarily as readable provenance,
      // but all production call sites route through these corrected adapters.
      function simulateCorrectMoveSubset(b, p, subset, dir){
        const result = simulateSwanMoveOn(
          b,
          p,
          subset.map(([r,c]) => ({r,c})),
          dir
        );
        return result ? result.board : null;
      }

      function simulateCorrectPushSubset(b, p, subset, dir){
        const result = simulatePushOn(
          b,
          p,
          subset.map(([r,c]) => ({r,c})),
          dir
        );
        return result ? result.board : null;
      }

      // ===== capability profiles =====
      const CAP = (()=>{
        if (difficulty==='hard')   return { MAX_SUBSET: 99, LOCAL_R: 99, MAX_STONES: 999, BEAM: 999, PROBE: 0, MUST_TACTICS: true };
        if (difficulty==='medium') return { MAX_SUBSET: 2,  LOCAL_R: 3,  MAX_STONES: 18,  BEAM: 10,  PROBE: 3, MUST_TACTICS: true };
        return /* easy */          { MAX_SUBSET: 1,  LOCAL_R: 2,  MAX_STONES: 10,  BEAM: 6,   PROBE: 0, MUST_TACTICS: true };
      })();

      // ===== unconditional tactical pre-pass: take any freeze/seal-in-one by stone =====
      {
        const wins = [];
        for (const [r,c] of legalStonePlacements(board, current)){
          const b2 = clone(board); set(b2,r,c,STONE);
          // freezeEncircled mutates, so pass a clone
          const res = freezeEncircled(clone(b2));
          const enemySealed = (current===SUN) ? res.sealedMoon : res.sealedSun;
          const selfSealed = (current===SUN) ? res.sealedSun : res.sealedMoon;
          const enemyFrozen = (current===SUN) ? (res.frozeMoon+res.sealedMoon)
                                              : (res.frozeSun +res.sealedSun);
          if (enemyFrozen > 0 && !(enemySealed > 0 && selfSealed > 0)){
            // Heavily prioritize true seals, then general freezes; add eval as a tie-breaker
            const sc = (enemySealed>0 ? 1e9 : 1e7) + (enemyFrozen*1000) + evaluateStyled(board, b2);
            wins.push({ type:'stone', r, c, score: sc });
          }
        }
        if (wins.length){
          wins.sort((a,b)=>b.score-a.score);
          return wins[0];
        }
      }

      // ===== locality mask around our active groups =====
      function inLocality(r,c){
        if (CAP.LOCAL_R>=90) return true;
        const coords = activeSwansOf(board, current);
        for (const [sr,sc] of coords){
          if (Math.max(Math.abs(sr-r), Math.abs(sc-c)) <= CAP.LOCAL_R) return true;
        }
        return false;
      }

      // ===== candidate collection with capability limits =====
      const cands = [];
      const pushStone = (r,c,score)=> cands.push({ type:'stone', r, c, score });
      const pushSwan  = (r,c,score)=> cands.push({ type:'swan',  r, c, score });
      const pushMove  = (subset,dir,score)=> cands.push({ type:'move', dir, swans: subset.map(([r,c])=>({r,c})), score });
      const pushPush  = (subset,dir,score)=> cands.push({ type:'push', dir, swans: subset.map(([r,c])=>({r,c})), score });

      const myTotalSwans = (()=>{let n=0; for(let r=0;r<SIZE;r++)for(let c=0;c<SIZE;c++){ const v=get(board,r,c); if(samePlayerSwan(current,v)) n++; } return n;})();

      // stones - strict early rule, only advancing, locality, and cap
      const legalStones = shuffled(legalStonePlacements(board, current));
      const allStones = legalStones.filter(([r,c]) => inLocality(r,c));
      const advancing = [];
      for (const [r,c] of allStones){
        if (myTotalSwans < 6 && !decisiveStone(board, r, c, current)) continue;
        const b2 = clone(board); set(b2,r,c,STONE);
        if (stoneAdvancesGame(board, b2, current)) advancing.push([r,c]);
        if (advancing.length >= CAP.MAX_STONES) break;
      }
      for (const [r,c] of advancing){
        const b2=clone(board); set(b2,r,c,STONE);
        let sc = evaluateStyled(board, b2, rootMyLib, rootOppLib);
        sc += 0.6 * enemyRingPressure(b2, current);
        let prox = 0; for(const [dr,dc] of DIRS8){ const rr=r+dr, cc=c+dc; if(inb(rr,cc) && get(board,rr,cc)!==EMPTY) prox += 0.15; }
        pushStone(r,c, sc + prox);
      }

      // swan placements (under six), locality respected
      if(myTotalSwans<6){
        for(const [r,c] of shuffled(legalSwanPlacements(board, current))){
          if (!inLocality(r,c)) continue;
          const b2=clone(board); set(b2,r,c, current===SUN?SWAN_SUN:SWAN_MOON);
          pushSwan(r,c, evaluateStyled(board, b2));
        }
      }

      // moves with subset-size cap and locality - beginners won’t coordinate many
      const coords = activeSwansOf(board, current);
      if(coords.length){
        for(const subset of allSwanSubsets(coords)){
          if (subset.length > CAP.MAX_SUBSET) continue;
          // locality - at least one moving swan must be in local zone
          if (!subset.some(([r,c])=>inLocality(r,c))) continue;
          for(const dir of DIRS8){
            const b2 = simulateCorrectMoveSubset(board, current, subset, dir);
            if(!b2) continue;
            pushMove(subset, dir, evaluateStyled(board, b2));
          }
        }
      }

      // pushes: consider subsets of ENEMY active swans adjacent to us, with same locality gate
      const enemyCoords = activeSwansOf(board, OPP);
      if (enemyCoords.length){
        for (const subset of allSwanSubsets(enemyCoords)){
          if (subset.length > CAP.MAX_SUBSET) continue;
          // locality: at least one of the pushed swans must be near our activity
          if (!subset.some(([r,c])=>inLocality(r,c))) continue;
          for (const dir of DIRS8){
            const b2 = simulateCorrectPushSubset(board, current, subset, dir);
            if (!b2) continue;
            // light heuristic bonus: pushing usually creates tactical pressure
            let sc = evaluateStyled(board, b2);
            sc += 0.4 * enemyRingPressure(b2, current);
            pushPush(subset, dir, sc);
          }
        }
      }

      // no cands fallbacks
      if (!cands.length){
        if (myTotalSwans < 6){
          const sp = legalSwanPlacements(board, current).filter(([r,c])=>inLocality(r,c));
          if (sp.length){ const [r,c]=sp[0]; return { type:'swan', r, c, score:0 }; }
        }
        for (const [r,c] of coords){
          for (const dir of DIRS8){
            const b2 = simulateCorrectMoveSubset(board, current, [[r,c]], dir);
            if (b2) return { type:'move', dir, swans:[{r,c}], score:0 };
          }
        }
        if (legalStones.length){ const [r,c]=legalStones[0]; return { type:'stone', r, c, score:0 }; }
        return null;
      }

      // ===== tactical layer (all difficulties) - take wins, prevent losses =====
      function afterBoard(b, a, player){
        const nb = clone(b);
        if (a.type === 'stone') {
          set(nb, a.r, a.c, STONE);
          return nb;
        }
        if (a.type === 'swan') {
          set(nb, a.r, a.c, player === SUN ? SWAN_SUN : SWAN_MOON);
          return nb;
        }
        if (a.type === 'move') {
          return simulateCorrectMoveSubset(b, player, a.swans.map(s => [s.r, s.c]), a.dir);
        }
        if (a.type === 'push') {
          return simulateCorrectPushSubset(b, player, a.swans.map(s => [s.r, s.c]), a.dir);
        }
        return null;
      }

      function freezeDeltaForPlayer(bBefore, a, player){
        const b2 = afterBoard(bBefore, a, player);
        if (!b2) return 0;
        const nb = clone(b2);
        const res = freezeEncircled(nb);
        if (res.sealedSun > 0 && res.sealedMoon > 0) return 0;
        return (player === SUN
          ? (res.frozeMoon + res.sealedMoon)
          : (res.frozeSun + res.sealedSun));
      }

      const winningNow = cands.filter(a => freezeDeltaForPlayer(board, a, current) > 0);
      if (CAP.MUST_TACTICS && winningNow.length){
        // choose the highest-eval among winning actions
        return winningNow.sort((a,b)=>b.score-a.score)[0];
      }

      // prevent loss-in-one
      function oppHasFreezeInOne(b){
        // Fast exhaustive check for opponent stones that immediately freeze or seal us
        for (const [r,c] of legalStonePlacements(b, OPP)){
          const b2 = clone(b); set(b2,r,c,STONE);
          const res = freezeEncircled(clone(b2));
          const ourFrozen = (current===SUN) ? (res.frozeSun + res.sealedSun)
                                            : (res.frozeMoon + res.sealedMoon);
          if (ourFrozen > 0) return true;
        }
        // Fall back to a greedy sample of other opponent actions (moves/placements)
        const oppActs = generateGreedyCandidates(b, OPP).slice(0, 12);
        for (const oa of oppActs){
          const delta = freezeDeltaForPlayer(b, oa, OPP);
          if (delta > 0) return true;
        }
        return false;
      }

      function generateGreedyCandidates(b, player){
        const out=[];
        const mySwans = activeSwansOf(b, player);
        const push=(x)=>out.push(x);
        // stones
        for (const [r,c] of legalStonePlacements(b, player)){
          const b2=clone(b); set(b2,r,c,STONE);
          const sc = evaluateStyled(b, b2, null, null, player);
          push({type:'stone', r, c, score:sc});
        }
        // placements
        if (countTotalSwansOn(b, player)<6){
          for (const [r,c] of legalSwanPlacements(b, player)){
            const b2=clone(b); set(b2,r,c, player===SUN?SWAN_SUN:SWAN_MOON);
            push({type:'swan', r, c, score:evaluateStyled(b,b2,null,null,player)});
          }
        }
        // single moves
        for (const [r,c] of mySwans){
          for (const dir of DIRS8){
            const b2 = simulateCorrectMoveSubset(b, player, [[r,c]], dir);
            if(!b2) continue;
            push({type:'move', dir, swans:[{r,c}], score:evaluateStyled(b,b2,null,null,player)});
          }
        }
        // Pushes were generated by the main selector but omitted from reply
        // simulation. Include every immediately tactical push plus ordinary
        // single-Swan pushes so the response probe can see this action class.
        const pushed = activeSwansOf(b, player===SUN ? MOON : SUN);
        for (const subset of allSwanSubsets(pushed)){
          if (subset.length > CAP.MAX_SUBSET) continue;
          for (const dir of DIRS8){
            const b2 = simulateCorrectPushSubset(b, player, subset, dir);
            if (!b2) continue;
            const action = {type:'push', dir, swans:subset.map(([r,c])=>({r,c}))};
            const tactical = freezeDeltaForPlayer(b, action, player) > 0;
            if (!tactical && subset.length > 1) continue;
            push({...action, score:evaluateStyled(b,b2,null,null,player)});
          }
        }
        return out.sort((a,b)=>b.score-a.score);
      }

      const sorted = cands.sort((a,b)=> b.score - a.score);
      const epsilon = 0.25;
      const best = sorted[0];
      const bestNonStone = sorted.find(a => a.type!=='stone');
      const topBest = (best && best.type==='stone' && bestNonStone && (best.score - bestNonStone.score) <= epsilon) ? bestNonStone : best;

      const defenders = sorted.filter(a => {
        const b2 = afterBoard(board, a, current);
        if (!b2) return false;
        return !oppHasFreezeInOne(b2);
      });
      if (CAP.MUST_TACTICS && defenders.length && oppHasFreezeInOne(board)){
        return defenders.slice(0, CAP.BEAM).sort((a,b)=>b.score-a.score)[0];
      }

      // ===== selection per difficulty =====
      // Helper: avoid choosing a stone that makes our position worse if a non-stone with non-negative score exists
      function avoidSelfHarm(pick, pool){
        if (!pick) return pick;
        if (pick.type !== 'stone' || pick.score >= 0) return pick;
        const alt = pool.find(a => a.type !== 'stone' && a.score >= 0);
        return alt || pick;
      }

      // Additional guard: avoid any pick that newly gives the opponent a freeze-in-one,
      // when there exists a safe alternative in the current pool.
      function avoidCreatingTacticLoss(pick, pool){
        if (!pick) return pick;
        const nb = afterBoard(board, pick, current);
        if (nb && oppHasFreezeInOne(nb)){
          const safeAlt = pool.find(a => {
            const b2 = afterBoard(board, a, current);
            return b2 && !oppHasFreezeInOne(b2);
          });
          if (safeAlt) return safeAlt;
        }
        return pick;
      }

      if (difficulty==='hard'){
        return avoidCreatingTacticLoss(avoidSelfHarm(topBest, sorted), sorted);
      }

      if (difficulty==='medium'){
        const beam = sorted.slice(0, CAP.BEAM);
        let bestByProbe = null, bestProbeScore = -Infinity;
        for (let i=0; i<beam.length && i<CAP.PROBE; i++){
          const a = beam[i];
          const b2 = afterBoard(board, a, current);  // ✅ pass board + player
          if(!b2) continue;
          const oppActs = generateGreedyCandidates(b2, OPP);
          const oppBest = oppActs[0];
          const finalScore = oppBest ? (a.score - oppBest.score) : a.score;
          if (finalScore > bestProbeScore){ bestProbeScore = finalScore; bestByProbe = a; }
        }
        if (bestByProbe && Math.random()<0.70)
          return avoidCreatingTacticLoss(avoidSelfHarm(bestByProbe, beam), beam);
        return avoidCreatingTacticLoss(avoidSelfHarm(beam[0], beam), beam);
      }

      // easy — pure 1-ply, but with reduced complexity already enforced by capability caps
      // pick from a mid-lower beam so it plays to win but is clumsy
      const E_BEAM = sorted.slice(0, CAP.BEAM);
      // prefer non-stone / multi-swan (simple “looks good” for beginners)
      const pref = E_BEAM.filter(a => a.type!=='move' || (a.swans && a.swans.length>1));
      if (pref.length) return pref[Math.min(pref.length-1, Math.floor(pref.length*0.6))]; // around 60th percentile of the preferred beam
      return E_BEAM[Math.min(E_BEAM.length-1, Math.floor(E_BEAM.length*0.7))];
    }
