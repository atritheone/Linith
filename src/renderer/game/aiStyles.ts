export interface AiStyleWeights {
  wFreeze?: number;
  wSelfFreeze?: number;
  wMyLib?: number;
  wOpLib?: number;
  wRing?: number;
  wSpace?: number;
}

export const AI_STYLES: Record<string, AiStyleWeights> = {
  doctrinal: {},
  constrictor: { wFreeze: 500, wSelfFreeze: -600, wMyLib: 2, wOpLib: -11, wRing: 1.4, wSpace: 1.5 },
  rupture: { wFreeze: 420, wSelfFreeze: -650, wMyLib: 2, wOpLib: -7, wRing: 0.6, wSpace: 0.8 },
  blizzard: { wFreeze: 600, wSelfFreeze: -450, wMyLib: 1, wOpLib: -5, wRing: 0.4, wSpace: 0.5 },
  librarian: { wFreeze: 380, wSelfFreeze: -700, wMyLib: 5, wOpLib: -10, wRing: 0.3, wSpace: 1.2 },
  swarm: { wFreeze: 520, wSelfFreeze: -620, wMyLib: 2, wOpLib: -8, wRing: 0.8, wSpace: 0.6 },
  fortress: { wFreeze: 480, wSelfFreeze: -680, wMyLib: 1, wOpLib: -6, wRing: 1.6, wSpace: 2 }
};
