declare const __APP_VERSION__: string;

declare module "*?worker&inline" {
  const InlineWorker: new () => Worker;
  export default InlineWorker;
}

declare module "*?url&inline" {
  const dataUrl: string;
  export default dataUrl;
}

interface LinithSoundApi {
  getVolume(): number;
  setVolume(value: number): void;
  play(name: LinithSoundName, options?: LinithSoundOptions): Promise<void>;
}

type LinithSoundName = "place" | "move1" | "moveMany" | "win" | "draw" | "loss";

interface LinithSoundOptions {
  gain?: number;
  rate?: number;
  pan?: number;
}

interface Window {
  __LINITH_SOUND_BASE64__?: Partial<Record<LinithSoundName, string>>;
  SFX?: LinithSoundApi;
  linithDesktop?: {
    appName: string;
    platform: string;
  };
  linithGetDifficulty?: () => string;
  linithSetDifficulty?: (difficulty: string) => void;
  linithGetStyle?: () => string;
  linithSetStyle?: (style: string) => void;
  linithGetClockMode?: () => string;
  linithSetClockMode?: (mode: string) => void;
  linithGetMoveHighlights?: () => boolean;
  linithSetMoveHighlights?: (mode: string) => void;
}
