export type SoundName = LinithSoundName;
export type SoundOptions = LinithSoundOptions;

const soundFiles: Record<SoundName, string> = {
  place: "place.wav",
  move1: "move.wav",
  moveMany: "movemany.wav",
  win: "win.wav",
  draw: "draw.wav",
  loss: "lose.wav"
};

const DEFAULT_VOLUME = 0.4;

class SoundEngine implements LinithSoundApi {
  private context?: AudioContext;
  private readonly buffers = new Map<SoundName, AudioBuffer>();
  private masterGain?: GainNode;
  private outputGain?: GainNode;
  private compressor?: DynamicsCompressorNode;
  private connected = false;

  private getContext(): AudioContext {
    this.context ??= new AudioContext();
    return this.context;
  }

  private ensureMaster(): GainNode {
    const context = this.getContext();
    this.masterGain ??= context.createGain();
    this.outputGain ??= context.createGain();
    this.compressor ??= context.createDynamicsCompressor();

    const stored = Number.parseFloat(localStorage.getItem("linith_sfx_volume") ?? "");
    if (!this.connected) {
      this.masterGain.gain.value = Number.isFinite(stored) ? clamp(stored) : DEFAULT_VOLUME;
      this.outputGain.gain.value = 4.3;
      this.compressor.threshold.setValueAtTime(-8, context.currentTime);
      this.compressor.knee.setValueAtTime(6, context.currentTime);
      this.compressor.ratio.setValueAtTime(3, context.currentTime);
      this.compressor.attack.setValueAtTime(0.003, context.currentTime);
      this.compressor.release.setValueAtTime(0.2, context.currentTime);
      this.masterGain.connect(this.compressor);
      this.compressor.connect(this.outputGain);
      this.outputGain.connect(context.destination);
      this.connected = true;
    }

    return this.masterGain;
  }

  async load(name: SoundName): Promise<void> {
    const response = await fetch(resolveSoundUrl(name));
    if (!response.ok) {
      throw new Error(`Could not load ${name}: ${response.status} ${response.statusText}`);
    }

    const buffer = await this.getContext().decodeAudioData(await response.arrayBuffer());
    this.buffers.set(name, buffer);
  }

  setVolume(value: number): void {
    const context = this.getContext();
    const volume = clamp(value);
    this.ensureMaster().gain.setValueAtTime(volume, context.currentTime);
    localStorage.setItem("linith_sfx_volume", String(volume));
  }

  getVolume(): number {
    if (this.masterGain) {
      return this.masterGain.gain.value;
    }

    const stored = Number.parseFloat(localStorage.getItem("linith_sfx_volume") ?? "");
    return Number.isFinite(stored) ? clamp(stored) : DEFAULT_VOLUME;
  }

  async play(name: SoundName, options: SoundOptions = {}): Promise<void> {
    const buffer = this.buffers.get(name);
    if (!buffer) {
      return;
    }

    const context = this.getContext();
    this.ensureMaster();
    if (context.state !== "running") {
      await context.resume().catch(() => undefined);
    }
    if (context.state !== "running") {
      return;
    }

    const source = context.createBufferSource();
    const gain = context.createGain();
    const panner = context.createStereoPanner();
    source.buffer = buffer;
    source.playbackRate.value = options.rate ?? 1;
    gain.gain.value = options.gain ?? 0.85;
    panner.pan.value = options.pan ?? 0;
    source.connect(gain);
    gain.connect(panner);
    panner.connect(this.masterGain!);
    source.start();
  }
}

export const SFX = new SoundEngine();
window.SFX = SFX;

export const sfxReady = Promise.all(
  (Object.keys(soundFiles) as SoundName[]).map((name) => SFX.load(name))
).catch((error: unknown) => {
  console.error("[SFX] preload failed:", error);
});

export function playReady(name: SoundName, options?: SoundOptions): Promise<void> {
  return sfxReady.then(() => SFX.play(name, options));
}

function resolveSoundUrl(name: SoundName): string {
  const embedded = window.__LINITH_SOUND_BASE64__?.[name];
  return embedded ? `data:audio/wav;base64,${embedded}` : `./sound/${soundFiles[name]}`;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
