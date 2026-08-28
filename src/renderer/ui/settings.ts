import { SFX } from "../sound";

export function initSettings(): void {
  const root = document.documentElement;
  const gearButton = byId<HTMLButtonElement>("gear-btn");
  const panel = byId<HTMLElement>("gear-panel");
  const difficulty = byId<HTMLSelectElement>("aiDifficulty");
  const style = byId<HTMLSelectElement>("aiStyle");
  const clock = byId<HTMLSelectElement>("clockMode");
  const highlights = byId<HTMLSelectElement>("moveHighlightsMode");
  const volume = byId<HTMLInputElement>("sfxVolume");
  const volumeValue = byId<HTMLElement>("sfxVolumeVal");
  const grey = byId<HTMLInputElement>("emptyGreyRange");
  const grid = byId<HTMLInputElement>("gridLineRange");
  const greySwatch = byId<HTMLElement>("emptyGreySwatch");
  const gridSwatch = byId<HTMLElement>("gridLineSwatch");

  if (!panel || !grey || !grid || !greySwatch || !gridSwatch) {
    return;
  }

  gearButton?.addEventListener("click", (event) => {
    event.stopPropagation();
    panel.style.display = panel.style.display === "block" ? "none" : "block";
  });
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof Node && !panel.contains(target) && !gearButton?.contains(target)) {
      panel.style.display = "none";
    }
  });

  bindStoredSelect(difficulty, "linith_ai_difficulty", "medium", window.linithSetDifficulty);
  bindStoredSelect(style, "linith_ai_style", "doctrinal", window.linithSetStyle);
  bindStoredSelect(clock, "linith_clock_mode", "off", window.linithSetClockMode);
  bindStoredSelect(highlights, "linith_move_highlights", "on", window.linithSetMoveHighlights);

  if (volume && volumeValue) {
    const applyVolume = (): void => {
      const unitValue = Math.max(0, Math.min(1, Number(volume.value) / 100));
      SFX.setVolume(unitValue);
      volumeValue.textContent = String(Math.round(unitValue * 100));
    };
    volume.value = String(Math.round(SFX.getVolume() * 100));
    volumeValue.textContent = volume.value;
    volume.addEventListener("input", applyVolume);
    volume.addEventListener("change", applyVolume);
    volume.addEventListener("dblclick", () => {
      volume.value = "40";
      applyVolume();
    });
  }

  const defaults = {
    "--grey": readCssVariable(root, "--grey"),
    "--grid": readCssVariable(root, "--grid")
  };
  bindColourSlider(root, grey, greySwatch, "--grey", defaults["--grey"]);
  bindColourSlider(root, grid, gridSwatch, "--grid", defaults["--grid"]);
}

function bindStoredSelect(
  select: HTMLSelectElement | null,
  key: string,
  fallback: string,
  setter: ((value: string) => void) | undefined
): void {
  if (!select) {
    return;
  }

  const stored = localStorage.getItem(key);
  const value = stored && Array.from(select.options).some((option) => option.value === stored)
    ? stored
    : fallback;
  select.value = value;
  setter?.(value);
  select.addEventListener("change", () => setter?.(select.value || fallback));
}

function bindColourSlider(
  root: HTMLElement,
  input: HTMLInputElement,
  swatch: HTMLElement,
  variable: "--grey" | "--grid",
  defaultColour: string
): void {
  const setColour = (colour: string): void => {
    const value = luminance(parseColour(colour));
    input.value = String(value);
    swatch.style.background = grey(value);
    root.style.setProperty(variable, colour);
  };
  const apply = (): void => setColour(grey(Number(input.value)));

  setColour(defaultColour);
  input.addEventListener("input", apply);
  input.addEventListener("change", apply);
  input.addEventListener("dblclick", () => setColour(defaultColour));
}

function parseColour(colour: string): { r: number; g: number; b: number } {
  const rgb = colour.trim().match(/^rgb\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)$/i);
  if (rgb) {
    return { r: Number(rgb[1]), g: Number(rgb[2]), b: Number(rgb[3]) };
  }

  const hex = colour.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i)?.[1];
  if (!hex) {
    return { r: 0, g: 0, b: 0 };
  }
  const expanded = hex.length === 3 ? [...hex].map((part) => part + part).join("") : hex;
  const value = Number.parseInt(expanded, 16);
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
}

function luminance({ r, g, b }: { r: number; g: number; b: number }): number {
  return Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
}

function grey(value: number): string {
  return `rgb(${value}, ${value}, ${value})`;
}

function readCssVariable(root: HTMLElement, variable: string): string {
  return getComputedStyle(root).getPropertyValue(variable).trim();
}

function byId<T extends HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}
