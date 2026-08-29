import { SFX } from "../sound";
import { AI_STYLE_LIST, aiPersonality } from "../game/aiStyles";

export function initSettings(): void {
  const root = document.documentElement;
  const gearButton = byId<HTMLButtonElement>("gear-btn");
  const panel = byId<HTMLElement>("gear-panel");
  const closeButton = byId<HTMLButtonElement>("settingsClose");
  const difficulty = byId<HTMLSelectElement>("aiDifficulty");
  const style = byId<HTMLSelectElement>("aiStyle");
  const styleDescription = byId<HTMLElement>("aiStyleDescription");
  const clock = byId<HTMLSelectElement>("clockMode");
  const highlights = byId<HTMLSelectElement>("moveHighlightsMode");
  const volume = byId<HTMLInputElement>("sfxVolume");
  const volumeValue = byId<HTMLElement>("sfxVolumeVal");
  const grey = byId<HTMLInputElement>("emptyGreyRange");
  const grid = byId<HTMLInputElement>("gridLineRange");
  const greySwatch = byId<HTMLElement>("emptyGreySwatch");
  const gridSwatch = byId<HTMLElement>("gridLineSwatch");

  if (!panel) {
    return;
  }

  if (style) {
    style.replaceChildren(...AI_STYLE_LIST.map((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.label;
      option.title = profile.goal;
      return option;
    }));
  }

  const setPanelOpen = (open: boolean, restoreFocus = false): void => {
    panel.hidden = !open;
    gearButton?.setAttribute("aria-expanded", String(open));
    if (open) {
      closeButton?.focus();
    } else if (restoreFocus) {
      gearButton?.focus();
    }
  };

  gearButton?.addEventListener("click", (event) => {
    event.stopPropagation();
    setPanelOpen(panel.hidden);
  });
  closeButton?.addEventListener("click", () => setPanelOpen(false, true));
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof Node && !panel.contains(target) && !gearButton?.contains(target)) {
      setPanelOpen(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) {
      event.preventDefault();
      setPanelOpen(false, true);
    }
  });

  bindStoredSelect(difficulty, "linith_ai_difficulty", "hard", (value) => window.linithSetDifficulty?.(value));
  bindStoredSelect(style, "linith_ai_style", "doctrinal", (value) => window.linithSetStyle?.(value));
  if (style && styleDescription) {
    const describeStyle = (): void => {
      const profile = aiPersonality(style.value);
      styleDescription.textContent = profile.goal;
    };
    describeStyle();
    style.addEventListener("change", describeStyle);
  }
  bindStoredSelect(clock, "linith_clock_mode", "off", (value) => window.linithSetClockMode?.(value));
  bindStoredSelect(highlights, "linith_move_highlights", "on", (value) => window.linithSetMoveHighlights?.(value));

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

  if (grey && grid && greySwatch && gridSwatch) {
    const defaults = {
      "--grey": readCssVariable(root, "--grey"),
      "--grid": readCssVariable(root, "--grid")
    };
    bindColourSlider(root, grey, greySwatch, "--grey", defaults["--grey"], "linith_board_grey");
    bindColourSlider(root, grid, gridSwatch, "--grid", defaults["--grid"], "linith_board_grid");
  }
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
  defaultColour: string,
  storageKey: string
): void {
  const setColour = (colour: string, persist: boolean): void => {
    const value = luminance(parseColour(colour));
    input.value = String(value);
    swatch.style.background = grey(value);
    root.style.setProperty(variable, colour);
    if (persist) localStorage.setItem(storageKey, String(value));
  };
  const apply = (): void => setColour(grey(Number(input.value)), true);

  const stored = localStorage.getItem(storageKey);
  const storedValue = stored === null ? Number.NaN : Number(stored);
  const initialColour = Number.isFinite(storedValue) && storedValue >= 0 && storedValue <= 255
    ? grey(storedValue)
    : defaultColour;
  setColour(initialColour, false);
  input.addEventListener("input", apply);
  input.addEventListener("change", apply);
  input.addEventListener("dblclick", () => {
    localStorage.removeItem(storageKey);
    setColour(defaultColour, false);
  });
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
