import { chmod, readdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const iconsDir = resolve(rootDir, "build/icons");
const expectedIcons = [
  "32x32.png",
  "64x64.png",
  "128x128.png",
  "256x256.png",
  "512x512.png"
];
const availableIcons = new Set(await readdir(iconsDir));

for (const icon of expectedIcons) {
  if (!availableIcons.has(icon)) {
    throw new Error(`Missing required Linux icon: build/icons/${icon}`);
  }

  await chmod(resolve(iconsDir, icon), 0o644);
}

console.log(`Normalized ${expectedIcons.length} Linux icons to mode 0644.`);
