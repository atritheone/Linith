import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const output = resolve(root, "src/renderer/game/veryHard/native/linith-core.wasm");
mkdirSync(dirname(output), { recursive: true });

const bundledNpx = resolve(dirname(process.execPath), "node_modules/npm/bin/npx-cli.js");
const executable = existsSync(bundledNpx) ? process.execPath : process.platform === "win32" ? "npx.cmd" : "npx";
const prefixArgs = existsSync(bundledNpx) ? [bundledNpx] : [];
const result = spawnSync(
  executable,
  [...prefixArgs, "--yes", "--package", "assemblyscript@0.28.20", "asc", "linith-core.ts", "--config", "asconfig.json", "--target", "release"],
  { cwd: here, stdio: "inherit", shell: process.platform === "win32" && !existsSync(bundledNpx) }
);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
