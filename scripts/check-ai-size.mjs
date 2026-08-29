import { existsSync, readdirSync, statSync } from "node:fs";
import { extname, relative, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const limits = readLimits(process.argv.slice(2));
const wasmDirectory = resolve(root, "src", "renderer", "game", "veryHard", "native");
const rendererDirectory = resolve(root, "dist", "renderer");
const standaloneFile = resolve(root, "dist", "linith-standalone.html");
const releaseDirectory = resolve(root, "release");

const wasmFiles = filesUnder(wasmDirectory).filter((path) => extname(path) === ".wasm");
const wasmBytes = totalBytes(wasmFiles);
const rendererBytes = totalBytes(filesUnder(rendererDirectory));
const standaloneBytes = existsSync(standaloneFile) ? statSync(standaloneFile).size : null;
const releaseFiles = filesUnder(releaseDirectory).filter((path) => statSync(path).isFile());
const largestRelease = releaseFiles
  .map((path) => ({ path: relative(root, path), bytes: statSync(path).size }))
  .sort((left, right) => right.bytes - left.bytes)[0] ?? null;

const report = {
  kind: "linith-ai-size-gate",
  wasm: {
    bytes: wasmBytes,
    mebibytes: mib(wasmBytes),
    limitMebibytes: limits.maxWasmMiB,
    files: wasmFiles.map((path) => relative(root, path))
  },
  renderer: existsSync(rendererDirectory)
    ? { bytes: rendererBytes, mebibytes: mib(rendererBytes) }
    : null,
  standalone: standaloneBytes === null
    ? null
    : { bytes: standaloneBytes, mebibytes: mib(standaloneBytes) },
  largestRelease: largestRelease === null
    ? null
    : { ...largestRelease, gibibytes: gib(largestRelease.bytes), limitGibibytes: limits.maxAppGiB }
};

console.log(JSON.stringify(report, null, 2));

const failures = [];
if (wasmBytes > limits.maxWasmMiB * 1024 * 1024) {
  failures.push(`Very Hard Wasm is ${mib(wasmBytes)} MiB (limit ${limits.maxWasmMiB} MiB)`);
}
if (largestRelease && largestRelease.bytes > limits.maxAppGiB * 1024 * 1024 * 1024) {
  failures.push(`${largestRelease.path} is ${gib(largestRelease.bytes)} GiB (limit ${limits.maxAppGiB} GiB)`);
}
if (failures.length > 0) {
  console.error(`Size gate failed: ${failures.join("; ")}`);
  process.exitCode = 1;
}

function readLimits(args) {
  const values = new Map();
  for (const argument of args) {
    const match = /^--([^=]+)=(.+)$/.exec(argument);
    if (!match) throw new Error(`Unexpected argument: ${argument}`);
    values.set(match[1], match[2]);
  }
  return {
    maxWasmMiB: positiveNumber(values.get("max-wasm-mib") ?? "25", "max-wasm-mib"),
    maxAppGiB: positiveNumber(values.get("max-app-gib") ?? "4", "max-app-gib")
  };
}

function positiveNumber(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`--${name} must be positive`);
  return parsed;
}

function filesUnder(path) {
  if (!existsSync(path)) return [];
  const stat = statSync(path);
  if (stat.isFile()) return [path];
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) =>
    filesUnder(resolve(path, entry.name))
  );
}

function totalBytes(paths) {
  return paths.reduce((sum, path) => sum + statSync(path).size, 0);
}

function mib(bytes) {
  return Number((bytes / (1024 * 1024)).toFixed(3));
}

function gib(bytes) {
  return Number((bytes / (1024 * 1024 * 1024)).toFixed(3));
}
