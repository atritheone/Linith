import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rendererDir = resolve(rootDir, "dist", "renderer");
const inputPath = resolve(rendererDir, "index.html");
const outputPath = resolve(rootDir, "dist", "linith-standalone.html");
const soundFiles = {
  place: "place.wav",
  move1: "move.wav",
  moveMany: "movemany.wav",
  win: "win.wav",
  draw: "draw.wav",
  loss: "lose.wav"
};

let html = await readFile(inputPath, "utf8");
html = await inlineStyles(html);
html = await inlineScripts(html);
html = await inlineFavicon(html);
html = await injectSounds(html);
html = authorizeInlineScripts(html);
html = html.replace(/\s+crossorigin(?:="[^"]*")?/g, "");

const externalReferences = Array.from(
  html.matchAll(/\b(?:src|href)=["'](?!data:|#)([^"']+)["']/gi),
  (match) => match[1]
);
if (externalReferences.length > 0) {
  throw new Error(`Standalone build still contains external assets: ${externalReferences.join(", ")}`);
}

// Vite can hide worker/Wasm URLs inside an inlined JavaScript string. Those
// are just as external as an HTML src attribute and make file:// standalone
// builds silently fall back to a weaker engine.
const hiddenBuildAssets = [
  ...html.matchAll(/\blinith-core-[A-Za-z0-9_-]+\.wasm\b/g),
  ...html.matchAll(/["'`]\/?assets\/[^"'`]+["'`]/g)
].map((match) => match[0]);
if (hiddenBuildAssets.length > 0) {
  throw new Error(
    `Standalone build still contains JavaScript-referenced assets: ${[...new Set(hiddenBuildAssets)].join(", ")}`
  );
}

await writeFile(outputPath, html, "utf8");
console.log(`Standalone file written to ${outputPath}`);

async function inlineStyles(source) {
  return replaceAsync(
    source,
    /<link\b(?=[^>]*\brel=["']stylesheet["'])(?=[^>]*\bhref=["']\.\/([^"']+)["'])[^>]*>/gi,
    async (_match, href) => {
      const stylesheetPath = resolve(rendererDir, href);
      const stylesheet = await readFile(stylesheetPath, "utf8");
      return `<style>\n${await inlineCssAssets(stylesheet, dirname(stylesheetPath))}\n</style>`;
    },
    "compiled stylesheet"
  );
}

async function inlineCssAssets(stylesheet, stylesheetDir) {
  return replaceAsync(
    stylesheet,
    /url\(\s*(["']?)(?!data:|https?:|#)([^"')]+)\1\s*\)/gi,
    async (_match, _quote, assetReference) => {
      const cleanReference = assetReference.split(/[?#]/, 1)[0];
      const assetPath = resolve(stylesheetDir, cleanReference);
      const base64 = (await readFile(assetPath)).toString("base64");
      return `url("data:${cssAssetMimeType(assetPath)};base64,${base64}")`;
    },
    "compiled stylesheet asset"
  );
}

function cssAssetMimeType(path) {
  switch (extname(path).toLowerCase()) {
    case ".ttf": return "font/ttf";
    case ".otf": return "font/otf";
    case ".woff": return "font/woff";
    case ".woff2": return "font/woff2";
    case ".png": return "image/png";
    case ".svg": return "image/svg+xml";
    default: return "application/octet-stream";
  }
}

async function inlineScripts(source) {
  return replaceAsync(
    source,
    /<script\b(?=[^>]*\btype=["']module["'])(?=[^>]*\bsrc=["']\.\/([^"']+)["'])[^>]*><\/script>/gi,
    async (_match, src) => {
      const script = await readFile(resolve(rendererDir, src), "utf8");
      return `<script type="module">\n${script.replaceAll("</script>", "<\\/script>")}\n</script>`;
    },
    "compiled module script"
  );
}

async function inlineFavicon(source) {
  const base64 = (await readFile(resolve(rootDir, "build", "favicon.png"))).toString("base64");
  const link = `<link rel="icon" type="image/png" href="data:image/png;base64,${base64}" />`;
  return replaceRequired(
    source,
    /<link\b[^>]*\brel=["'](?:shortcut icon|icon)["'][^>]*>/i,
    link,
    "favicon link"
  );
}

async function injectSounds(source) {
  const sounds = Object.fromEntries(
    await Promise.all(
      Object.entries(soundFiles).map(async ([name, file]) => [
        name,
        (await readFile(resolve(rootDir, "build", "sound", file))).toString("base64")
      ])
    )
  );
  const script = `<script>window.__LINITH_SOUND_BASE64__=${JSON.stringify(sounds)};</script>`;
  return replaceRequired(source, "</head>", `    ${script}\n  </head>`, "closing head tag");
}

function authorizeInlineScripts(source) {
  const hashes = Array.from(
    source.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi),
    (match) => `'sha256-${createHash("sha256").update(match[1], "utf8").digest("base64")}'`
  );
  if (hashes.length === 0) {
    throw new Error("Could not build standalone file: missing inline scripts to authorize.");
  }
  return replaceRequired(
    source,
    /script-src 'self'/,
    `script-src 'self' ${hashes.join(" ")}`,
    "script content security policy"
  );
}

async function replaceAsync(source, pattern, replacer, label) {
  const replacements = await Promise.all(Array.from(source.matchAll(pattern), (match) => replacer(...match)));
  if (replacements.length === 0) {
    throw new Error(`Could not build standalone file: missing ${label}.`);
  }
  let index = 0;
  return source.replace(pattern, () => replacements[index++]);
}

function replaceRequired(source, search, replacement, label) {
  if (!source.match(search)) {
    throw new Error(`Could not build standalone file: missing ${label}.`);
  }
  return source.replace(search, replacement);
}
