import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
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
    async (_match, href) => `<style>\n${await readFile(resolve(rendererDir, href), "utf8")}\n</style>`,
    "compiled stylesheet"
  );
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
