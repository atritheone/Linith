import { cp, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const wrapperDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rootDir = resolve(wrapperDir, "..", "..");
const sourceDir = resolve(rootDir, "dist", "renderer");
const targetDir = resolve(wrapperDir, "www");
const androidAssetsDir = resolve(wrapperDir, "android-assets");

await assertRendererBuild();
await rm(targetDir, { recursive: true, force: true });
await mkdir(targetDir, { recursive: true });
await cp(sourceDir, targetDir, { recursive: true });
await cp(androidAssetsDir, resolve(targetDir, "android"), { recursive: true });
await injectAndroidShell(resolve(targetDir, "index.html"));

console.log(`Copied ${sourceDir} to ${targetDir}`);

async function assertRendererBuild() {
  try {
    if ((await stat(sourceDir)).isDirectory()) return;
  } catch {
    // Fall through to the explicit error.
  }
  throw new Error(`Missing renderer build at ${sourceDir}. Run "npm run build" from the repository root first.`);
}

async function injectAndroidShell(path) {
  let html = await readFile(path, "utf8");
  html = replaceRequired(
    html,
    /<meta\s+name="viewport"\s+content="[^"]*"\s*\/>/i,
    '<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />',
    "viewport metadata"
  );
  html = replaceRequired(
    html,
    "</head>",
    '    <link rel="stylesheet" href="./android/android-shell.css" />\n  </head>',
    "closing head tag"
  );
  html = replaceRequired(
    html,
    "</body>",
    '    <script type="module" src="./android/android-shell.js"></script>\n  </body>',
    "closing body tag"
  );
  await writeFile(path, html, "utf8");
}

function replaceRequired(source, search, replacement, label) {
  if (!source.match(search)) {
    throw new Error(`Could not inject Android shell: missing ${label}.`);
  }
  return source.replace(search, replacement);
}
