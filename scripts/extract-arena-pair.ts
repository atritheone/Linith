import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const raw = new Map<string, string>();
for (const argument of process.argv.slice(2)) {
  const equals = argument.indexOf("=");
  if (!argument.startsWith("--") || equals < 0) throw new Error(`Expected --name=value, received ${argument}`);
  raw.set(argument.slice(2, equals), argument.slice(equals + 1));
}
const reportPath = raw.get("report");
const pairId = raw.get("pair");
const outputPath = raw.get("output");
if (!reportPath || !pairId || !outputPath) {
  throw new Error("Usage: --report=arena-report.json --pair=p00001 --output=pair-manifest.json");
}

const report = JSON.parse(readFileSync(resolve(reportPath), "utf8")) as {
  manifest?: {
    format: string;
    version: number;
    seed: number;
    challenger: string;
    opponent: string;
    maxActions: number;
    repetitionCount: number;
    games: Array<{ id?: string; pairId: string }>;
  };
};
if (!report.manifest) throw new Error("The arena report does not contain a full manifest.");
const games = report.manifest.games.filter((game) => game.pairId === pairId);
if (games.length !== 2) throw new Error(`Expected exactly two games for ${pairId}; found ${games.length}.`);
const manifest = { ...report.manifest, games };
mkdirSync(dirname(resolve(outputPath)), { recursive: true });
writeFileSync(resolve(outputPath), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`Extracted ${pairId} (${games.map((game) => game.id).join(", ")}) to ${resolve(outputPath)}`);
