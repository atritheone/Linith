import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  mergeTeacherCorpusShards,
  serializeTeacherCorpus,
  type TeacherCorpus
} from "../src/renderer/game/veryHard/teacherCorpus";

function argumentsNamed(name: string): string[] {
  const prefix = `--${name}=`;
  return process.argv.filter((value) => value.startsWith(prefix)).map((value) => value.slice(prefix.length));
}

const inputPaths = argumentsNamed("input");
const [outputPath] = argumentsNamed("output");
if (inputPaths.length === 0 || !outputPath) {
  throw new Error(
    "Usage: node --import tsx scripts/merge-very-hard-teacher-corpus.ts "
    + "--input=shard-0.json --input=shard-1.json --output=teacher-corpus.json"
  );
}

const shards = await Promise.all(inputPaths.map(async (path) => (
  JSON.parse(await readFile(resolve(path), "utf8")) as TeacherCorpus
)));
const corpus = mergeTeacherCorpusShards(shards);
await writeFile(resolve(outputPath), serializeTeacherCorpus(corpus), "utf8");
process.stdout.write(
  `Merged ${inputPaths.length} teacher shards: ${corpus.samples.length} samples from `
  + `${corpus.completedGames}/${corpus.gamesAttempted} terminal games.\n`
);
