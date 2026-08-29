export const TUNABLE_EVALUATION_FEATURES = [
  "frozen",
  "activity",
  "liberties",
  "pressure",
  "territory",
  "development",
  "centre",
  "tempo"
] as const;

export type TunableEvaluationFeature = (typeof TUNABLE_EVALUATION_FEATURES)[number];
export type EvaluationMultipliers = Record<TunableEvaluationFeature, number>;

export interface EvaluationFeatureVector extends EvaluationMultipliers {}

export interface EvaluationTuningSample {
  id: string;
  /** Samples with the same group are never divided across train and holdout. */
  groupId?: string;
  features: EvaluationFeatureVector;
  /** Teacher value from the evaluated side's perspective, bounded to [-1, 1]. */
  target: number;
  weight?: number;
}

export interface EvaluationTuningOptions {
  algorithm?: "coordinate" | "spsa";
  seed?: number;
  iterations?: number;
  initial?: Partial<EvaluationMultipliers>;
  minimumMultiplier?: number;
  maximumMultiplier?: number;
  scoreScale?: number;
  regularization?: number;
  coordinateStep?: number;
  coordinateDecay?: number;
  spsaLearningRate?: number;
  spsaPerturbation?: number;
  /** Stable identifier for the terminal teacher corpus/engine. */
  teacherProvenance?: string;
  /** Exact engine artifact used to generate terminal labels. */
  engineFingerprint?: string;
}

export interface EvaluationTuningResult {
  format: "linith-evaluation-weights";
  version: 1;
  algorithm: "coordinate" | "spsa";
  seed: number;
  iterations: number;
  sampleCount: number;
  teacherProvenance: string | null;
  engineFingerprint: string | null;
  scoreScale: number;
  bounds: { minimum: number; maximum: number };
  initial: EvaluationMultipliers;
  weights: EvaluationMultipliers;
  initialLoss: number;
  finalLoss: number;
  validation?: EvaluationTuningValidation;
}

export interface EvaluationTuningSplit {
  training: EvaluationTuningSample[];
  holdout: EvaluationTuningSample[];
  trainingGroups: string[];
  holdoutGroups: string[];
  seed: number;
  holdoutFraction: number;
}

export interface EvaluationTuningValidation {
  splitSeed: number;
  holdoutFraction: number;
  trainingSampleCount: number;
  holdoutSampleCount: number;
  trainingGroupCount: number;
  holdoutGroupCount: number;
  heldOutInitialLoss: number;
  heldOutFinalLoss: number;
  minimumRelativeImprovement: number;
  relativeImprovement: number;
  accepted: boolean;
}

interface NormalizedOptions {
  algorithm: "coordinate" | "spsa";
  seed: number;
  iterations: number;
  initial: EvaluationMultipliers;
  minimumMultiplier: number;
  maximumMultiplier: number;
  scoreScale: number;
  regularization: number;
  coordinateStep: number;
  coordinateDecay: number;
  spsaLearningRate: number;
  spsaPerturbation: number;
}

export const IDENTITY_EVALUATION_MULTIPLIERS: Readonly<EvaluationMultipliers> = Object.freeze({
  frozen: 1,
  activity: 1,
  liberties: 1,
  pressure: 1,
  territory: 1,
  development: 1,
  centre: 1,
  tempo: 1
});

function finite(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function positiveInteger(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function normalizedOptions(options: EvaluationTuningOptions): NormalizedOptions {
  const minimumMultiplier = Math.max(0, finite(options.minimumMultiplier, 0.25));
  const maximumMultiplier = Math.max(minimumMultiplier, finite(options.maximumMultiplier, 2.5));
  const initial = Object.fromEntries(TUNABLE_EVALUATION_FEATURES.map((feature) => [
    feature,
    Math.max(minimumMultiplier, Math.min(
      maximumMultiplier,
      finite(options.initial?.[feature], IDENTITY_EVALUATION_MULTIPLIERS[feature])
    ))
  ])) as unknown as EvaluationMultipliers;
  return {
    algorithm: options.algorithm === "spsa" ? "spsa" : "coordinate",
    seed: (positiveInteger(options.seed, 0x51a71c) >>> 0),
    iterations: positiveInteger(options.iterations, 80),
    initial,
    minimumMultiplier,
    maximumMultiplier,
    scoreScale: Math.max(1, finite(options.scoreScale, 20_000)),
    regularization: Math.max(0, finite(options.regularization, 0.0005)),
    coordinateStep: Math.max(0.0001, finite(options.coordinateStep, 0.2)),
    coordinateDecay: Math.max(0.1, Math.min(0.999, finite(options.coordinateDecay, 0.94))),
    spsaLearningRate: Math.max(0.0001, finite(options.spsaLearningRate, 0.08)),
    spsaPerturbation: Math.max(0.0001, finite(options.spsaPerturbation, 0.12))
  };
}

function validateSamples(samples: readonly EvaluationTuningSample[]): EvaluationTuningSample[] {
  if (samples.length === 0) throw new Error("Evaluation tuning requires at least one labelled sample.");
  return [...samples]
    .sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0)
    .map((sample) => {
      if (!sample.id) throw new Error("Every evaluation sample requires a stable id.");
      if (sample.groupId !== undefined && sample.groupId.trim().length === 0) {
        throw new Error(`Sample ${sample.id} has an empty group id.`);
      }
      if (!Number.isFinite(sample.target) || sample.target < -1 || sample.target > 1) {
        throw new Error(`Sample ${sample.id} has a target outside [-1, 1].`);
      }
      const weight = sample.weight ?? 1;
      if (!Number.isFinite(weight) || weight <= 0) {
        throw new Error(`Sample ${sample.id} has a non-positive weight.`);
      }
      for (const feature of TUNABLE_EVALUATION_FEATURES) {
        if (!Number.isFinite(sample.features[feature])) {
          throw new Error(`Sample ${sample.id} has a non-finite ${feature} feature.`);
        }
      }
      return { ...sample, weight };
    });
}

function splitHash(groupId: string, seed: number): number {
  let hash = (0x811c9dc5 ^ seed) >>> 0;
  for (let index = 0; index < groupId.length; index += 1) {
    hash ^= groupId.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  hash ^= hash >>> 16;
  return Math.imul(hash, 0x85ebca6b) >>> 0;
}

/**
 * Produce a deterministic game-grouped split.  Ranking groups by a seeded
 * hash gives an exact, non-empty holdout size and prevents paired player views
 * (or several samples from one game) leaking into training.
 */
export function splitEvaluationTuningSamples(
  inputSamples: readonly EvaluationTuningSample[],
  options: { seed?: number; holdoutFraction?: number } = {}
): EvaluationTuningSplit {
  const samples = validateSamples(inputSamples);
  const seed = (positiveInteger(options.seed, 0x4f1d0a7) >>> 0);
  const requestedFraction = finite(options.holdoutFraction, 0.2);
  const holdoutFraction = Math.max(0.01, Math.min(0.5, requestedFraction));
  const groups = [...new Set(samples.map((sample) => sample.groupId ?? sample.id))];
  if (groups.length < 2) {
    throw new Error("Held-out evaluation requires samples from at least two independent groups.");
  }
  const ranked = groups.sort((left, right) => {
    const hashDifference = splitHash(left, seed) - splitHash(right, seed);
    return hashDifference || (left < right ? -1 : left > right ? 1 : 0);
  });
  const holdoutGroupCount = Math.max(1, Math.min(
    ranked.length - 1,
    Math.round(ranked.length * holdoutFraction)
  ));
  const holdoutGroupSet = new Set(ranked.slice(0, holdoutGroupCount));
  const training = samples.filter((sample) => !holdoutGroupSet.has(sample.groupId ?? sample.id));
  const holdout = samples.filter((sample) => holdoutGroupSet.has(sample.groupId ?? sample.id));
  const trainingGroups = ranked.slice(holdoutGroupCount).sort();
  const holdoutGroups = ranked.slice(0, holdoutGroupCount).sort();
  return { training, holdout, trainingGroups, holdoutGroups, seed, holdoutFraction };
}

export function assessEvaluationTuning(
  result: EvaluationTuningResult,
  split: EvaluationTuningSplit,
  minimumRelativeImprovement = 0
): EvaluationTuningResult {
  if (split.training.length === 0 || split.holdout.length === 0) {
    throw new Error("Evaluation assessment requires non-empty training and holdout sets.");
  }
  const minimum = Math.max(0, finite(minimumRelativeImprovement, 0));
  // No regularization in validation: the gate measures predictive error on
  // unseen terminal games, while regularization remains a training concern.
  const heldOutInitialLoss = evaluationTuningLoss(split.holdout, result.initial, result.scoreScale, 0);
  const heldOutFinalLoss = evaluationTuningLoss(split.holdout, result.weights, result.scoreScale, 0);
  const relativeImprovement = heldOutInitialLoss <= 0
    ? (heldOutFinalLoss < heldOutInitialLoss ? 1 : 0)
    : (heldOutInitialLoss - heldOutFinalLoss) / heldOutInitialLoss;
  return {
    ...result,
    validation: {
      splitSeed: split.seed,
      holdoutFraction: split.holdoutFraction,
      trainingSampleCount: split.training.length,
      holdoutSampleCount: split.holdout.length,
      trainingGroupCount: split.trainingGroups.length,
      holdoutGroupCount: split.holdoutGroups.length,
      heldOutInitialLoss,
      heldOutFinalLoss,
      minimumRelativeImprovement: minimum,
      relativeImprovement,
      accepted: result.finalLoss <= result.initialLoss
        && heldOutFinalLoss < heldOutInitialLoss
        && relativeImprovement >= minimum
    }
  };
}

function cloneWeights(weights: EvaluationMultipliers): EvaluationMultipliers {
  return { ...weights };
}

function clampWeights(weights: EvaluationMultipliers, options: NormalizedOptions): EvaluationMultipliers {
  for (const feature of TUNABLE_EVALUATION_FEATURES) {
    weights[feature] = Math.max(
      options.minimumMultiplier,
      Math.min(options.maximumMultiplier, weights[feature])
    );
  }
  return weights;
}

export function evaluationPrediction(
  features: EvaluationFeatureVector,
  weights: EvaluationMultipliers,
  scoreScale = 20_000
): number {
  let score = 0;
  for (const feature of TUNABLE_EVALUATION_FEATURES) score += features[feature] * weights[feature];
  return Math.tanh(score / Math.max(1, scoreScale));
}

export function evaluationTuningLoss(
  samples: readonly EvaluationTuningSample[],
  weights: EvaluationMultipliers,
  scoreScale = 20_000,
  regularization = 0.0005
): number {
  let error = 0;
  let totalWeight = 0;
  for (const sample of samples) {
    const sampleWeight = sample.weight ?? 1;
    const difference = evaluationPrediction(sample.features, weights, scoreScale) - sample.target;
    error += difference * difference * sampleWeight;
    totalWeight += sampleWeight;
  }
  let penalty = 0;
  for (const feature of TUNABLE_EVALUATION_FEATURES) {
    const offset = weights[feature] - 1;
    penalty += offset * offset;
  }
  return error / Math.max(1, totalWeight) + regularization * penalty;
}

function tuneCoordinate(
  samples: readonly EvaluationTuningSample[],
  options: NormalizedOptions
): EvaluationMultipliers {
  const weights = cloneWeights(options.initial);
  let bestLoss = evaluationTuningLoss(samples, weights, options.scoreScale, options.regularization);
  let step = options.coordinateStep;
  for (let iteration = 0; iteration < options.iterations; iteration += 1) {
    const feature = TUNABLE_EVALUATION_FEATURES[iteration % TUNABLE_EVALUATION_FEATURES.length];
    const original = weights[feature];
    let chosen = original;
    for (const candidate of [original - step, original + step]) {
      weights[feature] = Math.max(options.minimumMultiplier, Math.min(options.maximumMultiplier, candidate));
      const loss = evaluationTuningLoss(samples, weights, options.scoreScale, options.regularization);
      if (loss + 1e-15 < bestLoss) {
        bestLoss = loss;
        chosen = weights[feature];
      }
    }
    weights[feature] = chosen;
    if ((iteration + 1) % TUNABLE_EVALUATION_FEATURES.length === 0) step *= options.coordinateDecay;
  }
  return weights;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

function tuneSpsa(
  samples: readonly EvaluationTuningSample[],
  options: NormalizedOptions
): EvaluationMultipliers {
  const random = seededRandom(options.seed);
  const weights = cloneWeights(options.initial);
  let best = cloneWeights(weights);
  let bestLoss = evaluationTuningLoss(samples, best, options.scoreScale, options.regularization);
  for (let iteration = 0; iteration < options.iterations; iteration += 1) {
    const index = iteration + 1;
    const learningRate = options.spsaLearningRate / Math.pow(index + 10, 0.602);
    const perturbation = options.spsaPerturbation / Math.pow(index, 0.101);
    const deltas = Object.fromEntries(TUNABLE_EVALUATION_FEATURES.map((feature) => [
      feature,
      random() < 0.5 ? -1 : 1
    ])) as unknown as EvaluationMultipliers;
    const plus = cloneWeights(weights);
    const minus = cloneWeights(weights);
    for (const feature of TUNABLE_EVALUATION_FEATURES) {
      plus[feature] += perturbation * deltas[feature];
      minus[feature] -= perturbation * deltas[feature];
    }
    clampWeights(plus, options);
    clampWeights(minus, options);
    const difference = (
      evaluationTuningLoss(samples, plus, options.scoreScale, options.regularization)
      - evaluationTuningLoss(samples, minus, options.scoreScale, options.regularization)
    ) / (2 * perturbation);
    for (const feature of TUNABLE_EVALUATION_FEATURES) {
      weights[feature] -= learningRate * difference / deltas[feature];
    }
    clampWeights(weights, options);
    const loss = evaluationTuningLoss(samples, weights, options.scoreScale, options.regularization);
    if (loss < bestLoss) {
      bestLoss = loss;
      best = cloneWeights(weights);
    }
  }
  return best;
}

export function tuneEvaluationWeights(
  inputSamples: readonly EvaluationTuningSample[],
  inputOptions: EvaluationTuningOptions = {}
): EvaluationTuningResult {
  const samples = validateSamples(inputSamples);
  const options = normalizedOptions(inputOptions);
  const initialLoss = evaluationTuningLoss(
    samples,
    options.initial,
    options.scoreScale,
    options.regularization
  );
  const weights = options.algorithm === "spsa"
    ? tuneSpsa(samples, options)
    : tuneCoordinate(samples, options);
  const finalLoss = evaluationTuningLoss(samples, weights, options.scoreScale, options.regularization);
  return {
    format: "linith-evaluation-weights",
    version: 1,
    algorithm: options.algorithm,
    seed: options.seed,
    iterations: options.iterations,
    sampleCount: samples.length,
    teacherProvenance: inputOptions.teacherProvenance?.trim() || null,
    engineFingerprint: inputOptions.engineFingerprint?.trim() || null,
    scoreScale: options.scoreScale,
    bounds: { minimum: options.minimumMultiplier, maximum: options.maximumMultiplier },
    initial: cloneWeights(options.initial),
    weights: cloneWeights(weights),
    initialLoss,
    finalLoss
  };
}

export function serializeEvaluationTuningResult(result: EvaluationTuningResult): string {
  return `${JSON.stringify(result, null, 2)}\n`;
}
