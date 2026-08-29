import test from "node:test";
import assert from "node:assert/strict";
import {
  TUNABLE_EVALUATION_FEATURES,
  assessEvaluationTuning,
  serializeEvaluationTuningResult,
  splitEvaluationTuningSamples,
  tuneEvaluationWeights,
  type EvaluationFeatureVector,
  type EvaluationTuningSample
} from "../src/renderer/game/veryHard/evaluationTuner";

test("coordinate tuning is deterministic, bounded, and improves its labelled corpus", () => {
  const samples = syntheticCorpus();
  const first = tuneEvaluationWeights(samples, {
    algorithm: "coordinate",
    iterations: 320,
    coordinateStep: 0.3,
    regularization: 0,
    scoreScale: 100
  });
  const second = tuneEvaluationWeights([...samples].reverse(), {
    algorithm: "coordinate",
    iterations: 320,
    coordinateStep: 0.3,
    regularization: 0,
    scoreScale: 100
  });
  assert.ok(first.finalLoss < first.initialLoss);
  assert.equal(serializeEvaluationTuningResult(first), serializeEvaluationTuningResult(second));
  for (const feature of TUNABLE_EVALUATION_FEATURES) {
    assert.ok(first.weights[feature] >= first.bounds.minimum);
    assert.ok(first.weights[feature] <= first.bounds.maximum);
  }
});

test("seeded SPSA is reproducible without random or neural dependencies", () => {
  const options = {
    algorithm: "spsa" as const,
    iterations: 200,
    seed: 0x12345678,
    regularization: 0,
    scoreScale: 100,
    spsaLearningRate: 0.5
  };
  const first = tuneEvaluationWeights(syntheticCorpus(), options);
  const second = tuneEvaluationWeights(syntheticCorpus(), options);
  assert.deepEqual(first, second);
  assert.ok(Number.isFinite(first.finalLoss));
  assert.ok(first.finalLoss <= first.initialLoss);
});

test("tuner rejects invalid teacher labels and feature values", () => {
  const sample = syntheticCorpus()[0];
  assert.throws(() => tuneEvaluationWeights([{ ...sample, target: 2 }]), /outside/);
  assert.throws(() => tuneEvaluationWeights([{
    ...sample,
    features: { ...sample.features, pressure: Number.NaN }
  }]), /non-finite/);
});

test("held-out splitting is deterministic and never leaks a game group", () => {
  const samples = syntheticCorpus().flatMap((sample, index) => [
    { ...sample, id: `${sample.id}-sun`, groupId: `game-${index}` },
    { ...sample, id: `${sample.id}-moon`, groupId: `game-${index}`, target: -sample.target }
  ]);
  const first = splitEvaluationTuningSamples(samples, { seed: 91, holdoutFraction: 0.34 });
  const second = splitEvaluationTuningSamples([...samples].reverse(), { seed: 91, holdoutFraction: 0.34 });
  assert.deepEqual(first, second);
  assert.ok(first.training.length > 0);
  assert.ok(first.holdout.length > 0);
  assert.deepEqual(
    first.trainingGroups.filter((group) => first.holdoutGroups.includes(group)),
    []
  );
});

test("held-out assessment rejects a candidate that overfits training outcomes", () => {
  const training: EvaluationTuningSample[] = [
    { id: "train-a", groupId: "train-a", features: vector({ pressure: 80 }), target: 0.99 },
    { id: "train-b", groupId: "train-b", features: vector({ pressure: -80 }), target: -0.99 }
  ];
  const holdout: EvaluationTuningSample[] = [
    { id: "hold-a", groupId: "hold-a", features: vector({ pressure: 80 }), target: -0.99 },
    { id: "hold-b", groupId: "hold-b", features: vector({ pressure: -80 }), target: 0.99 }
  ];
  const candidate = tuneEvaluationWeights(training, {
    iterations: 160,
    coordinateStep: 0.3,
    regularization: 0,
    scoreScale: 100
  });
  const assessed = assessEvaluationTuning(candidate, {
    training,
    holdout,
    trainingGroups: ["train-a", "train-b"],
    holdoutGroups: ["hold-a", "hold-b"],
    seed: 7,
    holdoutFraction: 0.5
  });
  assert.equal(assessed.validation?.accepted, false);
  assert.ok(assessed.validation!.heldOutFinalLoss > assessed.validation!.heldOutInitialLoss);
});

function vector(overrides: Partial<EvaluationFeatureVector>): EvaluationFeatureVector {
  return Object.fromEntries(
    TUNABLE_EVALUATION_FEATURES.map((feature) => [feature, overrides[feature] ?? 0])
  ) as unknown as EvaluationFeatureVector;
}

function syntheticCorpus(): EvaluationTuningSample[] {
  // Teacher outcomes deliberately care much more about pressure than the
  // identity vector does, giving both optimizers a real, deterministic signal.
  return [
    { id: "a", features: vector({ pressure: 80, centre: -40 }), target: 0.9 },
    { id: "b", features: vector({ pressure: 60, centre: -50 }), target: 0.75 },
    { id: "c", features: vector({ pressure: -80, centre: 40 }), target: -0.9 },
    { id: "d", features: vector({ pressure: -60, centre: 50 }), target: -0.75 },
    { id: "e", features: vector({ pressure: 20, centre: -18 }), target: 0.25 },
    { id: "f", features: vector({ pressure: -20, centre: 18 }), target: -0.25 }
  ];
}
