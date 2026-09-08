import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { test } from "node:test";
import { createRequire } from "node:module";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

function loadHistory() {
  const source = fs.readFileSync(new URL("rlx-history.ts", import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports,
    process,
    Buffer,
    require: loadDependency,
  }, { filename: "rlx-history.ts" });
  return exports;
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function sha256(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "rlx-history-"));
  const runDirectory = path.join(root, "runs", "studio", "swing", "test");
  fs.mkdirSync(runDirectory, { recursive: true });
  const metadataFile = path.join(runDirectory, "swing.safetensors.json");
  const metricsFile = path.join(runDirectory, "training-metrics.jsonl");
  return {
    root,
    runDirectory,
    metadataFile,
    metricsFile,
    metadata(metadata) {
      fs.writeFileSync(metadataFile, JSON.stringify({ metadata }));
    },
    metrics(lines) {
      fs.writeFileSync(metricsFile, `${lines.join("\n")}\n`);
    },
    cleanup() {
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

const collection = (envSteps, meanReward) =>
  JSON.stringify({
    phase: "collection",
    steps: 10,
    seconds: 0.1,
    env_steps: envSteps,
    mean_reward: meanReward,
  });

const update = (envSteps, meanLoss) =>
  JSON.stringify({
    phase: "update",
    steps: 10,
    seconds: 0.1,
    env_steps: envSteps,
    mean_loss: meanLoss,
    policy_loss: meanLoss / 2,
    value_loss: meanLoss / 3,
    entropy: 0.4,
    approximate_kl: 0.02,
    clip_fraction: 0.1,
    explained_variance: 0.8,
  });

test("summary samples use actual steps without merging asynchronous metrics", () => {
  const f = fixture();
  try {
    f.metadata({ steps: 2048, ppo: { normalize_rewards: false } });
    f.metrics([
      JSON.stringify({
        phase: "episodes",
        env_steps: 528,
        mean_raw_return: 7,
        returns: [6, 8],
        lengths: [264, 264],
      }),
      collection(2048, 1.25),
      update(2048, 3),
    ]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segment = history.segments[0];

    assert.deepEqual(plain(segment.initialSample), {
      step: 528,
      meanReward: null,
      meanRawReturn: 7,
      meanLoss: null,
      policyLoss: null,
      valueLoss: null,
    });
    assert.deepEqual(plain(segment.finalSample), {
      step: 2048,
      meanReward: 1.25,
      meanRawReturn: null,
      meanLoss: 3,
      policyLoss: 1.5,
      valueLoss: 1,
    });
  } finally {
    f.cleanup();
  }
});

test("history survives process restart and keeps resumed normalization segments separate", () => {
  const f = fixture();
  try {
    const first = loadHistory();
    f.metadata({
      steps: 20,
      ppo: { normalize_rewards: false },
    });
    f.metrics([collection(10, 1), update(10, 3), collection(20, 2), update(20, 2)]);
    assert.equal(
      first.prepareTrainingHistory(
        f.root,
        f.runDirectory,
        f.metadataFile,
        f.metricsFile
      ),
      20
    );
    assert.equal(fs.readFileSync(f.metricsFile, "utf8"), "");

    f.metrics([collection(10, 0.25), update(10, 1.5)]);
    const active = first.collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      {
        id: "active-resume",
        startStep: 20,
        normalizeRewards: true,
        totalTimesteps: 40,
        status: "active",
      }
    );
    assert.deepEqual(
      plain(active.segments.filter((segment) => segment.kind === "ppo")
        .map((segment) => [
          segment.startStep,
          segment.endStep,
          segment.normalization,
          segment.status,
        ])),
      [
        [0, 20, "raw", "complete"],
        [20, 30, "normalized", "active"],
      ]
    );

    const restarted = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    assert.deepEqual(
      plain(restarted.segments.filter((segment) => segment.kind === "ppo")
        .map((segment) => [segment.startStep, segment.endStep, segment.status])),
      [[0, 20, "complete"], [20, 30, "complete"]]
    );
    assert.equal(restarted.segments[1].updates[0].meanLoss, 1.5);
    assert.equal(restarted.segments[1].initialSample.meanReward, 0.25);
    assert.equal(restarted.segments[1].finalSample.valueLoss, 0.5);
    const restored = loadHistory().restoreTrainingInvocation(
      f.runDirectory,
      f.metadataFile,
      restarted
    );
    assert.deepEqual(plain(restored.rewardHistory), [{ step: 10, reward: 0.25 }]);
    assert.equal(restored.trainingSteps, 10);
    assert.equal(restored.trainingTotal, 40);
    assert.equal(restored.normalizeRewards, true);
  } finally {
    f.cleanup();
  }
});

test("completed latest invocation is not duplicated on cold access", () => {
  const f = fixture();
  try {
    f.metadata({
      steps: 20,
      ppo: { normalize_rewards: false, total_timesteps: 20 },
    });
    f.metrics([collection(10, 1), update(10, 3), collection(20, 2)]);
    const active = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      {
        id: "completed-invocation",
        startStep: 0,
        normalizeRewards: false,
        totalTimesteps: 20,
        status: "complete",
      }
    );
    assert.equal(active.segments.filter((segment) => segment.kind === "ppo").length, 1);

    const cold = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segments = cold.segments.filter((segment) => segment.kind === "ppo");
    assert.equal(segments.length, 1);
    assert.equal(segments[0].id, "completed-invocation");
    assert.deepEqual(
      plain(segments[0].collections),
      [{ step: 10, meanReward: 1 }, { step: 20, meanReward: 2 }]
    );

    assert.equal(
      loadHistory().prepareTrainingHistory(
        f.root,
        f.runDirectory,
        f.metadataFile,
        f.metricsFile
      ),
      20
    );
    const archived = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    ).segments.filter((segment) => segment.kind === "ppo");
    assert.equal(archived.length, 1);
    assert.equal(archived[0].id, "completed-invocation");
    assert.equal(archived[0].collections.length, 2);
  } finally {
    f.cleanup();
  }
});

test("prepare preserves the journal when atomic history archival fails", () => {
  const f = fixture();
  try {
    f.metadata({ steps: 20, ppo: { normalize_rewards: false } });
    f.metrics([collection(10, 1), collection(20, 2)]);
    const originalJournal = fs.readFileSync(f.metricsFile, "utf8");
    fs.mkdirSync(path.join(f.runDirectory, "training-history.json.tmp"));

    assert.throws(() =>
      loadHistory().prepareTrainingHistory(
        f.root,
        f.runDirectory,
        f.metadataFile,
        f.metricsFile
      )
    );
    assert.equal(fs.readFileSync(f.metricsFile, "utf8"), originalJournal);
    assert.equal(
      fs.existsSync(path.join(f.runDirectory, "training-history.json")),
      false
    );
  } finally {
    f.cleanup();
  }
});

test("malformed and torn journal records are ignored without losing valid samples", () => {
  const f = fixture();
  try {
    f.metadata({ steps: 20, ppo: { normalize_rewards: false } });
    f.metrics([
      "{bad json",
      collection(10, 1),
      JSON.stringify({ phase: "collection", env_steps: "ten", mean_reward: 99 }),
      update(10, 3),
      '{"phase":"collection","env_steps":20,"mean_reward":',
      collection(20, 2),
    ]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segment = history.segments[0];
    assert.deepEqual(plain(segment.collections.map((sample) => sample.meanReward)), [1, 2]);
    assert.equal(segment.updates.length, 1);
    assert.equal(segment.endStep, 20);
  } finally {
    f.cleanup();
  }
});

test("step resets create explicit segments instead of connecting scales", () => {
  const f = fixture();
  try {
    f.metadata({ steps: 30, ppo: { normalize_rewards: true } });
    f.metrics([
      collection(10, 1),
      update(10, 2),
      collection(20, 2),
      collection(5, 10),
      update(5, 4),
      collection(10, 11),
    ]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    assert.deepEqual(
      plain(history.segments.map((segment) => [
        segment.startStep,
        segment.endStep,
        segment.normalizationLabel,
      ])),
      [
        [0, 20, "Normalized rollout reward"],
        [20, 30, "Normalized rollout reward"],
      ]
    );
  } finally {
    f.cleanup();
  }
});

test("trusted parent metadata contributes cumulative lineage and BC initializer", () => {
  const f = fixture();
  try {
    const parentDirectory = path.join(
      f.root,
      "runs",
      "studio",
      "swing",
      "parent"
    );
    fs.mkdirSync(parentDirectory, { recursive: true });
    const checkpoint = path.join(parentDirectory, "swing.safetensors");
    const sidecar = `${checkpoint}.json`;
    const parentMetadata = {
      recipe: "swing",
      steps: 20,
      ppo: { normalize_rewards: true },
      initialization: {
        loaded_metadata: {
          teacher_assisted: true,
          bootstrap: "BC + DAgger",
          teacher_samples: 38400,
          dagger_iteration: 3,
          normalizer: "frozen teacher observations",
        },
      },
    };
    fs.writeFileSync(checkpoint, "checkpoint");
    fs.writeFileSync(sidecar, JSON.stringify({ metadata: parentMetadata }));
    fs.writeFileSync(
      path.join(parentDirectory, "training-metrics.jsonl"),
      `${collection(20, 0.5)}\n`
    );
    f.metadata({
      steps: 30,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: checkpoint,
        source_sha256: sha256(checkpoint),
        source_sidecar_sha256: sha256(sidecar),
        loaded_metadata: parentMetadata,
      },
    });
    f.metrics([collection(10, 5)]);

    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    assert.equal(history.segments[0].kind, "bc-initializer");
    assert.equal(history.segments[0].initializer.teacherSamples, 38400);
    assert.deepEqual(
      plain(history.segments.filter((segment) => segment.kind === "ppo")
        .map((segment) => [
          segment.startStep,
          segment.endStep,
          segment.normalization,
        ])),
      [[0, 20, "normalized"], [20, 30, "raw"]]
    );
  } finally {
    f.cleanup();
  }
});

test("self-copied checkpoint resolves a unique exact-hash sibling journal", () => {
  const f = fixture();
  try {
    const siblingDirectory = path.join(
      path.dirname(f.runDirectory),
      "earlier-run"
    );
    fs.mkdirSync(siblingDirectory);
    const siblingCheckpoint = path.join(siblingDirectory, "saved-base.bin");
    const siblingSidecar = `${siblingCheckpoint}.json`;
    const parentMetadata = {
      steps: 20,
      ppo: { normalize_rewards: true },
    };
    fs.writeFileSync(siblingCheckpoint, "exact copied checkpoint");
    fs.writeFileSync(
      siblingSidecar,
      JSON.stringify({ metadata: parentMetadata })
    );
    fs.writeFileSync(
      path.join(siblingDirectory, "training-metrics.jsonl"),
      `${collection(10, 1)}\n${update(10, 3)}\n${collection(20, 2)}\n`
    );

    const selfCheckpoint = path.join(f.runDirectory, "swing.safetensors");
    fs.writeFileSync(selfCheckpoint, "continued checkpoint replaced the copy");
    f.metadata({
      steps: 30,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: selfCheckpoint,
        source_sha256: sha256(siblingCheckpoint),
        source_sidecar_sha256: sha256(siblingSidecar),
        loaded_metadata: parentMetadata,
      },
    });
    f.metrics([collection(10, 5)]);

    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segments = history.segments.filter((segment) => segment.kind === "ppo");
    assert.deepEqual(
      plain(segments.map((segment) => [
        segment.startStep,
        segment.endStep,
        segment.normalization,
        segment.collections.map((sample) => sample.meanReward),
      ])),
      [
        [0, 20, "normalized", [1, 2]],
        [20, 30, "raw", [5]],
      ]
    );
  } finally {
    f.cleanup();
  }
});

test("self-copied checkpoint with wrong hashes keeps an explicit unsampled gap", () => {
  const f = fixture();
  try {
    const siblingDirectory = path.join(
      path.dirname(f.runDirectory),
      "unrelated-run"
    );
    fs.mkdirSync(siblingDirectory);
    const siblingCheckpoint = path.join(siblingDirectory, "candidate.bin");
    fs.writeFileSync(siblingCheckpoint, "different checkpoint");
    fs.writeFileSync(
      `${siblingCheckpoint}.json`,
      JSON.stringify({
        metadata: { steps: 20, ppo: { normalize_rewards: true } },
      })
    );
    fs.writeFileSync(
      path.join(siblingDirectory, "training-metrics.jsonl"),
      `${collection(20, 99)}\n`
    );

    const selfCheckpoint = path.join(f.runDirectory, "swing.safetensors");
    fs.writeFileSync(selfCheckpoint, "continued checkpoint");
    f.metadata({
      steps: 30,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: selfCheckpoint,
        source_sha256: "0".repeat(64),
        source_sidecar_sha256: "1".repeat(64),
        loaded_metadata: {
          steps: 20,
          ppo: { normalize_rewards: true },
        },
      },
    });
    f.metrics([collection(10, 5)]);

    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segments = history.segments.filter((segment) => segment.kind === "ppo");
    assert.deepEqual(
      plain(segments.map((segment) => [
        segment.startStep,
        segment.endStep,
        segment.collections.length,
      ])),
      [[0, 20, 0], [20, 30, 1]]
    );
  } finally {
    f.cleanup();
  }
});

test("lineage never follows metadata paths outside the RLX root", () => {
  const f = fixture();
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "rlx-history-outside-"));
  try {
    const checkpoint = path.join(outside, "secret.safetensors");
    const sidecar = `${checkpoint}.json`;
    fs.writeFileSync(checkpoint, "secret");
    fs.writeFileSync(
      sidecar,
      JSON.stringify({
        metadata: { steps: 999, ppo: { normalize_rewards: true } },
      })
    );
    const hash = createHash("sha256")
      .update(fs.readFileSync(sidecar))
      .digest("hex");
    f.metadata({
      steps: 10,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: checkpoint,
        source_sha256: sha256(checkpoint),
        source_sidecar_sha256: hash,
        loaded_metadata: { steps: 999 },
      },
    });
    f.metrics([collection(10, 1)]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    assert.deepEqual(
      plain(history.segments.filter((segment) => segment.kind === "ppo")
        .map((segment) => [segment.startStep, segment.endStep])),
      [[0, 10]]
    );
  } finally {
    f.cleanup();
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test("lineage rejects in-root symlinks that resolve outside the RLX root", () => {
  const f = fixture();
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "rlx-history-symlink-"));
  try {
    const outsideCheckpoint = path.join(outside, "secret.safetensors");
    const outsideSidecar = `${outsideCheckpoint}.json`;
    fs.writeFileSync(outsideCheckpoint, "secret");
    fs.writeFileSync(
      outsideSidecar,
      JSON.stringify({
        metadata: { steps: 999, ppo: { normalize_rewards: true } },
      })
    );
    const linkedCheckpoint = path.join(f.root, "linked.safetensors");
    fs.symlinkSync(outsideCheckpoint, linkedCheckpoint);
    fs.symlinkSync(outsideSidecar, `${linkedCheckpoint}.json`);
    const hash = createHash("sha256")
      .update(fs.readFileSync(outsideSidecar))
      .digest("hex");
    f.metadata({
      steps: 10,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: linkedCheckpoint,
        source_sha256: sha256(outsideCheckpoint),
        source_sidecar_sha256: hash,
        loaded_metadata: { steps: 999 },
      },
    });
    f.metrics([collection(10, 1)]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    assert.deepEqual(
      plain(history.segments.filter((segment) => segment.kind === "ppo")
        .map((segment) => [segment.startStep, segment.endStep])),
      [[0, 10]]
    );
  } finally {
    f.cleanup();
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test("same-run resume metadata preserves an explicit unsampled ancestor segment", () => {
  const f = fixture();
  try {
    const checkpoint = path.join(f.runDirectory, "swing.safetensors");
    fs.writeFileSync(checkpoint, "current checkpoint");
    f.metadata({
      steps: 30,
      ppo: { normalize_rewards: false },
      initialization: {
        kind: "checkpoint",
        source_checkpoint: checkpoint,
        source_sidecar_sha256: "the prior sidecar was replaced by this resume",
        loaded_metadata: {
          steps: 20,
          ppo: { normalize_rewards: true },
        },
      },
    });
    f.metrics([collection(10, 5)]);
    const history = loadHistory().collectTrainingHistory(
      f.root,
      f.runDirectory,
      f.metadataFile,
      f.metricsFile,
      null
    );
    const segments = history.segments.filter((segment) => segment.kind === "ppo");
    assert.deepEqual(
      plain(segments.map((segment) => [
        segment.startStep,
        segment.endStep,
        segment.normalization,
        segment.collections.length,
      ])),
      [[0, 20, "normalized", 0], [20, 30, "raw", 1]]
    );
  } finally {
    f.cleanup();
  }
});
