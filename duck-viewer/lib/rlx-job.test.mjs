import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { test } from "node:test";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

// Compile with the project's installed TypeScript; all subprocesses and artifact I/O are fakes.
function load(filename, dependencies = {}) {
  const source = fs.readFileSync(new URL(filename, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports, process, Buffer,
    require: (id) => id in dependencies ? dependencies[id] : loadDependency(id),
  }, { filename });
  return exports;
}
const experiments = load("experiments.ts");
const { evaluationVerdict } = load("evaluation.ts");

function fixture() {
  const files = new Map();
  const children = [];
  const reads = [];
  let writeImplementation;
  const api = load("rlx-job.ts", {
    "@/lib/experiments": experiments,
    "node:fs": {
      existsSync: (file) => file.endsWith("examples/ppo_microduck_dance.py") || files.has(file),
    },
    "node:fs/promises": {
      readFile: async (file) => {
        reads.push(file);
        if (!files.has(file)) throw new Error("ENOENT");
        return files.get(file);
      },
      writeFile: async (file, data) => {
        if (writeImplementation) return writeImplementation(file, data);
        files.set(file, data);
      },
    },
    "node:child_process": {
      spawn: (command, args, options) => {
        const child = new EventEmitter();
        Object.assign(child, {
          command, args, options,
          stdout: new EventEmitter(), stderr: new EventEmitter(),
          kill: (signal) => { child.killed = signal; return true; },
        });
        children.push(child);
        return child;
      },
    },
  });
  const artifact = (runName, kind = "checkpoint", experimentId = "swing") =>
    api.artifactPath(experimentId, runName, kind);
  return {
    api, files, children, artifact, reads,
    add: (runName, kind = "checkpoint", experimentId = "swing") => files.set(artifact(runName, kind, experimentId), "artifact"),
    deferWrites: (implementation) => { writeImplementation = implementation; },
  };
}
const input = (runName = "test-a", extra = {}) => ({ experimentId: "swing", runName, ...extra });
const arg = (child, flag) => child.args[child.args.indexOf(flag) + 1];
function sourceReport(f, runName = "test-a", sourceType = "policy") {
  const source = f.artifact(runName, sourceType === "policy" ? "onnx" : "checkpoint");
  const files = sourceType === "policy" ? [source] : [source, f.artifact(runName, "metadata")];
  const hashes = Object.fromEntries(files.map((file) => [file, createHash("sha256").update(f.files.get(file)).digest("hex")]));
  return {
    source_type: sourceType, source_sha256: hashes[source], source_files_sha256: hashes,
    evaluation_mode: "skill", skill_status: "passed", passed: true, pipeline_passed: true,
  };
}
function finish(child, report, code = 0) {
  if (report) child.stdout.emit("data", `${JSON.stringify(report)}\n`);
  child.emit("close", code, null);
}
function progress(child) {
  child.stdout.emit("data", '{"event":"training_progress","steps":4,"total":4,"mean_reward":2.5}\n');
}

test("export includes the CLI-required recipe and checkpoint/output without environment arguments", () => {
  const f = fixture();
  f.add("test-a", "onnx");
  assert.throws(() => f.api.startJob("export", input()), /Export requires a checkpoint/);
  assert.equal(f.children.length, 0);
  f.add("test-a");
  f.api.startJob("export", input());
  const child = f.children[0];
  assert.equal(child.command, "uv");
  assert.equal(arg(child, "--recipe"), "swing");
  assert.equal(child.args.includes("--num-envs"), false);
  assert.equal(child.args.includes("--weight-overrides"), false);
  assert.deepEqual(Array.from(child.args.slice(child.args.indexOf("export") + 1)), [
    "--recipe", "swing", "--checkpoint", f.artifact("test-a"), "--output", f.artifact("test-a", "onnx"),
  ]);
});

for (const profile of ["smoke", "full"]) {
  test(`${profile} Swing evaluation declares its scope, horizon, and explicit default target`, () => {
    const f = fixture();
    f.add("test-a");
    const recipe = f.api.startJob("eval", input("test-a", { profile }));
    const child = f.children[0];
    assert.equal(recipe.swingMinSpanDeg, 150);
    assert.equal(arg(child, "--evaluation-mode"), profile === "smoke" ? "pipeline" : "skill");
    assert.equal(arg(child, "--eval-steps"), profile === "smoke" ? "4" : "1200");
    assert.equal(arg(child, "--max-episode-s"), profile === "smoke" ? "1" : "24");
    assert.equal(arg(child, "--swing-min-span-deg"), "150");
    assert.equal(arg(child, "--checkpoint"), f.artifact("test-a"));
  });
}

test("evaluation selects ONNX when present and forwards an edited target", () => {
  const f = fixture();
  f.add("test-a", "onnx");
  f.api.startJob("eval", input("test-a", { swingMinSpanDeg: 160 }));
  assert.equal(arg(f.children[0], "--policy"), f.artifact("test-a", "onnx"));
  assert.equal(arg(f.children[0], "--swing-min-span-deg"), "160");
  assert.equal(f.api.normalizeRecipe(input("a", { swingMinSpanDeg: Infinity })).swingMinSpanDeg, 150);
});

test("non-Swing evaluation declares scope without Swing-specific criteria", () => {
  const f = fixture();
  f.add("test-a", "checkpoint", "dance");
  f.api.startJob("eval", input("test-a", { experimentId: "dance", profile: "full" }));
  assert.equal(arg(f.children[0], "--evaluation-mode"), "skill");
  assert.equal(f.children[0].args.includes("--swing-min-span-deg"), false);
});

test("render accepts ONNX-only runs while artifact-free runs cannot evaluate or render", () => {
  const f = fixture();
  assert.throws(() => f.api.startJob("eval", input()), /checkpoint before continuing/);
  assert.throws(() => f.api.startJob("render", input()), /checkpoint before continuing/);
  assert.equal(f.children.length, 0);
  f.add("test-a", "onnx");
  f.api.startJob("render", input());
  assert.equal(arg(f.children[0], "--policy"), f.artifact("test-a", "onnx"));
  assert.equal(f.children[0].args.includes("--evaluation-mode"), false);
});

test("selecting another run loads only that run's persisted evaluation", async () => {
  const f = fixture();
  f.add("test-a");
  f.add("test-b");
  const savedPath = path.join(path.dirname(f.artifact("test-b")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify({ marker: "selected-run", passed: false }));
  f.api.startJob("eval", input());
  finish(f.children[0], { marker: "previous-run", passed: true });
  f.api.startJob("render", input("test-b"));
  assert.equal((await f.api.snapshot()).evaluation.marker, "selected-run");
});

test("training keeps its PPO arguments and never receives evaluation-only flags", () => {
  const f = fixture();
  f.api.startJob("train", input());
  const child = f.children[0];
  assert.equal(arg(child, "--num-steps"), "2");
  assert.equal(arg(child, "--onnx-output"), f.artifact("test-a", "onnx"));
  assert.equal(child.args.includes("--evaluation-mode"), false);
  assert.equal(child.args.includes("--swing-min-span-deg"), false);
  assert.throws(() => f.api.startJob("train", input("test-b")), /already running/);
});

for (const operation of ["eval", "render"]) {
  for (const otherExperiment of [false, true]) {
    test(`${operation} clears old-run telemetry and evaluation on ${otherExperiment ? "experiment" : "name"} change`, async () => {
      const f = fixture();
      f.api.startJob("train", input());
      progress(f.children[0]);
      finish(f.children[0]);
      f.add("test-a");
      f.api.startJob("eval", input());
      finish(f.children[1], { passed: true, source_sha256: "old-source" });
      const next = otherExperiment ? input("test-a", { experimentId: "dance" }) : input("test-b");
      f.add(next.runName, "checkpoint", next.experimentId);
      f.api.startJob(operation, next);
      const snapshot = await f.api.snapshot();
      assert.equal(snapshot.trainingSteps, 0);
      assert.equal(snapshot.trainingTotal, 0);
      assert.equal(snapshot.rewardHistory.length, 0);
      assert.equal(snapshot.evaluation, null);
      assert.equal(snapshot.result, null);
    });
  }
}

test("same-run eval/render preserve training stats and persist settings with authoritative outcomes", async () => {
  const f = fixture();
  f.api.startJob("train", input());
  progress(f.children[0]);
  finish(f.children[0]);
  f.add("test-a");
  f.add("test-a", "onnx");
  const report = {
    ...sourceReport(f),
    evaluation_mode: "skill", passed: false, pipeline_passed: true, skill_status: "failed",
    finite: true,
    evaluation: { mode: "skill", steps_per_env: 1200, swing_criteria: { min_bidirectional_span_deg: 160 } },
    swing_assessment: { passed: false, failures: ["incomplete"] },
  };
  f.api.startJob("eval", input("test-a", { profile: "full", swingMinSpanDeg: 160 }));
  finish(f.children[1], report);
  const evaluated = await f.api.snapshot();
  assert.equal(evaluated.evaluation.passed, false);
  assert.equal(evaluated.evaluation.skill_status, "failed");
  assert.equal(evaluated.evaluation.source_sha256, report.source_sha256);
  assert.equal(evaluated.evaluation.evaluation.steps_per_env, 1200);
  assert.equal(evaluated.evaluation.evaluation_request.swing_min_span_deg, 160);
  assert.equal(evaluated.evaluation.evaluation_request.evaluation_mode, "skill");
  assert.equal(evaluated.evaluation.evaluation_request.recipe.profile, "full");
  const saved = JSON.parse(f.files.get(path.join(path.dirname(f.artifact("test-a")), "evaluation.json")));
  assert.equal(saved.source_sha256, report.source_sha256);
  assert.equal(saved.evaluation_request.eval_steps, 1200);
  f.api.startJob("render", input());
  finish(f.children[2], { rendered: true });
  const rendered = await f.api.snapshot();
  assert.equal(rendered.trainingSteps, 4);
  assert.equal(rendered.trainingTotal, 4);
  assert.equal(rendered.rewardHistory[0].reward, 2.5);
  assert.equal(rendered.evaluation.source_sha256, report.source_sha256);
  f.api.startJob("train", input());
  const restarted = await f.api.snapshot();
  assert.equal(restarted.trainingSteps, 0);
  assert.equal(restarted.rewardHistory.length, 0);
  assert.equal(restarted.evaluation, null);
});

for (const newRunName of ["test-a", "test-b"]) {
  test(`cancelled callbacks cannot mutate a replacement job (${newRunName})`, async () => {
    const f = fixture();
    f.api.startJob("train", input());
    const old = f.children[0];
    old.stdout.emit("data", "old partial");
    assert.equal(f.api.cancelJob(), true);
    assert.equal(old.killed, "SIGTERM");
    f.api.startJob("train", input(newRunName));
    const current = f.children[1];
    const before = JSON.stringify(await f.api.snapshot());
    progress(old);
    old.stderr.emit("data", "late stderr\n");
    old.emit("error", new Error("late error"));
    finish(old, { passed: true }, 0);
    assert.equal(JSON.stringify(await f.api.snapshot()), before);
    progress(current);
    assert.equal((await f.api.snapshot()).trainingSteps, 4);
    assert.equal(f.api.cancelJob(), true);
    assert.equal(current.killed, "SIGTERM");
  });
}

test("cancelled evaluation cannot publish its buffered report", async () => {
  const f = fixture();
  f.add("test-a");
  f.api.startJob("eval", input());
  f.children[0].stdout.emit("data", '{"passed":true}');
  f.api.cancelJob();
  finish(f.children[0]);
  const snapshot = await f.api.snapshot();
  assert.equal(snapshot.phase, "cancelled");
  assert.equal(snapshot.evaluation, null);
  assert.equal(snapshot.artifacts.evaluation, false);
});

test("a previous evaluation persistence error cannot append to a new job", async () => {
  const f = fixture();
  let rejectWrite;
  f.deferWrites(() => new Promise((_, reject) => { rejectWrite = reject; }));
  f.add("test-a");
  f.api.startJob("eval", input());
  finish(f.children[0], { passed: false });
  f.api.startJob("train", input("test-b"));
  rejectWrite(new Error("old write failed"));
  await new Promise(setImmediate);
  assert.equal((await f.api.snapshot()).logs.some((line) => line.includes("old write failed")), false);
});

for (const sourceType of ["policy", "checkpoint"]) {
  test(`${sourceType} evaluation remains bound only to the evaluated current bytes`, async () => {
    const f = fixture();
    f.add("test-a");
    f.add("test-a", "metadata");
    if (sourceType === "policy") f.add("test-a", "onnx");
    const report = sourceReport(f, "test-a", sourceType);
    f.api.startJob("eval", input("test-a", { profile: "full" }));
    finish(f.children[0], report);
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    const source = f.artifact("test-a", sourceType === "policy" ? "onnx" : "checkpoint");
    f.files.set(source, "changed policy bytes");
    assert.equal((await f.api.snapshot()).evaluation, null);
    f.files.set(source, "artifact");
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    f.files.delete(source);
    assert.equal((await f.api.snapshot()).evaluation, null);
  });
}

for (const changedKind of ["checkpoint", "metadata"]) {
  test(`same-run retrain/render cannot revive saved evidence after ${changedKind} changes`, async () => {
    const f = fixture();
    f.add("test-a");
    f.add("test-a", "metadata");
    f.api.startJob("eval", input("test-a", { profile: "full" }));
    finish(f.children[0], sourceReport(f, "test-a", "checkpoint"));
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    f.api.startJob("train", input());
    f.files.set(f.artifact("test-a", changedKind), "retrained bytes");
    finish(f.children[1]);
    f.api.startJob("render", input());
    assert.equal((await f.api.snapshot()).evaluation, null);
    assert.equal(f.files.has(path.join(path.dirname(f.artifact("test-a")), "evaluation.json")), true);
  });
}

test("missing checkpoint sidecars and missing sidecar hashes invalidate saved evaluations", async () => {
  const f = fixture();
  f.add("test-a");
  f.add("test-a", "metadata");
  const report = sourceReport(f, "test-a", "checkpoint");
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify(report));
  f.files.delete(f.artifact("test-a", "metadata"));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
  f.add("test-a", "metadata");
  delete report.source_files_sha256[f.artifact("test-a", "metadata")];
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
});

test("source hashes never authorize reads of report-supplied paths", async () => {
  const f = fixture();
  f.add("test-a", "onnx");
  const report = sourceReport(f);
  report.source = "/unowned/secret";
  report.source_files_sha256["/unowned/secret"] = "ignored";
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation.passed, true);
  assert.equal(f.reads.includes("/unowned/secret"), false);
  report.source_type = "unsupported";
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
  assert.equal(f.reads.includes("/unowned/secret"), false);
});

test("reports without source hashes remain visible only as unassessed", async () => {
  const f = fixture();
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify({ passed: true, skill_status: "passed", evaluation_mode: "skill", finite: true }));
  const report = (await f.api.snapshot("swing", "test-a")).evaluation;
  assert.equal(report.finite, true);
  assert.equal(report.skill_status, "not_assessed");
  assert.equal(evaluationVerdict(report, "swing").taskPassed, false);
  assert.equal(evaluationVerdict(report, "dance").taskPassed, false);
});

test("snapshot captures run ownership before asynchronous evidence reads", async () => {
  const f = fixture();
  f.api.startJob("train", input());
  progress(f.children[0]);
  finish(f.children[0]);
  const pending = f.api.snapshot();
  f.api.startJob("train", input("test-b"));
  const previous = await pending;
  assert.equal(previous.runName, "test-a");
  assert.equal(previous.trainingSteps, 4);
  assert.equal(previous.rewardHistory[0].reward, 2.5);
  assert.equal((await f.api.snapshot()).trainingSteps, 0);
});

test("UI requires authoritative scoped skill verdict, not finite output or large spans", () => {
  const pipeline = { evaluation_mode: "pipeline", passed: true, finite: true, pipeline_passed: true, skill_status: "not_assessed" };
  assert.equal(evaluationVerdict(pipeline, "swing").taskPassed, false);
  assert.equal(evaluationVerdict(pipeline, "swing").skillAssessed, false);
  assert.equal(evaluationVerdict(pipeline, "dance").taskPassed, false);
  assert.equal(evaluationVerdict({ passed: true, finite: true }, "swing").taskPassed, false);
  const skill = { ...pipeline, evaluation_mode: "skill", skill_status: "failed", swing_span_deg: 175 };
  assert.equal(evaluationVerdict(skill, "swing").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "passed" }, "swing").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "passed", passed: false }, "swing").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "not_assessed" }, "dance").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, evaluation: { swing_criteria: { min_bidirectional_span_deg: 150 } } }, "swing").swingMinSpanDeg, 150);
});
