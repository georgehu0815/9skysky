import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { test } from "node:test";
import ts from "typescript";

test("run catalog includes unevaluated checkpoints and excludes empty or linked directories", async () => {
  const files = new Map([
    ["/runs/dance/evaluated/evaluation.json", "2026-09-07T12:00:00Z"],
    ["/runs/dance/evaluated/policy.safetensors", "2026-09-06T12:00:00Z"],
    ["/runs/dance/checkpoint-only/policy.safetensors", "2026-09-08T12:00:00Z"],
    ["/runs/dance/linked/policy.safetensors", "2026-09-08T13:00:00Z"],
  ]);
  const dependencies = {
    "node:path": path,
    "node:fs/promises": {
      readdir: async () => ["evaluated", "checkpoint-only", "empty", "linked"].map((name) => ({ name, isDirectory: () => true })),
      realpath: async (directory) => directory.endsWith("/linked") ? "/outside" : directory,
      stat: async (file) => {
        if (!files.has(file)) throw new Error("ENOENT");
        return { mtime: new Date(files.get(file)) };
      },
    },
    "next/server": { NextResponse: { json: (body) => JSON.parse(JSON.stringify(body)) } },
    "@/lib/experiments": { EXPERIMENTS: [{ id: "dance" }] },
    "@/lib/evaluation": { evaluationVerdict: (evaluation) => ({ taskPassed: Boolean(evaluation), skillAssessed: Boolean(evaluation) }) },
    "@/lib/rlx-job": {
      artifactPath: (experiment, run) => `/runs/${experiment}/${run}/policy.safetensors`,
      sanitizeRunName: (name) => name,
      snapshot: async (_experiment, run) => ({ evaluation: run === "evaluated" ? {} : null, artifacts: { checkpoint: true, renderVideo: run === "evaluated" }, renderVerified: run === "evaluated", renderEvidenceId: run === "evaluated" ? "verified-hash" : null }),
    },
  };
  const source = readFileSync(new URL("../app/api/rlx/runs/route.ts", import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  });
  const exports = {};
  vm.runInNewContext(outputText, { exports, require: (name) => {
    assert.ok(name in dependencies, `Unexpected dependency: ${name}`);
    return dependencies[name];
  } });
  const { runs } = await exports.GET();
  assert.deepEqual(runs.map((run) => run.runName), ["checkpoint-only", "evaluated"]);
  assert.equal(runs[0].checkpoint, true);
  assert.equal(runs[0].skillAssessed, false);
  assert.equal(runs[1].taskPassed, true);
  assert.equal(runs[1].trainedAt, "2026-09-06T12:00:00.000Z");
  assert.equal(runs[1].renderVerified, true);
  assert.equal(runs[1].renderEvidenceId, "verified-hash");
  assert.equal(runs[0].renderVerified, false);
  assert.equal(runs[0].renderEvidenceId, null);
});
