import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("studio-run.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports });
const { availableProfileRunName } = exports;
const { latestVerifiedRun } = exports;

function previewRun(runName, trainedAt, overrides = {}) {
  return { experimentId: "dance", runName, trainedAt, modifiedAt: trainedAt,
    checkpoint: true, taskPassed: true, skillAssessed: true, video: true,
    renderVerified: true, renderEvidenceId: "matched-video-hash", ...overrides };
}

test("previews choose the newest successful training, not the newest evaluation", () => {
  const earlier = previewRun("earlier", "2026-09-07T00:00:00Z", { modifiedAt: "2026-09-09T00:00:00Z" });
  const later = previewRun("later", "2026-09-08T00:00:00Z");
  const runs = [earlier, later];
  assert.equal(latestVerifiedRun(runs, "dance").runName, "later");
  assert.equal(runs[0], earlier);
});

test("previews exclude failed, stale, missing and unmatched artifacts", () => {
  const accepted = previewRun("accepted", "2026-09-07T00:00:00Z");
  for (const invalid of [{ taskPassed: false }, { checkpoint: false }, { video: false },
    { renderVerified: false }, { renderEvidenceId: null }, { trainedAt: "bad date" }, { experimentId: "swing" }]) {
    assert.equal(latestVerifiedRun([previewRun("rejected", "2026-09-08T00:00:00Z", invalid), accepted], "dance").runName, "accepted");
  }
});

test("each scenario selects its own verified rollout; missing evidence has no fallback", () => {
  const scenarios = ["dance", "swing", "running", "stilts"];
  const runs = scenarios.map((experimentId) => previewRun(`${experimentId}-trained`, "2026-09-08T00:00:00Z", { experimentId }));
  for (const experimentId of scenarios) assert.equal(latestVerifiedRun(runs, experimentId).runName, `${experimentId}-trained`);
  assert.equal(latestVerifiedRun([], "dance"), undefined);
  assert.equal(latestVerifiedRun([previewRun("unverified", "2026-09-08T00:00:00Z", { renderVerified: false })], "dance"), undefined);
});

test("completed smoke gets a separate Full run for every scenario", () => {
  for (const scenario of ["dance", "swing", "running", "stilts"]) {
    const run = `${scenario}-studio`;
    assert.equal(availableProfileRunName(run, "full", [run]), `${run}-full`);
  }
});

test("unoccupied custom names are preserved", () => {
  assert.equal(availableProfileRunName("my-custom-run", "full", ["dance-studio"]), "my-custom-run");
});

test("repeated presets avoid existing runs without chaining suffixes", () => {
  const reserved = ["dance-studio", "dance-studio-full", "dance-studio-full-2"];
  assert.equal(availableProfileRunName("dance-studio-full", "full", reserved), "dance-studio-full-3");
  assert.equal(availableProfileRunName("dance-studio-full", "smoke", reserved), "dance-studio-smoke");
});

test("collision checks match API canonicalization and the 48-character limit", () => {
  assert.equal(availableProfileRunName("Dance Studio", "full", ["dance-studio"]), "dance-studio-full");
  const long = "a".repeat(48);
  const full = `${"a".repeat(43)}-full`;
  const next = availableProfileRunName(long, "full", [long, full]);
  assert.equal(next.length, 48);
  assert.match(next, /-full-2$/);
});
