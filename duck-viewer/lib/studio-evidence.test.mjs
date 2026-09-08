import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

function load(name) {
  const source = readFileSync(new URL(name, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } });
  const exports = {};
  vm.runInNewContext(outputText, { exports });
  return exports;
}

const { skillEvidence, evidenceLabel } = load("studio-evidence.ts");
const { evaluationVerdict } = load("evaluation.ts");

test("evidence shows worst/best episodes without substituting average reward for acceptance", () => {
  const evidence = skillEvidence({ mean_return: 9999, dance_assessment: { episodes: [
    { passed: true, pose_rmse_rad: .08, dynamic_gain: .4 },
    { passed: false, pose_rmse_rad: .3, dynamic_gain: -.1 },
    { pose_rmse_rad: null },
  ], criteria: { max_pose_rmse_rad: .15 } } }, "dance");
  assert.equal(evidence.passed, 1);
  assert.equal(evidence.episodes.length, 3);
  assert.equal(evidence.ranges[0].maximum, .3);
  assert.equal(evidence.ranges[0].measured, 2);
  assert.equal(evidence.criteria.max_pose_rmse_rad, .15);
  assert.equal(evidenceLabel("pose_rmse_rad"), "pose rmse rad");
});

test("missing, malformed and nonfinite measurements stay unmeasured", () => {
  assert.equal(skillEvidence(null, "swing").episodes.length, 0);
  assert.equal(skillEvidence({ swing_assessment: { episodes: [null, 4, { max_alignment: Infinity }] } }, "swing").ranges.length, 0);
  assert.equal(skillEvidence({ dance_assessment: { episodes: [] } }, "running").episodes.length, 0);
});

test("acceptance requires skill mode, pipeline validity, matching scenario and settings", () => {
  const passing = { recipe: "running", evaluation_mode: "skill", skill_status: "passed", pipeline_passed: true, passed: true };
  assert.equal(evaluationVerdict(passing, "running").taskPassed, true);
  for (const override of [{ pipeline_passed: false }, { recipe: "dance" }, { recipe: undefined }, { evaluation_mode: "pipeline" }, { evaluation_settings_match: false }, { passed: false }]) {
    assert.equal(evaluationVerdict({ ...passing, ...override }, "running").taskPassed, false);
  }
});
