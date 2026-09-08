import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const script = fileURLToPath(new URL("../scripts/verify-rlx-video.mjs", import.meta.url));

for (const failure of ["missing recipe", "wrong run", "stale policy"]) {
  test(`video verification invalidates old success on ${failure}`, async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "rlx-video-identity-"));
    try {
      const run = path.join(directory, "run");
      await mkdir(run);
      await writeFile(path.join(directory, "video-validation.json"), '{"passed":true}');
      if (failure !== "missing recipe") {
        await writeFile(path.join(directory, "recipe.json"), JSON.stringify({experimentId: "running"}));
        await writeFile(path.join(run, "running.onnx"), "new policy");
        await writeFile(path.join(directory, "audit.json"), JSON.stringify({
          recipe: {experimentId: "running"}, run_name: failure === "wrong run" ? "other" : "run",
          policy_sha256: createHash("sha256").update("old policy").digest("hex"),
        }));
      }
      const result = spawnSync(process.execPath, [script,
        "--recipe-json", path.join(directory, "recipe.json"),
        "--run", run, "--output", directory, "--playwright-package", "unused.json",
      ], {encoding: "utf8"});
      assert.notEqual(result.status, 0);
      const receipt = JSON.parse(await readFile(path.join(directory, "video-validation.json"), "utf8"));
      assert.equal(receipt.passed, false);
      assert.match(result.stderr, failure === "missing recipe" ? /ENOENT/ : /AssertionError/);
    } finally {
      await rm(directory, {recursive: true, force: true});
    }
  });
}
