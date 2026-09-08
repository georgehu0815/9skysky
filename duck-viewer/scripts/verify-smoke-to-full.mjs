import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(
  process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json"
)("playwright");

const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve(
  process.env.STUDIO_EVIDENCE_DIR ||
    "../rlx/artifacts/smoke-to-full-regression-20260908"
);
const clip = {
  path: "dance-clip/cumbia_microduck_v2.clip.json",
  name: "Cumbia Microduck v2",
  durationSeconds: 9.6,
};
const smokeRun = "dance-smoke-complete";
const fullRun = "dance-full-complete";
const occupiedRuns = new Set([smokeRun, fullRun]);

await mkdir(output, { recursive: true });

function recipe(runName, profile) {
  const full = profile === "full";
  return {
    experimentId: "dance",
    runName,
    profile,
    totalTimesteps: full ? 1_000_000 : 4,
    numEnvs: full ? 16 : 2,
    numSteps: full ? 24 : 2,
    numMinibatches: full ? 4 : 1,
    maxEpisodeS: full ? clip.durationSeconds : 1,
    evalSteps: full ? Math.ceil(clip.durationSeconds * 50) : 4,
    renderSeconds: full ? clip.durationSeconds : 120,
    danceClip: clip.path,
    dancePoseSigma: null,
    locomotionForwardCommand: null,
    initialStd: Math.exp(-0.5),
    normalizeRewards: false,
    freezeObservationNormalization: false,
    checkpointInterval: 0,
    seed: 1,
    learningRate: 0.0001,
    gamma: 0.995,
    clipCoefficient: 0.1,
    updateEpochs: 3,
    entropyCoefficient: 0.002,
    maxGradNorm: 1,
    domainRand: full,
    obsNoise: full,
    actionDelay: full,
    randomYaw: full,
    stiltHeightCm: 10,
    stiltBlend: 0.5,
    stiltMassKg: 0.029,
    swingInitialAngleDeg: 0,
    swingInitialRateRadS: 0,
    swingPlanarActions: false,
    swingMinSpanDeg: 150,
    resumeFromCheckpoint: false,
    rewardWeights: {},
  };
}

function artifacts(runName, checkpoint) {
  const root = `runs/studio/dance/${runName}`;
  return {
    checkpoint,
    metadata: checkpoint,
    onnx: checkpoint,
    renderSheet: false,
    renderVideo: false,
    evaluation: false,
    checkpointPath: `${root}/dance.safetensors`,
    onnxPath: `${root}/dance.onnx`,
    renderDirectory: `${root}/render`,
  };
}

function snapshot(runName, options = {}) {
  const checkpoint = options.checkpoint ?? occupiedRuns.has(runName);
  const profile = options.profile ?? (runName === fullRun ? "full" : "smoke");
  const phase = options.phase ?? (checkpoint ? "succeeded" : "idle");
  const operation =
    options.operation === undefined
      ? checkpoint
        ? "train"
        : null
      : options.operation;
  return {
    renderVerified: false,
    renderEvidenceId: null,
    phase,
    operation,
    activeJob:
      phase === "running" && operation
        ? {
            operation,
            experimentId: "dance",
            runName,
            startedAt: "2026-09-08T12:00:00.000Z",
          }
        : null,
    experimentId: "dance",
    runName,
    startedAt: checkpoint ? "2026-09-08T11:59:00.000Z" : null,
    finishedAt: phase === "succeeded" ? "2026-09-08T12:00:00.000Z" : null,
    exitCode: phase === "succeeded" ? 0 : null,
    logs:
      phase === "succeeded"
        ? ["[studio] train request accepted", "[studio] training succeeded"]
        : phase === "running"
          ? ["[studio] train request accepted"]
          : [],
    result: phase === "succeeded" ? { returncode: 0 } : null,
    evaluation: null,
    rewardHistory:
      phase === "succeeded" ? [{ step: profile === "smoke" ? 4 : 1_000_000, reward: 1 }] : [],
    normalizeRewards: false,
    trainingSteps: phase === "succeeded" ? (profile === "smoke" ? 4 : 1_000_000) : 0,
    trainingTotal: profile === "smoke" ? 4 : 1_000_000,
    savedRecipe: checkpoint ? recipe(runName, profile) : null,
    trainingHistory: { segments: [], rewardHistory: [] },
    artifacts: artifacts(runName, checkpoint),
  };
}

function savedRuns() {
  return [
    {
      experimentId: "dance",
      runName: smokeRun,
      modifiedAt: "2026-09-08T12:00:00.000Z",
      taskPassed: false,
      skillAssessed: false,
      video: false,
      checkpoint: true,
    },
    {
      experimentId: "dance",
      runName: fullRun,
      modifiedAt: "2026-09-08T11:00:00.000Z",
      taskPassed: false,
      skillAssessed: false,
      video: false,
      checkpoint: true,
    },
  ];
}

async function installMocks(page, postHandler, getSnapshot = snapshot) {
  await page.route("**/api/rlx**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/rlx/clips") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ clips: [clip] }),
      });
      return;
    }
    if (url.pathname === "/api/rlx/runs") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ runs: savedRuns() }),
      });
      return;
    }
    if (url.pathname !== "/api/rlx") {
      await route.continue();
      return;
    }
    if (request.method() === "POST") {
      await postHandler(route, request.postDataJSON());
      return;
    }
    const runName = url.searchParams.get("run") || "dance-studio";
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(getSnapshot(runName)),
    });
  });
}

async function loadCompletedRun(page, runName) {
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  const saved = page.getByLabel("Saved runs", { exact: true });
  await saved.waitFor();
  await saved.selectOption(runName);
  await assertInputValue(page, runName);
  await page.locator("#rlx-action-status").getByText(`Training completed for ${runName}`, { exact: false }).waitFor();
}

async function assertInputValue(page, expected) {
  await page.waitForFunction(
    (value) =>
      document.querySelector('#recipe input[pattern="[A-Za-z0-9_-]+"]')?.value === value,
    expected
  );
}

async function selectFull(page) {
  await page.getByRole("button", { name: /Default full/ }).click();
}

async function freshFullRunName(page, previousRun) {
  await selectFull(page);
  const nextRun = await page.getByLabel("Run name", { exact: true }).inputValue();
  assert.notEqual(nextRun, previousRun, "Default full must not reuse an occupied checkpoint run.");
  assert.equal(occupiedRuns.has(nextRun), false, `Default full chose occupied run ${nextRun}.`);
  assert.match(nextRun, /^[A-Za-z0-9_-]+$/);
  return nextRun;
}

async function runCase(browser, name, test) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(8_000);
  const result = { name, passed: false, pageErrors: [], requests: [] };
  page.on("pageerror", (error) => result.pageErrors.push(error.message));
  try {
    await test(page, result);
    assert.deepEqual(result.pageErrors, []);
    result.passed = true;
  } catch (error) {
    result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    await page.screenshot({
      path: path.join(output, `${name}-failure.png`),
      fullPage: true,
    }).catch(() => {});
    await writeFile(path.join(output, `${name}-failure.html`), await page.content());
  } finally {
    await context.close();
  }
  return result;
}

const browser = await chromium.launch({ headless: true });
const receipt = {
  passed: false,
  checkedAt: new Date().toISOString(),
  baseUrl,
  cases: [],
};

try {
  receipt.cases.push(
    await runCase(browser, "completed-smoke-selects-fresh-full-run", async (page, result) => {
      await installMocks(page, async (route, payload) => {
        result.requests.push(payload);
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ error: "Profile selection must not launch training." }),
        });
      });
      await loadCompletedRun(page, smokeRun);
      await page.getByLabel("Dance reference clip", { exact: true }).selectOption(clip.path);
      const nextRun = await freshFullRunName(page, smokeRun);
      assert.equal(result.requests.length, 0, "Selecting Default full must not POST.");
      assert.equal(await page.getByLabel("Dance reference clip", { exact: true }).inputValue(), clip.path);
      result.freshRunName = nextRun;
    })
  );

  receipt.cases.push(
    await runCase(browser, "full-request-shows-pending-then-running", async (page, result) => {
      let releasePost;
      let runningRun = null;
      const postGate = new Promise((resolve) => {
        releasePost = resolve;
      });
      await installMocks(
        page,
        async (route, payload) => {
          result.requests.push(payload);
          runningRun = payload.recipe.runName;
          await postGate;
          await route.fulfill({
            status: 202,
            contentType: "application/json",
            body: JSON.stringify({ accepted: true, action: "train", recipe: payload.recipe }),
          });
        },
        (runName) =>
          runName === runningRun
            ? snapshot(runName, {
                checkpoint: false,
                profile: "full",
                phase: "running",
                operation: "train",
              })
            : snapshot(runName)
      );
      await loadCompletedRun(page, smokeRun);
      await page.getByLabel("Dance reference clip", { exact: true }).selectOption(clip.path);
      const nextRun = await freshFullRunName(page, smokeRun);
      const click = page.getByRole("button", { name: "Start RLX", exact: true }).click();
      await page.locator("#rlx-action-status").getByText(
        `Sending full training request for ${nextRun}`,
        { exact: false }
      ).waitFor();
      result.pendingVisible = true;
      releasePost();
      await click;
      await page.locator("#rlx-action-status").getByText(
        `Training ${nextRun}. Reward history will update`,
        { exact: false }
      ).waitFor();
      assert.equal(result.requests.length, 1);
      const payload = result.requests[0];
      assert.equal(payload.action, "train");
      assert.equal(payload.recipe.runName, nextRun);
      assert.equal(payload.recipe.profile, "full");
      assert.equal(payload.recipe.resumeFromCheckpoint, false);
      assert.equal(payload.recipe.danceClip, clip.path);
      assert.equal(payload.recipe.maxEpisodeS, clip.durationSeconds);
      assert.equal(payload.recipe.evalSteps, Math.ceil(clip.durationSeconds * 50));
      assert.equal(payload.recipe.renderSeconds, clip.durationSeconds);
      assert.equal(payload.recipe.totalTimesteps, 1_000_000);
      assert.equal(payload.recipe.numEnvs, 16);
      assert.equal(payload.recipe.domainRand, true);
      assert.equal(payload.recipe.obsNoise, true);
      assert.equal(payload.recipe.actionDelay, true);
      assert.equal(payload.recipe.randomYaw, true);
      assert.equal(runningRun, nextRun);
      result.runningVisible = true;
    })
  );

  receipt.cases.push(
    await runCase(browser, "repeated-full-on-occupied-run-gets-fresh-name", async (page, result) => {
      await installMocks(page, async (route, payload) => {
        result.requests.push(payload);
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ error: "Profile selection must not launch training." }),
        });
      });
      await loadCompletedRun(page, fullRun);
      const nextRun = await freshFullRunName(page, fullRun);
      assert.equal(result.requests.length, 0);
      assert.equal(await page.getByLabel("Dance reference clip", { exact: true }).inputValue(), clip.path);
      result.freshRunName = nextRun;
    })
  );

  receipt.cases.push(
    await runCase(browser, "conflict-is-visible-over-old-success", async (page, result) => {
      const conflict = "a409 checkpoint already exists for regression";
      await installMocks(page, async (route, payload) => {
        result.requests.push(payload);
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({ error: conflict }),
        });
      });
      await loadCompletedRun(page, smokeRun);
      await page.getByRole("button", { name: "Start RLX", exact: true }).click();
      const actionStatus = page.locator("#rlx-action-status");
      await actionStatus.getByText(conflict, { exact: false }).waitFor();
      assert.match(await actionStatus.innerText(), /a409 checkpoint already exists/);
      assert.equal(result.requests.length, 1);
      result.visibleStatus = await actionStatus.innerText();
    })
  );

  receipt.passed = receipt.cases.every((testCase) => testCase.passed);
} finally {
  await writeFile(
    path.join(output, "verification.json"),
    `${JSON.stringify(receipt, null, 2)}\n`
  );
  await browser.close();
}

console.log(JSON.stringify({
  passed: receipt.passed,
  cases: receipt.cases.map(({ name, passed, error }) => ({ name, passed, error })),
  output,
}, null, 2));

if (!receipt.passed) {
  process.exitCode = 1;
}
