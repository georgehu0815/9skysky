import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const assets = fileURLToPath(new URL("./assets/", import.meta.url));
const packagePath = process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json";
const { chromium } = createRequire(packagePath)("playwright");
const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const browser = await chromium.launch({ headless: true });
const receipts = [];
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 3 });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.getByRole("radio").filter({ hasText: "Fast running" }).click();
  await page.getByLabel("Run name", { exact: true }).fill("running-e2e-20260907-v4");
  await page.locator("#evaluation").getByText("Accepted", { exact: true }).waitFor({ timeout: 30000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `${assets}studio-overview.png` });
  const advanced = page.getByRole("button", { name: /Advanced PPO and environment settings/ });
  await advanced.click();
  await page.locator("#recipe").screenshot({ path: `${assets}studio-parameters.png` });
  for (const [scenario, title, run, duration] of [
    ["dance", "Dance imitation", "dance-e2e-20260907-low-noise", 8],
    ["running", "Fast running", "running-e2e-20260907-v4", 12],
    ["stilts", "Stilt walking", "stilts-e2e-20260907-v3", 10],
    ["swing", "Self-pumped swing", "swing-e2e-20260907-v3", 24],
  ]) {
    await page.getByRole("radio").filter({ hasText: title }).click();
    await page.getByLabel("Run name", { exact: true }).fill(run);
    const panel = page.locator("#evaluation");
    await panel.getByText("Accepted", { exact: true }).waitFor({ timeout: 30000 });
    const video = panel.locator("video");
    await video.scrollIntoViewIfNeeded();
    await video.evaluate(async element => { await element.play(); });
    await page.waitForFunction(expected => {
      const element = document.querySelector("#evaluation video");
      return element && element.currentTime > 0.5 && Math.abs(element.duration - expected) < 0.05;
    }, duration);
    await video.evaluate(element => element.pause());
    const actual = await video.evaluate(element => ({ src: element.currentSrc, duration: element.duration, time: element.currentTime, error: element.error?.message ?? null }));
    assert.ok(actual.src.includes(run));
    assert.equal(actual.error, null);
    await panel.locator('section[aria-label="Skill verification summary"]').screenshot({ path: `${assets}studio-${scenario}-summary.png` });
    const contentBottomPixels = await panel.evaluate(element => {
      const review = element.querySelector('label:last-child');
      if (!review) throw new Error("Missing visual-review boundary");
      return Math.ceil((review.getBoundingClientRect().bottom - element.getBoundingClientRect().top + 16) * window.devicePixelRatio);
    });
    await panel.screenshot({ path: `${assets}studio-${scenario}-evaluation.png` });
    if (scenario === "running") await panel.screenshot({ path: `${assets}studio-evaluation.png` });
    receipts.push({ scenario, run, accepted: true, playback: actual, contentBottomPixels });
  }
} finally {
  await browser.close();
}
await writeFile(`${assets}ui-capture.json`, JSON.stringify({ capturedAt: new Date().toISOString(), baseUrl, deviceScaleFactor: 3, viewport: [1600, 1100], note: "Read-only inspection of saved runs. No training actions were submitted. Recipe controls show current UI values, not reconstructed historical recipes.", receipts }, null, 2) + "\n");
console.log(JSON.stringify({ passed: true, captured: receipts.length }));
