import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  "recipe-json": { type: "string" },
  run: { type: "string" },
  output: { type: "string" },
  "base-url": { type: "string", default: "http://127.0.0.1:63317" },
  "playwright-package": { type: "string" },
} });
for (const key of ["recipe-json", "run", "output", "playwright-package"]) {
  assert.ok(values[key], `--${key} is required`);
}
const recipe = JSON.parse(await readFile(values["recipe-json"], "utf8"));
const runName = path.basename(path.resolve(values.run));
const output = path.resolve(values.output);
function expectedFrames(seconds) {
  const controlSteps = seconds * 50;
  const lower = Math.floor(controlSteps);
  const rounded = controlSteps - lower === .5
    ? (lower % 2 === 0 ? lower : lower + 1)
    : Math.round(controlSteps);
  return Math.ceil(Math.max(1, rounded) / 2);
}
await mkdir(output, { recursive: true });
const url = new URL("/api/rlx/artifact", values["base-url"]);
url.search = new URLSearchParams({ experiment: recipe.experimentId, run: runName, kind: "video", inline: "1" }).toString();
const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
assert.equal(response.status, 200);
assert.equal(response.headers.get("content-type"), "video/mp4");
const downloaded = Buffer.from(await response.arrayBuffer());
const local = await readFile(path.join(values.run, "render/ep0.mp4"));
assert.deepEqual(downloaded, local);
await writeFile(path.join(output, "api-video.mp4"), downloaded);
const range = await fetch(url, { headers: { Range: "bytes=0-1023" }, signal: AbortSignal.timeout(30000) });
assert.equal(range.status, 206);
assert.deepEqual(Buffer.from(await range.arrayBuffer()), local.subarray(0, 1024));
const { chromium } = createRequire(path.resolve(values["playwright-package"]))("playwright");
const browser = await chromium.launch({ headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
let playback;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
  await page.locator("video").evaluate(async video => { video.currentTime = 0; await video.play(); });
  await page.waitForFunction(() => document.querySelector("video")?.currentTime > 1);
  const seekTime = recipe.renderSeconds / 2;
  await page.locator("video").evaluate((video, seconds) => { video.pause(); video.currentTime = seconds; }, seekTime);
  await page.waitForFunction(seconds => {
    const video = document.querySelector("video");
    return video && !video.seeking && Math.abs(video.currentTime - seconds) < .05;
  }, seekTime);
  playback = await page.locator("video").evaluate(video => ({ duration: video.duration, width: video.videoWidth, height: video.videoHeight, currentTime: video.currentTime, error: video.error?.message ?? null }));
  assert.equal(playback.error, null);
  assert.ok(Math.abs(playback.duration - expectedFrames(recipe.renderSeconds) / 25) < .001);
  await page.screenshot({ path: path.join(output, "browser-video.png") });
} finally {
  await browser.close();
}
const videos = [];
for (const [name, seconds] of [["api-video.mp4", recipe.renderSeconds], ["comparison.mp4", recipe.maxEpisodeS]]) {
  const file = path.join(output, name);
  const probe = JSON.parse(execFileSync("ffprobe", ["-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", file], { encoding: "utf8" }));
  const stream = probe.streams.find(entry => entry.codec_type === "video");
  assert.equal(stream.codec_name, "h264");
  assert.equal(Number(stream.nb_read_frames), expectedFrames(seconds));
  assert.ok(Math.abs(Number(probe.format.duration) - expectedFrames(seconds) / 25) < .001);
  assert.equal(stream.avg_frame_rate, "25/1");
  videos.push({ file, sha256: createHash("sha256").update(await readFile(file)).digest("hex"), probe });
}
const result = { passed: true, experiment: recipe.experimentId, run: runName, download_matches_local: true, byte_range_passed: true, playback, videos };
await writeFile(path.join(output, "video-validation.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({ passed: true, playback, videos: videos.map(({ file }) => file) }));
