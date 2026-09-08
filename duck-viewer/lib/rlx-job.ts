import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, realpathSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  defaultRewardWeights,
  getExperiment,
  isExperimentId,
  type ExperimentId,
} from "@/lib/experiments";

export type RlxOperation = "train" | "eval" | "render" | "export";
export type RlxJobPhase =
  | "idle"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface RlxRecipe {
  experimentId: ExperimentId;
  runName: string;
  profile: "smoke" | "full";
  totalTimesteps: number;
  numEnvs: number;
  numSteps: number;
  numMinibatches: number;
  maxEpisodeS: number;
  evalSteps: number;
  renderSeconds: number;
  danceClip: string | null;
  dancePoseSigma: number | null;
  locomotionForwardCommand: number | null;
  initialStd: number;
  normalizeRewards: boolean;
  freezeObservationNormalization: boolean;
  checkpointInterval: number;
  seed: number;
  learningRate: number;
  gamma: number;
  clipCoefficient: number;
  updateEpochs: number;
  entropyCoefficient: number;
  maxGradNorm: number;
  domainRand: boolean;
  obsNoise: boolean;
  actionDelay: boolean;
  randomYaw: boolean;
  stiltHeightCm: number;
  stiltBlend: number;
  stiltMassKg: number;
  swingInitialAngleDeg: number;
  swingInitialRateRadS: number;
  swingPlanarActions: boolean;
  swingMinSpanDeg: number;
  resumeFromCheckpoint: boolean;
  rewardWeights: Record<string, number>;
}

export interface RlxArtifacts {
  checkpoint: boolean;
  metadata: boolean;
  onnx: boolean;
  renderSheet: boolean;
  renderVideo: boolean;
  evaluation: boolean;
  checkpointPath: string;
  onnxPath: string;
  renderDirectory: string;
}

export interface RlxRewardPoint {
  step: number;
  reward: number;
}

export interface RlxJobSnapshot {
  phase: RlxJobPhase;
  operation: RlxOperation | null;
  activeJob: {
    operation: RlxOperation;
    experimentId: ExperimentId;
    runName: string;
    startedAt: string | null;
  } | null;
  experimentId: ExperimentId;
  runName: string;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  evaluation: Record<string, unknown> | null;
  rewardHistory: RlxRewardPoint[];
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  artifacts: RlxArtifacts;
}

interface MutableRlxJob {
  environmentKeys: Record<string, string>;
  generation: number;
  phase: RlxJobPhase;
  operation: RlxOperation | null;
  experimentId: ExperimentId;
  runName: string;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  evaluation: Record<string, unknown> | null;
  rewardHistory: RlxRewardPoint[];
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  stdoutBuffer: string;
  stderrBuffer: string;
  child: ChildProcess | null;
}

declare global {
  var __microduckRlxJob: MutableRlxJob | undefined;
}

const INITIAL_JOB: MutableRlxJob = {
  environmentKeys: {},
  generation: 0,
  phase: "idle",
  operation: null,
  experimentId: "dance",
  runName: "dance-studio",
  startedAt: null,
  finishedAt: null,
  exitCode: null,
  logs: [],
  result: null,
  evaluation: null,
  rewardHistory: [],
  normalizeRewards: false,
  trainingSteps: 0,
  trainingTotal: 0,
  stdoutBuffer: "",
  stderrBuffer: "",
  child: null,
};

const job = (globalThis.__microduckRlxJob ??= { ...INITIAL_JOB });
job.environmentKeys ??= {};
job.generation ??= 0;
job.experimentId ??= "dance";
job.rewardHistory ??= [];
job.normalizeRewards ??=
  job.child?.spawnargs?.includes("--normalize-rewards") ?? false;
job.trainingSteps ??= 0;
job.trainingTotal ??= 0;
job.stdoutBuffer ??= "";
job.stderrBuffer ??= "";

export function sanitizeRunName(value: unknown): string {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  if (!normalized) throw new Error("Run name must contain a letter or number.");
  return normalized;
}

function rlxRoot(): string {
  const candidates = [
    path.resolve(process.cwd(), "../rlx"),
    path.resolve(process.cwd(), "rlx"),
  ];
  const root = candidates.find((candidate) =>
    existsSync(path.join(candidate, "examples/ppo_microduck_dance.py"))
  );
  if (!root) throw new Error("Cannot locate the sibling RLX checkout.");
  return root;
}

function pathsFor(experimentId: ExperimentId, runName: string) {
  const root = rlxRoot();
  const experiment = getExperiment(experimentId);
  const runDirectory = path.join(root, "runs", "studio", experimentId, runName);
  return {
    root,
    runDirectory,
    checkpoint: path.join(runDirectory, `${experiment.artifactStem}.safetensors`),
    metadata: path.join(
      runDirectory,
      `${experiment.artifactStem}.safetensors.json`
    ),
    onnx: path.join(runDirectory, `${experiment.artifactStem}.onnx`),
    renderDirectory: path.join(runDirectory, "render"),
    renderSheet: path.join(runDirectory, "render", "ep0_sheet.png"),
    renderVideo: path.join(runDirectory, "render", "ep0.mp4"),
    evaluation: path.join(runDirectory, "evaluation.json"),
  };
}

export function artifactPath(
  experimentIdValue: unknown,
  runNameValue: unknown,
  kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video"
): string {
  const experimentId = normalizeExperimentId(experimentIdValue);
  const runName = sanitizeRunName(runNameValue);
  const paths = pathsFor(experimentId, runName);
  return {
    checkpoint: paths.checkpoint,
    metadata: paths.metadata,
    onnx: paths.onnx,
    sheet: paths.renderSheet,
    video: paths.renderVideo,
  }[kind];
}

async function artifactState(
  experimentId: ExperimentId,
  runName: string
): Promise<RlxArtifacts> {
  const paths = pathsFor(experimentId, runName);
  return {
    checkpoint: existsSync(paths.checkpoint),
    metadata: existsSync(paths.metadata),
    onnx: existsSync(paths.onnx),
    renderSheet: existsSync(paths.renderSheet),
    renderVideo: existsSync(paths.renderVideo),
    evaluation: existsSync(paths.evaluation),
    checkpointPath: path.relative(paths.root, paths.checkpoint),
    onnxPath: path.relative(paths.root, paths.onnx),
    renderDirectory: path.relative(paths.root, paths.renderDirectory),
  };
}

function appendLine(line: string) {
  const trimmed = line.trimEnd();
  if (!trimmed) return;
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    try {
      const payload = JSON.parse(trimmed) as Record<string, unknown>;
      if (payload.event === "training_progress") {
        const steps =
          typeof payload.steps === "number" ? payload.steps : Number.NaN;
        const total =
          typeof payload.total === "number" ? payload.total : Number.NaN;
        const reward =
          typeof payload.mean_reward === "number"
            ? payload.mean_reward
            : Number.NaN;
        if (Number.isFinite(steps)) job.trainingSteps = Math.max(0, steps);
        if (Number.isFinite(total)) job.trainingTotal = Math.max(0, total);
        if (Number.isFinite(steps) && Number.isFinite(reward)) {
          const point = { step: Math.max(0, steps), reward };
          const previous = job.rewardHistory.at(-1);
          if (previous?.step === point.step) {
            job.rewardHistory[job.rewardHistory.length - 1] = point;
          } else {
            job.rewardHistory.push(point);
            if (job.rewardHistory.length > 240) {
              job.rewardHistory.splice(0, job.rewardHistory.length - 240);
            }
          }
        }
        return;
      }
    } catch {
      // Non-JSON process output remains visible in the log.
    }
  }
  job.logs.push(trimmed);
  if (job.logs.length > 240) job.logs.splice(0, job.logs.length - 240);
}

function appendStream(stream: "stdout" | "stderr", chunk: Buffer | string) {
  const bufferKey = stream === "stdout" ? "stdoutBuffer" : "stderrBuffer";
  const combined = job[bufferKey] + String(chunk);
  const lines = combined.split(/\r?\n/);
  job[bufferKey] = lines.pop() ?? "";
  lines.forEach(appendLine);
}

function flushStreams() {
  if (job.stdoutBuffer) appendLine(job.stdoutBuffer);
  if (job.stderrBuffer) appendLine(job.stderrBuffer);
  job.stdoutBuffer = "";
  job.stderrBuffer = "";
}

function positiveInt(value: unknown, fallback: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.min(max, Math.round(parsed)));
}

function boundedFloat(
  value: unknown,
  fallback: number,
  min: number,
  max: number
): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function explicitPositiveInt(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 1 ||
    value > max
  ) {
    throw new Error(`${label} must be an integer between 1 and ${max}.`);
  }
  return value;
}

function explicitPositiveFloat(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    value > max
  ) {
    throw new Error(`${label} must be a number greater than 0 and at most ${max}.`);
  }
  return value;
}

function explicitNonNegativeInt(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value > max
  ) {
    throw new Error(`${label} must be an integer between 0 and ${max}.`);
  }
  return value;
}

function explicitBoolean(
  value: unknown,
  fallback: boolean,
  label: string
): boolean {
  if (value == null) return fallback;
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be a boolean.`);
  }
  return value;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function normalizeDanceClip(
  experimentId: ExperimentId,
  value: unknown
): string | null {
  if (value == null) return null;
  if (typeof value !== "string") {
    throw new Error("Dance clip must be a file path string.");
  }
  const input = value.trim();
  if (!input) throw new Error("Dance clip path cannot be empty.");
  if (experimentId !== "dance") {
    throw new Error("Dance clips can only be used with the dance experiment.");
  }

  const root = rlxRoot();
  const workspaceRoot = path.dirname(root);
  const allowedRoots = [
    path.join(workspaceRoot, "dance-clip"),
    path.join(root, "artifacts"),
  ];
  const candidates = path.isAbsolute(input)
    ? [path.resolve(input)]
    : [path.resolve(workspaceRoot, input), path.resolve(root, input)];
  const candidate = candidates.find((item) =>
    allowedRoots.some((allowedRoot) => isWithin(allowedRoot, item))
  );
  if (!candidate) {
    throw new Error(
      "Dance clip must be under workspace dance-clip/ or rlx/artifacts/."
    );
  }
  if (!existsSync(candidate)) {
    throw new Error(`Dance clip does not exist: ${candidate}`);
  }

  const resolved = realpathSync(candidate);
  const resolvedRoots = allowedRoots.map((allowedRoot) =>
    existsSync(/* turbopackIgnore: true */ allowedRoot)
      ? realpathSync(/* turbopackIgnore: true */ allowedRoot)
      : allowedRoot
  );
  if (!resolvedRoots.some((allowedRoot) => isWithin(allowedRoot, resolved))) {
    throw new Error(
      "Dance clip must resolve under workspace dance-clip/ or rlx/artifacts/."
    );
  }
  if (!statSync(resolved).isFile()) {
    throw new Error(`Dance clip is not a regular file: ${resolved}`);
  }
  return resolved;
}

function normalizeDancePoseSigma(
  experimentId: ExperimentId,
  value: unknown
): number | null {
  if (value == null) return null;
  if (experimentId !== "dance") {
    throw new Error("Dance pose sigma can only be used with the dance experiment.");
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error("Dance pose sigma must be a finite number greater than 0.");
  }
  return value;
}

function normalizeLocomotionForwardCommand(
  experimentId: ExperimentId,
  value: unknown
): number | null {
  if (value == null) return null;
  if (experimentId !== "running" && experimentId !== "stilts") {
    throw new Error("Locomotion forward command can only be used with running or stilts.");
  }
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    value > 1.5
  ) {
    throw new Error("Locomotion forward command must be a finite number greater than 0 and at most 1.5.");
  }
  return value;
}

function normalizeRewardWeights(
  experimentId: ExperimentId,
  value: unknown
): Record<string, number> {
  const defaults = defaultRewardWeights(experimentId);
  if (value == null) return defaults;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Reward weights must be an object.");
  }
  const input = value as Record<string, unknown>;
  for (const key of Object.keys(input)) {
    if (!(key in defaults)) {
      throw new Error(`Unknown reward weight for ${experimentId}: ${key}`);
    }
  }
  return Object.fromEntries(
    Object.entries(defaults).map(([key, fallback]) => {
      const parsed = Number(input[key] ?? fallback);
      if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`Reward weight ${key} must be a non-negative number.`);
      }
      return [key, Math.min(parsed, 10_000)];
    })
  );
}

export function normalizeRecipe(input: Partial<RlxRecipe>): RlxRecipe {
  const experimentId = normalizeExperimentId(input.experimentId);
  const freezeObservationNormalization = explicitBoolean(
    input.freezeObservationNormalization,
    false,
    "Freeze observation normalization"
  );
  if (freezeObservationNormalization && input.resumeFromCheckpoint !== true) {
    throw new Error("Freeze observation normalization requires resumeFromCheckpoint to be true.");
  }
  const experiment = getExperiment(experimentId);
  const profile = input.profile === "full" ? "full" : "smoke";
  const smoke = profile === "smoke";
  const numEnvs = positiveInt(
    input.numEnvs,
    smoke ? 2 : experiment.fullEnvs,
    64
  );
  const numSteps = explicitPositiveInt(
    input.numSteps,
    smoke ? 2 : experimentId === "swing" ? 64 : 24,
    100_000,
    "Rollout steps"
  );
  const numMinibatches = explicitPositiveInt(
    input.numMinibatches,
    smoke ? 1 : 4,
    100_000,
    "Minibatches"
  );
  const batch = numEnvs * numSteps;
  if (numMinibatches > batch || batch % numMinibatches !== 0) {
    throw new Error(
      "Minibatches must divide numEnvs * numSteps and cannot exceed that batch."
    );
  }
  const maxEpisodeS = explicitPositiveFloat(
    input.maxEpisodeS,
    smoke ? 1 : experiment.maxEpisodeSeconds,
    3_600,
    "Maximum episode seconds"
  );
  const evalSteps = explicitPositiveInt(
    input.evalSteps,
    smoke ? 4 : Math.max(experimentId === "running" ? 600 : 500, Math.ceil(maxEpisodeS * 50)),
    10_000_000,
    "Evaluation steps"
  );
  if (!smoke && experimentId === "running" && (maxEpisodeS < 12 || evalSteps < 600)) {
    throw new Error("Full Running evaluation requires at least 12 episode seconds and 600 evaluation steps.");
  }
  const renderSeconds = explicitPositiveFloat(
    input.renderSeconds,
    experimentId === "dance" ? 120 : maxEpisodeS,
    3_600,
    "Render seconds"
  );
  let totalTimesteps = positiveInt(
    input.totalTimesteps,
    smoke ? 4 : experiment.fullTimesteps,
    40_000_000
  );
  totalTimesteps = Math.max(batch, Math.ceil(totalTimesteps / batch) * batch);
  return {
    experimentId,
    runName: sanitizeRunName(input.runName ?? experiment.defaultRunName),
    profile,
    totalTimesteps,
    numEnvs,
    numSteps,
    numMinibatches,
    maxEpisodeS,
    evalSteps,
    renderSeconds,
    danceClip: normalizeDanceClip(experimentId, input.danceClip),
    dancePoseSigma: normalizeDancePoseSigma(
      experimentId,
      input.dancePoseSigma
    ),
    locomotionForwardCommand: normalizeLocomotionForwardCommand(
      experimentId,
      input.locomotionForwardCommand
    ),
    initialStd: explicitPositiveFloat(
      input.initialStd,
      experimentId === "swing" ? 0.1 : Math.exp(-0.5),
      10,
      "Initial policy standard deviation"
    ),
    normalizeRewards: explicitBoolean(
      input.normalizeRewards,
      experimentId === "swing",
      "Reward normalization"
    ),
    freezeObservationNormalization,
    checkpointInterval: explicitNonNegativeInt(
      input.checkpointInterval,
      100_000,
      40_000_000,
      "Checkpoint interval"
    ),
    seed: positiveInt(input.seed, 1, 2_147_483_647),
    learningRate: boundedFloat(
      input.learningRate,
      experiment.ppo.learningRate,
      0.000001,
      0.1
    ),
    gamma: boundedFloat(input.gamma, experiment.ppo.gamma, 0.8, 1),
    clipCoefficient: boundedFloat(
      input.clipCoefficient,
      experiment.ppo.clipCoefficient,
      0.01,
      1
    ),
    updateEpochs: positiveInt(
      input.updateEpochs,
      experiment.ppo.updateEpochs,
      20
    ),
    entropyCoefficient: boundedFloat(
      input.entropyCoefficient,
      experiment.ppo.entropyCoefficient,
      0,
      1
    ),
    maxGradNorm: boundedFloat(
      input.maxGradNorm,
      experiment.ppo.maxGradNorm,
      0.01,
      10
    ),
    domainRand: smoke ? false : input.domainRand !== false,
    obsNoise: smoke ? false : input.obsNoise !== false,
    actionDelay: smoke ? false : input.actionDelay !== false,
    randomYaw: smoke ? false : input.randomYaw !== false,
    stiltHeightCm: boundedFloat(
      input.stiltHeightCm,
      experiment.controls?.stiltHeightCm ?? 10,
      0.8,
      300
    ),
    stiltBlend: boundedFloat(
      input.stiltBlend,
      experiment.controls?.stiltBlend ?? 0.5,
      0,
      1
    ),
    stiltMassKg: boundedFloat(
      input.stiltMassKg,
      experiment.controls?.stiltMassKg ?? 0.029,
      0.001,
      2
    ),
    swingInitialAngleDeg: boundedFloat(
      input.swingInitialAngleDeg,
      0,
      0,
      30
    ),
    swingInitialRateRadS: boundedFloat(
      input.swingInitialRateRadS,
      0,
      0,
      1
    ),
    swingPlanarActions:
      experimentId === "swing" && input.swingPlanarActions !== false,
    swingMinSpanDeg: boundedFloat(input.swingMinSpanDeg ?? 150, 150, 1, 180),
    resumeFromCheckpoint: input.resumeFromCheckpoint === true,
    rewardWeights: normalizeRewardWeights(experimentId, input.rewardWeights),
  };
}

export function normalizeExperimentId(value: unknown): ExperimentId {
  if (!isExperimentId(value)) {
    if (value == null || value === "") return "dance";
    throw new Error(`Unknown Microduck experiment: ${String(value)}`);
  }
  return value;
}

function pythonArgs(): string[] {
  const configured = process.env.MICRODUCK_STUDIO_PYTHON;
  const preferred = configured || "/usr/local/bin/python3.12";
  return [
    "run",
    "--isolated",
    "--no-project",
    "--python",
    preferred,
    "--with-editable",
    ".",
    "--with-editable",
    "../microduck_local",
  ];
}

function commonArgs(recipe: RlxRecipe): string[] {
  const args = [
    "--recipe",
    recipe.experimentId,
    "--num-envs",
    String(recipe.numEnvs),
    "--seed",
    String(recipe.seed),
    "--max-episode-s",
    String(recipe.maxEpisodeS),
    recipe.domainRand ? "--domain-rand" : "--no-domain-rand",
    recipe.obsNoise ? "--obs-noise" : "--no-obs-noise",
    recipe.actionDelay ? "--action-delay" : "--no-action-delay",
    recipe.randomYaw ? "--random-yaw" : "--no-random-yaw",
    "--weight-overrides",
    JSON.stringify(recipe.rewardWeights),
  ];
  if (recipe.danceClip) {
    args.push("--dance-clip", recipe.danceClip);
  }
  if (recipe.dancePoseSigma !== null) {
    args.push("--dance-pose-sigma", String(recipe.dancePoseSigma));
  }
  if (recipe.locomotionForwardCommand !== null) {
    args.push("--locomotion-forward-command", String(recipe.locomotionForwardCommand));
  }
  if (recipe.experimentId === "stilts") {
    args.push(
      "--stilt-height-cm",
      String(recipe.stiltHeightCm),
      "--stilt-blend",
      String(recipe.stiltBlend),
      "--stilt-mass-kg",
      String(recipe.stiltMassKg)
    );
  }
  if (recipe.experimentId === "swing") {
    args.push(
      recipe.swingPlanarActions
        ? "--swing-planar-actions"
        : "--no-swing-planar-actions"
    );
  }
  return args;
}

function evaluationEnvironmentKey(recipe: RlxRecipe, operation: RlxOperation = "eval"): string {
  return JSON.stringify(commonArgs({
    ...recipe,
    ...(operation === "render" ? {
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
      maxEpisodeS: recipe.renderSeconds,
    } : {}),
    rewardWeights: Object.fromEntries(Object.entries(recipe.rewardWeights).sort(([left], [right]) => left.localeCompare(right))),
  }));
}

function evaluationSettings(recipe: RlxRecipe) {
  return {
    evaluation_mode: recipe.profile === "smoke" ? "pipeline" : "skill",
    eval_steps: recipe.evalSteps,
    ...(recipe.locomotionForwardCommand !== null
      ? { locomotion_forward_command: recipe.locomotionForwardCommand }
      : {}),
    ...(recipe.experimentId === "swing"
      ? { swing_min_span_deg: recipe.swingMinSpanDeg }
      : {}),
    recipe,
  };
}

function commandFor(operation: RlxOperation, recipe: RlxRecipe): string[] {
  const paths = pathsFor(recipe.experimentId, recipe.runName);
  const base = [
    ...pythonArgs(),
    "examples/ppo_microduck_studio.py",
    operation,
  ];
  if (operation === "train") {
    return [
      ...base,
      "--checkpoint",
      paths.checkpoint,
      ...(recipe.resumeFromCheckpoint && existsSync(paths.checkpoint)
        ? ["--init-from", paths.checkpoint]
        : []),
      ...(recipe.freezeObservationNormalization
        ? ["--freeze-observation-normalization"]
        : []),
      "--onnx-output",
      paths.onnx,
      "--total-timesteps",
      String(recipe.totalTimesteps),
      "--num-steps",
      String(recipe.numSteps),
      "--num-minibatches",
      String(recipe.numMinibatches),
      "--checkpoint-interval",
      String(recipe.checkpointInterval),
      "--initial-std",
      String(recipe.initialStd),
      recipe.normalizeRewards
        ? "--normalize-rewards"
        : "--no-normalize-rewards",
      "--update-epochs",
      recipe.profile === "smoke" ? "1" : String(recipe.updateEpochs),
      "--learning-rate",
      String(recipe.learningRate),
      "--gamma",
      String(recipe.gamma),
      "--gae-lambda",
      recipe.experimentId === "swing" ? "0.98" : "0.95",
      "--clip-coefficient",
      String(recipe.clipCoefficient),
      "--entropy-coefficient",
      String(recipe.entropyCoefficient),
      "--max-grad-norm",
      String(recipe.maxGradNorm),
      ...(recipe.experimentId === "swing"
        ? [
            "--swing-initial-angle-deg",
            String(recipe.swingInitialAngleDeg),
            "--swing-initial-rate-rad-s",
            String(recipe.swingInitialRateRadS),
          ]
        : []),
      ...commonArgs(recipe),
    ];
  }
  if (operation === "eval") {
    const settings = evaluationSettings(recipe);
    return [
      ...base,
      ...(existsSync(paths.onnx)
        ? ["--policy", paths.onnx]
        : ["--checkpoint", paths.checkpoint]),
      "--backend",
      "dummy",
      "--evaluation-mode",
      settings.evaluation_mode,
      "--eval-steps",
      String(settings.eval_steps),
      ...(recipe.experimentId === "swing"
        ? ["--swing-min-span-deg", String(recipe.swingMinSpanDeg)]
        : []),
      ...commonArgs(recipe),
    ];
  }
  if (operation === "export") {
    return [
      ...base,
      "--recipe",
      recipe.experimentId,
      "--checkpoint",
      paths.checkpoint,
      "--output",
      paths.onnx,
    ];
  }
  return [
    ...base,
    ...(existsSync(paths.onnx)
      ? ["--policy", paths.onnx]
      : ["--checkpoint", paths.checkpoint]),
    "--output",
    paths.renderDirectory,
    "--episodes",
    "1",
    "--width",
    "640",
    "--height",
    "360",
    "--camera",
    "three-quarter",
    "--render-seconds",
    String(recipe.renderSeconds),
    ...commonArgs(recipe),
  ];
}

/** Only current owned artifact bytes can supply an authoritative evaluation verdict. */
async function boundEvaluation(
  report: Record<string, unknown> | null,
  experimentId: ExperimentId,
  runName: string,
  environmentKey: string | undefined
): Promise<Record<string, unknown> | null> {
  if (!report) return null;
  if (report.source_sha256 == null) {
    return { ...report, passed: false, skill_status: "not_assessed" };
  }
  const paths = pathsFor(experimentId, runName);
  const source = report.source_type === "policy"
    ? paths.onnx
    : report.source_type === "checkpoint" ? paths.checkpoint : null;
  if (!source || typeof report.source_sha256 !== "string") return null;
  const files = report.source_files_sha256;
  if (typeof files !== "object" || files === null || Array.isArray(files)) return null;
  const hashes = files as Record<string, unknown>;
  const ownedSources = report.source_type === "checkpoint" ? [source, paths.metadata] : [source];
  for (const file of ownedSources) {
    if (typeof hashes[file] !== "string") return null;
    let bytes: Buffer;
    try {
      bytes = await readFile(file);
    } catch {
      // Missing or unreadable owned artifacts cannot retain a measured verdict.
      return null;
    }
    const hash = createHash("sha256").update(bytes).digest("hex");
    if (hash !== hashes[file] || (file === source && hash !== report.source_sha256)) return null;
  }
  if (environmentKey !== undefined) {
    const request = report.evaluation_request as { recipe?: Partial<RlxRecipe> } | undefined;
    let matches = false;
    try {
      matches = request?.recipe != null &&
        evaluationEnvironmentKey(normalizeRecipe(request.recipe)) === environmentKey;
    } catch {
      matches = false;
    }
    if (!matches) {
      return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
    }
  }
  return report;
}

export async function snapshot(
  experimentIdValue?: unknown,
  runNameValue?: unknown
): Promise<RlxJobSnapshot> {
  const state = { ...job, logs: [...job.logs], rewardHistory: [...job.rewardHistory] };
  const experimentId = experimentIdValue
    ? normalizeExperimentId(experimentIdValue)
    : state.experimentId;
  const runName = runNameValue
    ? sanitizeRunName(runNameValue)
    : state.runName;
  const sameRun =
    runName === state.runName && experimentId === state.experimentId;
  const environmentKey = state.environmentKeys[`${experimentId}/${runName}`];
  let savedEvaluation: Record<string, unknown> | null = null;
  try {
    savedEvaluation = JSON.parse(
      await readFile(pathsFor(experimentId, runName).evaluation, "utf8")
    ) as Record<string, unknown>;
  } catch {
    savedEvaluation = null;
  }
  const evaluation = await boundEvaluation(
    sameRun && state.operation === "train"
      ? state.evaluation
      : (sameRun ? state.evaluation : null) ?? savedEvaluation,
    experimentId,
    runName,
    environmentKey
  );
  return {
    phase: sameRun ? state.phase : "idle",
    operation: sameRun ? state.operation : null,
    activeJob:
      state.phase === "running" && state.operation
        ? {
            operation: state.operation,
            experimentId: state.experimentId,
            runName: state.runName,
            startedAt: state.startedAt,
          }
        : null,
    experimentId,
    runName,
    startedAt: sameRun ? state.startedAt : null,
    finishedAt: sameRun ? state.finishedAt : null,
    exitCode: sameRun ? state.exitCode : null,
    logs: sameRun ? state.logs : [],
    result: sameRun ? state.result : null,
    evaluation,
    rewardHistory: sameRun ? state.rewardHistory : [],
    normalizeRewards: sameRun ? state.normalizeRewards : false,
    trainingSteps: sameRun ? state.trainingSteps : 0,
    trainingTotal: sameRun ? state.trainingTotal : 0,
    artifacts: await artifactState(experimentId, runName),
  };
}

export function startJob(
  operation: RlxOperation,
  recipeInput: Partial<RlxRecipe>
): RlxRecipe {
  if (job.child && job.phase === "running") {
    throw new Error(`${job.operation ?? "RLX"} is already running.`);
  }
  const recipe = normalizeRecipe(recipeInput);
  const paths = pathsFor(recipe.experimentId, recipe.runName);
  if (operation === "export" && !existsSync(paths.checkpoint)) {
    throw new Error("Export requires a checkpoint; an ONNX policy cannot be re-exported.");
  }
  if (
    operation !== "train" &&
    !existsSync(paths.checkpoint) &&
    !existsSync(paths.onnx)
  ) {
    throw new Error("Train or select a run with a checkpoint before continuing.");
  }
  if (
    operation === "train" &&
    recipe.resumeFromCheckpoint &&
    !existsSync(paths.checkpoint)
  ) {
    throw new Error(
      "This run has no checkpoint to continue. Run the Discovery stage first."
    );
  }

  const child = spawn("uv", commandFor(operation, recipe), {
    cwd: paths.root,
    env: {
      ...process.env,
      VIRTUAL_ENV: undefined,
      UV_PYTHON_PREFERENCE: "only-system",
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const changedRun =
    job.experimentId !== recipe.experimentId || job.runName !== recipe.runName;
  if (operation !== "export") {
    job.environmentKeys[`${recipe.experimentId}/${recipe.runName}`] = evaluationEnvironmentKey(recipe, operation);
  }
  job.phase = "running";
  job.operation = operation;
  job.experimentId = recipe.experimentId;
  job.runName = recipe.runName;
  job.startedAt = new Date().toISOString();
  job.finishedAt = null;
  job.exitCode = null;
  job.logs = [
    `[studio] ${operation} started`,
    `[studio] experiment: ${recipe.experimentId}`,
    `[studio] run: ${recipe.runName}`,
  ];
  job.result = null;
  if (operation === "train") {
    job.evaluation = null;
    job.rewardHistory = [];
    job.normalizeRewards = recipe.normalizeRewards;
    job.trainingSteps = 0;
    job.trainingTotal = recipe.totalTimesteps;
  } else if (changedRun) {
    job.evaluation = null;
    job.rewardHistory = [];
    job.trainingSteps = 0;
    job.trainingTotal = 0;
  }
  job.stdoutBuffer = "";
  job.stderrBuffer = "";
  job.child = child;

  const generation = ++job.generation;
  const ownsJob = () => job.generation === generation;
  child.stdout?.on("data", (chunk) => {
    if (ownsJob() && job.phase === "running") appendStream("stdout", chunk);
  });
  child.stderr?.on("data", (chunk) => {
    if (ownsJob() && job.phase === "running") appendStream("stderr", chunk);
  });
  child.on("error", (error) => {
    if (ownsJob() && job.phase === "running") appendLine(`[studio] ${error.message}`);
  });
  child.on("close", (code, signal) => {
    if (!ownsJob()) return;
    if (job.phase === "cancelled") {
      job.child = null;
      return;
    }
    flushStreams();
    job.exitCode = code;
    job.finishedAt = new Date().toISOString();
    job.child = null;
    job.phase = code === 0 ? "succeeded" : "failed";
    if (signal) appendLine(`[studio] process ended by ${signal}`);
    const jsonLine = [...job.logs]
      .reverse()
      .find((line) => line.startsWith("{") && line.endsWith("}"));
    if (jsonLine) {
      try {
        job.result = JSON.parse(jsonLine) as Record<string, unknown>;
        if (operation === "eval") {
          job.result = {
            ...job.result,
            evaluation_request: evaluationSettings(recipe),
          };
          job.evaluation = job.result;
          void writeFile(
            paths.evaluation,
            `${JSON.stringify(job.result, null, 2)}\n`
          ).catch((error: unknown) => {
            if (!ownsJob()) return;
            appendLine(
              `[studio] could not persist evaluation: ${
                error instanceof Error ? error.message : String(error)
              }`
            );
          });
        }
      } catch {
        job.result = null;
      }
    }
  });
  return recipe;
}

export function cancelJob(): boolean {
  if (!job.child || job.phase !== "running") return false;
  job.phase = "cancelled";
  job.finishedAt = new Date().toISOString();
  job.child.kill("SIGTERM");
  appendLine("[studio] cancellation requested");
  return true;
}

export async function readArtifact(
  experimentIdValue: unknown,
  runNameValue: unknown,
  kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video"
) {
  const filePath = artifactPath(experimentIdValue, runNameValue, kind);
  const [data, fileStat] = await Promise.all([readFile(filePath), stat(filePath)]);
  return { data, filePath, size: fileStat.size };
}

export async function readDanceChoreography(): Promise<string> {
  return readFile(
    path.join(rlxRoot(), "assets", "clips", "dance-120bpm.json"),
    "utf8"
  );
}
