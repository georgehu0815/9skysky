"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Link from "next/link";

import {
  LAB_HTTP,
  LabClient,
  fetchPolicies,
  type Frame,
  type Policy,
} from "@/lib/lab";
import {
  defaultRewardWeights,
  EXPERIMENTS,
  getExperiment,
  type ExperimentDefinition,
  type ExperimentId,
} from "@/lib/experiments";
import { evaluationVerdict } from "@/lib/evaluation";
import { evidenceLabel, skillEvidence } from "@/lib/studio-evidence";
import type { RlxRecipe } from "@/lib/rlx-job";
import type { RlxTrainingHistory } from "@/lib/rlx-history";
import { AnimPanel } from "./AnimPanel";
import { PolicyPanel } from "./PolicyPanel";
import { TeachPanel } from "./TeachPanel";
import Viewer from "./Viewer";
import styles from "./Studio.module.css";

type Operation = "train" | "eval" | "render" | "export";
type Phase = "idle" | "running" | "succeeded" | "failed" | "cancelled";

type Recipe = RlxRecipe;

interface SavedRun {
  experimentId: ExperimentId;
  runName: string;
  taskPassed: boolean;
  skillAssessed: boolean;
  video: boolean;
}

interface JobState {
  renderVerified?: boolean;
  renderEvidenceId?: string | null;
  phase: Phase;
  operation: Operation | null;
  activeJob?: {
    operation: Operation;
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
  rewardHistory: RewardPoint[];
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  savedRecipe?: Recipe | null;
  trainingHistory?: RlxTrainingHistory;
  artifacts: {
    checkpoint: boolean;
    metadata: boolean;
    onnx: boolean;
    renderSheet: boolean;
    renderVideo: boolean;
    evaluation: boolean;
    checkpointPath: string;
    onnxPath: string;
    renderDirectory: string;
  };
}

interface RewardPoint {
  step: number;
  reward: number;
}

function recipeDefaults(experimentId: ExperimentId): Recipe {
  const experiment = getExperiment(experimentId);
  return {
    experimentId,
    runName: experiment.defaultRunName,
    profile: "smoke",
    totalTimesteps: 4,
    numEnvs: 2,
    numSteps: 2,
    numMinibatches: 1,
    maxEpisodeS: 1,
    evalSteps: 4,
    renderSeconds: experiment.maxEpisodeSeconds,
    danceClip: null,
    dancePoseSigma: null,
    locomotionForwardCommand: null,
    initialStd: experimentId === "swing" ? 0.1 : Math.exp(-0.5),
    normalizeRewards: experimentId === "swing",
    freezeObservationNormalization: false,
    checkpointInterval: 0,
    seed: 1,
    learningRate: experiment.ppo.learningRate,
    gamma: experiment.ppo.gamma,
    clipCoefficient: experiment.ppo.clipCoefficient,
    updateEpochs: experiment.ppo.updateEpochs,
    entropyCoefficient: experiment.ppo.entropyCoefficient,
    maxGradNorm: experiment.ppo.maxGradNorm,
    domainRand: false,
    obsNoise: false,
    actionDelay: false,
    randomYaw: false,
    stiltHeightCm: experiment.controls?.stiltHeightCm ?? 10,
    stiltBlend: experiment.controls?.stiltBlend ?? 0.5,
    stiltMassKg: experiment.controls?.stiltMassKg ?? 0.029,
    swingInitialAngleDeg: 0,
    swingInitialRateRadS: 0,
    swingPlanarActions: experimentId === "swing",
    swingMinSpanDeg: 150,
    resumeFromCheckpoint: false,
    rewardWeights: defaultRewardWeights(experimentId),
  };
}

function fullHorizon(experimentId: ExperimentId, danceDuration?: number) {
  const seconds = experimentId === "dance" && danceDuration !== undefined ? danceDuration : Math.max(experimentId === "running" ? 12 : 0, getExperiment(experimentId).maxEpisodeSeconds);
  return { numSteps: experimentId === "swing" ? 64 : 24, numMinibatches: 4, maxEpisodeS: seconds, evalSteps: experimentId === "dance" ? Math.ceil(seconds * 50) : Math.max(500, Math.ceil(seconds * 50)), renderSeconds: seconds };
}

const DEFAULT_RECIPE = recipeDefaults("dance");

const EMPTY_JOB: JobState = {
  phase: "idle",
  operation: null,
  activeJob: null,
  experimentId: "dance",
  runName: DEFAULT_RECIPE.runName,
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
  artifacts: {
    checkpoint: false,
    metadata: false,
    onnx: false,
    renderSheet: false,
    renderVideo: false,
    evaluation: false,
    checkpointPath: "runs/studio/dance/dance-studio/dance.safetensors",
    onnxPath: "runs/studio/dance/dance-studio/dance.onnx",
    renderDirectory: "runs/studio/dance/dance-studio/render",
  },
};

const STEPS = [
  ["01", "Train", "Learn in simulation"],
  ["02", "Evaluate", "Measure the policy"],
  ["03", "Render", "Review the motion"],
  ["04", "Artifacts", "Export the policy"],
] as const;

function Icon({ children }: { children: React.ReactNode }) {
  return <span className={styles.icon} aria-hidden="true">{children}</span>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatAxisNumber(value: number) {
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.01) return value.toExponential(1);
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (absolute >= 1_000) return `${Math.round(value / 1_000)}k`;
  if (absolute >= 100) return Math.round(value).toString();
  if (absolute >= 10) return value.toFixed(0);
  return value.toFixed(absolute < 1 ? 3 : 1);
}

function rewardChart(points: RewardPoint[], totalSteps: number) {
  if (!points.length) return null;
  const left = 34;
  const right = 414;
  const top = 14;
  const bottom = 116;
  const rewards = points.map((point) => point.reward);
  let minimum = Math.min(...rewards);
  let maximum = Math.max(...rewards);
  if (minimum === maximum) {
    const padding = Math.max(1, Math.abs(minimum) * 0.1);
    minimum -= padding;
    maximum += padding;
  } else {
    const padding = (maximum - minimum) * 0.12;
    minimum -= padding;
    maximum += padding;
  }
  const xMaximum = Math.max(totalSteps, points.at(-1)?.step ?? 0, 1);
  const coordinates = points.map((point) => {
    const x = left + (Math.max(0, point.step) / xMaximum) * (right - left);
    const y = bottom - ((point.reward - minimum) / (maximum - minimum)) * (bottom - top);
    return [x, y] as const;
  });
  const line = coordinates
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`)
    .join(" ");
  const [firstX] = coordinates[0];
  const [lastX, lastY] = coordinates[coordinates.length - 1];
  return {
    line,
    area: `${line} L${lastX.toFixed(1)} ${bottom} L${firstX.toFixed(1)} ${bottom} Z`,
    lastX,
    lastY,
    yLabels: [maximum, (maximum + minimum) / 2, minimum],
    xLabels: [0, xMaximum / 3, (xMaximum * 2) / 3, xMaximum],
  };
}

function HistoryPlot({ title, points, start, end }: { title: string; points: RewardPoint[]; start: number; end: number }) {
  const plot = rewardChart(points.map((point) => ({ ...point, step: point.step - start })), end - start);
  return <div>
    <h4>{title} · {points.length} samples</h4>
    {plot ? <svg viewBox="0 0 450 145" role="img" aria-label={`${title}, complete segment from ${start} to ${end} transitions`}>
      <path d={plot.line} />
      {plot.yLabels.map((label, index) => <text key={`y-${index}`} x="0" y={18 + index * 49}>{formatAxisNumber(label)}</text>)}
      {plot.xLabels.map((label, index) => <text key={`x-${index}`} x={34 + index * 126} y="132">{formatAxisNumber(label + start)}</text>)}
      <circle cx={plot.lastX} cy={plot.lastY} r="2" fill="currentColor" />
    </svg> : <p>No measurements recorded for this series.</p>}
  </div>;
}

function TrainingLifecycle({ history }: { history: RlxTrainingHistory | undefined }) {
  if (!history?.segments.length) return null;
  const steps = history.segments.reduce((total, segment) => total + segment.trainedSteps, 0);
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(history, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "ppo-training-lifecycle.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className={styles.lifecycle} aria-label="Full training lifecycle">
    <h3>Full training lifecycle</h3>
    <p>{formatNumber(steps)} recorded PPO transitions · {history.segments.length} stage{history.segments.length === 1 ? "" : "s"}. All recorded samples, from the first rollout through the final update; no 240-sample tail window.</p>
    <p>Stages remain separate: running reward normalization changes the scale. Rollout reward is not deterministic evaluation return, and PPO losses need not decrease monotonically. Missing history is not reconstructed as invented data.</p>
    <button type="button" className={styles.button} onClick={download}>Download full reward and loss history</button>
    {history.segments.map((segment) => <article key={segment.id} data-history-segment={segment.id}>
      <h4 title={segment.id}>{segment.kind === "bc-initializer" ? "Teacher initialization (not PPO)" : "PPO training stage"} · {segment.id.split("/").at(-1)}</h4>
      {segment.initializer ? <p>{segment.initializer.label} · {segment.initializer.teacherSamples ?? "Unknown"} teacher samples. Skill acquisition can precede PPO refinement.</p> : <>
        <p>{formatNumber(segment.startStep)}–{formatNumber(segment.endStep)} cumulative transitions · {segment.normalizationLabel} · {segment.status === "active" ? "recording" : "recorded history (not a skill verdict)"}</p>
        {segment.episodes.length > 0 && <HistoryPlot title="Raw training episode return" points={segment.episodes.map((sample) => ({ step: sample.step, reward: sample.meanRawReturn }))} start={segment.startStep} end={segment.endStep} />}
        {segment.episodes.length > 0 && <p>Raw episode returns mix episode lengths and training seeds/resets. Increasing returns can reflect longer survival; physical skill is checked independently below.</p>}
        <HistoryPlot title={segment.normalizationLabel} points={segment.collections.map((sample) => ({ step: sample.step, reward: sample.meanReward }))} start={segment.startStep} end={segment.endStep} />
        <HistoryPlot title="Policy loss" points={segment.updates.flatMap((sample) => sample.policyLoss === null ? [] : [{ step: sample.step, reward: sample.policyLoss }])} start={segment.startStep} end={segment.endStep} />
        <HistoryPlot title="Value loss" points={segment.updates.flatMap((sample) => sample.valueLoss === null ? [] : [{ step: sample.step, reward: sample.valueLoss }])} start={segment.startStep} end={segment.endStep} />
      </>}
    </article>)}
  </section>;
}

function elapsed(startedAt: string | null, endedAt: string | null, now: number) {
  if (!startedAt) return "00:00:00";
  const end = endedAt ? Date.parse(endedAt) : now;
  const seconds = Math.max(0, Math.floor((end - Date.parse(startedAt)) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return [hours, minutes, seconds % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function resultNumber(result: Record<string, unknown> | null, key: string) {
  const value = result?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatReward(value: number | null) {
  if (value === null) return "—";
  if (value !== 0 && Math.abs(value) < 0.05) return value.toExponential(2);
  return value.toFixed(2);
}

function recipeMetricNumber(
  result: Record<string, unknown> | null,
  experiment: ExperimentDefinition
) {
  const direct = resultNumber(result, experiment.metricKey);
  if (direct !== null) return direct;

  const recipeMetrics = result?.recipe_metrics;
  if (!recipeMetrics || typeof recipeMetrics !== "object") return null;
  const summary = (recipeMetrics as Record<string, unknown>)[experiment.metricKey];
  if (!summary || typeof summary !== "object") return null;
  const values = summary as Record<string, unknown>;

  return resultNumber(values, "mean");
}

function recipeMetricSummaryNumber(
  result: Record<string, unknown> | null,
  experiment: ExperimentDefinition,
  statistic: "mean" | "median" | "max"
) {
  const recipeMetrics = result?.recipe_metrics;
  if (!recipeMetrics || typeof recipeMetrics !== "object") return null;
  const summary = (recipeMetrics as Record<string, unknown>)[experiment.metricKey];
  if (!summary || typeof summary !== "object") return null;
  return resultNumber(summary as Record<string, unknown>, statistic);
}

function StatusPill({
  tone,
  children,
}: {
  tone: "live" | "warn" | "muted";
  children: React.ReactNode;
}) {
  return <span className={`${styles.statusPill} ${styles[tone]}`}>{children}</span>;
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <label className={styles.toggle}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span aria-hidden="true" />
      {label}
    </label>
  );
}

export default function Studio() {
  const [selectedExperimentId, setSelectedExperimentId] =
    useState<ExperimentId>("dance");
  const [recipe, setRecipe] = useState(DEFAULT_RECIPE);
  const [receivedJob, setJob] = useState<JobState>(EMPTY_JOB);
  const job = receivedJob.experimentId === recipe.experimentId && receivedJob.runName === recipe.runName
    ? receivedJob : { ...EMPTY_JOB, experimentId: recipe.experimentId, runName: recipe.runName };
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [danceClips, setDanceClips] = useState<{ path: string; name: string; durationSeconds: number }[]>([]);
  const [clipError, setClipError] = useState<string | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [advanced, setAdvanced] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  const [reviewedEvidence, setReviewedEvidence] = useState<string | null>(null);
  const reviewIdentity = `${recipe.experimentId}/${recipe.runName}/${job.evaluation?.source_sha256 ?? "unverified"}/${job.renderEvidenceId ?? "unbound"}`;
  const visualReviewed = reviewedEvidence === reviewIdentity;
  function setVisualReviewed(reviewed: boolean) { setReviewedEvidence(reviewed ? reviewIdentity : null); }
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pendingAction, setPendingAction] = useState<Operation | "cancel" | null>(null);
  const [now, setNow] = useState(Date.now());
  const [labRewardHistory, setLabRewardHistory] = useState<RewardPoint[]>([]);
  const [viewerExpanded, setViewerExpanded] = useState(false);
  const [rewardHistoryOpen, setRewardHistoryOpen] = useState(false);
  const [rewardHistoryCopied, setRewardHistoryCopied] = useState(false);
  const [guidanceExperimentId, setGuidanceExperimentId] =
    useState<ExperimentId | null>(null);
  const [sessionTab, setSessionTab] = useState<
    "training" | "animate" | "policies" | "teach"
  >("training");
  const [homeHref] = useState(() =>
    typeof window === "undefined" ? "/" : `${window.location.pathname}${window.location.search}`
  );
  const clientRef = useRef<LabClient | null>(null);
  const labRunRef = useRef<string | null>(null);
  const guidanceDialogRef = useRef<HTMLDialogElement | null>(null);
  const rewardHistoryDialogRef = useRef<HTMLDialogElement | null>(null);
  const selectedExperiment = getExperiment(selectedExperimentId);
  const guidanceExperiment = guidanceExperimentId
    ? getExperiment(guidanceExperimentId)
    : null;
  const selectedDanceDuration = danceClips.find((clip) => recipe.danceClip === clip.path || recipe.danceClip?.endsWith(`/${clip.path}`))?.durationSeconds;

  useEffect(() => {
    window.localStorage.setItem(
      "microduck-studio-experiment",
      selectedExperimentId
    );
  }, [selectedExperimentId]);

  useEffect(() => {
    fetchPolicies().then(setPolicies).catch(() => setPolicies([]));
  }, []);

  useEffect(() => {
    let active = true;
    fetch("/api/rlx/clips", { cache: "no-store" }).then(async (response) => {
      if (!response.ok) throw new Error("Dance clip catalog unavailable.");
      const data = await response.json();
      if (active) setDanceClips(data.clips);
    }).catch((error) => { if (active) setClipError(String(error)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetch("/api/rlx/runs", { cache: "no-store" }).then(async (response) => {
      if (!response.ok) throw new Error("Saved run catalog unavailable.");
      const data = await response.json();
      if (active) { setSavedRuns(data.runs); setCatalogError(null); }
    }).catch((error) => { if (active) setCatalogError(String(error)); });
    return () => { active = false; };
  }, [job.phase]);

  useEffect(() => {
    if (!viewerExpanded) return;
    const previousOverflow = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setViewerExpanded(false);
    };
    window.addEventListener("keydown", close);
    return () => {
      document.documentElement.style.overflow = previousOverflow;
      window.removeEventListener("keydown", close);
    };
  }, [viewerExpanded]);

  useEffect(() => {
    const dialog = guidanceDialogRef.current;
    if (!dialog) return;
    if (guidanceExperiment && !dialog.open) {
      dialog.showModal();
    } else if (!guidanceExperiment && dialog.open) {
      dialog.close();
    }
  }, [guidanceExperiment]);

  useEffect(() => {
    const dialog = rewardHistoryDialogRef.current;
    if (!dialog) return;
    if (rewardHistoryOpen && !dialog.open) {
      dialog.showModal();
    } else if (!rewardHistoryOpen && dialog.open) {
      dialog.close();
    }
  }, [rewardHistoryOpen]);

  function handleViewerFrame(nextFrame: Frame | null) {
    setFrame(nextFrame);
    const training = nextFrame?.training;
    if (!training) return;
    const step =
      training.progress.overallSteps ?? training.progress.steps ?? 0;
    const reward = training.progress.ep_rew;
    setLabRewardHistory((history) => {
      const existingHistory =
        labRunRef.current === training.runName ? history : [];
      labRunRef.current = training.runName;
      if (!Number.isFinite(reward) || step <= 0) return existingHistory;
      const point = { step, reward: reward as number };
      const previous = existingHistory.at(-1);
      if (previous?.step === point.step && previous.reward === point.reward) {
        return existingHistory;
      }
      if (previous?.step === point.step) {
        return [...existingHistory.slice(0, -1), point];
      }
      return [...existingHistory, point].slice(-240);
    });
  }

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const response = await fetch(
          `/api/rlx?experiment=${recipe.experimentId}&run=${encodeURIComponent(recipe.runName)}`,
          { cache: "no-store" }
        );
        const data = (await response.json()) as JobState & { error?: string };
        if (active && response.ok) setJob(data);
      } catch {
        if (active) setNotice("The local Studio API is unavailable.");
      }
    }
    poll();
    const id = window.setInterval(poll, job.phase === "running" ? 1000 : 2500);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [recipe.experimentId, recipe.runName, job.phase]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const liveTraining = frame?.training;
  const labSteps = liveTraining?.progress.overallSteps ?? liveTraining?.progress.steps ?? 0;
  const labTotal = liveTraining?.progress.overallTotal ?? liveTraining?.progress.total ?? 0;

  const labTrainingActive =
    selectedExperimentId === "dance" && liveTraining?.status === "training";
  const rlxTrainingActive = job.phase === "running" && job.operation === "train";
  const useRlxTelemetry =
    rlxTrainingActive || (!labTrainingActive && job.rewardHistory.length > 0);
  const chartPoints = useRlxTelemetry
    ? job.rewardHistory
    : selectedExperimentId === "dance"
      ? labRewardHistory
      : [];
  const trainingSteps = useRlxTelemetry
    ? job.trainingSteps
    : selectedExperimentId === "dance"
      ? labSteps
      : 0;
  const trainingTotal = useRlxTelemetry
    ? job.trainingTotal
    : selectedExperimentId === "dance"
      ? labTotal
      : 0;
  const progress = trainingTotal > 0
    ? Math.min(100, (trainingSteps / trainingTotal) * 100)
    : 0;
  const chart = rewardChart(chartPoints, trainingTotal);
  const hasTrainingTelemetry =
    rlxTrainingActive || labTrainingActive || chartPoints.length > 0;
  const rewardIsNormalized = useRlxTelemetry && job.normalizeRewards;
  const rewardHistoryLabel = rewardIsNormalized
    ? "Normalized rollout reward history"
    : "Rollout reward history";
  const evaluation = job.evaluation;
  const evidence = skillEvidence(evaluation, selectedExperimentId);
  const recipeMetric = recipeMetricNumber(evaluation, selectedExperiment);
  const verdict = evaluationVerdict(evaluation, selectedExperimentId);
  const evalPassed = verdict.pipelinePassed;
  const swingMeanSpan =
    selectedExperimentId === "swing"
      ? recipeMetricSummaryNumber(evaluation, selectedExperiment, "mean")
      : null;
  const swingBestSpan =
    selectedExperimentId === "swing"
      ? recipeMetricSummaryNumber(evaluation, selectedExperiment, "max")
      : null;
  const swingDiscoveryPassed =
    verdict.scope === "skill" && evalPassed &&
    ((swingMeanSpan ?? 0) >= 10 || (swingBestSpan ?? 0) >= 20);
  const swingConsolidationPassed =
    verdict.scope === "skill" && evalPassed &&
    ((swingMeanSpan ?? 0) >= 30 || (swingBestSpan ?? 0) >= 60);
  const swingTargetPassed = selectedExperimentId === "swing" && verdict.taskPassed;
  const taskPassed = verdict.taskPassed;
  const operationTime = elapsed(job.startedAt, job.finishedAt, now);
  const operationLabel = job.operation
    ? `${job.operation.charAt(0).toUpperCase()}${job.operation.slice(1)} time`
    : "Operation time";
  const stage = !job.artifacts.checkpoint
    ? 0
    : !job.artifacts.evaluation
      ? 1
      : !job.artifacts.renderVideo || !visualReviewed
        ? 2
        : 3;
  const deployReady =
    job.artifacts.onnx && taskPassed && job.renderVerified && job.artifacts.renderSheet && visualReviewed;
  const reward = chartPoints.at(-1)?.reward ?? null;
  const activeJob = job.activeJob;
  const anotherRunActive = Boolean(
    activeJob &&
      (activeJob.experimentId !== recipe.experimentId ||
        activeJob.runName !== recipe.runName)
  );
  const trainButtonLabel =
    pendingAction === "train"
      ? "Starting RLX..."
      : job.phase === "running" && job.operation === "train"
        ? "RLX training"
        : anotherRunActive
          ? "RLX busy"
          : "Start RLX";
  const recipeActionStatus =
    pendingAction === "train"
      ? `Sending ${recipe.profile === "smoke" ? "pipeline smoke" : "full training"} request for ${recipe.runName}...`
      : anotherRunActive && activeJob
        ? `RLX is already running ${activeJob.operation} for ${activeJob.experimentId}/${activeJob.runName}. Only one local RLX job can run at a time.`
        : job.phase === "running" && job.operation === "train"
          ? `Training ${job.runName}. Reward history will update after each PPO rollout.`
          : job.phase === "succeeded" && job.operation === "train"
            ? `Training completed for ${job.runName} with ${job.rewardHistory.length} reward sample${job.rewardHistory.length === 1 ? "" : "s"}.`
          : job.phase === "failed" && job.operation === "train"
            ? `Training failed for ${job.runName}. Open View logs for the exact error.`
            : job.phase === "cancelled" && job.operation === "train"
              ? `Training was stopped for ${job.runName}.`
            : notice;
  const rewardHistoryText = [
    `step\t${rewardIsNormalized ? "normalized_rollout_reward" : "rollout_reward"}`,
    ...chartPoints.map((point) => `${point.step}\t${point.reward}`),
  ].join("\n");
  const activePolicies = policies.filter((policy) => policy.group === "runs").length;

  const recipeCommand = `POST /api/rlx\n${JSON.stringify({ action: "train", recipe }, null, 2)}`;

  async function runAction(action: Operation | "cancel", recipeOverride?: Recipe) {
    setBusy(true);
    setPendingAction(action);
    let activeRecipe = recipeOverride ?? recipe;
    const actionLabel =
      action === "train"
        ? "Training"
        : action === "eval"
          ? "Evaluation"
          : action === "render"
            ? "Rendering"
            : action === "export"
              ? "Export"
              : "Cancellation";
    setNotice(
      action === "cancel"
        ? "Stopping the active RLX job..."
        : `Starting ${actionLabel.toLowerCase()} for ${activeRecipe.runName}...`
    );
    try {
      const response = await fetch("/api/rlx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, recipe: activeRecipe }),
      });
      const data = (await response.json()) as { error?: string; recipe?: Recipe };
      if (!response.ok) throw new Error(data.error || "The action could not start.");
      if (data.recipe) {
        activeRecipe = data.recipe;
        setRecipe(activeRecipe);
      }
      if (action !== "cancel") {
        setVisualReviewed(false);
        setJob((current) => ({
          ...current,
          phase: "running",
          operation: action,
          activeJob: {
            operation: action,
            experimentId: activeRecipe.experimentId,
            runName: activeRecipe.runName,
            startedAt: new Date().toISOString(),
          },
          experimentId: activeRecipe.experimentId,
          runName: activeRecipe.runName,
          startedAt: new Date().toISOString(),
          finishedAt: null,
          exitCode: null,
          logs: [
            `[studio] ${action} request accepted`,
            `[studio] experiment: ${activeRecipe.experimentId}`,
            `[studio] run: ${activeRecipe.runName}`,
          ],
          ...(action === "train"
            ? {
                rewardHistory: [],
                trainingSteps: 0,
                trainingTotal: activeRecipe.totalTimesteps,
              }
            : {}),
        }));
      }
      setNotice(
        action === "cancel"
          ? "Cancellation requested."
          : `${actionLabel} started.`
      );
      try {
        const status = await fetch(
          `/api/rlx?experiment=${activeRecipe.experimentId}&run=${encodeURIComponent(activeRecipe.runName)}`,
          { cache: "no-store" }
        );
        if (status.ok) setJob(await status.json());
      } catch {
        setNotice(`${actionLabel} started. Status will refresh automatically.`);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The action failed.");
    } finally {
      setPendingAction(null);
      setBusy(false);
    }
  }

  function selectExperiment(experimentId: ExperimentId) {
    setSelectedExperimentId(experimentId);
    setRecipe(recipeDefaults(experimentId));
    setVisualReviewed(false);
    setNotice(null);
  }

  async function loadSavedRun(experimentId: ExperimentId, runName: string) {
    setBusy(true);
    try {
      const response = await fetch(`/api/rlx?experiment=${experimentId}&run=${encodeURIComponent(runName)}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Cannot load saved run.");
      const data = await response.json() as JobState;
      setSelectedExperimentId(experimentId);
      setRecipe(data.savedRecipe ?? { ...recipeDefaults(experimentId), runName });
      setJob(data);
      setVisualReviewed(false);
      setNotice(data.savedRecipe ? "Loaded the saved evaluation recipe. Changes affect future jobs only." : "Artifacts loaded; historical recipe unavailable. Current controls are defaults, not training provenance.");
      document.getElementById("evaluation")?.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  function previewSelectedExperiment() {
    if (selectedExperimentId === "dance") {
      setSessionTab("animate");
      document
        .getElementById("training")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
      window.requestAnimationFrame(() => {
        window.dispatchEvent(new Event("microduck:play-choreography"));
      });
      return;
    }
    window.open(selectedExperiment.video, "_blank", "noopener,noreferrer");
  }

  function trainRecommendedRecipe() {
    const fullRecipe: Recipe = {
      ...recipe,
      profile: "full",
      ...fullHorizon(recipe.experimentId, selectedDanceDuration),
      totalTimesteps: Math.max(
        recipe.totalTimesteps,
        selectedExperiment.fullTimesteps
      ),
      numEnvs: Math.max(recipe.numEnvs, selectedExperiment.fullEnvs),
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: true,
    };
    setRecipe(fullRecipe);
    void runAction("train", fullRecipe);
  }

  function applySwingDiscovery() {
    setAdvanced(true);
    setRecipe((current) => ({
      ...current,
      runName: "swing-curriculum-01",
      profile: "full",
      ...fullHorizon("swing"),
      totalTimesteps: 250_000,
      numEnvs: 16,
      seed: 2,
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
      swingInitialAngleDeg: 12,
      swingInitialRateRadS: 0.35,
      swingPlanarActions: true,
      resumeFromCheckpoint: false,
      rewardWeights: defaultRewardWeights("swing"),
    }));
    document
      .getElementById("recipe")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function applySwingConsolidation() {
    setAdvanced(true);
    setRecipe((current) => ({
      ...current,
      profile: "full",
      ...fullHorizon("swing"),
      totalTimesteps: 500_000,
      numEnvs: 16,
      seed: 2,
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: false,
      swingInitialAngleDeg: 4,
      swingInitialRateRadS: 0.12,
      swingPlanarActions: true,
      resumeFromCheckpoint: true,
      rewardWeights: defaultRewardWeights("swing"),
    }));
    document
      .getElementById("recipe")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function startLabTraining(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`${LAB_HTTP}/teach`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: selectedExperiment.goal,
          steps: recipe.profile === "smoke" ? 100_000 : recipe.totalTimesteps,
        }),
      });
      if (!response.ok) throw new Error(`duck-lab returned ${response.status}`);
      setNotice("Duck Lab training request accepted.");
    } catch {
      setNotice(
        `Duck Lab is offline. Use Run RLX training for the local ${selectedExperiment.shortTitle.toLowerCase()} pipeline.`
      );
    } finally {
      setBusy(false);
    }
  }

  function artifactHref(kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video") {
    return `/api/rlx/artifact?experiment=${recipe.experimentId}&run=${encodeURIComponent(recipe.runName)}&kind=${kind}`;
  }

  async function copyRewardHistory() {
    try {
      await navigator.clipboard.writeText(rewardHistoryText);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = rewardHistoryText;
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }
    setRewardHistoryCopied(true);
    window.setTimeout(() => setRewardHistoryCopied(false), 1800);
  }

  return (
    <div className={styles.app}>
      <aside className={styles.sidebar}>
        <Link className={styles.brand} href={homeHref} aria-label="Microduck Studio home">
          <span className={styles.brandMark}><Icon>⌁</Icon></span>
          <span>microduck<small>STUDIO</small></span>
        </Link>
        <div className={styles.workspace}>
          <span className={styles.avatar}>ML</span>
          <span>Local workspace<small>Apple Silicon lab</small></span>
          <span className={styles.chevron}>⌄</span>
        </div>
        <p className={styles.navHeading}>WORKSPACE</p>
        <nav aria-label="Workspace">
          <a className={styles.navItem} href="#overview"><Icon>▦</Icon><span>Overview</span></a>
          <a className={styles.navItem} href="#recipe"><Icon>◇</Icon><span>Projects</span><b>1</b></a>
          <a className={`${styles.navItem} ${styles.active}`} href="#experiments"><Icon>♙</Icon><span>Experiments</span><b>4</b></a>
          <a className={styles.navItem} href="#artifacts"><Icon>▱</Icon><span>Policies</span><b>{activePolicies}</b></a>
          <a className={styles.navItem} href="#evaluation"><Icon>⌁</Icon><span>Evaluations</span></a>
        </nav>
        <p className={styles.navHeading}>CONTROL CENTER</p>
        <nav aria-label="Control center">
          <a className={styles.navItem} href="#deployment"><Icon>⌾</Icon><span>Robot fleet</span></a>
          <a className={styles.navItem} href="#deployment"><Icon>♢</Icon><span>Deployments</span></a>
          <a className={styles.navItem} href="#system"><Icon>⚙</Icon><span>Settings</span></a>
        </nav>
        <div className={styles.sidebarBottom}>
          <div className={styles.localStatus}>
            <strong><i className={connected ? styles.onlineDot : styles.offlineDot} />{connected ? "Duck Lab connected" : "Local studio ready"}</strong>
            <p>{connected ? "Live frames on :8788" : "RLX jobs run on this Mac"}</p>
          </div>
          <div className={styles.profile}>
            <span className={styles.profileAvatar}>GH</span>
            <span>Local operator<small>No cloud required</small></span>
          </div>
        </div>
      </aside>

      <main className={styles.main} id="overview">
        <header className={styles.topbar}>
          <div className={styles.breadcrumbs}>
            <span>Workspace</span><b>/</b><span>Experiments</span><b>/</b><strong>{selectedExperiment.shortTitle}</strong>
          </div>
          <div className={styles.topActions}>
            <StatusPill tone={connected ? "live" : "muted"}>
              {connected ? "LIVE LAB" : "LOCAL MODE"}
            </StatusPill>
            <button
              className={styles.iconButton}
              onClick={() => setViewerExpanded(true)}
              title="Expand the simulation workspace"
              aria-label="Expand the simulation workspace"
            >
              ⛶
            </button>
            <span className={styles.avatar}>GH</span>
          </div>
        </header>

        <section className={styles.heading}>
          <div>
            <p className={styles.eyebrow}>SMALL ROBOT. COMPLETE POLICY PIPELINE.</p>
            <h1>Learning in motion.</h1>
            <p>Train a behavior. Prove it in simulation. Export what the robot can run.</p>
          </div>
          <div className={styles.headingActions}>
            {job.artifacts.metadata && (
              <a className={styles.button} href={artifactHref("metadata")}>
                <Icon>↓</Icon> Metadata
              </a>
            )}
            <button
              className={`${styles.button} ${styles.primary}`}
              onClick={() => {
                const suffix = new Date().toISOString().slice(11, 19).replaceAll(":", "");
                setRecipe((current) => ({
                  ...current,
                  runName: `${selectedExperiment.id}-${suffix}`,
                }));
                setVisualReviewed(false);
                document.getElementById("recipe")?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <Icon>＋</Icon> New experiment
            </button>
          </div>
        </section>

        <section className={styles.experimentCatalog} id="experiments" aria-labelledby="experiment-catalog-title">
          <div className={styles.catalogHead}>
            <div>
              <p className={styles.eyebrow}>MACOS RLX RECIPES</p>
              <h2 id="experiment-catalog-title">Choose what the duck should learn</h2>
            </div>
            <span>Reference previews are published simulation evidence.</span>
          </div>
          <div className={styles.experimentTable} role="radiogroup" aria-label="Training experiments">
            <div className={styles.experimentHeader} aria-hidden="true">
              <span>Experiment</span><span>Preview</span><span>Result and artifacts</span>
            </div>
            {EXPERIMENTS.map((experiment) => {
              const selected = experiment.id === selectedExperimentId;
              return (
                <div
                  role="radio"
                  aria-checked={selected}
                  tabIndex={0}
                  className={`${styles.experimentRow} ${selected ? styles.selectedExperiment : ""}`}
                  key={experiment.id}
                  onClick={() => selectExperiment(experiment.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      selectExperiment(experiment.id);
                    }
                  }}
                >
                  <span className={styles.experimentIdentity}>
                    <span className={styles.recipeCardTopline}>
                      <i>{selected ? "SELECTED" : "RLX RECIPE"}</i>
                      <button
                        type="button"
                        className={styles.recipeInfoButton}
                        aria-label={`Show ${experiment.title} guidance`}
                        title={`How to train ${experiment.title.toLowerCase()}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setGuidanceExperimentId(experiment.id);
                        }}
                        onKeyDown={(event) => event.stopPropagation()}
                      >
                        i
                      </button>
                    </span>
                    <strong>{experiment.title}</strong>
                    <small>{experiment.goal}</small>
                  </span>
                  <span className={styles.previewFrame}>
                    {experiment.preview.endsWith(".mp4") ? (
                      <video
                        src={experiment.preview}
                        poster={experiment.poster}
                        autoPlay={selected}
                        muted
                        loop
                        playsInline
                      />
                    ) : (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={experiment.preview} alt="" />
                    )}
                    <b>REFERENCE</b>
                  </span>
                  <span className={styles.experimentResult}>
                    <strong>{experiment.result}</strong>
                    <small>{experiment.task}</small>
                    <span className={styles.referenceLinks}>
                      <a
                        href={experiment.video}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(event) => event.stopPropagation()}
                      >
                        Full video ↗
                      </a>
                      <a
                        href={`/experiments/${experiment.id}/evaluation.json`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(event) => event.stopPropagation()}
                      >
                        Evaluation ↗
                      </a>
                      <a
                        href={`/guides/microduck-studio-experiments-guide.pdf#${experiment.guideAnchor}`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(event) => event.stopPropagation()}
                      >
                        Guide ↗
                      </a>
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <section className={styles.savedEvidence} aria-label="Saved scenario evidence">
          <h2>Review trained policies</h2>
          <p>Current checkpoint-bound evaluations, not preview animations. Nominal MuJoCo simulation only; not hardware certification.</p>
          {catalogError && <p role="alert">{catalogError}</p>}
          <div className={styles.evidenceCards}>
            {EXPERIMENTS.map((experiment) => {
              const run = savedRuns.find((candidate) => candidate.experimentId === experiment.id && candidate.taskPassed)
                ?? savedRuns.find((candidate) => candidate.experimentId === experiment.id);
              return <article key={experiment.id}>
                <strong>{experiment.title}</strong>
                <span>{run ? run.taskPassed ? "Skill passed · saved evaluation" : run.skillAssessed ? "Skill failed" : "Skill not assessed" : "No evaluated run found"}</span>
                <small>{run?.runName ?? "Train and evaluate to create evidence"}</small>
                <button className={styles.button} disabled={!run || busy} onClick={() => run && void loadSavedRun(experiment.id, run.runName)}>Review {experiment.shortTitle}</button>
              </article>;
            })}
          </div>
        </section>

        <section className={styles.steps} aria-label={`${selectedExperiment.title} workflow`}>
          {STEPS.map(([number, label, detail], index) => (
            <a
              key={label}
              className={`${styles.step} ${index < stage ? styles.done : ""} ${index === stage ? styles.current : ""}`}
              href={index === 0 ? "#recipe" : index === 1 ? "#evaluation" : index === 2 ? "#evaluation" : "#artifacts"}
            >
              <span className={styles.stepNumber}>{index < stage ? "✓" : number}</span>
              <span><strong>{label}</strong><small>{detail}</small></span>
            </a>
          ))}
        </section>

        {notice && (
          <div className={styles.notice} role="status">
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} aria-label="Dismiss message">×</button>
          </div>
        )}

        <section className={styles.coach} aria-labelledby="experiment-coach-title">
          <div className={styles.coachIntro}>
            <p className={styles.eyebrow}>SELECTED RLX WORKFLOW</p>
            <h2 id="experiment-coach-title">{selectedExperiment.title}</h2>
            <p>{selectedExperiment.description}</p>
            <div className={styles.evidenceList}>
              {selectedExperiment.evidence.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
            <div className={styles.coachActions}>
              <button className={styles.button} onClick={previewSelectedExperiment}>
                <Icon>▶</Icon> Preview reference
              </button>
              {!job.artifacts.checkpoint ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={trainRecommendedRecipe}
                  disabled={busy || job.phase === "running"}
                >
                  <Icon>✦</Icon> Train {selectedExperiment.shortTitle.toLowerCase()}
                </button>
              ) : !job.artifacts.evaluation ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("eval")}
                  disabled={busy || job.phase === "running"}
                >
                  <Icon>✓</Icon> Evaluate policy
                </button>
              ) : !job.artifacts.renderVideo ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("render")}
                  disabled={busy || job.phase === "running"}
                >
                  <Icon>◫</Icon> Render rollout
                </button>
              ) : (
                <a className={`${styles.button} ${styles.primary}`} href="#evaluation">
                  <Icon>▶</Icon> Review result
                </a>
              )}
            </div>
          </div>
          <ol className={styles.coachSteps}>
            <li className={styles.coachDone}>
              <span>1</span>
              <div><strong>Plan</strong><small>{selectedExperiment.task}</small></div>
            </li>
            <li className={job.artifacts.checkpoint ? styles.coachDone : styles.coachCurrent}>
              <span>2</span>
              <div><strong>Train</strong><small>{job.artifacts.checkpoint ? "Checkpoint ready" : `${formatNumber(recipe.totalTimesteps)} planned steps · ${recipe.numEnvs} environments`}</small></div>
            </li>
            <li className={job.artifacts.evaluation ? styles.coachDone : job.artifacts.checkpoint ? styles.coachCurrent : ""}>
              <span>3</span>
              <div><strong>Evaluate</strong><small>{selectedExperiment.metricLabel} · finite actions</small></div>
            </li>
            <li className={visualReviewed ? styles.coachDone : job.artifacts.renderVideo ? styles.coachCurrent : ""}>
              <span>4</span>
              <div><strong>Review</strong><small>Watch motion · compare against null controls</small></div>
            </li>
          </ol>
        </section>

        <div className={styles.dashboard}>
          <section
            className={`${styles.panel} ${styles.simulationPanel} ${viewerExpanded ? styles.workspaceExpanded : ""}`}
            id="simulation"
          >
            {viewerExpanded && (
              <button
                className={styles.fullscreenClose}
                onClick={() => setViewerExpanded(false)}
                title="Restore Studio layout"
                aria-label="Restore Studio layout"
              >
                ×
              </button>
            )}
            <div className={styles.panelHead}>
              <div>
                <h2><Icon>◇</Icon>Simulation workspace</h2>
                <p>Policies, teaching, animation, recording, and live MuJoCo controls</p>
              </div>
              <div className={styles.panelTools}>
                <StatusPill tone={connected ? "live" : "muted"}>
                  {connected ? "LIVE MUJOCO" : "WAITING FOR LAB"}
                </StatusPill>
                <button
                  className={styles.iconButton}
                  onClick={() => setViewerExpanded((expanded) => !expanded)}
                  title={viewerExpanded ? "Restore Studio layout" : "Expand the simulation workspace"}
                  aria-label={viewerExpanded ? "Restore Studio layout" : "Expand the simulation workspace"}
                >
                  {viewerExpanded ? "×" : "⛶"}
                </button>
              </div>
            </div>
            <div className={styles.viewer}>
              <Viewer
                layout="studio"
                showAnimationTools={false}
                showPolicyTools={false}
                showTeachTools={false}
                onClientReady={(client) => {
                  clientRef.current = client;
                }}
                onConnectionChange={setConnected}
                onFrame={handleViewerFrame}
              />
            </div>
            <div className={styles.viewerControls}>
              <button onClick={() => clientRef.current?.sendReset()} disabled={!connected} title="Reset every live simulation">
                ↺
              </button>
              <span>{connected ? `${frame?.ducks.length ?? 0} live duck${frame?.ducks.length === 1 ? "" : "s"}` : "Lab offline"}</span>
              <span>61 observations</span>
              <span>14 actions</span>
              <span>50 Hz control</span>
              <button
                className={styles.workspaceButton}
                onClick={() => setViewerExpanded((expanded) => !expanded)}
              >
                {viewerExpanded ? "Restore layout" : "Expand workspace"}
              </button>
            </div>
          </section>

          <section className={styles.panel} id="training">
            <div className={styles.sessionHead}>
              <div className={styles.sessionTabs} role="tablist" aria-label={`${selectedExperiment.title} workspace`}>
                <button
                  role="tab"
                  aria-selected={sessionTab === "training"}
                  className={sessionTab === "training" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("training")}
                >
                  <Icon>♙</Icon> Training
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "animate"}
                  className={sessionTab === "animate" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("animate")}
                >
                  <Icon>▶</Icon> Animate
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "policies"}
                  className={sessionTab === "policies" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("policies")}
                >
                  <Icon>◇</Icon> Policies
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "teach"}
                  className={sessionTab === "teach" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("teach")}
                >
                  <Icon>✦</Icon> Teach
                </button>
              </div>
            </div>
            <div
              className={styles.trainingBody}
              role="tabpanel"
              hidden={sessionTab !== "training"}
            >
              <div className={styles.experimentTop}>
                <div>
                  <h3>{recipe.runName}</h3>
                  <p>{selectedExperiment.shortTitle} · RLX PPO · MLX Metal · Seed {recipe.seed}</p>
                </div>
                <StatusPill tone={hasTrainingTelemetry ? "live" : "muted"}>
                  {hasTrainingTelemetry ? "LIVE DATA" : "WAITING FOR DATA"}
                </StatusPill>
              </div>
              <div className={styles.metricGrid}>
                <div>
                  <span>{rewardIsNormalized ? "Normalized training reward" : "Training reward"}</span>
                  <strong>{formatReward(reward)}</strong>
                  <small>
                    {reward == null
                      ? "Waiting for the first rollout"
                      : rewardIsNormalized
                        ? "Latest normalized rollout-buffer mean"
                        : "Latest rollout mean"}
                  </small>
                </div>
                <div>
                  <span>Evaluation score</span>
                  <strong>
                    {recipeMetric == null
                      ? "Not measured"
                      : `${recipeMetric.toFixed(2)}${selectedExperiment.metricSuffix}`}
                  </strong>
                  <small>
                    {evaluation
                      ? evalPassed
                        ? "Rollout checks passed"
                        : "Evaluation needs review"
                      : "Run evaluation after export"}
                  </small>
                </div>
                <div className={styles.timerMetric}>
                  <span>{operationLabel}</span>
                  <strong>{operationTime}</strong>
                  <small>
                    {job.phase === "running"
                      ? `${job.operation} in progress`
                      : job.startedAt
                        ? `Last ${job.operation} ${job.phase}`
                        : "Starts with training or evaluation"}
                  </small>
                </div>
              </div>
              <div className={styles.progressLabel}>
                <span>{job.phase === "running" ? `${job.operation} process` : "Training progress"}</span>
                <b>{trainingTotal > 0 ? `${formatNumber(trainingSteps)} / ${formatNumber(trainingTotal)}` : operationTime}</b>
              </div>
              <div
                className={`${styles.progress} ${job.phase === "running" && job.operation !== "train" ? styles.indeterminate : ""}`}
                role="progressbar"
                aria-label="Training progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={trainingTotal > 0 ? Math.round(progress) : undefined}
              >
                <span style={{ width: `${trainingTotal > 0 ? progress : job.artifacts.checkpoint ? 100 : 0}%` }} />
              </div>
              <div className={styles.chartHead}>
                <div className={styles.chartTitle}>
                  <strong>{rewardHistoryLabel} · {useRlxTelemetry ? "current PPO stage" : "live Duck Lab stream (up to 240 samples)"}</strong>
                  <button
                    type="button"
                    className={styles.historyIndicator}
                    onClick={() => setRewardHistoryOpen(true)}
                    aria-label={`Open reward history table with ${chartPoints.length} samples`}
                    title="Open reward history table"
                  >
                    <Icon>▦</Icon>
                    <b>{chartPoints.length}</b>
                  </button>
                </div>
                <span>
                  <i />
                  {rewardIsNormalized
                    ? "Normalized PPO rollout curve · evaluate exported ONNX separately"
                    : "PPO rollout curve · evaluate exported ONNX separately"}
                </span>
              </div>
              <svg
                className={styles.chart}
                viewBox="0 0 420 128"
                preserveAspectRatio="none"
                role="img"
                aria-label={rewardHistoryLabel}
                data-reward-points={chartPoints.length}
              >
                <g className={styles.gridLines}>
                  <path d="M34 14H414M34 48H414M34 82H414M34 116H414" />
                </g>
                {chart ? (
                  <>
                    <g className={styles.chartLabels}>
                      {chart.yLabels.map((label, index) => (
                        <text key={`y-${index}`} x="2" y={18 + index * 49}>{formatAxisNumber(label)}</text>
                      ))}
                      {chart.xLabels.map((label, index) => (
                        <text key={`x-${index}`} x={34 + index * 126} y="127">{formatAxisNumber(label)}</text>
                      ))}
                    </g>
                    <path className={styles.chartArea} d={chart.area} />
                    <path className={styles.chartLine} d={chart.line} data-testid="reward-history-line" />
                    <circle cx={chart.lastX} cy={chart.lastY} r="3" className={styles.chartPoint} />
                  </>
                ) : (
                  <text x="224" y="67" textAnchor="middle" className={styles.chartEmpty}>
                    {hasTrainingTelemetry ? "Waiting for first PPO rollout" : "No training telemetry yet"}
                  </text>
                )}
              </svg>
              <TrainingLifecycle history={job.trainingHistory} />
              <div className={styles.trainingActions}>
                {job.phase === "running" ? (
                  <button className={styles.button} onClick={() => runAction("cancel")} disabled={busy}>
                    <Icon>■</Icon> Stop {job.operation}
                  </button>
                ) : (
                  <button
                    className={styles.button}
                    onClick={() =>
                      document
                        .getElementById("recipe")
                        ?.scrollIntoView({ behavior: "smooth", block: "start" })
                    }
                  >
                    <Icon>⚙</Icon> Configure training
                  </button>
                )}
                <button className={styles.button} onClick={() => setLogsOpen((open) => !open)}>
                  {logsOpen ? "Hide logs" : "View logs"} <Icon>→</Icon>
                </button>
              </div>
              {logsOpen && (
                <pre className={styles.logs}>{job.logs.length ? job.logs.join("\n") : recipeCommand}</pre>
              )}
            </div>
            <div
              className={styles.animationBody}
              role="tabpanel"
              hidden={sessionTab !== "animate"}
            >
              <AnimPanel variant="embedded" active={sessionTab === "animate"} />
            </div>
            <div
              className={styles.toolBody}
              role="tabpanel"
              hidden={sessionTab !== "policies"}
            >
              <PolicyPanel
                clientRef={clientRef}
                variant="embedded"
                active={sessionTab === "policies"}
              />
            </div>
            <div
              className={styles.toolBody}
              role="tabpanel"
              hidden={sessionTab !== "teach"}
            >
              <TeachPanel clientRef={clientRef} variant="embedded" />
            </div>
          </section>
        </div>

        <div className={styles.lower}>
          <section className={styles.panel} id="recipe">
            <div className={styles.panelHead}>
              <h2><Icon>✦</Icon>Microduck PPO Recipe</h2>
              <StatusPill tone="live">RECOMMENDED</StatusPill>
            </div>
            <form className={styles.recipeBody} onSubmit={(event) => { event.preventDefault(); if (!busy) void runAction("train"); }}>
              <div className={styles.recipeIntro}>
                <div className={styles.recipeMark}>PPO</div>
                <div>
                  <strong>{selectedExperiment.title}</strong>
                  <span>{selectedExperiment.task} · 512 → 256 → 128 ELU</span>
                </div>
              </div>
              <label className={styles.field}>
                <span>Run name</span>
                <input
                  value={recipe.runName}
                  onChange={(event) => setRecipe((current) => ({ ...current, runName: event.target.value }))}
                  pattern="[A-Za-z0-9_-]+"
                  maxLength={48}
                  required
                />
              </label>
              <label className={styles.field}>
                <span>Saved runs</span>
                <select aria-label="Saved runs" value="" disabled={busy} onChange={(event) => { if (event.target.value) void loadSavedRun(recipe.experimentId, event.target.value); }}>
                  <option value="">Load a run and its saved evaluation recipe…</option>
                  {savedRuns.filter((run) => run.experimentId === recipe.experimentId).map((run) => <option key={run.runName} value={run.runName}>{run.taskPassed ? "Passed" : run.skillAssessed ? "Failed" : "Not assessed"} · {run.runName}</option>)}
                </select>
              </label>
              <p className={styles.evidenceNote}>Typing a run name selects artifacts only. Use Saved runs to restore the exact saved evaluation settings, including clip, command, horizon, and PPO parameters. The checkpoint metadata records training provenance.</p>
              {job.artifacts.checkpoint && <p className={styles.evidenceNote}>This run already has a checkpoint. Choose a new run name for fresh training, or explicitly enable Continue current checkpoint in Advanced settings. Timesteps on continuation are additional, not the lifetime total.</p>}
              {recipe.experimentId === "dance" && <section className={styles.clipSelection}>
                <label className={styles.field}>
                  <span>Dance reference clip</span>
                  <select aria-label="Dance reference clip" value={recipe.danceClip ?? ""} onChange={(event) => {
                    const selected = danceClips.find((clip) => clip.path === event.target.value);
                    setRecipe((current) => ({ ...current, danceClip: selected?.path ?? null, ...(current.profile === "full" ? {
                      maxEpisodeS: selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds,
                      evalSteps: Math.ceil((selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds) * 50),
                      renderSeconds: selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds,
                    } : {}) }));
                  }}>
                    <option value="">Built-in default choreography</option>
                    {recipe.danceClip && !danceClips.some((clip) => clip.path === recipe.danceClip) && <option value={recipe.danceClip}>Saved reference · {recipe.danceClip.split("/").at(-1)}</option>}
                    {danceClips.map((clip) => <option key={clip.path} value={clip.path}>{clip.name} · {clip.durationSeconds.toFixed(2)} s</option>)}
                  </select>
                </label>
                <p>Joint-angle reference for PPO, evaluation, and rendering—not an animation-only preview. In Full mode, selection sets the episode, evaluation, and video horizons to one complete clip. Changing the clip does not retrain an existing checkpoint or change its saved verdict. Catalog clips are not guaranteed physically learnable; train and verify each one.</p>
                <code>{recipe.danceClip ?? "Built-in reference"}</code>
                {clipError && <p role="alert">{clipError}</p>}
              </section>}
              <div className={styles.segmented} aria-label="Training profile">
                <button
                  type="button"
                  className={recipe.profile === "smoke" ? styles.selected : ""}
                  onClick={() => setRecipe((current) => ({
                    ...current,
                    profile: "smoke",
                    totalTimesteps: 4,
                    numEnvs: 2,
                    numSteps: 2,
                    numMinibatches: 1,
                    maxEpisodeS: 1,
                    evalSteps: 4,
                    domainRand: false,
                    obsNoise: false,
                    actionDelay: false,
                    randomYaw: false,
                  }))}
                >
                  Pipeline smoke<small>4 steps · wiring only</small>
                </button>
                <button
                  type="button"
                  className={
                    recipe.profile === "full" &&
                    recipe.totalTimesteps === selectedExperiment.fullTimesteps &&
                    recipe.domainRand &&
                    recipe.obsNoise &&
                    recipe.actionDelay
                      ? styles.selected
                      : ""
                  }
                  onClick={() => setRecipe((current) => ({
                    ...current,
                    profile: "full",
                    ...fullHorizon(current.experimentId, selectedDanceDuration),
                    totalTimesteps: selectedExperiment.fullTimesteps,
                    numEnvs: selectedExperiment.fullEnvs,
                    domainRand: true,
                    obsNoise: true,
                    actionDelay: true,
                    randomYaw: true,
                  }))}
                >
                  Default full<small>{formatNumber(selectedExperiment.fullTimesteps)} steps · randomized</small>
                </button>
              </div>
              {recipe.experimentId === "swing" && (
                <section className={styles.swingPlan} aria-labelledby="swing-plan-title">
                  <div className={styles.swingPlanHead}>
                    <div>
                      <p className={styles.eyebrow}>RECOVERY PLAN</p>
                      <h3 id="swing-plan-title">Build motion before spending 4M steps</h3>
                    </div>
                    <span className={swingTargetPassed ? styles.planPass : styles.planReject}>
                      {recipeMetric == null
                        ? "No evaluation yet"
                        : !verdict.skillAssessed
                          ? "Skill not assessed"
                          : swingTargetPassed
                            ? `${recipeMetric.toFixed(2)}° accepted`
                            : `${recipeMetric.toFixed(2)}° rejected`}
                    </span>
                  </div>
                  <div className={styles.swingSettings} aria-label="Exact Swing recovery settings">
                    <div><strong>Setting</strong><b>Discovery</b><b>Consolidation</b></div>
                    <div><span>Run</span><code>swing-curriculum-01</code><code>same checkpoint</code></div>
                    <div><span>New PPO steps</span><code>250,000</code><code>500,000</code></div>
                    <div><span>Parallel envs / seed</span><code>16 / 2</code><code>16 / 2</code></div>
                    <div><span>Initial angle / rate</span><code>12° / 0.35 rad/s</code><code>4° / 0.12 rad/s</code></div>
                    <div><span>Randomization / noise / delay</span><code>off / off / off</code><code>on / on / on</code></div>
                    <div><span>LR / gamma / clip</span><code>1e-4 / .995 / .10</code><code>same</code></div>
                    <div><span>Epochs / entropy / grad norm</span><code>3 / .002 / 1.0</code><code>same</code></div>
                  </div>
                  <ol className={styles.swingPlanSteps}>
                    <li>
                      <b>1</b>
                      <div>
                        <strong>Preserve the failed baseline</strong>
                        <p>Keep <code>swing-studio-01</code>. Use a new run named <code>swing-curriculum-01</code>.</p>
                      </div>
                    </li>
                    <li>
                      <b>2</b>
                      <div>
                        <strong>Discovery: 250k assisted steps</strong>
                        <p>16 envs, seed 2, 12° initial angle, 0.35 rad/s initial rate, randomization/noise/delay off, default Swing rewards.</p>
                        <button type="button" className={styles.textButton} onClick={applySwingDiscovery}>Apply exact Discovery settings</button>
                      </div>
                    </li>
                    <li>
                      <b>3</b>
                      <div>
                        <strong>Evaluate from a still start</strong>
                        <p>Run Evaluate, then Render. Evaluation uses 16 envs × 1,200 control steps (24 s), with 0° initial angle and 0 rad/s. Continue only if mean span is at least 10° or best span is at least 20°.</p>
                        {evaluation && (
                          <p className={swingDiscoveryPassed ? styles.decisionPass : styles.decisionReject}>
                            Current gate: mean {swingMeanSpan?.toFixed(2) ?? "—"}°, best {swingBestSpan?.toFixed(2) ?? "—"}° · {swingDiscoveryPassed ? "continue" : "reject and revise curriculum"}
                          </p>
                        )}
                      </div>
                    </li>
                    <li>
                      <b>4</b>
                      <div>
                        <strong>Consolidate: 500k additional steps</strong>
                        <p>Resume the same checkpoint, reduce assistance to 4° and 0.12 rad/s, then enable domain randomization, observation noise, and action delay.</p>
                        <button
                          type="button"
                          className={styles.textButton}
                          onClick={applySwingConsolidation}
                          disabled={
                            !job.artifacts.checkpoint ||
                            recipe.runName !== "swing-curriculum-01" ||
                            !swingDiscoveryPassed
                          }
                          title={
                            !job.artifacts.checkpoint
                              ? "Run Discovery first"
                              : !swingDiscoveryPassed
                                ? "Evaluate Discovery and pass the 10° mean or 20° best gate first"
                                : undefined
                          }
                        >
                          Apply exact Consolidation settings
                        </button>
                      </div>
                    </li>
                    <li>
                      <b>5</b>
                      <div>
                        <strong>Decide with evidence</strong>
                        <p>After 500k, continue toward 1M only if mean span reaches 30° or best span reaches 60°. Default skill acceptance requires a symmetric 150° span (at least 75° each side), a complete 24-second episode, valid geometry, and tensioned strings in every episode. Review the video separately; pipeline smoke is not a skill assessment.</p>
                        {evaluation && (
                          <p className={swingConsolidationPassed ? styles.decisionPass : styles.decisionReject}>
                            1M gate: {swingConsolidationPassed ? "eligible to continue" : "do not extend yet"}
                          </p>
                        )}
                      </div>
                    </li>
                  </ol>
                </section>
              )}
              <button type="button" className={styles.advancedButton} onClick={() => setAdvanced((open) => !open)}>
                <span>Advanced PPO and environment settings</span><b>{advanced ? "−" : "+"}</b>
              </button>
              {advanced && (
                <div className={styles.advancedSettings}>
                  <div className={styles.advancedGrid}>
                    {([
                      ["numSteps", "Rollout steps per environment", 1, 100000, 1],
                      ["numMinibatches", "Minibatches per PPO epoch", 1, 100000, 1],
                      ["maxEpisodeS", "Episode horizon (seconds)", 0.02, 3600, 0.02],
                      ["evalSteps", "Evaluation steps per environment", 1, 10000000, 1],
                      ["renderSeconds", "Video horizon (seconds)", 0.02, 3600, 0.02],
                      ["initialStd", "Initial exploration standard deviation", 0.001, 10, 0.01],
                      ["checkpointInterval", "Checkpoint interval (transitions; 0 disables)", 0, 40000000, 1],
                    ] as const).map(([key, label, minimum, maximum, step]) => <label className={styles.field} key={key}><span>{label}</span><input type="number" min={minimum} max={maximum} step={step} value={recipe[key]} onChange={(event) => setRecipe((current) => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}
                    <Toggle label="Normalize rewards" checked={recipe.normalizeRewards} onChange={(checked) => setRecipe((current) => ({ ...current, normalizeRewards: checked }))} />
                    <Toggle label="Freeze observation normalization" checked={recipe.freezeObservationNormalization} onChange={(checked) => setRecipe((current) => ({ ...current, freezeObservationNormalization: checked }))} />
                    {recipe.experimentId !== "swing" && <Toggle label="Continue current checkpoint" checked={recipe.resumeFromCheckpoint} onChange={(checked) => setRecipe((current) => ({ ...current, resumeFromCheckpoint: checked }))} />}
                    {recipe.experimentId === "dance" && <label className={styles.field}><span>Dance pose sigma (blank uses default)</span><input type="number" min="0.001" step="0.01" value={recipe.dancePoseSigma ?? ""} onChange={(event) => setRecipe((current) => ({ ...current, dancePoseSigma: event.target.value === "" ? null : Number(event.target.value) }))} /></label>}
                    {(recipe.experimentId === "running" || recipe.experimentId === "stilts") && <label className={styles.field}><span>Fixed forward command (m/s; blank samples commands)</span><input type="number" min="0.01" max="1.5" step="0.01" value={recipe.locomotionForwardCommand ?? ""} onChange={(event) => setRecipe((current) => ({ ...current, locomotionForwardCommand: event.target.value === "" ? null : Number(event.target.value) }))} /></label>}
                    <label className={styles.field}><span>Total timesteps</span><input type="number" min="4" max="40000000" value={recipe.totalTimesteps} onChange={(event) => setRecipe((current) => ({ ...current, totalTimesteps: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Parallel envs</span><input type="number" min="1" max="64" value={recipe.numEnvs} onChange={(event) => setRecipe((current) => ({ ...current, numEnvs: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Learning rate</span><input type="number" min="0.000001" max="0.1" step="0.0001" value={recipe.learningRate} onChange={(event) => setRecipe((current) => ({ ...current, learningRate: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Gamma</span><input type="number" min="0.8" max="1" step="0.01" value={recipe.gamma} onChange={(event) => setRecipe((current) => ({ ...current, gamma: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>PPO clip</span><input type="number" min="0.01" max="1" step="0.01" value={recipe.clipCoefficient} onChange={(event) => setRecipe((current) => ({ ...current, clipCoefficient: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Update epochs</span><input type="number" min="1" max="20" step="1" value={recipe.updateEpochs} onChange={(event) => setRecipe((current) => ({ ...current, updateEpochs: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Entropy coefficient</span><input type="number" min="0" max="1" step="0.001" value={recipe.entropyCoefficient} onChange={(event) => setRecipe((current) => ({ ...current, entropyCoefficient: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>Max gradient norm</span><input type="number" min="0.01" max="10" step="0.1" value={recipe.maxGradNorm} onChange={(event) => setRecipe((current) => ({ ...current, maxGradNorm: Number(event.target.value) }))} /></label>
                    <Toggle label="Domain randomization" checked={recipe.domainRand} onChange={(checked) => setRecipe((current) => ({ ...current, domainRand: checked }))} />
                    <Toggle label="Observation noise" checked={recipe.obsNoise} onChange={(checked) => setRecipe((current) => ({ ...current, obsNoise: checked }))} />
                    <Toggle label="Action delay" checked={recipe.actionDelay} onChange={(checked) => setRecipe((current) => ({ ...current, actionDelay: checked }))} />
                    <Toggle label="Random yaw" checked={recipe.randomYaw} onChange={(checked) => setRecipe((current) => ({ ...current, randomYaw: checked }))} />
                    {recipe.experimentId === "stilts" && (
                      <>
                        <label className={styles.field}><span>Stilt height (cm)</span><input type="number" min="0.8" max="300" step="0.5" value={recipe.stiltHeightCm} onChange={(event) => setRecipe((current) => ({ ...current, stiltHeightCm: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>Support blend</span><input type="number" min="0" max="1" step="0.05" value={recipe.stiltBlend} onChange={(event) => setRecipe((current) => ({ ...current, stiltBlend: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>Mass per stilt (kg)</span><input type="number" min="0.001" max="2" step="0.001" value={recipe.stiltMassKg} onChange={(event) => setRecipe((current) => ({ ...current, stiltMassKg: Number(event.target.value) }))} /></label>
                      </>
                    )}
                    {recipe.experimentId === "swing" && (
                      <>
                        <label className={styles.field}><span>Initial angle assistance (degrees)</span><input type="number" min="0" max="30" step="1" value={recipe.swingInitialAngleDeg} onChange={(event) => setRecipe((current) => ({ ...current, swingInitialAngleDeg: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>Initial rate assistance (rad/s)</span><input type="number" min="0" max="1" step="0.01" value={recipe.swingInitialRateRadS} onChange={(event) => setRecipe((current) => ({ ...current, swingInitialRateRadS: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>Skill target: symmetric total span (degrees)</span><input type="number" min="1" max="180" step="1" value={recipe.swingMinSpanDeg} onChange={(event) => setRecipe((current) => ({ ...current, swingMinSpanDeg: Number(event.target.value) }))} /></label>
                        <Toggle label="Planar discovery actions" checked={recipe.swingPlanarActions} onChange={(checked) => setRecipe((current) => ({ ...current, swingPlanarActions: checked }))} />
                        <Toggle label="Continue current checkpoint" checked={recipe.resumeFromCheckpoint} onChange={(checked) => setRecipe((current) => ({ ...current, resumeFromCheckpoint: checked }))} />
                      </>
                    )}
                  </div>
                  <section className={styles.rewardEditor} aria-labelledby="reward-weights-title">
                    <div className={styles.rewardEditorHead}>
                      <div>
                        <h3 id="reward-weights-title">Reward weights</h3>
                        <p>Absolute non-negative coefficients. Penalty measurements are already negative.</p>
                      </div>
                      <button
                        type="button"
                        className={styles.textButton}
                        onClick={() => setRecipe((current) => ({
                          ...current,
                          rewardWeights: defaultRewardWeights(current.experimentId),
                        }))}
                      >
                        Reset defaults
                      </button>
                    </div>
                    <div className={styles.rewardWeightGrid}>
                      {selectedExperiment.reward.terms
                        .filter((term) => term.editable !== false)
                        .map((term) => (
                          <label className={styles.rewardWeightRow} key={term.key}>
                            <span>
                              <strong>{term.label}</strong>
                              <small>{term.penalty ? "Penalty" : "Reward"} · {term.key}</small>
                            </span>
                            <input
                              type="number"
                              min="0"
                              max="10000"
                              step="any"
                              value={recipe.rewardWeights[term.key] ?? term.defaultWeight}
                              onChange={(event) => {
                                const value = Math.max(0, Number(event.target.value));
                                setRecipe((current) => ({
                                  ...current,
                                  rewardWeights: {
                                    ...current.rewardWeights,
                                    [term.key]: Number.isFinite(value) ? value : term.defaultWeight,
                                  },
                                }));
                              }}
                              aria-label={`${term.label} reward weight`}
                            />
                          </label>
                        ))}
                    </div>
                    <p className={styles.rewardRetrainNote}>Changing a reward weight defines a new experiment. Start a new training run before evaluating or rendering it.</p>
                  </section>
                </div>
              )}
              <div className={styles.recipeActions}>
                <button
                  type="button"
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("train")}
                  disabled={busy || job.phase === "running" || anotherRunActive}
                  aria-describedby="rlx-action-status"
                >
                  <Icon>{pendingAction === "train" ? "…" : "▶"}</Icon> {trainButtonLabel}
                </button>
                {recipe.experimentId === "dance" ? (
                  <button type="button" onClick={startLabTraining} className={styles.button} disabled={busy || !connected} title="Separate free-text teaching workflow; does not use this RLX clip or recipe">
                    <Icon>⌁</Icon> Separate Duck Lab teaching
                  </button>
                ) : (
                  <a
                    className={styles.button}
                    href={`/guides/microduck-studio-experiments-guide.pdf#${selectedExperiment.guideAnchor}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Icon>?</Icon> Open recipe guide
                  </a>
                )}
              </div>
              <p
                id="rlx-action-status"
                className={`${styles.recipeActionStatus} ${
                  job.phase === "failed" && job.operation === "train"
                    ? styles.recipeActionError
                    : ""
                }`}
                role="status"
                aria-live="polite"
              >
                <i
                  className={
                    pendingAction === "train" || job.phase === "running"
                      ? styles.statusWorking
                      : job.phase === "failed"
                        ? styles.statusError
                        : styles.statusReady
                  }
                />
                <span>
                  {recipeActionStatus ??
                    "Ready to launch this recipe with the selected training settings."}
                </span>
              </p>
              <p className={styles.honesty}>Smoke mode validates the pipeline only. A learned {selectedExperiment.shortTitle.toLowerCase()} policy requires full training plus deterministic and visual review.</p>
            </form>
          </section>

          <section className={styles.panel} id="evaluation">
            <div className={styles.panelHead}>
              <div>
                <h2><Icon>⌁</Icon>Evaluation gate</h2>
                <p>Smoke checks the pipeline (4 steps); full evaluates the skill. Swing requires all 1,200 steps (24 s).</p>
                {evaluation && <p>Saved evaluation scope: {verdict.scope ?? "unknown (legacy report)"}{verdict.swingMinSpanDeg == null ? "" : ` · target ${verdict.swingMinSpanDeg}° total / ${verdict.swingMinSpanDeg / 2}° each side`}. Current recipe edits do not change saved results.</p>}
              </div>
              <button className={styles.textButton} onClick={() => runAction("eval")} disabled={busy || job.phase === "running" || !job.artifacts.checkpoint}>
                Run evaluation ↗
              </button>
            </div>
            <div className={styles.evaluationBody}>
              <section className={styles.skillSummary} aria-label="Skill verification summary">
                <h3>{taskPassed ? "Skill verified in simulation" : verdict.skillAssessed ? "Skill criteria failed" : "Skill not yet verified"}</h3>
                <p><strong>{evidence.passed} / {evidence.episodes.length} episodes passed</strong> · {recipe.runName}</p>
                <p>Acceptance requires every complete episode, not just a high average reward. The verdict is bound to current policy bytes; preview clips are not training evidence.</p>
                {selectedExperimentId === "swing" && <p>Swing acquisition used teacher BC/DAgger initialization followed by PPO refinement. This is not a demonstrated pure-PPO-from-scratch success.</p>}
                {selectedExperimentId === "dance" && <p>Dance verifies the saved reference horizon and joint-angle tracking, not automatic imitation of the entire source video.</p>}
                {typeof evaluation?.source_sha256 === "string" && <p>Evaluated source SHA-256: <code>{evaluation.source_sha256}</code></p>}
                {evidence.ranges.length > 0 && <table aria-label="Measured skill ranges"><thead><tr><th>Measurement</th><th>Episode minimum</th><th>Episode maximum</th></tr></thead><tbody>
                  {evidence.ranges.map((range) => <tr key={range.metric}><td>{evidenceLabel(range.metric)}<small> · {range.measured}/{evidence.episodes.length} measured</small></td><td>{range.minimum.toFixed(4)}</td><td>{range.maximum.toFixed(4)}</td></tr>)}
                </tbody></table>}
                <details><summary>Exact acceptance criteria and per-episode results</summary>
                  <table><tbody>{Object.entries(evidence.criteria).map(([key, value]) => <tr key={key}><th>{evidenceLabel(key)}</th><td>{JSON.stringify(value)}</td></tr>)}</tbody></table>
                  {evidence.episodes.map((episode, index) => <details key={index}><summary>Environment {String(episode.env_index)} · episode {String(episode.episode_index)} · {episode.passed === true ? "Passed" : "Failed"} · {String(episode.measured_steps)} measured steps</summary><pre>{JSON.stringify(episode, null, 2)}</pre></details>)}
                </details>
                <details><summary>Saved evaluation recipe and raw report</summary><pre>{JSON.stringify(evaluation, null, 2)}</pre></details>
              </section>
              {job.artifacts.renderVideo && (
                <section className={styles.rolloutPlayer} aria-labelledby="rollout-player-title">
                  <div>
                    <p className={styles.eyebrow}>SELECTED RUN · SAVED VIDEO</p>
                    <h3 id="rollout-player-title">{recipe.runName} rollout</h3>
                    <p>{job.renderVerified ? "Policy, clip, and media hashes match. Video settings are the nominal projection of the saved evaluation recipe (randomization disabled); this does not visualize every evaluation domain. Watch the complete rollout before marking reviewed." : "Unbound or stale video: re-render using the saved evaluation recipe before marking reviewed. Legacy videos cannot satisfy the package gate."}</p>
                  </div>
                  <video
                    key={reviewIdentity}
                    controls
                    playsInline
                    preload="metadata"
                    src={`${artifactHref("video")}&inline=1&source=${encodeURIComponent(String(evaluation?.source_sha256 ?? "unknown"))}`}
                  >
                    Your browser cannot play this MP4.
                  </video>
                  <div className={styles.rolloutPlayerActions}>
                    <a className={styles.button} href={`${artifactHref("video")}&inline=1`} target="_blank" rel="noreferrer">
                      <Icon>▶</Icon> Open full video
                    </a>
                    <a className={styles.button} href={artifactHref("sheet")} target="_blank" rel="noreferrer">
                      <Icon>▦</Icon> Open contact sheet
                    </a>
                  </div>
                </section>
              )}
              <table>
                <thead><tr><th>Check</th><th>Evidence</th><th>Result</th></tr></thead>
                <tbody>
                  <tr><td><i className={job.artifacts.checkpoint ? styles.passDot : styles.mutedDot} />Checkpoint</td><td>Safetensors + sidecar</td><td>{job.artifacts.checkpoint ? "Ready" : "Missing"}</td></tr>
                  <tr><td><i className={job.artifacts.onnx ? styles.passDot : styles.mutedDot} />ONNX contract</td><td>obs[batch,61] → actions[batch,14]</td><td>{job.artifacts.onnx ? "Ready" : "Missing"}</td></tr>
                  <tr><td><i className={evalPassed ? styles.passDot : styles.warnDot} />Rollout validity</td><td>Finite observations, rewards, and actions</td><td>{typeof evaluation?.pipeline_passed === "boolean" ? (evalPassed ? "Passed" : "Failed") : "Unavailable"}</td></tr>
                  <tr><td><i className={taskPassed ? styles.passDot : styles.warnDot} />{selectedExperiment.metricLabel}</td><td>{recipeMetric == null ? "Not measured" : `${recipeMetric.toFixed(3)}${selectedExperiment.metricSuffix}`}</td><td>{!verdict.skillAssessed ? "Not assessed" : taskPassed ? "Accepted" : "Failed"}</td></tr>
                  <tr><td><i className={job.artifacts.renderSheet && job.artifacts.renderVideo ? styles.passDot : styles.warnDot} />Visual artifact</td><td>MP4 + contact sheet (separate visual review)</td><td>{job.artifacts.renderSheet && job.artifacts.renderVideo ? "Ready" : "Required"}</td></tr>
                  <tr><td><i className={job.renderVerified ? styles.passDot : styles.warnDot} />Video provenance</td><td>Source, clip, settings and media hashes</td><td>{job.renderVerified ? "Matched" : "Re-render required"}</td></tr>
                </tbody>
              </table>
              <div className={styles.evaluationActions}>
                <button className={styles.button} onClick={() => runAction("render")} disabled={busy || job.phase === "running" || !job.artifacts.checkpoint}>
                  <Icon>◫</Icon> Render rollout
                </button>
                {!job.artifacts.onnx && job.artifacts.checkpoint && (
                  <button className={styles.button} onClick={() => runAction("export")} disabled={busy || job.phase === "running"}>
                    <Icon>↓</Icon> Export ONNX
                  </button>
                )}
                {job.artifacts.renderVideo && <a className={styles.button} href={`${artifactHref("video")}&inline=1`} target="_blank" rel="noreferrer"><Icon>▶</Icon> Play rollout</a>}
                {job.artifacts.renderSheet && <a className={styles.button} href={artifactHref("sheet")} target="_blank" rel="noreferrer"><Icon>▦</Icon> Contact sheet</a>}
              </div>
              <label className={styles.reviewCheck}>
                <input type="checkbox" checked={visualReviewed && Boolean(job.renderVerified)} onChange={(event) => setVisualReviewed(event.target.checked)} disabled={!job.renderVerified} />
                <span><strong>I reviewed the rendered motion</strong><small>Confirm the policy moves as intended and outperforms null controls.</small></span>
              </label>
            </div>
          </section>

          <section className={styles.panel} id="deployment">
            <div className={styles.panelHead}>
              <h2><Icon>♢</Icon>Deployment handoff</h2>
              <StatusPill tone={deployReady ? "live" : "warn"}>{deployReady ? "PACKAGE READY" : "GATED"}</StatusPill>
            </div>
            <div className={styles.deployBody}>
              <div className={styles.deployArtifact}>
                <span className={styles.fileIcon}>ONNX</span>
                <div><strong>{selectedExperiment.artifactStem}.onnx</strong><small>Normalizer baked into the graph</small></div>
                <span>{job.artifacts.onnx ? "Ready" : "Pending"}</span>
              </div>
              <ul className={styles.gateList}>
                <li className={job.artifacts.onnx ? styles.complete : ""}>61-observation / 14-action contract</li>
                <li className={taskPassed ? styles.complete : ""}>
                  {selectedExperimentId === "swing"
                    ? "Swing skill passed: saved symmetric target, 24 s, geometry and string tension"
                    : "Deterministic evaluation passed"}
                </li>
                <li className={job.artifacts.renderSheet && visualReviewed ? styles.complete : ""}>Rendered motion reviewed</li>
                <li>Complete hardware-specific validation before deployment</li>
              </ul>
              <div className={styles.deployActions}>
                <a
                  className={`${styles.button} ${deployReady ? styles.primary : styles.disabled}`}
                  href={deployReady ? artifactHref("onnx") : undefined}
                  aria-disabled={!deployReady}
                  onClick={(event) => {
                    if (!deployReady) event.preventDefault();
                  }}
                >
                  <Icon>↓</Icon> Download policy
                </a>
                <button className={styles.button} disabled title="Physical robot transport is not configured">
                  <Icon>⌾</Icon> Send to robot
                </button>
              </div>
              <p className={styles.safetyNote}><Icon>✓</Icon><span>Local RLX output is a prototyping artifact. Hardware deployment remains disabled until an authenticated robot transport and hardware-specific validation are configured.</span></p>
            </div>
          </section>
        </div>

        <section className={`${styles.panel} ${styles.artifactPanel}`} id="artifacts">
          <div className={styles.panelHead}>
            <h2><Icon>▱</Icon>Run artifacts</h2>
            <span>{job.artifacts.checkpointPath}</span>
          </div>
          <div className={styles.artifactGrid}>
            {[
              ["Checkpoint", "checkpoint", job.artifacts.checkpoint, "Training weights"],
              ["Metadata", "metadata", job.artifacts.metadata, "Recipe and contract"],
              ["ONNX policy", "onnx", deployReady, "Simulation graph · requires matched visual review"],
              ["Contact sheet", "sheet", job.artifacts.renderSheet, "Visual verification"],
              ["Rollout video", "video", job.artifacts.renderVideo, "Motion review"],
            ].map(([label, kind, ready, detail]) => (
              <a
                key={String(kind)}
                className={`${styles.artifact} ${ready ? styles.ready : ""}`}
                href={ready ? artifactHref(kind as "checkpoint" | "metadata" | "onnx" | "sheet" | "video") : undefined}
                aria-disabled={!ready}
                onClick={(event) => {
                  if (!ready) event.preventDefault();
                }}
              >
                <Icon>{ready ? "✓" : "·"}</Icon>
                <span><strong>{label}</strong><small>{detail}</small></span>
                <b>{ready ? "↓" : "Pending"}</b>
              </a>
            ))}
          </div>
        </section>

        <section className={styles.systemStrip} id="system">
          <div><Icon>⌘</Icon><span><strong>Apple Silicon</strong><small>MLX · Metal learning path</small></span><b>Local</b></div>
          <div><Icon>◇</Icon><span><strong>MuJoCo physics</strong><small>{connected ? "Duck Lab streaming" : "Waiting on :8788"}</small></span><b>{connected ? "Live" : "Offline"}</b></div>
          <div><Icon>⌾</Icon><span><strong>Physical robot</strong><small>Authenticated transport</small></span><b className={styles.mutedText}>Not configured</b></div>
          <div><Icon>▱</Icon><span><strong>Policy library</strong><small>Assignable local runs</small></span><b>{activePolicies}</b></div>
        </section>

        <footer className={styles.footer}>
          <span><i className={styles.onlineDot} />Microduck Studio</span>
          <span>RLX PPO · 61 observations · 14 actions</span>
          <span>Local-first · Hardware gated</span>
        </footer>
      </main>

      <dialog
        ref={rewardHistoryDialogRef}
        className={styles.rewardHistoryDialog}
        aria-labelledby="reward-history-title"
        onClose={() => {
          setRewardHistoryOpen(false);
          setRewardHistoryCopied(false);
        }}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            event.currentTarget.close();
          }
        }}
      >
        <div className={styles.rewardHistoryContent}>
          <header className={styles.rewardHistoryHeader}>
            <div>
              <p className={styles.eyebrow}>TRAINING TELEMETRY</p>
              <h2 id="reward-history-title">{rewardHistoryLabel}</h2>
              <p>
                {chartPoints.length
                  ? `${chartPoints.length} available ${rewardIsNormalized ? "normalized " : ""}rollout samples for ${recipe.runName}.`
                  : `No rollout samples are available for ${recipe.runName} yet.`}
              </p>
            </div>
            <button
              type="button"
              className={styles.dialogClose}
              aria-label="Close reward history"
              onClick={() => rewardHistoryDialogRef.current?.close()}
            >
              ×
            </button>
          </header>
          <div className={styles.rewardHistoryActions}>
            <span>Tab-separated text · paste directly into a spreadsheet</span>
            <button
              type="button"
              className={`${styles.button} ${chartPoints.length ? styles.primary : ""}`}
              disabled={!chartPoints.length}
              onClick={copyRewardHistory}
            >
              <Icon>{rewardHistoryCopied ? "✓" : "▣"}</Icon>
              {rewardHistoryCopied ? "Copied" : "Copy table"}
            </button>
          </div>
          <div className={styles.rewardHistoryTableWrap}>
            <table className={styles.rewardHistoryTable}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Training step</th>
                  <th>{rewardIsNormalized ? "Normalized mean rollout reward" : "Mean rollout reward"}</th>
                </tr>
              </thead>
              <tbody>
                {chartPoints.length ? (
                  chartPoints.map((point, index) => (
                    <tr key={`${point.step}-${index}`}>
                      <td>{index + 1}</td>
                      <td>{formatNumber(point.step)}</td>
                      <td>{point.reward}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3}>Start training to collect reward samples.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </dialog>

      <dialog
        ref={guidanceDialogRef}
        className={styles.recipeGuidanceDialog}
        aria-labelledby="recipe-guidance-title"
        onClose={() => setGuidanceExperimentId(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            event.currentTarget.close();
          }
        }}
      >
        {guidanceExperiment && (
          <div className={styles.recipeGuidanceContent}>
            <header className={styles.recipeGuidanceHeader}>
              <div>
                <p className={styles.eyebrow}>RECIPE GUIDANCE</p>
                <h2 id="recipe-guidance-title">{guidanceExperiment.title}</h2>
                <p>{guidanceExperiment.goal}</p>
              </div>
              <button
                type="button"
                className={styles.dialogClose}
                aria-label="Close recipe guidance"
                onClick={() => guidanceDialogRef.current?.close()}
              >
                ×
              </button>
            </header>

            <section className={styles.guidanceInput}>
              <strong>Required input</strong>
              <p>{guidanceExperiment.guidance.requiredInput}</p>
            </section>

            <section className={styles.guidanceSteps}>
              <h3>Train, evaluate, and render</h3>
              <ol>
                {guidanceExperiment.guidance.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </section>

            <section className={styles.guidanceConcepts}>
              <h3>What clip, command, and PPO mean here</h3>
              <ul>
                {guidanceExperiment.guidance.distinctions.map((distinction) => (
                  <li key={distinction}>{distinction}</li>
                ))}
              </ul>
            </section>

            <section className={styles.guidanceReward}>
              <h3>PPO reward function</h3>
              <p>{guidanceExperiment.reward.summary}</p>
              <code>{guidanceExperiment.reward.formula}</code>
              <p className={styles.rewardRule}>
                Each measured term is multiplied by its weight and summed every
                control step. Penalty terms already return negative values, so
                their editable weights stay non-negative.
              </p>
              <div className={styles.guidanceRewardTerms}>
                {guidanceExperiment.reward.terms.map((term) => (
                  <div key={term.key}>
                    <span>
                      <strong>{term.label}</strong>
                      <small>{term.description}</small>
                    </span>
                    <b>
                      {term.editable === false ? "dynamic x " : ""}
                      {term.defaultWeight}
                    </b>
                  </div>
                ))}
              </div>
              <p className={styles.rewardRule}>
                Customize editable weights in Advanced PPO and environment
                settings. Retrain after any change; the checkpoint metadata
                records the chosen weights.
              </p>
            </section>

            <footer className={styles.guidanceOutput}>
              <strong>Outputs</strong>
              <p>{guidanceExperiment.guidance.output}</p>
            </footer>
          </div>
        )}
      </dialog>
    </div>
  );
}
