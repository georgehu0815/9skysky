import {
  existsSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";

export type RlxRewardNormalization = "raw" | "normalized" | "unknown";

export interface RlxTrainingCollectionSample {
  step: number;
  meanReward: number;
}

export interface RlxTrainingUpdateSample {
  step: number;
  meanLoss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
  entropy: number | null;
  approximateKl: number | null;
  clipFraction: number | null;
  explainedVariance: number | null;
}

export interface RlxTrainingEpisodeSample {
  step: number;
  meanRawReturn: number;
  returns: number[];
  lengths: number[];
}

export interface RlxTrainingSummarySample {
  step: number;
  meanReward: number | null;
  meanRawReturn: number | null;
  meanLoss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
}

export interface RlxTrainingInitializer {
  kind: "bc-dagger";
  label: string;
  teacherSamples: number | null;
  daggerIteration: number | null;
  normalizer: string | null;
}

export interface RlxTrainingSegment {
  id: string;
  kind: "ppo" | "bc-initializer";
  status: "active" | "complete" | "initializer";
  startStep: number;
  endStep: number;
  trainedSteps: number;
  normalization: RlxRewardNormalization;
  normalizationLabel: string;
  collections: RlxTrainingCollectionSample[];
  updates: RlxTrainingUpdateSample[];
  episodes: RlxTrainingEpisodeSample[];
  initialSample: RlxTrainingSummarySample | null;
  finalSample: RlxTrainingSummarySample | null;
  initializer: RlxTrainingInitializer | null;
}

export interface RlxTrainingHistory {
  segments: RlxTrainingSegment[];
}

export interface RlxActiveTrainingHistory {
  id: string;
  startStep: number;
  normalizeRewards: boolean;
  totalTimesteps: number;
  status: "active" | "complete";
}

export interface RlxRestoredTrainingInvocation {
  rewardHistory: Array<{ step: number; reward: number }>;
  trainingSteps: number;
  trainingTotal: number;
  normalizeRewards: boolean;
}

interface HistoryDocument {
  version: 1;
  segments: RlxTrainingSegment[];
  latestInvocation?: {
    id: string;
    totalTimesteps: number;
    normalizeRewards: boolean;
  };
}

interface MetadataEnvelope {
  metadata?: Record<string, unknown>;
}

interface JournalEvent {
  phase?: unknown;
  env_steps?: unknown;
  mean_reward?: unknown;
  mean_raw_return?: unknown;
  returns?: unknown;
  lengths?: unknown;
  mean_loss?: unknown;
  policy_loss?: unknown;
  value_loss?: unknown;
  entropy?: unknown;
  approximate_kl?: unknown;
  clip_fraction?: unknown;
  explained_variance?: unknown;
}

const HISTORY_FILE = "training-history.json";
const journalCache = new Map<string, {
  key: string;
  groups: JournalEvent[][];
}>();
const historyWriteCache = new Map<string, string>();
const fileHashCache = new Map<string, {
  key: string;
  hash: string;
}>();
const MAX_SIBLING_RUNS = 256;
const MAX_FILES_PER_RUN = 128;

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonNegativeInt(value: unknown): number {
  const parsed = finite(value);
  return parsed === null ? 0 : Math.max(0, Math.round(parsed));
}

function numbers(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number =>
        typeof item === "number" && Number.isFinite(item)
      )
    : [];
}

function normalizationFrom(metadata: Record<string, unknown> | null) {
  const ppo = metadata?.ppo;
  const value =
    typeof ppo === "object" && ppo !== null && !Array.isArray(ppo)
      ? (ppo as Record<string, unknown>).normalize_rewards
      : undefined;
  const normalization: RlxRewardNormalization =
    value === true ? "normalized" : value === false ? "raw" : "unknown";
  return {
    normalization,
    normalizationLabel:
      normalization === "normalized"
        ? "Normalized rollout reward"
        : normalization === "raw"
          ? "Raw rollout reward"
          : "Reward scale unknown",
  };
}

function safeJson(file: string): Record<string, unknown> | null {
  try {
    const value = JSON.parse(readFileSync(file, "utf8")) as unknown;
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function metadataAt(file: string): Record<string, unknown> | null {
  return (safeJson(file) as MetadataEnvelope | null)?.metadata ?? null;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" ||
    (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function cachedFileHash(file: string): string | null {
  try {
    const fileStat = statSync(file);
    if (!fileStat.isFile()) return null;
    const key = `${fileStat.size}:${fileStat.mtimeMs}:${fileStat.ctimeMs}`;
    const cached = fileHashCache.get(file);
    if (cached?.key === key) return cached.hash;
    const hash = createHash("sha256").update(readFileSync(file)).digest("hex");
    fileHashCache.set(file, { key, hash });
    return hash;
  } catch {
    return null;
  }
}

function initializationRecord(
  metadata: Record<string, unknown> | null
): Record<string, unknown> | null {
  const initialization = metadata?.initialization;
  if (
    typeof initialization !== "object" ||
    initialization === null ||
    Array.isArray(initialization)
  ) return null;
  const record = initialization as Record<string, unknown>;
  return record.kind === "checkpoint" ? record : null;
}

function verifiedParentAt(
  resolvedRoot: string,
  checkpointInput: string,
  checkpointHash: string,
  sidecarHash: string
): { checkpoint: string; metadata: Record<string, unknown> } | null {
  const sidecarInput = `${checkpointInput}.json`;
  try {
    const checkpoint = realpathSync(checkpointInput);
    const sidecar = realpathSync(sidecarInput);
    if (!isWithin(resolvedRoot, checkpoint) || !isWithin(resolvedRoot, sidecar)) {
      return null;
    }
    if (
      cachedFileHash(checkpoint) !== checkpointHash ||
      cachedFileHash(sidecar) !== sidecarHash
    ) return null;
    const envelope = safeJson(sidecar) as MetadataEnvelope | null;
    if (
      typeof envelope?.metadata !== "object" ||
      envelope.metadata === null ||
      Array.isArray(envelope.metadata)
    ) return null;
    return { checkpoint, metadata: envelope.metadata };
  } catch {
    return null;
  }
}

function copiedSiblingParent(
  root: string,
  runDirectory: string,
  record: Record<string, unknown>
): { checkpoint: string; metadata: Record<string, unknown> } | null {
  if (
    typeof record.source_checkpoint !== "string" ||
    typeof record.source_sha256 !== "string" ||
    typeof record.source_sidecar_sha256 !== "string"
  ) return null;
  try {
    const resolvedRoot = realpathSync(root);
    const resolvedRun = realpathSync(runDirectory);
    const source = realpathSync(path.resolve(record.source_checkpoint));
    if (
      !isWithin(resolvedRoot, source) ||
      path.dirname(source) !== resolvedRun
    ) return null;
    const scenarioDirectory = realpathSync(path.dirname(resolvedRun));
    if (
      !isWithin(resolvedRoot, scenarioDirectory) ||
      path.dirname(resolvedRun) !== scenarioDirectory
    ) return null;

    const matches: Array<{
      checkpoint: string;
      metadata: Record<string, unknown>;
    }> = [];
    const siblings = readdirSync(scenarioDirectory, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .sort((left, right) => left.name.localeCompare(right.name))
      .slice(0, MAX_SIBLING_RUNS);
    for (const sibling of siblings) {
      const siblingDirectory = realpathSync(
        path.join(scenarioDirectory, sibling.name)
      );
      if (
        siblingDirectory === resolvedRun ||
        path.dirname(siblingDirectory) !== scenarioDirectory
      ) continue;
      const files = readdirSync(siblingDirectory, { withFileTypes: true })
        .filter((entry) => entry.isFile() && !entry.name.endsWith(".json"))
        .sort((left, right) => left.name.localeCompare(right.name))
        .slice(0, MAX_FILES_PER_RUN);
      for (const file of files) {
        const match = verifiedParentAt(
          resolvedRoot,
          path.join(siblingDirectory, file.name),
          record.source_sha256,
          record.source_sidecar_sha256
        );
        if (!match) continue;
        matches.push(match);
        if (matches.length > 1) return null;
      }
    }
    return matches[0] ?? null;
  } catch {
    return null;
  }
}

function trustedParentMetadata(
  root: string,
  runDirectory: string,
  metadata: Record<string, unknown> | null
): { checkpoint: string; metadata: Record<string, unknown> } | null {
  const record = initializationRecord(metadata);
  if (
    !record ||
    typeof record.source_checkpoint !== "string" ||
    typeof record.source_sha256 !== "string" ||
    typeof record.source_sidecar_sha256 !== "string"
  ) return null;
  const checkpointInput = path.resolve(record.source_checkpoint);
  try {
    const resolvedRoot = realpathSync(root);
    if (!isWithin(root, checkpointInput)) return null;
    return verifiedParentAt(
      resolvedRoot,
      checkpointInput,
      record.source_sha256,
      record.source_sidecar_sha256
    ) ?? copiedSiblingParent(root, runDirectory, record);
  } catch {
    return null;
  }
}

function readJournal(file: string): JournalEvent[][] {
  if (!existsSync(file)) return [];
  let cacheKey: string;
  try {
    const fileStat = statSync(file);
    cacheKey = `${fileStat.size}:${fileStat.mtimeMs}`;
  } catch {
    return [];
  }
  const cached = journalCache.get(file);
  if (cached?.key === cacheKey) return cached.groups;

  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return [];
  }
  const groups: JournalEvent[][] = [];
  let current: JournalEvent[] = [];
  let previousStep = -1;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let event: JournalEvent;
    try {
      event = JSON.parse(line) as JournalEvent;
    } catch {
      continue;
    }
    if (
      typeof event !== "object" ||
      event === null ||
      !["collection", "update", "episodes"].includes(String(event.phase))
    ) continue;
    const step = finite(event.env_steps);
    if (step === null || step < 0) continue;
    if (previousStep >= 0 && step < previousStep && current.length) {
      groups.push(current);
      current = [];
    }
    current.push(event);
    previousStep = step;
  }
  if (current.length) groups.push(current);
  journalCache.set(file, { key: cacheKey, groups });
  return groups;
}

function summaryAt(
  fallbackStep: number,
  collection: RlxTrainingCollectionSample | undefined,
  update: RlxTrainingUpdateSample | undefined,
  episode: RlxTrainingEpisodeSample | undefined,
  edge: "initial" | "final"
): RlxTrainingSummarySample {
  const sampleSteps = [collection?.step, update?.step, episode?.step].filter(
    (step): step is number => step !== undefined
  );
  const step = sampleSteps.length
    ? edge === "initial"
      ? Math.min(...sampleSteps)
      : Math.max(...sampleSteps)
    : fallbackStep;
  return {
    step,
    meanReward: collection?.step === step ? collection.meanReward : null,
    meanRawReturn: episode?.step === step ? episode.meanRawReturn : null,
    meanLoss: update?.step === step ? update.meanLoss : null,
    policyLoss: update?.step === step ? update.policyLoss : null,
    valueLoss: update?.step === step ? update.valueLoss : null,
  };
}

function segmentFromEvents(
  events: JournalEvent[],
  id: string,
  startStep: number,
  metadata: Record<string, unknown> | null,
  status: "active" | "complete"
): RlxTrainingSegment {
  const scale = normalizationFrom(metadata);
  const collections: RlxTrainingCollectionSample[] = [];
  const updates: RlxTrainingUpdateSample[] = [];
  const episodes: RlxTrainingEpisodeSample[] = [];
  for (const event of events) {
    const step = startStep + nonNegativeInt(event.env_steps);
    if (event.phase === "collection") {
      const meanReward = finite(event.mean_reward);
      if (meanReward !== null) collections.push({ step, meanReward });
    } else if (event.phase === "update") {
      updates.push({
        step,
        meanLoss: finite(event.mean_loss),
        policyLoss: finite(event.policy_loss),
        valueLoss: finite(event.value_loss),
        entropy: finite(event.entropy),
        approximateKl: finite(event.approximate_kl),
        clipFraction: finite(event.clip_fraction),
        explainedVariance: finite(event.explained_variance),
      });
    } else {
      const meanRawReturn = finite(event.mean_raw_return);
      if (meanRawReturn !== null) {
        episodes.push({
          step,
          meanRawReturn,
          returns: numbers(event.returns),
          lengths: numbers(event.lengths).map(Math.round),
        });
      }
    }
  }
  const localEnd = events.reduce(
    (maximum, event) => Math.max(maximum, nonNegativeInt(event.env_steps)),
    0
  );
  const endStep = startStep + localEnd;
  return {
    id,
    kind: "ppo",
    status,
    startStep,
    endStep,
    trainedSteps: localEnd,
    ...scale,
    collections,
    updates,
    episodes,
    initialSample: summaryAt(
      startStep,
      collections[0],
      updates[0],
      episodes[0],
      "initial"
    ),
    finalSample: summaryAt(
      endStep,
      collections.at(-1),
      updates.at(-1),
      episodes.at(-1),
      "final"
    ),
    initializer: null,
  };
}

function initializerSegment(
  metadata: Record<string, unknown> | null
): RlxTrainingSegment | null {
  const source = metadata?.initialization;
  const loaded =
    typeof source === "object" && source !== null && !Array.isArray(source)
      ? (source as Record<string, unknown>).loaded_metadata
      : metadata;
  if (
    typeof loaded !== "object" ||
    loaded === null ||
    Array.isArray(loaded) ||
    (loaded as Record<string, unknown>).teacher_assisted !== true
  ) return null;
  const record = loaded as Record<string, unknown>;
  return {
    id: "bc-initializer",
    kind: "bc-initializer",
    status: "initializer",
    startStep: 0,
    endStep: nonNegativeInt(record.ppo_steps),
    trainedSteps: nonNegativeInt(record.ppo_steps),
    normalization: "unknown",
    normalizationLabel: "BC/DAgger initializer; no PPO reward scale",
    collections: [],
    updates: [],
    episodes: [],
    initialSample: null,
    finalSample: null,
    initializer: {
      kind: "bc-dagger",
      label:
        typeof record.bootstrap === "string"
          ? record.bootstrap
          : "Behavior cloning / DAgger initializer",
      teacherSamples: finite(record.teacher_samples),
      daggerIteration: finite(record.dagger_iteration),
      normalizer:
        typeof record.normalizer === "string" ? record.normalizer : null,
    },
  };
}

function selfResumeMetadata(
  root: string,
  runDirectory: string,
  metadata: Record<string, unknown> | null
): Record<string, unknown> | null {
  const initialization = metadata?.initialization;
  if (
    typeof initialization !== "object" ||
    initialization === null ||
    Array.isArray(initialization)
  ) return null;
  const record = initialization as Record<string, unknown>;
  if (
    record.kind !== "checkpoint" ||
    typeof record.source_checkpoint !== "string" ||
    typeof record.loaded_metadata !== "object" ||
    record.loaded_metadata === null ||
    Array.isArray(record.loaded_metadata)
  ) return null;
  const checkpointInput = path.resolve(record.source_checkpoint);
  if (!isWithin(root, checkpointInput)) return null;
  try {
    const resolvedRoot = realpathSync(root);
    const checkpoint = realpathSync(checkpointInput);
    const resolvedRun = realpathSync(runDirectory);
    if (
      !isWithin(resolvedRoot, checkpoint) ||
      path.dirname(checkpoint) !== resolvedRun
    ) return null;
    return record.loaded_metadata as Record<string, unknown>;
  } catch {
    return null;
  }
}

function embeddedLineage(
  metadata: Record<string, unknown> | null
): RlxTrainingSegment[] {
  if (!metadata) return [];
  const nested = metadata.initialization;
  const loaded =
    typeof nested === "object" && nested !== null && !Array.isArray(nested) &&
    typeof (nested as Record<string, unknown>).loaded_metadata === "object" &&
    (nested as Record<string, unknown>).loaded_metadata !== null &&
    !Array.isArray((nested as Record<string, unknown>).loaded_metadata)
      ? (nested as Record<string, unknown>).loaded_metadata as Record<string, unknown>
      : null;
  const ancestors = embeddedLineage(loaded);
  const startStep = ancestors
    .filter((segment) => segment.kind === "ppo")
    .reduce((maximum, segment) => Math.max(maximum, segment.endStep), 0);
  const endStep = nonNegativeInt(metadata.steps);
  const ppo = metadata.ppo;
  if (
    endStep <= startStep ||
    typeof ppo !== "object" ||
    ppo === null ||
    Array.isArray(ppo)
  ) return ancestors;
  return [
    ...ancestors,
    {
      id: `metadata-lineage-${startStep}-${endStep}`,
      kind: "ppo",
      status: "complete",
      startStep,
      endStep,
      trainedSteps: endStep - startStep,
      ...normalizationFrom(metadata),
      collections: [],
      updates: [],
      episodes: [],
      initialSample: null,
      finalSample: null,
      initializer: null,
    },
  ];
}

function readHistoryDocument(runDirectory: string): HistoryDocument | null {
  const document = safeJson(path.join(runDirectory, HISTORY_FILE));
  if (document?.version !== 1 || !Array.isArray(document.segments)) return null;
  return {
    version: 1,
    segments: document.segments.filter((segment): segment is RlxTrainingSegment =>
      typeof segment === "object" && segment !== null &&
      (segment as RlxTrainingSegment).kind !== undefined
    ),
    ...(typeof document.latestInvocation === "object" &&
      document.latestInvocation !== null &&
      !Array.isArray(document.latestInvocation)
      ? {
          latestInvocation:
            document.latestInvocation as HistoryDocument["latestInvocation"],
        }
      : {}),
  };
}

function readHistory(runDirectory: string): RlxTrainingSegment[] {
  return readHistoryDocument(runDirectory)?.segments ?? [];
}

function writeHistory(
  runDirectory: string,
  segments: RlxTrainingSegment[],
  latestInvocation?: HistoryDocument["latestInvocation"],
  cacheKey?: string
) {
  const destination = path.join(runDirectory, HISTORY_FILE);
  if (cacheKey && historyWriteCache.get(destination) === cacheKey) return;
  const temporary = `${destination}.tmp`;
  const document: HistoryDocument = {
    version: 1,
    segments,
    ...(latestInvocation ? { latestInvocation } : {}),
  };
  writeFileSync(temporary, `${JSON.stringify(document, null, 2)}\n`);
  renameSync(temporary, destination);
  if (cacheKey) historyWriteCache.set(destination, cacheKey);
}

function deduplicate(segments: RlxTrainingSegment[]): RlxTrainingSegment[] {
  const byId = new Map<string, RlxTrainingSegment>();
  for (const segment of segments) {
    byId.set(segment.id, segment);
  }
  return [...byId.values()];
}

function parentSegments(
  root: string,
  runDirectory: string,
  metadata: Record<string, unknown> | null,
  seen = new Set<string>()
): RlxTrainingSegment[] {
  const parent = trustedParentMetadata(root, runDirectory, metadata);
  if (!parent) return [];
  const parentDirectory = path.dirname(parent.checkpoint);
  if (parentDirectory === runDirectory || seen.has(parentDirectory)) return [];
  seen.add(parentDirectory);
  const archived = readHistory(parentDirectory);
  const metrics = path.join(parentDirectory, "training-metrics.jsonl");
  const groups = readJournal(metrics);
  const journalSteps = groups.reduce(
    (sum, group) =>
      sum + group.reduce(
        (maximum, event) => Math.max(maximum, nonNegativeInt(event.env_steps)),
        0
      ),
    0
  );
  let offset = Math.max(0, nonNegativeInt(parent.metadata.steps) - journalSteps);
  const journal = groups.map((group, index) => {
    const segment = segmentFromEvents(
      group,
      `lineage-${parentDirectory}-${offset}-${index}`,
      offset,
      parent.metadata,
      "complete"
    );
    offset = segment.endStep;
    return segment;
  });
  const initializer = initializerSegment(parent.metadata);
  return deduplicate([
    ...parentSegments(root, parentDirectory, parent.metadata, seen),
    ...archived,
    ...(initializer ? [initializer] : []),
    ...journal,
  ]);
}

export function prepareTrainingHistory(
  root: string,
  runDirectory: string,
  metadataFile: string,
  metricsFile: string
): number {
  const metadata = metadataAt(metadataFile);
  const historyDocument = readHistoryDocument(runDirectory);
  const archived = historyDocument?.segments ?? [];
  const groups = readJournal(metricsFile);
  const finalStep = nonNegativeInt(metadata?.steps);
  const journalSteps = groups.reduce(
    (sum, group) =>
      sum + group.reduce(
        (maximum, event) => Math.max(maximum, nonNegativeInt(event.env_steps)),
        0
      ),
    0
  );
  const interrupted = archived.find((segment) =>
    segment.kind === "ppo" && segment.status === "active"
  );
  const persistedInvocation = archived.find((segment) =>
    segment.kind === "ppo" &&
    segment.id === historyDocument?.latestInvocation?.id
  );
  let offset = interrupted?.startStep ?? Math.max(0, finalStep - journalSteps);
  const additions = groups.map((group, index) => {
    const persistedId = interrupted?.id ?? persistedInvocation?.id;
    const segment = segmentFromEvents(
      group,
      persistedId && index === groups.length - 1
        ? persistedId
        : persistedId
          ? `journal-${offset}-${index}`
        : `archived-${offset}-${index}`,
      offset,
      metadata,
      "complete"
    );
    offset = segment.endStep;
    return segment;
  });
  const initializer =
    initializerSegment(metadata) ??
    initializerSegment(
      trustedParentMetadata(root, runDirectory, metadata)?.metadata ?? null
    );
  const parents = parentSegments(root, runDirectory, metadata);
  const fallbackLineage = parents.some((segment) => segment.kind === "ppo")
    ? []
    : embeddedLineage(selfResumeMetadata(root, runDirectory, metadata));
  const segments = deduplicate([
    ...parents,
    ...fallbackLineage,
    ...(initializer ? [initializer] : []),
    ...archived,
    ...additions,
  ]);
  let archivedSuccessfully = false;
  if (segments.length) {
    writeHistory(
      runDirectory,
      segments,
      readHistoryDocument(runDirectory)?.latestInvocation
    );
    archivedSuccessfully = true;
  }
  if (archivedSuccessfully && existsSync(metricsFile)) {
    try {
      writeFileSync(metricsFile, "");
      journalCache.delete(metricsFile);
    } catch {
      // The trainer also truncates the journal; archival has already completed.
    }
  }
  return Math.max(finalStep, ...segments.map((segment) => segment.endStep), 0);
}

export function collectTrainingHistory(
  root: string,
  runDirectory: string,
  metadataFile: string,
  metricsFile: string,
  active: RlxActiveTrainingHistory | null
): RlxTrainingHistory {
  const metadata = metadataAt(metadataFile);
  const historyDocument = readHistoryDocument(runDirectory);
  const archived = historyDocument?.segments ?? [];
  const groups = readJournal(metricsFile);
  const journalSteps = groups.reduce(
    (sum, group) =>
      sum + group.reduce(
        (maximum, event) => Math.max(maximum, nonNegativeInt(event.env_steps)),
        0
      ),
    0
  );
  const interrupted = !active
    ? archived.find((segment) =>
        segment.kind === "ppo" && segment.status === "active"
      )
    : null;
  const persistedInvocation = !active
    ? archived.find((segment) =>
        segment.kind === "ppo" &&
        segment.id === historyDocument?.latestInvocation?.id
      )
    : null;
  let offset = active?.startStep ?? interrupted?.startStep ??
    Math.max(0, nonNegativeInt(metadata?.steps) - journalSteps);
  const current = groups.map((group, index) => {
    const persistedSegment = interrupted ?? persistedInvocation;
    const persistedNormalization =
      persistedSegment?.normalization === "normalized"
        ? true
        : persistedSegment?.normalization === "raw"
          ? false
          : undefined;
    const segmentMetadata = active || persistedNormalization !== undefined
      ? {
          ...(metadata ?? {}),
          ppo: {
            ...(typeof metadata?.ppo === "object" && metadata.ppo !== null
              ? metadata.ppo as Record<string, unknown>
              : {}),
            normalize_rewards:
              active?.normalizeRewards ?? persistedNormalization,
          },
        }
      : metadata;
    const segment = segmentFromEvents(
      group,
      active && index === groups.length - 1
        ? active.id
        : persistedInvocation && index === groups.length - 1
          ? persistedInvocation.id
        : interrupted && index === 0
          ? interrupted.id
          : `journal-${offset}-${index}`,
      offset,
      segmentMetadata,
      active?.status ?? "complete"
    );
    offset = segment.endStep;
    return segment;
  });
  const parent = trustedParentMetadata(root, runDirectory, metadata);
  const initializer =
    initializerSegment(metadata) ??
    initializerSegment(parent?.metadata ?? null);
  const parents = parentSegments(root, runDirectory, metadata);
  const fallbackLineage = parents.some((segment) => segment.kind === "ppo")
    ? []
    : embeddedLineage(selfResumeMetadata(root, runDirectory, metadata));
  const segments = deduplicate([
    ...parents,
    ...fallbackLineage,
    ...(initializer ? [initializer] : []),
    ...archived,
    ...current,
  ]);
  if (active && segments.length && current.length) {
    const tail = current.at(-1);
    writeHistory(
      runDirectory,
      segments,
      {
        id: active.id,
        totalTimesteps: active.totalTimesteps,
        normalizeRewards: active.normalizeRewards,
      },
      tail
        ? [
            active.id,
            active.status,
            tail.endStep,
            tail.collections.length,
            tail.updates.length,
            tail.episodes.length,
          ].join(":")
        : undefined
    );
  }
  return { segments };
}

export function restoreTrainingInvocation(
  runDirectory: string,
  metadataFile: string,
  history: RlxTrainingHistory
): RlxRestoredTrainingInvocation {
  const document = readHistoryDocument(runDirectory);
  const ppoSegments = history.segments.filter((segment) => segment.kind === "ppo");
  const segment =
    ppoSegments.find((candidate) =>
      candidate.id === document?.latestInvocation?.id
    ) ?? ppoSegments.at(-1);
  const metadata = metadataAt(metadataFile);
  const ppo =
    typeof metadata?.ppo === "object" &&
    metadata.ppo !== null &&
    !Array.isArray(metadata.ppo)
      ? metadata.ppo as Record<string, unknown>
      : null;
  const metadataTotal = nonNegativeInt(ppo?.total_timesteps);
  if (!segment) {
    return {
      rewardHistory: [],
      trainingSteps: 0,
      trainingTotal: metadataTotal,
      normalizeRewards:
        document?.latestInvocation?.normalizeRewards ??
        ppo?.normalize_rewards === true,
    };
  }
  return {
    rewardHistory: segment.collections.map((sample) => ({
      step: Math.max(0, sample.step - segment.startStep),
      reward: sample.meanReward,
    })),
    trainingSteps: segment.trainedSteps,
    trainingTotal:
      document?.latestInvocation?.totalTimesteps ?? metadataTotal,
    normalizeRewards:
      segment.normalization === "normalized" ||
      (segment.normalization === "unknown" &&
        document?.latestInvocation?.normalizeRewards === true),
  };
}
