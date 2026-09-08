import type { ExperimentId } from "./experiments";

export interface SavedRun {
  experimentId: ExperimentId;
  runName: string;
  taskPassed: boolean;
  skillAssessed: boolean;
  video: boolean;
  checkpoint?: boolean;
  modifiedAt?: string;
  trainedAt?: string | null;
  renderVerified?: boolean;
  renderEvidenceId?: string | null;
}

export function latestVerifiedRun(runs: readonly SavedRun[], experimentId: ExperimentId): SavedRun | undefined {
  const trainedTime = (run: SavedRun) => Date.parse(run.trainedAt ?? run.modifiedAt ?? "");
  return runs.filter((run) => run.experimentId === experimentId && run.taskPassed &&
    run.checkpoint && run.video && run.renderVerified && run.renderEvidenceId && Number.isFinite(trainedTime(run)))
    .sort((left, right) => trainedTime(right) - trainedTime(left) || left.runName.localeCompare(right.runName))[0];
}

export function availableProfileRunName(
  runName: string,
  profile: "smoke" | "full",
  reservedNames: readonly string[],
): string {
  const canonical = (name: string) => name.trim().toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
  const reserved = new Set(reservedNames.map(canonical));
  const current = canonical(runName);
  if (!reserved.has(current)) return runName;
  const base = current.replace(/-(smoke|full)(-\d+)?$/, "") || "run";
  for (let sequence = 1; ; sequence += 1) {
    const suffix = `-${profile}${sequence === 1 ? "" : `-${sequence}`}`;
    const candidate = `${base.slice(0, 48 - suffix.length)}${suffix}`;
    if (!reserved.has(candidate)) return candidate;
  }
}
