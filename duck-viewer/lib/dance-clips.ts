import { readdir, readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";

export interface DanceClipCatalogEntry {
  path: string;
  label: string;
  durationSeconds: number;
  frameCount: number;
}

interface CatalogOptions {
  workspaceRoot?: string;
}

interface ClipKey {
  t?: unknown;
  joints?: unknown;
  rootPitch?: unknown;
}

interface ClipJson {
  name?: unknown;
  duration?: unknown;
  keys?: unknown;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === "" ||
    (!relative.startsWith("..") && !path.isAbsolute(relative))
  );
}

function workspaceRootFrom(cwd: string): string {
  return path.basename(cwd) === "duck-viewer" ? path.dirname(cwd) : cwd;
}

async function candidateFiles(
  root: string,
  include: (fileName: string) => boolean
): Promise<string[]> {
  const files: string[] = [];

  async function visit(directory: string): Promise<void> {
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === "ENOENT") return;
      throw error;
    }

    await Promise.all(
      entries.map(async (entry) => {
        const candidate = path.join(directory, entry.name);
        if (entry.isDirectory()) {
          await visit(candidate);
        } else if (
          (entry.isFile() || entry.isSymbolicLink()) &&
          include(entry.name)
        ) {
          files.push(candidate);
        }
      })
    );
  }

  await visit(root);
  return files;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clipMetadata(
  raw: ClipJson,
  fallbackLabel: string
): Omit<DanceClipCatalogEntry, "path"> | null {
  if (
    !finiteNumber(raw.duration) ||
    raw.duration <= 0 ||
    raw.duration > 120
  ) {
    return null;
  }
  if (
    !Array.isArray(raw.keys) ||
    raw.keys.length === 0 ||
    raw.keys.length > 512
  ) {
    return null;
  }

  let previousTime = -1;
  for (let index = 0; index < raw.keys.length; index += 1) {
    const key = raw.keys[index] as ClipKey;
    if (
      !key ||
      typeof key !== "object" ||
      Array.isArray(key) ||
      !finiteNumber(key.t)
    ) {
      return null;
    }
    if (index === 0 && key.t !== 0) return null;
    if (key.t <= previousTime || key.t < 0 || key.t > raw.duration) return null;
    if (
      !Array.isArray(key.joints) ||
      key.joints.length !== 14 ||
      !key.joints.every(finiteNumber)
    ) {
      return null;
    }
    if (key.rootPitch != null && !finiteNumber(key.rootPitch)) return null;
    previousTime = key.t;
  }

  const name = typeof raw.name === "string" ? raw.name.trim() : "";
  return {
    label: name || fallbackLabel,
    durationSeconds: raw.duration,
    frameCount: raw.keys.length,
  };
}

async function catalogEntry(
  workspaceRoot: string,
  allowedRoot: string,
  filePath: string
): Promise<DanceClipCatalogEntry | null> {
  try {
    const [resolvedRoot, resolvedFile] = await Promise.all([
      realpath(allowedRoot),
      realpath(filePath),
    ]);
    if (!isWithin(resolvedRoot, resolvedFile)) return null;
    if (!(await stat(resolvedFile)).isFile()) return null;

    const raw = JSON.parse(await readFile(resolvedFile, "utf8")) as ClipJson;
    const metadata = clipMetadata(
      raw,
      path.basename(filePath, path.extname(filePath))
    );
    if (!metadata) return null;

    return {
      path: path.relative(workspaceRoot, filePath).split(path.sep).join("/"),
      ...metadata,
    };
  } catch {
    return null;
  }
}

export async function listDanceClips(
  options: CatalogOptions = {}
): Promise<DanceClipCatalogEntry[]> {
  const workspaceRoot = path.resolve(
    options.workspaceRoot ?? workspaceRootFrom(process.cwd())
  );
  const authoredRoot = path.join(workspaceRoot, "dance-clip");
  const artifactsRoot = path.join(workspaceRoot, "rlx", "artifacts");

  const [authoredFiles, artifactFiles] = await Promise.all([
    candidateFiles(authoredRoot, (name) => name.endsWith(".json")),
    candidateFiles(artifactsRoot, (name) => name === "reference.clip.json"),
  ]);
  const entries = await Promise.all([
    ...authoredFiles.map((file) =>
      catalogEntry(workspaceRoot, authoredRoot, file)
    ),
    ...artifactFiles.map((file) =>
      catalogEntry(workspaceRoot, artifactsRoot, file)
    ),
  ]);

  return entries
    .filter((entry): entry is DanceClipCatalogEntry => entry !== null)
    .sort((left, right) => left.path.localeCompare(right.path));
}
