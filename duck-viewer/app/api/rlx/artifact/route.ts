import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

import { readArtifact } from "@/lib/rlx-job";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const KINDS = {
  checkpoint: "application/octet-stream",
  metadata: "application/json",
  onnx: "application/octet-stream",
  sheet: "image/png",
  video: "video/mp4",
} as const;

export async function GET(request: NextRequest) {
  const run = request.nextUrl.searchParams.get("run");
  const experiment = request.nextUrl.searchParams.get("experiment");
  const kind = request.nextUrl.searchParams.get("kind") as keyof typeof KINDS | null;
  if (!experiment || !run || !kind || !(kind in KINDS)) {
    return NextResponse.json(
      { error: "A valid run and artifact kind are required." },
      { status: 400 }
    );
  }
  try {
    const artifact = await readArtifact(experiment, run, kind);
    const inline = kind === "video" && request.nextUrl.searchParams.get("inline") === "1";
    const range = kind === "video" ? request.headers.get("range") : null;
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!match) {
        return new NextResponse(null, {
          status: 416,
          headers: { "Content-Range": `bytes */${artifact.size}` },
        });
      }
      const requestedStart = match[1] ? Number(match[1]) : null;
      const requestedEnd = match[2] ? Number(match[2]) : null;
      const start =
        requestedStart ??
        Math.max(0, artifact.size - Math.max(0, requestedEnd ?? 0));
      const end = Math.min(
        requestedEnd ?? artifact.size - 1,
        artifact.size - 1
      );
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) ||
        start < 0 ||
        end < start ||
        start >= artifact.size
      ) {
        return new NextResponse(null, {
          status: 416,
          headers: { "Content-Range": `bytes */${artifact.size}` },
        });
      }
      const data = artifact.data.subarray(start, end + 1);
      return new NextResponse(data, {
        status: 206,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Type": KINDS[kind],
          "Content-Length": String(data.byteLength),
          "Content-Range": `bytes ${start}-${end}/${artifact.size}`,
          "Content-Disposition": `inline; filename="${path.basename(artifact.filePath)}"`,
          "Cache-Control": "no-store",
        },
      });
    }
    return new NextResponse(artifact.data, {
      headers: {
        "Accept-Ranges": kind === "video" ? "bytes" : "none",
        "Content-Type": KINDS[kind],
        "Content-Length": String(artifact.size),
        "Content-Disposition": `${inline ? "inline" : "attachment"}; filename="${path.basename(artifact.filePath)}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json({ error: "Artifact not found." }, { status: 404 });
  }
}
