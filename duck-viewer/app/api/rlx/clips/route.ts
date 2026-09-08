import { NextResponse } from "next/server";

import { listDanceClips } from "@/lib/dance-clips";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const clips = (await listDanceClips()).map((clip) => ({
      ...clip,
      name: clip.label,
    }));
    return NextResponse.json(
      { clips },
      { headers: { "Cache-Control": "no-store" } }
    );
  } catch {
    return NextResponse.json(
      { error: "The dance clip catalog could not be loaded." },
      { status: 500 }
    );
  }
}
