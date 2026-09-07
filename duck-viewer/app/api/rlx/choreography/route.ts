import { NextResponse } from "next/server";

import { readDanceChoreography } from "@/lib/rlx-job";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return new NextResponse(await readDanceChoreography(), {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "The packaged dance choreography could not be loaded." },
      { status: 404 }
    );
  }
}
