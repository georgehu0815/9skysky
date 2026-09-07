import { NextRequest, NextResponse } from "next/server";

import {
  cancelJob,
  snapshot,
  startJob,
  type RlxOperation,
  type RlxRecipe,
} from "@/lib/rlx-job";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    return NextResponse.json(
      await snapshot(
        request.nextUrl.searchParams.get("experiment") || undefined,
        request.nextUrl.searchParams.get("run") || undefined
      )
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Status failed." },
      { status: 400 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      action?: RlxOperation | "cancel";
      recipe?: Partial<RlxRecipe>;
    };
    if (body.action === "cancel") {
      return NextResponse.json({ cancelled: cancelJob() });
    }
    if (
      !body.action ||
      !["train", "eval", "render", "export"].includes(body.action)
    ) {
      return NextResponse.json({ error: "Unknown RLX action." }, { status: 400 });
    }
    const recipe = startJob(body.action as RlxOperation, body.recipe ?? {});
    return NextResponse.json(
      { accepted: true, action: body.action, recipe },
      { status: 202 }
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Job launch failed." },
      { status: 409 }
    );
  }
}
