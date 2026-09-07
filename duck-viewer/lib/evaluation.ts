import type { ExperimentId } from "./experiments";

/** Project evaluator verdicts without deriving skill acceptance from motion metrics. */
export function evaluationVerdict(
  report: Record<string, unknown> | null,
  experimentId: ExperimentId
) {
  const scope = report?.evaluation_mode;
  const skillAssessed = scope === "skill" &&
    (experimentId !== "swing" || report?.skill_status === "passed" || report?.skill_status === "failed");
  const settings = report?.evaluation as Record<string, unknown> | undefined;
  const criteria = settings?.swing_criteria as Record<string, unknown> | undefined;
  const target = criteria?.min_bidirectional_span_deg;
  return {
    swingMinSpanDeg: typeof target === "number" && Number.isFinite(target) ? target : null,
    scope: scope === "skill" || scope === "pipeline" ? scope : null,
    pipelinePassed: report?.pipeline_passed === true,
    skillAssessed,
    taskPassed: skillAssessed && report?.passed === true &&
      (experimentId !== "swing" || report?.skill_status === "passed"),
  };
}
