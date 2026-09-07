"""Pure acceptance checks for control-rate Swing samples; no physics or training."""

import json

import pytest

from rlx.environments.swing_evaluation import (
    REQUIRED_METRICS,
    SwingEvaluation,
    SwingEvaluationPlan,
)


def physical_metrics(angle=75.0):
    return {
        "swing_angle_deg": angle,
        "lateral_offset_m": 0.0,
        "alignment_penalty": 0.0,
        "valid_geometry": 1.0,
        "string_left_m": 0.3822,
        "string_right_m": 0.3822,
        "string_left_tension_n": 4.4,
        "string_right_tension_n": 4.4,
    }


def observe_episode(
    evaluation,
    *,
    env_index=0,
    steps=1200,
    angles=(75.0, -75.0),
    terminated=False,
    truncated=True,
    bad_sample=None,
):
    for step in range(steps):
        metrics = physical_metrics(angles[step % len(angles)])
        if step == 17 and bad_sample is not None:
            metrics = bad_sample
        last = step == steps - 1
        evaluation.observe(
            env_index,
            metrics,
            terminated=last and terminated,
            truncated=last and truncated,
        )


def assert_finite_json(report):
    encoded = json.dumps(report, allow_nan=False)
    assert json.loads(encoded) == report


def test_default_plan_is_still_start_24_second_symmetric_criterion():
    assert SwingEvaluationPlan().to_dict() == {
        "version": 1,
        "min_bidirectional_span_deg": 150.0,
        "required_steps": 1200,
        "max_abs_lateral_m": 0.020,
        "max_alignment": 0.050,
        "min_string_length_m": 0.370,
        "max_string_length_m": 0.394,
        "min_string_tension_n": 0.0,
        "all_episodes_required": True,
    }


@pytest.mark.parametrize("threshold", [0.0, 181.0, float("nan")])
def test_plan_rejects_invalid_span_threshold(threshold):
    with pytest.raises(ValueError, match="finite and in"):
        SwingEvaluationPlan(min_bidirectional_span_deg=threshold)


def test_complete_bidirectional_episode_passes_at_exact_150_degree_boundary():
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    observe_episode(evaluation)

    report = evaluation.report()

    assert report["passed"] is True
    assert report["passed_episodes"] == report["completed_episodes"] == 1
    assert report["incomplete_episodes"] == 0
    assert report["failures"] == []
    episode = report["episodes"][0]
    assert episode["steps"] == episode["measured_steps"] == 1200
    assert episode["positive_peak_deg"] == 75.0
    assert episode["negative_peak_deg"] == -75.0
    assert episode["peak_to_peak_span_deg"] == episode["bidirectional_span_deg"] == 150.0
    assert episode["valid_geometry_fraction"] == 1.0
    assert episode["both_strings_tensioned_fraction"] == 1.0
    assert episode["complete"] is True
    assert report["hardware_safe"] is False
    assert "constraint forces" in report["measurement_scope"]
    assert_finite_json(report)


@pytest.mark.parametrize(
    "angles, expected_span, expected_bidirectional",
    [
        ((74.999, -75.0), 149.999, 149.998),
        ((170.0, 0.0), 170.0, 0.0),
        ((170.0, -10.0), 180.0, 20.0),
    ],
)
def test_span_requires_both_sides_to_reach_half_target(angles, expected_span, expected_bidirectional):
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    observe_episode(evaluation, angles=angles)

    report = evaluation.report()

    assert report["passed"] is False
    episode = report["episodes"][0]
    assert episode["peak_to_peak_span_deg"] == pytest.approx(expected_span)
    assert episode["bidirectional_span_deg"] == pytest.approx(expected_bidirectional)
    assert episode["failures"] == ["bidirectional span below target"]


def test_opposite_peaks_in_separate_reset_episodes_are_not_pooled():
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    observe_episode(evaluation, angles=(80.0, 0.0))
    observe_episode(evaluation, angles=(-80.0, 0.0))

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 2
    assert report["passed_episodes"] == 0
    assert [episode["episode_index"] for episode in report["episodes"]] == [0, 1]
    assert [episode["bidirectional_span_deg"] for episode in report["episodes"]] == [0.0, 0.0]


def test_vector_lanes_track_episodes_and_peaks_independently():
    evaluation = SwingEvaluation(3, SwingEvaluationPlan())
    for step in range(1200):
        for lane, angles in enumerate(((75.0, -75.0), (90.0, 0.0), (-90.0, 0.0))):
            evaluation.observe(
                lane,
                physical_metrics(angles[step % 2]),
                terminated=False,
                truncated=step == 1199,
            )

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 3
    assert report["passed_episodes"] == 1
    assert [episode["env_index"] for episode in report["episodes"]] == [0, 1, 2]
    assert [episode["episode_index"] for episode in report["episodes"]] == [0, 0, 0]
    assert [episode["passed"] for episode in report["episodes"]] == [True, False, False]
    assert [episode["bidirectional_span_deg"] for episode in report["episodes"]] == [150.0, 0.0, 0.0]


def test_full_episode_followed_by_partial_episode_fails_overall():
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    observe_episode(evaluation)
    observe_episode(evaluation, steps=2, truncated=False)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == report["incomplete_episodes"] == 1
    assert report["passed_episodes"] == 1
    assert [episode["episode_index"] for episode in report["episodes"]] == [0, 1]
    assert report["episodes"][1]["bidirectional_span_deg"] == 150.0
    assert report["episodes"][1]["failures"] == ["incomplete 24-second episode"]
    assert evaluation.report() == report


@pytest.mark.parametrize(
    "steps, terminated, truncated",
    [(1199, False, False), (1199, False, True), (1200, False, False),
     (1200, True, False), (1200, True, True)],
)
def test_finite_samples_require_full_horizon_truncation_without_termination(steps, terminated, truncated):
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    observe_episode(evaluation, steps=steps, terminated=terminated, truncated=truncated)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 0
    episode = report["episodes"][0]
    assert episode["measured_steps"] == steps
    assert episode["bidirectional_span_deg"] == 150.0
    assert episode["complete"] is False
    assert "incomplete 24-second episode" in episode["failures"]
    assert ("episode terminated" in episode["failures"]) is terminated
    assert_finite_json(report)


@pytest.mark.parametrize(
    "key, value",
    [
        ("lateral_offset_m", 0.020001),
        ("lateral_offset_m", -0.020001),
        ("alignment_penalty", 0.050001),
        ("alignment_penalty", -0.001),
        ("string_left_m", 0.369999),
        ("string_left_m", 0.394001),
        ("string_right_m", 0.369999),
        ("string_right_m", 0.394001),
        ("valid_geometry", 0.0),
    ],
)
def test_geometry_violation_fails_even_after_later_samples_recover(key, value):
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    bad_sample = {**physical_metrics(), key: value}
    observe_episode(evaluation, bad_sample=bad_sample)

    report = evaluation.report()

    assert report["passed"] is False
    episode = report["episodes"][0]
    assert episode["complete"] is True
    assert episode["measured_steps"] == 1200
    assert episode["valid_geometry_fraction"] == pytest.approx(1199 / 1200)
    assert episode["failures"] == ["invalid swing geometry"]
    assert_finite_json(report)


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_string_must_keep_positive_spring_tension_even_after_recovery(side):
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    bad_sample = {**physical_metrics(), f"string_{side}_tension_n": 0.0}
    observe_episode(evaluation, bad_sample=bad_sample)

    report = evaluation.report()

    assert report["passed"] is False
    episode = report["episodes"][0]
    assert episode["min_spring_tension_n"] == 0.0
    assert episode["valid_geometry_fraction"] == 1.0
    assert episode["both_strings_tensioned_fraction"] == pytest.approx(1199 / 1200)
    assert episode["failures"] == ["one or both strings lost positive spring tension"]


@pytest.mark.parametrize("key", REQUIRED_METRICS)
@pytest.mark.parametrize("invalid", ["missing", float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_physical_metric_fails_with_finite_json(key, invalid):
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    bad_sample = physical_metrics()
    if invalid == "missing":
        del bad_sample[key]
    else:
        bad_sample[key] = invalid
    observe_episode(evaluation, bad_sample=bad_sample)

    report = evaluation.report()

    assert report["passed"] is False
    episode = report["episodes"][0]
    assert episode["steps"] == 1200
    assert episode["measured_steps"] == 1199
    assert "missing or non-finite physical metrics" in episode["failures"]
    assert "incomplete physical measurement coverage" in episode["failures"]
    assert_finite_json(report)


def test_all_missing_physical_samples_report_null_extrema_not_nan():
    evaluation = SwingEvaluation(1, SwingEvaluationPlan())
    evaluation.observe(0, {}, terminated=False, truncated=True)

    report = evaluation.report()

    assert report["passed"] is False
    episode = report["episodes"][0]
    assert episode["measured_steps"] == 0
    assert episode["min_string_length_m"] is None
    assert episode["max_string_length_m"] is None
    assert episode["min_spring_tension_n"] is None
    assert_finite_json(report)


def test_no_observations_cannot_pass():
    report = SwingEvaluation(1, SwingEvaluationPlan()).report()

    assert report["passed"] is False
    assert report["episodes"] == []
    assert report["completed_episodes"] == report["passed_episodes"] == 0
    assert report["failures"] == ["missing evaluation lane coverage"]
    assert_finite_json(report)


def test_unobserved_lane_prevents_passing_completed_other_lane():
    evaluation = SwingEvaluation(2, SwingEvaluationPlan())
    observe_episode(evaluation, env_index=1)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["passed_episodes"] == 1
    assert report["episodes"][0]["env_index"] == 1
    assert report["failures"] == ["missing evaluation lane coverage"]
    assert_finite_json(report)
