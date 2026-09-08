"""Pure dance skill-gate checks plus the recipe wrapper evidence contract."""

import json

import numpy as np
import pytest

from rlx.environments.dance_evaluation import (
    DanceEvaluation,
    DanceEvaluationCriteria,
)
from rlx.environments.microduck_recipes import make_single_recipe_env


LEG_JOINTS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)


def dance_state(
    step: int,
    total_steps: int,
    *,
    tracking_fraction: float = 1.0,
    upright: float = 1.0,
    pose_offset: float = 0.0,
    moving_legs: tuple[int, ...] = (0, 1, 9),
    head_amplitude: float = 0.0,
) -> dict:
    phase = 2.0 * np.pi * step / total_steps
    target = np.zeros(14, dtype=np.float64)
    for offset, joint in enumerate(moving_legs):
        target[joint] = (0.10 + 0.02 * offset) * np.sin(phase + offset * 0.4)
    target[7] = head_amplitude * np.cos(phase)
    current = tracking_fraction * target + pose_offset
    return {
        "phase_step": step,
        "current_joints": current.tolist(),
        "target_joints": target.tolist(),
        "upright": upright,
        "height_m": 0.22,
    }


def observe_episode(
    evaluation: DanceEvaluation,
    *,
    env_index: int = 0,
    steps: int = 100,
    total_steps: int = 100,
    tracking_fraction: float = 1.0,
    moving_legs: tuple[int, ...] = (0, 1, 9),
    head_amplitude: float = 0.0,
    low_upright_steps: int = 0,
    pose_offset: float = 0.0,
    terminated: bool = False,
    truncated: bool = True,
) -> None:
    for step in range(steps):
        state = dance_state(
            step,
            total_steps,
            tracking_fraction=tracking_fraction,
            upright=0.89 if step < low_upright_steps else 0.9,
            pose_offset=pose_offset,
            moving_legs=moving_legs,
            head_amplitude=head_amplitude,
        )
        last = step == steps - 1
        evaluation.observe(
            env_index,
            state,
            terminated=last and terminated,
            truncated=last and truncated,
        )


def assert_finite_json(report: dict) -> None:
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_default_criteria_are_fixed_skill_thresholds():
    evaluation = DanceEvaluation(num_envs=2, required_steps=400)

    assert evaluation.report()["criteria"] == {
        "version": 1,
        "min_upright": 0.9,
        "min_upright_fraction": 0.95,
        "max_pose_rmse_rad": 0.15,
        "min_dynamic_gain": 0.20,
        "min_leg_dynamic_gain": 0.0,
        "moving_joint_std_rad": 0.03,
        "min_moving_leg_joints": 2,
        "all_episodes_required": True,
        "required_steps": 400,
    }


@pytest.mark.parametrize(
    "num_envs,required_steps",
    [(0, 100), (1.5, 100), (1, 0), (1, 1.5)],
)
def test_evaluation_rejects_invalid_dimensions(num_envs, required_steps):
    with pytest.raises(ValueError):
        DanceEvaluation(num_envs, required_steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_upright": float("nan")},
        {"min_upright_fraction": 1.01},
        {"max_pose_rmse_rad": 0.0},
        {"min_dynamic_gain": 1.01},
        {"min_leg_dynamic_gain": 1.0},
        {"moving_joint_std_rad": -0.1},
        {"min_moving_leg_joints": 0},
        {"all_episodes_required": False},
    ],
)
def test_criteria_reject_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        DanceEvaluationCriteria(**kwargs)


def test_full_dynamic_episode_passes_with_raw_aligned_tracking():
    evaluation = DanceEvaluation(1, 100)
    observe_episode(evaluation)

    report = evaluation.report()

    assert report["passed"] is True
    assert report["completed_episodes"] == report["passed_episodes"] == 1
    episode = report["episodes"][0]
    assert episode["complete"] is True
    assert episode["upright_fraction"] == 1.0
    assert episode["pose_rmse_rad"] == pytest.approx(0.0)
    assert episode["leg_pose_rmse_rad"] == pytest.approx(0.0)
    assert episode["moving_leg_joint_count"] == 3
    assert episode["dynamic_gain"] == pytest.approx(1.0)
    assert episode["leg_dynamic_gain"] == pytest.approx(1.0)
    assert episode["mean_joint_correlation"] == pytest.approx(1.0)
    assert episode["leg_mean_joint_correlation"] == pytest.approx(1.0)
    assert "no smoothing" in report["measurement_scope"]
    assert_finite_json(report)


def test_episode_may_exceed_requested_horizon():
    evaluation = DanceEvaluation(1, 100)
    observe_episode(evaluation, steps=101, total_steps=101)

    assert evaluation.report()["passed"] is True


def test_constant_mean_pose_cannot_pass_as_dance():
    evaluation = DanceEvaluation(1, 100)
    observe_episode(evaluation, tracking_fraction=0.0)

    episode = evaluation.report()["episodes"][0]

    assert episode["pose_rmse_rad"] < 0.15
    assert episode["dynamic_gain"] == pytest.approx(0.0)
    assert episode["leg_dynamic_gain"] == pytest.approx(0.0)
    assert episode["mean_joint_correlation"] == pytest.approx(0.0)
    assert episode["passed"] is False
    assert "dynamic tracking gain below target" in episode["failures"]
    assert "leg dynamic tracking did not improve on baseline" in episode["failures"]


def test_twenty_percent_dynamic_gain_is_required_without_lag_search():
    passing = DanceEvaluation(1, 100)
    observe_episode(passing, tracking_fraction=0.21)
    failing = DanceEvaluation(1, 100)
    observe_episode(failing, tracking_fraction=0.19)

    assert passing.report()["episodes"][0]["dynamic_gain"] == pytest.approx(0.21)
    assert passing.report()["passed"] is True
    assert failing.report()["episodes"][0]["dynamic_gain"] == pytest.approx(0.19)
    assert failing.report()["passed"] is False


def test_head_tracking_cannot_hide_zero_leg_improvement():
    evaluation = DanceEvaluation(1, 100)
    for step in range(100):
        state = dance_state(
            step,
            100,
            tracking_fraction=0.0,
            moving_legs=(0, 1),
            head_amplitude=1.0,
        )
        state["current_joints"][7] = state["target_joints"][7]
        evaluation.observe(
            0,
            state,
            terminated=False,
            truncated=step == 99,
        )

    episode = evaluation.report()["episodes"][0]

    assert episode["dynamic_gain"] > 0.20
    assert episode["leg_dynamic_gain"] == pytest.approx(0.0)
    assert episode["passed"] is False
    assert episode["failures"] == [
        "leg dynamic tracking did not improve on baseline"
    ]


def test_static_or_single_leg_reference_is_rejected():
    static = DanceEvaluation(1, 100)
    observe_episode(static, moving_legs=())
    one_leg = DanceEvaluation(1, 100)
    observe_episode(one_leg, moving_legs=(0,))

    for report in (static.report(), one_leg.report()):
        episode = report["episodes"][0]
        assert episode["moving_leg_joint_count"] < 2
        assert "insufficient moving leg joints in reference" in episode["failures"]
        assert report["passed"] is False


@pytest.mark.parametrize(
    "steps,terminated,truncated,expected_failure",
    [
        (99, False, True, "episode did not cover the requested clip horizon"),
        (100, False, False, "episode did not cover the requested clip horizon"),
        (100, True, False, "episode terminated or fell"),
        (100, True, True, "episode terminated or fell"),
    ],
)
def test_only_full_non_falling_completed_episodes_can_pass(
    steps, terminated, truncated, expected_failure
):
    evaluation = DanceEvaluation(1, 100)
    observe_episode(
        evaluation,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )

    episode = evaluation.report()["episodes"][0]

    assert episode["passed"] is False
    assert expected_failure in episode["failures"]


def test_upright_fraction_and_pose_rmse_thresholds_are_enforced():
    low_upright = DanceEvaluation(1, 100)
    observe_episode(low_upright, low_upright_steps=6)
    bad_pose = DanceEvaluation(1, 100)
    observe_episode(bad_pose, pose_offset=0.16)

    upright_episode = low_upright.report()["episodes"][0]
    pose_episode = bad_pose.report()["episodes"][0]
    assert upright_episode["upright_fraction"] == pytest.approx(0.94)
    assert "upright fraction below target" in upright_episode["failures"]
    assert pose_episode["pose_rmse_rad"] == pytest.approx(0.16)
    assert "pose RMSE above target" in pose_episode["failures"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.pop("target_joints"),
        lambda state: state.update(current_joints=[0.0] * 13),
        lambda state: state["target_joints"].__setitem__(0, float("nan")),
        lambda state: state.update(height_m=float("inf")),
        lambda state: state.update(phase_step=0.5),
    ],
)
def test_missing_malformed_or_nonfinite_state_fails(mutation):
    evaluation = DanceEvaluation(1, 1)
    state = dance_state(0, 100)
    mutation(state)
    evaluation.observe(0, state, terminated=False, truncated=True)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["episodes"][0]["measured_steps"] == 0
    assert "missing or non-finite dance state" in report["failures"]
    assert_finite_json(report)


def test_repeated_or_skipped_phase_steps_are_rejected():
    evaluation = DanceEvaluation(1, 100)
    for step in range(100):
        state = dance_state(step, 100)
        if step == 50:
            state["phase_step"] = 49
        evaluation.observe(
            0,
            state,
            terminated=False,
            truncated=step == 99,
        )

    episode = evaluation.report()["episodes"][0]

    assert episode["phase_step"]["contiguous"] is False
    assert "non-contiguous dance phase evidence" in episode["failures"]


def test_vector_lanes_and_episodes_are_not_pooled():
    evaluation = DanceEvaluation(2, 100)
    observe_episode(evaluation, env_index=0)
    observe_episode(evaluation, env_index=1, tracking_fraction=0.0)

    report = evaluation.report()

    assert report["completed_episodes"] == 2
    assert report["passed_episodes"] == 1
    assert [episode["passed"] for episode in report["episodes"]] == [True, False]
    assert report["passed"] is False


def test_missing_lane_and_empty_evidence_are_rejected():
    assert DanceEvaluation(1, 100).report()["failures"] == [
        "missing evaluation lane coverage"
    ]

    evaluation = DanceEvaluation(2, 100)
    observe_episode(evaluation, env_index=0)
    report = evaluation.report()
    assert report["passed"] is False
    assert "missing evaluation lane coverage" in report["failures"]


def test_dance_recipe_wrapper_emits_aligned_json_safe_state():
    env = make_single_recipe_env(
        "dance",
        seed=7,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
    )
    try:
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
        state = info["dance_state"]
        target, _ = env.unwrapped.clip.at(state["phase_step"])
        assert len(state["current_joints"]) == len(state["target_joints"]) == 14
        np.testing.assert_allclose(state["target_joints"], target)
        assert state["upright"] == info["recipe_metrics"]["upright"]
        assert state["height_m"] == info["recipe_metrics"]["height_m"]
        json.dumps(state, allow_nan=False)
    finally:
        env.close()
