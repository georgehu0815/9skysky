"""Pure per-episode checks for running and stilt skill acceptance."""

import json
import math

import pytest

from rlx.environments.locomotion_evaluation import LocomotionEvaluation


def samples(
    recipe,
    *,
    steps=None,
    command=(-0.5, 0.0),
    velocity=(-0.4, 0.0),
    contacts=None,
):
    required = 600 if recipe == "running" else 500
    steps = required if steps is None else steps
    contacts = contacts or (
        (True, False),
        (False, False),
        (False, True),
        (False, False),
    )
    position = [0.0, 0.0]
    heading = (0.0, 1.0)
    side = (-1.0, 0.0)
    command_speed = (command[0] ** 2 + command[1] ** 2) ** 0.5
    local_direction = (
        command[0] / command_speed,
        command[1] / command_speed,
    )
    world_direction = (
        heading[0] * local_direction[0] + side[0] * local_direction[1],
        heading[1] * local_direction[0] + side[1] * local_direction[1],
    )
    directed_speed = (
        velocity[0] * local_direction[0]
        + velocity[1] * local_direction[1]
    )
    result = []
    for step in range(steps):
        position[0] += world_direction[0] * directed_speed * 0.02
        position[1] += world_direction[1] * directed_speed * 0.02
        left, right = contacts[step % len(contacts)]
        result.append(
            {
                "command_forward_m_s": command[0],
                "command_lateral_m_s": command[1],
                "command_yaw_rad_s": 0.0,
                "heading_forward_speed_m_s": velocity[0],
                "heading_lateral_speed_m_s": velocity[1],
                "world_x_m": position[0],
                "world_y_m": position[1],
                "heading_forward_x": heading[0],
                "heading_forward_y": heading[1],
                "upright": 1.0,
                "left_foot_contact": float(left),
                "right_foot_contact": float(right),
            }
        )
    return result


def observe_episode(evaluation, frames, *, terminated=False, truncated=True):
    for step, metrics in enumerate(frames):
        last = step == len(frames) - 1
        evaluation.observe(
            0,
            metrics,
            last and terminated,
            last and truncated,
        )


def test_running_accepts_backward_command_projected_motion_with_real_flight():
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(evaluation, samples("running"))

    report = evaluation.report()

    assert report["passed"] is True
    episode = report["episodes"][0]
    assert episode["complete"] is True
    assert episode["mean_command_directed_speed_m_s"] == pytest.approx(0.4)
    assert episode["command_directed_displacement_m"] > 4.7
    assert episode["aerial_fraction"] == pytest.approx(0.5)
    assert episode["alternating_support_switches"] > 8
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_stilts_accepts_lateral_command_in_heading_frame():
    evaluation = LocomotionEvaluation("stilts", 1, 500)
    observe_episode(
        evaluation,
        samples(
            "stilts",
            command=(0.0, 0.2),
            velocity=(0.0, 0.08),
        ),
    )

    report = evaluation.report()

    assert report["passed"] is True
    episode = report["episodes"][0]
    assert episode["mean_command_directed_speed_m_s"] == pytest.approx(0.08)
    assert episode["command_directed_displacement_m"] > 0.79
    assert report["criteria"]["required_steps"] == 500


def test_walking_contact_pattern_is_not_credited_as_running():
    walking_contacts = (
        (True, False),
        (True, True),
        (False, True),
        (True, True),
    )
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(
        evaluation,
        samples("running", contacts=walking_contacts),
    )

    report = evaluation.report()

    assert report["passed"] is False
    assert report["episodes"][0]["aerial_fraction"] == 0.0
    assert "aerial fraction below running target" in report["failures"]


@pytest.mark.parametrize(
    "mutate, failure",
    [
        (lambda metric: metric.pop("world_x_m"), "missing or non-finite locomotion metrics"),
        (
            lambda metric: metric.__setitem__("upright", float("nan")),
            "missing or non-finite locomotion metrics",
        ),
    ],
)
def test_missing_or_nonfinite_sample_fails_closed(mutate, failure):
    frames = samples("stilts")
    mutate(frames[17])
    evaluation = LocomotionEvaluation("stilts", 1, 500)
    observe_episode(evaluation, frames)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["episodes"][0]["measured_steps"] == 499
    assert failure in report["failures"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "steps, terminated, truncated",
    [
        (599, False, True),
        (600, False, False),
        (600, True, True),
    ],
)
def test_running_requires_complete_nonterminated_twelve_second_episode(
    steps, terminated, truncated
):
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(
        evaluation,
        samples("running", steps=steps),
        terminated=terminated,
        truncated=truncated,
    )

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 0
    assert "episode did not complete the required 12-second horizon" in report[
        "failures"
    ]


def test_completed_and_partial_episodes_are_not_pooled():
    evaluation = LocomotionEvaluation("stilts", 1, 500)
    observe_episode(evaluation, samples("stilts"))
    observe_episode(
        evaluation,
        samples("stilts", steps=100),
        truncated=False,
    )

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 1
    assert report["incomplete_episodes"] == 1
    assert report["passed_episodes"] == 1
    assert [episode["episode_index"] for episode in report["episodes"]] == [0, 1]


def test_requires_meaningful_commanded_motion_coverage():
    frames = samples("running")
    for metric in frames[100:]:
        metric["command_forward_m_s"] = 0.0
        metric["command_lateral_m_s"] = 0.0
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(evaluation, frames)

    report = evaluation.report()

    assert report["passed"] is False
    assert report["episodes"][0]["commanded_fraction"] == pytest.approx(1 / 6)
    assert "insufficient commanded-motion coverage" in report["failures"]


@pytest.mark.parametrize("recipe,steps", [("running", 600), ("stilts", 500)])
def test_public_signature_and_fixed_criteria(recipe, steps):
    evaluation = LocomotionEvaluation(recipe, 1, steps)
    criteria = evaluation.criteria.to_dict()
    assert criteria["required_steps"] == steps
    assert criteria["min_upright"] == 0.9
    assert criteria["min_upright_fraction"] == 0.95
    assert criteria["all_episodes_required"] is True


def test_rejects_unsupported_recipe():
    with pytest.raises(ValueError, match="running.*stilts"):
        LocomotionEvaluation("dance", 1, 500)


@pytest.mark.parametrize(
    "recipe,steps,duration",
    [("running", 599, "12 seconds"), ("stilts", 499, "10 seconds")],
)
def test_cannot_relax_minimum_episode_horizon(recipe, steps, duration):
    with pytest.raises(ValueError, match=duration):
        LocomotionEvaluation(recipe, 1, steps)


@pytest.mark.parametrize("recipe,steps", [("running", 600), ("stilts", 500)])
@pytest.mark.parametrize("turn_sign", [-1, 1])
@pytest.mark.parametrize("commanded_turn", [False, True])
@pytest.mark.parametrize("command", [(0.6, 0.0), (-0.6, 0.0), (0.0, 0.6)])
def test_circle_only_counts_when_turn_is_commanded(
    recipe, steps, turn_sign, commanded_turn, command
):
    velocity = tuple(component * 2 / 3 for component in command)
    frames = samples(recipe, command=command, velocity=velocity)
    yaw_rate = turn_sign * 2 * math.pi / (steps * 0.02)
    radius = 0.4 / yaw_rate
    direction_angle = math.atan2(command[1], command[0])
    for index, metric in enumerate(frames):
        angle = yaw_rate * (index + 1) * 0.02
        metric.update(
            command_yaw_rad_s=yaw_rate if commanded_turn else 0.0,
            heading_forward_x=math.cos(angle),
            heading_forward_y=math.sin(angle),
            world_x_m=radius * (math.sin(angle + direction_angle) - math.sin(direction_angle)),
            world_y_m=radius * (math.cos(direction_angle) - math.cos(angle + direction_angle)),
        )
    evaluation = LocomotionEvaluation(recipe, 1, steps)
    observe_episode(evaluation, frames)
    report = evaluation.report()
    episode = report["episodes"][0]
    assert report["passed"] is commanded_turn
    if commanded_turn:
        assert episode["command_directed_displacement_m"] == pytest.approx(
            (steps - 1) * 0.02 * 0.4, abs=0.001
        )
    else:
        assert abs(episode["command_directed_displacement_m"]) < 0.01
        assert report["failures"] == ["command-directed displacement below target"]


def test_command_change_uses_current_transition_direction_and_reanchors():
    evaluation = LocomotionEvaluation("running", 1, 600)
    frames = samples("running", steps=3, command=(0.6, 0.0), velocity=(0.4, 0.0))
    frames[0].update(world_x_m=0.0, world_y_m=0.0)
    frames[1].update(
        command_forward_m_s=-0.6, heading_forward_speed_m_s=-0.4,
        world_x_m=0.0, world_y_m=-0.008,
        heading_forward_x=1.0, heading_forward_y=0.0,
    )
    frames[2].update(
        command_forward_m_s=0.0, command_lateral_m_s=0.6,
        heading_forward_speed_m_s=0.0, heading_lateral_speed_m_s=0.4,
        world_x_m=0.0, world_y_m=0.0,
        heading_forward_x=1.0, heading_forward_y=0.0,
    )
    observe_episode(evaluation, frames, truncated=False)
    episode = evaluation.report()["episodes"][0]
    assert episode["command_directed_displacement_m"] == pytest.approx(0.016)


@pytest.mark.parametrize("yaw", [None, float("nan"), float("inf")])
def test_yaw_command_is_required_finite_evidence(yaw):
    frames = samples("running")
    if yaw is None:
        frames[17].pop("command_yaw_rad_s")
    else:
        frames[17]["command_yaw_rad_s"] = yaw
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(evaluation, frames)
    report = evaluation.report()
    assert report["passed"] is False
    assert "missing or non-finite locomotion metrics" in report["failures"]
    json.dumps(report, allow_nan=False)


def test_zero_yaw_out_and_back_displacement_is_signed_not_clipped():
    frames = samples("running", command=(0.6, 0.0), velocity=(0.4, 0.0))
    for index, metric in enumerate(frames):
        metric["world_y_m"] = min(index, len(frames) - 1 - index) * 0.008
    evaluation = LocomotionEvaluation("running", 1, 600)
    observe_episode(evaluation, frames)
    report = evaluation.report()
    assert report["passed"] is False
    assert report["episodes"][0]["command_directed_displacement_m"] == pytest.approx(0.0)


def test_segment_anchor_does_not_leak_across_episodes_or_lanes():
    evaluation = LocomotionEvaluation("running", 2, 600)
    for _ in range(2):
        for index, metric in enumerate(samples("running")):
            for lane in range(2):
                direction = 1.0 if lane else -1.0
                rotated = {
                    **metric,
                    "heading_forward_y": metric["heading_forward_y"] * direction,
                    "world_y_m": metric["world_y_m"] * direction,
                }
                evaluation.observe(lane, rotated, False, index == 599)
    report = evaluation.report()
    assert report["passed"] is True
    assert report["passed_episodes"] == 4
    assert all(episode["command_directed_displacement_m"] > 4.7 for episode in report["episodes"])
