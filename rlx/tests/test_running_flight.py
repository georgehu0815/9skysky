"""Opt-in RLX aerial shaping, without changing the upstream running recipe."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from microduck_local.behaviors import BehaviorEnv
from rlx.environments.microduck_recipes import (
    _running_flight,
    make_single_recipe_env,
    validate_reward_weights,
)


ENV_OPTIONS = dict(
    seed=7, actuator_force="xml", domain_rand=False, obs_noise=False,
    action_delay=False, random_yaw=False, max_episode_s=0.06,
)


def make_running(weights=None):
    options = {**ENV_OPTIONS, "actuator": ENV_OPTIONS["actuator_force"]}
    del options["actuator_force"]
    return make_single_recipe_env("running", weight_overrides=weights, **options)


def flight(*, contacts=(False, False), upright=1.0,
           velocity=(0.4, 0.0, 0.0), command=(0.6, 0.0, 0.0)):
    env = SimpleNamespace(
        _foot_contacts=lambda: {"left": contacts[0], "right": contacts[1]},
        _projected_gravity=lambda: np.array([0.0, 0.0, -upright]),
        body_lin_vel=lambda: np.array(velocity),
        twist_cmd=np.array(command),
    )
    return _running_flight(env)


@pytest.mark.parametrize("kwargs", [
    {"contacts": (True, True)},
    {"contacts": (True, False)},
    {"contacts": (False, True)},
    {"command": (0.0, 0.0, 0.0)},
    {"command": (0.0, 0.0, 1.0)},
    {"velocity": (0.0, 0.0, 0.0)},
    {"velocity": (-0.4, 0.0, 0.0)},
    {"velocity": (0.0, 0.4, 0.0)},
    {"velocity": (0.0, 0.0, 1.0)},
    {"upright": 0.8},
    {"upright": 0.0},
    {"upright": -1.0},
])
def test_no_flight_credit_for_grounded_standing_falling_or_opposing_motion(kwargs):
    assert flight(**kwargs) == 0.0


@pytest.mark.parametrize("upright,speed,expected", [
    (1.0, 0.4, 1.0), (1.0, 4.0, 1.0), (1.1, 0.8, 1.0),
    (0.9, 0.4, 0.5), (1.0, 0.2, 0.5), (0.9, 0.2, 0.25),
    (0.85, 0.1, 0.0625),
])
def test_moving_flight_is_bounded_and_uses_both_gates(upright, speed, expected):
    value = flight(upright=upright, velocity=(speed, 0.0, 0.0))
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(expected)
    json.dumps({"flight": value}, allow_nan=False)


@pytest.mark.parametrize("command,velocity", [
    ((-0.6, 0.0, 0.0), (-0.4, 0.0, 0.0)),
    ((0.0, 0.6, 0.0), (0.0, 0.4, 0.0)),
    ((0.3, 0.4, 0.0), (0.24, 0.32, 0.0)),
    ((0.6, 0.0, 2.0), (0.4, 0.0, 0.0)),
])
def test_command_projection_respects_backward_sideways_and_turning(command, velocity):
    assert flight(command=command, velocity=velocity) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_state_has_no_credit(bad):
    assert flight(upright=bad) == 0.0
    assert flight(velocity=(bad, 0.0, 0.0)) == 0.0
    assert flight(command=(bad, 0.0, 0.0)) == 0.0


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_flight_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_running({"flight": weight})


@pytest.mark.parametrize("recipe", ["dance", "swing", "stilts"])
def test_flight_is_running_only(recipe):
    with pytest.raises(ValueError, match="unknown"):
        validate_reward_weights(recipe, {"flight": 1.0})


@pytest.mark.parametrize("weights", [None, {}, {"flight": 0.0},
                                      {"flight": 0.0, "track_turn": 8.0}])
def test_disabled_flight_matches_upstream_rewards_observations_and_term_rows(weights):
    original = {key: value for key, value in (weights or {}).items() if key != "flight"}
    baseline = BehaviorEnv(behavior_id="run", weight_overrides=original, **ENV_OPTIONS)
    candidate = make_running(weights)
    try:
        assert candidate.unwrapped._term_rows == baseline._term_rows
        assert all(row[0] != "flight" for row in candidate.unwrapped._term_rows)
        np.testing.assert_array_equal(candidate.reset(seed=7)[0], baseline.reset(seed=7)[0])
        for _ in range(3):
            action = np.zeros(14, dtype=np.float32)
            actual = candidate.step(action)
            expected = baseline.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]
            assert candidate.unwrapped.reward_sums == baseline.reward_sums
        assert actual[4]["episode_rewards"] == expected[4]["episode_rewards"]
    finally:
        candidate.close()
        baseline.close()


def test_positive_flight_changes_physical_reward_and_captures_episode_terms(monkeypatch):
    baseline = make_running({"track_turn": 8.0, "keep_pace": 0.0})
    candidate = make_running({"flight": 4.0, "track_turn": 8.0, "keep_pace": 0.0})
    try:
        captured_terms = {}
        compute_reward = candidate.unwrapped._compute_reward

        def capture_reward():
            reward, terms = compute_reward()
            captured_terms.clear()
            captured_terms.update(terms)
            return reward, terms

        monkeypatch.setattr(candidate.unwrapped, "_compute_reward", capture_reward)
        assert candidate.unwrapped._term_rows[:-1] == baseline.unwrapped._term_rows
        assert candidate.unwrapped._term_rows[-1] == ("flight", "flight", 0.0, _running_flight)
        for wrapper in (baseline, candidate):
            wrapper.reset(seed=7)
            env = wrapper.unwrapped
            env.twist_cmd[:] = (0.6, 0.0, 0.0)
            env.data.qpos[2] += 0.5
            env.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            env.data.qvel[:] = 0.0
            env.data.qvel[0] = 0.6
            mujoco.mj_forward(env.model, env.data)

        flight_sum = 0.0
        for _ in range(3):
            action = np.zeros(14, dtype=np.float32)
            expected = baseline.step(action)
            actual = candidate.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(candidate.unwrapped.data.qpos, baseline.unwrapped.data.qpos)
            assert actual[2:4] == expected[2:4]
            raw_flight = _running_flight(candidate.unwrapped)
            assert 0.0 < raw_flight <= 1.0
            assert actual[1] - expected[1] == pytest.approx(4.0 * raw_flight)
            assert captured_terms["flight"] == pytest.approx(4.0 * raw_flight)
            assert captured_terms["keep_pace"] == 0.0
            flight_sum += captured_terms["flight"]
            json.dumps(captured_terms, allow_nan=False)

        assert actual[3] is True
        assert actual[4]["episode_rewards"]["flight"] == pytest.approx(flight_sum)
        assert "flight" not in expected[4]["episode_rewards"]
        for key, value in expected[4]["episode_rewards"].items():
            assert actual[4]["episode_rewards"][key] == pytest.approx(value)
        json.dumps(actual[4]["episode_rewards"], allow_nan=False)
    finally:
        candidate.close()
        baseline.close()
