"""Opt-in yaw-only tracking for the running and stilt recipes."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_local.behaviors import BehaviorEnv
from rlx.environments.microduck_recipes import (
    _locomotion_yaw_tracking,
    make_single_recipe_env,
    validate_reward_weights,
)


ENV_OPTIONS = dict(
    seed=11,
    actuator_force="xml",
    domain_rand=False,
    obs_noise=False,
    action_delay=False,
    random_yaw=False,
    max_episode_s=0.06,
)


def yaw_score(*, gyro=(0.0, 0.0, 0.0), command=(0.6, 0.0, 0.0)):
    env = SimpleNamespace(
        _gyro=np.asarray(gyro, dtype=float),
        twist_cmd=np.asarray(command, dtype=float),
    )
    return _locomotion_yaw_tracking(env)


def make_recipe(recipe, weights=None):
    options = {**ENV_OPTIONS, "actuator": ENV_OPTIONS["actuator_force"]}
    del options["actuator_force"]
    return make_single_recipe_env(recipe, weight_overrides=weights, **options)


def test_yaw_tracking_rewards_target_and_distinguishes_opposite_yaw():
    assert yaw_score(
        gyro=(0.0, 0.0, 0.25),
        command=(0.6, 0.0, 0.25),
    ) == pytest.approx(1.0)
    assert yaw_score(
        gyro=(0.0, 0.0, -0.25),
        command=(0.6, 0.0, 0.25),
    ) == pytest.approx(math.exp(-4.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_yaw_tracking_returns_zero_for_nonfinite_yaw_or_command(bad):
    assert yaw_score(gyro=(0.0, 0.0, bad)) == 0.0
    assert yaw_score(command=(0.6, 0.0, bad)) == 0.0


def test_yaw_tracking_is_independent_of_roll_and_pitch_rates():
    expected = yaw_score(
        gyro=(0.0, 0.0, 0.15),
        command=(0.6, 0.0, 0.25),
    )
    assert yaw_score(
        gyro=(12.0, -9.0, 0.15),
        command=(0.6, 0.0, 0.25),
    ) == pytest.approx(expected)


@pytest.mark.parametrize("recipe", ["running", "stilts"])
def test_yaw_tracking_weight_is_allowed_for_locomotion_recipes(recipe):
    assert validate_reward_weights(recipe, {"yaw_tracking": 2}) == {
        "yaw_tracking": 2.0
    }


@pytest.mark.parametrize("recipe", ["running", "stilts"])
@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_yaw_tracking_weight_must_be_finite_and_nonnegative(recipe, weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_reward_weights(recipe, {"yaw_tracking": weight})


@pytest.mark.parametrize("recipe", ["dance", "swing"])
def test_yaw_tracking_weight_is_rejected_for_other_recipes(recipe):
    with pytest.raises(ValueError, match=f"unknown {recipe} reward"):
        validate_reward_weights(recipe, {"yaw_tracking": 1.0})


def test_running_default_and_zero_override_leave_upstream_reward_unchanged():
    for weights in (None, {}, {"yaw_tracking": 0.0}):
        baseline = BehaviorEnv(behavior_id="run", **ENV_OPTIONS)
        candidate = make_recipe("running", weights)
        try:
            assert candidate.unwrapped._term_rows == baseline._term_rows
            candidate.reset(seed=11)
            baseline.reset(seed=11)
            candidate_reward, candidate_terms = candidate.unwrapped._compute_reward()
            baseline_reward, baseline_terms = baseline._compute_reward()
            assert candidate_reward == pytest.approx(baseline_reward)
            assert candidate_terms == pytest.approx(baseline_terms)
            assert "yaw_tracking" not in candidate_terms
        finally:
            candidate.close()
            baseline.close()


def test_stilts_default_and_zero_override_leave_reward_unchanged():
    baseline = make_recipe("stilts")
    candidate = make_recipe("stilts", {"yaw_tracking": 0.0})
    try:
        baseline.reset(seed=11)
        candidate.reset(seed=11)
        baseline_reward, baseline_terms = baseline.unwrapped._compute_reward()
        candidate_reward, candidate_terms = candidate.unwrapped._compute_reward()
        assert candidate_reward == pytest.approx(baseline_reward)
        assert candidate_terms == pytest.approx(baseline_terms)
        assert "yaw_tracking" not in candidate_terms
    finally:
        baseline.close()
        candidate.close()


@pytest.mark.parametrize("recipe", ["running", "stilts"])
def test_per_instance_yaw_tracking_override_affects_reward(recipe):
    weight = 3.0
    baseline = make_recipe(recipe)
    candidate = make_recipe(recipe, {"yaw_tracking": weight})
    try:
        baseline.reset(seed=11)
        candidate.reset(seed=11)
        for env in (baseline.unwrapped, candidate.unwrapped):
            env.twist_cmd[2] = 0.2
            env._gyro[2] = 0.2

        baseline_reward, baseline_terms = baseline.unwrapped._compute_reward()
        candidate_reward, candidate_terms = candidate.unwrapped._compute_reward()

        assert "yaw_tracking" not in baseline_terms
        assert candidate_terms["yaw_tracking"] == pytest.approx(weight)
        assert candidate_reward - baseline_reward == pytest.approx(weight)
    finally:
        baseline.close()
        candidate.close()
