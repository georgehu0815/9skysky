"""Native Swing reset, actuator-target, and signed-reward regressions."""

import mujoco
import numpy as np
import pytest

from rlx.environments.microduck_recipes import (
    SWING_REWARD_WEIGHTS,
    SWING_STRING_LENGTH,
    SWING_STRING_STIFFNESS,
    make_single_recipe_env,
    validate_reward_weights,
)


@pytest.fixture
def swing_factory():
    environments = []

    def make(**kwargs):
        env = make_single_recipe_env(
            "swing",
            actuator="xml",
            domain_rand=False,
            obs_noise=False,
            action_delay=True,
            random_yaw=False,
            max_episode_s=0.1,
            **kwargs,
        )
        environments.append(env)
        return env

    yield make
    for env in environments:
        env.close()


def test_assisted_reset_repeats_seed_across_resets_and_environments(swing_factory):
    environments = [
        swing_factory(
            seed=seed,
            swing_initial_angle_deg=12.0,
            swing_initial_rate_rad_s=0.35,
        )
        for seed in (1, 99)
    ]
    reference = None
    for env in environments:
        for _ in range(3):
            observation, _ = env.reset(seed=7)
            actual = (observation, env.unwrapped.data.qpos, env.unwrapped.data.qvel)
            assert observation.shape == (61,)
            assert np.isfinite(observation).all()
            if reference is None:
                reference = tuple(value.copy() for value in actual)
            for value, expected in zip(actual, reference):
                np.testing.assert_array_equal(value, expected)
            env.step(np.full(14, 0.1, dtype=np.float32))


def test_assisted_reset_different_seeds_differ_within_bounds(swing_factory):
    env = swing_factory(
        swing_initial_angle_deg=12.0,
        swing_initial_rate_rad_s=0.35,
    )
    samples = []
    for seed in range(8):
        env.reset(seed=seed)
        metrics = env.unwrapped.recipe_metrics()
        angle, rate = metrics["swing_angle_deg"], metrics["swing_rate_rad_s"]
        assert np.isfinite([angle, rate]).all()
        assert abs(angle) <= 12.0
        assert abs(rate) <= 0.35
        samples.append((angle, rate))
    assert len(set(samples)) == len(samples)


@pytest.mark.parametrize("lag", [0, 1])
def test_zero_action_keeps_seated_target_and_raw_history_after_reset(swing_factory, lag):
    from microduck_local.contract import DEFAULT_POSE

    env = swing_factory()
    native = env.unwrapped
    for _ in range(2):
        observation, _ = env.reset(seed=7)
        native._action_lag = lag
        np.testing.assert_array_equal(observation[34:48], np.zeros(14))
        np.testing.assert_array_equal(native._policy_last_action, np.zeros(14))
        np.testing.assert_array_equal(native._policy_prev_action, np.zeros(14))
        np.testing.assert_allclose(
            DEFAULT_POSE + native._delayed_action, native._SEATED_POSE, atol=1e-7
        )
        for _ in range(2):
            observation, reward, _, _, _ = env.step(np.zeros(14, dtype=np.float32))
            np.testing.assert_allclose(native.data.ctrl, native._SEATED_POSE, atol=1e-7)
            np.testing.assert_array_equal(observation[34:48], np.zeros(14))
            assert observation.shape == (61,)
            assert np.isfinite(observation).all()
            assert np.isfinite(reward)
            assert native.reward_sums["action_rate_penalty"] == 0.0
            assert np.isfinite(list(native.reward_sums.values())).all()
            assert all(
                value <= 0.0
                for key, value in native.reward_sums.items()
                if key.endswith("penalty")
            )
        env.step(np.full(14, 0.2, dtype=np.float32))


def test_swing_actions_are_bounded_to_reference_contract(swing_factory):
    env = swing_factory()
    env.reset(seed=7)
    native = env.unwrapped
    env.step(np.full(14, 3.0, dtype=np.float32))
    np.testing.assert_array_equal(native._policy_last_action, np.ones(14))


def test_planar_swing_actions_project_to_sagittal_symmetry(swing_factory):
    env = swing_factory(swing_planar_actions=True)
    env.reset(seed=7)
    native = env.unwrapped
    env.step(np.arange(14, dtype=np.float32) / 5.0 - 1.0)
    action = native._policy_last_action
    np.testing.assert_array_equal(action[[0, 1, 7, 8, 9, 10]], np.zeros(6))
    assert action[2] == pytest.approx(-action[11])
    assert action[3] == pytest.approx(-action[12])
    assert action[4] == pytest.approx(-action[13])


@pytest.mark.parametrize("boundary", ["lower", "upper"])
@pytest.mark.parametrize("excess", [-0.01, 0.0, 0.1])
def test_neck_pitch_limit_penalty_uses_each_actual_boundary(swing_factory, boundary, excess):
    env = swing_factory(weight_overrides={"joint_limit_penalty": 2.5})
    env.reset(seed=7)
    native = env.unwrapped
    joint = native.model.joint("neck_pitch")
    lower, upper = native.model.jnt_range[joint.id]
    assert abs(lower) > abs(upper)
    position = lower - excess if boundary == "lower" else upper + excess
    native.data.qpos[joint.qposadr[0]] = position
    mujoco.mj_forward(native.model, native.data)

    reward, terms = native._compute_reward()

    assert terms["joint_limit_penalty"] == pytest.approx(-2.5 * max(excess, 0.0) ** 2)
    assert reward == pytest.approx(sum(terms.values()))
    assert np.isfinite(list(terms.values())).all()
    assert all(value <= 0.0 for key, value in terms.items() if key.endswith("penalty"))


def test_joint_limit_penalty_sums_lower_and_upper_excess_in_actuator_order(swing_factory):
    env = swing_factory()
    env.reset(seed=7)
    native = env.unwrapped
    joint_ids = native.model.actuator_trnid[:, 0]
    qpos_addresses = native.model.jnt_qposadr[joint_ids]
    ranges = native.model.jnt_range[joint_ids]
    native.data.qpos[qpos_addresses] = ranges.mean(axis=1)
    native.data.qpos[qpos_addresses[0]] = ranges[0, 0] - 0.1
    native.data.qpos[qpos_addresses[-1]] = ranges[-1, 1] + 0.2
    mujoco.mj_forward(native.model, native.data)

    reward, terms = native._compute_reward()

    assert terms["joint_limit_penalty"] == pytest.approx(-(0.1**2 + 0.2**2))
    assert reward == pytest.approx(sum(terms.values()))
    assert set(terms) == set(SWING_REWARD_WEIGHTS)
    assert np.isfinite(list(terms.values())).all()
    for key, value in terms.items():
        assert value <= 0.0 if key.endswith("penalty") else value >= 0.0


@pytest.mark.parametrize("invalid_weight", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_swing_weights_reject_negative_or_nonfinite_values(invalid_weight):
    for key in SWING_REWARD_WEIGHTS:
        with pytest.raises(ValueError, match="finite and non-negative"):
            validate_reward_weights("swing", {key: invalid_weight})


@pytest.mark.parametrize("height_offset", [0.0, 0.02, -0.005])
def test_string_metrics_report_sampled_lengths_and_spring_tension(swing_factory, height_offset):
    env = swing_factory()
    env.reset(seed=7)
    native = env.unwrapped
    native.data.qpos[2] += height_offset
    native.data.qpos[3:7] = (np.cos(0.015), np.sin(0.015), 0.0, 0.0)
    mujoco.mj_forward(native.model, native.data)
    state = native._swing_state()

    metrics = native.recipe_metrics()

    old_metrics = {
        "swing_angle_deg", "swing_abs_angle_deg", "swing_rate_rad_s",
        "lateral_offset_m", "string_slack_m", "string_imbalance_m",
        "alignment_penalty", "valid_geometry",
    }
    assert old_metrics <= metrics.keys()
    assert np.isfinite(list(metrics.values())).all()
    assert metrics["string_left_m"] != metrics["string_right_m"]
    for index, side in enumerate(("left", "right")):
        length = metrics[f"string_{side}_m"]
        tension = metrics[f"string_{side}_tension_n"]
        assert length == state["lengths"][index]
        assert length == native.data.ten_length[native._swing_tendon_ids[index]]
        assert tension >= 0.0
        assert tension == pytest.approx(
            SWING_STRING_STIFFNESS * max(length - SWING_STRING_LENGTH, 0.0)
        )
        if height_offset == 0.02:
            assert tension == 0.0
        elif height_offset == -0.005:
            assert tension > 0.0
    assert metrics["valid_geometry"] == 1.0
    native._swing_invalid = True
    assert native.recipe_metrics()["valid_geometry"] == 0.0


def test_swing_default_weights_are_finite_nonnegative_and_accept_zero():
    assert validate_reward_weights("swing", SWING_REWARD_WEIGHTS) == SWING_REWARD_WEIGHTS
    assert np.isfinite(list(SWING_REWARD_WEIGHTS.values())).all()
    assert all(weight >= 0.0 for weight in SWING_REWARD_WEIGHTS.values())
    zeros = dict.fromkeys(SWING_REWARD_WEIGHTS, 0.0)
    assert validate_reward_weights("swing", zeros) == zeros
