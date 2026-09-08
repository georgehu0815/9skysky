"""Studio integration for authoritative running and stilt skill verdicts."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import gymnasium as gym

from rlx.environments.microduck_recipes import (
    _locomotion_recipe_metrics,
    _RecipeMetricsWrapper,
    make_single_recipe_env,
)
from test_locomotion_evaluation import samples


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recipe", ["running", "stilts"])
@pytest.mark.parametrize("done", [False, True])
def test_metrics_preserve_command_active_before_physics(recipe, done):
    class ResamplingEnv(gym.Env):
        def __init__(self):
            self.twist_cmd = np.array([0.6, -0.1, 0.2])
            self.observation = np.zeros(61, dtype=np.float32)

        def step(self, action):
            self.twist_cmd[:] = (-0.3, 0.2, -0.4)
            self.observation[48:51] = self.twist_cmd
            return self.observation, 1.25, False, done, {"original": True}

        def recipe_metrics(self):
            return {
                "command_forward_m_s": float(self.twist_cmd[0]),
                "command_lateral_m_s": float(self.twist_cmd[1]),
                "command_yaw_rad_s": float(self.twist_cmd[2]),
                "world_x_m": 0.008,
            }

    env = ResamplingEnv()
    wrapper = _RecipeMetricsWrapper(env, recipe)
    observation, reward, terminated, truncated, info = wrapper.step(np.zeros(14))
    metrics = info["recipe_metrics"]
    assert metrics["command_forward_m_s"] == 0.6
    assert metrics["command_lateral_m_s"] == -0.1
    assert metrics["command_yaw_rad_s"] == 0.2
    assert metrics["world_x_m"] == 0.008
    assert observation is env.observation
    assert observation[48:51] == pytest.approx([-0.3, 0.2, -0.4])
    assert (reward, terminated, truncated) == (1.25, False, done)
    assert info["original"] is True
    json.dumps(metrics, allow_nan=False)
    _, _, _, _, next_info = wrapper.step(np.zeros(14))
    assert next_info["recipe_metrics"]["command_forward_m_s"] == -0.3
    assert next_info["recipe_metrics"]["command_lateral_m_s"] == 0.2
    assert next_info["recipe_metrics"]["command_yaw_rad_s"] == -0.4


@pytest.mark.parametrize("recipe,fixed_command", [("running", 0.6), ("stilts", 0.15)])
@pytest.mark.parametrize("pinned", [False, True])
def test_real_resampling_reports_previous_command_without_changing_observation(
    recipe, fixed_command, pinned
):
    env = make_single_recipe_env(
        recipe, seed=7, actuator="xml", domain_rand=False, obs_noise=False,
        action_delay=False, random_yaw=False,
        locomotion_forward_command=fixed_command if pinned else None,
    )
    try:
        observation, _ = env.reset(seed=7)
        env.unwrapped.resample_steps = 1
        for _ in range(3):
            active_command = observation[48:51].copy()
            observation, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
            metrics = info["recipe_metrics"]
            assert [metrics[key] for key in (
                "command_forward_m_s", "command_lateral_m_s", "command_yaw_rad_s"
            )] == pytest.approx(active_command)
            assert observation[48:51] == pytest.approx(env.unwrapped.twist_cmd)
            if pinned:
                assert observation[48:51] == pytest.approx([fixed_command, 0.0, 0.0])
            json.dumps(metrics, allow_nan=False)
    finally:
        env.close()


@pytest.fixture
def studio():
    spec = importlib.util.spec_from_file_location(
        "studio_locomotion_tests",
        ROOT / "examples/ppo_microduck_studio.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEnvironment:
    def __init__(self, frames):
        self.frames = frames
        self.step_index = 0
        self.closed = False

    def reset(self, key):
        return np.zeros((1, 61), np.float32), {}, {}

    def step(self, key, state, action):
        index = self.step_index
        self.step_index += 1
        done = self.step_index == len(self.frames)
        return (
            np.zeros((1, 61), np.float32),
            {},
            np.array([1.0]),
            np.array([False]),
            np.array([done]),
            {
                "infos": [{"recipe_metrics": self.frames[index]}],
                "_episode": np.array([done]),
                "episode": {
                    "r": np.array([float(len(self.frames))]),
                    "l": np.array([len(self.frames)]),
                },
            },
        )

    def close(self):
        self.closed = True


def evaluate(studio, monkeypatch, tmp_path, recipe, frames, mode="skill"):
    source = tmp_path / f"{recipe}.onnx"
    source.write_bytes(b"fake model")
    env = FakeEnvironment(frames)
    monkeypatch.setattr(studio, "_environment", lambda args, normalize: env)
    monkeypatch.setattr(
        studio,
        "_policy_from_onnx",
        lambda path: lambda observation: np.zeros((1, 14), np.float32),
    )
    args = studio.parse_args(
        [
            "eval",
            "--recipe",
            recipe,
            "--policy",
            str(source),
            "--num-envs",
            "1",
            "--eval-steps",
            str(len(frames)),
            "--evaluation-mode",
            mode,
            "--no-domain-rand",
            "--no-obs-noise",
            "--no-action-delay",
            "--no-random-yaw",
        ]
    )
    result = studio.evaluate(args)
    assert env.closed
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("recipe", ["running", "stilts"])
def test_skill_mode_publishes_authoritative_locomotion_status(
    studio, monkeypatch, tmp_path, recipe
):
    result = evaluate(studio, monkeypatch, tmp_path, recipe, samples(recipe))

    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "passed"
    assert result["passed"] is True
    assert result["locomotion_assessment"]["passed"] is True
    assert result["evaluation"]["locomotion_criteria"]["recipe"] == recipe
    assert result["dance_assessment"] is None
    assert result["swing_assessment"] is None


def test_pipeline_mode_never_claims_running_skill(
    studio, monkeypatch, tmp_path
):
    result = evaluate(
        studio,
        monkeypatch,
        tmp_path,
        "running",
        samples("running"),
        mode="pipeline",
    )

    assert result["pipeline_passed"] is True
    assert result["passed"] is True
    assert result["skill_status"] == "not_assessed"
    assert result["locomotion_assessment"]["passed"] is True


def test_stationary_running_fails_skill_but_not_pipeline_execution(
    studio, monkeypatch, tmp_path
):
    frames = samples("running")
    for metric in frames:
        metric["heading_forward_speed_m_s"] = 0.0
        metric["heading_lateral_speed_m_s"] = 0.0
        metric["world_x_m"] = 0.0
        metric["world_y_m"] = 0.0
    result = evaluate(studio, monkeypatch, tmp_path, "running", frames)

    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "failed"
    assert result["passed"] is False
    assert "command-directed speed below target" in result["failures"]


def test_locomotion_eval_defaults_cover_full_recipe_horizons(studio):
    running = studio.parse_args(["eval", "--recipe", "running"])
    stilts = studio.parse_args(["eval", "--recipe", "stilts"])

    assert running.eval_steps == 600
    assert stilts.eval_steps == 500
    assert running.backend == stilts.backend == "dummy"


@pytest.mark.parametrize(
    "command,recipe",
    [(0.6, "running"), (0.15, "stilts")],
)
def test_fixed_forward_command_is_visible_on_reset_and_resample(command, recipe):
    env = make_single_recipe_env(
        recipe,
        seed=7,
        actuator="xml",
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        locomotion_forward_command=command,
    )
    try:
        observation, _ = env.reset(seed=7)
        assert observation[48:51].tolist() == pytest.approx([command, 0.0, 0.0])
        env.unwrapped._sample_commands()
        assert env.unwrapped.twist_cmd.tolist() == pytest.approx(
            [command, 0.0, 0.0]
        )
    finally:
        env.close()


@pytest.mark.parametrize("command,recipe", [(0.6, "running"), (0.15, "stilts")])
def test_cli_plumbs_fixed_command_through_all_environment_modes(
    studio, monkeypatch, tmp_path, command, recipe
):
    captured = []

    def fake_make_recipe_env(name, **kwargs):
        captured.append((name, kwargs))
        return object()

    monkeypatch.setattr(studio, "make_recipe_env", fake_make_recipe_env)
    for mode in ("train", "eval"):
        args = studio.parse_args(
            [
                mode,
                "--recipe",
                recipe,
                "--locomotion-forward-command",
                str(command),
            ]
        )
        assert args.locomotion_forward_command == command
        assert studio._recipe_options(args)["locomotion_forward_command"] == command
        if mode == "train":
            assert (
                studio._metadata(args, 0)["recipe_options"][
                    "locomotion_forward_command"
                ]
                == command
            )
        studio._environment(args, normalize=mode == "train")

    assert [kwargs["locomotion_forward_command"] for _, kwargs in captured] == [
        command,
        command,
    ]

    policy = tmp_path / f"{recipe}.onnx"
    policy.touch()
    render_args = studio.parse_args(
        [
            "render",
            "--recipe",
            recipe,
            "--policy",
            str(policy),
            "--output",
            str(tmp_path / "render"),
            "--locomotion-forward-command",
            str(command),
        ]
    )
    assert render_args.locomotion_forward_command == command
    assert (
        studio._recipe_options(render_args)["locomotion_forward_command"]
        == command
    )


@pytest.mark.parametrize("recipe", ["dance", "swing"])
def test_fixed_forward_command_is_locomotion_only(studio, recipe):
    with pytest.raises(SystemExit):
        studio.parse_args(
            [
                "eval",
                "--recipe",
                recipe,
                "--locomotion-forward-command",
                "0.6",
            ]
        )


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5001", "nan", "inf"])
def test_fixed_forward_command_rejects_invalid_values(studio, value):
    with pytest.raises(SystemExit):
        studio.parse_args(
            [
                "eval",
                "--recipe",
                "running",
                "--locomotion-forward-command",
                value,
            ]
        )


@pytest.mark.parametrize("recipe", ["running", "stilts"])
def test_locomotion_evaluation_rejects_terminal_only_fork_backend(studio, recipe):
    with pytest.raises(SystemExit):
        studio.parse_args(["eval", "--recipe", recipe, "--backend", "fork"])


def test_recipe_metrics_are_finite_serializable_and_use_heading_frame():
    class FakeEnv:
        twist_cmd = np.array([-0.4, 0.2, 0.0])
        _trunk_xpos = np.array([1.25, -0.5, 0.2])
        _trunk_xmat = np.array(
            [
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ]
        )

        def _projected_gravity(self):
            return np.array([0.0, 0.0, -0.95])

        def heading_lin_vel(self):
            return -0.3, 0.12, 0.0

        def _foot_contacts(self):
            return {"left": True, "right": False}

    metrics = _locomotion_recipe_metrics(FakeEnv())

    assert metrics["command_forward_m_s"] == pytest.approx(-0.4)
    assert metrics["command_lateral_m_s"] == pytest.approx(0.2)
    assert metrics["heading_forward_speed_m_s"] == pytest.approx(-0.3)
    assert metrics["heading_forward_x"] == pytest.approx(0.0)
    assert metrics["heading_forward_y"] == pytest.approx(1.0)
    assert metrics["left_foot_contact"] == 1.0
    assert metrics["right_foot_contact"] == 0.0
    json.dumps(metrics, allow_nan=False)
