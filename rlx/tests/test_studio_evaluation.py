"""CLI scope, checkpoint inference and failed Swing assessments remain explicit."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def studio():
    spec = importlib.util.spec_from_file_location("studio_evaluation_tests", ROOT / "examples/ppo_microduck_studio.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def physical(angle=0.0):
    return dict(swing_angle_deg=angle, lateral_offset_m=0.0, alignment_penalty=0.0,
                valid_geometry=1.0, string_left_m=.382, string_right_m=.382,
                string_left_tension_n=4.0, string_right_tension_n=4.0)


class FakeEnvironment:
    def __init__(self, frames, *, episode_steps=1200, mutate=None, reward=1.0):
        self.frames = frames
        self.episode_steps = episode_steps
        self.mutate = mutate
        self.reward = reward
        self.step_index = 0
        self.closed = False

    def reset(self, key):
        return np.zeros((1, 61), np.float32), {}, {}

    def step(self, key, state, action):
        index = self.step_index
        self.step_index += 1
        if self.mutate is not None:
            self.mutate(index)
        done = self.step_index % self.episode_steps == 0
        return (np.zeros((1, 61), np.float32), {}, np.array([self.reward]),
                np.array([False]), np.array([done]),
                {"infos": [{"recipe_metrics": self.frames[index]}], "_episode": np.array([done]),
                 "episode": {"r": np.array([float(self.episode_steps)]), "l": np.array([self.episode_steps])}})

    def close(self):
        self.closed = True


def evaluate(studio, monkeypatch, tmp_path, frames, *, mode="skill", episode_steps=1200, mutate=None, reward=1.0):
    source = tmp_path / "swing.onnx"
    source.write_bytes(b"fake model bytes")
    env = FakeEnvironment(frames, episode_steps=episode_steps,
                          mutate=None if mutate is None else lambda index: mutate(source, index), reward=reward)
    monkeypatch.setattr(studio, "_environment", lambda args, normalize: env)
    monkeypatch.setattr(studio, "_policy_from_onnx", lambda path: lambda obs: np.zeros((1, 14), np.float32))
    args = studio.parse_args(["eval", "--recipe", "swing", "--policy", str(source), "--num-envs", "1",
                              "--eval-steps", str(len(frames)), "--evaluation-mode", mode,
                              "--no-domain-rand", "--no-obs-noise", "--no-action-delay", "--no-random-yaw"])
    result = studio.evaluate(args)
    assert env.closed
    json.dumps(result, allow_nan=False)
    return result


def test_stationary_full_rollout_passes_pipeline_but_fails_skill(studio, monkeypatch, tmp_path):
    result = evaluate(studio, monkeypatch, tmp_path, [physical()] * 1200)
    assert result["finite"] and result["pipeline_passed"]
    assert result["passed"] is False
    assert result["skill_status"] == "failed"
    assert result["swing_assessment"]["completed_episodes"] == 1
    assert "bidirectional span below target" in result["failures"]
    assert result["evaluation"]["swing_criteria"]["min_bidirectional_span_deg"] == 150
    assert result["evaluation"]["swing_criteria"]["required_steps"] == 1200
    assert result["evaluation"]["environment"]["action_delay"] is False
    assert result["evaluation"]["environment"]["reward_weights"]["swing_peak_progress"] == 224
    assert len(result["source_sha256"]) == 64


def test_full_valid_bidirectional_episode_passes_skill(studio, monkeypatch, tmp_path):
    result = evaluate(studio, monkeypatch, tmp_path, [physical(75), physical(-75)] + [physical()] * 1198)
    assert result["passed"] is True
    assert result["skill_status"] == "passed"
    assert result["recipe_metrics"]["swing_bidirectional_span_deg"]["min"] == 150


def test_smoke_never_claims_skill_and_short_skill_evaluation_fails(studio, monkeypatch, tmp_path):
    frames = [physical(80), physical(-80)] * 2
    smoke = evaluate(studio, monkeypatch, tmp_path, frames, mode="pipeline")
    assert smoke["passed"] and smoke["pipeline_passed"]
    assert smoke["skill_status"] == "not_assessed"
    assert smoke["swing_assessment"]["passed"] is False
    full = evaluate(studio, monkeypatch, tmp_path, frames)
    assert full["passed"] is False
    assert "incomplete 24-second episode" in full["failures"]


def test_opposite_reset_episodes_do_not_pool_span(studio, monkeypatch, tmp_path):
    frames = [physical(90)] * 600 + [physical(-90)] * 600
    result = evaluate(studio, monkeypatch, tmp_path, frames, episode_steps=600)
    assert result["recipe_metrics"]["swing_span_deg"]["max"] == 90
    assert result["recipe_metrics"]["swing_bidirectional_span_deg"]["max"] == 0
    assert result["swing_assessment"]["incomplete_episodes"] == 2
    assert result["passed"] is False


def test_changed_source_fails_even_pipeline_evaluation(studio, monkeypatch, tmp_path):
    def replace(source, index):
        source.write_bytes(b"different model bytes")
    result = evaluate(studio, monkeypatch, tmp_path, [physical()], mode="pipeline", mutate=replace)
    assert result["pipeline_passed"] is False
    assert result["passed"] is False
    assert "policy source changed during evaluation" in result["failures"]


def test_nonfinite_rollout_emits_failed_strict_json(studio, monkeypatch, tmp_path):
    result = evaluate(studio, monkeypatch, tmp_path, [physical()], mode="pipeline", reward=float("nan"))
    assert result["finite"] is False
    assert result["pipeline_passed"] is False
    assert result["mean_return"] is None
    assert result["rollout_return"]["mean"] is None


@pytest.mark.parametrize("recipe", ["swing", "dance", "running", "stilts"])
def test_export_parser_has_no_environment_dependencies(studio, recipe):
    args = studio.parse_args(["export", "--recipe", recipe, "--checkpoint", "input.safetensors", "--output", "output.onnx"])
    assert args.checkpoint == Path("input.safetensors")
    assert not hasattr(args, "swing_initial_angle_deg")
    assert not hasattr(args, "weight_overrides")


def test_swing_default_evaluation_covers_24_seconds_and_starts_still(studio):
    args = studio.parse_args(["eval", "--recipe", "swing"])
    assert args.eval_steps == 1200 and args.evaluation_mode == "skill"
    assert args.backend == "dummy"
    assert args.swing_min_span_deg == 150
    assert args.swing_initial_angle_deg == args.swing_initial_rate_rad_s == 0
    with pytest.raises(SystemExit):
        studio.parse_args(["eval", "--recipe", "swing", "--swing-initial-angle-deg", "12"])
    assisted = studio.parse_args(["eval", "--recipe", "swing", "--evaluation-mode", "pipeline", "--swing-initial-angle-deg", "12"])
    assert assisted.swing_initial_angle_deg == 12


@pytest.mark.parametrize("option,value", [("--swing-min-span-deg", "0"), ("--swing-min-span-deg", "181"),
    ("--swing-min-span-deg", "nan"), ("--swing-min-span-deg", "inf"), ("--swing-initial-rate-rad-s", "nan"),
    ("--max-episode-s", "nan"), ("--min-episodes", "-1"), ("--min-mean-return", "nan")])
def test_evaluation_rejects_invalid_numeric_criteria(studio, option, value):
    with pytest.raises(SystemExit):
        studio.parse_args(["eval", "--recipe", "swing", option, value])


def test_checkpoint_render_callback_accepts_numpy_and_matches_model(studio, tmp_path):
    mx = pytest.importorskip("mlx.core")
    from rlx.models.microduck import create_actor_critic, save_checkpoint, normalize_observations
    mx.random.seed(9)
    model = create_actor_critic(initial_std=.1)
    path = tmp_path / "swing.safetensors"
    mean = np.arange(61, dtype=np.float32) / 100
    var = np.full(61, 2.0, np.float32)
    save_checkpoint(path, model, mean, var, 5)
    infer = studio._render_policy_source(path, "checkpoint")
    observation = np.ones(61, np.float32)
    expected = np.asarray(model.deterministic(normalize_observations(mx.array(observation[None, :]), mean, var)))[0]
    actual = infer(observation)
    assert actual.shape == (14,) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["pipeline", "skill"])
def test_swing_evaluation_rejects_terminal_only_fork_backend(studio, mode):
    with pytest.raises(SystemExit):
        studio.parse_args(["eval", "--recipe", "swing", "--backend", "fork", "--evaluation-mode", mode])
    args = studio.parse_args(["train", "--recipe", "swing"])
    assert args.backend == "fork"


def test_swing_reports_resolved_fixed_yaw_in_training_and_evaluation(studio, monkeypatch, tmp_path):
    args = studio.parse_args(["train", "--recipe", "swing", "--random-yaw"])
    assert studio._metadata(args, 0)["randomization"]["random_yaw"] is False
    path = tmp_path / "swing.onnx"
    path.touch()
    args = studio.parse_args(["eval", "--recipe", "swing", "--policy", str(path), "--num-envs", "1",
                              "--eval-steps", "1", "--evaluation-mode", "pipeline", "--random-yaw"])
    monkeypatch.setattr(studio, "_environment", lambda args, normalize: FakeEnvironment([physical()]))
    monkeypatch.setattr(studio, "_policy_from_onnx", lambda path: lambda obs: np.zeros((1, 14), np.float32))
    assert studio.evaluate(args)["evaluation"]["environment"]["random_yaw"] is False


def test_cli_failed_evaluation_uses_nonzero_status(studio, monkeypatch, capsys):
    monkeypatch.setattr(studio, "evaluate", lambda args: {"passed": False, "skill_status": "failed"})
    with pytest.raises(SystemExit) as error:
        studio.main(["eval", "--recipe", "swing"])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["skill_status"] == "failed"
