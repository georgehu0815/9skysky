"""Focused regression coverage for the studio dance pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "ppo_microduck_studio.py"


@pytest.fixture
def studio():
    spec = importlib.util.spec_from_file_location("dance_pipeline_studio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dance_clip(tmp_path):
    path = tmp_path / "custom-dance.json"
    path.write_bytes(b'{"version":3,"name":"test dance"}\n')
    return path


def _argv(command: str, clip: Path, tmp_path: Path) -> list[str]:
    argv = [command, "--recipe", "dance", "--dance-clip", str(clip)]
    if command == "eval":
        argv.extend(["--policy", str(tmp_path / "dance.onnx")])
    elif command == "render":
        argv.extend(
            [
                "--policy",
                str(tmp_path / "dance.onnx"),
                "--output",
                str(tmp_path / "render"),
            ]
        )
    return argv


@pytest.mark.parametrize("command", ["train", "eval"])
def test_dance_clip_is_forwarded_and_hashed_in_command_metadata(
    studio, monkeypatch, dance_clip, tmp_path, command
):
    args = studio.parse_args(_argv(command, dance_clip, tmp_path))
    expected_path = str(dance_clip.resolve())
    expected_hash = hashlib.sha256(dance_clip.read_bytes()).hexdigest()

    options = studio._recipe_options(args)
    assert options == {
        "dance_clip": expected_path,
        "dance_clip_sha256": expected_hash,
    }

    if command == "train":
        assert studio._metadata(args, 17)["recipe_options"] == options

    captured = {}
    sentinel = object()

    def fake_factory(recipe, **kwargs):
        captured.update(recipe=recipe, **kwargs)
        return sentinel

    monkeypatch.setattr(studio, "make_recipe_env", fake_factory)
    assert studio._environment(args, normalize=command == "train") is sentinel
    assert captured["recipe"] == "dance"
    assert captured["dance_clip"] == dance_clip


def test_render_forwards_dance_clip_and_reports_hashed_metadata(
    studio, monkeypatch, dance_clip, tmp_path
):
    import imageio.v2 as imageio
    import mujoco
    import microduck_local.render_rollout as render_rollout

    policy = tmp_path / "dance.onnx"
    policy.write_bytes(b"fake policy")
    captured = {}

    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.model = object()
            self.data = SimpleNamespace(xpos=np.array([[0.0, 0.0, 0.2]]))
            self.trunk_body_id = 0
            self.terminate_on_fall = True
            self.closed = False

        def reset(self, seed):
            return np.zeros(61, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(61, dtype=np.float32),
                1.0,
                False,
                True,
                {},
            )

        def recipe_metrics(self):
            return {"pose_rmse_rad": 0.0}

        def close(self):
            self.closed = True

    env = FakeEnv()

    def fake_factory(recipe, **kwargs):
        captured.update(recipe=recipe, **kwargs)
        return env

    class FakeRenderer:
        def __init__(self, model, *, height, width):
            self.height = height
            self.width = width

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def append_data(self, frame):
            pass

    monkeypatch.setattr(studio, "make_single_recipe_env", fake_factory)
    monkeypatch.setattr(
        studio,
        "_render_policy_source",
        lambda source, source_type: lambda observation: np.zeros(
            14, dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        studio,
        "_camera",
        lambda recipe, name, distance: SimpleNamespace(lookat=np.zeros(3)),
    )
    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: FakeWriter())
    monkeypatch.setattr(render_rollout, "sheet_indices", lambda count, limit: [0])
    monkeypatch.setattr(render_rollout, "build_sheet", lambda *args, **kwargs: None)

    args = studio.parse_args(
        [
            "render",
            "--recipe",
            "dance",
            "--dance-clip",
            str(dance_clip),
            "--policy",
            str(policy),
            "--output",
            str(tmp_path / "render"),
            "--render-seconds",
            "0.02",
        ]
    )
    result = studio.render(args)

    assert env.closed
    assert captured["dance_clip"] == dance_clip
    assert result["environment"]["recipe_options"] == {
        "dance_clip": str(dance_clip.resolve()),
        "dance_clip_sha256": hashlib.sha256(dance_clip.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    ("recipe", "normalize_rewards", "initial_std"),
    [
        ("dance", False, math.exp(-0.5)),
        ("running", False, math.exp(-0.5)),
        ("stilts", False, math.exp(-0.5)),
        ("swing", True, 0.1),
    ],
)
def test_training_defaults_keep_recipe_specific_reward_normalization_and_std(
    studio, recipe, normalize_rewards, initial_std
):
    args = studio.parse_args(["train", "--recipe", recipe])
    metadata = studio._metadata(args, 0)

    assert args.normalize_rewards is normalize_rewards
    assert args.initial_std == pytest.approx(initial_std)
    assert metadata["ppo"]["normalize_rewards"] is normalize_rewards
    assert metadata["ppo"]["initial_std"] == pytest.approx(initial_std)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--normalize-rewards", True), ("--no-normalize-rewards", False)],
)
def test_explicit_training_normalization_and_initial_std_are_preserved(
    studio, flag, expected
):
    args = studio.parse_args(
        ["train", "--recipe", "dance", flag, "--initial-std", "0.27"]
    )
    metadata = studio._metadata(args, 9)

    assert args.normalize_rewards is expected
    assert args.initial_std == pytest.approx(0.27)
    assert metadata["ppo"]["normalize_rewards"] is expected
    assert metadata["ppo"]["initial_std"] == pytest.approx(0.27)


def test_rollout_journal_keeps_phase_steps_and_cumulative_env_steps(
    studio, tmp_path
):
    journal = tmp_path / "training-metrics.jsonl"
    algorithm = SimpleNamespace(
        step=96,
        buffer=SimpleNamespace(
            rewards=np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        ),
    )
    observer = studio._training_rollout_observer(200, algorithm, journal)

    observer({"phase": "collection", "steps": 24, "seconds": 0.5})
    algorithm.step = 120
    observer(
        {
            "phase": "update",
            "steps": 24,
            "seconds": 0.25,
            "optimizer_steps": 8,
            "mean_loss": 1.5,
        }
    )

    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert entries[0] == {
        "phase": "collection",
        "steps": 24,
        "seconds": 0.5,
        "env_steps": 96,
        "mean_reward": 4.0,
    }
    assert entries[1]["phase"] == "update"
    assert entries[1]["steps"] == 24
    assert entries[1]["env_steps"] == 120


def test_episode_journal_persists_raw_returns_and_lengths(studio, tmp_path):
    journal = tmp_path / "training-metrics.jsonl"
    callback = studio._training_progress_callback(100, 2, 10, journal)

    callback(
        {
            "_episode": np.array([True, False, True]),
            "episode": {
                "r": np.array([3.25, 999.0, -1.5], dtype=np.float32),
                "l": np.array([12, 999, 8], dtype=np.int64),
            },
        },
        18,
    )

    entry = json.loads(journal.read_text())
    assert entry == {
        "phase": "episodes",
        "env_steps": 20,
        "returns": [3.25, -1.5],
        "lengths": [12, 8],
        "mean_raw_return": 0.875,
    }


def _dance_state(step: int, *, tracking: bool) -> dict[str, object]:
    target = np.zeros(14, dtype=np.float64)
    target[0] = (-0.12, 0.12, -0.12, 0.12)[step]
    target[1] = (0.12, -0.12, 0.12, -0.12)[step]
    return {
        "phase_step": step,
        "current_joints": target.copy() if tracking else np.zeros(14),
        "target_joints": target,
        "upright": 1.0,
        "height_m": 0.2,
    }


class _DanceEnvironment:
    def __init__(self, *, tracking: bool):
        self.tracking = tracking
        self.step_index = 0
        self.closed = False

    def reset(self, key):
        return np.zeros((1, 61), dtype=np.float32), {}, {}

    def step(self, key, state, action):
        step = self.step_index
        self.step_index += 1
        done = self.step_index == 4
        return (
            np.zeros((1, 61), dtype=np.float32),
            {},
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            np.array([done], dtype=bool),
            {
                "infos": [
                    {
                        "recipe_metrics": {},
                        "dance_state": _dance_state(
                            step, tracking=self.tracking
                        ),
                    }
                ],
                "_episode": np.array([done], dtype=bool),
                "episode": {
                    "r": np.array([4.0], dtype=np.float32),
                    "l": np.array([4], dtype=np.int64),
                },
            },
        )

    def close(self):
        self.closed = True


def _evaluate_dance(studio, monkeypatch, tmp_path, *, tracking: bool, mode: str, clip_duration: float = 0.08):
    policy = tmp_path / f"dance-{mode}-{tracking}.onnx"
    policy.write_bytes(b"fake policy")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "duration": clip_duration, "loop": True,
        "keys": [{"t": 0, "joints": [0.0] * 14},
                 {"t": clip_duration, "joints": [0.1] * 14}],
    }))
    env = _DanceEnvironment(tracking=tracking)
    monkeypatch.setattr(studio, "_environment", lambda args, normalize: env)
    monkeypatch.setattr(
        studio,
        "_policy_from_onnx",
        lambda path: lambda observation: np.zeros((1, 14), dtype=np.float32),
    )
    args = studio.parse_args(
        [
            "eval",
            "--recipe",
            "dance",
            "--dance-clip",
            str(reference),
            "--policy",
            str(policy),
            "--num-envs",
            "1",
            "--eval-steps",
            "4",
            "--max-episode-s",
            "0.08",
            "--evaluation-mode",
            mode,
        ]
    )
    result = studio.evaluate(args)
    assert env.closed
    return result


def test_dance_skill_cannot_pass_by_shortening_the_reference_horizon(studio, monkeypatch, tmp_path):
    result = _evaluate_dance(studio, monkeypatch, tmp_path, tracking=True, mode="skill", clip_duration=0.16)
    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "failed"
    assert result["evaluation"]["dance_criteria"]["required_steps"] == 8
    assert "episode did not cover the requested clip horizon" in result["failures"]


def test_dance_skill_evaluation_passes_with_time_aligned_tracking(
    studio, monkeypatch, tmp_path
):
    result = _evaluate_dance(
        studio, monkeypatch, tmp_path, tracking=True, mode="skill"
    )

    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "passed"
    assert result["passed"] is True
    assert result["dance_assessment"]["passed"] is True
    assert result["dance_assessment"]["completed_episodes"] == 1


def test_dance_skill_failure_does_not_relabel_pipeline_failure(
    studio, monkeypatch, tmp_path
):
    result = _evaluate_dance(
        studio, monkeypatch, tmp_path, tracking=False, mode="skill"
    )

    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "failed"
    assert result["passed"] is False
    assert "dynamic tracking gain below target" in result["failures"]


def test_dance_pipeline_mode_reports_but_does_not_enforce_skill(
    studio, monkeypatch, tmp_path
):
    result = _evaluate_dance(
        studio, monkeypatch, tmp_path, tracking=False, mode="pipeline"
    )

    assert result["pipeline_passed"] is True
    assert result["skill_status"] == "not_assessed"
    assert result["passed"] is True
    assert result["dance_assessment"]["passed"] is False
