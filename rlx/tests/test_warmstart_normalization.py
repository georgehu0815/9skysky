"""Warmstart normalizer and provenance contracts without a Metal dependency."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from rlx.environments import microduck


class FakeVecEnv:
    num_envs = 2
    observation_space = gym.spaces.Box(-np.inf, np.inf, (61,), dtype=np.float32)
    action_space = gym.spaces.Box(-4.0, 4.0, (14,), dtype=np.float32)

    def __init__(self):
        self.steps = 0

    def reset(self):
        return np.full((2, 61), 5.0, dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        return (
            np.full((2, 61), 5.0 + self.steps, dtype=np.float32),
            np.array([1.0, 3.0]),
            np.array([True, False]),
            [{"TimeLimit.truncated": True,
              "terminal_observation": np.full(61, 9.0, dtype=np.float32)}, {}],
        )

    def close(self):
        pass


@pytest.fixture
def numpy_adapter(monkeypatch):
    monkeypatch.setattr(microduck, "_import_mlx", lambda: np)


def seed_statistics(env):
    env.observation_rms.mean[:] = 1.0
    env.observation_rms.var[:] = 4.0
    env.observation_rms.count = 17.0


@pytest.mark.parametrize("freeze", [False, True])
def test_freeze_only_observation_statistics_on_reset_step_and_timeout(numpy_adapter, freeze):
    env = microduck.MicroDuckVecEnv(
        FakeVecEnv(), normalize_observations=True, normalize_rewards=True,
        freeze_observation_normalization=freeze,
    )
    seed_statistics(env)
    initial_mean = env.observation_rms.mean.copy()
    initial_variance = env.observation_rms.var.copy()
    observations, _, _ = env.reset()
    if freeze:
        np.testing.assert_allclose(observations, 2.0)
    for step in range(1, 4):
        observations, _, rewards, _, _, info = env.step(None, {}, np.zeros((2, 14)))
        assert np.isfinite(rewards).all()
        assert env.return_rms.count == pytest.approx(1e-4 + 2 * step)
        assert float(env.return_rms.mean) > 0.0
        assert float(env.return_rms.var) != 1.0
        np.testing.assert_allclose(
            rewards, np.array([1.0, 3.0]) / np.sqrt(env.return_rms.var + env.epsilon),
        )
        if freeze:
            np.testing.assert_allclose(observations, (4.0 + step) / 2.0)
            np.testing.assert_allclose(info["terminal_observation"][0], 4.0)
            np.testing.assert_array_equal(env.observation_rms.mean, initial_mean)
            np.testing.assert_array_equal(env.observation_rms.var, initial_variance)
            assert env.observation_rms.count == 17.0
        else:
            assert env.observation_rms.count == 17.0 + 2 * (step + 1)
            assert not np.array_equal(env.observation_rms.mean, initial_mean)
            assert not np.array_equal(env.observation_rms.var, initial_variance)
    env.reset()
    assert env.observation_rms.count == (17.0 if freeze else 27.0)
    assert env.return_rms.count == pytest.approx(6.0001)


def test_default_updates_and_explicit_update_false_does_not(numpy_adapter):
    env = microduck.MicroDuckVecEnv(FakeVecEnv(), normalize_observations=True)
    assert env.freeze_observation_normalization is False
    env.reset()
    count = env.observation_rms.count
    env._normalize_observation(np.full((2, 61), 100.0), update=False)
    assert env.observation_rms.count == count
    env._normalize_observation(np.full((2, 61), 100.0))
    assert env.observation_rms.count == count + 2


@pytest.fixture
def studio():
    path = Path(__file__).resolve().parents[1] / "examples/ppo_microduck_studio.py"
    spec = importlib.util.spec_from_file_location("studio_warmstart_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_freeze_requires_checkpoint_and_defaults_false(studio, tmp_path, capsys):
    defaults = studio.parse_args(["train", "--recipe", "swing"])
    assert defaults.freeze_observation_normalization is False
    assert studio._metadata(defaults, 0)["freeze_observation_normalization"] is False
    assert "initialization" not in studio._metadata(defaults, 0)
    with pytest.raises(SystemExit):
        studio.parse_args(["train", "--recipe", "swing", "--freeze-observation-normalization"])
    assert "requires --init-from" in capsys.readouterr().err
    source = tmp_path / "teacher.safetensors"
    source.write_bytes(b"checkpoint")
    args = studio.parse_args([
        "train", "--recipe", "swing", "--init-from", str(source), "--freeze-observation-normalization",
    ])
    assert args.freeze_observation_normalization is True
    assert studio.parse_args([
        "train", "--recipe", "swing", "--no-freeze-observation-normalization",
    ]).freeze_observation_normalization is False


@pytest.mark.parametrize("command,extra", [("eval", []), ("render", ["--output", "/tmp/unused.mp4"]),
                                          ("export", [])])
def test_freeze_is_train_only(studio, command, extra, capsys):
    with pytest.raises(SystemExit):
        studio.parse_args([command, "--recipe", "swing", *extra, "--freeze-observation-normalization"])
    assert "unrecognized arguments: --freeze-observation-normalization" in capsys.readouterr().err


def test_initialization_is_a_snapshot_including_teacher_origin(studio, tmp_path):
    source = tmp_path / "stage3.safetensors"
    sidecar = source.with_suffix(".safetensors.json")
    source.write_bytes(b"best genuine teacher checkpoint")
    loaded_metadata = {"teacher_assisted": True, "steps": 0, "ppo_steps": 0,
                       "bootstrap": "BC + on-policy DAgger", "stage": 3}
    sidecar.write_text(json.dumps({"metadata": loaded_metadata}))
    initialization = studio._initialization_provenance(source, loaded_metadata)
    assert initialization == {
        "kind": "checkpoint", "source_checkpoint": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "loaded_metadata": loaded_metadata,
    }
    expected = deepcopy(initialization)
    loaded_metadata["stage"] = 5
    source.write_bytes(b"overwritten by PPO")
    sidecar.write_text("{}")
    args = studio.parse_args(["train", "--recipe", "swing", "--init-from", str(source), "--freeze-observation-normalization"])
    first = studio._metadata(args, 10, initialization=initialization)
    assert first["initialization"] == expected
    assert first["teacher_assisted"] is True
    assert first["steps"] == 10
    assert first["freeze_observation_normalization"] is True
    first["initialization"]["loaded_metadata"]["stage"] = 10
    assert studio._metadata(args, 20, initialization=initialization)["initialization"] == expected


@pytest.mark.parametrize("freeze", [False, True])
def test_train_snapshots_and_final_preserve_source_before_overwrite(
    studio, monkeypatch, numpy_adapter, tmp_path, freeze,
):
    source = tmp_path / "stage3.safetensors"
    source.write_bytes(b"initial BC checkpoint")
    sidecar = source.with_suffix(".safetensors.json")
    origin = {"teacher_assisted": True, "bootstrap": "BC + DAgger", "steps": 0, "stage": 3}
    sidecar.write_text(json.dumps({"metadata": origin}))
    expected_initialization = studio._initialization_provenance(source, origin)
    flags = ["--freeze-observation-normalization"] if freeze else []
    args = studio.parse_args([
        "train", "--recipe", "swing", "--init-from", str(source),
        "--checkpoint", str(source), "--num-envs", "2", "--num-steps", "1",
        "--num-minibatches", "1", "--total-timesteps", "2",
        "--checkpoint-interval", "2", "--no-export-onnx", *flags,
    ])
    env = microduck.MicroDuckVecEnv(
        FakeVecEnv(), normalize_observations=True, normalize_rewards=True,
    )
    monkeypatch.setattr(studio, "_environment", lambda *_args, **_kwargs: env)
    network = SimpleNamespace(parameters=lambda: {})
    loaded = {
        "model": network, "mean": np.full(61, 1.0), "variance": np.full(61, 4.0),
        "count": 17.0, "return_mean": np.array(0.0), "return_variance": np.array(1.0),
        "return_count": 2.0, "epsilon": 1e-6, "clip": 8.0, "metadata": origin,
    }
    snapshots = []

    def save_checkpoint(path, model, mean, variance, count, **kwargs):
        snapshots.append({"path": path, "mean": mean.copy(), "variance": variance.copy(),
                          "count": count, **deepcopy(kwargs)})
        source.write_bytes(b"updated PPO checkpoint")
        sidecar.write_text(json.dumps({"metadata": kwargs["metadata"]}))

    class FakePPO:
        def __init__(self, **kwargs):
            self.step = 0
            self.env = kwargs["env"]
            assert self.env.freeze_observation_normalization is freeze
            assert self.env.observation_rms.count == 17.0
            self.env.reset()

        def train(self, steps, *, callback, observer):
            self.env.step(None, {}, np.zeros((2, 14)))
            self.step = steps
            observer({"phase": "update"})

    modules = {
        "mlx": {},
        "mlx.core": {"eval": lambda *_: None,
                     "random": SimpleNamespace(seed=lambda _: None, key=lambda _: None)},
        "mlx.optimizers": {"Adam": lambda **_: None},
        "rlx.algorithms.ppo": {"PPO": FakePPO, "PPOConfig": SimpleNamespace},
        "rlx.buffers.rollout_buffer": {"RolloutBuffer": lambda *_, **__: None},
        "rlx.models.microduck": {"load_checkpoint": lambda _: loaded,
                                 "save_checkpoint": save_checkpoint,
                                 "create_actor_critic": lambda **_: pytest.fail("must load initial model")},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    result = studio.train(args)
    assert env.closed
    assert result["trained_steps"] == 2
    assert len(snapshots) == 3
    assert [snapshot["metadata"]["steps"] for snapshot in snapshots] == [0, 2, 2]
    for snapshot in snapshots:
        assert snapshot["metadata"]["initialization"] == expected_initialization
        assert snapshot["metadata"]["freeze_observation_normalization"] is freeze
        assert snapshot["metadata"]["teacher_assisted"] is True
        json.dumps(snapshot["metadata"], allow_nan=False)
        if freeze:
            np.testing.assert_array_equal(snapshot["mean"], loaded["mean"])
            np.testing.assert_array_equal(snapshot["variance"], loaded["variance"])
            assert snapshot["count"] == 17.0
            assert snapshot["epsilon"] == loaded["epsilon"]
            assert snapshot["clip"] == loaded["clip"]
    assert snapshots[-1]["return_count"] == 4.0
    if not freeze:
        assert snapshots[-1]["count"] == 21.0
    assert json.loads(sidecar.read_text())["metadata"]["initialization"] == expected_initialization
