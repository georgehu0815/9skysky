import ast
import importlib.util
from pathlib import Path
import time

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bootstrap_swing_e2e.py"
SPEC = importlib.util.spec_from_file_location("swing_e2e_bootstrap", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.mark.parametrize("arguments", [
    ["--iterations", "6"], ["--iterations", "-1"], ["--max-samples", "60001"],
    ["--train-seconds", "301"], ["--total-seconds", "601"], ["--train-seconds", "nan"],
])
def test_hard_limits(arguments):
    with pytest.raises(SystemExit):
        bootstrap.parse_args(arguments)


def test_teacher_label_bounded_and_symmetric():
    parameters = np.arange(13, dtype=float) - 3
    actions = bootstrap.teacher_action(parameters, 0.5, -1.0)
    assert actions.shape == (14,)
    assert np.isfinite(actions).all()
    assert np.max(np.abs(actions)) <= 1
    np.testing.assert_array_equal(actions[[0, 1, 7, 8, 9, 10]], np.zeros(6))
    np.testing.assert_array_equal(actions[2:5], -actions[11:14])


def test_only_label_capture_reads_privileged_state():
    tree = ast.parse(SCRIPT.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    readers = [node.name for node in functions if any(
        isinstance(child, ast.Attribute) and child.attr == "_swing_state"
        for child in ast.walk(node)
    )]
    assert readers == ["label_observation"]
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert not any(isinstance(node, ast.Name) and node.id in {"env", "parameters"}
                   for node in ast.walk(actor))


def test_frozen_normalizer_floors_handle_constant_observations():
    observations = np.zeros((4, 61), dtype=np.float32)
    mean, variance = bootstrap.observation_statistics(observations)
    assert mean.shape == variance.shape == (61,)
    assert (variance > 0).all()
    assert np.isfinite((observations - mean) / np.sqrt(variance)).all()


def test_strict_success_beats_larger_invalid_span():
    def report(passed, span, tension):
        return {"passed": passed, "episodes": [{"valid_geometry_fraction": 1.,
                "both_strings_tensioned_fraction": tension, "bidirectional_span_deg": span}]}
    assert bootstrap.selection_key(report(True, 150, 1)) > bootstrap.selection_key(report(False, 170, 0.99))


def test_dagger_labels_pre_action_states_but_executes_learner(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.index = 0
            self.applied = []
            self.closed = False

        def reset(self, seed):
            return np.zeros(61, np.float32), {}

        def _swing_state(self):
            return {"angle": self.index * 0.1, "angle_rate": 0.0}

        def step(self, action):
            self.applied.append(action.copy())
            self.index += 1
            return np.full(61, self.index, np.float32), 0., False, False, {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(bootstrap, "make_env", lambda *_: env)
    parameters = np.zeros(13)
    parameters[1] = 1.0
    parameters[8:13] = 1.0
    observations, labels, batches = bootstrap.collect(
        parameters, lambda observation: np.full(14, 0.9, np.float32),
        seed=1, remaining=3, deadline=time.monotonic() + 10, episodes=1,
    )
    np.testing.assert_array_equal(observations[:, 0], [0, 1, 2])
    np.testing.assert_allclose(labels[:, 2], np.tanh([0., 0.1, 0.2]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(env.applied), 0.9)
    assert batches[0]["samples"] == 3
    assert batches[0]["behavior"] == "learned_61d_onnx"
    assert env.closed


def test_strict_evaluation_does_not_query_teacher(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.index = 0

        def reset(self, seed):
            return np.zeros(61, np.float32), {}

        def recipe_metrics(self):
            return dict(swing_angle_deg=76. if self.index % 2 else -76.,
                        lateral_offset_m=0., alignment_penalty=0., valid_geometry=1.,
                        string_left_m=0.382, string_right_m=0.382,
                        string_left_tension_n=4., string_right_tension_n=4.)

        def step(self, action):
            assert action.shape == (14,)
            self.index += 1
            return np.zeros(61, np.float32), 0., False, self.index == 1200, {
                "recipe_metrics": self.recipe_metrics()
            }

        def close(self):
            pass

    def forbidden(*args):
        raise AssertionError("teacher queried during evaluation")

    monkeypatch.setattr(bootstrap, "make_env", FakeEnv)
    monkeypatch.setattr(bootstrap, "label_observation", forbidden)
    report, traces = bootstrap.evaluate_actor(lambda observation: np.zeros(14), seeds=(1701,))
    assert report["passed"]
    assert report["criteria"]["min_bidirectional_span_deg"] == 150
    assert report["criteria"]["required_steps"] == 1200
    assert len(traces) == 1200
