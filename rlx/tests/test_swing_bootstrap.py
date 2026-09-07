import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "bootstrap_microduck_swing.py"


def load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_microduck_swing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_action_is_bounded_and_sagittally_symmetric():
    module = load_bootstrap()
    action = module.teacher_action(angle=0.2, rate=-0.4)
    assert action.shape == (14,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0
    np.testing.assert_array_equal(action[[0, 1, 7, 8, 9, 10]], np.zeros(6))
    np.testing.assert_allclose(action[2:5], -action[11:14])
