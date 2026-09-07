"""Focused tests for the real Duck Lab Hello World architecture demo."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.hello_ducks import build_parser, run_demo, zero_policy
from microduck_local.walk_env import clear_shared_models


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--ducks", "1"], "between 2 and 20"),
        (["--ducks", "21"], "between 2 and 20"),
        (["--steps", "0"], "positive"),
    ],
)
def test_argument_validation(argv, message, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_zero_policy_checks_contract():
    action = zero_policy(np.zeros(C.OBS_DIM, dtype=np.float32))
    assert action.shape == (C.NUM_JOINTS,)
    assert action.dtype == np.float32
    with pytest.raises(ValueError, match="observation shape"):
        zero_policy(np.zeros(C.OBS_DIM - 1, dtype=np.float32))


def test_real_two_duck_invariant_demo():
    clear_shared_models()
    try:
        result = run_demo(duck_count=2, steps=2)
        assert len({duck.env_id for duck in result.ducks}) == 2
        assert len({duck.data_id for duck in result.ducks}) == 2
        assert len({duck.model_id for duck in result.ducks}) == 1
        assert len({duck.seed for duck in result.ducks}) == 2
        assert result.isolated_qpos_unchanged
        assert result.observation_shape == (61,)
        assert result.action_shape == (14,)
        assert all(duck.initial_trunk != duck.final_trunk for duck in result.ducks)
    finally:
        clear_shared_models()
