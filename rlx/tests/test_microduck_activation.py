import numpy as np
import pytest


def test_elu_preserves_finite_large_positive_gradients():
    import mlx.core as mx
    from rlx.models.microduck import stable_elu

    values = mx.array([-1000.0, -5.0, -1.0, 0.0, 1.0, 90.0, 1000.0])
    derivative = mx.grad(lambda inputs: stable_elu(inputs).sum())
    for operation in (derivative, mx.compile(derivative)):
        actual = np.asarray(operation(values))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, [0, np.exp(-5), np.exp(-1), 1, 1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(np.asarray(stable_elu(values)), [-1, np.expm1(-5), np.expm1(-1), 0, 1, 90, 1000], atol=1e-6)


@pytest.mark.parametrize("seed", [7, 17])
def test_stable_elu_matches_legacy_forward_on_normal_values(seed):
    import mlx.core as mx
    import mlx.nn as nn
    from rlx.models.microduck import stable_elu

    inputs = mx.array(np.random.default_rng(seed).normal(size=(64, 128)).astype(np.float32))
    np.testing.assert_allclose(np.asarray(stable_elu(inputs)), np.asarray(nn.elu(inputs)), atol=2e-7)
