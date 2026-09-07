"""Nonzero terminal values distinguish time limits from autoreset observations."""
from types import SimpleNamespace

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from rlx.algorithms.ppo import PPO, PPOConfig
from rlx.environments.microduck import MicroDuckVecEnv
from rlx.environments.environment import Environment
from rlx.utils.utils import compute_generalized_advantage_estimate as gae


def test_mixed_transition_values_stop_traces_but_bootstrap_timeouts():
    rewards = mx.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=mx.float32)
    values = mx.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]], dtype=mx.float32)
    terminated = mx.array([[False] * 4, [False, True, False, False], [False, False, True, True]])
    truncated = mx.array([[False] * 4, [True, False, False, False], [False, True, False, True]])
    terminals = mx.array([[0, 0, 0, 0], [20, 0, 0, 0], [0, 30, 0, 40]], dtype=mx.float32)
    actual = gae(rewards, values, terminated, mx.array([14., 15., 16., 17.]), mx.zeros(4),
                 gamma=.9, gae_lambda=.8, truncations=truncated, truncation_values=terminals)
    np.testing.assert_allclose(np.asarray(actual),
        [[16.64, 4.58, 12.7376, 14.2856], [17, -1, 9.08, 9.98], [11.6, 26, -1, -1]], atol=2e-6)


def test_missing_timeout_values_rejects_and_true_terminal_ignores_nonfinite_tail():
    with pytest.raises(ValueError, match="truncation_values"):
        gae(mx.ones((1, 1)), mx.zeros((1, 1)), mx.array([[False]]), mx.array([9.]), mx.zeros(1),
            truncations=mx.array([[True]]))
    result = gae(mx.ones((1, 1)), mx.zeros((1, 1)), mx.array([[True]]), mx.array([float('nan')]), mx.zeros(1),
                 truncations=mx.array([[True]]))
    np.testing.assert_array_equal(np.asarray(result), [[1.]])


def critic_only_ppo():
    agent = object.__new__(PPO)
    agent.config = PPOConfig(num_envs=3)
    agent.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (61,), dtype=np.float32))
    agent.network = lambda x: (None, x[:, :1])
    return agent


def test_collector_uses_final_value_and_true_terminal_wins():
    agent = critic_only_ppo()
    info = {"terminal_observation": mx.full((3, 61), 23.),
            "_terminal_observation": mx.array([True, False, False])}
    result = agent._truncation_values(mx.array([False, True, False]), mx.array([True, True, False]), info)
    np.testing.assert_array_equal(np.asarray(result), [23., 0., 0.])
    agent.network = lambda x: pytest.fail("must not evaluate critic without a timeout")
    np.testing.assert_array_equal(np.asarray(agent._truncation_values(mx.array([True] * 3), mx.array([True] * 3), {})), [0.] * 3)


@pytest.mark.parametrize("info", [{}, {"terminal_observation": np.ones((3, 61)), "_terminal_observation": [False] * 3},
    {"terminal_observation": np.ones((3, 60)), "_terminal_observation": [True] * 3},
    {"terminal_observation": np.ones((3, 61)), "_terminal_observation": [1] * 3}])
def test_collector_rejects_missing_or_malformed_timeout_metadata(info):
    with pytest.raises(ValueError):
        critic_only_ppo()._truncation_values(mx.array([False] * 3), mx.array([True, False, False]), info)


class TimeoutVec:
    num_envs = 2
    observation_space = gym.spaces.Box(-np.inf, np.inf, (61,), dtype=np.float32)
    action_space = gym.spaces.Box(-4., 4., (14,), dtype=np.float32)

    def reset(self):
        return np.zeros((2, 61), dtype=np.float32)

    def step(self, actions):
        self.infos = [{"TimeLimit.truncated": True, "terminal_observation": np.full(61, 8., dtype=np.float32)}, {}]
        return np.ones((2, 61), dtype=np.float32), np.ones(2), np.array([True, False]), self.infos

    def close(self):
        pass


@pytest.mark.parametrize("normalize", [False, True])
def test_terminal_normalization_counts_only_returned_batch(normalize):
    inner = TimeoutVec()
    env = MicroDuckVecEnv(inner, normalize_observations=normalize)
    env.reset(None)
    count = env.observation_rms.count
    *_, info = env.step(None, {}, mx.zeros((2, 14)))
    assert env.observation_rms.count == pytest.approx(count + (2 if normalize else 0))
    expected = np.full(61, 8., dtype=np.float32)
    if normalize:
        expected = np.clip((expected - env.observation_rms.mean) / np.sqrt(env.observation_rms.var + env.epsilon), -env.clip, env.clip).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(info['terminal_observation'])[0], expected)
    np.testing.assert_array_equal(np.asarray(info['_terminal_observation']), [True, False])
    np.testing.assert_array_equal(inner.infos[0]['terminal_observation'], np.full(61, 8.))


def test_native_environment_preserves_pre_reset_observation():
    class Native(Environment):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, key):
            return mx.array([-100.]), {}, {}

        def step_env(self, key, state, action):
            return mx.array([7.]), {}, mx.array(1.), mx.array(False), mx.array(True), {}

    observation, _, _, _, _, info = Native().step(mx.random.key(1), {}, mx.array(0))
    np.testing.assert_array_equal(np.asarray(observation), [-100.])
    np.testing.assert_array_equal(np.asarray(info['terminal_observation']), [7.])
    assert bool(info['_terminal_observation'])
