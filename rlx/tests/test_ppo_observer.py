"""Observational telemetry preserves real compiled MLX PPO training."""
from types import SimpleNamespace

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import pytest

import rlx.algorithms.ppo as ppo_module
from rlx.algorithms.ppo import PPO, PPOConfig
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils.distributions import Gaussian


class TinyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(2, 1)
        self.critic = nn.Linear(2, 1)
        self.log_std = mx.array([-0.5])

    def constrain_actor_log_std(self):
        self.log_std = mx.clip(self.log_std, -5.0, 2.0)

    def __call__(self, observation):
        return Gaussian(mx.tanh(self.actor(observation)), self.log_std), self.critic(observation)


class TinyEnvironment:
    """Deterministic two-lane dynamics with timeouts and true terminations."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-4.0, 4.0, (1,), dtype=np.float32)

    def __init__(self):
        self.resets = []
        self.transitions = []

    def reset(self, keys):
        self.resets.append(np.asarray(keys).copy())
        observation = mx.array([[0.1, 0.2], [-0.3, 0.4]])
        return observation, {"observation": observation, "tick": mx.array(0)}, {}

    def step(self, keys, state, action):
        self.transitions.append((np.asarray(keys).copy(), np.asarray(action).copy()))
        tick = state["tick"] + 1
        terminal = state["observation"] + mx.concatenate([action, -action], axis=-1) * 0.05
        truncated = mx.array([True, False]) & (tick % 2 == 0)
        terminated = mx.array([False, True]) & (tick % 3 == 0)
        done = truncated | terminated
        observation = mx.where(done[:, None], mx.zeros_like(terminal), terminal)
        reward = 1.0 - mx.sum(mx.square(terminal), axis=-1)
        info = {"terminal_observation": terminal, "_terminal_observation": done,
                "episode": {"tick": int(tick.item())}}
        return observation, {"observation": observation, "tick": tick}, reward, terminated, truncated, info


@pytest.fixture(autouse=True)
def restore_numpy_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def make_agent(**overrides):
    np.random.seed(1234)
    mx.random.seed(5678)
    settings = dict(num_envs=2, num_steps=3, num_minibatches=4, update_epochs=2)
    settings.update(overrides)
    config = PPOConfig(**settings)
    env = TinyEnvironment()
    buffer = RolloutBuffer(config.num_steps, env.observation_space, env.action_space,
                           num_envs=config.num_envs)
    return PPO(config, env, TinyActorCritic(), optim.Adam(learning_rate=1e-3),
               buffer, mx.random.key(9012))


def snapshot_tree(tree):
    return {name: np.asarray(value).copy() for name, value in tree_flatten(tree)}


def assert_trees_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def run_training(observed):
    agent = make_agent()
    events, callbacks, losses = [], [], []
    compiled_update_step = agent.update_step

    def record_loss(*args):
        result = compiled_update_step(*args)
        losses.append(result[0])
        return result

    agent.update_step = record_loss
    callback = lambda info, step: callbacks.append((info["episode"]["tick"], step))
    observer = events.append if observed else None
    assert agent.train(7, callback, observer=observer) is None
    assert agent.train(13, callback, observer=observer) is None
    assert agent.train(18, callback, observer=observer) is None
    return SimpleNamespace(
        parameters=snapshot_tree(agent.network.parameters()),
        optimizer=snapshot_tree(agent.optimizer.state),
        buffer=snapshot_tree({name: value for name, value in vars(agent.buffer).items()
                              if isinstance(value, mx.array)}),
        key=np.asarray(agent.key).copy(), step=agent.step,
        next_numpy_random=np.random.uniform(size=5),
        next_mlx_random=np.asarray(mx.random.uniform(shape=(5,))).copy(),
        resets=agent.env.resets, transitions=agent.env.transitions,
        callbacks=callbacks, events=events, losses=[float(loss.item()) for loss in losses],
    )


def test_observer_preserves_parameters_optimizer_rng_resets_and_callbacks(monkeypatch):
    evaluations = []
    evaluate = mx.eval

    def record_evaluation(*args):
        evaluations.append(len(args))
        return evaluate(*args)

    monkeypatch.setattr(mx, "eval", record_evaluation)
    plain = run_training(False)
    plain_evaluations = evaluations[:]
    evaluations.clear()
    observed = run_training(True)
    assert evaluations == plain_evaluations

    assert_trees_equal(plain.parameters, observed.parameters)
    assert_trees_equal(plain.optimizer, observed.optimizer)
    assert_trees_equal(plain.buffer, observed.buffer)
    for name in ("key", "next_numpy_random", "next_mlx_random", "resets", "losses"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(observed, name))
    assert len(plain.transitions) == len(observed.transitions) == 9
    for first, second in zip(plain.transitions, observed.transitions):
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
    assert plain.callbacks == observed.callbacks == [
        (1, 0), (2, 2), (3, 4), (4, 6), (5, 8), (6, 10), (1, 12), (2, 14), (3, 16),
    ]
    assert plain.step == observed.step == 18
    assert len(observed.resets) == 3  # Even a reached target retains train's reset.
    assert plain.events == []
    assert [event["phase"] for event in observed.events] == ["collection", "update"] * 3
    for index, event in enumerate(observed.events):
        assert event["steps"] == 6 and type(event["steps"]) is int
        assert type(event["seconds"]) is float
        assert np.isfinite(event["seconds"]) and event["seconds"] >= 0
        if event["phase"] == "collection":
            assert set(event) == {"phase", "steps", "seconds"}
        else:
            assert set(event) == {"phase", "steps", "seconds", "optimizer_steps", "mean_loss"}
            # Six one-sample minibatches per epoch, not the requested four.
            assert event["optimizer_steps"] == 12 and type(event["optimizer_steps"]) is int
            assert type(event["mean_loss"]) is float and np.isfinite(event["mean_loss"])
            actual = observed.losses[(index // 2) * 12:(index // 2 + 1) * 12]
            assert event["mean_loss"] == sum(actual) / len(actual)


def test_phase_clock_includes_reset_and_gae_but_excludes_observers(monkeypatch):
    agent = make_agent()
    clock = SimpleNamespace(now=0.0)
    events = []
    monkeypatch.setattr(ppo_module, "perf_counter", lambda: clock.now)

    def timed(function, duration):
        def call(*args, **kwargs):
            clock.now += duration
            return function(*args, **kwargs)
        return call

    monkeypatch.setattr(agent.buffer, "reset", timed(agent.buffer.reset, 2.0))
    monkeypatch.setattr(agent.env, "step", timed(agent.env.step, 3.0))
    monkeypatch.setattr(ppo_module, "compute_generalized_advantage_estimate",
                        timed(ppo_module.compute_generalized_advantage_estimate, 5.0))
    network_call = TinyActorCritic.__call__
    update = agent.update
    in_update = False

    def timed_network(network, observation):
        if agent.buffer.full and not in_update:
            clock.now += 4.0  # Last-value construction before GAE.
        return network_call(network, observation)

    def timed_update(*args, **kwargs):
        nonlocal in_update
        in_update = True
        clock.now += 7.0
        try:
            return update(*args, **kwargs)
        finally:
            in_update = False

    monkeypatch.setattr(TinyActorCritic, "__call__", timed_network)
    monkeypatch.setattr(agent, "update", timed_update)

    def observe(event):
        events.append(event)
        clock.now += 1000.0

    agent.train(7, observer=observe)
    assert [event["seconds"] for event in events] == [11.0, 16.0, 11.0, 16.0]


def test_absent_observer_keeps_update_call_and_return_conventions(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(ppo_module, "perf_counter", lambda: pytest.fail("unexpected telemetry clock"))
    update = agent.update
    calls = []

    def legacy_update(advantages, returns):
        calls.append(True)
        assert update(advantages, returns) is None

    monkeypatch.setattr(agent, "update", legacy_update)
    agent.train(1)
    assert calls == [True]
    with pytest.raises(TypeError):
        agent.train(1, None, lambda event: None)


@pytest.mark.parametrize("phase", ["collection", "update"])
def test_observer_exceptions_propagate_and_stop_training(phase):
    agent = make_agent()
    events = []
    failure = RuntimeError("observer failed")

    def observe(event):
        events.append(event)
        if event["phase"] == phase:
            raise failure

    with pytest.raises(RuntimeError) as raised:
        agent.train(100, observer=observe)
    assert raised.value is failure
    assert agent.step == 6
    assert [event["phase"] for event in events] == (["collection"] if phase == "collection"
                                                 else ["collection", "update"])
    assert int(agent.optimizer.step.item()) == (0 if phase == "collection" else 12)


@pytest.mark.parametrize("failure_kind", ["exception", "loss", "gradients", "model parameters"])
def test_partial_update_never_emits_success(failure_kind):
    agent = make_agent()
    events = []
    update_step = agent.update_step
    calls = 0

    def fail_third_minibatch(*args):
        nonlocal calls
        calls += 1
        if calls == 3 and failure_kind == "exception":
            raise RuntimeError("partial update")
        result = list(update_step(*args))
        if calls == 3:
            result[{"loss": 1, "gradients": 2, "model parameters": 3}[failure_kind]] = mx.array(False)
        return tuple(result)

    agent.update_step = fail_third_minibatch
    with pytest.raises(RuntimeError, match="partial update|non-finite"):
        agent.train(100, observer=events.append)
    assert calls == 3 and agent.step == 6
    assert [event["phase"] for event in events] == ["collection"]
    assert int(agent.optimizer.step.item()) >= 2


def test_failed_collection_emits_no_event():
    agent = make_agent()
    events = []

    def fail_step(*args):
        raise RuntimeError("collection failed")

    agent.env.step = fail_step
    with pytest.raises(RuntimeError, match="collection failed"):
        agent.train(1, observer=events.append)
    assert events == [] and agent.step == 0


def test_empty_update_does_not_report_an_invented_mean():
    agent = make_agent(update_epochs=0)
    events = []
    with pytest.raises(ValueError, match="at least one optimizer step"):
        agent.train(1, observer=events.append)
    assert [event["phase"] for event in events] == ["collection"]
    agent = make_agent(update_epochs=0)
    assert agent.train(1) is None


def test_extreme_finite_policy_ratio_keeps_loss_and_gradients_finite():
    agent = make_agent(num_envs=2, num_steps=2, num_minibatches=1, update_epochs=1)
    observations = mx.array(np.full((4, 2), 0.25, dtype=np.float32))
    actions = mx.array(np.full((4, 1), 12.0, dtype=np.float32))
    distribution, _ = agent.network(observations)
    old_log_probabilities = distribution.log_prob(actions) - 100.0

    loss, loss_finite, gradients_finite, parameters_finite = agent.update_step(
        observations,
        actions,
        old_log_probabilities,
        mx.zeros((4,), dtype=mx.float32),
        mx.array([1.0, -1.0, 0.5, -0.5], dtype=mx.float32),
        mx.array([500.0, -500.0, 250.0, -250.0], dtype=mx.float32),
    )
    mx.eval(
        loss,
        loss_finite,
        gradients_finite,
        parameters_finite,
        agent.network.state,
        agent.optimizer.state,
    )

    assert np.isfinite(float(loss.item()))
    assert bool(np.asarray(loss_finite))
    assert bool(np.asarray(gradients_finite))
    assert bool(np.asarray(parameters_finite))


def test_extreme_finite_critic_error_keeps_loss_and_gradients_finite():
    agent = make_agent(num_envs=2, num_steps=2, num_minibatches=1, update_epochs=1)
    observations = mx.array(np.full((4, 2), 1e18, dtype=np.float32))
    actions = mx.zeros((4, 1), dtype=mx.float32)
    distribution, old_values = agent.network(observations)
    old_log_probabilities = distribution.log_prob(actions)

    loss, loss_finite, gradients_finite, parameters_finite = agent.update_step(
        observations,
        actions,
        old_log_probabilities,
        old_values.squeeze(-1),
        mx.array([1.0, -1.0, 0.5, -0.5], dtype=mx.float32),
        mx.array([500.0, -500.0, 250.0, -250.0], dtype=mx.float32),
    )
    mx.eval(
        loss,
        loss_finite,
        gradients_finite,
        parameters_finite,
        agent.network.state,
        agent.optimizer.state,
    )

    assert np.isfinite(float(loss.item()))
    assert bool(np.asarray(loss_finite))
    assert bool(np.asarray(gradients_finite))
    assert bool(np.asarray(parameters_finite))
