<div align="center">

# 🦾 RLX

**Reinforcement learning that runs end-to-end on [MLX](https://github.com/ml-explore/mlx), Apple's array framework.**

Single-file, CleanRL-style algorithms and vectorized environments that live entirely on device: over a million environment steps per second on Apple silicon.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/backend-MLX-black.svg)](https://github.com/ml-explore/mlx)
[![Status: Production/Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://github.com/noahfarr/rlx)

</div>

---

## Why RLX

- **On-device, end-to-end.** Environments, buffers, and the learner all run on the GPU, with no host round-trips between steps.
- **Fused updates.** Every update step is wrapped in `mx.compile`, fusing the training graph for maximum throughput.
- **Fast.** The bundled MLX-native environments reach well over a million environment steps per second on Apple silicon.
- **Readable.** Each algorithm is a single, self-contained, CleanRL-style file you can read top to bottom.
- **Portable.** The correct MLX backend (Metal on Apple silicon, CUDA on Linux) is selected automatically.

## Algorithms

| Algorithm | File | Action space |
| --- | --- | --- |
| DQN | `rlx/algorithms/dqn.py` | discrete |
| REINFORCE | `rlx/algorithms/reinforce.py` | discrete |
| A2C | `rlx/algorithms/a2c.py` | discrete |
| PPO | `rlx/algorithms/ppo.py` | discrete & continuous |
| SAC | `rlx/algorithms/sac.py` | continuous |
| TD3 | `rlx/algorithms/td3.py` | continuous |

### PPO time limits and autoreset observations

PPO bootstraps a time-limit truncation from the final observation's critic value, while ending the advantage trace at every reset. True termination takes precedence when both flags are set. An autoresetting environment must return batched `info["terminal_observation"]` in the same observation space as its policy input, plus the boolean validity mask `info["_terminal_observation"]`. PPO rejects a timeout without valid final-state metadata instead of using the reset state. The native Environment and MicroDuck adapter provide this metadata; other adapters must establish their final-state semantics before using time limits.

The MicroDuck adapter normalizes final observations with the current observation statistics without updating those statistics a second time. [Timeout regressions](tests/test_ppo_timeouts.py) cover nonzero terminal values, reset separation, true termination, and mixed vector lanes.

### Optional PPO phase telemetry

`PPO.train(num_steps, callback=None, *, observer=None)` accepts an optional synchronous observer without changing the episode callback's `callback(info, step)` convention. The observer receives a fresh dictionary after each completed collection and update:

```python
# Given an initialized PPO instance and an existing episode callback:
phase_events = []
algorithm.train(total_timesteps, callback, observer=phase_events.append)
# Collection: {"phase": "collection", "steps": int, "seconds": float}
# Update: {"phase": "update", "steps": int, "seconds": float,
#          "optimizer_steps": int, "mean_loss": float}
```

Counts and times are **per event**, not cumulative. `steps` is the number of environment transitions in that collection (also attached to its update), including any full-rollout overshoot of the requested target. `optimizer_steps` counts actual completed minibatch updates; `mean_loss` is the arithmetic mean of their already-materialized weighted total objectives (policy loss minus weighted entropy plus weighted value loss), not a separate policy-only loss. `update(advantages, returns, *, collect_metrics=False)` retains its default `None` return; opting in returns `optimizer_steps` and `mean_loss`. Metrics require at least one optimizer step and raise `ValueError` for an empty update rather than inventing a loss.

Collection wall time starts before buffer reset and ends after the rollout loop, including existing episode callbacks. Update wall time starts after the collection observer returns, before last-value and GAE construction, and ends after the update finishes. It includes GAE and deferred critic work; neither interval isolates CPU/GPU execution or transfer costs. Initial environment reset and observer notification time are excluded. Telemetry adds no MLX evaluation barriers and reads losses only after the existing update evaluation.

Observer exceptions propagate, and failed or partial phases emit no completion event. Observers must not mutate training state or consume training RNGs. With `observer=None`, no phase timing or metric gathering occurs, and reset, sampling, normalization, optimizer, and callback behavior are unchanged. [Real-MLX observer tests](tests/test_ppo_observer.py) compare complete parameters, optimizer state, RNG continuation, resets, and callback traces with telemetry on and off.

## Environments

Every environment is MLX-native and vectorized, so resets, steps, and the learner all run on device. Classic-control
environments import from `rlx.environments`; the MinAtar suite lives under `rlx.environments.minatar`. The MinAtar ports
are branchless and validated against the reference [`minatar`](https://github.com/kenjyoung/MinAtar) package.

| Environment | Import | Action space |
| --- | --- | --- |
| CartPole | `rlx.environments.CartPole` | discrete |
| MountainCar | `rlx.environments.MountainCar` | discrete |
| Acrobot | `rlx.environments.Acrobot` | discrete |
| Pendulum | `rlx.environments.Pendulum` | continuous |
| MinAtar Breakout | `rlx.environments.minatar.Breakout` | discrete |
| MinAtar Freeway | `rlx.environments.minatar.Freeway` | discrete |
| MinAtar SpaceInvaders | `rlx.environments.minatar.SpaceInvaders` | discrete |
| MinAtar Asterix | `rlx.environments.minatar.Asterix` | discrete |
| MinAtar Seaquest | `rlx.environments.minatar.Seaquest` | discrete |

Need something outside these suites? The `EnvPool` adapter wraps a pre-vectorized
[EnvPool](https://github.com/sail-sg/envpool) environment behind the same `Environment` interface.

## Quickstart

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv), and macOS on Apple silicon (Metal) or Linux (CUDA).

```bash
git clone https://github.com/noahfarr/rlx.git
cd rlx
uv sync
```

Each algorithm ships with a runnable example wired to a [tyro](https://github.com/brentyi/tyro) CLI:

```bash
uv run examples/ppo_cartpole.py
uv run examples/sac_pendulum.py

# Scale the vectorized rollout right from the CLI
uv run examples/ppo_cartpole.py --ppo.num-envs 8192 --ppo.num-steps 16
```

Experiment-level flags (`--seed`, `--total-timesteps`, `--learning-rate`, `--track`) live on the example.
Algorithm hyperparameters are nested under the algorithm name, e.g. `--ppo.gamma` or `--sac.tau`. Append `--help` to
any example to see every option.

## Train and render the MicroDuck dance

The MicroDuck examples use the sibling `microduck_local` checkout for MuJoCo
simulation and rendering. Install it into the RLX environment first. Its
dependencies provide ONNX for policy export and ONNX Runtime for policy
evaluation and rendering, so RLX does not duplicate those dependencies.

```bash
uv sync
uv pip install -e ../microduck_local
```

Run the dance workflow from the RLX repository root:

```bash
# Train and save an RLX checkpoint with its JSON metadata sidecar
uv run examples/ppo_microduck_dance.py train \
  --checkpoint runs/dance/dance.safetensors

# Evaluate the deterministic checkpoint in the MuJoCo environment
uv run examples/ppo_microduck_dance.py eval \
  --checkpoint runs/dance/dance.safetensors

# Export a reusable ONNX policy with observation normalization baked in
uv run examples/ppo_microduck_dance.py export \
  --checkpoint runs/dance/dance.safetensors \
  --output runs/dance/dance.onnx

# Render directly from the checkpoint
uv run examples/ppo_microduck_dance.py render \
  --checkpoint runs/dance/dance.safetensors \
  --output runs/dance/render
```

The render command exports a temporary ONNX policy automatically, keeps it
available until `microduck_local.render_rollout` exits, and then removes it.
The output directory receives an MP4 video and a captioned PNG contact sheet
for each episode, such as `ep0.mp4` and `ep0_sheet.png`. Use the explicit
export command when you need a persistent ONNX policy for deployment or
separate evaluation.

## Evaluate the native Studio Swing recipe

`examples/ppo_microduck_studio.py` separates execution checks from Swing skill acceptance. `--evaluation-mode pipeline` checks finite execution and any explicit return/episode limits; its `skill_status` is `not_assessed`, even when `passed` is true. The default `skill` mode requires every observed Swing episode to complete 1,200 control steps (24 seconds) without termination, from zero initial angle and rate. Partial episodes cannot pass.

The default target is a **150° symmetric bidirectional span**: at least +75° and −75° in the same episode, not pooled across resets or vector lanes. `--swing-min-span-deg` makes that target explicit and editable. Every 50 Hz sample must have valid geometry (absolute lateral offset ≤0.020 m, alignment penalty ≤0.050, both string lengths within 0.370–0.394 m, and the environment's validity flag) and positive spring tension in both strings. Reported tension is a sampled elastic estimate, excluding safety-limit constraint forces; these criteria do not establish hardware safety.

```bash
# Full, deterministic, still-start assessment; failed skill exits with status 2.
uv run examples/ppo_microduck_studio.py eval --recipe swing \
  --checkpoint runs/studio/swing/my-run/swing.safetensors \
  --evaluation-mode skill --swing-min-span-deg 150 \
  --num-envs 1 --backend dummy --eval-steps 1200 --max-episode-s 24 \
  --no-domain-rand --no-obs-noise --no-action-delay --no-random-yaw

# Short execution smoke: this cannot report learned Swing skill.
uv run examples/ppo_microduck_studio.py eval --recipe swing \
  --checkpoint runs/studio/swing/my-run/swing.safetensors \
  --evaluation-mode pipeline --num-envs 1 --backend dummy --eval-steps 4

# Export needs a checkpoint, not environment arguments or an existing ONNX file.
uv run examples/ppo_microduck_studio.py export --recipe swing \
  --checkpoint runs/studio/swing/my-run/swing.safetensors \
  --output runs/studio/swing/my-run/swing.onnx
```

Evaluation defaults to the `dummy` backend, which supplies physical metrics at every control step. Swing evaluation rejects `fork`, whose transport provides only terminal metrics; `subproc` also preserves per-step metrics. Training retains its `fork` default. Swing fixes yaw to the seat axis regardless of the common yaw flag, and reports that resolved setting as `false`.

Reports bind the evaluation settings and per-episode measurements to SHA-256 hashes of the policy bytes and, for checkpoint inference, its normalization metadata sidecar. They distinguish `pipeline_passed`, `skill_status`, and the requested scope's `passed`. Rendering accepts either `--checkpoint` or `--policy`, reports the source hashes and reset count, and does not itself certify skill. The actor and critic still share the 61-element observation; pipeline correctness is not evidence that training has learned a substantial swing.

## Train MicroDuck with mjlab

RLX also packages the complete GPU-oriented MicroDuck mjlab task suite. Install
the optional robotics stack, inspect the registered tasks, and run the mandatory
64-environment smoke test before a long training job:

```bash
uv python pin 3.12
uv sync --extra mjlab-microduck
uv run rlx-mjlab-list-envs
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

See the [mjlab MicroDuck training and evaluation guide](docs/mjlab-microduck-guide.md)
for local and Hugging Face training, resume, playback, deterministic evaluation,
ONNX export, deployment rehearsal, task contracts, and troubleshooting.

## Use it as a library

```python
from rlx.algorithms import PPO, PPOConfig
from rlx.environments import CartPole
from rlx.environments.minatar import Breakout
```

The top-level `rlx` package also re-exports the core algorithms and their configs (`from rlx import PPO, PPOConfig`).

## Project layout

| Path | What's inside |
| --- | --- |
| `rlx/algorithms/` | Algorithm implementations (DQN, REINFORCE, A2C, PPO, SAC, TD3) and their configs |
| `rlx/environments/` | MLX-native environments (`classic_control/` and `minatar/`) plus the `Environment` interface and `EnvPool` adapter |
| `rlx/buffers/` | `RolloutBuffer` (on-policy) and `ReplayBuffer` (off-policy) |
| `rlx/utils/` | Action distributions, the `Logger`, and shared helpers (GAE, returns, `soft_update`) |
| `examples/` | One runnable training script per algorithm |

## Contributing

Contributions are welcome! Fork the repo, create a branch, commit your changes, and open a pull request.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Acknowledgments

Thanks to the [MLX](https://github.com/ml-explore/mlx) team for the framework and to
[CleanRL](https://github.com/vwxyzjn/cleanrl) for the reference implementations this project draws on.
