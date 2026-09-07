---
title: RLX mjlab MicroDuck training and evaluation
description: Install, train, resume, evaluate, export, and rehearse MicroDuck policies with the mjlab integration distributed in RLX
ms.date: 2026-09-05
ms.topic: tutorial
keywords:
  - MicroDuck
  - mjlab
  - reinforcement learning
  - PPO
  - sim2real
estimated_reading_time: 18
---

## Scope

RLX now distributes the complete MicroDuck mjlab task suite under
`rlx.mjlab_microduck`. The integration includes the task configurations, custom
MDP terms, BAM actuator model, domain randomization, curricula, symmetry,
backlash variants, MJCF models, meshes, local and Hugging Face training, policy
playback, deterministic evaluations, ONNX export, and deployment rehearsal.

The production training path deliberately uses mjlab's `rsl_rl` runner. This
preserves every source training feature, including manager-based environments,
MuJoCo Warp vectorization, privileged critic observations, empirical
normalization, symmetry loss, checkpoint resume, curriculum managers, and
W&B checkpoint handling. RLX owns packaging, discovery, command entry points,
and the colocated MicroDuck implementation. The older
`rlx.environments.microduck` adapter remains a separate CPU-MuJoCo and MLX PPO
prototype path.

> [!IMPORTANT]
> MicroDuck mjlab runs require Python 3.12. BAM currently excludes Python 3.13,
> and the imported task suite is validated against mjlab 1.3.0, Warp 1.12.0,
> and Torch 2.9.1.

## What was imported

The integration lives in these areas:

* `rlx/mjlab_microduck/actuator` contains the BAM actuator with friction and
  backlash-aware encoder behavior
* `rlx/mjlab_microduck/robot` contains robot constants, MJCF scenes, models,
  meshes, roller, stilt, swing, ball, and test-bench assets
* `rlx/mjlab_microduck/tasks` contains all task registrations, environment
  configurations, MDP functions, terrain definitions, curricula, symmetry,
  and backlash wrappers
* `rlx/mjlab_microduck/tools` contains export, inference, evaluation,
  checkpoint surgery, visualization, and Hugging Face upload utilities
* `tests/mjlab_microduck` contains the source configuration and MDP regression
  suite

The `mjlab.tasks` entry point loads `rlx.mjlab_microduck.tasks`, so stock mjlab
registry APIs discover the imported tasks after installation.

## Install the mjlab integration

Run commands from the RLX repository root. Create or select a Python 3.12
environment, then sync the optional dependency group:

```bash
uv python pin 3.12
uv sync --extra mjlab-microduck
```

The base RLX installation does not pull the CUDA and robotics stack. The extra
keeps mjlab optional for users who only need the small RLX examples.

> [!WARNING]
> On managed machines where Microsoft Defender blocks
> `files.pythonhosted.org`, `uv sync` and `pip` cannot download PyPI wheels even
> when `pypi.org` resolves. Request an enterprise allow indicator for the
> blocked host. Do not bypass the managed network policy.

Linux resolves Torch 2.9.1 from the CUDA 12.9 index so Torch and MLX share a compatible CUDA runtime. This also avoids the CPU-only PyPI wheel on `aarch64`. macOS can run configuration tests
and CPU-capable utilities, but high-throughput MuJoCo Warp training is intended
for a CUDA machine.

## Verify task discovery

List the registry after installation:

```bash
uv run rlx-mjlab-list-envs
```

The catalog includes these task families and their registered backlash twins
where applicable:

* Flat and rough velocity walking
* Flat and rough VelStand walking plus recovery
* Flat and rough stand-up
* Flat and rough commanded sit-to-stand
* Flat and rough ground pick
* Forward running and running with a flight reward
* Ball kick
* Stilt locomotion
* Roller velocity, swizzle, crouch, slope, stand-up, and spin
* Forward roulade
* Self-pumped swing
* XL330 test-bench tasks registered by the imported suite

Task names are case-sensitive. Use the registry output rather than guessing an
identifier.

## Run the mandatory smoke test

Every new configuration or training change starts with 64 environments and
five iterations:

```bash
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

A successful smoke test proves that the model compiles, vector environments
step, all configured reward and observation terms execute, and the runner can
write a checkpoint. It does not prove policy quality or sim2real transfer.

Check the run for these invariants:

* Actor observations have 61 values: 48 proprioceptive values followed by the
  13-value command block
* Actions have 14 values in the deployed servo order
* No observation, reward, return, or action contains NaN or infinity
* Every self-negating penalty contributes a value less than or equal to zero
* Episode length and the main task reward move in the expected direction
* Curriculum transitions occur at the configured environment-step boundary

## Train a policy

A local walking run uses the same overrides accepted by mjlab:

```bash
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 4000
```

Logs and checkpoints are written below
`logs/rsl_rl/<experiment_name>/`. The task configuration controls observation
noise, action delay, domain randomization, command sampling, reward schedules,
and curriculum stages. Override manager configuration through mjlab's CLI
instead of editing copied configuration objects after environment creation.
Managers deep-copy term configurations during initialization.

### Resume training

Resume from a local checkpoint with the standard runner options:

```bash
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --agent.resume true \
  --agent.load-checkpoint /absolute/path/to/model_1000.pt \
  --agent.max_iterations 4000
```

Use an absolute checkpoint path when runs from several experiments share the
same working directory. A resume restores the policy, optimizer, and runner
iteration state. Confirm the active curriculum stage and observation
normalizer after loading.

### Train on Hugging Face Jobs

The training wrapper preserves the source remote-submission workflow:

```bash
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 4000 \
  --hf-jobs
```

Authenticate with Hugging Face and W&B before submission. Add `--namespace`
for unattended use, `--detach` to return after submission, or `--dry-run` to
inspect the job specification. The job snapshots tracked and untracked,
non-ignored repository files, starts a checkpoint uploader, trains through the
same RLX command, and makes a best-effort final ONNX export.

## Evaluate checkpoints

### Interactive playback

Play a local or W&B checkpoint through mjlab's viewer:

```bash
uv run rlx-mjlab-play Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path entity/project/run_id
```

Use the task's play configuration. It reduces environment count and disables
training-only behavior while preserving the observation and action contract.

### Deterministic running battery

Measure a running checkpoint at an exact speed across many environments:

```bash
uv run rlx-mjlab-eval-running \
  --checkpoint-file /absolute/path/to/model_1000.pt \
  --task-id Mjlab-Running-Flat-MicroDuck \
  --speed 0.8 \
  --num-envs 512 \
  --duration-s 8.0 \
  --output-file runs/running-eval.json
```

The report includes survival, commanded and displacement speed, lateral drift,
yaw rate, heading error, flight fraction, flight events, maximum tilt, and the
actor observation dimension.

### Swing evaluation and ONNX parity

Run the swing-specific checkpoint battery:

```bash
uv run rlx-mjlab-eval-swing \
  --checkpoint-file /absolute/path/to/model_1000.pt \
  --output runs/swing-eval.json
```

After export, compare Torch and ONNX policy outputs in the same swing state:

```bash
uv run rlx-mjlab-verify-swing-onnx \
  --checkpoint-file /absolute/path/to/model_1000.pt \
  --onnx-file runs/swing.onnx
```

Run both visual playback and quantitative batteries. Reward totals alone can
hide a collapsed pose, spawn-assisted behavior, or cyclic behavior that never
holds the goal.

## Export a deployment policy

Export through the supplied command, not a generic checkpoint converter:

```bash
uv run rlx-mjlab-export Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/to/model_4000.pt \
  --num-envs 1 \
  --onnx-file runs/walking.onnx
```

The exporter performs three deployment-critical operations:

1. It exports the runner's actor with empirical observation normalization in
   the graph.
2. It inserts the action clip used by `RslRlVecEnvWrapper` into the ONNX graph.
3. It attaches mjlab environment metadata to the model.

Never deploy a raw checkpoint or an ONNX graph without the normalizer. In-sim
play applies normalization and can conceal that export mistake.

Record a checkpoint video during export when a visual artifact is required:

```bash
uv run rlx-mjlab-export Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/to/model_4000.pt \
  --onnx-file runs/walking.onnx \
  --video \
  --video-length 500
```

## Rehearse ONNX in CPU MuJoCo

Run the deployment-style ONNX loop against the packaged MJCF assets:

```bash
uv run rlx-mjlab-infer \
  --walking runs/walking.onnx \
  --new-cmd-obs
```

The inference utility supports walking, standing, sit-to-stand, ground pick,
roller slope, left and right kick, and roulade policy switching. It computes
actuated joint indices from MuJoCo transmissions, so passive wheel and backlash
joints do not corrupt the 14-servo ordering.

## Run regression tests

Start with the fast package and existing RLX tests:

```bash
uv run --extra mjlab-microduck --with pytest pytest \
  tests/test_import_order.py \
  tests/test_microduck.py \
  tests/mjlab_microduck
```

Run GPU smoke tests separately because configuration tests passing on CPU does
not prove MuJoCo Warp execution:

```bash
uv run rlx-mjlab-train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

Before a long run, verify at least one representative task from every changed
family. Backlash changes require a base-versus-backlash pair using the same
robot collision model.

## Training contracts

### Observation and action layouts

Every deployable policy uses a 61-value actor observation and 14 actions. The
command block is ordered as twist (3), head pose (4), and body pose (6). Tasks
that do not use a command still retain and zero-pad its slots. Removing slots
breaks hot-swapping and deployment.

The 14 controls are left leg joints 0 through 4, neck and head joints 5 through
8, and right leg joints 9 through 13. Roller and backlash models interleave
passive joints. MDP code must use servo selectors rather than hardcoded MuJoCo
joint indices.

### BAM and domain randomization

BAM computes XL330 friction in the actuator model. Joint-friction
randomization scales `friction_scale`; changing MuJoCo `dof_frictionloss` is a
silent no-op for this actuator. Standalone configurations must install the BAM
friction expansion startup event.

Domain randomization must restore compile-time defaults before applying a new
sample. Custom additive or scaling randomizers cannot accumulate across
resets. Observation remapping, including backlash encoder and encoder-bias
views, must also be used by rewards that track the same quantity.

### Curriculum and reward checks

Curriculum time is measured in environment steps. With the standard rollout,
one runner iteration is 24 environment steps per environment. Weight schedules
are discrete stages, not interpolation. Update live manager term
configurations, not the original environment configuration objects.

Audit weighted term values during every run. A MicroDuck function named as a
penalty often returns a non-positive value and therefore needs a positive
weight. Stock mjlab costs often return non-negative values and need a negative
weight. A double negative rewards the violation.

## Troubleshooting

### A task is absent from the registry

Confirm the extra is installed and run `uv run rlx-mjlab-list-envs`. Inspect the
installed distribution's `mjlab.tasks` entry point if the command lists only
stock mjlab tasks. Avoid importing Torch, Warp, or MLX merely to test package
metadata.

### Training fails before iteration zero on Linux

Print the Torch build and CUDA availability. The expected installation uses the
configured CUDA 12.9 index and reports CUDA support. A `+cpu` Torch wheel causes
mjlab GPU selection to index an empty device list.

### Training works but exported ONNX fails

Use `rlx-mjlab-export`, then inspect the ONNX input and output dimensions. The
expected shapes end in 61 and 14. Run the parity utility for swing or compare a
fixed observation through the runner and ONNX Runtime. Do not compensate for a
missing normalizer in the deployment caller.

### A policy earns reward but looks wrong

Play or record the exact checkpoint, then compare it with zero-action or random
controls where the task supports them. Check contact geoms, orientation axes,
spawn type, and hold duration. Fix an exploration gap through physics or spawn
curriculum rather than adding payment for states the rollout never visits.

## Updating the vendored suite

The current import is a source snapshot from
`/Volumes/ExternalSSD/geoagent/microduck-playground/src/mjlab_microduck`, with
its operational scripts and tests copied alongside it. When refreshing the
snapshot, preserve the RLX namespace alias, package-relative tool imports,
package-relative MJCF paths, RLX command names in the Hugging Face bootstrap,
optional dependency table, entry points, and this guide. Run the complete
copied regression suite and a 64-environment, five-iteration smoke test before
accepting the refresh.
