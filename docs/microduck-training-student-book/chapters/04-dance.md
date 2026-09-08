# Dance Imitation: Learning a Timed Joint Choreography

## Learning Objectives

After this chapter, you should be able to:

- explain why playing an authored animation is different from controlling a physical robot;
- trace the dance experiment from clip preparation through PPO, ONNX export, and independent evaluation;
- derive the pose and dynamic-tracking metrics used to reject a stationary imitation;
- distinguish training reward, optimizer telemetry, deterministic skill evidence, and visual evidence;
- reproduce the recorded experiment without expanding its claim beyond the selected excerpt.

This chapter studies one verified result: a Microduck policy learned an **eight-second excerpt** from `dance-clip/bachata_microduck_v2.clip.json`. The source clip lasts 60.923 seconds, but the experiment did not train or validate the complete choreography. A separate sixteen-second test repeats the learned eight-second excerpt twice. It does not evaluate the first sixteen seconds of the original clip.

![Dance training and evaluation flow](assets/dance-flow.svg)

### Evidence labels used in this chapter

- **Measured** means a number is read from the retained recipe, checkpoint sidecar, JSONL optimizer journal, evaluation JSON, NPZ trace, or media validation artifact.
- **Implemented contract** means a behavior is enforced by the current source or focused tests, but the sentence is not itself a new experiment.
- **Inference** means a conclusion follows from measured evidence and code structure but was not isolated by a controlled ablation.
- **Pseudocode** is explanatory and is not directly executable.
- **Runnable** marks commands that can be executed from the workspace root. Commands that launch training are separately labeled because they are expensive and create new artifacts.

## Source-of-Truth Map

The chapter is grounded in the following workspace files:

| Evidence role | Authoritative local source |
|---|---|
| Deployment observation and action order | `microduck_local/src/microduck_local/contract.py` |
| Physics step, raw/applied action handling, observations | `microduck_local/src/microduck_local/walk_env.py` |
| Clip JSON loader and 50 Hz interpolation | `microduck_local/src/microduck_local/motion.py` |
| Imitation behavior and default reward terms | `microduck_local/src/microduck_local/behaviors/imitate.py` |
| Phase injection and weighted reward assembly | `microduck_local/src/microduck_local/behaviors/env.py` |
| Dance-specific Gaussian pose replacement and state evidence | `rlx/rlx/environments/microduck_recipes.py` |
| RLX vector adapter and normalization | `rlx/rlx/environments/microduck.py` |
| Actor-critic architecture and checkpoint format | `rlx/rlx/models/microduck.py` |
| PPO rollout, GAE, loss, numerical guards, and sample counting | `rlx/rlx/algorithms/ppo.py` |
| Deterministic ONNX graph | `rlx/rlx/export/microduck_onnx.py` |
| Studio train/eval/render/export CLI | `rlx/examples/ppo_microduck_studio.py` |
| Strict dance gate | `rlx/rlx/environments/dance_evaluation.py` |
| Excerpt and recipe generator | `rlx/scripts/dance_e2e.py` |
| Independent checkpoint/control audit | `rlx/scripts/audit_dance.py` |
| Recorded successful recipe | `rlx/artifacts/dance-e2e-20260907/recipe-low-noise.json` |
| Recorded successful checkpoint metadata | `rlx/runs/studio/dance/dance-e2e-20260907-low-noise/dance.safetensors.json` |
| Recorded independent result | `rlx/artifacts/dance-e2e-20260907/final-audit/audit.json` |

The local training playbook in `microduck_local/AGENTS.md` supplies the interpretation rules used here: do not change the 61/14 interface per task, do not treat reward as proof, export the observation normalizer with ONNX, compare with null controls, and visually inspect deterministic rollouts.

## The Task: Animation Becomes Control

An animation clip supplies desired joint angles over time. At control step $t$, let the fourteen-joint target be

$$
\mathbf{q}^{*}_t \in \mathbb{R}^{14}.
$$

The simulator contains the robot's actual joint positions $\mathbf{q}_t$, velocities, orientation, contacts, and actuator dynamics. A policy does not directly set $\mathbf{q}_t=\mathbf{q}^{*}_t$. It emits fourteen raw actions,

$$
\mathbf{a}_t = \pi_\theta(\mathbf{o}_t) \in \mathbb{R}^{14},
$$

which the environment transforms into servo targets through $\mathbf q^{\mathrm{target}}_t=\mathbf q^{\mathrm{default}}+\operatorname{clip}(\mathbf a_t,-4,4)$. The actor itself has a linear mean and Gaussian sampling, not a tanh-bounded output. The base environment records the raw previous action while clipping the applied offset. MuJoCo then advances the physical state. Balance, inertia, contact, and actuator response determine what pose is reached.

This distinction matters. A diagnostic controller that directly chased the reference joint angles fell after 0.98 seconds. The successful policy had to approximate the choreography while also learning stabilization. Therefore, the output is not an animation player. It is a feedback controller whose behavior is conditioned on the standard 61-dimensional observation.

The clip loader uses fourteen joint values and `rootPitch`. It does not imitate authored `rootYaw`, `rootRoll`, `rootPosition`, or contact annotations. The demonstrated result is consequently **time-aligned joint-pose imitation with physical balance**, not exact reconstruction of the full authored world-space motion.

## Inputs and Contract

The experiment preserves the shared deployment interface:

| Item | Contract |
|---|---|
| Observation | 61 floating-point values |
| Action | 14 raw joint-target offsets; applied offsets clipped to $[-4,4]$ rad |
| Control rate | 50 Hz |
| Selected reference | First contiguous 8 seconds |
| Reference samples | 400 control steps |
| Training actuator | Nominal XML servo model |
| Policy output | Deterministic ONNX actor |

### Exact observation tensor

**Implemented contract.** One unnormalized observation is a `float32` vector in this fixed order:

| Slice | Width | Meaning | Dance-specific behavior |
|---|---:|---|---|
| `0:3` | 3 | trunk angular velocity | ordinary physical state |
| `3:6` | 3 | projected gravity in the trunk frame | exposes pitch and roll, not world yaw |
| `6:20` | 14 | joint position minus `DEFAULT_POSE` | ordinary physical state |
| `20:34` | 14 | joint velocity | intentionally delayed one control step |
| `34:48` | 14 | previous raw policy action | raw value, even if the actuator command was clipped |
| `48:51` | 3 | twist command | zero for this Dance recipe |
| `51:55` | 4 | head command | tiny keep-alive noise in the base behavior |
| `55:59` | 4 | first four body-command values | tiny keep-alive noise in the base behavior |
| `59:61` | 2 | last two body-command values | replaced by clip phase $(\sin\phi,\cos\phi)$ |

The vector environment presents a batch with shape `(num_envs, 61)`. During training, running mean and variance are updated from these raw observations, normalization uses

$$
\widehat{\mathbf o}
=
\operatorname{clip}
\left(
\frac{\mathbf o-\boldsymbol\mu_o}
{\sqrt{\boldsymbol\sigma_o^2+10^{-8}}},
-10,10
\right),
$$

and the normalized `float32` batch is passed to the MLX network. Evaluation gives raw observations to the exported ONNX graph because the graph contains the same subtraction, variance offset, square root, division, and clipping operations.

### Exact action tensor and actuator path

**Implemented contract.** Training samples a batch shaped `(num_envs, 14)` from a diagonal Gaussian. The deterministic actor returns its mean. There is no actor-side `tanh`:

$$
\mathbf a_t \sim
\mathcal N\!\left(\boldsymbol\mu_\theta(\widehat{\mathbf o}_t),
\operatorname{diag}(\boldsymbol\sigma_\theta^2)\right).
$$

The environment stores the full raw action in `last_action`, exposes that raw value in the next observation, and only then computes

$$
\widetilde{\mathbf a}_t
=
\operatorname{clip}(\mathbf a_t,-4,4).
$$

With action delay disabled in the recorded recipe, the MuJoCo servo target is

$$
\mathbf q^{\mathrm{ctrl}}_t
=
\mathbf q^{\mathrm{default}}+\widetilde{\mathbf a}_t.
$$

The physics timestep is 0.005 seconds and each control action is held for four MuJoCo steps, giving a 0.02-second control interval. The recorded policy's exported ONNX interface is named `observations -> actions`, uses `float32`, and has dynamic batch shapes `("batch", 61) -> ("batch", 14)`.

The policy is memoryless, so it needs an observable clock. The existing imitation environment places a cyclic phase signal into two otherwise unused body-command slots:

$$
\phi_t = 2\pi \frac{t \bmod T}{T},
\qquad
\mathbf{p}_t =
\begin{bmatrix}
\sin \phi_t \\
\cos \phi_t
\end{bmatrix}.
$$

Using sine and cosine avoids a discontinuous scalar clock at the loop boundary. It also preserves the 61-value interface: the observation layout is not enlarged or reordered. The frozen-phase negative control confirms that this timing information matters. With the two phase slots held fixed, the final network survived but failed all five held-out trials because its dynamic tracking gain became negative.

The reference-preparation script records hashes for both the original source and generated excerpt. It retains the first eight seconds without amplitude scaling or teacher actions, interpolates the endpoint on the 50 Hz grid, and keeps looping enabled. Those provenance details prevent a later run from silently training on a different clip.

## Motion Dataset Construction

### Authored JSON contract

**Implemented contract.** `load_clip` accepts version-1 JSON with this logical schema:

```json
{
  "version": 1,
  "name": "clip name",
  "duration": 8.0,
  "loop": true,
  "keys": [
    {
      "t": 0.0,
      "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "rootPitch": 0.0
    }
  ]
}
```

Each `joints` array is interpreted in the deployment joint order:

```text
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

The source Bachata file has 401 authored keys over 60.92307692307693 seconds. The prepared eight-second file has 55 retained/interpolated keys, while the loaded training array has 400 control-rate rows. Authored key count and training sample count are therefore different quantities.

### Resampling onto the control grid

For duration $D$ and control frequency $f_c=50$ Hz, the loader computes

$$
T=\max(\operatorname{round}(Df_c),1),
\qquad
t_i=\frac{i}{f_c},
\quad
i\in\{0,\ldots,T-1\}.
$$

Every joint and `rootPitch` is independently linearly interpolated at each $t_i$. The eight-second reference therefore contains samples at 0.00 through 7.98 seconds. `Clip.at(400)` wraps to index zero because the excerpt is looping. The preparation script nevertheless appends an authored endpoint key at exactly 8.0 seconds, obtained by querying the original source clip at source step 400. That endpoint makes the generated JSON explicit at the excerpt boundary; the runtime loader still constructs exactly 400 rows.

**Pseudocode, not runnable:**

```text
source = parse_json(source_path)
sorted_keys = sort(source.keys by key.t)
full_clip = resample_each_joint_and_pitch(sorted_keys, 50_hz)
excerpt_steps = round(seconds * 50)
excerpt_keys = authored_keys_where_t_is_less_than_seconds
endpoint_joints, endpoint_pitch = full_clip.at(excerpt_steps)
append_key(t=seconds, joints=endpoint_joints, rootPitch=endpoint_pitch)
write_looping_excerpt_and_sha256_provenance()
```

**Measured.** The recorded source hash is `306164a79a0edb494c856cda09c9f0b4a36546fcf85bb53f940ed0dc697f6a12`; the generated reference hash is `108bf9e56b374b36d2d86c0698bace538ac5d1db0b57d4669d94d4c739190eac`. Twelve of fourteen reference joints exceed the evaluator's 0.03-rad moving-joint threshold; eight of those are leg joints. Left and right hip-yaw remain below that threshold in the full eight-second excerpt.

### What is and is not a dataset

This workflow does not construct observation-action demonstrations. There are no teacher actions, behavior-cloning labels, replay transitions, or privileged-state inputs. The "dataset" is a deterministic time-indexed target table:

$$
\mathcal D_{\mathrm{motion}}
=
\left\{
\left(i,\mathbf q_i^*,p_i^*\right)
\right\}_{i=0}^{399},
$$

where $p_i^*$ is reference root pitch. PPO still collects its own on-policy tuples $(\mathbf o_t,\mathbf a_t,r_t,\mathbf o_{t+1})$ from physics. Calling the clip a motion dataset is useful, but calling it a demonstration-action dataset would be incorrect.

## Policy and Optimizer Architecture

### Actor-critic

**Implemented contract.** Actor and critic are separate multilayer perceptrons. Each uses hidden widths 512, 256, and 128 with ELU activations:

$$
\text{actor mean}: 61 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 14,
$$

$$
\text{critic}: 61 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 1.
$$

The actor also owns fourteen state-independent log-standard-deviation parameters. For the recorded run, all are initialized to $\log(0.1)$. The Microduck model constrains log standard deviation to `[-5.0, -0.5]` before sampling and after every optimizer update. The deterministic ONNX export discards the critic and stochastic standard deviation, retaining only normalization plus the actor mean.

The complete trainable actor-critic contains 393,885 scalar parameters: 197,774 in the actor mean, 14 actor log-standard deviations, and 196,097 in the critic. The two networks do not share hidden layers.

### Rollout storage

One rollout buffer stores:

- observations and next observations: `(128, 16, 61)`;
- actions: `(128, 16, 14)`;
- rewards, terminations, truncations, values, and log probabilities: `(128, 16)`.

The rollout therefore holds 2,048 environment transitions. Timeout states are handled separately: if an episode truncates, PPO requires the true `terminal_observation`, evaluates its critic value, and uses that value in GAE instead of bootstrapping from the autoreset observation.

## Reward Design

The most important shaping term is a per-joint Gaussian pose score. With precision parameter $\sigma=0.2$ rad,

$$
r_{\text{pose},t}
=
\frac{1}{14}
\sum_{j=1}^{14}
\exp\left[
-\left(
\frac{q_{t,j}-q^{*}_{t,j}}{0.2}
\right)^2
\right].
$$

The final recipe multiplies this term by 40. A joint exactly on target contributes 1 before weighting. An error equal to $\sigma$ contributes $e^{-1}\approx0.368$. This sharper, per-joint kernel makes a broad stationary compromise less rewarding than the inherited pose score did.

The complete reward also retains terms for root-pitch matching, staying on the feet, avoiding slip and spin, gentle head contact, soft landings, joint-limit avoidance, and energy use. Forward travel is disabled. In compact form,

$$
r_t
=
40r_{\text{pose},t}
+4r_{\text{rotation},t}
+5r_{\text{feet},t}
-0.5c_{\text{slip},t}
-0.3c_{\text{spin},t}
-c_{\text{head},t}
-0.75c_{\text{landing},t}
-c_{\text{limit},t}
-0.0005c_{\text{energy},t}.
$$

This equation communicates the structure, not permission to infer causality from weights alone. The experiment did not perform a complete one-factor ablation. It established that the recorded combination worked, while earlier two-million-transition variants did not meet the dynamic-tracking gate.

![Dance reward and checkpoint evidence](assets/dance-reward.png)

### Exact reward assembly

**Implemented contract.** `BehaviorEnv` evaluates every configured term after the physics step, multiplies it by the active non-negative weight, and uses Python's built-in compensated `sum` for the scalar reward. Penalty functions return non-positive values; their weights remain non-negative.

| Output contribution | Recorded weight | Active meaning |
|---|---:|---|
| `pose_match` | 40.0 | per-joint Gaussian pose score with $\sigma=0.2$ |
| `rotation_match` | 4.0 | instantaneous root-pitch agreement for a looping clip |
| `stick_it` | 5.0 | zero for this looping clip |
| `on_feet` | 5.0 | upright, standing-height, foot-contact score |
| `no_slip_penalty` | 0.5 | squared horizontal velocity of contacting feet |
| `no_spin_penalty` | 0.3 | negative squared yaw rate |
| `travel` | 0.0 | explicitly disabled |
| `gentle_head_penalty` | 1.0 | head-floor impact penalty |
| `soft_landings_penalty` | 0.75 | vertical landing-impact penalty |
| `no_limit_parking_penalty` | 1.0 | joint-limit parking penalty |
| `save_energy_penalty` | 0.0005 | motor-strain penalty |

The `dancePoseSigma` option does not add a second pose reward. It replaces the inherited whole-body two-kernel function in the environment's precomputed reward rows with the per-joint Gaussian shown above. This distinction matters when reproducing earlier stages: setting `pose_match` to 40 without also setting `dancePoseSigma` uses a different function.

**Inference.** The per-joint mean makes errors local: tracking thirteen joints cannot fully hide one badly wrong joint. The inherited implementation first sums squared error over all fourteen joints and applies broad whole-pose kernels, so it is easier for a nearly static compromise to retain substantial credit. The experiment history is consistent with that mechanism, but it is not a complete controlled ablation.

## Training Process

The successful actor began from random initialization. It used no behavior cloning, teacher policy, physical assistance, joint-target feed-forward, or warm-start checkpoint. The recorded run used:

| Setting | Value |
|---|---:|
| Seed | 7 |
| Environment workers | 16 |
| Rollout length | 128 steps per worker |
| Batch size | 2,048 transitions |
| PPO epochs | 4 |
| Minibatches per epoch | 4 |
| Learning rate | $10^{-4}$ |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\epsilon$ | 0.2 |
| Initial action standard deviation | 0.1 rad |
| Total transitions | 4,001,792 |

### Requested versus actual PPO samples

Let

$$
B=N_{\mathrm{env}}N_{\mathrm{step}}
$$

be transitions per rollout. The PPO loop always collects a complete rollout before checking its outer `while self.step < target` condition again. Therefore, for a generic requested target $S$,

$$
U=\left\lceil\frac{S}{B}\right\rceil,
\qquad
S_{\mathrm{actual}}=UB.
$$

The Studio CLI checks only that $S\ge B$; it does not require divisibility. A request for 4,000,000 transitions with this recipe would actually collect 4,001,792. Progress messages clamp their displayed `steps` to the requested target, while checkpoint metadata and the training journal retain `algorithm.step`, the actual count. Students should use the latter for sample accounting.

**Measured successful run:**

$$
B=16\times128=2{,}048,
$$

$$
U=\frac{4{,}001{,}792}{2{,}048}=1{,}954.
$$

Each update performs four epochs and four minibatches per epoch:

$$
K=4\times4=16
$$

optimizer steps per rollout, for

$$
1{,}954\times16=31{,}264
$$

optimizer minibatches in total. Each minibatch contains $2{,}048/4=512$ transitions. Because the same rollout is revisited for four epochs, PPO performs 8,192 transition-presentations per update, but only 2,048 are newly sampled environment transitions. Do not report 16,007,168 transition-presentations as fresh experience.

PPO compares the updated policy with the policy that collected a rollout. For advantage estimate $\hat A_t$ and probability ratio

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)},
$$

the clipped policy objective is

$$
L^{\text{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

Clipping limits how strongly one update can exploit the current batch. It does not guarantee monotonic reward or skill improvement. The run performed 1,954 rollout/update cycles and 31,264 optimizer minibatches. The first update had approximate KL 7.76 and clip fraction 0.905, an aggressive startup retained in the evidence. Over the final 100 updates, mean approximate KL was 0.05317 and explained variance was 0.8951.

![Dance PPO loss telemetry](assets/dance-loss.png)

Training used observation normalization and running discounted-return reward normalization. The exported ONNX actor contains the frozen observation normalizer. Normalized rollout reward is therefore an optimizer input, not a directly comparable physical score. The evidence keeps raw completed-episode return, normalized collection reward, and deterministic checkpoint return separate.

### Actual PPO implementation details

The current RLX implementation adds several details beyond the compact textbook equation:

- advantages are normalized independently inside every optimizer minibatch;
- log probability ratios are clipped to `[-20, 20]` before exponentiation;
- value prediction uses a clipped value target and a Huber-like loss with transition point 10;
- value loss coefficient is 1.0 in the recorded recipe;
- entropy coefficient is 0, although entropy is still measured;
- gradient global norm is clipped to 0.5;
- loss, gradients, critic timeout values, update inputs, model parameters, and saved tensors are checked for finite values;
- a non-finite gradient aborts with an eager diagnostic naming unstable tensors rather than silently saving a corrupt checkpoint.

The recorded checkpoint also stores the running return-normalizer statistics. ONNX does not need those statistics: reward normalization affects optimization, not inference.

### Recorded experiment stages

The retained evidence shows an engineering sequence, not a single predeclared hyperparameter sweep:

| Stage | Retained evidence | Fresh transitions | Key change | Recorded outcome |
|---|---|---:|---|---|
| Pipeline smoke | `rlx/artifacts/dance-e2e-20260907/smoke/` | 4,096 | prove train/checkpoint/export plumbing | no skill claim |
| Broad-kernel Bachata | `recipe.json`, run `dance-e2e-20260907-bachata` | 2,000,896 | pose weight 40, inherited pose function, initial std 0.3 | standing/stability without passing dynamic gate |
| Precision tracking | `recipe-tracking.json`, run `dance-e2e-20260907-tracking` | 2,000,896 | replace pose term with $\sigma=0.2$ per-joint kernel | better pose discrimination, still no final skill pass |
| Low-noise full run | `recipe-low-noise.json`, run `dance-e2e-20260907-low-noise` | 4,001,792 | initial std 0.1 and longer budget | selected run; independent audit passed |
| Independent audit | `final-audit/audit.json` | 0 training transitions | ONNX, controls, hashes, traces, plots, parity | five of five held-out episodes passed |
| Repeated-cycle extension | `final-audit/extended-16s.json` | 0 training transitions | 800-step horizon over looping excerpt | three of three episodes passed two repeats |

All three substantive PPO recipes use seed 7 and fresh initialization. They are not continuation stages. The final five evaluation seeds test rollout/reset variation under deterministic policy inference; they are not five independently trained policies.

## Outputs and Reproducible Workflow

The workflow produces a prepared reference, recipe JSON, checkpoint and metadata, optimizer journal, deterministic ONNX model, skill report, plots, MP4 files, and contact sheets. Run these commands from the repository root. They are the recorded reproduction path, not a quick tutorial run:

### Runnable: inspect the retained evidence without training

```bash
python3 -m json.tool \
  rlx/artifacts/dance-e2e-20260907/reference-provenance.json
python3 -m json.tool \
  rlx/artifacts/dance-e2e-20260907/recipe-low-noise.json
python3 -m json.tool \
  rlx/runs/studio/dance/dance-e2e-20260907-low-noise/dance.safetensors.json
python3 -m json.tool \
  rlx/artifacts/dance-e2e-20260907/final-audit/audit.json
```

### Runnable: regenerate only the excerpt and recipe

This command writes a prepared reference and recipe. Despite its name, `dance_e2e.py` does **not** launch PPO:

```bash
(
set -eu
test ! -e rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS
rlx/.venv-microduck/bin/python rlx/scripts/dance_e2e.py \
  --source dance-clip/bachata_microduck_v2.clip.json \
  --output rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS \
  --seconds 8 --train-steps 4001792 \
  --run-name dance-e2e-reproduction-YYYYMMDD-HHMMSS
)
```

### Runnable and expensive: execute the full recorded train/eval/render/export path

Use a fresh run name and artifact path. The Studio server must already be available at `http://127.0.0.1:63317`.

```bash
(
set -eu
test ! -e rlx/runs/studio/dance/dance-e2e-reproduction-YYYYMMDD-HHMMSS
test ! -e rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/api-e2e.json
node duck-viewer/scripts/rlx-dance-api-e2e.mjs \
  --execute --base-url http://127.0.0.1:63317 \
  --recipe-json rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/recipe.json \
  --run dance-e2e-reproduction-YYYYMMDD-HHMMSS \
  --report rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/api-e2e.json \
  --timeout-seconds 7200
)
```

### Runnable: independently audit a completed reproduction

This runs deterministic simulation and optional rendering, but no PPO updates:

```bash
(
set -eu
test ! -e rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/audit
rlx/.venv-microduck/bin/python rlx/scripts/audit_dance.py \
  --run rlx/runs/studio/dance/dance-e2e-reproduction-YYYYMMDD-HHMMSS \
  --recipe rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/recipe.json \
  --output rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS/audit \
  --seeds 101 102 103 104 105 --history-stride 2 --render
)
```

The API driver performs train, evaluation, render, and export. It requires `--execute`, preserves render/export evidence after a failed full-profile skill evaluation, and exits nonzero when the final gate fails. The independent audit evaluates the exported ONNX actor and controls rather than trusting the training process.

### Runnable: focused tests without training

```bash
rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_dance_evaluation.py \
  rlx/tests/test_dance_pipeline.py \
  rlx/tests/test_microduck_dance.py
```

These tests cover clip routing, hashes, normalization choices, training telemetry shape, strict gate behavior, malformed evidence rejection, checkpoint/export contracts, and smoke integration surfaces. The gated integration test that actually trains is excluded unless its opt-in environment variable is set.

## Evaluation: Proving Motion, Not Merely Pose

Ordinary pose RMSE is

$$
\operatorname{RMSE}_{\text{pose}}
=
\sqrt{
\frac{1}{14T}
\sum_{t=1}^{T}
\sum_{j=1}^{14}
(q_{t,j}-q^{*}_{t,j})^2
}.
$$

A stationary policy can sometimes obtain a moderate RMSE by holding the reference's average pose. The evaluator therefore identifies moving joints whose reference standard deviation is at least 0.03 rad. It compares policy tracking error with a constant-reference-mean baseline:

$$
G_{\text{dynamic}}
=
1-
\frac{
\operatorname{RMSE}(\mathbf{q}_{M}-\mathbf{q}^{*}_{M})
}{
\operatorname{RMSE}(\mathbf{q}^{*}_{M}-\overline{\mathbf{q}^{*}_{M}})
}.
$$

Here $M$ is the selected moving-joint set; the leg metric restricts it to moving leg joints. A gain of zero means the policy is no better than holding the average target pose. Positive gain means it tracks the changing motion better than that static baseline. The strict pass requires:

- the complete 400-step horizon with no fall;
- contiguous, finite, time-aligned state measurements;
- upright score at least 0.9 for at least 95% of samples;
- fourteen-joint pose RMSE at most 0.15 rad;
- dynamic gain at least 20%;
- at least two moving leg joints and positive leg dynamic gain.

No smoothing, fitted lag, or temporal realignment is applied. All episodes must pass independently.

![Dance joint tracking](assets/dance-tracking.png)

The held-out ONNX audit passed five of five eight-second episodes. Mean pose RMSE was 0.08438 rad, mean dynamic gain was 38.39%, mean leg gain was 52.52%, and upright coverage was 100%. Three sixteen-second, two-cycle episodes also passed. The zero-action control passed zero of five and fell after 0.86 to 0.98 seconds; an untrained network fell after 0.74 seconds. Native and ONNX actions agreed within $3.58\times10^{-7}$ maximum absolute error.

These results support one local simulation claim: this recorded PPO recipe learned this selected excerpt. They do not establish the full Bachata clip, arbitrary dances, training-seed robustness, BAM equivalence, disturbance robustness, or hardware readiness.

## Reading the Evaluation JSON

The Studio report and independent audit answer related but different questions.

| Field | Meaning | Limitation |
|---|---|---|
| `pipeline_passed` | execution, finite values, and requested generic checks passed | does not alone prove dance skill |
| `skill_status` | `passed` or `failed` from the strict dance assessor in full mode | applies only to the configured horizon and conditions |
| `terminated` | count of fall/terminal events | zero does not imply accurate imitation |
| `truncated` | count of time-limit completions | expected for complete non-falling episodes |
| `pose_rmse_rad` | all-joint, all-step time-aligned RMSE | a static mean pose can still score moderately |
| `dynamic_tracking_rmse_rad` | RMSE on joints moving at least 0.03 rad standard deviation | moving set is horizon-dependent |
| `constant_mean_baseline_rmse_rad` | error of holding each moving joint at its reference mean | not a learned baseline policy |
| `dynamic_gain` | fractional improvement over the constant-mean baseline | negative values mean worse than static |
| `leg_dynamic_gain` | same calculation on moving leg joints | excludes head-only success |
| `mean_joint_correlation` | mean centered trajectory correlation over moving joints | diagnostic only; not a pass threshold |
| `upright_fraction` | fraction of samples with upright score at least 0.9 | does not measure foot placement quality |
| `height_m` | trunk-height mean and minimum | diagnostic only in the current dance gate |
| `raw_return` | sum of unnormalized task reward for that rollout | reward-weight dependent and shortened by early falls |
| `reward_per_step` | raw return divided by observed steps | can look high for an incomplete episode |
| `finite` | all stored observations/actions/rewards/traces were finite | necessary, never sufficient |

**Measured.** In the final independent audit, the five trained episodes have pose RMSE 0.0782-0.0983 rad, dynamic gain 28.05%-42.86%, leg gain 45.74%-54.74%, 400 measured steps, no termination, and upright fraction 1.0. Their mean values are summaries of five individually passing episodes, not pooled evidence that can hide a failing seed.

The final Studio `evaluation.json` used sixteen vector lanes at seed 7 and also passed all sixteen. The independent audit used seeds 101-105 in single environments. Agreement between those paths is useful because the Studio report exercises the production orchestration while the audit controls policy source, null policies, traces, checkpoint history, hashes, and ONNX parity.

### Why early-fall metrics can mislead

The moving-joint set is computed from the target samples that were actually observed. A zero-action rollout that falls after 43-49 steps sees only the early portion of the dance; in the retained audit that shortened window contains only three or four joints above the movement threshold and no moving leg joints. Its pose RMSE near 0.10 rad is therefore not comparable to a full 400-step pass. Completeness, contiguous phase, and leg-motion gates must be checked before interpreting the scalar RMSE.

## Failure Modes and Iteration Lessons

The experiment history shows why a complete evidence loop matters. A 4,096-transition smoke run proved only that training, checkpointing, and export were connected. It could not support a skill claim. A broader inherited pose kernel produced stable standing after roughly two million transitions, but the actor failed dynamic and leg tracking. Tightening the pose kernel improved pose accuracy without immediately solving timed motion. The final successful combination used the sharper $\sigma=0.2$ kernel, lower initial exploration standard deviation, and approximately four million transitions.

That sequence should not be read as a controlled proof that each adjustment was necessary. For example, the experiment did not include a four-million-transition run with the earlier 0.3 action standard deviation. It instead demonstrates an engineering progression: identify the stationary exploit, make the objective discriminate it, reduce destructive exploration for a precision task, and allow enough samples for the dynamic behavior to emerge.

The checkpoint history also rejects simple early stopping by reward. The policy remained near a standing compromise for a long interval, suffered a visible setback around 2.4 million transitions, and then improved its measured dynamic gain from about $-2\%$ at two million transitions to 17% at 3.2 million and 36% at 3.6 million. The final artifact was selected after the predetermined run completed and was then subjected to held-out physical tests.

When diagnosing a new dance run, ask in order: Did every episode cover the full reference? Did the actor remain upright? Is pose RMSE acceptable? Does it beat the constant-pose baseline? Do multiple leg joints move? Does the deterministic ONNX video agree with the metrics? A "no" at an earlier question cannot be repaired by a favorable answer later.

### Failure diagnosis matrix

| Symptom | Evidence to inspect first | Likely interpretation | Next safe experiment |
|---|---|---|---|
| Falls in under one second | `terminated`, steps, contact sheet, zero control | stabilization not acquired or action noise too destructive | inspect deterministic checkpoint history before changing reward |
| Survives but `dynamic_gain <= 0` | current/target traces and frozen-phase control | stationary compromise or timing input ignored | verify phase slots 59:61 vary and use the per-joint pose kernel |
| Good overall gain, non-positive leg gain | moving-joint counts and leg traces | head motion is carrying the aggregate metric | inspect leg-joint amplitudes and keep the leg-specific gate |
| Pose RMSE improves while gain remains below 0.20 | constant-mean baseline and checkpoint history | more accurate average pose without enough choreography | continue only if deterministic history shows dynamic progress |
| High training reward, failed ONNX | raw episode returns versus strict audit | optimizer objective and task evidence disagree | trust deterministic audit; inspect normalizer/export identity |
| Native actor and ONNX disagree | `onnx_parity_max_abs_error`, source hashes | export, normalization, or wrong-artifact problem | stop skill interpretation until parity is restored |
| Phase is repeated or skipped | `phase_step.contiguous` and trace length | evidence plumbing or autoreset alignment bug | fix measurement path; do not smooth or realign |
| Evaluation varies across nominal seeds although randomization is off | per-seed reset traces | worker seed changes initial numerical/reset trajectory | report spread; do not call them independent training seeds |
| Sixteen-second test passes | reference hash and loop flag | two repeats of the same learned excerpt | do not relabel as unseen sixteen-second choreography |

### Controls and what they falsify

- **Zero action** asks whether the spawn pose, passive servos, and gravity already satisfy the gate.
- **Untrained network** asks whether the architecture's random initial mean accidentally survives or tracks.
- **Frozen phase** preserves the trained network and physical feedback while removing the changing clock from its policy input. Its five episodes survived upright but had dynamic gains from -16.28% to -6.47%, which supports the claim that timed conditioning is functionally used.
- **Direct reference chasing** asks whether simply applying the authored joint targets is already a stable controller. The retained diagnostic fell at step 49, despite a favorable partial-window dynamic gain, showing why complete-horizon criteria come first.
- **Native/ONNX parity** asks whether the deployable graph represents the deterministic checkpoint actor on the same recorded observations.

The frozen-phase result is strong evidence for phase dependence, but it is not a complete feature attribution study. Setting two normalized inputs to fixed raw values also moves them away from their training distribution. The correct claim is that removing the changing phase signal breaks dynamic tracking, not that every learned timing mechanism has been isolated.

## How to Read the Figures

The reward figure separates stochastic training data from deterministic checkpoints. A falling exploratory episode can produce a downward raw-return spike even while the deterministic mean policy improves. Conversely, a high normalized reward cannot prove choreography.

The loss figure reports the actual PPO minibatch objective terms. Policy loss is a batch-relative surrogate, so it need not steadily approach zero. Value loss measures critic error under changing targets. Approximate KL and clip fraction indicate update size. Entropy describes the Gaussian action distribution; differential entropy may be negative.

The tracking figure is closest to the task claim because it compares physical joint trajectories against the authored targets at the same time steps. It should still be read with the acceptance table and video: the final motion retains visible timing lag, hip-roll offsets, and head-yaw overshoot.

## Read the saved Dance result in Studio

![Actual saved Dance verification panel. The source hash, episode count and tracking ranges bind the displayed result to a specific evaluated policy. This browser capture does not run new training.](assets/studio-dance-summary.png)

Select **Dance imitation**, then use **Saved runs** to restore the historical run and evaluation settings. Merely typing a run name selects artifacts; it does not restore its reference clip or recipe. Check the run identifier, 16/16 complete API episodes, joint-tracking ranges and the source hash. The five independent audit seeds in this chapter are a different evaluation set, not a contradiction of the UI's sixteen environments. Inspect the reference horizon before claiming the whole source choreography was learned, and play the complete saved video rather than treating this screenshot as proof of movement.

## Exercises

**Easy: verify the excerpt boundary.** Read the generated `reference-provenance.json` and `reference.clip.json`.

Acceptance check: report an excerpt start of 0 seconds, duration of 8 seconds, 400 control-rate poses, looping enabled, and no amplitude scaling. State explicitly that the source is longer.

**Easy: reconstruct the 61-value observation map.** Starting from `microduck_local/src/microduck_local/contract.py` and `_get_obs` in `walk_env.py`, write every half-open slice and its width. Then identify the exact two indices overwritten by `BehaviorEnv` for imitation.

Acceptance check: widths sum to 61; joint position, joint velocity, and previous action each have width 14; phase is at indices 59 and 60.

**Intermediate: account for PPO samples.** For a hypothetical request of 4,000,000 transitions under the recorded 16-by-128 rollout, calculate actual collected transitions, rollout updates, optimizer minibatches, minibatch size, and transition-presentations.

Acceptance check: distinguish newly collected environment transitions from repeated epoch presentations and explain why displayed progress may differ from checkpoint metadata.

**Intermediate: recompute dynamic gain.** Load one `trained-seed*.npz` trace from the final audit. Select joints whose target standard deviation is at least 0.03 rad, compute tracking RMSE and the constant-mean baseline RMSE, then calculate $G_{\text{dynamic}}$.

Acceptance check: your formula matches the evaluator, uses no fitted time shift, and produces a gain above 0.20 for the selected passing trace.

**Runnable exercise scaffold:**

```bash
python3 - <<'PY'
import numpy as np
path = "rlx/artifacts/dance-e2e-20260907/final-audit/trained-seed101.npz"
trace = np.load(path)
current = trace["current"]
target = trace["target"]
moving = np.flatnonzero(target.std(axis=0) >= 0.03)
tracking = np.sqrt(np.mean((current[:, moving] - target[:, moving]) ** 2))
baseline = np.sqrt(np.mean((target[:, moving] - target[:, moving].mean(axis=0)) ** 2))
print({"moving": moving.tolist(), "tracking": tracking, "baseline": baseline, "gain": 1 - tracking / baseline})
PY
```

**Intermediate: verify ONNX graph identity.** Inspect the retained ONNX graph's input/output names, element types, shapes, and normalization initializers.

Acceptance check: report `observations` and `actions`, `float32`, dynamic batch axes, widths 61 and 14, and the presence of observation mean, variance, epsilon, and clip constants.

**Advanced: design a falsification audit.** Compare the trained actor with zero action, an untrained actor, and the trained actor with frozen phase inputs. Specify what each control tests.

Acceptance check: require complete episodes, per-episode results, ONNX/native parity, physical tracking, and visual review. Reject any argument that uses reward alone or treats a repeated excerpt as unseen choreography.

**Advanced: test a timing-lag hypothesis without weakening the gate.** Compute cross-correlation between current and target trajectories as a diagnostic, but keep the official zero-lag RMSE and gain unchanged.

Acceptance check: clearly label the fitted lag as exploratory analysis. Do not use it to convert a failed official episode into a pass.

**Advanced: design a training-seed study.** Specify at least five independently initialized PPO runs with fixed recipe, fixed sample budget, distinct training seeds, and the same held-out evaluation seed set.

Acceptance check: separate within-policy rollout variation from between-training-seed variation; predeclare how many policies must pass; retain failed checkpoints and videos.

## Evidence Gaps and Claim Boundary

The retained artifacts do not answer several important questions:

- only one successful training seed is recorded, so scratch-PPO reliability is unknown;
- the recipe sequence changes both exploration standard deviation and sample budget between the tracking and successful runs, so their individual causal effects are not isolated;
- there is no four-million-transition run with initial standard deviation 0.3;
- there is no domain-randomized or BAM audit of the selected Dance actor;
- the source choreography beyond the first eight seconds was not trained or evaluated;
- evaluator thresholds are fixed and regression-tested, but they are engineering acceptance criteria rather than hardware-derived tolerances;
- the independent audit uses nominal simulation with randomization disabled and does not apply pushes, terrain changes, sensor corruption, latency variation, or actuator mismatch;
- visible lag and joint-specific overshoot are documented qualitatively, but no frequency-domain or phase-delay tolerance is part of the gate.

The stop condition for this chapter is therefore narrow: the recorded deterministic ONNX policy is verified to balance and dynamically track the selected looping eight-second joint excerpt under the retained nominal MuJoCo conditions. Nothing in the evidence upgrades that result to complete-song imitation, robust learning across seeds, sim2real readiness, or safe hardware deployment.
