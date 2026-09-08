# Swing: Teacher-Assisted Acquisition and PPO Refinement

## Learning Objectives

After this chapter, you should be able to:

- explain why the successful Swing result is **teacher acquisition followed by PPO refinement**, not scratch-PPO discovery;
- write the exact privileged teacher equation and map its 13 parameters into the 14-action contract;
- unpack every block of the student's 61-observation input without claiming that privileged swing angle or rate is appended;
- reconstruct the saved behavior-cloning and DAgger datasets, including shapes, counts, start bounds, and the actual no-mixture beta schedule;
- explain the BC loss, optimizer, minibatching, normalization statistics, and why a lower BC loss did not determine checkpoint selection;
- calculate exactly how 524,288 PPO transitions become 128 rollouts and 1,024 optimizer minibatches;
- explain what happens to the actor mean, critic, state-independent `actor_log_std`, normalizers, and optimizer state at the imitation-to-RL handoff;
- read bootstrap JSON, checkpoint sidecars, PPO JSONL telemetry, API reports, audits, and rollout traces without confusing their meanings;
- apply the strict physical gate and explain why the million-transition continuation is a failed policy even though it still swings widely;
- distinguish runnable repository examples from conceptual pseudocode and reproduce the staged pipeline without overwriting evidence.

The central attribution rule is non-negotiable:

> The selected observation-only actor acquired Swing through behavior cloning and on-policy DAgger. It already passed the strict Swing gate with `ppo_steps: 0`. PPO then refined that competent initializer for 524,288 transitions.

Fresh PPO did not discover the large-amplitude behavior. A scratch-PPO run completed 1,048,576 transitions but produced only about 2.23 degrees of still-start bidirectional span. Conversely, calling the final policy "just imitation" would also be wrong: PPO changed both actor and critic, raised the audited deterministic return and span, and produced its own optimization telemetry.

![Swing acquisition and refinement flow](assets/swing-flow.svg)

## The Task and Its Physical Measurements

The simulated robot sits in a retained swing seat suspended by two compliant tendons. Control runs at 50 Hz. The policy always uses the shared 61-observation, 14-action interface.

The swing recipe measures the average anchor location, the average seat-attachment location, and the average attachment velocity. If the anchor-to-attachment vector is $(x,y,z)$, then the sagittal swing angle and rate are

$$
\theta = \operatorname{atan2}(x,-z),
$$

$$
\dot\theta =
\frac{-z\dot x+x\dot z}{x^2+z^2}.
$$

The angle is zero when the seat hangs below the anchors. Positive and negative values are the two directions in the vertical swing plane. The teacher consumes $\theta$ in radians and $\dot\theta$ in radians per second. Reports convert $\theta$ to degrees.

The strict task begins from rest:

$$
\theta_0=0,\qquad \dot\theta_0=0.
$$

The bootstrap collector may reset inside bounded intervals around nonzero angle and rate. Those starts are data-acquisition aids, not accepted evaluation conditions. Skill-mode argument validation rejects nonzero Swing initial-angle or initial-rate bounds.

The tendon length metrics are sampled from MuJoCo. The reported spring-only tension is

$$
F_{\text{spring}}=k\max(l-l_0,0).
$$

Safety-limit constraint forces are excluded. A positive sampled value therefore means the modeled tendon spring was stretched at that 50 Hz sample; it is not a continuous-time hardware load guarantee.

### Planar action projection

The actor still emits 14 values. The environment clips each to $[-1,1]$ and, for this recipe, projects them into a sagittal subspace:

- action indices `0, 1, 7, 8, 9, 10` are set to zero;
- pairs `(2,11)`, `(3,12)`, and `(4,13)` are replaced by equal-and-opposite values;
- indices `5` and `6`, the neck-pitch and head-pitch actions, remain independently available;
- the projected action is stored in the observation's `last_action` block;
- the Swing wrapper passes the base environment the offset
  `SEATED_POSE - DEFAULT_POSE + 0.7 * projected_action`;
- the base environment clips that offset to `[-4,4]` and adds `DEFAULT_POSE`,
  so the effective joint target is
  `SEATED_POSE + 0.7 * projected_action` when the final offset clip is
  inactive.

This is an environment-side action restriction, not an extra teacher input and not a smaller deployed action tensor.

## Why Scratch PPO Failed

PPO estimates a policy gradient from sampled trajectories:

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid o_t)\hat A_t
\right].
$$

If rollouts contain only stillness or tiny oscillation, PPO receives little evidence about the coordinated phase-dependent motion that injects energy into the swing. A larger reward coefficient cannot retrospectively assign credit to an action sequence that was never sampled.

The failed scratch run is therefore evidence about acquisition under that recipe, not proof that PPO can never learn Swing. Its optimizer ran, but the state-action distribution did not contain the required behavior. The successful pipeline deliberately changed that distribution:

1. a bounded privileged controller generated useful action labels;
2. BC placed an observation-only student near those labels;
3. DAgger queried the teacher on states reached by the student;
4. strict evaluation selected a competent imitation checkpoint;
5. PPO refined that checkpoint under the real reward and on-policy exploration.

## Stage 0: The Exact Privileged Teacher

The teacher parameter vector has 13 finite values:

$$
\mathbf{p}=(b,k_\theta,k_{\dot\theta},
\mathbf{o}_{0:5},\mathbf{a}_{0:5}).
$$

For a simulator state $(\theta,\dot\theta)$, it computes

$$
q=\tanh(b+k_\theta\theta+k_{\dot\theta}\dot\theta),
$$

$$
\mathbf{s}=
\operatorname{clip}(\mathbf{o}+\mathbf{a}q,-1,1),
$$

where $\mathbf{s}\in\mathbb{R}^5$. The saved source gives:

```text
bias                 0.14797230529360234
angle gain         -12.002270229601733
rate gain           -0.5333935911538304

offsets [-0.2263374287, -0.2579816298,  0.0721266818,
         -0.2121294270,  0.1456699723]

amplitudes [-0.4865735091, -0.9590667100, -1.0000000000,
             0.6472595167, -0.6942870027]
```

The five sagittal values are inserted into the 14-action array as follows:

```python
# Conceptual transcription of teacher_action; see the source for runnable code.
action = zeros(14)
action[2:7] = sagittal
action[11:14] = -sagittal[:3]
```

Thus:

- `action[2:7]` controls left hip pitch, left knee, left ankle, neck pitch, and head pitch;
- `action[11:14]` mirrors the first three values onto right hip pitch, right knee, and right ankle with opposite sign;
- yaw, roll, head yaw, and head roll remain zero.

Only `label_observation()` calls the private simulator method `_swing_state()`. Tests inspect the script's syntax tree to enforce that boundary. The exported `ObservationActor` stores only an ONNX session and accepts finite arrays shaped `[batch, 61]`; it has no environment or teacher reference.

## The Student's Exact 61 Observations

Each raw student observation is a fresh `float32[61]` array:

| Slice | Width | Meaning for Swing |
|---|---:|---|
| `0:3` | 3 | base angular velocity |
| `3:6` | 3 | gravity projected into the trunk frame |
| `6:20` | 14 | joint position relative to the shared default pose |
| `20:34` | 14 | one-control-step-lagged joint velocity |
| `34:48` | 14 | previous projected policy action |
| `48:51` | 3 | Swing command: `(0, body_y_axis_x, body_y_axis_z)` |
| `51:55` | 4 | head command, zero in Swing |
| `55:61` | 6 | body command, zero in Swing |

The student does **not** receive a separately appended $\theta$ or $\dot\theta$. It must infer useful phase information from ordinary proprioception: angular velocity, projected gravity, joint state, prior action, and the retained command slots. The Swing-specific command assignment changes values within the existing contract; it does not alter the size or ordering.

The 14 outputs remain normalized action offsets. The ONNX export contains the observation normalizer and deterministic actor mean. It exports neither a Gaussian sample nor the critic.

## Stage 1: Initial Behavior Cloning

### Demonstration collection

Initial collection executes teacher actions. The four reset bounds are:

```text
(maximum |angle| in degrees, maximum |rate| in rad/s)
(0, 0), (2, 0.05), (5, 0.1), (10, 0.2)
```

`reset()` samples uniformly inside the signed interval for each bound. The collector runs eight episodes, cycling through those four bounds twice. Every episode can contribute at most 1,200 control steps. The saved run reached the full count:

$$
8\text{ episodes}\times1{,}200
=9{,}600\text{ examples}.
$$

For each pre-action state it saves:

- the raw student observation $\mathbf{o}_i\in\mathbb{R}^{61}$;
- the teacher label $\mathbf{a}^T_i\in[-1,1]^{14}$;
- batch provenance containing seed, reset bounds, sample count, executed behavior, and label source.

### Frozen input normalization

The bootstrap computes an observation mean $\mathbf{m}_{\text{obs}}$ and
observation variance $\mathbf{v}_{\text{obs}}$ once from these initial 9,600
raw observations:

$$
\mathbf{m}_{\text{obs}}=
\operatorname{mean}(\mathbf{O},\text{axis}=0),
$$

$$
\mathbf{v}_{\text{obs}}=
\max\left(\operatorname{var}(\mathbf{O},\text{axis}=0),
\mathbf{f}^2\right).
$$

The per-dimension standard-deviation floors are:

| Observation block | Floor |
|---|---:|
| angular velocity `0:3` | 0.03 |
| projected gravity `3:6` | 0.01 |
| joint position `6:20` | 0.01 |
| joint velocity `20:34` | 0.25 |
| last action `34:48` | 0.05 |
| all 13 command values `48:61` | 0.01 |

Training inputs are

$$
\tilde{\mathbf{o}}_i=
\operatorname{clip}
\left(
\frac{\mathbf{o}_i-\mathbf{m}_{\text{obs}}}
{\sqrt{\mathbf{v}_{\text{obs}}+10^{-8}}},
-10,10
\right).
$$

Crucially, later DAgger observations do not update
$\mathbf{m}_{\text{obs}}$ or $\mathbf{v}_{\text{obs}}$. The saved provenance
calls this `normalizer_frozen_sample_count: 9600`. PPO also freezes this
observation normalizer, preserving the coordinate system in which the actor
was fitted. These symbols denote input-normalization statistics;
$\mu_\theta(\cdot)$ below denotes the actor's mean action.

### Objective and optimizer

Only the actor-mean network is optimized:

$$
L_{\text{BC}}(\theta)
=
\frac{1}{14N}
\sum_{i=1}^{N}
\left\|
\mu_\theta(\tilde{\mathbf{o}}_i)-\mathbf{a}^{T}_i
\right\|_2^2.
$$

The factor $14$ is required because MLX `mean` reduces over both the batch
dimension and all 14 action components. Writing $1/N$ with the same squared
Euclidean norm would describe a loss scaled 14 times larger.

Implementation details:

- actor and critic architecture: separate `61 -> 512 -> 256 -> 128` ELU MLPs;
- actor output width: 14;
- optimizer: MLX Adam with learning rate $3\times10^{-4}$;
- shuffled minibatch size: at most 512;
- requested epochs per stage: 80;
- loss reduction: mean over batch and action dimensions;
- no validation split, explicit regularizer, learning-rate schedule, early-stopping criterion, or BC gradient clipping is present;
- optimizer state persists across BC/DAgger stages in one bootstrap process;
- checkpoint serialization stores model weights and normalizers, not the BC Adam state.

The initial dataset has $\lceil9600/512\rceil=19$ minibatches per epoch, so 80 epochs produce $19\times80=1{,}520$ optimizer updates.

## Stage 2: On-Policy DAgger

Pure BC trains on the teacher's state distribution $d_{\pi_T}$. During deployment, small student errors change future observations, so the student instead encounters $d_{\pi_\theta}$. DAgger reduces this covariate shift by asking the teacher what should have been done on states the current student actually visits:

$$
\mathcal{D}_{k+1}
=
\mathcal{D}_k
\cup
\{(\mathbf{o},\pi_T(s)):
\mathbf{o}\sim d_{\pi_{\theta_k}}\}.
$$

In this implementation the ordering is exact:

1. reset a bounded-start environment;
2. read raw 61D observation $\mathbf{o}_t$;
3. query privileged state and compute teacher label $\mathbf{a}^T_t$ **before acting**;
4. append $(\mathbf{o}_t,\mathbf{a}^T_t)$;
5. pass raw $\mathbf{o}_t$ to the exported ONNX actor, whose embedded
   normalizer computes
   $\tilde{\mathbf{o}}_t=\operatorname{clip}((\mathbf{o}_t-\mathbf{m}_{\text{obs}})
   /\sqrt{\mathbf{v}_{\text{obs}}+10^{-8}},-10,10)$ before evaluating
   $\mu_{\theta_k}(\tilde{\mathbf{o}}_t)$, and execute that returned action;
6. receive $\mathbf{o}_{t+1}$ and repeat;
7. append the new batch to all earlier data and retrain on the accumulated dataset.

This is DAgger because the labels come from the teacher while the visited states come from the learner.

### The actual beta schedule

Many DAgger descriptions define a mixture policy

$$
\pi_{\beta_k}=\beta_k\pi_T+(1-\beta_k)\pi_{\theta_k}.
$$

The saved Swing bootstrap does **not** sample a stochastic or linearly decaying mixture. Its effective execution schedule is the hard switch:

| Collection stage | Executed action source | Effective $\beta$ |
|---|---|---:|
| initial BC collection | teacher | 1 |
| DAgger stages 1-5 | student ONNX actor | 0 |

The teacher still labels every DAgger state. Saying that beta "decays over five iterations" would invent behavior not present in the source.

### Dataset shapes and stage counts

Each stage attempts eight more 1,200-step episodes, so it adds up to 9,600 rows. The saved run completed all batches:

| Stage | Interpretation | Cumulative rows | Observation shape | Label shape | Updates | Strict result |
|---|---|---:|---|---|---:|---|
| 0 | teacher-distribution BC | 9,600 | `[9600,61]` | `[9600,14]` | 1,520 | fail: span, geometry, tension |
| 1 | first learner-distribution aggregation | 19,200 | `[19200,61]` | `[19200,14]` | 3,040 | fail: span |
| 2 | second aggregation | 28,800 | `[28800,61]` | `[28800,14]` | 4,560 | fail: one tension sample |
| 3 | third aggregation | 38,400 | `[38400,61]` | `[38400,14]` | 6,000 | **pass; selected** |
| 4 | fourth aggregation | 48,000 | `[48000,61]` | `[48000,14]` | 7,520 | fail: geometry |
| 5 | fifth aggregation | 57,600 | `[57600,61]` | `[57600,14]` | 9,040 | fail: span |

The final saved `demonstrations.npz` contains finite `float32` arrays:

```text
observations: (57600, 61)
labels:       (57600, 14)
```

This final archive includes data collected after the selected stage. The selected checkpoint itself was trained through stage 3 on 38,400 rows. Therefore `demonstrations.npz` row count and selected checkpoint `teacher_samples` intentionally differ.

### Why BC loss does not select the policy

Stage 3 had last BC loss about 0.00346 and passed. Stage 5 had a lower last loss, about 0.00158, but failed the 150-degree span gate. Stage 4 also had lower mean loss than stage 3 but violated geometry.

Supervised error averages all action dimensions over the collected dataset. It does not directly encode long-horizon phase stability, bidirectional amplitude, tendon tension, or the worst physical sample. Therefore selection is lexicographic:

```text
(strict pass,
 minimum valid-geometry fraction,
 minimum both-strings-tensioned fraction,
 minimum bidirectional span)
```

Pass status dominates all other fields. A larger but invalid swing cannot beat a smaller valid pass.

The selected stage-3 strict episode had approximately:

- positive peak: $79.09^\circ$;
- negative peak: $-77.60^\circ$;
- symmetric bidirectional span: $155.20^\circ$;
- valid geometry: 100%;
- both strings tensioned: 100%;
- maximum alignment: 0.00224;
- minimum sampled spring tension: 0.290 N.

Both bootstrap evaluation lanes were effectively identical under nominal deterministic resets. They verify repeatable execution, not disturbance robustness.

## The Imitation-to-PPO Handoff

The selected checkpoint is a full actor-critic container, but imitation trained only `model.actor_mean`.

### What is competent

- actor mean weights have been fitted through BC/DAgger;
- observation mean, variance, count, epsilon, and clip are saved;
- deterministic ONNX parity has been checked;
- the observation-only actor passes the strict still-start gate.

### What is not pretrained

- the critic remains at its random initialization;
- no BC optimizer state is loaded into PPO;
- reward-return normalization has not learned a useful scale: bootstrap saves mean 0, variance 1, and count $10^{-4}$;
- the bootstrap metadata correctly states `steps: 0`, `ppo_steps: 0`, and `critic: "random, not pretrained"`.

### Exploration standard deviation

The actor-critic owns a state-independent 14-vector `actor_log_std`. The bootstrap creates the model with `initial_std=0.1`, so initially

$$
\log\sigma_j=\log(0.1)\approx-2.302585
$$

for every action dimension. BC does not optimize this vector because its optimizer targets `model.actor_mean` only. It is nevertheless serialized and loaded with the checkpoint.

At PPO construction and after every update, MicroDuck constrains

$$
-5\le\log\sigma_j\le-0.5.
$$

The inherited value $\log(0.1)$ is already inside that interval. PPO samples

$$
\mathbf{a}_t=
\mu_\theta(\tilde{\mathbf{o}}_t)
+\boldsymbol\sigma\odot\boldsymbol\epsilon_t,
\qquad
\boldsymbol\epsilon_t\sim\mathcal{N}(0,I),
$$

while deterministic evaluation and ONNX use only $\mu_\theta$. Entropy coefficient zero means entropy is logged but not rewarded; `actor_log_std` can still change through the PPO likelihood objective.

### Normalizer and optimizer handling

PPO loads the bootstrap observation statistics and freezes observation-statistic updates. Reward normalization remains enabled and updates `return_rms` from discounted raw rewards:

$$
R_t^{\text{disc}}=\gamma R_{t-1}^{\text{disc}}+r_t,
\qquad
\hat r_t=
\operatorname{clip}
\left(
\frac{r_t}{\sqrt{\operatorname{Var}(R^{\text{disc}})+10^{-8}}},
-10,10
\right).
$$

PPO starts a new Adam optimizer at learning rate $3\times10^{-6}$. Both actor and critic parameters are then optimized by the PPO objective. The actor begins from a useful solution; the critic begins by catching up to on-policy returns.

## Stage 3: Exactly 524,288 PPO Transitions

The selected recipe is:

| Setting | Value |
|---|---:|
| PPO transitions | 524,288 |
| parallel environments | 16 |
| rollout steps per environment | 256 |
| rollout transitions | 4,096 |
| update epochs | 2 |
| minibatches per epoch | 4 |
| minibatch rows | 1,024 |
| optimizer steps per rollout | 8 |
| learning rate | $3\times10^{-6}$ |
| $\gamma$ | 0.995 |
| GAE $\lambda$ | 0.98 |
| normalize advantages | true |
| policy/value clip coefficient | 0.05 |
| clipped value loss | true |
| value coefficient | 1.0 |
| entropy coefficient | 0 |
| max gradient norm | 0.5 |
| reward normalization | enabled |
| observation normalization | frozen |
| checkpoint interval | 262,144 transitions |

A **transition** is one environment control step from one vector lane. It is not one optimizer step and not one MuJoCo physics substep.

The arithmetic is:

$$
16\text{ envs}\times256\text{ steps}=4{,}096
\text{ transitions per rollout},
$$

$$
524{,}288/4{,}096=128
\text{ rollout/update cycles},
$$

$$
4{,}096/4=1{,}024
\text{ rows per minibatch},
$$

$$
128\times2\times4=1{,}024
\text{ optimizer minibatches}.
$$

Because checkpoints are written at step 0 and every 262,144 transitions, the run contains snapshots at:

```text
0, 262144, 524288
```

The 19,200 transitions used by the later 16-lane, 1,200-step deterministic evaluation are evaluation data, not part of the 524,288 training budget.

### PPO objective details

The implementation forms GAE, normalizes advantages within each optimizer minibatch, and applies the clipped PPO policy loss. Log probability ratios are clipped to `[-20,20]` before exponentiation to prevent finite but extreme differences from overflowing.

The critic uses clipped Huber loss with transition at absolute error 10. Value predictions may also be clipped around the old value by the same 0.05 coefficient. Gradients are checked for finiteness, clipped to norm 0.5, and the save helper refuses non-finite tensors.

Timeouts are not treated as true terminal failures. For truncated lanes, PPO requires the original terminal observation, evaluates its critic value, and uses that value in GAE. Autoreset observations are not substituted for the timed-out final state.

![Swing PPO loss telemetry](assets/swing-loss.png)

The saved v3 JSONL has:

- 128 collection rows;
- 128 update rows;
- 27 completed-episode rows containing 432 episode returns;
- 283 total rows;
- 1,024 optimizer steps reported across all updates;
- finite numeric update telemetry.

Across the final 100 update rows, mean approximate KL was about 0.000472 and clip fraction about 0.0969. Mean explained variance was about $-0.8999$.

Negative explained variance means the critic's value errors varied more than returns around a constant baseline on those minibatch samples. It is a real optimization limitation. It does not automatically invalidate the deterministic actor's separately measured physical pass, but it warns against claiming a well-fitted value function or treating PPO loss telemetry as proof of skill.

## Reward Structure

The reward weights used by the selected recipe are:

| Term | Weight |
|---|---:|
| new peak progress | 224 |
| height gain | 8 |
| late height gain | 24 |
| energy gain | 0.5 |
| lateral penalty | 3 |
| lateral barrier | 8 |
| lateral velocity | 3 |
| out-of-plane angular velocity | 1 |
| alignment penalty | 18 |
| alignment barrier | 4 |
| string slack/imbalance | 8 |
| string extension | 12 |
| invalid episode | 10 |
| action rate | 0.03 |
| joint torque | 0.001 |
| joint limit | 1 |

The peak term pays only newly reached positive and negative angular frontiers. If $p_t$ and $n_t$ are the best positive and negative magnitudes so far, the implementation pays bounded increments of their squared normalized values. The per-step frontier advance is capped at $8\Delta t$ before squaring:

$$
r_{\text{frontier},t}
\propto
\Delta\left(\frac{p_t}{\pi}\right)^2
+
\Delta\left(\frac{n_t}{\pi}\right)^2.
$$

Repeatedly revisiting an old peak does not farm the same progress reward.

Height and energy are

$$
h(\theta)=\operatorname{clip}(1-\cos\theta,0,2),
$$

$$
E(\theta,\dot\theta)=
\operatorname{clip}
\left(
h(\theta)+\frac{1}{2}\frac{L}{g}\dot\theta^2,
0,2
\right).
$$

The environment subtracts the reset's assisted-energy baseline before paying height or energy gain. A validity multiplier decays near lateral, alignment, slack, and extension boundaries. Two consecutive invalid samples latch the episode as invalid, set positive validity to zero, and activate the invalid-episode penalty.

Reward remains a training signal, not the acceptance definition. The gate below inspects physical measurements directly.

![Swing reward and checkpoint evidence](assets/swing-reward.png)

## Reading the Artifact and Log Schemas

### Bootstrap outputs

`stage-NN/training.json` describes one supervised stage:

- `stage`: 0 for initial BC, 1-5 for DAgger stages;
- `samples`: cumulative dataset rows used at that stage;
- `epochs_started`: epochs entered before a time budget stopped training;
- `optimizer_updates`: completed BC minibatches;
- `mean_bc_loss`, `last_bc_loss`: supervised MSE, not return;
- `onnx_parity_max_error`: maximum difference on a sampled parity batch;
- `strict_passed`: result of the two-lane strict evaluation;
- `selection_key`: lexicographic physical checkpoint-ranking tuple;
- `strict_episodes`: complete per-lane physical reports.

`provenance.json` records source/script hashes, hard limits, collection batches, all stage summaries, selected stage, frozen-normalizer count, artifact hashes, timings, and explicit limitations.

`demonstrations.npz` stores the final accumulated raw observations and teacher labels. It does not identify which rows alone trained the selected checkpoint; use stage metadata for that.

`swing.safetensors.json` is the checkpoint sidecar. The bootstrap sidecar records teacher source, parameters, sample count, normalizer policy, `ppo_steps: 0`, and random-critic status. The PPO sidecar nests that record under `initialization.loaded_metadata` and adds the PPO recipe and current `steps`.

### PPO `training-metrics.jsonl`

Each line is one JSON object:

- `phase: "collection"`: one 4,096-transition rollout, wall seconds, ending `env_steps`, and mean **normalized stochastic collection reward per transition**;
- `phase: "update"`: optimizer time, eight minibatch updates, policy/value/total losses, Gaussian entropy, approximate KL, clip fraction, explained variance, and ending `env_steps`;
- `phase: "episodes"`: raw completed-episode returns and lengths emitted when lanes finish, plus their mean raw return.

Collection `mean_reward` and episode `mean_raw_return` are intentionally different scales. The former is normalized and per transition; the latter is an unnormalized sum over an episode. Neither equals deterministic evaluation return.

### Evaluation and audit outputs

`evaluation.json` records one vectorized deterministic evaluation, including:

- recipe and source hashes;
- finite/pipeline/skill status;
- 16 per-lane Swing episode reports;
- raw rollout return;
- action magnitudes;
- metric summaries;
- termination and truncation counts.

The independent audit runs trained, zero, and random controls over named seeds and writes per-seed `.npz` traces, an `audit.json`, plots, comparison media, and hashes. Each trace preserves time-series values such as angle, tension, and lateral offset; the summary is not a substitute when diagnosing the exact failing sample.

The API report records requested and normalized recipes, operation state, captured process logs, final evaluation, render/export evidence, and any failure. A succeeded training operation plus a failed skill evaluation is still a failed experiment outcome.

## Strict Evaluation

For each 24-second, 1,200-step episode, let $\theta^+$ be the maximum positive angle and $\theta^-$ the magnitude of the most negative angle. The evaluator reports:

$$
S_{\text{peak-to-peak}}=\theta^+ + \theta^-,
$$

$$
S_{\text{bi}}=2\min(\theta^+,\theta^-).
$$

The symmetric metric prevents one large excursion and one weak return from passing. For example, $+90^\circ$ and $-40^\circ$ yield $130^\circ$ peak-to-peak but only $80^\circ$ bidirectional span.

Every episode must satisfy all of the following:

- $S_{\text{bi}}\ge150^\circ$;
- exactly 1,200 measured steps;
- time-limit truncation after the complete horizon, with no termination;
- finite required metrics at every sample;
- absolute lateral offset at most 0.020 m;
- alignment between 0 and 0.050;
- both string lengths between 0.370 m and 0.394 m;
- `valid_geometry == 1` at every measured sample;
- strictly positive spring tension in both strings at every sample.

The evaluator accumulates vector lanes and reset episodes separately. It never pools positive peaks from one episode with negative peaks from another.

![Swing physical tracking](assets/swing-tracking.png)

## Checkpoint Selection and the Bad Continuation

The selected v3 final policy produced:

- 16/16 passing API-evaluation episodes;
- five passing independent audit seeds, 501-505;
- positive peak about $82.94^\circ$;
- negative peak about $-81.16^\circ$;
- symmetric span about $162.32^\circ$;
- maximum lateral offset about 0.00902 m;
- maximum alignment about 0.04607 against a 0.050 limit;
- minimum sampled spring tension about 0.209 N;
- deterministic raw return about 7,016.89;
- native/ONNX maximum action difference about $1.132\times10^{-6}$.

The selected BC/DAgger initializer already passed. The final audit's saved initial-policy trace reports about $154.62^\circ$ span and raw return 6,126.39. PPO raised the same nominal seed's return by about 890.50, or 14.5%, and raised the span to $162.32^\circ$. Those are refinement measurements, not acquisition credit.

The seeds are nearly identical because the nominal Swing reset has no enabled domain randomization, observation noise, action delay, random yaw, or nonzero start bounds. Repeating deterministic dynamics checks reproducibility and pipeline consistency; it does not estimate a broad success probability.

### Why the 1,048,576-transition continuation is rejected

Swing v2 used the same teacher-assisted family and trained to 1,048,576 transitions. Its 524,288 checkpoint passed, but its final checkpoint failed:

- bidirectional span remained large, about $160.03^\circ$;
- valid geometry fraction fell to about 0.9475;
- maximum alignment reached about 0.1633, over three times the 0.050 limit;
- both-strings-tensioned fraction was about 0.99917, meaning one of 1,200 samples failed;
- minimum sampled spring tension was 0 N;
- deterministic raw return became about -711.97 under the physical evaluation.

This is not "almost a pass." The gate is fail-closed and requires every sampled step. Large amplitude does not compensate for invalid geometry or lost tension.

The million-transition failure also explains the v3 lineage: v3 did **not** continue from the bad v2 final checkpoint or copy back the v2 524k snapshot as its final result. It repeated the teacher-assisted initialization and declared a 524,288-transition budget, then required a fresh evaluation and audit. The shorter schedule was selected from evidence that further PPO updates could damage the skill.

## Complete Staged Pipeline

### What is runnable and what is conceptual

Code blocks marked `bash` below are repository commands. They perform real supervised training, PPO, evaluation, rendering, or audit and require the documented environment.

Short Python snippets earlier in the chapter are conceptual transcriptions. They explain equations and indexing but omit imports, environment construction, budgets, hashing, and overwrite guards. Use `rlx/scripts/bootstrap_swing_e2e.py` and `rlx/examples/ppo_microduck_studio.py` as the runnable implementations.

### Safe reproduction

Run from the repository root. Use one fresh timestamp in every matching path. The fail-closed block requires both the bootstrap output and the entire destination run directory to be new; any failed command stops the block before subsequent copies.

```bash
(
set -eu
test ! -e /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS
test ! -e rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS
rlx/.venv-microduck/bin/python rlx/scripts/bootstrap_swing_e2e.py \
  --source docs/remaining-scenarios-e2e/recipes/swing-teacher.json \
  --output /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS

test -f /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/swing.safetensors
test -f /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/swing.safetensors.json
test -f /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/provenance.json
mkdir -p rlx/runs/studio/swing
mkdir rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS
cp /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/swing.safetensors \
  /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/swing.safetensors.json \
  rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS/
)
```

Before PPO, inspect:

```bash
jq '{selected_stage, samples, normalizer_frozen_sample_count, limitations}' \
  /tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS/provenance.json

jq '.metadata | {steps, ppo_steps, dagger_iteration, teacher_samples, critic}' \
  rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS/swing.safetensors.json
```

The expected semantics are `steps: 0`, `ppo_steps: 0`, teacher assistance declared, and a random critic. Do not proceed from an initializer whose strict evaluation failed merely because its BC loss is low.

With Studio already running at `http://127.0.0.1:63317`, execute the tracked recipe:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs \
  --execute --base-url http://127.0.0.1:63317 \
  --experiment swing \
  --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json \
  --run swing-reproduction-YYYYMMDD-HHMMSS \
  --report rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-api.json \
  --timeout-seconds 7200
```

Then run the independent physical audit:

```bash
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py \
  --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json \
  --run rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS \
  --output rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-audit \
  --seeds 501 502 503 504 505 --render
```

For browser/video verification, follow the complete guarded Swing procedure in `docs/microduck-training-student-book/REPRODUCTION.md`. Its `PLAYWRIGHT_PACKAGE` default is workstation-specific and must be set appropriately elsewhere.

Important stop rules:

- a nonzero bootstrap or training exit means the stage did not complete;
- a nonzero audit exit is evidence of a failed gate and must remain visible;
- do not copy a checkpoint when its sidecar or provenance is missing;
- do not reinterpret a completed optimizer run as a successful skill;
- do not resume from the failed million-transition final policy;
- do not claim hardware safety, BAM equivalence, or robustness from nominal XML evidence.

## Source-Grounded Limits and Unknowns

The repository and saved artifacts establish the exact counts and outcomes reported above. They do **not** establish:

- the size or search algorithm of the earlier controller-parameter search population; `swing-teacher.json` preserves only the selected 13 values and a source-evidence hash;
- a BC train/validation generalization gap, because no held-out supervised split is recorded;
- robustness to randomized starts, disturbances, sensor noise, delay, actuator mismatch, or domain randomization;
- continuous-time tendon tension between 50 Hz samples;
- hardware-safe loads or sim-to-real transfer;
- five independent training seeds; the five audit seeds replay essentially identical nominal dynamics;
- scratch-PPO impossibility in principle; only failure of the recorded scratch recipe is shown.

Unknown quantities must remain unknown rather than being reconstructed from filenames or inferred from the final pass.

## Read the saved Swing result in Studio

![Actual saved Swing verification panel. The UI explicitly discloses BC/DAgger acquisition and PPO refinement, and displays bidirectional span separately from geometry and tension.](assets/studio-swing-summary.png)

Select **Self-pumped swing** and restore the historical run with **Saved runs**. Verify the 1,200-step still-start horizon, 150-degree bidirectional target and evaluated source hash. Inspect all geometry/tension rows rather than accepting the largest angular peak as a complete result. The screenshot's sixteen nominal API episodes and the separate five audit seeds do not imply randomized robustness. The saved video is the final mean actor—not the privileged teacher or an animation. This read-only UI capture launches neither imitation nor PPO training.

## Exercises

**Easy: reconstruct the teacher action.** Given the saved 13 parameters, $\theta=0.1$ rad, and $\dot\theta=-0.2$ rad/s, compute $q$, the five clipped sagittal values, and all 14 action entries.

Acceptance check: six specified action indices are zero; the right hip-pitch, knee, and ankle entries negate the first three left sagittal entries; all outputs are in $[-1,1]$.

**Easy: label the observation blocks.** Write the half-open slice for each of the eight 61D blocks and identify which values are Swing-specific.

Acceptance check: no extra angle/rate block is invented; the Swing command occupies `48:51`; head and body command blocks remain present and zero.

**Intermediate: verify dataset arithmetic.** Derive the cumulative sample count and BC optimizer-update count for every stage.

Acceptance check: use eight episodes times 1,200 samples per collection stage and `ceil(N/512) * 80` updates; explain why stage 3 uses 38,400 rows while the final NPZ contains 57,600.

**Intermediate: explain beta honestly.** Compare a textbook decaying-mixture DAgger schedule with the implementation's collection policy.

Acceptance check: report effective beta 1 for initial collection and 0 for every DAgger collection; distinguish executed student action from retained teacher label.

**Intermediate: audit imitation-to-RL state.** Make a table for actor mean, critic, `actor_log_std`, observation normalizer, return normalizer, and optimizer.

Acceptance check: actor mean is pretrained; critic is random; log standard deviation is inherited at $\log(0.1)$; observation statistics are inherited and frozen; return statistics start at the bootstrap defaults and update; PPO Adam is fresh.

**Intermediate: derive PPO work counts.** Starting from 16 environments, 256 steps, four minibatches, two epochs, and 524,288 transitions, compute rollout size, number of updates, minibatch size, and optimizer minibatches.

Acceptance check: 4,096 transitions per rollout, 128 updates, 1,024 rows per minibatch, and 1,024 optimizer minibatches.

**Advanced: parse telemetry without mixing scales.** Read `training-metrics.jsonl` and separately aggregate collection reward, completed raw episode return, KL, clip fraction, and explained variance.

Acceptance check: preserve the distinction between normalized stochastic per-transition reward, raw stochastic episode return, and deterministic evaluation return.

**Advanced: implement the strict span metric.** Given an angle series, compute positive peak, negative peak, peak-to-peak span, and symmetric bidirectional span. Reject incomplete episodes and missing metrics.

Acceptance check: $+90^\circ/-40^\circ$ gives $130^\circ$ peak-to-peak and $80^\circ$ bidirectional span; peaks from different episodes are never combined.

**Advanced: select a checkpoint.** Compare stages 2-5 using BC loss, strict pass, geometry fraction, tension fraction, and span.

Acceptance check: choose stage 3 even though later stages have lower BC loss; explain the lexicographic selection key and why one zero-tension sample fails stage 2.

**Advanced: diagnose the bad continuation.** Compare the 524,288-transition pass with the 1,048,576-transition final policy.

Acceptance check: do not call the longer policy better because its amplitude remains high; identify alignment, geometry fraction, and zero-tension evidence; recommend a fresh bounded rerun or checkpoint selection rather than continuing the failed final weights.

**Research extension: design a robustness study without claiming results.** Propose randomized initial angle/rate, observation noise, delay, actuator variation, and multiple training seeds.

Acceptance check: distinguish acquisition experiments, checkpoint selection data, and held-out robustness tests; predeclare the physical gate and report all failures.
