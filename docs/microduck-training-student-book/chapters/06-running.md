# Running: an aerial gait with intended progress

## Learning goals

After this chapter, you should be able to:

1. distinguish running from fast grounded stepping using contact-state evidence;
2. trace the exact 61-input observation and 14-output action through the
   Running environment;
3. derive every selected positive and penalty reward term;
4. explain which built-in curricula are active and which are bypassed by the
   fixed-command recipe;
5. count the base and continuation PPO work exactly; and
6. interpret the measured pass without claiming straight-line perfection or
   hardware readiness.

![Running experiment flow](assets/running-flow.svg)

## Case-study question

For this experiment, "running" means more than moving quickly. A passing policy
must:

- survive the full 12-second episode;
- make signed progress along the intended command direction;
- repeatedly exchange exclusive support between feet;
- lift and land both feet; and
- spend a declared fraction of samples with both feet off the ground.

The tracked command is

$$
\mathbf c=(0.75,\ 0,\ 0),
$$

or 0.75 m/s forward with no lateral or yaw-rate request. The selected physics
is nominal XML simulation: no domain randomization, observation noise, action
delay, or random initial yaw.

The evidence supports:

> One deterministic actor produced an aerial command-directed gait on five
> reset seeds under the selected 0.75 m/s nominal simulator condition.

It does not establish arbitrary-speed running, commanded turning, disturbance
recovery, energy efficiency, or safe hardware behavior.

## Why Running is not a faster Stilt task

Running and Stilt Walking share the tensor contract and target-offset action
interface, but their optimization problems differ:

| Aspect | Running | Stilt Walking |
| --- | --- | --- |
| Contact geometry | original feet | generated stilts |
| Command | 0.75 m/s | 0.25 m/s |
| Reward stack | dedicated `run` behavior | reweighted walking reward |
| Posture model | speed-regime leg tolerances | fixed walking Gaussian |
| Air-time window | 0.15-0.35 s | 0.125-0.300 s |
| Both-feet flight reward | explicit optional term | none |
| Both-feet flight gate | at least 3% | none |
| Episode | 600 steps | 500 steps |

Running requires dynamic lean and roll/pitch motion. Its upright Gaussian is
therefore broader than walking's. It also uses a speed-dependent pose width and
an explicit both-feet-flight feature. Simply raising the walking command would
retain incentives that can favor grounded stepping.

## Reset, command handling, and active curricula

Every episode starts from STAND with independent joint perturbations
$U(-0.03,0.03)$ rad, root-height perturbation $U(0,0.01)$ m, zero velocity, and
yaw zero because `randomYaw` is disabled.

The reusable `run` behavior contains a command curriculum:

- a standing fraction that rises from 0.02 to 0.10;
- a 55% straight-forward bucket;
- an omnidirectional remainder; and
- a staged speed ceiling from 0.4 to 1.1 m/s.

That curriculum is **not active in the selected E2E experiment**.
`_pin_locomotion_forward_command` calls the native sampler and then overwrites
the twist command with `[0.75, 0, 0]`. The head/body keep-alive commands remain
sampled. The tracked policy therefore specializes to one speed and direction.

One reward-internal curriculum remains active: `smooth_moves` calls
`_run_action_rate_pen`, whose internal coefficient is scheduled by
per-environment lifetime steps. With the default stage scale:

| Per-env steps | Total transitions at 16 envs | Internal coefficient | Effective after recipe weight 0.1 |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.1 | 0.01 |
| 250,000 | 4,000,000 | 0.2 | 0.02 |
| 375,000 | 6,000,000 | 0.3 | 0.03 |
| 562,500 | 9,000,000 | 0.4 | 0.04 |
| 750,000 | 12,000,000 | 0.5 | 0.05 |

The 6,000,640-transition base reaches the 0.3 rung only at its end. The
continuation process starts a new environment lifetime counter and runs only
2,097,152 transitions, so its effective smoothness coefficient remains 0.01.
The checkpoint restores policy and normalization state, not this environment
counter.

## Exact policy input and output

The raw observation is:

$$
o_t=[
\boldsymbol\omega_b(3),
\mathbf g_b(3),
\Delta\mathbf q(14),
\dot{\mathbf q}_{t-1}(14),
\mathbf a_{t-1}(14),
\mathbf c_{\text{twist}}(3),
\mathbf c_{\text{head}}(4),
\mathbf c_{\text{body}}(6)
]\in\mathbb R^{61}.
$$

Absolute world position and absolute yaw are absent. Yaw *rate* is present in
the gyro, and commanded yaw rate is present at index 50. This distinction is
why `yaw_tracking` is learnable while an absolute-heading reward is not.

The following is the **actual first raw observation** from
`running-v4-audit/trained-seed501.npz`:

```text
gyro:
[0.000000, 0.000000, 0.000000]
projected_gravity:
[0.000000, 0.000000, -1.000000]
joint_pos_rel:
[ 0.029786,  0.008408, -0.004889,  0.003520,  0.021663,
  0.029569, -0.025637,  0.019733, -0.020873,
  0.013535, -0.002452,  0.020996, -0.011165, -0.006701]
lagged_joint_vel:
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
 0.000000, 0.000000]
last_action:
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
 0.000000, 0.000000]
twist_cmd:
[0.750000, 0.000000, 0.000000]
head_cmd:
[0.037858, 0.021488, -0.049252, 0.009214]
body_cmd:
[-0.000709, 0.001986, 0.003957, 0.041533, -0.012620, -0.029673]
```

The matching **actual deterministic actor output** is:

```text
[ 2.288563,  0.807669,  3.515622, -0.709707, -6.582819,
  2.529023,  0.497528,  9.825796,  3.089725,
  6.087962, -3.619488, -9.271533, -8.283561, -0.033699]
```

These are raw actor means. Several exceed $\pm4$. The environment stores the
raw vector as `last_action` and uses it in the smoothness penalty, but clips
the applied offset:

$$
q_t^{\text{target}}
=q^{\text{default}}+\operatorname{clip}(a_t,-4,4).
$$

This recorded vector is therefore also concrete evidence that "14-action
space" does not mean the network output is tanh-bounded. XML actuator dynamics
convert target positions into physical motion; the actor does not command
torque directly.

The raw observation is normalized before the network. The exported ONNX actor
contains the observation normalizer, so evaluation supplies the raw 61-vector
and does not normalize it a second time.

## Reward trace: all effective selected terms

The recipe explicitly supplies:

```json
{
  "keep_pace": 12,
  "track_turn": 2,
  "flight": 4,
  "air_time": 1,
  "stay_upright": 2,
  "pose": 0.1,
  "head_up": 1,
  "foot_clearance": 0.5,
  "smooth_moves": 0.1,
  "plant_the_foot": 0.05,
  "yaw_tracking": 8
}
```

Because weight overrides replace only named terms, two default `run` terms are
also active:

```json
{
  "no_limit_parking": 1,
  "calm_roll": 0.025
}
```

The total selected reward is the sum of the following contributions.

### `keep_pace`: baseline-subtracted body-frame speed

Let

$$
e_v=(c_x-v_x)^2+(c_y-v_y)^2+v_z^2,
\qquad
b=\exp[-(c_x^2+c_y^2)/0.1].
$$

For a nonzero translational command,

$$
f_{\text{pace}}
=\frac{\max(0,\exp(-e_v/0.1)-b)}
{\max(1-b,10^{-6})},
$$

$$
r_{\text{pace}}=12f_{\text{pace}}.
$$

The subtraction makes standing worth zero at nonzero command while retaining a
gradient toward the command. Body-frame velocity is computed by inverse
rotation of world base velocity using the trunk quaternion. It is not the
MuJoCo COM-inertial local frame.

### `track_turn`: combined yaw and roll/pitch rate

$$
f_{\text{turn}}
=\exp\left[
-\frac{(c_\omega-\omega_z)^2+\omega_x^2+\omega_y^2}{0.5}
\right],
$$

$$
r_{\text{turn}}=2f_{\text{turn}}.
$$

Running v3 used weight 8. The continuation reduces it to 2 because this
combined term penalizes roll and pitch motion that a dynamic gait may need.

### `yaw_tracking`: focused observable yaw-rate control

$$
f_{\text{yaw}}
=\exp\left[-\left(\frac{\omega_z-c_\omega}{0.25}\right)^2\right],
\qquad
r_{\text{yaw}}=8f_{\text{yaw}}.
$$

This term uses only gyro yaw rate and the observable yaw-rate command. It
cannot directly know whether accumulated world heading has drifted.

### `air_time`: per-foot flight window

For each foot, an air timer resets on contact and advances by 0.02 s off
contact. While a motion command is active,

$$
f_{\text{air}}=
\sum_{f\in\{L,R\}}\mathbb 1[0.15<\tau_f<0.35],
\qquad
r_{\text{air}}=f_{\text{air}}.
$$

The value can be 0, 1, or 2 each step.

### `flight`: both feet airborne, upright, and moving as commanded

Let $L$ and $R$ denote contact and
$v_\parallel=\mathbf v_{b,xy}\cdot\hat{\mathbf c}_{xy}$.

$$
f_{\text{flight}}
=\mathbb 1[\neg L\land\neg R]
\operatorname{clip}\left(\frac{u-0.8}{0.2},0,1\right)
\operatorname{clip}\left(\frac{v_\parallel}{0.4},0,1\right),
$$

where $u=-g_z$. The contribution is

$$
r_{\text{flight}}=4f_{\text{flight}}.
$$

The multiplicative gates make stationary hopping, sideways flight, and an
airborne fall poor solutions.

### `stay_upright`: broad dynamic posture Gaussian

$$
r_{\text{upright}}
=2\exp[-(g_x^2+g_y^2)/0.12].
$$

The denominator 0.12 is broader than the walking value 0.05, so modest running
lean is less heavily taxed.

### `pose`: speed-regime leg posture

For the ten leg joints,

$$
f_{\text{pose}}
=\exp\left[-\frac1{10}\sum_j
\left(\frac{\Delta q_j}{\sigma_j}\right)^2\right].
$$

At the selected command norm 0.75, the running-regime widths are used:

```text
[0.4, 0.08, 0.6, 0.6, 0.4,
 0.4, 0.08, 0.6, 0.6, 0.4]
```

corresponding to yaw, roll, pitch, knee, and ankle for each leg. The selected
contribution is $0.1f_{\text{pose}}$.

### `head_up`: head command tracking

$$
f_{\text{head}}
=\frac14\sum_{j\in\text{head}}
\exp\left[-\left(\frac{\Delta q_j-h_j^{\text{cmd}}}{0.5}\right)^2\right],
$$

$$
r_{\text{head}}=f_{\text{head}}.
$$

The low selected weight 1 replaces the behavior default 3.5.

### `foot_clearance`: moving-foot height cost

For each foot geom, let $h_f$ be height above its calibrated planted height and
$s_f$ horizontal world speed:

$$
f_{\text{clear}}
=-\sum_f |h_f-0.03|s_f,
\qquad
r_{\text{clear}}=0.5f_{\text{clear}}.
$$

The term is zero when no motion command is active. It is a penalty, not a
Gaussian reward.

### `plant_the_foot`: contact slip cost

For contacting feet,

$$
f_{\text{slip}}
=-\sum_{f\text{ in contact}}(v_{f,x}^2+v_{f,y}^2),
\qquad
r_{\text{slip}}=0.05f_{\text{slip}}.
$$

### `smooth_moves`: scheduled raw-action change cost

$$
f_{\text{smooth}}
=-\alpha(n)\sum_j(a_{t,j}-a_{t-1,j})^2,
\qquad
r_{\text{smooth}}=0.1f_{\text{smooth}}.
$$

The exact active schedule was listed above. Because raw actions are used, large
saturated requests remain costly even when applied offsets clip at $\pm4$.

### `no_limit_parking`: hard-stop avoidance

For each joint, normalize position within its compile-time range:

$$
z_j=\frac{|q_j-\text{mid}_j|}{\text{halfRange}_j},
\qquad
p_j=\max(0,z_j-0.88).
$$

Then

$$
r_{\text{limit}}=-25\sum_jp_j^2.
$$

This inherited term has effective weight 1.

### `calm_roll`: body roll/pitch rate penalty

$$
r_{\text{calm}}
=-0.025(\omega_x^2+\omega_y^2).
$$

All penalty functions are non-positive before multiplication by non-negative
weights.

![Running reward and checkpoint evidence](assets/running-reward.png)

## PPO applied to Running

The actor and critic use 512, 256, and 128-unit ELU hidden layers. Collection
uses stochastic Gaussian actions; export uses the deterministic mean.

Both selected stages use:

| PPO quantity | Value |
| --- | ---: |
| Environments | 16 |
| Steps per environment per rollout | 128 |
| Transitions per rollout | 2,048 |
| Minibatches | 4 |
| Samples per minibatch | 512 |
| Update epochs | 4 |
| Optimizer minibatches per rollout | 16 |
| $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Ratio/value clip | 0.2 |
| Value coefficient | 1.0 |
| Entropy coefficient | 0 |
| Max gradient norm | 0.5 |

### How an aerial transition affects PPO

Suppose a sampled action produces both-feet flight while upright and moving
forward. It can increase `flight`, `air_time`, and `keep_pace`, but may also
increase `smooth_moves`, `calm_roll`, or clearance cost. GAE combines the
immediate net reward with later value estimates. If the resulting advantage is
positive, PPO increases that sampled action's likelihood, subject to the
clipped ratio. If the aerial phase leads to a fall, later lost reward and a
non-bootstrapped terminal value can make its advantage negative.

This is why "reward flight" does not mean "every airborne action is reinforced."
The full temporal outcome matters.

The following is a **conceptual calculation only**, not the repository
optimizer:

```python
# Conceptual only: one sample's PPO policy term.
ratio = exp(clip(new_logp - old_logp, -20, 20))
surrogate = ratio * normalized_advantage
bounded = clip(ratio, 0.8, 1.2) * normalized_advantage
policy_loss_sample = -min(surrogate, bounded)
```

The actual RLX code evaluates minibatches, uses a clipped Huber critic loss,
checks loss/gradient/parameter finiteness, and applies gradient-norm clipping.

### Exact stage counts and restored state

| Stage | Transitions | Rollout/update cycles | Optimizer minibatches | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| Running v3 base | 6,000,640 | 2,930 | 46,880 | $3\times10^{-4}$ |
| Running v4 continuation | 2,097,152 | 1,024 | 16,384 | $1\times10^{-4}$ |
| Entire selected lineage | 8,097,792 | 3,954 | 63,264 | two stages |

The base starts with randomly initialized network weights and standard
deviation 0.6. The continuation loads:

- actor and critic parameters;
- learned actor log standard deviation;
- observation mean, variance, and count; and
- reward-return normalization statistics.

It does not restore Adam optimizer moments or the behavior environment's
lifetime curriculum counter. `initialStd: 0.6` is passed on the command line but
does not replace the loaded actor's learned log standard deviation.

Running v4 changes the objective from combined angular weight 8 to 2 and adds
yaw-only weight 8. It does not change physics, command, or acceptance floors.

## Exact physical acceptance criteria

For commanded samples,

$$
\hat{\mathbf c}_t=\frac{\mathbf c_t}{\|\mathbf c_t\|},
\qquad
\bar v_\parallel=
\frac1{N_c}\sum_{t\in C}\mathbf v_{h,t}\cdot\hat{\mathbf c}_t,
$$

$$
\rho_v=
\frac{\sum_{t\in C}\mathbf v_{h,t}\cdot\hat{\mathbf c}_t}
{\sum_{t\in C}\|\mathbf c_t\|}.
$$

For signed displacement, evaluator version 2 anchors an intended world heading
when the twist command changes, advances it only by commanded yaw, projects
each world-position increment onto that intended direction, and sums:

$$
d_\parallel
=\sum_{t=2}^{T}
(\mathbf p_t-\mathbf p_{t-1})
\cdot\hat{\mathbf d}^{\,\text{intended}}_t.
$$

A policy can therefore have high body-frame speed and still fail if it turns
away from the intended path.

Every episode must satisfy:

| Criterion | Required value |
| --- | ---: |
| Complete horizon | 600 samples, truncated at 12 s without termination |
| Valid metric coverage | all samples finite; contacts binary; heading normalized |
| Upright sample | at least 0.9 |
| Upright fraction | at least 0.95 |
| Active translational command | at least 0.1 m/s |
| Commanded fraction | at least 0.50 |
| Mean command-directed speed | at least 0.35 m/s |
| Signed intended displacement | at least 2.0 m |
| Speed tracking ratio | at least 0.35 |
| Alternating exclusive-support switches | at least 8 |
| Air fraction for each foot | at least 0.08 |
| Contact fraction for each foot | at least 0.10 |
| Foot events | both feet lift off and touch down |
| Both-feet aerial fraction | at least 0.03 |

## Measured evidence and margins

The selected audit produced:

| Seed | Directed speed | Displacement | Speed ratio | Aerial fraction | Support switches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 501 | 0.7622 | 6.2711 | 1.0162 | 0.3867 | 167 |
| 502 | 0.7746 | 6.4911 | 1.0329 | 0.3983 | 168 |
| 503 | 0.7383 | 3.8483 | 0.9844 | 0.3900 | 169 |
| 504 | 0.7600 | 5.6116 | 1.0134 | 0.3500 | 170 |
| 505 | 0.7644 | 9.0130 | 1.0192 | 0.3983 | 167 |

All five completed 600 steps with upright fraction 1.0. Each foot had both
substantial air time and substantial contact time. The aerial fractions are
more than ten times the 0.03 floor.

Zero control fell after 44-49 steps; random control fell after 43-45 steps.
Their short release dynamics produced about 0.125-0.144 m/s directed speed, but
they failed the complete physical gate.

### What the continuation changed

On comparison seed 501:

| Continuation transitions | Raw return | Skill | Key outcome |
| ---: | ---: | --- | --- |
| 0 | 9,835.52 | FAIL | displacement $-0.2392$ m |
| 524,288 | 11,023.29 | PASS | displacement $4.7544$ m |
| 1,048,576 | 11,253.39 | PASS | displacement $8.5150$ m |
| 1,572,864 | 11,617.39 | PASS | displacement $6.2940$ m |
| 2,097,152 | 11,467.54 | PASS | displacement $5.7800$ m |

The initializer was already aerial: aerial fraction 0.275 with 156 support
switches and mean directed speed 0.6664 m/s. It failed only intended
displacement. Running v4 is therefore a steering/refinement case, not fresh
discovery of running.

The final raw-return change versus initialization was 1,683.18 under the final
reward configuration. The final return is lower than the preceding checkpoint;
that dip is retained. Model selection is based on the declared physical gate,
not a requirement that every scalar rise monotonically.

Training recorded 1,024 finite rollout/update cycles and 16,384 optimizer
minibatches. Last-100 means included approximate KL 0.0308, clip fraction
0.3393, and explained variance 0.5853. Maximum approximate KL was 2.8238, a
transient worth noticing but not, by itself, a failure verdict.

The ONNX audit reported `batch x 61` input, `batch x 14` output, native/ONNX
maximum absolute action error $1.431\times10^{-6}$, and parameter L2 change
7.6725 from the loaded initializer.

![Running physical tracking](assets/running-tracking.png)

## Failure history and reward hacking

### Running v1: recipe evidence only

Running v1 did not produce an accepted skill result.

### Running v2: fast grounded stepping

Running v2 learned speed but failed the aerial requirement. This is the central
classification lesson: a gait can be fast, alternating, and upright while
remaining walking.

### Running v3: aerial gait, wrong world progress

Running v3 acquired aerial phases around 0.7 m/s, but three of five recorded
episodes failed signed intended displacement. The actor's local velocity could
look correct while accumulated yaw produced a curved world path.

The v4 response was deliberately observable:

- reduce the combined roll/pitch/yaw Gaussian from 8 to 2;
- add yaw-rate-only tracking at weight 8; and
- keep the strict evaluator unchanged.

Adding absolute heading to the reward would score a hidden variable. Lowering
the displacement floor would redefine the task around the failure.

### Exploit map

| Candidate behavior | Reward weakness | Independent rejection |
| --- | --- | --- |
| fast grounded walk | speed and alternating support | aerial fraction |
| stationary hopping | both feet airborne | flight speed gate and displacement |
| sideways or backward flight | airborne motion | command-directed projection |
| circular run | local speed remains high | signed intended displacement |
| face-forward dive | transient speed | upright and full horizon |
| joint saturation | applied target clips | raw-action smoothness and limit parking |

## Reading reward and PPO diagnostics

![Running PPO losses](assets/running-loss.png)

Normalized stochastic collection reward, raw completed-episode return, and
deterministic checkpoint return are different measurements. Reward
normalization changes scale over training. Sampled actions can terminate early.
The deterministic audit uses the actor mean and raw reward.

Policy loss is batch-relative. Approximate KL measures policy movement on the
sampled batch. Clip fraction reports the fraction of probability ratios outside
the unclipped interval. Value loss and explained variance describe critic fit.
None establishes running without contact and trajectory evidence.

## Reproducing the tracked result safely

The continuation cannot discover the base checkpoint from another run. Studio
requires the checkpoint to exist in the continuation run directory before it
adds `--init-from`.

Use the complete guarded **Running: all stages** procedure in
`docs/microduck-training-student-book/REPRODUCTION.md`. It is intentionally
long because it verifies:

1. fresh non-overlapping paths;
2. the base API report identity;
3. successful base training;
4. only the specifically recognized base skill-failure outcome;
5. successful render/export evidence;
6. both checkpoint and normalization sidecar; and
7. a copy into a distinct continuation run.

Then run the appendix's continuation, five-seed audit, and video verifier
commands unchanged. Keep a failed audit as a failed artifact.

The first two post-guard commands are reproduced here with the same valid
syntax; replace the timestamp consistently:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute \
  --base-url http://127.0.0.1:63317 --experiment running \
  --recipe-json docs/remaining-scenarios-e2e/recipes/running.json \
  --run running-reproduction-YYYYMMDD-HHMMSS \
  --report rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-api.json \
  --timeout-seconds 7200

rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py \
  --recipe-json docs/remaining-scenarios-e2e/recipes/running.json \
  --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS \
  --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit \
  --seeds 501 502 503 504 505 --render
```

Run the appendix's matching `verify-rlx-video.mjs` command afterward; it also
requires `PLAYWRIGHT_PACKAGE` to resolve as described at the top of the
appendix.

This is a **conceptual sequence, not executable shell syntax**:

```text
running-base.json
  -> 6,000,640 fresh PPO transitions
  -> preserve failed base skill verdict
  -> guarded checkpoint + sidecar handoff
running.json
  -> 2,097,152 continuation transitions
  -> deterministic seeds 501..505
  -> negative controls
  -> ONNX parity, video, and browser checks
```

Do not shorten the process to "copy an old running.safetensors and run v4."
That would lose provenance and may pair weights with the wrong normalizer.

## Simulator versus hardware

The nominal result omits:

- BAM actuator dynamics, current limits, and communication latency;
- randomized mass, friction, noise, delay, and yaw;
- terrain variation, impacts, pushes, and recovery;
- battery sag, thermal behavior, cable forces, backlash, and wear;
- command-speed or turning generalization;
- hardware fall protection and emergency-stop validation.

The raw actor output also demonstrates substantial target saturation on the
first recorded step. That is not automatically unsafe in XML simulation, but
it is a reason not to treat the local pass as a hardware command-quality
certificate.

The appropriate next step is to port the environment and reward design to the
official `microduck_rl` mjlab/BAM stack, retrain with declared randomization and
larger-scale sampling, export through the official path, and validate on
hardware under a separate safety protocol.

## Read the saved Running result in Studio

![Actual saved Running verification panel. Speed, progress, aerial contacts and full-episode acceptance must agree; a large reward alone is insufficient.](assets/studio-running-summary.png)

Select **Fast running** and use **Saved runs** to restore the historical recipe. Check that the saved command is 0.75 m/s and the complete evaluation lasts twelve seconds. Read the physical metric ranges and source hash before opening the video. Sixteen API episodes and five independent audit trials are distinct evidence sets. Watch the path as well as the feet: the passing result includes aerial phases but still curves, so this screenshot must not be described as proof of straight-line path following. No new training is performed by the capture procedure.

## Exercises

1. **Classify a gait.** A policy reaches 0.72 m/s and completes 600 steps, but
   its aerial fraction is 0.01. List the exact failed gate.
2. **Decode observability.** Explain why yaw-rate tracking is learnable from
   indices `0:3` and `48:51`, while absolute-heading error is not.
3. **Apply clipping.** Convert the recorded first actor output into the applied
   clipped offset vector. Which entries saturate?
4. **Compute a flight reward.** Evaluate `flight` for no contacts, upright
   0.9, and directed speed 0.2 m/s.
5. **Count PPO reuse.** Explain why 2,048 collected transitions create 16
   optimizer minibatches without becoming 32,768 new physical transitions.
6. **Interpret the lineage.** State separately what Running v3 acquired and
   what Running v4 improved.
7. **Audit a circle.** Construct a trajectory with high body-frame speed and
   near-zero signed intended displacement.
8. **Design held-out commands.** Predeclare speeds and yaw-rate commands, then
   specify whether you are testing interpolation, extrapolation, or a new task.

Command curricula, turning, randomization, pushes, energy constraints, and
hardware deployment are future experiments. They must earn their own audit
results rather than inheriting the nominal v4 pass.
