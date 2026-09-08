# Parameters and the Microduck Studio UI

## What a Studio recipe controls

Microduck Studio turns a browser form into a command for
`rlx/examples/ppo_microduck_studio.py`. The browser sends a recipe object to
`POST /api/rlx`; `duck-viewer/lib/rlx-job.ts` validates and normalizes that
object, translates it into command-line arguments, and starts one of four
operations: `train`, `eval`, `render`, or `export`.

This distinction matters because the value visible in an input is a
**requested value**, while the value returned by the API is the
**normalized value**. The value that finally affects an operation is the
**effective value**. These can differ:

* Numeric inputs may be clamped to a supported range.
* `totalTimesteps` is rounded up to a whole rollout batch.
* The Pipeline smoke profile forces randomization off and forces one PPO
  update epoch, even if the submitted object contains other values.
* Render always disables domain randomization, observation noise, action
  delay, and random yaw.
* Swing evaluation always starts still. Training assistance does not carry
  into a skill evaluation.
* Resuming loads model and normalization state from a checkpoint. Some
  requested initialization values therefore no longer initialize anything.

The API response is the first authoritative record of the normalized recipe:

```json
{
  "accepted": true,
  "action": "train",
  "recipe": {
    "...": "the complete normalized RlxRecipe"
  }
}
```

The saved checkpoint sidecar is the authoritative record of the Python
training process. A saved `evaluation.json` additionally records
`evaluation_request`, including the normalized recipe, evaluation mode,
horizon, and Swing target. Do not reconstruct an experiment later from a
screenshot or from remembered slider positions.

![Microduck Studio overview: experiment selection, live simulation, training, evaluation, and artifacts](assets/studio-overview.png)

The screenshots inspect saved accepted runs, but the recipe form displays the current UI state. Entering an old run name retrieves its artifacts; it does not reconstruct every historical training parameter. Use the versioned JSON recipe and checkpoint sidecar, not the screenshot values, to reproduce the measured run.

## Select or create a run

Studio opens on the **Experiments** workspace. The first major table is headed
**Choose what the duck should learn** and offers four radio-style rows:
**Dance imitation**, **Self-pumped swing**, **Fast running**, and
**Stilt walking**.

To select an existing run:

1. Select the experiment row. This changes the active recipe and resets the
   form to that experiment's defaults.
2. In **Microduck PPO Recipe**, enter the existing run's name exactly in
   **Run name**.
3. Wait for the status poll to populate the checkpoint, ONNX, evaluation,
   render, and metadata indicators for that experiment/run pair.
4. Use the existing artifacts for evaluation or rendering. Do not select
   **Start RLX** unless you intend to replace the run or explicitly continue
   its checkpoint.

To create a run:

1. Select an experiment.
2. Select **New experiment**. Studio generates a name such as
   `running-143012` and scrolls to the recipe.
3. Replace that generated name with a descriptive identifier if needed.
4. Select **Pipeline smoke** first.
5. Expand **Advanced PPO and environment settings** to inspect the visible
   parameters and reward weights.
6. Select **Start RLX**.

`runName` is lowercased, spaces and unsupported characters become hyphens,
leading and trailing hyphens are removed, and the result is truncated to 48
characters. An empty result is rejected. Artifacts are owned by
`rlx/runs/studio/<experimentId>/<runName>/`; choosing the same experiment and
run name addresses the same files.

## Profiles: smoke is wiring, full is learning

The profile buttons are policy presets, not merely labels.

| Profile | Defaults and enforced settings | Meaning |
|---|---|---|
| `smoke` | Defaults: 2 envs, 2 rollout steps, 1 minibatch and 4 total timesteps. Enforced: 1 update epoch and all four randomization flags off. | Verifies process launch, environment stepping, PPO update, checkpoint, ONNX, evaluation, and render plumbing. It does not demonstrate a learned skill. |
| `full` | Scenario defaults, normally 16 envs; 24 rollout steps except Swing 64; 4 minibatches; scenario PPO defaults; randomization on unless explicitly disabled | Performs an actual training experiment. Full evaluation uses skill criteria rather than pipeline-only validity. |

Valid explicit API values can override smoke's environment count, rollout steps, minibatches, and budget. Those values are defaults, not hard limits; use the normalized response to see the actual batch. Only the stated enforced settings are forced by the smoke profile.

The full training budgets selected by **Default full** are 1,000,000 steps for
Dance and 4,000,000 steps for Swing, Running, and Stilts. These are experiment
budgets, not guarantees of success.

The rollout batch is

$$
B = N_{\text{env}}N_{\text{step}}.
$$

Studio normalizes the requested training budget to

$$
T_{\text{effective}} =
\max\left(B,\left\lceil\frac{T_{\text{requested}}}{B}\right\rceil B\right).
$$

For example, 16 environments and 24 rollout steps produce a batch of 384
transitions. A request for 1,000,000 steps becomes 1,000,320 so that PPO never
ends with a partial rollout.

## Identity, duration, and artifact fields

The following table covers every recipe field concerned with identity,
profiles, time horizons, and saved outputs. “Default” means the value chosen
by `normalizeRecipe` when an API caller omits the field.

| Field | Valid/default value | Purpose and effect |
|---|---|---|
| `experimentId` | `dance`, `swing`, `running`, or `stilts`; omitted becomes `dance` | Selects the environment, reward-key allowlist, artifact stem, PPO defaults, episode horizon, and skill evaluator. |
| `runName` | Sanitized lowercase name, `[a-z0-9_-]`, maximum 48 characters; scenario defaults are `dance-studio`, `swing-studio-01`, `running-studio`, and `stilts-10cm-studio` | Names the artifact directory. Reusing it addresses and may replace the same final checkpoint and ONNX files. |
| `profile` | `smoke` or `full`; omitted or unknown becomes `smoke` | Chooses pipeline versus skill evaluation and applies profile-dependent defaults and overrides. |
| `totalTimesteps` | Integer effectively clamped to 1–40,000,000, then raised to at least one batch and rounded up to a batch multiple; profile defaults are 4 or the scenario full budget | Number of **new** environment transitions requested from this training invocation. On resume, it is not the cumulative target. |
| `numEnvs` | 1–64 after rounding; default 2 in smoke, 16 in full | Number of parallel environments. It multiplies rollout size, throughput, memory use, and the step increment of the trainer. |
| `numSteps` | Integer 1–100,000; default 2 in smoke, 64 for full Swing, 24 for other full recipes | Control steps collected per environment before each PPO update. Longer rollouts improve temporal coverage but increase latency and batch size. |
| `numMinibatches` | Integer 1–100,000; default 1 in smoke, 4 in full; must not exceed or fail to divide `numEnvs * numSteps` | Splits one rollout batch into equal optimizer minibatches. Invalid divisibility is rejected rather than silently repaired. |
| `maxEpisodeS` | Finite value greater than 0 and at most 3,600; default 1 in smoke; full defaults: Dance 4, Swing 24, Running 12, Stilts 12 | Training/evaluation episode timeout. At 50 Hz, seconds multiply by 50 to give nominal control steps per complete episode. Full Running rejects values below 12 seconds. |
| `evalSteps` | Integer 1–10,000,000; default 4 in smoke; full default is the larger of 50 times the episode horizon and 500, except Running has a 600-step floor | Number of evaluation control steps **per environment**. Full Running rejects fewer than 600. Full Swing therefore defaults to 1,200. |
| `renderSeconds` | Finite value greater than 0 and at most 3,600; Dance defaults to 120, other recipes default to `maxEpisodeS` | Render-only horizon. It does not lengthen training episodes. Studio renders one 640 x 360 three-quarter-view episode. |
| `checkpointInterval` | Integer 0–40,000,000; default 100,000 | Interval in newly trained steps for snapshots under `checkpoints/`. Zero disables periodic snapshots, but the initial and final checkpoint behavior remains. |

The browser exposes `runName`, `profile`, `totalTimesteps`, and `numEnvs`.
The other fields in this table are API-capable fields. They can be supplied by
an HTTP recipe or the repository's API driver even though the current Studio
form does not show them.

## Scenario inputs and environment fields

| Field | Valid/default value | Purpose and effect |
|---|---|---|
| `danceClip` | `null`, or an existing JSON file resolving under workspace `dance-clip/` or `rlx/artifacts/`; Dance only | Replaces the built-in `dance-120bpm` motion source. The Python metadata records the resolved path and SHA-256 hash. |
| `dancePoseSigma` | `null`, or any finite number greater than 0; Dance only | Sets the width of the per-joint Gaussian pose-match term. Smaller values judge joint error more sharply; larger values grant partial credit farther from the target pose. |
| `locomotionForwardCommand` | `null`, or finite $0 < v_x \le 1.5$ m/s; Running or Stilts only | Pins every sampled command to `(v_x, 0, 0)`. `null` leaves the recipe's command sampler and curriculum active. The same pin is forwarded to train, evaluate, and render. |
| `domainRand` | Boolean; full defaults on, smoke forces off | Randomizes supported physical properties across resets. It improves robustness but broadens the learning problem. |
| `obsNoise` | Boolean; full defaults on, smoke forces off | Perturbs observations to reduce dependence on exact simulated sensing. |
| `actionDelay` | Boolean; full defaults on, smoke forces off | Introduces control delay so the policy must tolerate latency. |
| `randomYaw` | Boolean; full defaults on, smoke forces off | Randomizes initial yaw where the recipe supports it. Swing always resolves this to false because yaw is fixed to the seat axis. |
| `stiltHeightCm` | 0.8–300 cm; Studio default 10 cm | Defines stilt length. It changes morphology, leverage, center-of-mass geometry, and the policy's balance problem. Evaluate and render with the same value used in training. |
| `stiltBlend` | 0–1; Studio default 0.5 | Blends the stilt support/contact shape. Zero and one are the endpoints; intermediate values trade contact characteristics. |
| `stiltMassKg` | 0.001–2 kg in the API; Studio default 0.029 kg | Mass of each stilt. This changes inertia and actuator demand. The lower-level CLI can derive mass from height when omitted, but a normalized Studio recipe always sends an explicit value. |
| `swingInitialAngleDeg` | 0–30 degrees; default 0 | Training assistance that starts the swing displaced from rest. It is useful for discovery, but is forced back to 0 for skill evaluation. |
| `swingInitialRateRadS` | 0–1 rad/s; default 0 | Training assistance that gives initial angular motion. Like angle assistance, it does not count as learned self-pumping and is removed for skill evaluation. |
| `swingPlanarActions` | Boolean; effective only for Swing; default true in Studio/API | Restricts discovery actions to the mechanism's useful plane, reducing irrelevant exploration. Non-Swing recipes normalize it to false. |
| `swingMinSpanDeg` | 1–180 degrees; default 150 | Skill-evaluation threshold for **symmetric total span**. A 150-degree target requires at least 75 degrees on each side, plus complete episodes, valid geometry, and tensioned strings. |

The current browser exposes stilt and swing controls under **Advanced PPO and
environment settings**. It does not expose `danceClip`, `dancePoseSigma`, or
`locomotionForwardCommand`; those are API fields. Dance clips are normally
created in the **Animate** tab and can be selected by a caller that submits the
resulting path.

## PPO optimization fields

| Field | Valid/default value | Purpose and effect |
|---|---|---|
| `seed` | Normalized to an integer 1–2,147,483,647; default 1 | Seeds NumPy, MLX, and environment randomness. A seed is evidence for reproducibility, not proof that one result generalizes. |
| `learningRate` | Clamped to $10^{-6}$–0.1; Dance/Running/Stilts default $3\times10^{-4}$, Swing $10^{-4}$ | Adam step size. Too large destabilizes policy and critic updates; too small can make a fixed budget ineffective. |
| `gamma` | Clamped to 0.8–1; Dance/Running/Stilts default 0.99, Swing 0.995 | Reward discount. Higher values preserve more credit across long behaviors; Swing uses a longer horizon because pumping gains accumulate over many cycles. |
| `clipCoefficient` | Clamped to 0.01–1; Dance/Running/Stilts default 0.2, Swing 0.1 | PPO policy-ratio clip $\epsilon$. Smaller values make each update more conservative. |
| `updateEpochs` | Integer normalized to 1–20; Dance/Running/Stilts default 5, Swing 3; smoke effectively 1 | Number of passes over each rollout batch. More passes reuse data more aggressively and can overfit or increase KL drift. |
| `entropyCoefficient` | Clamped to 0–1; Dance/Running/Stilts default 0.01, Swing 0.002 | Weight on policy entropy. More entropy encourages exploration; too much can prevent a precise deterministic mean policy. |
| `maxGradNorm` | Clamped to 0.01–10; Dance/Running/Stilts default 0.5, Swing 1.0 | Global gradient-norm clipping threshold. It limits update spikes but cannot repair a malformed reward. |
| `initialStd` | Finite $0 < \sigma \le 10$; Dance/Running/Stilts default $e^{-0.5}\approx0.6065$, Swing 0.1 | Initial Gaussian action standard deviation for a **new** model. It is ignored when a checkpoint supplies the existing network and its learned log standard deviation. |
| `normalizeRewards` | Boolean; default true only for Swing | Enables return/reward normalization in the normalized training environment. It changes the scale seen by PPO and the meaning of the displayed rollout curve. |
| `freezeObservationNormalization` | Boolean; default false; true requires `resumeFromCheckpoint` | Loads observation-normalizer statistics from the checkpoint and prevents further updates. Useful when fine-tuning should preserve the old input scaling. |
| `resumeFromCheckpoint` | Boolean; default false; true requires the selected run's checkpoint to exist | Adds `--init-from` for training. The existing model, observation statistics, return statistics when present, and cumulative step count are loaded. |

The browser currently exposes the six main PPO values from `learningRate`
through `maxGradNorm`. `seed` is part of the browser recipe state but has no
general visible input; the Swing recovery buttons set it to 2.
`initialStd`, reward normalization, and normalization freezing are API-only.
The general browser shows checkpoint continuation only in the Swing controls,
although the API field itself is valid for all four recipes.

## Fixed PPO behavior not represented by recipe fields

Several important PPO settings are deliberately not editable through
`RlxRecipe`.

| Fixed setting | Effective Studio value | Role |
|---|---|---|
| `gae_lambda` | 0.95 for Dance, Running, and Stilts; 0.98 for Swing | Controls the bias/variance horizon of generalized advantage estimates. |
| `normalize_advantages` | `true` | Centers and scales advantages before the PPO policy loss. |
| `clip_value_loss` | `true` | Enables pessimistic clipping of critic changes. |
| `value_coefficient` | 1.0 in the Studio runner; core `PPOConfig` default is 0.5 | Scales critic loss in the combined PPO objective. |
| Huber delta | 10 | Makes critic loss quadratic near zero and linear for large errors. |
| Log-ratio cap | `[-20, 20]` | Prevents overflow before exponentiating the policy probability ratio. |

Generalized advantage estimation uses

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
$$

$$
\hat A_t = \sum_{l=0}^{L-1}(\gamma\lambda)^l\delta_{t+l}.
$$

Here, the bootstrap is zero for true termination, uses the terminal-observation value for time-limit truncation, and advantage propagation stops at either boundary. These compact equations describe a single uninterrupted segment; the foundations chapter gives the exact boundary-aware recursion. The critic consumes actor observations, even though the conventional shorthand above calls them $s$.

Studio passes `gae_lambda=0.95` for Dance, Running, and Stilts, and `0.98` for
Swing. Advantages are normalized before each loss calculation:

$$
\hat A \leftarrow
\frac{\hat A-\operatorname{mean}(\hat A)}
{\operatorname{std}(\hat A)+10^{-8}}.
$$

The policy ratio is

$$
r_t(\theta)=
\exp\left(
\operatorname{clip}\left[
\log\pi_\theta(a_t|s_t)-\log\pi_{\text{old}}(a_t|s_t),
-20,20
\right]\right).
$$

The `[-20, 20]` cap is a numerical guard applied before exponentiation. The
policy objective uses the standard pessimistic PPO maximum of the negative
unclipped and clipped surrogate losses.

The critic does **not** use ordinary mean-squared error. For error
$e=V_\theta(s)-R$, it uses a Huber loss with delta 10:

$$
H_{10}(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\le 10,\\
10(|e|-5), & |e|>10.
\end{cases}
$$

Value clipping is enabled. The implementation computes both the Huber loss of
the new value and the Huber loss after clipping the value change to the same
`clipCoefficient` interval used by PPO, then takes the larger loss. This is
the pessimistic value-clipping form:

$$
L_V=\operatorname{mean}
\left[
\max\left(
H_{10}(V_\theta-R),
H_{10}(V_{\text{old}}+
\operatorname{clip}(V_\theta-V_{\text{old}},-\epsilon,\epsilon)-R)
\right)
\right].
$$

The combined minimized loss is

$$
L=L_{\text{policy}}-c_H H(\pi)+c_V L_V.
$$

Core `PPOConfig` declares `value_coefficient=0.5`, but the current Studio
Python command parser explicitly defaults `--value-coefficient` to `1.0`.
Because `rlx-job.ts` does not pass that flag, **Studio's effective
`value_coefficient` is 1.0**, not the core dataclass default. The metadata
records this effective value. `clip_value_loss=true` and
`normalize_advantages=true` are also parser defaults and are not exposed by
the Studio recipe.

The Microduck actor clamps each learned action log standard deviation to
`[-5, -0.5]` before sampling and constrains the stored parameter after updates.
Thus the effective standard deviation remains between
$e^{-5}\approx0.0067$ and $e^{-0.5}\approx0.6065$. Although the API accepts an
`initialStd` as high as 10, a fresh model immediately uses the clamped upper
bound. This is separate from the wider generic distribution utility's bounds;
the Microduck model's tighter limits are the ones that govern these recipes.

## Requested versus effective values on resume

**Continue current checkpoint** means “train for another
`totalTimesteps`,” not “train until cumulative steps equal
`totalTimesteps`.” If a checkpoint contains 250,000 completed steps and the
new request contains 500,000, the resulting metadata reports approximately
750,000 cumulative steps, subject to rollout-batch rounding.

Resume loads:

* Actor and critic parameters, including learned action log standard deviation.
* Observation running mean, variance, and count.
* Return-normalization statistics when present.
* The previous cumulative step count for metadata.

Resume does not restore an optimizer state in this runner; a new Adam
optimizer is constructed with the newly requested learning rate. It also does
not use `initialStd`, because no new actor is created. Requested PPO
coefficients, rollout shape, rewards, randomization, and episode settings take
effect for the new training segment. This makes a resumed run a fine-tuning
stage, not a byte-for-byte continuation of an interrupted optimizer.

`freezeObservationNormalization=true` preserves the loaded observation
scaling while the policy changes and also restores the checkpoint's
normalizer epsilon and clipping values. With it false, loaded mean, variance,
and count seed the normalizer and continue updating under the new
environment's epsilon and clipping defaults. Reward normalization is
separately controlled by `normalizeRewards`; loaded return statistics are
restored when available.

Be careful with artifacts. Training writes the selected run's final checkpoint
and normally exports ONNX automatically. Starting a fresh run with the same
name and `resumeFromCheckpoint=false` replaces the final lineage at that path.
A changed reward weight defines a new experiment conceptually, so use a new run
name instead of evaluating an old checkpoint against a newly edited reward
form.

## Reward weights: sign and interpretation

`rewardWeights` is an object mapping known term keys to absolute,
non-negative coefficients. Unknown keys, negative values, infinities, and
NaNs are rejected; the API caps each weight at 10,000. Penalty measurements
are already negative, so their coefficients remain positive. Do not negate a
penalty weight.

Conceptually,

$$
r_t=\sum_i w_i q_i(s_t,a_t),\qquad w_i\ge0,
$$

where reward measurements $q_i$ are positive and penalty measurements are
non-positive. Increasing a coefficient increases that term's influence, but
does not guarantee the desired motion is explored. If no rollout ever enters
the target state, change the curriculum or physics assistance before endlessly
retuning weights.

### Dance reward terms

| Key and default | Kind | What it pays for |
|---|---|---|
| `pose_match` 4 | Reward | Joint pose at the current clip frame; `dancePoseSigma` controls tolerance. |
| `rotation_match` 4 | Reward | Body rotation synchronized to clip timing. |
| `stick_it` 5 | Reward | Finishing a one-shot trick standing on both feet. |
| `on_feet` 5 | Reward | Remaining upright, tall, and in foot contact during a loop. |
| `travel` 3 | Reward | Forward movement rather than dancing in place. |
| `no_slip` 0.5 | Penalty | Planted feet sliding over the floor. |
| `no_spin` 0.3 | Penalty | Unwanted yaw when travel should remain straight. |
| `gentle_head` 1 | Penalty | Head-floor impacts, scaled by severity. |
| `soft_landings` 0.75 | Penalty | Excessive landing impact. |
| `no_limit_parking` 1 | Penalty | Holding joints against end stops. |
| `save_energy` 0.5 | Penalty | Excessive actuator torque. |

### Swing reward terms

| Key and default | Kind | What it pays for |
|---|---|---|
| `swing_peak_progress` 224 | Reward | A new angular frontier on either side; prevents repeated payment for the same arc. |
| `swing_height` 8 | Reward | Height throughout the episode. |
| `swing_late_height` 24 | Reward | Height later in the attempt, encouraging sustained pumping. |
| `swing_energy` 0.5 | Reward | Useful pendulum height and angular speed. |
| `swing_lateral_penalty` 3 | Penalty | Sideways offset from the swing plane. |
| `swing_lateral_barrier_penalty` 8 | Penalty | Strong charge beyond the safe lateral band. |
| `swing_lateral_velocity_penalty` 3 | Penalty | Fast sideways movement. |
| `swing_out_of_plane_penalty` 1 | Penalty | Roll and yaw angular velocity. |
| `swing_alignment_penalty` 18 | Penalty | Misalignment of the two attachment axes. |
| `swing_alignment_barrier_penalty` 4 | Penalty | Strong charge after alignment exceeds its valid range. |
| `string_slack_penalty` 8 | Penalty | Slack or unequal string lengths. |
| `string_extension_penalty` 12 | Penalty | Strings stretched beyond intended length. |
| `invalid_episode_penalty` 10 | Penalty | Persistent invalid mechanism geometry. |
| `action_rate_penalty` 0.03 | Penalty | Abrupt changes between policy actions. |
| `joint_torque_penalty` 0.001 | Penalty | Excessive actuator force. |
| `joint_limit_penalty` 1 | Penalty | Motion beyond joint limits. |

### Running reward terms

| Key and default | Kind | What it pays for |
|---|---|---|
| `keep_pace` 4 | Reward | Matching commanded body-frame velocity. |
| `track_turn` 2 | Reward | Matching commanded yaw rate. |
| `air_time` 3 | Reward | Useful duration of individual foot flight. |
| `flight` 0 | Reward | Optional bounded both-feet flight, gated by upright posture and command-directed speed. |
| `yaw_tracking` 0 | Reward | Optional yaw-only tracking from observed gyro and commanded turn rate. |
| `stay_upright` 2 | Reward | Upright trunk while allowing running lean. |
| `pose` 1 | Reward | Speed-appropriate running posture. |
| `head_up` 3.5 | Reward | Head alignment with its command. |
| `foot_clearance` 2 | Penalty | A moving foot at the wrong height. |
| `plant_the_foot` 0.1 | Penalty | Skidding by a planted foot. |
| `smooth_moves` 1 | Penalty | Jerky actions as the gait curriculum advances. |
| `no_limit_parking` 1 | Penalty | Joints held against end stops. |
| `calm_roll` 0.025 | Penalty | Excessive trunk roll and pitch angular velocity. |

### Stilts reward terms

| Key and default | Kind | What it pays for |
|---|---|---|
| `track_lin_vel` 2.5 | Reward | Requested linear velocity. |
| `track_ang_vel` 1 | Reward | Requested yaw rate. |
| `yaw_tracking` 0 | Reward | Optional yaw-only tracking independent of roll and pitch. |
| `upright` 3 | Reward | A level trunk on raised contacts. |
| `pose` 0 | Reward | Optional pull toward the default leg pose. |
| `head_pose` 0.25 | Reward | Requested head posture without dominating balance. |
| `feet_air_time` 2 | Reward | Controlled swing phases while walking. |
| `action_rate_penalty` 0.2 | Penalty, fixed | Twenty percent of the walking curriculum's dynamic action-rate penalty. It is displayed but excluded from editable `rewardWeights`. |
| `ang_vel_xy_penalty` 0.05 | Penalty | Excessive roll and pitch angular velocity. |

![Recipe controls, upper panel. Native three-times-density browser capture, cropped without rescaling for legibility.](assets/studio-parameters-top.png)

![Advanced PPO, environment, and reward parameters, lower panel. Read historical values from the saved recipe rather than these current UI defaults.](assets/studio-parameters-bottom.png)

## Train from the UI

1. Select the experiment and choose a unique run name.
2. Select **Pipeline smoke** and leave reward defaults unchanged.
3. Select **Start RLX**. The button changes to a running state, the progress
   bar advances, and **Reward history** receives one point per completed PPO
   rollout. A normalized curve is labeled as normalized when Swing reward
   normalization is active.
4. Open **View logs** if launch or validation fails. The log is the right place
   to find an invalid minibatch divisor, missing checkpoint, invalid clip path,
   or Python error.
5. Confirm that checkpoint, metadata, and ONNX artifacts appear. Smoke proves
   the pipeline only.
6. Select **Default full**, inspect advanced settings, assign a new run name
   if this is a materially different reward or environment recipe, and select
   **Start RLX** again.

Only one local RLX process can run at a time. A job for another run makes the
button read **RLX busy**. **Stop train/eval/render** sends `SIGTERM` and marks
the current job cancelled.

For Swing, the UI supplies a staged recovery workflow. **Apply exact Discovery
settings** requests 250,000 new steps, seed 2, 16 environments, 12 degrees and
0.35 rad/s of initial assistance, planar actions, no randomization/noise/delay,
and the Swing PPO defaults. After still-start evaluation demonstrates at least
10 degrees mean span or 20 degrees best span, **Apply exact Consolidation
settings** requests 500,000 additional steps from the same checkpoint, reduces
assistance to 4 degrees and 0.12 rad/s, and enables randomization, noise, and
delay.

## Evaluate from the UI

1. Select the experiment and enter the run name containing a checkpoint or
   ONNX policy.
2. Set **Pipeline smoke** only when checking execution. Select **Default full**
   when asking whether the skill passes.
3. Select **Run evaluation** in **Evaluation gate**.
4. Read the saved scope. Smoke maps to `evaluation_mode=pipeline`; full maps
   to `evaluation_mode=skill`.
5. Inspect **Rollout validity** separately from the scenario metric. Finite
   observations, actions, and rewards prove numerical execution, not skill.
6. Require the scenario row to say **Accepted** before treating the skill gate
   as passed.

Evaluation prefers the run's ONNX file when present and otherwise evaluates
the checkpoint. The report hashes its source. Studio only retains an
authoritative verdict while those owned artifact bytes and the normalized
environment settings still match. Editing the current form does not rewrite a
saved result, and changing the evaluated environment invalidates its authority.

Full Dance evaluates at least 500 steps and the complete clip horizon. Full
Running requires 600 steps and a 12-second episode. Full Stilts defaults to at
least 600 steps in Studio because its full episode horizon is 12 seconds. Full
Swing uses 1,200 steps, starts from 0 degrees and 0 rad/s, and applies the
selected symmetric span target.

## Render and review from the UI

1. After a checkpoint exists, select **Render rollout**.
2. Studio chooses ONNX when available, otherwise the checkpoint.
3. Wait for the **LATEST RUN** video player and contact-sheet links to appear.
4. Play the complete MP4, then open the contact sheet for time-separated
   posture and metric checks.
5. Compare the visible behavior with the intended skill and, when relevant,
   with a null control. Reward and return alone can approve the wrong motion.
6. Select **I reviewed the rendered motion** only after completing that review.

Render is intentionally deterministic and clean: one episode, 640 x 360,
three-quarter camera, and all randomization/noise/delay/yaw perturbations off.
Dance disables fall termination so a requested 120-second choreography render
can continue; this does not erase a fall from visual review.

The deployment handoff remains gated until an ONNX file exists, deterministic
skill evaluation passes, the render exists, and the visual-review checkbox is
selected. Hardware validation remains a separate requirement.

![Evaluation panel, upper half: the saved Running verdict and policy video. Cropped from the native browser capture without rescaling.](assets/studio-evaluation-top.png)

![Evaluation panel, lower half: inspect the full saved evidence and review controls. The overlap with the preceding image preserves reading context.](assets/studio-evaluation-bottom.png)

## UI versus API reference

The UI is intentionally simpler than `RlxRecipe`. The API is the complete
surface.

| Surface | Fields you can set directly | Important behavior |
|---|---|---|
| Main Studio form | Experiment, run name, profile, total timesteps, parallel envs, six PPO controls, four randomization toggles, stilt controls, swing controls, editable reward weights | Optimized for normal local operation. Scenario selection resets the form to defaults. |
| Studio preset buttons | Full/smoke profiles and exact Swing Discovery/Consolidation values | Presets modify several fields together. Read the resulting form before launch. |
| `POST /api/rlx` | Every `RlxRecipe` field | Returns the normalized recipe. Use this surface for rollout shape, horizons, clip paths, pinned commands, initial exploration, normalization controls, and checkpoint cadence. |
| Python CLI | Studio recipe arguments plus lower-level source/output, backend, actuator, render geometry, and evaluator controls | The Studio wrapper intentionally fixes many CLI details to preserve a consistent local workflow. |

For API-driven evidence, save both the submitted recipe and the accepted
normalized recipe. The repository's HTTP driver does this as
`requestedRecipe` and `normalizedRecipe`. That pairing explains clamping,
defaults, profile overrides, and rollout rounding without guesswork.

## Parameter-reading checklist

Before accepting a run, answer these questions from artifacts rather than
memory:

1. Which `experimentId` and sanitized `runName` own the files?
2. Was the operation smoke/pipeline or full/skill?
3. How many **new** steps were requested, what batch multiple was effective,
   and what cumulative step count is in metadata?
4. Was the model new or loaded through `resumeFromCheckpoint`?
5. Which PPO values were effective, including fixed GAE, advantage
   normalization, value coefficient, value clipping, Huber delta, and
   log-standard-deviation limits?
6. Were reward normalization and observation-normalizer freezing active?
7. Which randomization and morphology settings governed training, evaluation,
   and rendering?
8. Do reward keys and weights match the checkpoint's experiment?
9. Does deterministic evaluation explicitly assess the skill, rather than
   merely pass the finite pipeline check?
10. Has the complete rendered motion been inspected?

That discipline turns the Studio from a collection of controls into a
reproducible experiment system.
