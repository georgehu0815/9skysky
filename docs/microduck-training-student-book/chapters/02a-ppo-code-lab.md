# PPO Code Laboratory: Follow One Batch End to End

## Learning goals and the four kinds of data

After this laboratory you should be able to trace one observation through the actor, environment, rollout buffer, advantage estimator and optimizer; calculate the exact sample budget; distinguish a demonstration from an on-policy transition; and independently read the four saved audit files. The mathematics below is grounded in `rlx/rlx/algorithms/ppo.py`, `rlx/rlx/buffers/rollout_buffer.py`, `rlx/rlx/utils/utils.py` and `rlx/rlx/models/microduck.py`. The accompanying [learning_lab.py](learning_lab.py) runs arithmetic and evidence inspection with the Python standard library. It is not a replacement PPO trainer.

| Data object | Who creates it? | Input and target | Where it is used |
|---|---|---|---|
| Dance reference clip | Motion preparation | Time-indexed target joint angles | Environment reward and independent tracking evaluator |
| Swing imitation dataset | Simulator teacher queried during teacher/student rollouts | Ordinary observation and teacher action | Supervised actor-mean training |
| PPO rollout buffer | Current stochastic student in the physical environment | Observation, sampled action, reward, boundary flags, old log probability and value | GAE and repeated PPO minibatches |
| Evaluation traces | Frozen exported mean actor | Physical states and task-specific measurements over a complete episode | Acceptance gates, plots and video evidence |

These are not four interchangeable forms of a training CSV. Dance uses imitation **as a reward objective** in a fresh PPO experiment; that does not imply Dance used Swing's supervised BC optimizer. Running and Stilts generate learning data through interaction rather than loading a labeled gait dataset. Swing explicitly adds a supervised acquisition stage. Never claim that all four activities used BC/DAgger, or that a Dance target clip is itself a trained policy.

![The four data routes, with supervised labels separated from PPO samples. This is an explanatory diagram, not a measured result.](assets/data-routes.svg)

## The actual actor–critic, not an unspecified neural network

There are **two separate multilayer perceptrons**, not one shared hidden trunk: an actor mean network `61 → 512 → 256 → 128 → 14` and a critic `61 → 512 → 256 → 128 → 1`. Each hidden layer uses the numerically stable ELU implementation. The actor also owns fourteen trainable log-standard-deviation values, one per joint; they do not depend on the current observation. The critic predicts one scalar discounted value for each observation. The architecture is feed-forward, not recurrent: previous action and other observation fields provide limited history, but there is no LSTM hidden state.

Counting weights and biases gives 197,774 actor-mean parameters, fourteen log-standard deviations, and 196,097 critic parameters: **393,885 trainable parameters** in total. This is a derived architecture count, not a measurement of model quality. Each linear layer contributes `(input_width + 1) × output_width`. The deterministic exported policy does not need the critic or exploration standard deviation. Its required computation is the learned observation transformation followed by the actor mean.

At each control tick the actor returns `mean.shape == (16, 14)` for sixteen environments. The critic returns `(16, 1)`. The policy distribution samples

$$a_{n,j}=\mu_{n,j}+\exp(\ell_j)\epsilon_{n,j},\qquad \epsilon_{n,j}\sim\mathcal N(0,1).$$

Microduck clips its log standard deviation to `[-5.0, -0.5]`, corresponding approximately to standard deviations `[0.00674, 0.60653]`. This is the Microduck model's restriction, not the generic distribution helper's range. `initialStd=0.1` starts near `log(0.1)=-2.30259` for fresh weights. A checkpoint can restore a different learned value; inspect the initializer rather than assuming the fresh-model flag won.

The Gaussian joint log probability is summed across action dimensions:

$$\log\pi(a\mid o)=\sum_{j=1}^{14}\left[-\frac12\left(\frac{a_j-\mu_j}{\sigma_j}\right)^2-\log\sigma_j-\frac12\log(2\pi)\right].$$

Consequently, `log_probs` has one scalar per environment transition, not fourteen independent PPO objectives. Correlated movement is represented through the observation-dependent action means and the physics; the exploration covariance is diagonal.

**Important action bookkeeping.** The rollout buffer stores the sampled Gaussian action and its matching log probability. The environment subsequently clips/transforms that action into a joint target. Do not replace the saved sample with its clipped target when recomputing Gaussian log probabilities: the denominator would no longer refer to the same action. Large saturation is still undesirable because many distinct samples produce nearly identical applied targets; logging saturation is a useful proposed diagnostic, not an existing success certificate.

## Trace a single collection tick

Read `PPO.train()` and locate the following sequence. This is an execution map, not runnable substitute code:

1. Evaluate `network(observation)` to produce distribution and critic value.
2. Sample the action and compute its old-policy joint log probability.
3. Call `env.step(...)`. CPU MuJoCo advances each world; the adapter returns the next normalized observation, reward, terminal flags and diagnostic information.
4. Extract timeout values from the actual terminal observation when needed. Autoreset observations belong to a new episode and cannot substitute for that state.
5. Append one vectorized row to the buffer. Store observations, actions, rewards, flags, values and log probabilities without losing the environment index.
6. Advance the transition counter by `num_envs`, not by one. Move to the next tick using the returned observation.

For sixteen environments one collection tick represents sixteen transitions and 0.02 seconds of simulated time **in each world**. The policy does not receive CPU wall time or the entire MuJoCo state as an input. The training process alternates collection and updates; “MLX accelerated” does not mean MuJoCo physics in this route has moved onto a GPU. The ordinary Mac path here uses CPU physics plus MLX/Metal neural-network work.

## Inspect the rollout tensors

For Swing, `T=256`, `N=16`, observation dimension `61` and action dimension `14`. `RolloutBuffer.reset()` allocates the following float32 arrays:

| Buffer field | Shape before flattening | Meaning |
|---|---|---|
| `observations` | `(256, 16, 61)` | Normalized actor/critic inputs at action time |
| `next_observations` | `(256, 16, 61)` | Adapter-returned next inputs; special timeout information is handled separately |
| `actions` | `(256, 16, 14)` | Sampled, pre-environment-transform actions |
| `rewards` | `(256, 16)` | Training rewards, normalized when configured |
| `values`, `log_probs` | Each `(256, 16)` | Old critic predictions and old-policy action log probabilities |
| `terminations`, `truncations` | Each `(256, 16)` | Separate physical-failure and time-limit boundaries |
| `advantages`, `returns` | Each `(256, 16)` | Reserved buffer fields; this PPO path passes computed GAE/returns into `update()` |

The timeout-value list is collected outside those buffer fields and stacked for GAE. The return target is `advantages + buffer.values`. Some allocated buffer fields are not the active source of the update argument; verify actual reads, not only class member names.

![Tensor shapes through one Swing collection and update. Dimensions and counts are derived from the saved recipe.](assets/tensor-batch.svg)

`flatten()` reshapes the first two dimensions together: `(256,16,61)` becomes `(4096,61)`. The flat sample index is `time_index × 16 + environment_index`. All arrays must be flattened and indexed consistently. It would be a serious silent error to permute actions independently from observations, advantages or old log probabilities.

Each epoch generates a new permutation of the 4,096 sample indices. Four minibatches contain 1,024 transitions each; two epochs give eight optimizer steps per iteration. The next collection starts only after those updates. The buffer is reset for the next iteration: these PPO updates do **not** train indefinitely from a permanent replay pool of old policies.

## Why “524,288 transitions” is a budget, not a giant batch

The selected Swing continuation uses:

$$16\times256=4,096\text{ transitions per rollout},$$
$$524,288/4,096=128\text{ collect/update iterations},$$
$$128\times2\times4=1,024\text{ optimizer steps}.$$

There are 1,048,576 sample presentations across the two epochs, but still only 524,288 newly collected environment transitions. BC training steps, DAgger teacher labels, audits and rendering are not counted in that PPO number. Nor does the number include earlier failed experiments or the Running/Stilt base stages.

At 50 Hz the Swing PPO phase represents `524288 × 0.02 = 10485.76` aggregate simulated seconds, about 2.913 hours summed over all worlds. Dividing by sixteen gives 655.36 simulated seconds per environment stream. Those streams can contain many resets; neither number is a single continuous 24-second episode or measured wall-clock runtime. Do not use either quantity to estimate real-robot battery time.

![Same arithmetic, different quantities: rollout size, PPO transition budget, and repeated optimizer presentations. This diagram excludes imitation data and is not a wall-time chart.](assets/transition-accounting.svg)

`PPO.train()` finishes complete rollouts. For a hypothetical request of 4,000,000 transitions with batch size 2,048:

$$\left\lceil4,000,000/2,048\right\rceil\times2,048=4,001,792.$$

The audit stores Dance's effective 4,001,792-step recipe; its training summary does not separately store `requested_transitions`. Our evidence reader reports that difference instead of inventing the original user request. Treat UI rounding, requested budget, effective recipe, observer counters and cumulative checkpoint lineage as different records that should reconcile.

## Work through timeout-aware GAE numerically

This is a one-transition boundary test. Continue with the **GAE Workshop** for the full three-transition backward calculation, when PPO uses GAE, and why lambda one still retains a value bootstrap at a nonterminal cutoff.

Suppose one final step has reward `1`, old value `2`, discount `0.99`, terminal-state value `5`, and reset-state value `99`. These are **synthetic arithmetic inputs** chosen to expose a bug, not robot measurements.

- Physical termination: bootstrap is zero, so the advantage is `1 - 2 = -1`.
- Time-limit truncation without physical termination: bootstrap the terminal observation, so the advantage is `1 + 0.99 × 5 - 2 = 3.95`.
- Incorrectly bootstrapping the reset observation would give `97.01`. That rewards a transition based on a different episode.
- When both flags are true, termination wins: the correct result is `-1`.

In a multi-step sequence the backward recursion is

$$\delta_t=r_t+\gamma b_t-V(o_t),$$
$$\widehat A_t=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1},$$

where `d_t` is termination **or** truncation, while `b_t` distinguishes their different bootstrap rules. A timeout bootstraps its own terminal value, but does not propagate advantages from the next reset episode. At a rollout boundary that is not an episode boundary, the last critic value provides the bootstrap.

The selected `gamma` and GAE `lambda` govern the discounted objective and estimator trade-off. A useful intuition is the decay scale, not a hard planning horizon: `gamma=0.99` has `1/(1-gamma)=100` control steps, about two seconds, while `0.995` gives 200 steps, about four seconds. Reward information is not literally discarded after that time. GAE's `lambda` further changes the advantage weighting; it is not the simulator timestep.

## Read the clipped objective with its sign intact

For a fixed batch sample let `ratio = exp(new_log_prob - old_log_prob)`. A positive advantage encourages increased probability; a negative one encourages decreased probability. The maximized surrogate is the smaller of unclipped and clipped products. The source minimizes its negative using `maximum` of the two negative terms.

For `clip=0.2`:

| Advantage | Ratio | Unclipped product | Clipped product | Maximized surrogate |
|---|---|---|---|---|
| +2 | 0.7 | 1.4 | 1.6 | 1.4 |
| +2 | 1.0 | 2.0 | 2.0 | 2.0 |
| +2 | 1.3 | 2.6 | 2.4 | 2.4 |
| −2 | 0.7 | −1.4 | −1.6 | −1.6 |
| −2 | 1.0 | −2.0 | −2.0 | −2.0 |
| −2 | 1.3 | −2.6 | −2.4 | −2.6 |

The plateau is on the helpful-update side; clipping does not make every out-of-range ratio harmless. It also does not constrain all future observations or guarantee monotonically better robot performance. Swing's rejected longer continuation is a concrete reason to keep a physically validated earlier checkpoint.

![Synthetic PPO objective curves for positive and negative advantages. These are exact scalar arithmetic, not measured training loss.](assets/clipped-objective.svg)

The implementation additionally clips **log ratios** to `[-20,20]` before exponentiation to avoid numerical overflow. That finite guard is distinct from the small PPO ratio interval, such as Swing's `[0.95,1.05]`. Gradient norm clipping is a third mechanism, operating on the update gradient rather than on action probabilities. None of the three replaces finite-value checks or a physical skill evaluator.

When enabled, advantage normalization occurs inside each minibatch: subtract that minibatch's mean and divide by its standard deviation plus `1e-8`. Therefore the table demonstrates the objective for already chosen advantages; it does not claim those exact unnormalized values survive the real minibatch transformation. Old log probabilities, old values and return targets remain fixed while the same rollout is reused across epochs.

## The critic's Huber loss and a misleading total loss

The current implementation uses Huber value error with threshold 10:

$$h(e)=\begin{cases}\tfrac12e^2,& |e|\le10,\\10|e|-50,&|e|>10.\end{cases}$$

For an error of 2, Huber loss is 2; for 20 it is 150, rather than squared-error loss 200 under the half-square convention. This limits growth of the error derivative but does not repair invalid observations or NaN gradients. With value clipping enabled the code takes the **larger** of Huber error under the new value and under the clipped change from the old value, then averages.

The selected Studio value coefficient is 1.0. Swing's saved last-100-update means contain `policy_loss=-0.0003224187`, `value_loss=0.3731174710` and `mean_loss=0.3727950529`. With entropy coefficient zero, adding the first two recovers the third. A positive total loss here mostly reflects critic fitting, not proof the actor is failing. Its negative explained variance remains a critic diagnostic that should be reported rather than hidden behind a passing skill badge.

The saved Swing entropy is also negative. This is **differential entropy** of a continuous Gaussian, not discrete Shannon entropy; negative values are possible for concentrated distributions. It does not mean a negative probability or zero stochastic exploration.

## Run the arithmetic lab before training

From the workspace root, these commands have no simulation or hardware side effects:

```bash
python3 docs/microduck-training-student-book/learning_lab.py
python3 docs/microduck-training-student-book/learning_lab.py --evidence
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/test_book.py
```

The first command prints the synthetic GAE and clipping cases plus budget calculations. The second reads the four bundled audit JSONs and cross-checks transition counts, iterations and optimizer steps against recipe arithmetic. Expected selected-phase counts are:

| Case | Actual transitions | PPO iterations | Optimizer minibatches | Saved trained audit passes |
|---|---|---|---|---|
| Dance | 4,001,792 | 1,954 | 31,264 | 5/5 |
| Running continuation | 2,097,152 | 1,024 | 16,384 | 5/5 |
| Stilts continuation | 1,048,576 | 512 | 8,192 | 5/5 |
| Swing refinement | 524,288 | 128 | 1,024 | 5/5 |

These pass counts are historical artifact contents, **not a new audit performed by this script**. The script's `policy_sha256` identifies the saved artifact claim; it does not hash a missing live ONNX or certify that today's code produced it. The book tests include regressions for timeout/termination precedence, no cross-reset advantage propagation, sign-correct clipping, invalid budgets, and agreement with every saved audit.

The scalar GAE implementation is a pedagogical port for one environment. It omits MLX, vectorization, automatic differentiation and integration with normalization. Compare it with the actual source, then run native RLX tests when changing the trainer. Do not replace production code with a teaching implementation simply because its small examples pass.

## Student checkpoint: explain it without slogans

1. Why is `(4096,14)` the action batch but `(4096,)` the log-probability batch?
2. If you double epochs but keep the rollout budget fixed, which counter doubles and which stays fixed? Calculate Swing's new optimizer count.
3. If you double `num_envs` while keeping `num_steps` and total transitions fixed, why do fewer policy refreshes not imply an equivalent experiment?
4. Change the scalar timeout example so the next episode has reward 1,000. The preceding timeout advantage must stay 3.95. Explain the masking operation.
5. Compare a zero-action control with a randomly initialized network. Why do they answer different causal questions?
6. State the full Swing result in one sentence including imitation acquisition, selected PPO budget, deterministic acceptance conditions, and the absence of hardware evidence.

**Answer checks.** Fourteen independent Gaussian log terms are summed to one joint action density; four Swing epochs would give 2,048 optimizer minibatches with the same 524,288 transitions; doubling environments halves the 128 policy refreshes at fixed horizon/budget. A null control tests passive physics, while an untrained actor tests whether generic network actions suffice. Neither replaces a teacher-only control when attributing improvement to Swing's PPO phase.
