# Mathematical foundations: from observations to PPO

## State is not the same as observation

MuJoCo maintains a simulator state $s_t$: body poses, velocities, contacts, joint states, and constraint information. The policy receives only an observation $o_t=g(s_t,c_t)$, where $c_t$ contains observable task commands. Privileged information can be used to score a task or train a teacher without becoming an actor input. Confusing these roles produces policies that cannot run under the intended interface.

The implementation preserves the **61-input / 14-output** contract. The following indices are zero-based half-open slices from `microduck_local/src/microduck_local/contract.py`:

| Slice | Contents | Interpretation |
|---|---|---|
| `0:3` | Base angular velocity | IMU-like rotational feedback |
| `3:6` | Projected gravity | Orientation relative to gravity |
| `6:20` | Joint position relative to default | Fourteen posture errors |
| `20:34` | Joint velocity | Fourteen motion rates |
| `34:48` | Last action | Recent target command and action history |
| `48:51` | Twist command | Desired forward/lateral/yaw motion |
| `51:55` | Head command slots | Task-conditioned signals |
| `55:61` | Body command slots | Task-conditioned signals |

This table is the base locomotion convention, not a claim that every recipe uses velocity commands. Dance encodes sine/cosine phase at indices `59:61`. Swing has no temporal phase: its twist slots hold `(0, body_y_axis.x, body_y_axis.z)`, and its head/body command slots are zeroed. Preserving dimensions is necessary but not sufficient: a policy must also receive the same slot meanings at evaluation. Do not append an absolute heading, a future reference frame, or a teacher-only pendulum state without explicitly changing the experimental contract.

The base Dance/Running/Stilt action is a raw joint-target offset in radians. The actor's Gaussian/linear outputs are not restricted to $[-1,1]$. The environment clips the applied offset:

$$q^{\mathrm{target}}_t=q^{\mathrm{default}}+\operatorname{clip}(a_t,-4,4).$$

Swing intentionally uses a different task wrapper: it bounds actions to $[-1,1]$, applies its planar projection $P$ when enabled, and scales around a seated pose. Its effective target is $q^{\mathrm{seated}}+0.7P(\operatorname{clip}(a_t,-1,1))$, subject to the base environment's final offset clipping. Preserve this wrapper when evaluating or exporting a Swing actor; matching only tensor dimensions is insufficient.

This is not a raw torque command. Actuator physics and joint limits mediate how targets become motion. The standard controller period is $\Delta t=0.02$ s: a 0.005-second physics timestep with four simulation substeps per action. Thus 400, 500, 600 and 1,200 control steps correspond to 8, 10, 12 and 24 seconds.

**Worked example.** An offset of 0.10 rad is about 5.73°. It changes one joint's target relative to its default, not the robot's global heading by 5.73°. This distinction matters when a reward appears to encourage turning but actually excites a leg.

## Normalization is part of the learned policy

Signals have different units and scales: radians, radians per second, gravity direction and commands. A running mean $\mu$ and variance $v$ transform observations before the network:

$$\widetilde{o}=\operatorname{clip}\!\left(\frac{o-\mu}{\sqrt{v+\epsilon}},-c,c\right).$$

Changing $\mu$ or $v$ while holding network weights fixed changes the physical policy. That is why a safetensors file without its metadata/normalization sidecar is not an interchangeable checkpoint, and why Swing freezes observation statistics when refining an already competent imitation policy.

The exported ONNX incorporates this transformation. Feed raw contract observations into that exported model; **do not normalize them a second time**. The training wrapper may normalize rewards independently using running discounted-return variance:

$$\widetilde r_t=\operatorname{clip}\!\left(\frac{r_t}{\sqrt{v_R+\epsilon}},-c,c\right).$$

This is a scale transformation, not subtraction of the mean reward. Observation-statistics freezing does not freeze reward statistics. For comparisons across checkpoints, a deterministic **raw** return measured in a fixed environment is easier to interpret than the evolving normalized training signal.

## Policy, critic and discounted return

The actor and critic use hidden layers of 512, 256 and 128 units with ELU activations in the Microduck model. The Gaussian actor defines a distribution over fourteen actions:

$$\pi_\theta(a\mid o)=\mathcal N\!\left(\mu_\theta(o),\operatorname{diag}(\sigma_\theta^2)\right).$$

Training samples this distribution to explore. Deterministic exported evaluation uses its mean. An initial action standard deviation is therefore an exploration setting, not a motor-noise or hardware-safety guarantee. A resumed checkpoint loads its learned log standard deviation; a displayed `initialStd` does not automatically reset that learned quantity.

The critic predicts discounted future reward:

$$V_\phi(o_t)\approx\mathbb E\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid o_t\right].$$

For intuition, $1/(1-\gamma)$ is an effective discount scale, not the episode length. At $\gamma=0.99$, that scale is about 100 control steps or two seconds; at 0.995 it is about 200 steps or four seconds. Long episodes still matter: the policy must survive and repeat good decisions beyond this approximate horizon.

Reward shaping determines what optimization prefers. A reward $r_t=\sum_i w_i f_i(s_t,a_t,c_t)$ combines feature terms with weights. The weights are not probabilities and need not sum to one. Before increasing a weight, inspect the feature's units, sign, range and how often the required states occur. No coefficient can make PPO learn from an aerial phase or a large swing that never appears in its sampled experience.

## GAE: distinguish a fall from a timeout

GAE stands for **Generalized Advantage Estimation**. PPO needs an advantage estimator but does not require GAE specifically; this implementation uses it to combine one-step TD residuals with information from later transitions. The dedicated **GAE Workshop** develops the bias–variance intuition, lambda endpoints, a corrected three-transition numerical example, and the code-level distinction between a true terminal, an external timeout and a rollout-buffer cutoff.

Let $d_t$ indicate a true terminal transition and $u_t$ a time-limit truncation. These belong to the transition after applying $a_t$, not to the next freshly reset episode. The critic bootstrap is:

$$b_t=\begin{cases}
0 & d_t=1,\\
V(o^{\mathrm{final}}_{t+1}) & d_t=0,\ u_t=1,\\
V(o_{t+1}) & \text{otherwise}.
\end{cases}$$

Then the temporal-difference error and generalized advantage estimate are:

$$\delta_t=r_t+\gamma b_t-V(o_t),$$
$$\widehat A_t=\delta_t+\gamma\lambda\left(1-\mathbf1[d_t\lor u_t]\right)\widehat A_{t+1}.$$

The value target is $\widehat R_t=\widehat A_t+V(o_t)$. The repository's `compute_generalized_advantage_estimate` uses this transition-aligned structure. A time limit bootstraps from the terminal observation but stops the advantage recursion; a fall both stops recursion and removes the bootstrap. Autoreset observations must not leak the next episode into the previous one's target.

**Numerical exercise.** Take $r_t=1$, $V(o_t)=3$, $\gamma=0.99$, final value 4 and next-episode reset value 100. A timeout has $\delta_t=1+0.99(4)-3=1.96$. A true termination has $\delta_t=-2$. Accidentally using the reset value gives 97, a large fictitious advantage. This is a data-boundary bug, not a hyperparameter problem.

## PPO's clipped update

![PPO iteration. The rollout batch is reused for a bounded number of optimization passes, then replaced by newly sampled experience.](assets/ppo-loop.svg)

The old policy generated the batch. For an action stored in that batch, define the likelihood ratio:

$$\rho_t(\theta)=\exp\!\left(\log\pi_\theta(a_t\mid o_t)-\log\pi_{\mathrm{old}}(a_t\mid o_t)\right).$$

PPO maximizes the conservative surrogate

$$L^{\mathrm{clip}}=\mathbb E_t\left[\min\left(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\widehat A_t\right)\right].$$

The implementation minimizes its negative. For positive advantage, an action was better than predicted; increasing its probability helps until the clipped surrogate stops rewarding an excessively large increase. Negative-advantage samples apply the complementary pressure. Clipping is not a proof of monotonic return improvement.

This short fragment is an explanatory NumPy equivalent of the policy-loss calculation, not a replacement optimizer:

```python
log_ratio = np.clip(new_log_prob - old_log_prob, -20.0, 20.0)
ratio = np.exp(log_ratio)
surrogate = ratio * advantage
bounded = np.clip(ratio, 1 - clip_coef, 1 + clip_coef)
policy_loss = -np.minimum(surrogate, bounded * advantage).mean()
```

The actual MLX code also normalizes minibatch advantages when configured. The log-ratio bound prevents an overflowing exponential before gradient clipping can act. It is a numerical guard separate from PPO's policy-ratio clip interval.

## The actual critic loss is robust, not plain MSE

Do not teach a generic squared-error objective as if it were the exact current code. RLX uses a Huber-style value error with $\kappa=10$:

$$h_\kappa(e)=\begin{cases}\frac12e^2,&|e|\le\kappa,\\
\kappa(|e|-\frac12\kappa),&|e|>\kappa.\end{cases}$$

With value clipping enabled, form $V_c=V_{\mathrm{old}}+\operatorname{clip}(V_{\mathrm{new}}-V_{\mathrm{old}},-\varepsilon,\varepsilon)$ and use

$$L_V=\mathbb E\left[\max\left(h_\kappa(V_{\mathrm{new}}-\widehat R),h_\kappa(V_c-\widehat R)\right)\right].$$

The minimized total objective is

$$L=-L^{\mathrm{clip}}+c_V L_V-c_H\mathcal H(\pi).$$

The selected recipes have entropy coefficient $c_H=0$, while entropy is still measured and the Gaussian policy still samples actions. Zero entropy bonus does **not** mean zero exploration. `value_coefficient` defaults to 0.5 in core `PPOConfig`, but the Studio entry point supplies its own default of **1.0**. The selected Studio runs therefore use 1.0 unless explicitly overridden at that lower-level CLI. A scalar “loss” combines these terms with different scales; it is not a universal score comparable across different rewards.

## Count the work correctly

For $N$ environments, $T$ steps per rollout, $M$ minibatches and $E$ epochs:

$$B=NT,\qquad B_{\mathrm{mini}}=B/M,\qquad U_{\mathrm{iteration}}=EM.$$

The **recorded E2E recipe overrides**, not the UI's current default presets, use $N=16,T=128,M=4,E=4$ for Dance/Running/Stilts: 2,048 transitions per iteration, minibatches of 512, and sixteen optimizer updates. Dance's 4,001,792 transitions give 1,954 iterations and 31,264 optimizer minibatches. The recorded Swing recipe uses $T=256,E=2$: 4,096 transitions per iteration, minibatches of 1,024 and eight updates. Its 524,288-transition refinement has 128 iterations and 1,024 optimizer minibatches. Behavior-cloning gradient steps are not part of this PPO count. Use the parameter chapter to distinguish these selected settings from current UI defaults.

More epochs reuse the same experience more heavily; they do not generate additional physical transitions. More environments change throughput and sample correlation. At a fixed total transition budget, increasing batch size reduces the number of policy refreshes. Tune these together rather than treating them as independent “more is better” sliders.

## Read diagnostics without inventing convergence

| Signal | What it measures | What it cannot establish |
|---|---|---|
| Raw deterministic return | Reward sum under a fixed evaluation setup | Skill if the reward can be exploited |
| Normalized rollout reward | Stochastic training reward on an evolving scale | Direct comparability across normalization histories |
| Policy/value/total losses | Actual minibatch optimization objectives | Successful motion or a requirement of monotonic decline |
| Approximate KL | Policy-distribution change on sampled data | A guarantee on unseen states |
| Clip fraction | Fraction of ratios outside the PPO interval | A target that should always be zero |
| Entropy | Spread of the sampled action distribution | Task-relevant exploration by itself |
| Explained variance | Agreement of critic values with sampled return targets | Actor competence; negative values can coexist with a passing actor |

Explained variance is approximately $1-\mathrm{Var}(\widehat R-V)/\mathrm{Var}(\widehat R)$. A negative value means a poor critic fit relative to a constant predictor on that data. Swing's critic is not consistently well explained; its physically validated actor must not be used to claim perfect value convergence.

The saved plots retain raw measurements and label any smoothing; they are not redrawn to look successful. Compare policies on the **same** raw reward/environment, identify whether a curve includes a continuation or a fresh start, and inspect the physical tracking plot next. An upward reward trend followed by a failed geometry test is a failed skill result, not an inconvenient outlier to erase.

## A numerical lesson from a failed Stilt run

An earlier Stilt run stopped with nonfinite critic gradients. A separate probe demonstrated an ELU implementation issue: an inactive exponential branch can overflow and poison gradients even when the forward output looks finite. The stable form used in the Microduck model is conceptually:

```python
def stable_elu(values):
    negative_branch = mx.exp(mx.minimum(values, 0.0)) - 1.0
    return mx.where(values >= 0.0, values, negative_branch)
```

The original failing minibatch was not preserved, so the probe supports a plausible mechanism rather than proving every detail of that historical failure. The lesson is precise: check forward values, gradients and updated parameters; save failure evidence; do not treat gradient clipping as a cure for a NaN that already exists.

### Foundation exercises

1. Compute batch size and optimizer updates for each selected recipe. Verify Dance and Swing counts against the arithmetic above.
2. Implement the one-transition timeout example in plain Python. Make termination take precedence when both flags are true.
3. For advantages +2 and −2, calculate the clipped objective at ratios 0.7, 1.0 and 1.3 with $\varepsilon=0.2$. Explain which direction of movement is discouraged.
4. Read an exported-policy audit's native/ONNX parity number. Explain why a small action error tests export fidelity, not physical robustness.
5. Design a reward ablation with one changed coefficient and an unchanged physical evaluator. Predict a failure mode before running it.
