# GAE Workshop: Why PPO Uses Advantages and How Timeouts Change Them

## Start with the question PPO must answer

**GAE means Generalized Advantage Estimation.** It estimates whether a sampled action led to a better or worse outcome than the policy's value baseline predicted. PPO needs an advantage estimate for its policy-gradient objective, but **does not require GAE specifically**. Returns minus a baseline and other estimators are possible. This RLX implementation chooses GAE to combine information from multiple future rewards without relying entirely on either a noisy complete return or a single critic prediction. See the original [GAE] and [PPO] references in the practicum bibliography.

Consider a Running action that briefly lifts both feet. Its immediate reward can look promising while the robot is already rotating toward a fall. Conversely, a stabilizing action may receive little immediate reward but enable useful motion later. PPO needs a credit-assignment signal for the sampled action, not just the current reward or a final episode-wide thumbs-up.

The theoretical advantage is

$$A^\pi(o_t,a_t)=Q^\pi(o_t,a_t)-V^\pi(o_t),$$

where $Q^\pi$ is the expected discounted return after taking the particular action and following the policy, and $V^\pi$ averages over the policy's actions from that observation. In a partially observed robot, $V(o_t)$ is the feed-forward critic's observation-based approximation; it is not an oracle for the complete simulator state or unobserved history.

An advantage of `+0.4` means better than that baseline, not “the robot earned exactly 0.4 reward” or “move the joint by 0.4 radians.” An advantage of `-0.4` means worse than the baseline, even if the immediate reward was positive. Estimates can be wrong when the critic, boundary handling, or sampled data is poor.

## When GAE runs—and when it does not

In `rlx/rlx/algorithms/ppo.py`, one training iteration has four stages:

1. **Collect:** use the current stochastic actor to gather a full vectorized rollout, saving rewards, old values, old action log probabilities, termination flags and timeout information.
2. **Estimate:** evaluate the last-state critic and compute GAE backward over the rollout. Form critic return targets as `advantages + old_values`.
3. **Optimize:** shuffle samples into minibatches. PPO uses the advantages in its clipped actor objective and the return targets in its critic loss. In this implementation, advantage normalization occurs inside each optimizer minibatch.
4. **Repeat:** discard the old rollout and collect new experience under the updated policy.

GAE belongs between collection and PPO optimization. It is not an extra action-selection network and it does not create more environment transitions. Two update epochs reuse the same computed rollout advantages; they are not two independent GAE datasets or two new physical rollouts.

The selected Swing refinement performs this computation for 128 rollouts of shape `(256,16)`, one scalar advantage for each of its 524,288 transitions. Dance, Running and Stilts use the same mechanism with their selected rollout horizons and rewards. Swing's earlier BC/DAgger stages do **not** need GAE: they train the actor mean against teacher action labels, and the selected initializer's critic is not pretrained. Deterministic ONNX inference and physical skill evaluation also do not compute GAE to choose an action.

## Build the estimate from TD residuals

At a transition that genuinely continues, the one-step temporal-difference residual is

$$\delta_t=r_t+\gamma V(o_{t+1})-V(o_t).$$

Interpret this as “reward just observed plus predicted remaining return, minus what the critic predicted before acting.” A positive residual is better news than the old prediction. GAE combines this news with subsequent residuals:

$$\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.$$

That compact expression assumes an appropriately continuing trajectory or terminal boundary. Real implementations use a **finite collected segment**, explicit episode-boundary masks, and a critic bootstrap at a nonterminal cutoff. They must not sum residuals from the next reset episode.

For three consecutive transitions with no intervening reset, the finite backward sweep is

$$\widehat A_2=\delta_2,$$
$$\widehat A_1=\delta_1+\gamma\lambda\widehat A_2,$$
$$\widehat A_0=\delta_0+\gamma\lambda\widehat A_1.$$

The final residual already contains the correct boundary bootstrap. Initializing the backward sweep's future advantage to zero does **not** imply that the final state's value should also be zero. Those are different quantities.

## What gamma and lambda each control

- **Gamma, $\gamma$:** discounts future rewards and affects the value objective being estimated. It is not a noise filter applied only to GAE.
- **Lambda, $\lambda$:** controls how strongly later TD residuals influence the current advantage. It changes the estimator without changing the environment's reward function.
- **Their product, $\gamma\lambda$:** is the per-transition weight in the residual trace. With `gamma=0.99` and `lambda=0.95`, this is exactly `0.9405`.

| Lambda | What the estimator becomes | Qualification |
|---|---|---|
| `0` | One-step TD advantage, $\widehat A_t=\delta_t$ | Usually less sampling variance but stronger reliance on an approximate next-state critic |
| Between `0` and `1` | A weighted multi-step residual trace | Trades sampling variance against bootstrap approximation; no universal best value |
| `1` | Discounted multi-step return minus the initial value baseline | Pure empirical return only when the segment reaches a true terminal boundary; otherwise a final critic bootstrap remains |

For a segment with $K$ transitions and no earlier boundary, setting $\lambda=1$ makes the intermediate value terms telescope:

$$\widehat A_t=
\sum_{l=0}^{K-1}\gamma^l r_{t+l}
+\gamma^K V(o_{t+K})-V(o_t).$$

At a true terminal, the continuation value is zero by definition, leaving the usual Monte Carlo return minus baseline. At a nonterminal rollout cutoff or external timeout, removing the last value term would change the target. Therefore “lambda one is always Monte Carlo” is too imprecise for this simulator. Similarly, “lambda zero is always highly biased” ignores the quality of the value function: the issue is reliance on its approximation, not a guaranteed fixed error.

## Timeout-aware means two different boundary decisions

A **true terminal** ends the return being modeled: for example, a fall that ends the task under its defined failure rule. An **external timeout** stops collection even though the modeled process could continue. Both may trigger an implementation reset, so “the simulator reset” is not sufficient to distinguish them.

The rule for a true terminal is to assign zero **continuation value**, not to declare its observation meaningless. That final observation can still be valid and useful for debugging and evaluation. What is zero is the future return beyond the task's terminal boundary.

Let $d_t$ be `terminated` and $u_t$ be `truncated`. These are flags for the transition just completed. Define

$$b_t=\begin{cases}
0,&d_t=1,\\
V(o^{\mathrm{final}}_{t+1}),&d_t=0\ \text{and}\ u_t=1,\\
V(o_{t+1}),&d_t=0\ \text{and}\ u_t=0.
\end{cases}$$

Then compute

$$\delta_t=r_t+\gamma b_t-V(o_t),$$
$$\widehat A_t=\delta_t+\gamma\lambda\bigl(1-\mathbf1[d_t\lor u_t]\bigr)\widehat A_{t+1}.$$

**Decision one:** may the TD residual bootstrap a final value? **Decision two:** may the advantage trace continue into the next stored transition? A timeout answers **yes** to the first but **no** to the second when the next stored transition belongs to a reset episode.

| Boundary | Value used in the last residual | Propagate the next episode's advantage? |
|---|---|---|
| Ordinary transition | Next-state value | Continue within the same episode |
| External timeout with autoreset | Value of the saved final observation | No |
| True terminal | Zero continuation value | No |
| Both flags true | Zero; termination takes precedence | No |
| Rollout buffer ends but episode continues | `last_value` from the ongoing state | No observed tail beyond this segment; initialize the backward tail to zero |

The last row is not a reason to mark the environment `terminated`. A 256-step Swing rollout can finish in the middle of a 1,200-step episode. The next rollout may continue that world, but this finite estimator still needs its current last-state bootstrap.

Not every clock-based ending is an external timeout. If a finite deadline is **part of the task definition**, it can be a true terminal; the state/observation needs the remaining-time information for the usual Markov formulation. Do not change an environment's flags based only on the word “timeout” in a log. The official Farama time-limit documentation makes this distinction [TimeLimits]. This RLX path follows the flags and terminal observations actually returned by its adapter.

![Where boundary-aware GAE sits in PPO. The terminal value decision and cross-reset trace decision are separate operations. This is a teaching diagram, not measured robot data.](assets/gae-boundaries.svg)

## The supplied three-transition example, corrected exactly

Use the following **synthetic** data. These are not saved Microduck reward measurements, and they are not normalized again inside this example:

$$\gamma=0.99,\qquad\lambda=0.95,\qquad\gamma\lambda=0.9405.$$

| Transition | Current value | Reward | Value after applying the action | Boundary |
|---|---|---|---|---|
| `t=0` | $V(o_0)=1.0$ | $r_0=0.5$ | $V(o_1)=1.2$ | Continues |
| `t=1` | $V(o_1)=1.2$ | $r_1=0.2$ | $V(o_2)=1.1$ | Continues |
| `t=2` | $V(o_2)=1.1$ | $r_2=0.0$ | $V(o_3^{\mathrm{final}})=0.9$ | External timeout |

There are three transitions, indexed `0,1,2`, and four observation values. The timeout occurs **after the third action**, producing the final observation $o_3$. That final observation is not the subsequently reset observation.

### First compute every TD residual

At the timeout:

$$\delta_2=0+0.99(0.9)-1.1=0.891-1.1=\boxed{-0.209}.$$

The value `-0.211` in the motivating draft is an arithmetic error, not a rounding convention. Keeping it would incorrectly change every preceding advantage.

At the earlier transitions:

$$\delta_1=0.2+0.99(1.1)-1.2=\boxed{0.089},$$
$$\delta_0=0.5+0.99(1.2)-1.0=\boxed{0.688}.$$

### Sweep backward without rounding intermediate values

$$\widehat A_2=\delta_2=\boxed{-0.209},$$
$$\widehat A_1=0.089+0.9405(-0.209)=\boxed{-0.1075645},$$
$$\widehat A_0=0.688+0.9405(-0.1075645)=\boxed{0.58683558775}.$$

Rounded to six decimals, the forward-ordered advantage vector is

```text
advantages = [0.586836, -0.107565, -0.209000]
```

Notice the credit assignment: `t=1` has a positive immediate reward and positive one-step residual, but a negative multi-step advantage because the later outcome is disappointing relative to its value prediction.

The critic's return targets are $\widehat R_t=\widehat A_t+V(o_t)$:

```text
value_targets = [1.58683558775, 1.0924355, 0.891]
```

At the final timeout, the critic target is `0.891`, exactly the zero immediate reward plus discounted final value. RLX fits these targets with its robust critic loss; the actor uses the advantages in the clipped objective. These displayed advantages are **before** the source's optimizer-minibatch normalization. A normalized advantage may have a different sign relative to the minibatch mean, so do not claim this tiny vector directly predicts every production gradient.

## Now make the third transition a true terminal

Keep the same recorded rewards and current values for a controlled mathematical comparison, but change only the final boundary semantics. In an actual newly defined terminal task, one would also need a value function trained for that objective; holding values fixed here isolates the mask's effect.

$$\delta_2^{\mathrm{terminal}}=0-1.1=-1.1,$$
$$\widehat A_1^{\mathrm{terminal}}=0.089+0.9405(-1.1)=-0.94555,$$
$$\widehat A_0^{\mathrm{terminal}}=0.688+0.9405(-0.94555)=-0.201289775.$$

| Quantity | Correct external-timeout calculation | Zero-bootstrap terminal calculation |
|---|---|---|
| $\widehat A_0$ | $0.58683558775$ | $-0.201289775$ |
| $\widehat A_1$ | $-0.1075645$ | $-0.94555$ |
| $\widehat A_2$ | $-0.209$ | $-1.1$ |
| $\widehat R_2$ | $0.891$ | $0$ |

If the boundary really was an external timeout, using the right column would be a bug. Here it even flips the **unnormalized** first advantage's sign. This illustrates why a final-step mask error can affect actions earlier in a rollout, not only its last entry.

The error is not always downward. At the boundary,

$$\delta_{\mathrm{wrong}}-\delta_{\mathrm{correct}}
=-\gamma V(o^{\mathrm{final}}).$$

With final value `+0.9`, the error is `-0.891`. With final value `-0.9`, it is `+0.891`. If the final value is zero there is no difference for that residual. Correct handling avoids the **wrong objective at a collection boundary**; it does not guarantee smoother curves, a well-fitted critic, PPO stability or successful robot motion.

## A lambda comparison on exactly the same data

| Lambda | $\widehat A_0$ | $\widehat A_1$ | $\widehat A_2$ |
|---|---|---|---|
| `0` | $0.688$ | $0.089$ | $-0.209$ |
| `0.95` | $0.58683558775$ | $-0.1075645$ | $-0.209$ |
| `1` | $0.5712691$ | $-0.11791$ | $-0.209$ |

For `lambda=1`, explicitly verify the nonterminal boundary term:

$$\widehat A_0=0.5+0.99(0.2)+0.99^2(0)+0.99^3(0.9)-1=0.5712691.$$

Dropping the final value would instead give `-0.302`. That latter number is the complete-terminal return-minus-baseline for these three rewards, not the correct external-timeout target. The example shows why an endpoint slogan about lambda is not enough; you must also state where the trajectory ends.

## Map the lesson to the actual RLX code

| Source location | Responsibility | Boundary failure it prevents |
|---|---|---|
| `rlx/rlx/environments/microduck.py` | Preserve the timeout's finite 61-vector as `terminal_observation`; normalize it with `update=False`; provide its validity mask | Using a reset observation or updating statistics just to score a saved final state |
| `PPO._truncation_values` in `rlx/rlx/algorithms/ppo.py` | Require `terminal_observation` and `_terminal_observation` for real timeouts, evaluate their critic values | Silently guessing a missing terminal value |
| `PPO.train` in the same file | Collect timeout values, evaluate `last_value`, call GAE, construct return targets | Confusing a rollout cutoff with an episode termination |
| `compute_generalized_advantage_estimate` in `rlx/rlx/utils/utils.py` | Reverse-scan rewards, values and flags with separate bootstrap and trace rules | Carrying advantage from a reset episode into its predecessor |
| `PPO._loss_and_metrics` | Normalize minibatch advantages and consume fixed advantage/return targets | Mistaking GAE for an action-selection or reward-generation stage |

At the vector level, these operations are per environment. If one of sixteen worlds falls while another times out, their masks must differ. Do not terminate the whole batch because any environment ended. Likewise, `info["terminal_observation"]` is a name for a stored boundary observation; it does not imply that every corresponding transition has `terminated=True`.

The backward scan is numerical credit-assignment arithmetic, **not backpropagation through time through the robot simulator**. The feed-forward actor does not become recurrent because GAE scans a trajectory backward. PPO subsequently differentiates its network loss; this implementation does not differentiate through MuJoCo physics to compute the advantages.

## Run and inspect the tested example

From the workspace root, run:

```bash
python3 docs/microduck-training-student-book/learning_lab.py --gae
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/test_book.py
```

`--gae` prints the inputs, TD residuals, advantages, critic targets, terminal comparison and all three lambda settings. It uses only the Python standard library and does not train, run MuJoCo, or contact a robot. Floating-point output may contain a tail such as `-0.20900000000000007`; the values above are the corresponding exact decimal arithmetic, not extra measured precision.

Read `scalar_gae()` in [learning_lab.py](learning_lab.py) alongside the vectorized RLX function. `gae_walkthrough()` deliberately passes `last_value=99.0` in the autoreset examples: the timeout/termination branches must ignore that reset-state value. For a genuinely unfinished rollout without a boundary flag, passing `last_value=0.9` reproduces this three-transition timeout estimate, because both use the same real boundary value and have no observed advantage tail beyond the segment.

The tests check the exact corrected numbers, critic targets, true-terminal comparison, lambda endpoints, nonterminal rollout bootstrap, termination precedence, no cross-reset advantage propagation and the sign of the error when a negative timeout value is dropped. The small scalar tests supplement—not replace—the native trainer's vectorized integration tests.

### Student exercises with answer checks

1. **Explain the pipeline:** which PPO stage consumes GAE? Answer: the actor update consumes advantage estimates; the critic fits `advantage + old_value`. Neither the BC label loss nor exported action inference needs this calculation.
2. **Change the final value:** set it to zero without changing the flags. At this fixed set of earlier values, the timeout and terminal calculations now coincide. Explain why that numerical coincidence does not make the flags interchangeable.
3. **Detect reset leakage:** make the following episode's first reward enormous. The prior timeout's advantage must not change. Its terminal observation still supplies a value bootstrap, but its trace stops.
4. **Check the endpoint:** derive `0.5712691` for lambda one, including `gamma**3 × 0.9`. Explain why calling it an unbootstrapped Monte Carlo return would be wrong.
5. **Connect to Swing:** why can correct GAE coexist with negative explained variance? Because boundary correctness does not ensure an accurate critic. It also does not prove the actor meets the independent tension and geometry gates.

The practical rule is: **PPO consumes advantages; GAE estimates them; the boundary determines the bootstrap; and the physical evaluator determines whether the robot learned the intended skill.**
