### What was tested, and what was not

This extends the Dance workflow to the Studio **Running**, **Stilt Walking**, and **Swing** recipes. The executable chain is real HTTP training, checkpoint export, deterministic skill evaluation, MuJoCo rendering, artifact download, H.264 decoding, and Chromium playback/seek. The separate audit uses the exported ONNX actor, not a scripted motion replay. Its comparison video reconstructs the recorded physics states without resetting a fallen control. The displayed zero-action failure is retained, not replaced by a new episode.

All final tests use the nominal XML actuator model. They do not establish BAM actuator equivalence, randomized robustness, arbitrary-speed command following, or hardware readiness. The locomotion recipes use fixed, observable forward commands; the stilt result applies to the selected **2 cm, blend 0, 0.014 kg-per-foot morphology**, not every height offered by the application. Swing uses planar action restrictions as declared in the recipe, starts at zero angle and angular rate, and must retain positive tension in both compliant strings at every 50 Hz sample.

The five final audit seeds are additional reset checks, not five independently trained policies. Swing's nominal reset is effectively identical across seeds, so its repeated passes demonstrate execution reproducibility, **not five independent robustness trials**. Locomotion resets vary, but all share the stated nominal configuration. Checkpoint-development seeds were used for tuning; final audit seed lists and all failed attempts remain visible in the evidence tables.

### Why the early locomotion attempts failed

Running v1 learned standing. Running v2 learned fast grounded stepping but did not satisfy the aerial-phase gate. Running v3 acquired an actual aerial gait around 0.7 m/s, but its trajectories often curved far enough to fail net intended displacement. A high body-frame speed alone was therefore insufficient. The evaluator was strengthened before final selection to measure signed progress against the intended command heading, rather than crediting accumulated travel around a circle. The acceptance floors were not reduced to accommodate these failures.

The optional `flight` term is zero by default and rewards both feet being airborne only with upright posture and command-directed speed. Its bounded form prevents stationary hopping or a falling robot from collecting full credit. Running's optional `yaw_tracking` term isolates yaw-rate error from the existing combined roll/pitch/yaw Gaussian: `exp(-((gyro_z - commanded_yaw_rate) / 0.25)^2)`. This gives a yaw-specific signal while leaving the native gyro, gravity, joint, prior-action, and command observations unchanged. It does not expose absolute world heading to the actor or reward an unobservable heading target. The same opt-in term is supported for stilts, but the selected Stilt v3 recipe does **not** enable it.

### Stilt Walking: PPO acquisition followed by refinement

The Stilt v2 base was trained from random initialization for 6,000,640 transitions. It learned moving, alternating-foot balance on the selected physical stilt geometry, but many episodes circled and failed intended progress. Stilt v3 resumes that base for 1,048,576 additional PPO transitions and increases the existing angular-velocity tracking weight from 0.5 to 4. The inherited exploration standard deviation is loaded from the checkpoint; the recipe's initial-standard-deviation field does not reset a resumed actor.

The final API passes all 16 episodes. The separate seeds 501–505 pass all five episodes with mean forward speeds 0.230–0.242 m/s and intended displacements 1.281–2.289 m over ten seconds, without a fall. On comparison seed 501, deterministic raw return rises by about 758 under the **same final reward configuration**. This is stronger evidence of beneficial refinement than merely showing changed weights. The final gait still has measurable curvature; passing the defined progress floor does not mean perfectly straight tracking.

### Swing: distinguish teacher acquisition from PPO improvement

Fresh PPO in Swing v1 did not discover the large-amplitude motion: its strict still-start span was only about 2.23 degrees. Rather than crediting assisted resets as success, a bounded controller feasibility search and behavior-cloning/DAgger initializer were used to place useful motion in the policy's reachable distribution. Privileged physical state supplies teacher labels during data collection only. The exported actor still receives the ordinary **61 inputs and produces 14 actions**; no teacher controller is called at evaluation time. The bootstrap has `ppo_steps=0` and must not be described as PPO learning.

The selected stage-three initializer already passes the Swing skill gate. PPO therefore refines an acquired skill; it does not acquire Swing from scratch in the successful branch. Observation statistics are frozen for this continuation so changing normalization cannot silently alter the competent baseline. The critic and actor are still optimized, rewards remain normalized with separately updated statistics, and all normal PPO finite-value guards remain active.

Swing v2 showed why a final checkpoint cannot be accepted solely because training finished: its 524,288-step checkpoint passed, but continued optimization to 1,048,576 steps caused geometry violations and one sampled loss of string tension. Swing v3 repeats the initialization and PPO recipe with the shorter **524,288-transition budget** selected from that development evidence. Its API result has approximately **162.32 degrees** of symmetric span, 100% valid geometry and tension, and at least **0.209 N** sampled spring tension. This is a model-selection decision with an explicitly retained failed longer run, not a smoothed graph or relaxed threshold. Alignment approaches the 0.05 limit, so the pass is nominal and has limited margin.

### How to read the reward and loss figures

Each figure comes from saved JSONL telemetry or actual deterministic checkpoint rollouts. No reward or loss series is invented, made monotonic, or truncated to remove startup spikes. The reward figure deliberately separates deterministic unnormalized checkpoint return, deterministic survival, normalized stochastic collection reward, and raw stochastic completed-episode return. Those are different measurements: normalizer scale evolves, training actions are sampled, and occasional exploratory falls shorten episodes. A downward normalized training curve is not by itself evidence that the exported deterministic policy got worse.

For example, Stilt v3's deterministic checkpoint return increases while its normalized stochastic reward decreases during part of training. Its occasional raw training-return drops coincide with shortened exploratory episodes; the independent deterministic gait is the acceptance measurement. PPO policy loss is a batch-relative clipped surrogate, not a supervised prediction-error curve. The reports retain value loss, approximate KL, clip fraction, entropy, explained variance, and update duration so a small scalar loss cannot conceal an unstable or uninformative critic. In particular, poor or negative explained variance in Swing is disclosed rather than presented as healthy value-function convergence.

### Numerical repair and evidence integrity

Stilt v1 aborted on nonfinite critic gradients. A separate installed-MLX probe reproduced finite ELU forward outputs but NaN gradients for large positive inputs. Bounding the inactive exponential branch preserves the ELU function and ONNX layout while eliminating that reproduced gradient-overflow mechanism. Regression tests cover eager and compiled gradients, forward parity, checkpoint compatibility, and the unchanged observation/action interface. The exact failed training minibatch was not saved, so the probe establishes a consistent failure mechanism rather than an exact replay of the original failure.

The evidence retains unsuccessful runs and distinguishes a completed pipeline from a learned skill. Null controls must fail, exported ONNX must match the checkpoint, final episodes must meet their physical gates, and video validation must establish actual downloadable and playable files. The native tests, viewer tests, lint/build results, and known broader-suite failures are reported separately; optional GPU-stack collection failures are not recast as passing CPU coverage. Nothing in this report certifies deployment to a real robot.

### Primary references for interpreting PPO

These references explain the optimization and evaluation concepts, not the outcome of these particular runs. This project executes RLX PPO; the SB3 implementation is cited for metric definitions and scaling cautions, not as the engine used here.

1. Schulman et al., *Proximal Policy Optimization Algorithms* (2017), original clipped-surrogate method: <https://arxiv.org/abs/1707.06347>.
2. OpenAI Spinning Up, *Proximal Policy Optimization*, stochastic exploration, policy/value objectives and approximate-KL diagnostics: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>.
3. Stable-Baselines3 2.6.0 PPO source documentation, policy/value/entropy losses, approximate KL, clipping fraction and reward-scaling caveat: <https://stable-baselines3.readthedocs.io/en/v2.6.0/_modules/stable_baselines3/ppo/ppo.html>.
4. Stable-Baselines3, *Reinforcement Learning Tips and Tricks*, separate evaluation of stochastic training policies: <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>.
