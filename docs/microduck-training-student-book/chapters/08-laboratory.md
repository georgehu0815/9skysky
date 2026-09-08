# Evaluation, debugging and the next research project

## Five outputs, five different questions

A complete experiment answers five questions independently. Do not compress them into one green badge.

| Evidence layer | Question | Minimum retained evidence |
|---|---|---|
| Training | Did the intended optimizer run on physical experience? | Recipe, transition counts, JSONL updates, finite parameters |
| Export | Does ONNX reproduce the native actor? | Normalizer-bearing ONNX, checkpoint identity, action parity |
| Skill | Does the robot perform the declared task? | Complete-episode metrics, all gates, controls, failure reasons |
| Media | Does the video actually contain the saved rollout? | Full decode, duration/frame count, source identity, playback/seek |
| Interpretation | Is the claim narrower than its evidence? | Scope, failed attempts, initializer provenance, untested extensions |

The separate `audit.json` files bundled with this book preserve the original checkpoint hashes, effective environment options, controls, physical measurements, and training diagnostics. The paired video-validation receipts preserve the earlier media checks. Fresh `assets/ui-capture.json` records which saved run the browser opened and played during book creation.

## A reproducible evidence notebook

Create a new directory before each experiment. Record the source revision and dirty diff, environment versions, model XML identity, input reference hash, requested recipe, API-normalized recipe, initializer, seed and purpose. Save failed runs alongside successful ones. A useful manifest is not a photograph of a terminal window: it is machine-readable data with enough identity to reject a stale file.

For each trained policy, retain:

```text
run/
  scenario.safetensors
  scenario.safetensors.json
  scenario.onnx
  training-metrics.jsonl
  evaluation.json
  checkpoints/
  render/ep0.mp4
audit/
  audit.json
  video-validation.json
  reward-learning.png
  ppo-losses.png
  physical-tracking.png  (joint-tracking.png for Dance)
  trained-seed*.npz
  zero-seed*.npz
  comparison.mp4
```

The full historical evidence folders are not included automatically by Git. The book copies the figures and audit receipts so it remains readable without those folders, but reproducing a plot from raw samples still requires the original journals/NPZ files or a new training run.

### Read a bundled audit without loading MLX

This complete inspection snippet runs from the workspace root using ordinary Python:

```python
import json
from pathlib import Path

root = Path("docs/microduck-training-student-book/assets")
audit = json.loads((root / "running-audit.json").read_text())
trials = audit["controls"]["trained"]
assert len(trials) == 5
assert all(trial["passed"] for trial in trials)
for trial in trials:
    episode = trial["episodes"][0]
    print(trial["seed"], episode["steps"],
          episode["command_directed_displacement_m"],
          episode["aerial_fraction"])
```

This verifies the saved report contents, not a newly simulated policy. To test a new checkpoint, run the audit command. To establish real-world transfer, neither this snippet nor the local simulator is sufficient.

## Why the controls matter

**Zero action** asks whether the spawn pose or passive physics can satisfy the reward. It does not mean frozen world state; a zero target offset can still fall or move. **Random/untrained policy** asks whether an arbitrary network already scores well. **Loaded initializer** measures what was present before refinement. **Frozen phase** in Dance tests whether dynamic timing actually contributes. A reference animation is a visualization of desired pose, not a physically valid control baseline.

Controls must share the relevant environment and horizon. If one terminates early, label that termination. Do not extend its last frame and imply it survived. If you compare normalized training return to raw deterministic control return, the comparison is invalid even if both axes say “reward.”

## Measuring progress when a robot turns

Running exposed an important difference between body-frame speed, path length and intended displacement. A robot can move quickly forward relative to its own rotating torso while returning near its starting position. Its path length is positive, but it did not obey a zero-turn forward command.

The locomotion evaluator anchors an intended heading for each command segment, advances it only by commanded yaw, and projects **signed world-position increments** onto that intended direction. It also measures body-frame directed velocity and contact transitions. These are complementary checks, not interchangeable formulas. The first position sample establishes the anchor; it does not receive unobserved displacement credit.

The actor need not observe world position for the evaluator to measure it. Observation restrictions constrain the deployed policy's inputs; an independent measurement system may use richer state to judge the outcome. Adding an absolute-heading reward to an actor that cannot infer heading would be a separate design decision, not a free fix.

## Debugging by layer

| Symptom | First diagnostic | Avoid this shortcut |
|---|---|---|
| No checkpoint | API operation state, Python stderr, paths, dependency imports | Increasing training timesteps |
| Immediate NaNs | Observation/action/return ranges, gradients, ELU branch and normalizers | Smoothing the curve or ignoring nonfinite updates |
| Good reward, stationary Dance | Moving-joint variance and static-pose baseline | Calling a balanced pose imitation |
| Fast but grounded Running | Both-foot aerial fraction and liftoff/contact events | Lowering the flight gate to make the run pass |
| High speed, no progress | Signed displacement and heading drift | Using traveled distance instead |
| Large Swing, failed geometry | Tension, lateral error, seat/string alignment at every sample | Reporting peak angle alone |
| Native works, ONNX fails | Input normalization once, tensor dimensions, checkpoint hashes | Re-exporting unrelated weights until something moves |
| MP4 opens but result is wrong | Run name, video URL, source hash, complete horizon | Treating browser playback as a skill test |

Move from the cheapest falsifying test to the more expensive one. A wrong file path should be fixed before rerunning millions of transitions. A valid file with a failed skill gate needs a new learning experiment, not another file-extension check.

## A disciplined improvement ladder

### Level 1: make the measurement reproducible

Rebuild the book, inspect the four saved videos, and explain one acceptance failure from a historical run. Change no reward. Deliver a one-page claim/evidence table with exact units and horizon. This teaches students that an experiment starts with a test, not with a neural network.

### Level 2: reproduce one selected recipe

Use a fresh run name, the exact clip or morphology, and the complete initializer lineage. Keep nominal dynamics and the same evaluator. Expect numerical differences across hardware/software revisions; a published count is not a guarantee of a bit-identical trajectory. Report pass/fail and the magnitude of differences instead of altering thresholds retrospectively.

### Level 3: one-variable ablation

For Dance, vary pose precision while preserving the excerpt and physical gate. For Stilts, compare angular tracking weights at the same morphology. For Running, remove the yaw-only term and predict circling before collecting data. For Swing, compare a frozen and updating observation normalizer from the **same** bootstrap. These are proposed experiments. Do not imply that the book's evidence already proves their outcomes.

### Level 4: held-out robustness

Use separate development and test seeds. The historical reset-seed sets in this book have been inspected and used in experiment development; once students repeatedly tune against them, they are no longer an untouched test set. Reserve new seeds, disturbances and commands before selecting the winner. If nominal resets have no effective randomness, introduce a declared controlled perturbation rather than claiming statistical independence from different integers alone.

Train multiple seeds when studying reliability of the learning algorithm. Resetting one actor five times estimates different behavior than training five independent actors. Keep both costs and both uncertainty sources explicit.

### Level 5: transfer the design, not the claim

Port the promising task to the official `microduck_rl` training stack with its actuator modeling, randomization and hardware-specific validation. The nominal Swing teacher showed poor BAM behavior in the separate feasibility search, so actuator transfer is a real missing capability, not paperwork. An ONNX file matching 61 inputs and 14 outputs is necessary for interface compatibility; it is not permission to put the policy on hardware.

## Scenario-specific advanced projects

| Project | What to change | How to decide whether it helped |
|---|---|---|
| Dance phrase curriculum | Progress from eight seconds to longer phrases; validate phase transitions and loader coverage | New complete-phrase audits, static baselines, balance and continuity; no claim about ignored root channels |
| Dance tempo conditioning | Expose a tempo command using an explicitly documented compatible slot convention | Held-out tempos without retiming the evaluation trace after the fact |
| Stilt morphology generalization | Controlled height/mass/blend grid or morphology-conditioned actor | Held-out geometries, clearance/contact checks, consistent units and evaluation morphology |
| Running generalization | Multiple speed commands and turning segments | Correct command-segment progress, flight and stability across unseen commands |
| Running efficiency | Add measured mechanical energy/torque constraints | Compare efficiency only among policies already passing the same skill gate |
| Swing scratch exploration | Progressively reduce teacher data or design a clearly labeled training curriculum | Still-start skill without inference-time teacher; separate data and PPO budgets |
| Swing robustness margin | Perturb dynamics and penalize unsafe near-boundary geometry | Preserve bidirectional span while improving minimum tension and alignment margins |

A curriculum can change the training distribution without relaxing the final evaluation. However, label assisted resets and teacher labels. If the actor fails from stillness without the assistance, the final skill remains unlearned under the original task.

## Selected exercise answers

**Stilt contact question:** $480/500=0.96$ upright coverage passes, but right-foot air fraction 0.04 fails the 0.05 minimum. The episode fails even if every other measure is excellent.

**Running speed question:** $0.30/0.75=0.40$ passes the 0.35 speed-ratio minimum. Absolute directed speed 0.30 m/s fails the 0.35 m/s floor. One cannot substitute the ratio for the speed floor.

**Swing span question:** peaks +90° and −40° give 130° peak-to-peak but only $2\min(90,40)=80$° symmetric span. A spectacular one-sided excursion is insufficient.

**Dance static baseline:** if tracking RMSE is 0.08 rad and the matching moving-joint static baseline is 0.10 rad, gain is $1-0.08/0.10=0.20$. It reaches the dynamic threshold but still needs every other gate, including leg motion and balance. Compare the same selected joints in numerator and denominator.

## Capstone: a defensible experiment report

Submit a Markdown report with: hypothesis; exact input and observation semantics; reward equation; requested/effective parameters; full checkpoint lineage; budget in transitions and optimizer steps; reward/loss/physical figures; per-episode gates; negative controls; full MP4 and validation receipt; failed attempts; and limitations. Add a short paragraph explaining what evidence would disprove your claim.

Suggested grading: 25% reproducibility, 25% physical evaluation, 20% mathematical/code explanation, 20% interpretation and honest limitations, 10% clarity of figures and media. A failed skill experiment can earn a strong grade if it is reproducible and teaches a falsifiable lesson. A polished video with untraceable weights cannot.

# Evidence index and implementation reading map

## Original rollouts

These links point to the local full-resolution source videos. If the large evidence directories were not shared, recreate them using the reproduction appendix instead of expecting them in a fresh clone.

- [Dance: sixteen uninterrupted seconds, two excerpt cycles](../../rlx/artifacts/dance-e2e-20260907/final-audit/extended-render/ep0.mp4).
- [Dance: reference, PPO and null comparison](../../rlx/artifacts/dance-e2e-20260907/final-audit/comparison.mp4).
- [Running: complete API-delivered twelve-second rollout](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/api-video.mp4).
- [Stilts: complete API-delivered ten-second rollout](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/api-video.mp4).
- [Swing: complete API-delivered twenty-four-second rollout](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/api-video.mp4).

The cover captures frames at Dance 3 s, Running 6 s, Stilts 5 s and Swing 18 s. The exact MP4 hashes and frame provenance are in [asset-manifest.json](asset-manifest.json). These are native 640×360 video frames, not fabricated high-resolution stills. On the cover each occupies about 3.69 inches horizontally, approximately 174 source pixels per inch. Design diagrams and equations are vector; original reward/loss figures retain their native pixel dimensions; UI screenshots were captured at three device pixels per CSS pixel.

## Read code in dependency order

| File | Study question |
|---|---|
| `microduck_local/src/microduck_local/contract.py` | Where do observation slots, joint order and timing come from? |
| `rlx/rlx/environments/microduck_recipes.py` | How does a scenario construct and instrument its environment? |
| `rlx/rlx/environments/microduck.py` | Where do normalization, autoresets and terminal observations cross the vector boundary? |
| `rlx/rlx/models/microduck.py` | How are policy, critic, stable ELU and checkpoint metadata defined? |
| `rlx/rlx/utils/utils.py` | How does GAE stop at resets while bootstrapping timeouts? |
| `rlx/rlx/algorithms/ppo.py` | Where are clipped objectives, robust critic loss and finite-update guards implemented? |
| `rlx/examples/ppo_microduck_studio.py` | Which defaults does Studio actually supply to PPO? |
| `duck-viewer/lib/rlx-job.ts` | Which API values are clamped, overridden or rounded? |
| `rlx/rlx/environments/dance_evaluation.py` | How are moving-joint baselines and temporal coverage measured? |
| `rlx/rlx/environments/locomotion_evaluation.py` | How do intended progress and aerial/contact gates prevent reward exploits? |
| `rlx/rlx/environments/swing_evaluation.py` | How are continuous geometry and symmetric span assessed? |
| `rlx/scripts/bootstrap_swing_e2e.py` | How are privileged teacher labels separated from actor inference? |

## Verification scope and remaining gaps

The source experiment reports record native RLX tests (418 passed, 2 skipped), Node tests (74 passed), and report-generator tests (18 passed), alongside successful targeted lint/build checks. They also record **12 existing broader local-harness test failures** and optional mjlab collection failures where that dependency was not installed. These are historical validation records, not a claim that the whole workspace currently passes every possible suite.

Book generation checks source/asset links, parameter coverage, preserved image hashes, script syntax, readable PDF output, mathematical typesetting and saved-run browser playback. It does not replace simulation audits or repeat expensive training. Consult `validation.json` for the fresh book-build checks and the original reports for experiment execution evidence.

## Further study

The executable repository is the source of truth for the exact formulas taught here. For background, search the original paper **Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347 (2017)** and **Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” arXiv:1506.02438 (2015)**. Treat a paper's generic objective as context, then read the current implementation for additions such as robust value loss and numerical guards. This book's measured outcomes come from local audit artifacts, not from those papers.
