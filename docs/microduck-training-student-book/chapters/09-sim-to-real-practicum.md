# From a Passing Simulation to a Robotics Research Practicum

## What this book has—and has not—trained

The experiments train neural controllers of a simulated Microduck using the robot's shared observation/action dimensions and model assets. They do **not** establish that these four exported actors are ready to execute on a physical Microduck. All selected final audit environments use nominal XML actuators, with domain randomization, observation noise, action delay and random yaw disabled. A green simulated skill badge is meaningful within those conditions, but it is not hardware approval.

The authoritative local playbook, `microduck_local/AGENTS.md`, specifies a progression: prototype locally, port the task into an upstream mjlab configuration, and retrain with the official `microduck_rl` sim-to-real recipe. The upstream README describes BAM actuator physics, friction and backlash modeling, randomization and a deployment contract. This chapter teaches how to prepare that progression; it deliberately does not invent a hardware deployment command for unvalidated Swing/Stilt mechanisms.

![Proposed progression from the established local result to separately reviewed hardware trials. Only the first local-simulation gate is established by the four cases.](assets/transfer-gates.svg)

## Compatibility has three layers

**Tensor compatibility.** Does ONNX accept the expected float tensor and return fourteen actions? Are values finite? Does native/ONNX parity hold? This catches serialization and normalizer mistakes, not a bad behavior.

**Semantic compatibility.** Does slot 59 still mean the same thing? Are joint order, sign, zero position, previous action, command values, units, clipping and the 50 Hz period identical? Dance needs its phase commands. Swing reuses command slots for body-axis information and applies a seated, planar action wrapper. A generic locomotion runtime that supplies different commands or adds actions to a standing default pose can be incompatible despite identical tensor sizes. The wrapper is part of the policy's physical meaning even when not all of it lives inside ONNX.

**Physical compatibility.** Are servo response, available torque, battery behavior, contact friction, compliance, inertia, delay and sensor quality sufficiently represented? A perfectly exported action can still ask for motion the real plant cannot produce. Adding a seat or a stilt changes the plant, not just a rendering mesh. The simulated retained Swing seat/tendon system is not evidence of a fabricated, instrumented and safety-reviewed physical apparatus.

Use these three layers as separate rows in your report. “ONNX works” answers only part of the first row. The source model also caps Gaussian exploration during training, but those numerical caps are not a motor current limit or a mechanical safety bound.

## A closed-loop timing exercise

The 50 Hz policy period allocates 20 ms per observation-to-action cycle. A **proposed measurement worksheet**, not an existing benchmark, should record:

| Interval | Measure on the target runtime | Why it matters |
|---|---|---|
| Sensor acquisition and timestamp age | Joint/IMU sample time versus decision time | The controller may act on stale state |
| Observation assembly | Exact transforms and ordering | Frame or sign errors create systematically wrong feedback |
| ONNX inference | Median, tail latency and deadline misses | A fast mean hides occasional missed control ticks |
| Command delivery | Decision time versus applied servo command | Variable delay changes closed-loop behavior |
| Logging and watchdog | Overhead and response to invalid output | Diagnostics must not silently break the control deadline |

For a synthetic example, 3 ms sensing, 1 ms feature assembly, 4 ms inference and 5 ms command delivery total 13 ms, leaving 7 ms within a 20 ms cycle. This arithmetic is not a measurement of this robot. If the 99th-percentile end-to-end time is 24 ms, an 8 ms average does not establish reliable 50 Hz operation. Do not infer hardware timing from a browser's frame rate or the trainer's aggregate transitions per second.

## Design a robustness matrix before changing the trainer

Start with a frozen baseline ONNX and preserve the original evaluator. Separate **evaluation perturbations** from **training randomization**: the former measures robustness of an existing actor; the latter changes the distribution from which a new actor learns. A single randomization flag is not a substitute for naming the distribution and recording its bounds.

| Axis | Proposed experiment | Hold fixed | Report beyond return |
|---|---|---|---|
| Actuator model | Re-evaluate nominal result with the documented BAM path where supported | Task, policy identity and episode gate | Completion, tracking, action saturation, physical failures |
| Sensor uncertainty | Perturb only documented sensor channels in simulation | Command semantics and noise seed set | Failures versus noise level and observation clipping |
| Timing | Compare documented delay settings | Episode duration and control rate | Phase lag, contact/geometry violations |
| Starting state | Expand allowable initial yaw/posture within a specified safe simulation set | Command and evaluator | Worst-case completion, not just mean return |
| Contact and morphology | Controlled friction or geometry studies using supported model options | Actuator, policy and movement command | Slips, impacts, unintended body contacts |
| Disturbance recovery | A declared simulator push protocol, if implemented | Push timing/magnitude distribution | Time to recover and fraction of complete accepted episodes |

These rows are a research plan, not a claim that every knob already exists in the Studio UI. Inspect the constructor and source support before writing a CLI flag. When a perturbation is unsupported, implement and test it as a separate scoped project rather than quietly ignoring the request. Do not vary every factor simultaneously in the first diagnostic experiment: it becomes difficult to explain a failure. Once individual sensitivities are understood, a joint held-out distribution can test interactions.

## How each case changes the transfer question

**Dance:** phase synchronization and joint calibration matter as much as the action shape. The learned target is an eight-second retargeted excerpt, not a semantic understanding of arbitrary choreography. Before extending the claim, test held-out excerpts, tempo changes and physical balance under the same well-defined tracking metric. Make any new phase-generation rule explicit; an ONNX without its intended phase source is not the complete controller.

**Running:** the model learned a fixed forward command and still follows curved paths. First separate velocity tracking from path-following. A heading controller or a broader command curriculum would be a new experiment, not a property already certified by the 0.75 m/s result. Foot-contact timing, impact dynamics and ground friction may change the aerial-phase gate on another plant.

**Stilts:** the selected morphology is two-centimeter extensions with a specific blend and foot mass. Another extension length changes contact leverage and inertia. It is not justified to attach longer stilts and assume the same policy is robust. A morphology-conditioned actor would require observable parameters, training coverage and new evaluation; adding privileged geometry at inference would violate the declared observation semantics unless the contract were deliberately redesigned.

**Swing:** the teacher uses privileged angle/rate only for simulator labeling, while the final actor uses ordinary contract observations. This distinction prevents one information leak, but does not prove that the relevant real sensor features are accurate enough. Seat retention, lateral alignment, compliant tendon behavior and string tension require separate mechanism and measurement work. The audit observes spring-only tension at 50 Hz control samples; it neither measures total continuous-time tensile force nor establishes a safe mechanical loading envelope.

For all four, keep a qualified operator's hardware procedure, motor/thermal limits, emergency stop and mechanical risk assessment outside the learning reward. A reward penalty for collisions cannot serve as an emergency stop. This book does not authorize unrestricted on-robot exploration or autonomous DAgger teacher interventions on physical hardware.

## Make reproduction portable, not workstation-dependent

The examples retain original local run names because they identify actual evidence. A fresh clone may not contain gitignored checkpoints, large MP4s, bootstrap NPZs or the original source dance files. The bundled audit and screenshot copies remain useful for reading, but do not synthesize missing weights. Record which evidence is available before promising a full replay.

A reproducible handoff has three inventories:

1. **Source inventory:** root and RLX revisions, upstream robot-model revision, uncommitted changes, Python/MLX/MuJoCo/ONNX Runtime versions, and the exact environment creation command.
2. **Experiment inventory:** input clip or teacher source and hashes, effective recipe, seeds, reset bounds, morphology/actuator settings, parent checkpoint and sidecar hashes, teacher versus PPO budget ledger.
3. **Output inventory:** final checkpoint and sidecar, ONNX, raw training JSONL, criteria and complete-episode audit, full video with decode receipt, negative controls, and known failed stages.

The following read-only notebook command prints a small source inventory without importing MLX or initializing MuJoCo. Package versions refer to the interpreter executing it; use the actual training interpreter, not an unrelated global Python:

```bash
rlx/.venv-microduck/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform
import subprocess

root = Path.cwd()
revisions = {}
for relative in (".", "rlx", "microduck_rl", "microduck"):
    directory = root / relative
    if not directory.exists():
        revisions[relative] = "directory not present"
        continue
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    revisions[relative] = result.stdout.strip() if result.returncode == 0 else "not available"
packages = {}
for package in ("mlx", "mujoco", "numpy", "onnx", "onnxruntime"):
    try:
        packages[package] = version(package)
    except PackageNotFoundError:
        packages[package] = "not installed in this interpreter"
print(json.dumps({"python": platform.python_version(),
                  "platform": platform.platform(),
                  "revisions": revisions, "packages": packages}, indent=2))
PY
git status --short
```

Git can report a parent repository revision when a subdirectory is not a separate checkout. Confirm repository boundaries with `git -C PATH rev-parse --show-toplevel` rather than interpreting identical hashes as independent pins. The command does not install packages, download data or reproduce training. Save its output in your own new run directory alongside the effective recipe.

Do not transplant a virtual environment between machines or overwrite original successful runs. In guarded reproduction commands replace the timestamp everywhere, keep base and continuation names distinct, and carry over both checkpoint files. A training process can finish normally while failing its skill gate; preserve that distinction and only use the narrowly specified base-checkpoint guard. A missing API report, failed optimizer process or nonfinite model is not an acceptable warm start.

## Build a fair experimental comparison

Choose one testable question: “Does a lower PPO learning rate preserve a passing Swing initializer better?” is narrower than “Which algorithm is best?” Define the metric and selection rule before viewing the final test set. Give candidates the same initializer and normalizer, use matched seeds where practical, and report the entire acquisition budget as well as the refinement budget. Comparing teacher-assisted Swing with scratch PPO while charging only the final 524,288 transitions hides the cost of teacher search and imitation.

Three seed types must be distinguished: training seeds, validation/checkpoint-selection seeds, and held-out evaluation seeds. Five evaluations of one checkpoint are not five independently trained policies. For deterministic nominal worlds, changing a seed may induce little or no physical variation, so repeated equal values are not evidence of broad robustness. The saved cases establish repeated checks under their stated conditions; do not attach a population-level confidence claim without a sampling design.

For a new study, record per-training-seed pass counts and raw metrics, the criterion used to select a checkpoint, and the number of failed attempts. If you inspect a “test” rollout and tune a reward from it, that rollout has become development evidence. Reserve a fresh final set. Keep thresholds unchanged when comparing methods; otherwise an apparent improvement may merely reflect an easier judge.

## A four-session course plan

| Session | Safe starting task | Student deliverable | Completion criterion |
|---|---|---|---|
| 1: Data and PPO | Run the standard-library arithmetic/evidence lab | Tensor diagram and transition ledger | Correct GAE boundary cases and all four audit budgets |
| 2: One complete case | Reproduce one chapter in a fresh named simulation run | Recipe, raw metrics, export, audit and full video | Report every gate, whether passed or failed |
| 3: Controlled change | One reward, exploration, or supported environment ablation | Matched baseline/candidate comparison | Same evaluator and documented initializer lineage |
| 4: Transfer proposal | Inspect upstream recipe and identify mismatches | Robustness matrix and hardware-readiness gap report | Separate established evidence from proposed experiments |

Actual training duration depends on the machine and chosen case; these are teaching units, not promises to finish a multi-million-transition run in a fixed lesson period. A rigorous failed experiment is a valid learning outcome. The objective is a student who can explain why a controller succeeds, show what it actually does, and name the conditions under which the claim could fail.

## Primary references and how to use them

The following archival primary sources were checked on September 8, 2026. They explain methods; they do not supply this repository's measurements. Exact recipe constants, reward implementations and task gates are established by the local source and audit artifacts cited throughout the chapters.

- **[PPO]** Schulman, Wolski, Dhariwal, Radford and Klimov. *Proximal Policy Optimization Algorithms*. 2017, arXiv:1707.06347. Read the clipped surrogate and alternating sampling/minibatch optimization. A pessimistic surrogate is not a guarantee that every trained robot policy improves.
- **[GAE]** Schulman, Moritz, Levine, Jordan and Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. First posted 2015, arXiv:1506.02438. Study the estimator's bias–variance trade-off, then compare with this repository's explicit truncation handling.
- **[DAgger]** Ross, Gordon and Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011, PMLR 15:627–635. Study teacher labeling on learner-induced states and dataset aggregation. The selected Swing implementation uses teacher-only initial collection and student-only later collection rather than a gradual action-mixture schedule.
- **[Transfer]** Peng, Andrychowicz, Zaremba and Abbeel. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. ICRA 2018, arXiv:1710.06537. Study the physical mismatch motivation. This is background for a proposed robustness study, not validation of these four Microduck policies.
- **[TimeLimits]** Farama Foundation. *Gymnasium: Handling Time Limits*, version 0.26.3 tutorial; cross-checked against the official `TimeLimit` wrapper documentation on September 8, 2026. Distinguishes external truncation from an inherent finite-horizon termination and explains why the latter needs remaining-time information in the Markov state. Exact adapter field names in this book still come from local RLX code.

Read papers and code together. The paper gives the algorithmic idea; code specifies the actual loss reduction, clipping, normalization, terminal handling and optimizer; experiment artifacts establish what happened. A high-quality robotics report needs all three.
