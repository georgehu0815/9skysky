# A laboratory book, not a victory montage

**Author: George Hu**  
**Evidence edition: September 2026**  
**Audience:** students who can read Python and want to connect reinforcement-learning mathematics to a robot that actually moves in simulation.

This book teaches four related experiments: **Dance imitation, Self-pumped Swing, Running, and Stilt Walking**. “Stith” in the original request is interpreted as the repository's **Stilt walking** scenario. Ordinary walking is the conceptual prerequisite for the last two tasks; it is not a fifth claimed experiment.

You will build an experiment from a clean setup, specify observable inputs and physical goals, run the existing RLX implementation, export a policy, evaluate complete episodes, and investigate failures. The objective is not to memorize a magic recipe. It is to learn how to tell the difference between a functioning training pipeline and a learned behavior.

## How to study this expanded edition

Start with the mathematical foundations and **PPO Code Laboratory**, including the runnable `learning_lab.py` arithmetic and saved-evidence exercises. Then read the Studio parameter map and one case from input to exported evaluation. Each expanded case gives code-reading anchors, concrete arrays, actual staged budgets, reward construction, recorded outputs, failed attempts and guarded reproduction. Read all four to compare different acquisition problems rather than assuming one reward recipe solves every movement.

Use **From a Passing Simulation to a Robotics Research Practicum** for the final comparison and sim-to-real planning assignment. The final **Complete reproduction appendix** carries the guarded base/continuation procedures. New diagrams are labeled explanatory; reward/loss plots remain original measured evidence; refreshed UI screenshots show saved runs and do not imply new training. Copy commands from the Markdown edition rather than the wrapped PDF.

## What “from scratch” means here

There are two meanings that must not be mixed:

1. **From a clean workspace:** you prepare the simulator, recipe, data, training process, evaluation, and evidence yourself. Every chapter supports this learning path.
2. **From randomly initialized policy weights using PPO alone:** this describes the selected Dance run and the initial Running/Stilt training stages. It **does not** describe the successful Swing initialization.

Swing required a disclosed behavior-cloning/data-aggregation teacher stage before PPO refinement. Fresh PPO did not discover the required large swing in the tested budget. A truthful textbook makes that failure useful instead of relabeling imitation as reinforcement learning.

All four cover panels are frames decoded from saved API-delivered videos of accepted policies. They are not AI-generated robots, rendered target poses, or demonstrations used as substitute test evidence. A still image identifies a rollout; it cannot prove motion. The accompanying full videos and per-step measurements supply that proof.

## The evidence you are learning from

| Experiment | Established result | Scope boundary |
|---|---|---|
| Dance | 16/16 API episodes; 5/5 separate eight-second audits; mean joint RMSE 0.08438 rad | First eight seconds of a Bachata clip; extended test repeats that excerpt |
| Running | 16/16 API episodes and 5/5 audits; 0.738–0.775 m/s; 35.0–39.8% aerial samples | Fixed 0.75 m/s command; curved trajectories remain |
| Stilt Walking | 16/16 API episodes and 5/5 audits; 0.230–0.242 m/s | Only 2 cm extensions, blend 0, 0.014 kg per foot, fixed 0.25 m/s command |
| Swing | 16/16 API episodes and 5/5 audits; 162.32° symmetric span | Teacher initialization plus PPO; nominal dynamics; narrow geometry margin |

The separate audits use different reset seeds, not independently trained policy seeds. Swing's nominal resets lead to effectively identical trajectories across those seeds. Five repeated passes therefore do not establish five independent robustness successes. None of these results establishes real-hardware safety, robustness to unknown terrain, arbitrary dance imitation, or general command following.

The underlying experiments already exist. **Building this book does not rerun training**: it packages their original measurements, code-grounded explanations, and a fresh visual inspection of saved Studio runs. Distinguish historical training validation from the book's own build and link checks.

## Suggested learning paths

| Path | Sequence | Deliverable |
|---|---|---|
| First encounter | Setup → observation contract → Studio → inspect one saved rollout | Annotated diagram and a correct explanation of one failure gate |
| First reproduction | PPO foundations → Dance preparation → short pipeline smoke → full selected recipe | New uniquely named run with policy, audit, and MP4 |
| Locomotion laboratory | Stilt chapter → Running chapter → heading/contact analysis | Comparison of reward success and physical skill success |
| Advanced learning | Swing bootstrap → PPO refinement → validation-led checkpoint selection | Separate imitation/PPO accounting and an honest ablation report |

A short smoke run only proves that the plumbing operates; it is not expected to learn a skill. Conversely, millions of transitions do not guarantee success. Set aside compute time for the chapter's complete schedule, monitor the first rollout, and do not keep spending on a broken environment.

### Prerequisites and notation

You should know arrays, functions, loops, basic derivatives, expectations, and mean/variance. An observation is $o_t$, an action is $a_t$, reward is $r_t$, policy parameters are $\theta$, and critic parameters are $\phi$. Time $t$ counts **control steps**, each 0.02 seconds, unless explicitly labeled wall-clock time. An environment transition is one action applied to one simulated robot. With sixteen robots, one vector step contributes sixteen transitions.

The chapters use three labels in prose:

- **Measured:** numbers read from saved experiment evidence.
- **Implementation:** behavior found in the repository's executable code.
- **Proposed:** an extension for students, not a claim that a new experiment has already passed.

## How to use the two editions

The Markdown edition is the editable source for study notes and reproducible commands. Paths in prose and shell examples are **workspace-root-relative** unless a command explicitly changes directory. Image links are relative to this book directory. The PDF contains vector mathematics and vector design diagrams, plus original raster plots and high-density UI captures. Zoom into a formula or diagram: edges should remain sharp.

The book bundles plots, decoded cover frames, selected audit JSON, and video-validation receipts under `assets/`. The large MP4 files, checkpoint lineages, raw traces, and full training journals remain in the existing gitignored experiment folders. Share those separately if a reader needs to replay the original experiment without retraining. A fresh Git clone is not an evidence archive.

Source reports: [Dance evidence](../dance-imitation-e2e/REPORT.md), [remaining-scenario evidence](../remaining-scenarios-e2e/REPORT.md), and [technical interpretation](../remaining-scenarios-e2e/INTERPRETATION.md). Those reports preserve the long audit tables; this book explains how to reason about them.

# Set up the complete learning system

## Architecture: who does what?

![Architecture. Follow the arrows for delivery, then return to the task specification when a physical test fails.](assets/architecture.svg)

The browser does not optimize neural-network weights. The Next.js API validates a recipe, runs the Studio Python entry point, exposes status and artifacts, and serves the saved video. MuJoCo computes the physical transitions on the **CPU**. RLX performs actor/critic optimization through MLX, using Apple Metal in this measured setup. Do not infer that the Microduck simulator runs on the GPU from generic RLX examples.

| Workspace component | Read first | Responsibility |
|---|---|---|
| `rlx/` | `rlx/README.md` | PPO, rollout storage, actor/critic, Studio training and audits |
| `microduck_local/` | `README.md` and `AGENTS.md` in that directory | Robot contract, CPU simulation, task physics and rendering |
| `duck-viewer/` | `duck-viewer/README.md` | Studio controls, API jobs, telemetry and artifact delivery |
| `microduck_rl/` | Upstream robot files and playbook | Source MJCF and separate official GPU/sim-to-real stack |
| `dance-clip/` | Selected JSON plus provenance | Dance reference data, not executable policy weights |

The older local SB3 training commands and the RLX Studio route are different execution paths. This book uses **RLX PPO**. Do not silently switch to SB3 because both tools have a command named “train.” The user-facing reference `rlx/examples/ppo_microduck_dance.py` explains the original standalone pattern; the API actually launches `rlx/examples/ppo_microduck_studio.py`.

## Inventory before installing anything

From the workspace root:

```bash
pwd
git status --short
test -f rlx/examples/ppo_microduck_studio.py
test -f microduck_local/src/microduck_local/contract.py
test -d microduck_rl/src/mjlab_microduck/robot/microduck
command -v uv node npm ffmpeg ffprobe
```

Existing students on this machine can use `rlx/.venv-microduck/bin/python`. That path is an environment created for the recorded experiments, **not a file supplied by Git**. On another Mac, create the documented environment rather than copying a virtual environment between machines:

```bash
cd rlx
uv sync --python 3.12
uv pip install --python .venv/bin/python -e ../microduck_local
.venv/bin/python examples/ppo_microduck_studio.py --help
cd ..
```

Pin Python 3.12 explicitly: the RLX checkout's own `.python-version` can select 3.13, while the sibling local harness requires Python 3.12. If your checkout lacks the upstream robot directory, follow the root setup instructions to obtain the sibling `microduck_rl` repository before training. Preserve its revision. The local harness resolves the model relative to the workspace; `MICRODUCK_RL_DIR` is available for a nonstandard layout. This is an Apple-Silicon RLX/MLX walkthrough, not a promise that the same Metal commands work on a generic CPU-only Linux machine.

Where chapter commands use `rlx/.venv-microduck/bin/python`, substitute `rlx/.venv/bin/python` if you followed the clean setup above. The API creates an isolated `uv` execution environment using editable RLX and local-harness packages; it does not simply reuse your shell's activated Python. Inspect `pythonArgs()` in `duck-viewer/lib/rlx-job.ts` when diagnosing a mismatch.

## Start Studio, then inspect without training

If Studio already runs, use its displayed URL. Do not kill somebody else's server to reclaim a preferred port. For a new local instance:

```bash
cd duck-viewer
npm ci
PORT=63317 MICRODUCK_STUDIO_PYTHON="$(command -v python3.12)" \
  npm run dev -- --hostname 127.0.0.1
```

Set `MICRODUCK_STUDIO_PYTHON` to a real compatible interpreter on your machine; an empty command substitution is not a valid setup. Dependency installation can require network access. The recorded Studio URL is `http://127.0.0.1:63317`; if you choose another port, pass that same URL to every API and video-check command.

The live `duck-lab` WebSocket viewer is useful for exploration, but it is distinct from the Studio API's saved evaluation video. This book's completion evidence is tied to the saved run name, checkpoint hash, ONNX, and audit, not whatever policy happens to be displayed in a live simulation pane.

## The smallest safe preflight

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --help
rlx/.venv-microduck/bin/python \
  rlx/scripts/dance_e2e.py --help
rlx/.venv-microduck/bin/python \
  rlx/scripts/audit_scenarios.py --help
```

These help commands do not launch training. Next, prepare a recipe and inspect its JSON before launching the shared driver with `--execute`. The driver refuses to launch without that flag; it does not provide a recipe-preview mode. Confirm that the command points at the intended scenario and a **new run name**. Never use the historical successful name for an exercise: overwriting a checkpoint destroys your control condition.

A complete preflight must establish that dependencies import, the model loads, input paths are real, the API is reachable, and the intended configuration is visible. The inventory and help commands above check only part of that contract: `--help` does not instantiate MuJoCo or contact Studio. A separate **Pipeline smoke** run checks environment construction and HTTP job delivery. Neither help output nor a smoke pass means a policy has learned. Keep these separate in your notebook.

### Exercise: write an experiment contract

Before touching a reward, write five sentences: (1) the robot's observable input; (2) the behavior desired in physical units; (3) the exact episode length; (4) the failure conditions; and (5) which artifacts would let a classmate challenge your conclusion. A useful answer says “both feet airborne for at least 3% of a complete 12-second Running episode,” not “the graph should look good.”
