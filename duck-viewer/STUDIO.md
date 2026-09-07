# Microduck Studio

The root route is the local control plane for four macOS RLX Microduck recipes:

```text
dance | swing | running | stilts
    -> train -> deterministic evaluation -> render review -> ONNX handoff
```

Run it with:

```bash
cd duck-viewer
npm run dev
```

Open the printed local URL. Studio is one application: the Three.js simulation,
HUD, policy assignment, teaching, animation, recording, training, evaluation,
and artifact workflow all run on `/`. There is no iframe or separate viewer
product; legacy `/viewer` requests return to the Studio.

## Local execution

The Studio API runs the fixed command surface in
`../rlx/examples/ppo_microduck_studio.py`. It accepts `train`, `eval`,
`render`, `export`, and `cancel`; run names are sanitized and artifact downloads
are restricted to known files under
`rlx/runs/studio/<experiment>/<run>/`.

All four recipes use RLX PPO on macOS. MuJoCo simulation runs on CPU and PPO
updates run through MLX on Apple Silicon; CUDA is not required. The bundled
Swing, Running, and Stilts videos and JSON files are labeled reference evidence
and remain separate from the selected run's artifacts.

Set `MICRODUCK_STUDIO_PYTHON` when the framework-linked Python 3.12 interpreter
is not `/usr/local/bin/python3.12`.

The smoke profile trains four timesteps and proves pipeline wiring only. It does
not prove that the policy learned the selected behavior. The deployment handoff remains
locked until an ONNX policy exists, deterministic evaluation passes, a rollout
is rendered, and a person confirms the visual review. Direct hardware upload is
intentionally unavailable.

## Swing recovery recipe

Selecting **Self-pumped swing** opens the preserved `swing-studio-01` baseline.
Play its latest-run MP4 inside the Evaluation gate. Its `2.516°` mean and
`5.661°` best span are valid rollout measurements, but the policy is rejected
because the target is approximately `163°` median span.

1. Select **Apply exact Discovery settings**. Studio creates
   `swing-curriculum-01` with 250,000 new PPO steps, 16 environments, seed 2,
   `1e-4` learning rate, `0.995` gamma, `0.10` clip, 3 update epochs, `0.002`
   entropy, and `1.0` max gradient norm. Initial-motion assistance is `12°`
   and `0.35 rad/s`; domain randomization, observation noise, action delay,
   and random yaw are off.
2. Select **Start RLX**. This is a new run; `swing-studio-01` is not changed.
3. Select **Run evaluation**. Studio evaluates 16 environments for 1,200
   control steps each, or 24 seconds at 50 Hz. Evaluation starts from rest:
   `0°` initial angle and `0 rad/s` initial rate.
4. Select **Render rollout**, then play the MP4 in the Evaluation gate.
5. Continue only when mean span is at least `10°` or best span is at least
   `20°`. Otherwise keep the 250k checkpoint as evidence and revise the
   curriculum or reward; do not spend another 500k steps automatically.
6. When the Discovery gate passes, select **Apply exact Consolidation
   settings**. Studio resumes the same checkpoint for 500,000 additional PPO
   steps, reduces assistance to `4°` and `0.12 rad/s`, and enables domain
   randomization, observation noise, and action delay. The PPO optimizer values
   remain unchanged.
7. Evaluate and render again. Continue toward 1M additional steps only when
   mean span reaches `30°` or best span reaches `60°`.
8. Do not deploy until the still-start evaluation reaches approximately
   `163°` median span, geometry remains valid, and the full video has been
   reviewed. A finite rollout pass alone does not satisfy this task gate.

The detailed guide is available at:

```text
docs/microduck-studio-experiments-guide/microduck-studio-experiments-guide.md
docs/microduck-studio-experiments-guide/microduck-studio-experiments-guide.pdf
```

## Verification

```bash
cd duck-viewer
npm run lint
npm run build

cd ..
node .omx/artifacts/visual-ralph/microduck-studio/verify.cjs
```

The browser script captures 1536 px and 390 px screenshots, checks horizontal
overflow, exercises recipe controls, verifies the read-only RLX status endpoint,
and confirms that the integrated WebGL workspace and its tool panels are mounted
directly in the Studio.
