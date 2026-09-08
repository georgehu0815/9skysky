# Microduck Remaining Scenarios: End-to-End Evidence

**Workflow dates:** September 7-8, 2026  
**Evidence root:** `rlx/artifacts/scenarios-e2e-20260907`  
**Report result:** **FAIL / INCOMPLETE**

## 1. Executive status

This report is generated from saved JSON and artifacts. It does not infer a successful skill from training completion, reward magnitude, finite output, an exported ONNX file, or a playable video. A selected scenario is complete only when its independent audit passes and its video validation passes.

| Scenario | Selected attempt | Skill audit | Held-out controls | Video | E2E |
| --- | --- | --- | --- | --- | --- |
| Running | `running-v2` | TRAINING ACTIVE | not available | missing/fail | incomplete |
| Stilt Walking | `stilts-v2` | NOT RUN | not available | missing/fail | incomplete |
| Swing | `swing-v1` | NOT RUN | not available | missing/fail | incomplete |

> The aggregate result remains failed/incomplete if any selected scenario is missing or failed. This snapshot must not be described as "all verified."

## 2. Evidence boundaries

- Training and deterministic audit are separate. A recipe may use assisted Swing resets, but evaluation and audit start at rest.
- The effective audit environment uses the XML actuator model only. It is nominal local MuJoCo evidence, not BAM, mjlab GPU, hardware, or sim-to-real certification.
- Stilt morphology is part of the experiment identity. The selected recipes request 2 cm morphology; reports must not substitute the API default of 10 cm.
- Fixed locomotion commands are taken from the selected recipe and reported as actual numeric values. They are not reconstructed from reward or displacement.
- The built-in recipe metrics wrapper captures the active locomotion command before `env.step()` and attaches physical metrics after the transition. This fixes the command/metric boundary when commands resample on the step boundary.
- Locomotion evaluator version 2 anchors intended heading at each twist-command change and advances it only by commanded yaw. Signed displacement therefore measures net progress along the intended command path; a closed circle cannot pass by accumulating path length while returning to its start.
- API outcome handling accepts an explicitly failed skill evaluation long enough to collect render/export evidence. A later successful render or export does not change the skill verdict.
- Outcome inference is deliberately limited: without `audit.json`, no skill verdict exists. API JSON and logs may establish TRAIN FAILED, TRAINING ACTIVE, TRAINED / UNAUDITED, or NOT RUN, but none is a pass. No pass is inferred from filenames, checkpoint existence, finite telemetry, HTTP success, screenshots, or videos.

## 3. Historical attempts discovered

| Scenario | Attempt | Evidence status | Evidence |
| --- | --- | --- | --- |
| running | `running-v1` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/audit.json) |
| running | `running-v2` | TRAINING LOG at 2,224,128/6,000,640 | [running-v2-api.log](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-api.log) |
| stilts | `stilts-v1` | TRAIN FAILED at 1,980,416/4,001,792 | [stilts-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1-api.json) |
| stilts | `stilts-v2` | RECIPE ONLY | [stilts-v2.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2.json) |
| swing | `swing-v1` | RECIPE ONLY | [swing-v1.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1.json) |

Only directories containing `audit.json` carry a skill verdict.

## 4. Repository-wide validation context

These are saved workflow logs, not newly rerun suites. They remain relevant limitations on the evidence package:

| Check | Saved result | Log |
| --- | --- | --- |
| RLX native suite | 297 passed, 2 skipped | rlx-native-tests.log |
| Broader microduck_local suite | 12 failed, 384 passed, 1 skipped | microduck-contract-tests.log |
| Full RLX collection | collection blocked: optional `mjlab` package missing | rlx-tests.log |
| Viewer Node tests | 46 / 46 / 0 | node-tests.log |
| Viewer build | Compiled successfully | node-build.log |
| Viewer lint | eslint | node-lint.log |

- The full RLX collection is blocked because the optional `mjlab` package is not installed in the captured environment. Native non-mjlab RLX tests passed separately.
- The broader `microduck_local` run has 12 existing failures: 11 BAM/step pre-optimization golden mismatches and one symmetry drift assertion. These failures are not hidden or reclassified by this report.

## 5. Running: `running-v2`

**Scenario status:** **TRAINING ACTIVE**

No `audit.json` exists because training is active. The retained log most recently recorded **2,224,128/6,000,640** transitions. The report remains incomplete until training, evaluation, audit, and video validation finish.

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 1000000,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 600,
  "experimentId": "running",
  "gamma": 0.99,
  "initialStd": 0.6,
  "learningRate": 0.0003,
  "locomotionForwardCommand": 0.6,
  "maxEpisodeS": 12,
  "maxGradNorm": 0.5,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 128,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 12,
  "resumeFromCheckpoint": false,
  "rewardWeights": {
    "air_time": 1,
    "foot_clearance": 0.5,
    "head_up": 1,
    "keep_pace": 12,
    "plant_the_foot": 0.05,
    "pose": 0.1,
    "smooth_moves": 0.1,
    "stay_upright": 2
  },
  "runName": "running-e2e-20260907-v2",
  "seed": 7,
  "totalTimesteps": 6000640,
  "updateEpochs": 4
}
```

No effective-environment object is available in the selected audit. The audit implementation fixes XML actuators, disables domain randomization, observation noise, action delay, and random yaw, and forces Swing audit starts to `0 degrees` and `0 rad/s`.

The requested fixed forward command is **0.6000 m/s**. If this value is `n/a`, the environment's native command sequence is evaluated rather than a fixed command.

### Per-episode skill evidence

No trained-policy episode records are available.

### Controls

#### Zero control

No records available.

#### Initial control

No records available.

### Physics metric means

No physical metric aggregates are available.

### PPO transitions and losses

No training audit summary is available.

### ONNX, controls, and source identity

No ONNX or control audit is available.

### API workflow

No API workflow JSON is available for this attempt.

### Video and graph evidence

No `video-validation.json` is available. Images or MP4 files, if present, are linked below but are not described as browser/transport verified.

- [running-v2-api.log](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-api.log)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [running.json](recipes/running.json) | 840 | `d61c43873041c403b1fb4d4bada0b40206815007660c333af4f2c9f9ddcb703d` |
| [running-v2.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v2.json) | 780 | `b20c6273b9f0f67785a3d5c9dc1f0fa685a51c541f6cb404d153714102aaf211` |
| [running-v2-api.log](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-api.log) | 5,592 | `7d5254cf5f51f7e6174dafff683b267da2af415ec95d7e36831bc9c63733d82a` |

### Reproduction commands

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment running --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run running-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --seeds 101 102 103 104 105 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 6. Stilt Walking: `stilts-v2`

**Scenario status:** **NOT RUN**

No `audit.json` exists at `rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit`. The recipe may be queued. No result is inferred from a recipe file, checkpoint name, API state, or partial artifact.

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 1000000,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 500,
  "experimentId": "stilts",
  "gamma": 0.99,
  "initialStd": 0.6,
  "learningRate": 0.0003,
  "locomotionForwardCommand": 0.25,
  "maxEpisodeS": 10,
  "maxGradNorm": 0.5,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 128,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 10,
  "resumeFromCheckpoint": false,
  "rewardWeights": {
    "ang_vel_xy_penalty": 0.02,
    "feet_air_time": 4,
    "head_pose": 0.1,
    "track_ang_vel": 0.5,
    "track_lin_vel": 20,
    "upright": 1
  },
  "runName": "stilts-e2e-20260907-v2",
  "seed": 7,
  "stiltBlend": 0,
  "stiltHeightCm": 2,
  "stiltMassKg": 0.014,
  "totalTimesteps": 6000640,
  "updateEpochs": 4
}
```

No effective-environment object is available in the selected audit. The audit implementation fixes XML actuators, disables domain randomization, observation noise, action delay, and random yaw, and forces Swing audit starts to `0 degrees` and `0 rad/s`.

The requested morphology is **2 cm** stilts, blend **0**, mass **0.0140 kg**, with a fixed forward command of **0.2500 m/s**.

### Per-episode skill evidence

No trained-policy episode records are available.

### Controls

#### Zero control

No records available.

#### Initial control

No records available.

### Physics metric means

No physical metric aggregates are available.

### PPO transitions and losses

No training audit summary is available.

### ONNX, controls, and source identity

No ONNX or control audit is available.

### API workflow

No API workflow JSON is available for this attempt.

### Video and graph evidence

No `video-validation.json` is available. Images or MP4 files, if present, are linked below but are not described as browser/transport verified.

- No audit artifacts are present.

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [stilts.json](recipes/stilts.json) | 873 | `64bea730061e3e71fb493d4c1d78bdce17adca41929b55fa373bc56ceb68d9f4` |
| [stilts-v2.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2.json) | 817 | `cab5e12bed6e85c866d891ec09a563f7d7cfc19a01d337a64b9354bc02a68e09` |

### Reproduction commands

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment stilts --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run stilts-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --seeds 101 102 103 104 105 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 7. Swing: `swing-v1`

**Scenario status:** **NOT RUN**

No `audit.json` exists at `rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit`. The recipe may be queued. No result is inferred from a recipe file, checkpoint name, API state, or partial artifact.

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 262144,
  "clipCoefficient": 0.1,
  "domainRand": false,
  "entropyCoefficient": 0.005,
  "evalSteps": 1200,
  "experimentId": "swing",
  "gamma": 0.995,
  "initialStd": 0.6,
  "learningRate": 0.0001,
  "maxEpisodeS": 24,
  "maxGradNorm": 1,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 128,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 24,
  "resumeFromCheckpoint": false,
  "runName": "swing-e2e-20260907-v1",
  "seed": 7,
  "swingInitialAngleDeg": 30,
  "swingInitialRateRadS": 1,
  "swingMinSpanDeg": 150,
  "swingPlanarActions": true,
  "totalTimesteps": 1048576,
  "updateEpochs": 3
}
```

No effective-environment object is available in the selected audit. The audit implementation fixes XML actuators, disables domain randomization, observation noise, action delay, and random yaw, and forces Swing audit starts to `0 degrees` and `0 rad/s`.

Training assistance is explicit in the recipe: initial angle **30 degrees**, initial rate **1 rad/s**, planar actions **yes**. The independent audit must start at rest (`0 degrees`, `0 rad/s`); assisted training starts never count as acceptance evidence.

### Per-episode skill evidence

No trained-policy episode records are available.

### Controls

#### Zero control

No records available.

#### Initial control

No records available.

### Physics metric means

No physical metric aggregates are available.

### PPO transitions and losses

No training audit summary is available.

### ONNX, controls, and source identity

No ONNX or control audit is available.

### API workflow

No API workflow JSON is available for this attempt.

### Video and graph evidence

No `video-validation.json` is available. Images or MP4 files, if present, are linked below but are not described as browser/transport verified.

- No audit artifacts are present.

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [swing.json](recipes/swing.json) | 716 | `4b279c32981e3da6974355a50cea94a3c74a20d6306395cf66fd505c782bb125` |
| [swing-v1.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1.json) | 680 | `1ce7e50c8a89770f89c0cb20cd5694315bf732043aac1e2ed0ddd175bd7d8f44` |

### Reproduction commands

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment swing --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run swing-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-audit --seeds 101 102 103 104 105 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 8. Historical Running: `running-v1`

**Scenario status:** **FAIL**

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 500000,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 600,
  "experimentId": "running",
  "gamma": 0.99,
  "initialStd": 0.35,
  "learningRate": 0.0001,
  "maxEpisodeS": 12,
  "maxGradNorm": 0.5,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 128,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 12,
  "resumeFromCheckpoint": false,
  "runName": "running-e2e-20260907-v1",
  "seed": 7,
  "totalTimesteps": 4001792,
  "updateEpochs": 4
}
```

No effective-environment object is available in the selected audit. The audit implementation fixes XML actuators, disables domain randomization, observation noise, action delay, and random yaw, and forces Swing audit starts to `0 degrees` and `0 rad/s`.

The requested fixed forward command is **n/a m/s**. If this value is `n/a`, the environment's native command sequence is evaluated rather than a fixed command.

### Per-episode skill evidence

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.0009 | 0.0119 | 0.0039 | FAIL |
| 102 | 0 | 600 | yes | 1.0000 | 1.0000 | -0.0007 | -0.0115 | -0.0019 | FAIL |
| 103 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.0012 | 0.0100 | 0.0034 | FAIL |
| 104 | 0 | 600 | yes | 1.0000 | 1.0000 | -0.0013 | -0.0164 | -0.0040 | FAIL |
| 105 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.0014 | 0.0189 | 0.0046 | FAIL |

#### Per-episode physics details

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 0.9967 / 0.9967 | 0.0033 / 0.0033 | 0.0033 | 0 | 0 / 0 | 1 / 1 |
| 102 | 0 | 0.9967 / 0.9750 | 0.0033 / 0.0250 | 0.0033 | 0 | 0 / 13 | 1 / 14 |
| 103 | 0 | 0.9983 / 0.9867 | 0.0017 / 0.0133 | 0.0017 | 0 | 0 / 7 | 1 / 8 |
| 104 | 0 | 0.9983 / 0.9883 | 0.0017 / 0.0117 | 0.0017 | 0 | 0 / 6 | 1 / 7 |
| 105 | 0 | 0.9983 / 0.9867 | 0.0017 / 0.0133 | 0.0017 | 0 | 0 / 7 | 1 / 8 |

#### Per-episode failures

- **seed 101, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough
- **seed 102, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough
- **seed 103, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough
- **seed 104, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough
- **seed 105, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough

### Controls

#### Zero control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 49 | no | 0.7347 | 1.0000 | -0.0875 | -0.0801 | -0.7074 | FAIL |
| 102 | 0 | 46 | no | 0.7174 | 1.0000 | 0.1081 | 0.0928 | 0.2744 | FAIL |
| 103 | 0 | 45 | no | 0.7111 | 1.0000 | 0.1365 | 0.1149 | 0.4550 | FAIL |
| 104 | 0 | 44 | no | 0.7273 | 1.0000 | 0.1296 | 0.1064 | 0.4321 | FAIL |
| 105 | 0 | 43 | no | 0.6977 | 1.0000 | 0.1416 | 0.1137 | 0.4721 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 0.9388 / 0.9388 | 0.0612 / 0.0612 | 0.0612 | 0 | 1 / 1 | 1 / 1 |
| 102 | 0 | 0.9565 / 0.9565 | 0.0435 / 0.0435 | 0.0435 | 0 | 0 / 0 | 1 / 1 |
| 103 | 0 | 0.9556 / 0.9778 | 0.0444 / 0.0222 | 0.0222 | 0 | 1 / 0 | 1 / 1 |
| 104 | 0 | 0.9773 / 0.9773 | 0.0227 / 0.0227 | 0.0227 | 0 | 0 / 0 | 1 / 1 |
| 105 | 0 | 0.9535 / 0.9767 | 0.0465 / 0.0233 | 0.0233 | 0 | 1 / 0 | 1 / 1 |

Failures:

- **seed 101, episode 0:** command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 102, episode 0:** both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 103, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 104, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 105, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

#### Initial control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 38 | no | 0.7105 | 1.0000 | -0.1217 | -0.0853 | -0.9839 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 0 | 0.9211 / 0.9211 | 0.0789 / 0.0789 | 0.0789 | 0 | 1 / 1 | 1 / 1 |

Failures:

- **seed 101, episode 0:** command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

### Physics metric means

| Control | Seed | Metric | Mean |
| --- | --- | --- | --- |
| trained | 101 | command_forward_m_s | 0.1391 |
| trained | 101 | command_lateral_m_s | -0.0362 |
| trained | 101 | forward_speed_m_s | 0.0008 |
| trained | 101 | heading_forward_speed_m_s | 0.0008 |
| trained | 101 | heading_forward_x | 0.9991 |
| trained | 101 | heading_forward_y | -0.0375 |
| trained | 101 | heading_lateral_speed_m_s | -0.0007 |
| trained | 101 | left_foot_contact | 0.9967 |
| trained | 101 | right_foot_contact | 0.9967 |
| trained | 101 | upright | 0.9994 |
| trained | 101 | world_x_m | 0.0036 |
| trained | 101 | world_y_m | -0.0037 |
| trained | 102 | command_forward_m_s | 0.3497 |
| trained | 102 | command_lateral_m_s | -0.0841 |
| trained | 102 | forward_speed_m_s | -0.0003 |
| trained | 102 | heading_forward_speed_m_s | -0.0003 |
| trained | 102 | heading_forward_x | 0.9741 |
| trained | 102 | heading_forward_y | 0.2066 |
| trained | 102 | heading_lateral_speed_m_s | 0.0008 |
| trained | 102 | left_foot_contact | 0.9967 |
| trained | 102 | right_foot_contact | 0.9750 |
| trained | 102 | upright | 0.9985 |
| trained | 102 | world_x_m | -0.0019 |
| trained | 102 | world_y_m | 0.0117 |
| trained | 103 | command_forward_m_s | 0.3533 |
| trained | 103 | command_lateral_m_s | 0.0000 |
| trained | 103 | forward_speed_m_s | 0.0012 |
| trained | 103 | heading_forward_speed_m_s | 0.0012 |
| trained | 103 | heading_forward_x | 0.9886 |
| trained | 103 | heading_forward_y | 0.1146 |
| trained | 103 | heading_lateral_speed_m_s | 0.0021 |
| trained | 103 | left_foot_contact | 0.9983 |
| trained | 103 | right_foot_contact | 0.9867 |
| trained | 103 | upright | 0.9992 |
| trained | 103 | world_x_m | 0.0082 |
| trained | 103 | world_y_m | 0.0293 |
| trained | 104 | command_forward_m_s | 0.2858 |
| trained | 104 | command_lateral_m_s | 0.0924 |
| trained | 104 | forward_speed_m_s | -0.0010 |
| trained | 104 | heading_forward_speed_m_s | -0.0010 |
| trained | 104 | heading_forward_x | 0.9872 |
| trained | 104 | heading_forward_y | 0.0588 |
| trained | 104 | heading_lateral_speed_m_s | 0.0011 |
| trained | 104 | left_foot_contact | 0.9983 |
| trained | 104 | right_foot_contact | 0.9883 |
| trained | 104 | upright | 0.9988 |
| trained | 104 | world_x_m | 0.0061 |
| trained | 104 | world_y_m | 0.0214 |
| trained | 105 | command_forward_m_s | 0.3000 |
| trained | 105 | command_lateral_m_s | 0.0000 |
| trained | 105 | forward_speed_m_s | 0.0014 |
| trained | 105 | heading_forward_speed_m_s | 0.0014 |
| trained | 105 | heading_forward_x | 0.9986 |
| trained | 105 | heading_forward_y | -0.0032 |
| trained | 105 | heading_lateral_speed_m_s | -0.0003 |
| trained | 105 | left_foot_contact | 0.9983 |
| trained | 105 | right_foot_contact | 0.9867 |
| trained | 105 | upright | 0.9993 |
| trained | 105 | world_x_m | 0.0019 |
| trained | 105 | world_y_m | -0.0037 |
| zero | 101 | command_forward_m_s | -0.0878 |
| zero | 101 | command_lateral_m_s | -0.0872 |
| zero | 101 | forward_speed_m_s | 0.1238 |
| zero | 101 | heading_forward_speed_m_s | 0.1238 |
| zero | 101 | heading_forward_x | 0.9999 |
| zero | 101 | heading_forward_y | 0.0049 |
| zero | 101 | heading_lateral_speed_m_s | -0.0005 |
| zero | 101 | left_foot_contact | 0.9388 |
| zero | 101 | right_foot_contact | 0.9388 |
| zero | 101 | upright | 0.8985 |
| zero | 101 | world_x_m | 0.0222 |
| zero | 101 | world_y_m | 0.0003 |
| zero | 102 | command_forward_m_s | 0.3378 |
| zero | 102 | command_lateral_m_s | -0.2026 |
| zero | 102 | forward_speed_m_s | 0.1256 |
| zero | 102 | heading_forward_speed_m_s | 0.1256 |
| zero | 102 | heading_forward_x | 0.9999 |
| zero | 102 | heading_forward_y | 0.0166 |
| zero | 102 | heading_lateral_speed_m_s | -0.0007 |
| zero | 102 | left_foot_contact | 0.9565 |
| zero | 102 | right_foot_contact | 0.9565 |
| zero | 102 | upright | 0.8970 |
| zero | 102 | world_x_m | 0.0200 |
| zero | 102 | world_y_m | -4.119e-05 |
| zero | 103 | command_forward_m_s | 0.3000 |
| zero | 103 | command_lateral_m_s | 0.0000 |
| zero | 103 | forward_speed_m_s | 0.1365 |
| zero | 103 | heading_forward_speed_m_s | 0.1365 |
| zero | 103 | heading_forward_x | 1.0000 |
| zero | 103 | heading_forward_y | 0.0004 |
| zero | 103 | heading_lateral_speed_m_s | -0.0011 |
| zero | 103 | left_foot_contact | 0.9556 |
| zero | 103 | right_foot_contact | 0.9778 |
| zero | 103 | upright | 0.8895 |
| zero | 103 | world_x_m | 0.0259 |
| zero | 103 | world_y_m | -0.0008 |
| zero | 104 | command_forward_m_s | 0.3000 |
| zero | 104 | command_lateral_m_s | 0.0000 |
| zero | 104 | forward_speed_m_s | 0.1296 |
| zero | 104 | heading_forward_speed_m_s | 0.1296 |
| zero | 104 | heading_forward_x | 0.9999 |
| zero | 104 | heading_forward_y | -0.0123 |
| zero | 104 | heading_lateral_speed_m_s | -0.0017 |
| zero | 104 | left_foot_contact | 0.9773 |
| zero | 104 | right_foot_contact | 0.9773 |
| zero | 104 | upright | 0.8980 |
| zero | 104 | world_x_m | 0.0227 |
| zero | 104 | world_y_m | -0.0010 |
| zero | 105 | command_forward_m_s | 0.3000 |
| zero | 105 | command_lateral_m_s | 0.0000 |
| zero | 105 | forward_speed_m_s | 0.1416 |
| zero | 105 | heading_forward_speed_m_s | 0.1416 |
| zero | 105 | heading_forward_x | 1.0000 |
| zero | 105 | heading_forward_y | -0.0032 |
| zero | 105 | heading_lateral_speed_m_s | 0.0024 |
| zero | 105 | left_foot_contact | 0.9535 |
| zero | 105 | right_foot_contact | 0.9767 |
| zero | 105 | upright | 0.8880 |
| zero | 105 | world_x_m | 0.0262 |
| zero | 105 | world_y_m | 0.0016 |
| initial | 101 | command_forward_m_s | -0.0878 |
| initial | 101 | command_lateral_m_s | -0.0872 |
| initial | 101 | forward_speed_m_s | 0.1584 |
| initial | 101 | heading_forward_speed_m_s | 0.1584 |
| initial | 101 | heading_forward_x | 0.9989 |
| initial | 101 | heading_forward_y | -0.0273 |
| initial | 101 | heading_lateral_speed_m_s | 0.0133 |
| initial | 101 | left_foot_contact | 0.9211 |
| initial | 101 | right_foot_contact | 0.9211 |
| initial | 101 | upright | 0.8888 |
| initial | 101 | world_x_m | 0.0250 |
| initial | 101 | world_y_m | 0.0015 |

### PPO transitions and losses

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 1,954 |
| Optimizer minibatches | 31,264 |
| Observed transitions | 4,001,792 |
| Requested transitions | 4,001,792 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.3795 |
| Last-100 mean_loss | -0.0208 |
| Last-100 policy_loss | -0.0236 |
| Last-100 value_loss | 0.0028 |
| Last-100 entropy | -2.9254 |
| Last-100 approximate_kl | 0.0217 |
| Last-100 clip_fraction | 0.2639 |
| Last-100 explained_variance | 0.9255 |

| Transitions | Steps | Raw return | Skill | Failures |
| --- | --- | --- | --- | --- |
| 0 | 38 | 215.0771 | FAIL | command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target |
| 1,001,472 | 600 | 5,427.58 | FAIL | aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough |
| 2,000,896 | 600 | 5,259.10 | FAIL | aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough |
| 3,000,320 | 600 | 5,506.92 | FAIL | aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough |
| 4,001,792 | 600 | 5,520.36 | FAIL | aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; insufficient alternating foot support; one or both feet did not leave the ground enough |

### ONNX, controls, and source identity

| Check | Value |
| --- | --- |
| Policy SHA-256 | `35897dbac13f06163da7add449325c6f5859023b0882bebbb6940bfb122b56b3` |
| ONNX input | batch x 61 |
| ONNX output | batch x 14 |
| ONNX contract | PASS |
| Native/ONNX max absolute error | 3.576e-07 |
| Parameter L2 change | 14.1883 |
| Negative controls failed as required | yes |
| Audit scope | Nominal local simulation only; no hardware certification. No assisted Swing starts. |

### API workflow

| Action | Accepted | Phase | Exit code | Finished |
| --- | --- | --- | --- | --- |
| eval | yes | failed | 2 | 2026-09-08T00:12:59.817Z |
| render | yes | succeeded | 0 | 2026-09-08T00:13:10.249Z |
| export | yes | succeeded | 0 | 2026-09-08T00:13:15.470Z |

Recorded API failure: **Full Running skill evaluation failed; render and export evidence were collected.**

### Video and graph evidence

| Video validation | Value |
| --- | --- |
| Validation status | PASS |
| API bytes equal local render | yes |
| HTTP byte range | yes |
| Browser playback | {"currentTime": 6, "duration": 12, "error": null, "height": 360, "width": 640} |

| File | Codec | Dimensions | FPS | Frames | Seconds | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/api-video.mp4) | h264 | 640 x 360 | 25/1 | 300 | 12.000000 | `6b334ebb05e286a07566846a6485f7bfbdd6489d3db42d75d2adffad3dd156da` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison.mp4) | h264 | 960 x 412 | 25/1 | 300 | 12.000000 | `3c1a807fe6e7d319cb96ef5cc0cef61419f818a1aa7de655b2d69992e4586e3b` |

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/video-validation.json)
- [running-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-api.json)

![Running trained-policy and control comparison](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison_sheet.png)

![Running: Physical tracking](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/physical-tracking.png)

![Running: Reward and checkpoint learning](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/reward-learning.png)

![Running: PPO loss history](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/ppo-losses.png)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [running-v1.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1.json) | 570 | `ad3f470d5492a874c3acb652ad38ac71196ea02c20cd92e2b124290aba87567b` |
| [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/audit.json) | 72,213 | `481e327a90ee9794e0800b025f25d796aed698ffac6cd7931b1372f7a84b5c09` |
| [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/video-validation.json) | 6,215 | `7077ca733ad7c6eea6e3f195ced51f279d1487192a90838c9a8d4bd59ddea86d` |
| [running-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-api.json) | 307,396 | `a415b4196365b10d560799f26f8165f5769c4e72ae110ebbc30bbd707c1ad544` |
| [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison_sheet.png) | 908,199 | `41f380f327930207ea70eda75ec581f904d4f0c4a6d7aeaa04cae83a9d223c15` |
| [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/physical-tracking.png) | 199,917 | `2b13405b65fa5066632329ad36086e5296c20f1083e5dff0358d918b949e256b` |
| [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/reward-learning.png) | 217,715 | `5fc2c4835af8a53380e1a788ad9844735450c82005af0cce1f8bccc5d7263a8c` |
| [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/ppo-losses.png) | 370,488 | `3098b8569940697204be6330685ae42310c559e9a2a5967d91424ba5fa4d9d35` |
| [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/browser-video.png) | 124,963 | `d14cbd589ddf15608f2c508651d1d3f4c39168613c0c4d10b4993a024375e131` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/comparison.mp4) | 425,184 | `3c1a807fe6e7d319cb96ef5cc0cef61419f818a1aa7de655b2d69992e4586e3b` |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-audit/api-video.mp4) | 386,402 | `6b334ebb05e286a07566846a6485f7bfbdd6489d3db42d75d2adffad3dd156da` |

### Reproduction commands

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment running --recipe-json rlx/artifacts/scenarios-e2e-20260907/running-v1.json --run running-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json rlx/artifacts/scenarios-e2e-20260907/running-v1.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --seeds 101 102 103 104 105 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json rlx/artifacts/scenarios-e2e-20260907/running-v1.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

This historical command uses the retained artifact recipe `rlx/artifacts/scenarios-e2e-20260907/running-v1.json` because it does not match the tracked selected `recipes/running.json`. It is not a fresh-checkout reproduction unless that historical artifact recipe is separately retained.

## 9. Historical Stilt Walking: `stilts-v1`

**Scenario status:** **TRAIN FAILED**

No `audit.json` exists because training failed before a final checkpoint and deterministic audit were produced.

| Training failure evidence | Value |
| --- | --- |
| Transitions reached | 1,980,416 |
| Requested transitions | 4,001,792 |
| Exit code | 1 |
| Started | 2026-09-08T00:13:16.008Z |
| Finished | 2026-09-08T00:18:21.415Z |
| Final checkpoint present | no |
| Final ONNX present | no |

Recorded failure: `RuntimeError: PPO gradients contains non-finite values: gradient tensors=critic.layers.0.weight, critic.layers.0.bias, critic.layers.2.weight, critic.layers.2.bias; observations[-7.21088..10; nonfinite=0], actions[-1.39449..3.84149; nonfinite=0], old_log_probabilities[-11.5659..8.13558; nonfinite=0], old_values[-1.64562..3.35344; nonfinite=0], advantages[-1.87452..1.62534; nonfinite=0], returns[-0.430656..3.33284; nonfinite=0]`

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 500000,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 500,
  "experimentId": "stilts",
  "gamma": 0.99,
  "initialStd": 0.25,
  "learningRate": 0.0001,
  "locomotionForwardCommand": 0.15,
  "maxEpisodeS": 10,
  "maxGradNorm": 0.5,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 128,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 10,
  "resumeFromCheckpoint": false,
  "rewardWeights": {
    "feet_air_time": 3,
    "track_lin_vel": 10,
    "upright": 2
  },
  "runName": "stilts-e2e-20260907-v1",
  "seed": 7,
  "stiltBlend": 0,
  "stiltHeightCm": 2,
  "stiltMassKg": 0.014,
  "totalTimesteps": 4001792,
  "updateEpochs": 4
}
```

No effective-environment object is available in the selected audit. The audit implementation fixes XML actuators, disables domain randomization, observation noise, action delay, and random yaw, and forces Swing audit starts to `0 degrees` and `0 rad/s`.

The requested morphology is **2 cm** stilts, blend **0**, mass **0.0140 kg**, with a fixed forward command of **0.1500 m/s**.

### Per-episode skill evidence

No trained-policy episode records are available.

### Controls

#### Zero control

No records available.

#### Initial control

No records available.

### Physics metric means

No physical metric aggregates are available.

### PPO transitions and losses

No training audit summary is available.

### ONNX, controls, and source identity

No ONNX or control audit is available.

### API workflow

| Action | Accepted | Phase | Exit code | Finished |
| --- | --- | --- | --- | --- |
| train | yes | failed | 1 | 2026-09-08T00:18:21.415Z |

Recorded API failure: **train ended as phase=failed operation=train; last log=RuntimeError: PPO gradients contains non-finite values: gradient tensors=critic.layers.0.weight, critic.layers.0.bias, critic.layers.2.weight, critic.layers.2.bias; observations[-7.21088..10; nonfinite=0], actions[-1.39449..3.84149; nonfinite=0], old_log_probabilities[-11.5659..8.13558; nonfinite=0], old_values[-1.64562..3.35344; nonfinite=0], advantages[-1.87452..1.62534; nonfinite=0], returns[-0.430656..3.33284; nonfinite=0]**

### Video and graph evidence

No `video-validation.json` is available. Images or MP4 files, if present, are linked below but are not described as browser/transport verified.

- [stilts-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1-api.json)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [stilts-v1.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1.json) | 741 | `25507ddb7074919a4143f3c625f1178caeed92f19ec01fa8d63279c549d3a749` |
| [stilts-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1-api.json) | 135,906 | `d01e4a21f56b0a499365a6bdad0fbc58bf5bfcc7ab1697a40002e2c995c2129e` |

### Reproduction commands

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment stilts --recipe-json rlx/artifacts/scenarios-e2e-20260907/stilts-v1.json --run stilts-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json rlx/artifacts/scenarios-e2e-20260907/stilts-v1.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --seeds 101 102 103 104 105 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json rlx/artifacts/scenarios-e2e-20260907/stilts-v1.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

This historical command uses the retained artifact recipe `rlx/artifacts/scenarios-e2e-20260907/stilts-v1.json` because it does not match the tracked selected `recipes/stilts.json`. It is not a fresh-checkout reproduction unless that historical artifact recipe is separately retained.

## 10. Stilt training failure and ELU diagnosis

The retained `stilts-v1` API record establishes the observed failure: PPO aborted at 1,980,416 of 4,001,792 transitions because critic-layer gradients became non-finite. The logged observations, actions, old log probabilities, old values, advantages, and returns were finite, and no final checkpoint or ONNX artifact was produced.

A separate installed-MLX activation probe supplies a plausible mechanism. The stock `mlx.nn.elu` produced finite forward outputs for large positive inputs 90 and 1000 but non-finite gradients. This is consistent with the critic-only gradient failure because actor and critic use the same hidden activation structure while receiving different objectives and gradient paths. It is **not proof of the exact failing minibatch**, which was not saved.

The local model now uses a stable ELU expression:

```python
where(x >= 0, x, exp(min(x, 0)) - 1)
```

This keeps the same 512-256-128 hidden-layer topology, checkpoint tensor layout, and ELU forward behavior used by exported ONNX graphs. The PPO finite-value guards remain fail-closed: a non-finite loss, gradient, or model parameter still raises; bad updates are not silently skipped.

| Probe field | Captured value |
| --- | --- |
| Inputs | [-1.0, 0.0, 1.0, 90.0, 1000.0] |
| Outputs | [-0.6321205496788025, 0.0, 1.0, 90.0, 1000.0] |
| Gradients | [0.3678794503211975, 1.0, 1.0, null, null] |
| Finite outputs | yes |
| Finite gradients | no |
| Implementation | installed mlx.nn.elu: where(x > 0, x, alpha * (exp(x) - 1)) |

| Validation artifact | Result | Link |
| --- | --- | --- |
| Activation test before fix | 3 failed because `stable_elu` did not yet exist | [activation-before.log](../../rlx/artifacts/scenarios-e2e-20260907/activation-before.log) |
| Activation test after fix | 3 passed: eager/compiled large-input gradient boundary and forward parity | [activation-after.log](../../rlx/artifacts/scenarios-e2e-20260907/activation-after.log) |
| Installed MLX probe | finite forward, non-finite gradients at large positive inputs | [activation-probe.json](../../rlx/artifacts/scenarios-e2e-20260907/activation-probe.json) |

Source hashes:

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [activation-probe.json](../../rlx/artifacts/scenarios-e2e-20260907/activation-probe.json) | 375 | `0074fd2183fd765c729d5746162adc302e62bd84dd0755149d0dd682830fbbe5` |
| [activation-before.log](../../rlx/artifacts/scenarios-e2e-20260907/activation-before.log) | 2,068 | `0dab9d24d3d427ed44db37ea020c3680b9cac27c1c77ed79883208389d0b04f2` |
| [activation-after.log](../../rlx/artifacts/scenarios-e2e-20260907/activation-after.log) | 98 | `fc99c56adf2cb5702a4564ceead9e104cf6db01c3cd1a15756b5c9630e644ce6` |
| [activation_probe.py](../../rlx/artifacts/scenarios-e2e-20260907/activation_probe.py) | 769 | `c1a52763100be1cdf44f16da9def46a54964fc10c1445999d78d38066c9bd03c` |

## 11. Tracked reproduction recipes

These copies are tracked with the report so a fresh checkout does not depend on the gitignored evidence directory:

- [Running](recipes/running.json)
- [Stilt Walking](recipes/stilts.json)
- [Swing](recipes/swing.json)

## 12. Final conclusion

**Report result: FAIL / INCOMPLETE.**

The evidence package is not fully verified. Outstanding selected scenarios: Running (training active), Stilt Walking (not run), Swing (not run). Regenerate this report after their audit and video-validation artifacts exist; the generator will update the verdict from the files rather than from expectations.
