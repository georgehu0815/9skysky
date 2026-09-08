# Microduck Remaining Scenarios: End-to-End Evidence

**Workflow dates:** September 7-8, 2026  
**Evidence root:** `rlx/artifacts/scenarios-e2e-20260907`  
**Report result:** **PASS**

## 1. Executive status

This report is generated from saved JSON and artifacts. It does not infer a successful skill from training completion, reward magnitude, finite output, an exported ONNX file, or a playable video. A selected scenario is complete only when its API jobs, independent audit, video validation, and matching current artifact hashes all pass.

| Scenario | Selected attempt | Skill audit | Held-out controls | Video | E2E |
| --- | --- | --- | --- | --- | --- |
| Running | `running-v4` | PASS | 5/5 | PASS | complete |
| Stilt Walking | `stilts-v3` | PASS | 5/5 | PASS | complete |
| Swing | `swing-v3` | PASS | 5/5 | PASS | complete |

> All selected evidence chains pass under the stated nominal simulation conditions. Swing is teacher-initialized PPO; repeated nominal Swing seeds are not independent robustness trials.

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
| running | `running-v1` | RECIPE ONLY | [running-v1.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1.json) |
| running | `running-v2` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/audit.json) |
| running | `running-v3` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/audit.json) |
| running | `running-v4` | PASS | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/audit.json) |
| stilts | `stilts-v1` | TRAIN FAILED at 1,980,416/4,001,792 | [stilts-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1-api.json) |
| stilts | `stilts-v2` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/audit.json) |
| stilts | `stilts-v3` | PASS | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/audit.json) |
| swing | `swing-v1` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/audit.json) |
| swing | `swing-v2` | FAIL | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/audit.json) |
| swing | `swing-v3` | PASS | [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/audit.json) |

Only directories containing `audit.json` carry a skill verdict.

## 4. Repository-wide validation context

These are recorded validation results from commands run during this workflow. Known broader-suite limitations remain separate from the native acceptance checks:

| Check | Saved result | Log |
| --- | --- | --- |
| RLX native suite | 418 passed, 2 skipped | rlx-native-tests-final.log |
| Broader microduck_local suite | 12 failed, 384 passed, 1 skipped | microduck-contract-tests.log |
| Full RLX collection | collection blocked: optional `mjlab` package missing | rlx-tests.log |
| Viewer Node tests | 74 / 74 / 0 | node-tests.log |
| Viewer build | Compiled successfully | node-build.log |
| Viewer lint | eslint | node-lint.log |
| Report regression tests | Ran 18 tests / OK | report-tests.log |
| Python static correctness | All checks passed! | ruff.log |
| Studio UI playback | "passed":true | studio-ui-validation.log |

- The full RLX collection is blocked because the optional `mjlab` package is not installed in the captured environment. Native non-mjlab RLX tests passed separately.
- The broader `microduck_local` run has 12 existing failures: 11 BAM/step pre-optimization golden mismatches and one symmetry drift assertion. These failures are not hidden or reclassified by this report.

## 5. Running: `running-v4`

**Scenario status:** **PASS**
**Complete API/audit/video evidence chain:** verified

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 524288,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 600,
  "experimentId": "running",
  "gamma": 0.99,
  "initialStd": 0.6,
  "learningRate": 0.0001,
  "locomotionForwardCommand": 0.75,
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
  "resumeFromCheckpoint": true,
  "rewardWeights": {
    "air_time": 1,
    "flight": 4,
    "foot_clearance": 0.5,
    "head_up": 1,
    "keep_pace": 12,
    "plant_the_foot": 0.05,
    "pose": 0.1,
    "smooth_moves": 0.1,
    "stay_upright": 2,
    "track_turn": 2,
    "yaw_tracking": 8
  },
  "runName": "running-e2e-20260907-v4",
  "seed": 7,
  "totalTimesteps": 2097152,
  "updateEpochs": 4
}
```

| Audit setting | Actual value |
| --- | --- |
| action_delay | no |
| actuator | xml |
| domain_rand | no |
| locomotion_forward_command | 0.7500 |
| max_episode_s | 12 |
| obs_noise | no |
| random_yaw | no |
| stilt_blend | 0 |
| stilt_height_cm | 2 |
| stilt_mass_kg | n/a |
| swing_initial_angle_deg | 0 |
| swing_initial_rate_rad_s | 0 |
| swing_planar_actions | no |
| weight_overrides | {'keep_pace': 12, 'track_turn': 2, 'flight': 4, 'air_time': 1, 'stay_upright': 2, 'pose': 0.1, 'head_up': 1, 'foot_clearance': 0.5, 'smooth_moves': 0.1, 'plant_the_foot': 0.05, 'yaw_tracking': 8} |

The requested fixed forward command is **0.7500 m/s**. If this value is `n/a`, the environment's native command sequence is evaluated rather than a fixed command.

### Per-episode skill evidence

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.7622 | 6.2711 | 1.0162 | PASS |
| 502 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.7746 | 6.4911 | 1.0329 | PASS |
| 503 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.7383 | 3.8483 | 0.9844 | PASS |
| 504 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.7600 | 5.6116 | 1.0134 | PASS |
| 505 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.7644 | 9.0130 | 1.0192 | PASS |

#### Per-episode physics details

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.3050 / 0.4550 | 0.6950 / 0.5450 | 0.3867 | 167 | 85 / 85 | 85 / 85 |
| 502 | 0 | 0.3117 / 0.4367 | 0.6883 / 0.5633 | 0.3983 | 168 | 85 / 85 | 86 / 85 |
| 503 | 0 | 0.3167 / 0.4367 | 0.6833 / 0.5633 | 0.3900 | 169 | 85 / 85 | 85 / 85 |
| 504 | 0 | 0.3583 / 0.4333 | 0.6417 / 0.5667 | 0.3500 | 170 | 85 / 85 | 86 / 85 |
| 505 | 0 | 0.3050 / 0.4400 | 0.6950 / 0.5600 | 0.3983 | 167 | 85 / 85 | 85 / 85 |

#### Per-episode failures

- **seed 501, episode 0:** none
- **seed 502, episode 0:** none
- **seed 503, episode 0:** none
- **seed 504, episode 0:** none
- **seed 505, episode 0:** none

### Controls

#### Zero control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 49 | no | 0.7347 | 1.0000 | 0.1267 | 0.1165 | 0.1689 | FAIL |
| 502 | 0 | 44 | no | 0.7045 | 1.0000 | 0.1309 | 0.1074 | 0.1745 | FAIL |
| 503 | 0 | 46 | no | 0.7174 | 1.0000 | 0.1280 | 0.1098 | 0.1707 | FAIL |
| 504 | 0 | 48 | no | 0.7292 | 1.0000 | 0.1254 | 0.1126 | 0.1672 | FAIL |
| 505 | 0 | 47 | no | 0.7234 | 1.0000 | 0.1278 | 0.1123 | 0.1704 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.9388 / 0.9592 | 0.0612 / 0.0408 | 0.0408 | 0 | 2 / 1 | 2 / 2 |
| 502 | 0 | 0.9773 / 0.9773 | 0.0227 / 0.0227 | 0.0227 | 0 | 0 / 0 | 1 / 1 |
| 503 | 0 | 0.9783 / 0.9783 | 0.0217 / 0.0217 | 0.0217 | 0 | 0 / 0 | 1 / 1 |
| 504 | 0 | 0.9583 / 0.9167 | 0.0417 / 0.0833 | 0.0417 | 0 | 1 / 1 | 1 / 1 |
| 505 | 0 | 0.9787 / 0.9574 | 0.0213 / 0.0426 | 0.0213 | 0 | 0 / 1 | 1 / 1 |

Failures:

- **seed 501, episode 0:** command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 502, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 503, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 504, episode 0:** command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 505, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

#### Random control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 45 | no | 0.7778 | 1.0000 | 0.1358 | 0.1149 | 0.1811 | FAIL |
| 502 | 0 | 44 | no | 0.7273 | 1.0000 | 0.1366 | 0.1180 | 0.1822 | FAIL |
| 503 | 0 | 43 | no | 0.7442 | 1.0000 | 0.1426 | 0.1169 | 0.1901 | FAIL |
| 504 | 0 | 44 | no | 0.7500 | 1.0000 | 0.1444 | 0.1214 | 0.1926 | FAIL |
| 505 | 0 | 45 | no | 0.7778 | 1.0000 | 0.1329 | 0.1125 | 0.1773 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.9778 / 0.9111 | 0.0222 / 0.0889 | 0.0222 | 0 | 0 / 1 | 1 / 1 |
| 502 | 0 | 0.9545 / 0.9091 | 0.0455 / 0.0909 | 0.0227 | 1 | 1 / 1 | 1 / 2 |
| 503 | 0 | 0.9767 / 0.9302 | 0.0233 / 0.0698 | 0.0233 | 0 | 0 / 1 | 1 / 2 |
| 504 | 0 | 0.9773 / 0.9318 | 0.0227 / 0.0682 | 0.0227 | 0 | 0 / 1 | 1 / 1 |
| 505 | 0 | 0.9778 / 0.9556 | 0.0222 / 0.0444 | 0.0222 | 0 | 0 / 1 | 1 / 1 |

Failures:

- **seed 501, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 502, episode 0:** aerial fraction below running target; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 503, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 504, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 505, episode 0:** aerial fraction below running target; both feet must lift off and touch down; command speed tracking ratio below target; command-directed displacement below target; command-directed speed below target; episode did not complete the required 12-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

#### Initial pretrained baseline

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 600 | yes | 1.0000 | 1.0000 | 0.6664 | -0.2392 | 0.8886 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.3733 / 0.4867 | 0.6267 / 0.5133 | 0.2750 | 156 | 78 / 78 | 79 / 78 |

Failures:

- **seed 501, episode 0:** command-directed displacement below target

### Physics metric means

| Control | Seed | Metric | Mean |
| --- | --- | --- | --- |
| trained | 501 | command_forward_m_s | 0.7500 |
| trained | 501 | command_lateral_m_s | 0.0000 |
| trained | 501 | command_yaw_rad_s | 0.0000 |
| trained | 501 | forward_speed_m_s | 0.7622 |
| trained | 501 | heading_forward_speed_m_s | 0.7622 |
| trained | 501 | heading_forward_x | 0.7056 |
| trained | 501 | heading_forward_y | -0.6259 |
| trained | 501 | heading_lateral_speed_m_s | -0.0108 |
| trained | 501 | left_foot_contact | 0.3050 |
| trained | 501 | right_foot_contact | 0.4550 |
| trained | 501 | upright | 0.9990 |
| trained | 501 | world_x_m | 3.5580 |
| trained | 501 | world_y_m | -2.2906 |
| trained | 502 | command_forward_m_s | 0.7500 |
| trained | 502 | command_lateral_m_s | 0.0000 |
| trained | 502 | command_yaw_rad_s | 0.0000 |
| trained | 502 | forward_speed_m_s | 0.7746 |
| trained | 502 | heading_forward_speed_m_s | 0.7746 |
| trained | 502 | heading_forward_x | 0.6966 |
| trained | 502 | heading_forward_y | -0.6845 |
| trained | 502 | heading_lateral_speed_m_s | 0.0150 |
| trained | 502 | left_foot_contact | 0.3117 |
| trained | 502 | right_foot_contact | 0.4367 |
| trained | 502 | upright | 0.9987 |
| trained | 502 | world_x_m | 3.4742 |
| trained | 502 | world_y_m | -2.7409 |
| trained | 503 | command_forward_m_s | 0.7500 |
| trained | 503 | command_lateral_m_s | 0.0000 |
| trained | 503 | command_yaw_rad_s | 0.0000 |
| trained | 503 | forward_speed_m_s | 0.7383 |
| trained | 503 | heading_forward_speed_m_s | 0.7383 |
| trained | 503 | heading_forward_x | 0.4253 |
| trained | 503 | heading_forward_y | -0.7713 |
| trained | 503 | heading_lateral_speed_m_s | 0.0190 |
| trained | 503 | left_foot_contact | 0.3167 |
| trained | 503 | right_foot_contact | 0.4367 |
| trained | 503 | upright | 0.9991 |
| trained | 503 | world_x_m | 2.8724 |
| trained | 503 | world_y_m | -2.7943 |
| trained | 504 | command_forward_m_s | 0.7500 |
| trained | 504 | command_lateral_m_s | 0.0000 |
| trained | 504 | command_yaw_rad_s | 0.0000 |
| trained | 504 | forward_speed_m_s | 0.7600 |
| trained | 504 | heading_forward_speed_m_s | 0.7600 |
| trained | 504 | heading_forward_x | 0.6176 |
| trained | 504 | heading_forward_y | -0.6863 |
| trained | 504 | heading_lateral_speed_m_s | 0.0110 |
| trained | 504 | left_foot_contact | 0.3583 |
| trained | 504 | right_foot_contact | 0.4333 |
| trained | 504 | upright | 0.9991 |
| trained | 504 | world_x_m | 3.4001 |
| trained | 504 | world_y_m | -2.4661 |
| trained | 505 | command_forward_m_s | 0.7500 |
| trained | 505 | command_lateral_m_s | 0.0000 |
| trained | 505 | command_yaw_rad_s | 0.0000 |
| trained | 505 | forward_speed_m_s | 0.7644 |
| trained | 505 | heading_forward_speed_m_s | 0.7644 |
| trained | 505 | heading_forward_x | 0.9912 |
| trained | 505 | heading_forward_y | 0.1069 |
| trained | 505 | heading_lateral_speed_m_s | 0.0404 |
| trained | 505 | left_foot_contact | 0.3050 |
| trained | 505 | right_foot_contact | 0.4400 |
| trained | 505 | upright | 0.9992 |
| trained | 505 | world_x_m | 4.3959 |
| trained | 505 | world_y_m | 0.6080 |
| zero | 501 | command_forward_m_s | 0.7500 |
| zero | 501 | command_lateral_m_s | 0.0000 |
| zero | 501 | command_yaw_rad_s | 0.0000 |
| zero | 501 | forward_speed_m_s | 0.1267 |
| zero | 501 | heading_forward_speed_m_s | 0.1267 |
| zero | 501 | heading_forward_x | 0.9999 |
| zero | 501 | heading_forward_y | 0.0143 |
| zero | 501 | heading_lateral_speed_m_s | -0.0002 |
| zero | 501 | left_foot_contact | 0.9388 |
| zero | 501 | right_foot_contact | 0.9592 |
| zero | 501 | upright | 0.8967 |
| zero | 501 | world_x_m | 0.0236 |
| zero | 501 | world_y_m | 0.0006 |
| zero | 502 | command_forward_m_s | 0.7500 |
| zero | 502 | command_lateral_m_s | 0.0000 |
| zero | 502 | command_yaw_rad_s | 0.0000 |
| zero | 502 | forward_speed_m_s | 0.1309 |
| zero | 502 | heading_forward_speed_m_s | 0.1309 |
| zero | 502 | heading_forward_x | 0.9998 |
| zero | 502 | heading_forward_y | 0.0176 |
| zero | 502 | heading_lateral_speed_m_s | -0.0009 |
| zero | 502 | left_foot_contact | 0.9773 |
| zero | 502 | right_foot_contact | 0.9773 |
| zero | 502 | upright | 0.8934 |
| zero | 502 | world_x_m | 0.0220 |
| zero | 502 | world_y_m | -0.0005 |
| zero | 503 | command_forward_m_s | 0.7500 |
| zero | 503 | command_lateral_m_s | 0.0000 |
| zero | 503 | command_yaw_rad_s | 0.0000 |
| zero | 503 | forward_speed_m_s | 0.1280 |
| zero | 503 | heading_forward_speed_m_s | 0.1280 |
| zero | 503 | heading_forward_x | 1.0000 |
| zero | 503 | heading_forward_y | -0.0024 |
| zero | 503 | heading_lateral_speed_m_s | -0.0012 |
| zero | 503 | left_foot_contact | 0.9783 |
| zero | 503 | right_foot_contact | 0.9783 |
| zero | 503 | upright | 0.8947 |
| zero | 503 | world_x_m | 0.0211 |
| zero | 503 | world_y_m | -0.0012 |
| zero | 504 | command_forward_m_s | 0.7500 |
| zero | 504 | command_lateral_m_s | 0.0000 |
| zero | 504 | command_yaw_rad_s | 0.0000 |
| zero | 504 | forward_speed_m_s | 0.1254 |
| zero | 504 | heading_forward_speed_m_s | 0.1254 |
| zero | 504 | heading_forward_x | 0.9999 |
| zero | 504 | heading_forward_y | 0.0114 |
| zero | 504 | heading_lateral_speed_m_s | -0.0009 |
| zero | 504 | left_foot_contact | 0.9583 |
| zero | 504 | right_foot_contact | 0.9167 |
| zero | 504 | upright | 0.8968 |
| zero | 504 | world_x_m | 0.0223 |
| zero | 504 | world_y_m | -0.0003 |
| zero | 505 | command_forward_m_s | 0.7500 |
| zero | 505 | command_lateral_m_s | 0.0000 |
| zero | 505 | command_yaw_rad_s | 0.0000 |
| zero | 505 | forward_speed_m_s | 0.1278 |
| zero | 505 | heading_forward_speed_m_s | 0.1278 |
| zero | 505 | heading_forward_x | 0.9998 |
| zero | 505 | heading_forward_y | -0.0199 |
| zero | 505 | heading_lateral_speed_m_s | 0.0012 |
| zero | 505 | left_foot_contact | 0.9787 |
| zero | 505 | right_foot_contact | 0.9574 |
| zero | 505 | upright | 0.8998 |
| zero | 505 | world_x_m | 0.0249 |
| zero | 505 | world_y_m | 0.0008 |
| initial | 501 | command_forward_m_s | 0.7500 |
| initial | 501 | command_lateral_m_s | 0.0000 |
| initial | 501 | command_yaw_rad_s | 0.0000 |
| initial | 501 | forward_speed_m_s | 0.6664 |
| initial | 501 | heading_forward_speed_m_s | 0.6664 |
| initial | 501 | heading_forward_x | 0.0081 |
| initial | 501 | heading_forward_y | 0.0915 |
| initial | 501 | heading_lateral_speed_m_s | 0.0041 |
| initial | 501 | left_foot_contact | 0.3733 |
| initial | 501 | right_foot_contact | 0.4867 |
| initial | 501 | upright | 0.9989 |
| initial | 501 | world_x_m | 0.4349 |
| initial | 501 | world_y_m | 1.8893 |

### PPO transitions and losses

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 1,024 |
| Optimizer minibatches | 16,384 |
| Observed transitions | 2,097,152 |
| Requested transitions | 2,097,152 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 2.8238 |
| Last-100 mean_loss | -0.0225 |
| Last-100 policy_loss | -0.0229 |
| Last-100 value_loss | 0.0003 |
| Last-100 entropy | -0.2174 |
| Last-100 approximate_kl | 0.0308 |
| Last-100 clip_fraction | 0.3393 |
| Last-100 explained_variance | 0.5853 |

| Transitions | Steps | Raw return | Skill | Failures |
| --- | --- | --- | --- | --- |
| 0 | 600 | 9,835.52 | FAIL | command-directed displacement below target |
| 524,288 | 600 | 11,023.29 | PASS | none |
| 1,048,576 | 600 | 11,253.39 | PASS | none |
| 1,572,864 | 600 | 11,617.39 | PASS | none |
| 2,097,152 | 600 | 11,467.54 | PASS | none |

### Initialization and PPO contribution

| Initialization evidence | Value |
| --- | --- |
| Source checkpoint | /Volumes/ExternalSSD/geoagent/microduck-lab/rlx/runs/studio/running/running-e2e-20260907-v4/running.safetensors |
| Source SHA-256 | c7232327f930cd933eca57fa6454d992ee56976a785db8ca0c5913910eb52b8b |
| Teacher-assisted | n/a |
| Initial policy role | pretrained baseline, not a null control |
| Comparison seed | 501 |
| PPO raw-return change vs initialization | 1,683.18 |

This run continues a pretrained policy. Its initial skill is not credited to PPO. A nonzero parameter change proves optimization occurred, not that it improved the initializer. Compare the baseline and final physical outcomes and raw returns before attributing any benefit to PPO.

### ONNX, controls, and source identity

| Check | Value |
| --- | --- |
| Policy SHA-256 | `f48dc18840a428c66dbca23d83a1f36a82a6657e88b7450acd7e7c23d7cc5745` |
| ONNX input | batch x 61 |
| ONNX output | batch x 14 |
| ONNX contract | PASS |
| Native/ONNX max absolute error | 1.431e-06 |
| Parameter L2 change | 7.6725 |
| Negative controls failed as required | yes |
| Audit scope | Nominal local simulation only; no hardware certification. No assisted Swing starts. |

### API workflow

| Action | Accepted | Phase | Exit code | Finished |
| --- | --- | --- | --- | --- |
| train | yes | succeeded | 0 | 2026-09-08T01:30:42.069Z |
| eval | yes | succeeded | 0 | 2026-09-08T01:30:57.575Z |
| render | yes | succeeded | 0 | 2026-09-08T01:31:09.139Z |
| export | yes | succeeded | 0 | 2026-09-08T01:31:14.888Z |

### Video and graph evidence

| Video validation | Value |
| --- | --- |
| Validation status | PASS |
| API bytes equal local render | yes |
| HTTP byte range | yes |
| Browser playback | {"currentTime": 6, "duration": 12, "error": null, "height": 360, "width": 640} |

| File | Codec | Dimensions | FPS | Frames | Seconds | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/api-video.mp4) | h264 | 640 x 360 | 25/1 | 300 | 12.000000 | `cf1b3fc6abd932ae0693451993ff3fcf0f081a8046ab42c68e0f211966e23655` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison.mp4) | h264 | 960 x 412 | 25/1 | 300 | 12.000000 | `522aa57a3cbdebb95650bbfd312cc491fca44753c0992f1fee50e90754dfbb18` |

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/browser-video.png)
- [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/studio-evaluation.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/video-validation.json)
- [running-v4-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-api.json)

![Running trained-policy and control comparison](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison_sheet.png)

![Running: Physical tracking](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/physical-tracking.png)

![Running: Reward and checkpoint learning](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/reward-learning.png)

![Running: PPO loss history](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/ppo-losses.png)

![Running: Studio saved skill verdict and rollout player](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/studio-evaluation.png)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [running.json](recipes/running.json) | 900 | `2b1ee5544d4a3d812f612131f98625409e32d8d97dd16fe08cc07f1fdb061f26` |
| [running-v4.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4.json) | 900 | `2b1ee5544d4a3d812f612131f98625409e32d8d97dd16fe08cc07f1fdb061f26` |
| [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/audit.json) | 94,167 | `829b7a4bb9aaaab3a0ebc2d7a5abe28e980a57d433810dd6665a10c61d6227db` |
| [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/video-validation.json) | 6,388 | `df9f834b6c8ed2d8a8148ab9d3785ffdce3974bf07771d37c5aae3fc5a79ecb7` |
| [running-v4-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-api.json) | 339,936 | `965c03c41f7878b81c5c0282e45bde175ca71d653d95b796fa8c7572850fcf29` |
| [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison_sheet.png) | 921,619 | `bcac0e966a53e8ca202e59122f2631fa56183fa44dcc5bf42991a4c1e9152baa` |
| [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/physical-tracking.png) | 447,426 | `a7dc713266c57f04286df80c69dd228bcf0cce56f080ec1f39d620de93380fb3` |
| [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/reward-learning.png) | 288,472 | `78aa85a12c5ccdaf6e9c987f82781ad8196d3a6174a2989059d5951896d6423a` |
| [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/ppo-losses.png) | 399,967 | `e50c299b8f0aae0ab329de5ce50f6d8af2176763919bb5752c5c46d06ed16ed8` |
| [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/browser-video.png) | 127,819 | `2bf8cdb2765722b3975b50fe52a12cda6e411f58be288bf787e6643a12d22033` |
| [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/studio-evaluation.png) | 83,283 | `1a9f7434f2b98ea3ebc035fd4e983db09ddc023e598d62877eedb0997acb5a08` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/comparison.mp4) | 893,717 | `522aa57a3cbdebb95650bbfd312cc491fca44753c0992f1fee50e90754dfbb18` |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v4-audit/api-video.mp4) | 967,405 | `cf1b3fc6abd932ae0693451993ff3fcf0f081a8046ab42c68e0f211966e23655` |

### Reproduction commands

Run all stages from the repository root with the Studio server at `http://127.0.0.1:63317`. Replace `YYYYMMDD-HHMMSS` consistently with a fresh timestamp; base and continuation runs must remain distinct.

This continuation resumes the Running v3 final checkpoint. Train the tracked base recipe first. The recorded base completed training but failed its skill gate, so the API runner exits nonzero for that skill failure. Retain its failed verdict. Reuse its completed checkpoint only if the API train operation succeeded and both checkpoint and sidecar exist. The guarded block accepts only the runner's explicit skill-failure outcome; missing evidence, failed training, and other runner errors stop the copy. Do not ignore failures unconditionally or resume after this block fails.

```bash
(
set -eu
test ! -e rlx/runs/studio/running/running-base-reproduction-YYYYMMDD-HHMMSS
test ! -e rlx/artifacts/scenarios-e2e-20260907/running-base-reproduction-YYYYMMDD-HHMMSS-api.json
test ! -e rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS
base_status=0
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment running --recipe-json docs/remaining-scenarios-e2e/recipes/running-base.json --run running-base-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/running-base-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200 || base_status=$?
python3 - rlx/artifacts/scenarios-e2e-20260907/running-base-reproduction-YYYYMMDD-HHMMSS-api.json running-base-reproduction-YYYYMMDD-HHMMSS "$base_status" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
recipe = report.get('requestedRecipe') or {}
if recipe.get('experimentId') != 'running' or recipe.get('runName') != sys.argv[2]:
    raise SystemExit('Base API report does not match this run')
operations = {op['action']: op for op in report.get('operations', [])}
train = operations.get('train', {})
if (train.get('state') or {}).get('phase') != 'succeeded' or train.get('error'):
    raise SystemExit('Base training did not succeed; refusing checkpoint copy')
failure = report.get('failure')
if int(sys.argv[3]) != 0:
    expected = 'Full Running skill evaluation failed; render and export evidence were collected.'
    evaluation = (operations.get('eval', {}).get('state') or {}).get('evaluation') or {}
    if (failure or {}).get('message') != expected or evaluation.get('skill_status') != 'failed':
        raise SystemExit('Unexpected runner failure; refusing checkpoint copy')
    for action in ('render', 'export'):
        operation = operations.get(action, {})
        if (operation.get('state') or {}).get('phase') != 'succeeded' or operation.get('error'):
            raise SystemExit('Base evidence collection failed; refusing checkpoint copy')
elif failure:
    raise SystemExit('Runner status contradicts API failure; refusing checkpoint copy')
PY
test -s rlx/runs/studio/running/running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors
test -s rlx/runs/studio/running/running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors.json
mkdir -p rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS
cp rlx/runs/studio/running/running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors rlx/runs/studio/running/running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors.json rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS/
)
```

After this block succeeds, the continuation runner below uses the target-local checkpoint and the selected recipe's reward weights. It does not reuse the base run's skill verdict.

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment running --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run running-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --seeds 501 502 503 504 505 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/running.json --run rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/running-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 6. Stilt Walking: `stilts-v3`

**Scenario status:** **PASS**
**Complete API/audit/video evidence chain:** verified

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 262144,
  "clipCoefficient": 0.2,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 500,
  "experimentId": "stilts",
  "gamma": 0.99,
  "initialStd": 0.6,
  "learningRate": 0.0001,
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
  "resumeFromCheckpoint": true,
  "rewardWeights": {
    "ang_vel_xy_penalty": 0.02,
    "feet_air_time": 4,
    "head_pose": 0.1,
    "track_ang_vel": 4,
    "track_lin_vel": 20,
    "upright": 1
  },
  "runName": "stilts-e2e-20260907-v3",
  "seed": 7,
  "stiltBlend": 0,
  "stiltHeightCm": 2,
  "stiltMassKg": 0.014,
  "totalTimesteps": 1048576,
  "updateEpochs": 4
}
```

| Audit setting | Actual value |
| --- | --- |
| action_delay | no |
| actuator | xml |
| domain_rand | no |
| locomotion_forward_command | 0.2500 |
| max_episode_s | 10 |
| obs_noise | no |
| random_yaw | no |
| stilt_blend | 0 |
| stilt_height_cm | 2 |
| stilt_mass_kg | 0.0140 |
| swing_initial_angle_deg | 0 |
| swing_initial_rate_rad_s | 0 |
| swing_planar_actions | no |
| weight_overrides | {'track_lin_vel': 20, 'track_ang_vel': 4, 'upright': 1, 'feet_air_time': 4, 'head_pose': 0.1, 'ang_vel_xy_penalty': 0.02} |

The requested morphology is **2 cm** stilts, blend **0**, mass **0.0140 kg**, with a fixed forward command of **0.2500 m/s**.

### Per-episode skill evidence

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2314 | 1.8273 | 0.9254 | PASS |
| 502 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2366 | 1.2806 | 0.9463 | PASS |
| 503 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2416 | 2.1831 | 0.9664 | PASS |
| 504 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2387 | 2.2887 | 0.9549 | PASS |
| 505 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2300 | 1.4662 | 0.9199 | PASS |

#### Per-episode physics details

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.6320 / 0.6420 | 0.3680 / 0.3580 | 0.0020 | 90 | 46 / 45 | 47 / 46 |
| 502 | 0 | 0.6360 / 0.6400 | 0.3640 / 0.3600 | 0.0020 | 92 | 47 / 47 | 48 / 48 |
| 503 | 0 | 0.6100 / 0.6440 | 0.3900 / 0.3560 | 0.0020 | 92 | 47 / 46 | 47 / 47 |
| 504 | 0 | 0.5900 / 0.6480 | 0.4100 / 0.3520 | 0.0020 | 90 | 47 / 45 | 47 / 46 |
| 505 | 0 | 0.6300 / 0.6500 | 0.3700 / 0.3500 | 0.0020 | 91 | 46 / 46 | 47 / 46 |

#### Per-episode failures

- **seed 501, episode 0:** none
- **seed 502, episode 0:** none
- **seed 503, episode 0:** none
- **seed 504, episode 0:** none
- **seed 505, episode 0:** none

### Controls

#### Zero control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 46 | no | 0.7609 | 1.0000 | 0.1575 | 0.1375 | 0.6300 | FAIL |
| 502 | 0 | 39 | no | 0.7436 | 1.0000 | 0.1781 | 0.1315 | 0.7122 | FAIL |
| 503 | 0 | 40 | no | 0.7500 | 1.0000 | 0.1750 | 0.1327 | 0.6999 | FAIL |
| 504 | 0 | 42 | no | 0.7619 | 1.0000 | 0.1663 | 0.1324 | 0.6652 | FAIL |
| 505 | 0 | 45 | no | 0.7556 | 1.0000 | 0.1666 | 0.1427 | 0.6663 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.9348 / 0.9348 | 0.0652 / 0.0652 | 0.0652 | 0 | 2 / 2 | 2 / 2 |
| 502 | 0 | 0.9231 / 0.9231 | 0.0769 / 0.0769 | 0.0769 | 0 | 2 / 2 | 3 / 3 |
| 503 | 0 | 0.9250 / 0.9250 | 0.0750 / 0.0750 | 0.0750 | 0 | 2 / 2 | 2 / 2 |
| 504 | 0 | 0.9286 / 0.9286 | 0.0714 / 0.0714 | 0.0714 | 0 | 2 / 2 | 3 / 3 |
| 505 | 0 | 0.9778 / 0.9778 | 0.0222 / 0.0222 | 0.0222 | 0 | 0 / 0 | 1 / 1 |

Failures:

- **seed 501, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 502, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 503, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 504, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 505, episode 0:** both feet must lift off and touch down; command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

#### Random control

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 40 | no | 0.7750 | 1.0000 | 0.1722 | 0.1442 | 0.6886 | FAIL |
| 502 | 0 | 36 | no | 0.7500 | 1.0000 | 0.1662 | 0.1396 | 0.6647 | FAIL |
| 503 | 0 | 35 | no | 0.7429 | 1.0000 | 0.1939 | 0.1398 | 0.7757 | FAIL |
| 504 | 0 | 37 | no | 0.7568 | 1.0000 | 0.1840 | 0.1414 | 0.7360 | FAIL |
| 505 | 0 | 43 | no | 0.7907 | 1.0000 | 0.1695 | 0.1470 | 0.6778 | FAIL |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.9750 / 0.9000 | 0.0250 / 0.1000 | 0.0250 | 0 | 0 / 2 | 1 / 2 |
| 502 | 0 | 0.9444 / 0.9167 | 0.0556 / 0.0833 | 0.0278 | 1 | 1 / 1 | 1 / 2 |
| 503 | 0 | 0.9429 / 0.9143 | 0.0571 / 0.0857 | 0.0571 | 0 | 1 / 2 | 1 / 2 |
| 504 | 0 | 0.9459 / 0.8919 | 0.0541 / 0.1081 | 0.0541 | 0 | 1 / 2 | 1 / 2 |
| 505 | 0 | 0.9535 / 0.9302 | 0.0465 / 0.0698 | 0.0233 | 2 | 1 / 2 | 2 / 2 |

Failures:

- **seed 501, episode 0:** both feet must lift off and touch down; command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target
- **seed 502, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 503, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 504, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; upright fraction below target
- **seed 505, episode 0:** command-directed displacement below target; episode did not complete the required 10-second horizon; episode terminated or fell; insufficient alternating foot support; one or both feet did not leave the ground enough; upright fraction below target

#### Initial pretrained baseline

| Seed | Episode | Steps | Complete | Upright | Commanded | Directed speed | Displacement | Speed ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 500 | yes | 1.0000 | 1.0000 | 0.2448 | 0.7163 | 0.9794 | PASS |

Physical details:

| Seed | Episode | L/R contact | L/R air | Aerial | Support switches | L/R liftoffs | L/R touchdowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.5980 / 0.6560 | 0.4020 / 0.3440 | 0.0020 | 100 | 51 / 50 | 51 / 51 |

Failures:

- **seed 501, episode 0:** none

### Physics metric means

| Control | Seed | Metric | Mean |
| --- | --- | --- | --- |
| trained | 501 | command_forward_m_s | 0.2500 |
| trained | 501 | command_lateral_m_s | 0.0000 |
| trained | 501 | command_yaw_rad_s | 0.0000 |
| trained | 501 | forward_speed_m_s | 0.2314 |
| trained | 501 | heading_forward_speed_m_s | 0.2314 |
| trained | 501 | heading_forward_x | 0.8049 |
| trained | 501 | heading_forward_y | 0.4870 |
| trained | 501 | heading_lateral_speed_m_s | 0.0038 |
| trained | 501 | left_foot_contact | 0.6320 |
| trained | 501 | right_foot_contact | 0.6420 |
| trained | 501 | stilt_blend | 0.0000 |
| trained | 501 | stilt_height_cm | 2.0000 |
| trained | 501 | stilt_mass_kg | 0.0140 |
| trained | 501 | upright | 0.9993 |
| trained | 501 | world_x_m | 0.9699 |
| trained | 501 | world_y_m | 0.4088 |
| trained | 502 | command_forward_m_s | 0.2500 |
| trained | 502 | command_lateral_m_s | 0.0000 |
| trained | 502 | command_yaw_rad_s | 0.0000 |
| trained | 502 | forward_speed_m_s | 0.2366 |
| trained | 502 | heading_forward_speed_m_s | 0.2366 |
| trained | 502 | heading_forward_x | 0.5459 |
| trained | 502 | heading_forward_y | 0.6321 |
| trained | 502 | heading_lateral_speed_m_s | -0.0040 |
| trained | 502 | left_foot_contact | 0.6360 |
| trained | 502 | right_foot_contact | 0.6400 |
| trained | 502 | stilt_blend | 0.0000 |
| trained | 502 | stilt_height_cm | 2.0000 |
| trained | 502 | stilt_mass_kg | 0.0140 |
| trained | 502 | upright | 0.9989 |
| trained | 502 | world_x_m | 0.9195 |
| trained | 502 | world_y_m | 0.5335 |
| trained | 503 | command_forward_m_s | 0.2500 |
| trained | 503 | command_lateral_m_s | 0.0000 |
| trained | 503 | command_yaw_rad_s | 0.0000 |
| trained | 503 | forward_speed_m_s | 0.2416 |
| trained | 503 | heading_forward_speed_m_s | 0.2416 |
| trained | 503 | heading_forward_x | 0.8902 |
| trained | 503 | heading_forward_y | 0.2525 |
| trained | 503 | heading_lateral_speed_m_s | -0.0047 |
| trained | 503 | left_foot_contact | 0.6100 |
| trained | 503 | right_foot_contact | 0.6440 |
| trained | 503 | stilt_blend | 0.0000 |
| trained | 503 | stilt_height_cm | 2.0000 |
| trained | 503 | stilt_mass_kg | 0.0140 |
| trained | 503 | upright | 0.9992 |
| trained | 503 | world_x_m | 1.1818 |
| trained | 503 | world_y_m | 0.0529 |
| trained | 504 | command_forward_m_s | 0.2500 |
| trained | 504 | command_lateral_m_s | 0.0000 |
| trained | 504 | command_yaw_rad_s | 0.0000 |
| trained | 504 | forward_speed_m_s | 0.2387 |
| trained | 504 | heading_forward_speed_m_s | 0.2387 |
| trained | 504 | heading_forward_x | 0.9558 |
| trained | 504 | heading_forward_y | 0.2127 |
| trained | 504 | heading_lateral_speed_m_s | -0.0051 |
| trained | 504 | left_foot_contact | 0.5900 |
| trained | 504 | right_foot_contact | 0.6480 |
| trained | 504 | stilt_blend | 0.0000 |
| trained | 504 | stilt_height_cm | 2.0000 |
| trained | 504 | stilt_mass_kg | 0.0140 |
| trained | 504 | upright | 0.9992 |
| trained | 504 | world_x_m | 1.1402 |
| trained | 504 | world_y_m | 0.1147 |
| trained | 505 | command_forward_m_s | 0.2500 |
| trained | 505 | command_lateral_m_s | 0.0000 |
| trained | 505 | command_yaw_rad_s | 0.0000 |
| trained | 505 | forward_speed_m_s | 0.2300 |
| trained | 505 | heading_forward_speed_m_s | 0.2300 |
| trained | 505 | heading_forward_x | 0.6269 |
| trained | 505 | heading_forward_y | 0.5010 |
| trained | 505 | heading_lateral_speed_m_s | -0.0032 |
| trained | 505 | left_foot_contact | 0.6300 |
| trained | 505 | right_foot_contact | 0.6500 |
| trained | 505 | stilt_blend | 0.0000 |
| trained | 505 | stilt_height_cm | 2.0000 |
| trained | 505 | stilt_mass_kg | 0.0140 |
| trained | 505 | upright | 0.9990 |
| trained | 505 | world_x_m | 0.9842 |
| trained | 505 | world_y_m | 0.3125 |
| zero | 501 | command_forward_m_s | 0.2500 |
| zero | 501 | command_lateral_m_s | 0.0000 |
| zero | 501 | command_yaw_rad_s | 0.0000 |
| zero | 501 | forward_speed_m_s | 0.1575 |
| zero | 501 | heading_forward_speed_m_s | 0.1575 |
| zero | 501 | heading_forward_x | 1.0000 |
| zero | 501 | heading_forward_y | 0.0065 |
| zero | 501 | heading_lateral_speed_m_s | 0.0011 |
| zero | 501 | left_foot_contact | 0.9348 |
| zero | 501 | right_foot_contact | 0.9348 |
| zero | 501 | stilt_blend | 0.0000 |
| zero | 501 | stilt_height_cm | 2.0000 |
| zero | 501 | stilt_mass_kg | 0.0140 |
| zero | 501 | upright | 0.9109 |
| zero | 501 | world_x_m | 0.0276 |
| zero | 501 | world_y_m | 0.0003 |
| zero | 502 | command_forward_m_s | 0.2500 |
| zero | 502 | command_lateral_m_s | 0.0000 |
| zero | 502 | command_yaw_rad_s | 0.0000 |
| zero | 502 | forward_speed_m_s | 0.1781 |
| zero | 502 | heading_forward_speed_m_s | 0.1781 |
| zero | 502 | heading_forward_x | 1.0000 |
| zero | 502 | heading_forward_y | 0.0087 |
| zero | 502 | heading_lateral_speed_m_s | 0.0006 |
| zero | 502 | left_foot_contact | 0.9231 |
| zero | 502 | right_foot_contact | 0.9231 |
| zero | 502 | stilt_blend | 0.0000 |
| zero | 502 | stilt_height_cm | 2.0000 |
| zero | 502 | stilt_mass_kg | 0.0140 |
| zero | 502 | upright | 0.9001 |
| zero | 502 | world_x_m | 0.0283 |
| zero | 502 | world_y_m | -0.0008 |
| zero | 503 | command_forward_m_s | 0.2500 |
| zero | 503 | command_lateral_m_s | 0.0000 |
| zero | 503 | command_yaw_rad_s | 0.0000 |
| zero | 503 | forward_speed_m_s | 0.1750 |
| zero | 503 | heading_forward_speed_m_s | 0.1750 |
| zero | 503 | heading_forward_x | 1.0000 |
| zero | 503 | heading_forward_y | -0.0055 |
| zero | 503 | heading_lateral_speed_m_s | -0.0012 |
| zero | 503 | left_foot_contact | 0.9250 |
| zero | 503 | right_foot_contact | 0.9250 |
| zero | 503 | stilt_blend | 0.0000 |
| zero | 503 | stilt_height_cm | 2.0000 |
| zero | 503 | stilt_mass_kg | 0.0140 |
| zero | 503 | upright | 0.9002 |
| zero | 503 | world_x_m | 0.0274 |
| zero | 503 | world_y_m | -0.0014 |
| zero | 504 | command_forward_m_s | 0.2500 |
| zero | 504 | command_lateral_m_s | 0.0000 |
| zero | 504 | command_yaw_rad_s | 0.0000 |
| zero | 504 | forward_speed_m_s | 0.1663 |
| zero | 504 | heading_forward_speed_m_s | 0.1663 |
| zero | 504 | heading_forward_x | 0.9999 |
| zero | 504 | heading_forward_y | 0.0120 |
| zero | 504 | heading_lateral_speed_m_s | 7.341e-05 |
| zero | 504 | left_foot_contact | 0.9286 |
| zero | 504 | right_foot_contact | 0.9286 |
| zero | 504 | stilt_blend | 0.0000 |
| zero | 504 | stilt_height_cm | 2.0000 |
| zero | 504 | stilt_mass_kg | 0.0140 |
| zero | 504 | upright | 0.9105 |
| zero | 504 | world_x_m | 0.0275 |
| zero | 504 | world_y_m | -0.0003 |
| zero | 505 | command_forward_m_s | 0.2500 |
| zero | 505 | command_lateral_m_s | 0.0000 |
| zero | 505 | command_yaw_rad_s | 0.0000 |
| zero | 505 | forward_speed_m_s | 0.1666 |
| zero | 505 | heading_forward_speed_m_s | 0.1666 |
| zero | 505 | heading_forward_x | 0.9994 |
| zero | 505 | heading_forward_y | -0.0307 |
| zero | 505 | heading_lateral_speed_m_s | 0.0032 |
| zero | 505 | left_foot_contact | 0.9778 |
| zero | 505 | right_foot_contact | 0.9778 |
| zero | 505 | stilt_blend | 0.0000 |
| zero | 505 | stilt_height_cm | 2.0000 |
| zero | 505 | stilt_mass_kg | 0.0140 |
| zero | 505 | upright | 0.9051 |
| zero | 505 | world_x_m | 0.0305 |
| zero | 505 | world_y_m | 0.0004 |
| initial | 501 | command_forward_m_s | 0.2500 |
| initial | 501 | command_lateral_m_s | 0.0000 |
| initial | 501 | command_yaw_rad_s | 0.0000 |
| initial | 501 | forward_speed_m_s | 0.2448 |
| initial | 501 | heading_forward_speed_m_s | 0.2448 |
| initial | 501 | heading_forward_x | 0.3222 |
| initial | 501 | heading_forward_y | -0.3258 |
| initial | 501 | heading_lateral_speed_m_s | 0.0010 |
| initial | 501 | left_foot_contact | 0.5980 |
| initial | 501 | right_foot_contact | 0.6560 |
| initial | 501 | stilt_blend | 0.0000 |
| initial | 501 | stilt_height_cm | 2.0000 |
| initial | 501 | stilt_mass_kg | 0.0140 |
| initial | 501 | upright | 0.9977 |
| initial | 501 | world_x_m | 0.8582 |
| initial | 501 | world_y_m | -0.3439 |

### PPO transitions and losses

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 512 |
| Optimizer minibatches | 8,192 |
| Observed transitions | 1,048,576 |
| Requested transitions | 1,048,576 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.7690 |
| Last-100 mean_loss | -0.0193 |
| Last-100 policy_loss | -0.0206 |
| Last-100 value_loss | 0.0013 |
| Last-100 entropy | 1.1074 |
| Last-100 approximate_kl | 0.0191 |
| Last-100 clip_fraction | 0.2322 |
| Last-100 explained_variance | 0.1992 |

| Transitions | Steps | Raw return | Skill | Failures |
| --- | --- | --- | --- | --- |
| 0 | 500 | 10,431.73 | PASS | none |
| 262,144 | 500 | 10,886.09 | PASS | none |
| 524,288 | 500 | 10,934.60 | PASS | none |
| 786,432 | 500 | 11,016.86 | PASS | none |
| 1,048,576 | 500 | 11,189.72 | PASS | none |

### Initialization and PPO contribution

| Initialization evidence | Value |
| --- | --- |
| Source checkpoint | /Volumes/ExternalSSD/geoagent/microduck-lab/rlx/runs/studio/stilts/stilts-e2e-20260907-v3/stilts.safetensors |
| Source SHA-256 | 0f0c8e2291ee58018d6c60de3107330da4be6d6a50dedc40b49752afc5edf4bc |
| Teacher-assisted | n/a |
| Initial policy role | pretrained baseline, not a null control |
| Comparison seed | 501 |
| PPO raw-return change vs initialization | 757.9966 |

This run continues a pretrained policy. Its initial skill is not credited to PPO. A nonzero parameter change proves optimization occurred, not that it improved the initializer. Compare the baseline and final physical outcomes and raw returns before attributing any benefit to PPO.

### ONNX, controls, and source identity

| Check | Value |
| --- | --- |
| Policy SHA-256 | `4e1bdf737c10ef253cc209097509bf3229c08496e71e455b15d2683f37e6cb24` |
| ONNX input | batch x 61 |
| ONNX output | batch x 14 |
| ONNX contract | PASS |
| Native/ONNX max absolute error | 9.537e-07 |
| Parameter L2 change | 6.6147 |
| Negative controls failed as required | yes |
| Audit scope | Nominal local simulation only; no hardware certification. No assisted Swing starts. |

### API workflow

| Action | Accepted | Phase | Exit code | Finished |
| --- | --- | --- | --- | --- |
| train | yes | succeeded | 0 | 2026-09-08T01:23:38.059Z |
| eval | yes | succeeded | 0 | 2026-09-08T01:23:48.108Z |
| render | yes | succeeded | 0 | 2026-09-08T01:24:00.063Z |
| export | yes | succeeded | 0 | 2026-09-08T01:24:05.307Z |

### Video and graph evidence

| Video validation | Value |
| --- | --- |
| Validation status | PASS |
| API bytes equal local render | yes |
| HTTP byte range | yes |
| Browser playback | {"currentTime": 5, "duration": 10, "error": null, "height": 360, "width": 640} |

| File | Codec | Dimensions | FPS | Frames | Seconds | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/api-video.mp4) | h264 | 640 x 360 | 25/1 | 250 | 10.000000 | `7cfddb35396c563d568d805dcda302aa83c41b805d2f3deaaaae89d85a67ada8` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison.mp4) | h264 | 960 x 412 | 25/1 | 250 | 10.000000 | `ae6183848a1960f080a8929675406e3681fd35e090c76e7e7240a8b3ea41d512` |

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/browser-video.png)
- [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/studio-evaluation.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/video-validation.json)
- [stilts-v3-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-api.json)
- [stilts-v3-api.log](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-api.log)

![Stilt Walking trained-policy and control comparison](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison_sheet.png)

![Stilt Walking: Physical tracking](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/physical-tracking.png)

![Stilt Walking: Reward and checkpoint learning](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/reward-learning.png)

![Stilt Walking: PPO loss history](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/ppo-losses.png)

![Stilt Walking: Studio saved skill verdict and rollout player](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/studio-evaluation.png)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [stilts.json](recipes/stilts.json) | 869 | `e6e827808787a0c49582995cfe77476b2f4478a7cfe7068d1ecd7a37baecb467` |
| [stilts-v3.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3.json) | 813 | `0d618f0ea6a98e5f025b5ff302cbd06c9448b793be9b61976f44bd1be3dd23ea` |
| [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/audit.json) | 90,597 | `5bec34de99f0e943fd6c019435757e49ad5e4caaed292d498a597ccd627d13a8` |
| [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/video-validation.json) | 6,382 | `e51807073a20be895b96d14dd3f9d344cc75dcef5aa96021a65a8a8fe6be5681` |
| [stilts-v3-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-api.json) | 329,516 | `c078ff1019c9fdab7ca024c7c2a54376dcce61d403d986f63867a2edbf07dd32` |
| [stilts-v3-api.log](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-api.log) | 3,439 | `0d2a0a8ad9223e8661e7b88bfea490b6fdb0a95d924ae0a6a7ed9d7bfbb5e66f` |
| [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison_sheet.png) | 918,575 | `1a830deea68bc5d2ed65cfde07835a10daa286d0a6d3addbddc2a4bdac301310` |
| [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/physical-tracking.png) | 341,339 | `13130001961c57f258af22722954403f966b17140238f7b3a6a5699d812b5bd1` |
| [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/reward-learning.png) | 253,527 | `d09c4c1592d0cc76f32f37401a377351604083ffaaead2002819a07312405893` |
| [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/ppo-losses.png) | 421,287 | `52cb03b00487e24c5aef90112083973782c7fd5c4216698a6f2ce83940e00e97` |
| [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/browser-video.png) | 122,025 | `7355147cd34c189fc665bf507ed9f6de10b96d207318c71c41d2a4a7e3f2a66c` |
| [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/studio-evaluation.png) | 82,894 | `fbda1e3860662d67cd7531af14e697d22b750c278fddc34f463c1a6d785338f4` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/comparison.mp4) | 474,477 | `ae6183848a1960f080a8929675406e3681fd35e090c76e7e7240a8b3ea41d512` |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit/api-video.mp4) | 442,905 | `7cfddb35396c563d568d805dcda302aa83c41b805d2f3deaaaae89d85a67ada8` |

### Reproduction commands

Run all stages from the repository root with the Studio server at `http://127.0.0.1:63317`. Replace `YYYYMMDD-HHMMSS` consistently with a fresh timestamp; base and continuation runs must remain distinct.

This continuation resumes the Stilt v2 final checkpoint. Train the tracked base recipe first. The recorded base completed training but failed its skill gate, so the API runner exits nonzero for that skill failure. Retain its failed verdict. Reuse its completed checkpoint only if the API train operation succeeded and both checkpoint and sidecar exist. The guarded block accepts only the runner's explicit skill-failure outcome; missing evidence, failed training, and other runner errors stop the copy. Do not ignore failures unconditionally or resume after this block fails.

```bash
(
set -eu
test ! -e rlx/runs/studio/stilts/stilts-base-reproduction-YYYYMMDD-HHMMSS
test ! -e rlx/artifacts/scenarios-e2e-20260907/stilts-base-reproduction-YYYYMMDD-HHMMSS-api.json
test ! -e rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS
base_status=0
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment stilts --recipe-json docs/remaining-scenarios-e2e/recipes/stilts-base.json --run stilts-base-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/stilts-base-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200 || base_status=$?
python3 - rlx/artifacts/scenarios-e2e-20260907/stilts-base-reproduction-YYYYMMDD-HHMMSS-api.json stilts-base-reproduction-YYYYMMDD-HHMMSS "$base_status" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
recipe = report.get('requestedRecipe') or {}
if recipe.get('experimentId') != 'stilts' or recipe.get('runName') != sys.argv[2]:
    raise SystemExit('Base API report does not match this run')
operations = {op['action']: op for op in report.get('operations', [])}
train = operations.get('train', {})
if (train.get('state') or {}).get('phase') != 'succeeded' or train.get('error'):
    raise SystemExit('Base training did not succeed; refusing checkpoint copy')
failure = report.get('failure')
if int(sys.argv[3]) != 0:
    expected = 'Full Stilts skill evaluation failed; render and export evidence were collected.'
    evaluation = (operations.get('eval', {}).get('state') or {}).get('evaluation') or {}
    if (failure or {}).get('message') != expected or evaluation.get('skill_status') != 'failed':
        raise SystemExit('Unexpected runner failure; refusing checkpoint copy')
    for action in ('render', 'export'):
        operation = operations.get(action, {})
        if (operation.get('state') or {}).get('phase') != 'succeeded' or operation.get('error'):
            raise SystemExit('Base evidence collection failed; refusing checkpoint copy')
elif failure:
    raise SystemExit('Runner status contradicts API failure; refusing checkpoint copy')
PY
test -s rlx/runs/studio/stilts/stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors
test -s rlx/runs/studio/stilts/stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors.json
mkdir -p rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS
cp rlx/runs/studio/stilts/stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors rlx/runs/studio/stilts/stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors.json rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS/
)
```

After this block succeeds, the continuation runner below uses the target-local checkpoint and the selected recipe's reward weights. It does not reuse the base run's skill verdict.

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment stilts --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run stilts-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --seeds 501 502 503 504 505 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/stilts.json --run rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/stilts-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 7. Swing: `swing-v3`

**Scenario status:** **PASS**
**Complete API/audit/video evidence chain:** verified

### Recipe and effective audit environment

The exact requested recipe follows. These values describe training; they do not override the deterministic audit environment.

```json
{
  "actionDelay": false,
  "checkpointInterval": 262144,
  "clipCoefficient": 0.05,
  "domainRand": false,
  "entropyCoefficient": 0,
  "evalSteps": 1200,
  "experimentId": "swing",
  "freezeObservationNormalization": true,
  "gamma": 0.995,
  "initialStd": 0.1,
  "learningRate": 3e-06,
  "maxEpisodeS": 24,
  "maxGradNorm": 0.5,
  "normalizeRewards": true,
  "numEnvs": 16,
  "numMinibatches": 4,
  "numSteps": 256,
  "obsNoise": false,
  "profile": "full",
  "randomYaw": false,
  "renderSeconds": 24,
  "resumeFromCheckpoint": true,
  "runName": "swing-e2e-20260907-v3",
  "seed": 7,
  "swingInitialAngleDeg": 0,
  "swingInitialRateRadS": 0,
  "swingMinSpanDeg": 150,
  "swingPlanarActions": true,
  "totalTimesteps": 524288,
  "updateEpochs": 2
}
```

| Audit setting | Actual value |
| --- | --- |
| action_delay | no |
| actuator | xml |
| domain_rand | no |
| locomotion_forward_command | n/a |
| max_episode_s | 24 |
| obs_noise | no |
| random_yaw | no |
| stilt_blend | 0 |
| stilt_height_cm | 2 |
| stilt_mass_kg | n/a |
| swing_initial_angle_deg | 0 |
| swing_initial_rate_rad_s | 0 |
| swing_planar_actions | yes |
| weight_overrides | {} |

Training assistance is explicit in the recipe: initial angle **0 degrees**, initial rate **0 rad/s**, planar actions **yes**. The independent audit must start at rest (`0 degrees`, `0 rad/s`); assisted training starts never count as acceptance evidence.

### Per-episode skill evidence

| Seed | Episode | Steps | Complete | Bidirectional span | Negative peak | Positive peak | Tensioned | Valid geometry | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 1,200 | yes | 162.3216 | -81.1608 | 82.9435 | 1.0000 | 1.0000 | PASS |
| 502 | 0 | 1,200 | yes | 162.3216 | -81.1608 | 82.9435 | 1.0000 | 1.0000 | PASS |
| 503 | 0 | 1,200 | yes | 162.3216 | -81.1608 | 82.9435 | 1.0000 | 1.0000 | PASS |
| 504 | 0 | 1,200 | yes | 162.3216 | -81.1608 | 82.9435 | 1.0000 | 1.0000 | PASS |
| 505 | 0 | 1,200 | yes | 162.3216 | -81.1608 | 82.9435 | 1.0000 | 1.0000 | PASS |

#### Per-episode physics details

| Seed | Episode | Max lateral | Max alignment | Min/max string length | Min spring tension |
| --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.0090 | 0.0461 | 0.3801 / 0.3867 | 0.2090 |
| 502 | 0 | 0.0090 | 0.0461 | 0.3801 / 0.3867 | 0.2090 |
| 503 | 0 | 0.0090 | 0.0461 | 0.3801 / 0.3867 | 0.2090 |
| 504 | 0 | 0.0090 | 0.0461 | 0.3801 / 0.3867 | 0.2090 |
| 505 | 0 | 0.0090 | 0.0461 | 0.3801 / 0.3867 | 0.2090 |

#### Per-episode failures

- **seed 501, episode 0:** none
- **seed 502, episode 0:** none
- **seed 503, episode 0:** none
- **seed 504, episode 0:** none
- **seed 505, episode 0:** none

### Controls

#### Zero control

| Seed | Episode | Steps | Complete | Bidirectional span | Negative peak | Positive peak | Tensioned | Valid geometry | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 1,200 | yes | 0.4957 | -0.2478 | 0.2667 | 1.0000 | 1.0000 | FAIL |
| 502 | 0 | 1,200 | yes | 0.4957 | -0.2478 | 0.2667 | 1.0000 | 1.0000 | FAIL |
| 503 | 0 | 1,200 | yes | 0.4957 | -0.2478 | 0.2667 | 1.0000 | 1.0000 | FAIL |
| 504 | 0 | 1,200 | yes | 0.4957 | -0.2478 | 0.2667 | 1.0000 | 1.0000 | FAIL |
| 505 | 0 | 1,200 | yes | 0.4957 | -0.2478 | 0.2667 | 1.0000 | 1.0000 | FAIL |

Physical details:

| Seed | Episode | Max lateral | Max alignment | Min/max string length | Min spring tension |
| --- | --- | --- | --- | --- | --- |
| 501 | 0 | 5.015e-07 | 6.736e-10 | 0.3821 / 0.3822 | 4.2705 |
| 502 | 0 | 5.015e-07 | 6.736e-10 | 0.3821 / 0.3822 | 4.2705 |
| 503 | 0 | 5.015e-07 | 6.736e-10 | 0.3821 / 0.3822 | 4.2705 |
| 504 | 0 | 5.015e-07 | 6.736e-10 | 0.3821 / 0.3822 | 4.2705 |
| 505 | 0 | 5.015e-07 | 6.736e-10 | 0.3821 / 0.3822 | 4.2705 |

Failures:

- **seed 501, episode 0:** bidirectional span below target
- **seed 502, episode 0:** bidirectional span below target
- **seed 503, episode 0:** bidirectional span below target
- **seed 504, episode 0:** bidirectional span below target
- **seed 505, episode 0:** bidirectional span below target

#### Random control

| Seed | Episode | Steps | Complete | Bidirectional span | Negative peak | Positive peak | Tensioned | Valid geometry | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 1,200 | yes | 0.8067 | -0.4393 | 0.4034 | 1.0000 | 1.0000 | FAIL |
| 502 | 0 | 1,200 | yes | 0.8067 | -0.4393 | 0.4034 | 1.0000 | 1.0000 | FAIL |
| 503 | 0 | 1,200 | yes | 0.8067 | -0.4393 | 0.4034 | 1.0000 | 1.0000 | FAIL |
| 504 | 0 | 1,200 | yes | 0.8067 | -0.4393 | 0.4034 | 1.0000 | 1.0000 | FAIL |
| 505 | 0 | 1,200 | yes | 0.8067 | -0.4393 | 0.4034 | 1.0000 | 1.0000 | FAIL |

Physical details:

| Seed | Episode | Max lateral | Max alignment | Min/max string length | Min spring tension |
| --- | --- | --- | --- | --- | --- |
| 501 | 0 | 1.001e-06 | 3.282e-08 | 0.3821 / 0.3823 | 4.2690 |
| 502 | 0 | 1.001e-06 | 3.282e-08 | 0.3821 / 0.3823 | 4.2690 |
| 503 | 0 | 1.001e-06 | 3.282e-08 | 0.3821 / 0.3823 | 4.2690 |
| 504 | 0 | 1.001e-06 | 3.282e-08 | 0.3821 / 0.3823 | 4.2690 |
| 505 | 0 | 1.001e-06 | 3.282e-08 | 0.3821 / 0.3823 | 4.2690 |

Failures:

- **seed 501, episode 0:** bidirectional span below target
- **seed 502, episode 0:** bidirectional span below target
- **seed 503, episode 0:** bidirectional span below target
- **seed 504, episode 0:** bidirectional span below target
- **seed 505, episode 0:** bidirectional span below target

#### Initial pretrained baseline

| Seed | Episode | Steps | Complete | Bidirectional span | Negative peak | Positive peak | Tensioned | Valid geometry | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 0 | 1,200 | yes | 154.6207 | -77.3103 | 78.8132 | 1.0000 | 1.0000 | PASS |

Physical details:

| Seed | Episode | Max lateral | Max alignment | Min/max string length | Min spring tension |
| --- | --- | --- | --- | --- | --- |
| 501 | 0 | 0.0087 | 0.0061 | 0.3802 / 0.3867 | 0.3058 |

Failures:

- **seed 501, episode 0:** none

### Physics metric means

| Control | Seed | Metric | Mean |
| --- | --- | --- | --- |
| trained | 501 | alignment_penalty | 0.0010 |
| trained | 501 | lateral_offset_m | 5.340e-05 |
| trained | 501 | string_imbalance_m | 3.895e-05 |
| trained | 501 | string_left_m | 0.3825 |
| trained | 501 | string_left_tension_n | 4.9731 |
| trained | 501 | string_right_m | 0.3825 |
| trained | 501 | string_right_tension_n | 4.9823 |
| trained | 501 | string_slack_m | 0.0000 |
| trained | 501 | swing_abs_angle_deg | 26.1614 |
| trained | 501 | swing_angle_deg | -0.8205 |
| trained | 501 | swing_rate_rad_s | -0.0078 |
| trained | 501 | valid_geometry | 1.0000 |
| trained | 502 | alignment_penalty | 0.0010 |
| trained | 502 | lateral_offset_m | 5.340e-05 |
| trained | 502 | string_imbalance_m | 3.895e-05 |
| trained | 502 | string_left_m | 0.3825 |
| trained | 502 | string_left_tension_n | 4.9731 |
| trained | 502 | string_right_m | 0.3825 |
| trained | 502 | string_right_tension_n | 4.9823 |
| trained | 502 | string_slack_m | 0.0000 |
| trained | 502 | swing_abs_angle_deg | 26.1614 |
| trained | 502 | swing_angle_deg | -0.8205 |
| trained | 502 | swing_rate_rad_s | -0.0078 |
| trained | 502 | valid_geometry | 1.0000 |
| trained | 503 | alignment_penalty | 0.0010 |
| trained | 503 | lateral_offset_m | 5.340e-05 |
| trained | 503 | string_imbalance_m | 3.895e-05 |
| trained | 503 | string_left_m | 0.3825 |
| trained | 503 | string_left_tension_n | 4.9731 |
| trained | 503 | string_right_m | 0.3825 |
| trained | 503 | string_right_tension_n | 4.9823 |
| trained | 503 | string_slack_m | 0.0000 |
| trained | 503 | swing_abs_angle_deg | 26.1614 |
| trained | 503 | swing_angle_deg | -0.8205 |
| trained | 503 | swing_rate_rad_s | -0.0078 |
| trained | 503 | valid_geometry | 1.0000 |
| trained | 504 | alignment_penalty | 0.0010 |
| trained | 504 | lateral_offset_m | 5.340e-05 |
| trained | 504 | string_imbalance_m | 3.895e-05 |
| trained | 504 | string_left_m | 0.3825 |
| trained | 504 | string_left_tension_n | 4.9731 |
| trained | 504 | string_right_m | 0.3825 |
| trained | 504 | string_right_tension_n | 4.9823 |
| trained | 504 | string_slack_m | 0.0000 |
| trained | 504 | swing_abs_angle_deg | 26.1614 |
| trained | 504 | swing_angle_deg | -0.8205 |
| trained | 504 | swing_rate_rad_s | -0.0078 |
| trained | 504 | valid_geometry | 1.0000 |
| trained | 505 | alignment_penalty | 0.0010 |
| trained | 505 | lateral_offset_m | 5.340e-05 |
| trained | 505 | string_imbalance_m | 3.895e-05 |
| trained | 505 | string_left_m | 0.3825 |
| trained | 505 | string_left_tension_n | 4.9731 |
| trained | 505 | string_right_m | 0.3825 |
| trained | 505 | string_right_tension_n | 4.9823 |
| trained | 505 | string_slack_m | 0.0000 |
| trained | 505 | swing_abs_angle_deg | 26.1614 |
| trained | 505 | swing_angle_deg | -0.8205 |
| trained | 505 | swing_rate_rad_s | -0.0078 |
| trained | 505 | valid_geometry | 1.0000 |
| zero | 501 | alignment_penalty | 1.797e-10 |
| zero | 501 | lateral_offset_m | -2.003e-09 |
| zero | 501 | string_imbalance_m | 3.570e-07 |
| zero | 501 | string_left_m | 0.3822 |
| zero | 501 | string_left_tension_n | 4.3908 |
| zero | 501 | string_right_m | 0.3822 |
| zero | 501 | string_right_tension_n | 4.3915 |
| zero | 501 | string_slack_m | 0.0000 |
| zero | 501 | swing_abs_angle_deg | 0.1228 |
| zero | 501 | swing_angle_deg | -0.0017 |
| zero | 501 | swing_rate_rad_s | 9.188e-06 |
| zero | 501 | valid_geometry | 1.0000 |
| zero | 502 | alignment_penalty | 1.797e-10 |
| zero | 502 | lateral_offset_m | -2.003e-09 |
| zero | 502 | string_imbalance_m | 3.570e-07 |
| zero | 502 | string_left_m | 0.3822 |
| zero | 502 | string_left_tension_n | 4.3908 |
| zero | 502 | string_right_m | 0.3822 |
| zero | 502 | string_right_tension_n | 4.3915 |
| zero | 502 | string_slack_m | 0.0000 |
| zero | 502 | swing_abs_angle_deg | 0.1228 |
| zero | 502 | swing_angle_deg | -0.0017 |
| zero | 502 | swing_rate_rad_s | 9.188e-06 |
| zero | 502 | valid_geometry | 1.0000 |
| zero | 503 | alignment_penalty | 1.797e-10 |
| zero | 503 | lateral_offset_m | -2.003e-09 |
| zero | 503 | string_imbalance_m | 3.570e-07 |
| zero | 503 | string_left_m | 0.3822 |
| zero | 503 | string_left_tension_n | 4.3908 |
| zero | 503 | string_right_m | 0.3822 |
| zero | 503 | string_right_tension_n | 4.3915 |
| zero | 503 | string_slack_m | 0.0000 |
| zero | 503 | swing_abs_angle_deg | 0.1228 |
| zero | 503 | swing_angle_deg | -0.0017 |
| zero | 503 | swing_rate_rad_s | 9.188e-06 |
| zero | 503 | valid_geometry | 1.0000 |
| zero | 504 | alignment_penalty | 1.797e-10 |
| zero | 504 | lateral_offset_m | -2.003e-09 |
| zero | 504 | string_imbalance_m | 3.570e-07 |
| zero | 504 | string_left_m | 0.3822 |
| zero | 504 | string_left_tension_n | 4.3908 |
| zero | 504 | string_right_m | 0.3822 |
| zero | 504 | string_right_tension_n | 4.3915 |
| zero | 504 | string_slack_m | 0.0000 |
| zero | 504 | swing_abs_angle_deg | 0.1228 |
| zero | 504 | swing_angle_deg | -0.0017 |
| zero | 504 | swing_rate_rad_s | 9.188e-06 |
| zero | 504 | valid_geometry | 1.0000 |
| zero | 505 | alignment_penalty | 1.797e-10 |
| zero | 505 | lateral_offset_m | -2.003e-09 |
| zero | 505 | string_imbalance_m | 3.570e-07 |
| zero | 505 | string_left_m | 0.3822 |
| zero | 505 | string_left_tension_n | 4.3908 |
| zero | 505 | string_right_m | 0.3822 |
| zero | 505 | string_right_tension_n | 4.3915 |
| zero | 505 | string_slack_m | 0.0000 |
| zero | 505 | swing_abs_angle_deg | 0.1228 |
| zero | 505 | swing_angle_deg | -0.0017 |
| zero | 505 | swing_rate_rad_s | 9.188e-06 |
| zero | 505 | valid_geometry | 1.0000 |
| initial | 501 | alignment_penalty | 0.0004 |
| initial | 501 | lateral_offset_m | -1.971e-05 |
| initial | 501 | string_imbalance_m | 2.757e-05 |
| initial | 501 | string_left_m | 0.3825 |
| initial | 501 | string_left_tension_n | 4.9390 |
| initial | 501 | string_right_m | 0.3825 |
| initial | 501 | string_right_tension_n | 4.9425 |
| initial | 501 | string_slack_m | 0.0000 |
| initial | 501 | swing_abs_angle_deg | 23.4740 |
| initial | 501 | swing_angle_deg | -0.4467 |
| initial | 501 | swing_rate_rad_s | 0.0451 |
| initial | 501 | valid_geometry | 1.0000 |

### PPO transitions and losses

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 128 |
| Optimizer minibatches | 1,024 |
| Observed transitions | 524,288 |
| Requested transitions | 524,288 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.0080 |
| Last-100 mean_loss | 0.3728 |
| Last-100 policy_loss | -0.0003 |
| Last-100 value_loss | 0.3731 |
| Last-100 entropy | -12.3715 |
| Last-100 approximate_kl | 0.0005 |
| Last-100 clip_fraction | 0.0969 |
| Last-100 explained_variance | -0.8999 |

| Transitions | Steps | Raw return | Skill | Failures |
| --- | --- | --- | --- | --- |
| 0 | 1,200 | 6,126.39 | PASS | none |
| 262,144 | 1,200 | 5,797.53 | FAIL | invalid swing geometry |
| 524,288 | 1,200 | 7,016.89 | PASS | none |

### Initialization and PPO contribution

| Initialization evidence | Value |
| --- | --- |
| Source checkpoint | /Volumes/ExternalSSD/geoagent/microduck-lab/rlx/runs/studio/swing/swing-e2e-20260907-v3/swing.safetensors |
| Source SHA-256 | de820bf3f448374d12ab4dc3673f65dcc5d37f7c821455e093cd1a3c76b077fe |
| Teacher-assisted | yes |
| Initial policy role | pretrained baseline, not a null control |
| Comparison seed | 501 |
| PPO raw-return change vs initialization | 890.5029 |

This run continues a pretrained policy. Its initial skill is not credited to PPO. A nonzero parameter change proves optimization occurred, not that it improved the initializer. Compare the baseline and final physical outcomes and raw returns before attributing any benefit to PPO.

### ONNX, controls, and source identity

| Check | Value |
| --- | --- |
| Policy SHA-256 | `d0599d4453fe80bcac4e92cb61d858dae2c86310090885c2a5ecb07ed2bbc916` |
| ONNX input | batch x 61 |
| ONNX output | batch x 14 |
| ONNX contract | PASS |
| Native/ONNX max absolute error | 1.132e-06 |
| Parameter L2 change | 0.4898 |
| Negative controls failed as required | yes |
| Audit scope | Nominal local simulation only; no hardware certification. No assisted Swing starts. |

### API workflow

| Action | Accepted | Phase | Exit code | Finished |
| --- | --- | --- | --- | --- |
| train | yes | succeeded | 0 | 2026-09-08T01:25:13.226Z |
| eval | yes | succeeded | 0 | 2026-09-08T01:25:25.071Z |
| render | yes | succeeded | 0 | 2026-09-08T01:25:46.250Z |
| export | yes | succeeded | 0 | 2026-09-08T01:25:51.887Z |

### Video and graph evidence

| Video validation | Value |
| --- | --- |
| Validation status | PASS |
| API bytes equal local render | yes |
| HTTP byte range | yes |
| Browser playback | {"currentTime": 12, "duration": 24, "error": null, "height": 360, "width": 640} |

| File | Codec | Dimensions | FPS | Frames | Seconds | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/api-video.mp4) | h264 | 640 x 360 | 25/1 | 600 | 24.000000 | `dee5b2805efcd8339f9e2389c78fbc6bcd4d095e4caa693d2f2e872bdfeb54ea` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison.mp4) | h264 | 960 x 412 | 25/1 | 600 | 24.000000 | `7e0ee67139bf014e8f513d7fe4c4c325c4a84bd68f7d15ec59760106310cac64` |

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/browser-video.png)
- [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/studio-evaluation.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/video-validation.json)
- [swing-v3-api.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-api.json)

![Swing trained-policy and control comparison](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison_sheet.png)

![Swing: Physical tracking](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/physical-tracking.png)

![Swing: Reward and checkpoint learning](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/reward-learning.png)

![Swing: PPO loss history](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/ppo-losses.png)

![Swing: Studio saved skill verdict and rollout player](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/studio-evaluation.png)

### Source hashes

| Source/artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [swing.json](recipes/swing.json) | 756 | `37c683b8146aec303e7ae63c9d328846717a2e4781f8fee0a37250c77530fe0a` |
| [swing-v3.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3.json) | 756 | `37c683b8146aec303e7ae63c9d328846717a2e4781f8fee0a37250c77530fe0a` |
| [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/audit.json) | 52,563 | `e571d732c0f3b7ff19bdcd8c12efbfebda2b49d53533a6e0ad4725334cb5c3a4` |
| [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/video-validation.json) | 6,379 | `62ff5ca8c66d556cf3e1d2178b59da339f5330bae1b3e67cf0d1b13796838ba2` |
| [swing-v3-api.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-api.json) | 282,887 | `b64ac543dd98fd30ff7bf855b0a88dcbaaa439e8183c9665833eee6ffcabded7` |
| [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison_sheet.png) | 1,492,424 | `8b5c97fab468760e2b903212802d9e717d11eb0a5ed2ab7f6bd063c4a0cbff71` |
| [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/physical-tracking.png) | 359,762 | `932448909624d357c2895170b756c23f0d2b91ddd15f72b2ae39a2f35bd4d09d` |
| [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/reward-learning.png) | 203,453 | `c9577b684f4f2478a6eeb873f26478a43e57fe002e3bb596c86aca9ebd975520` |
| [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/ppo-losses.png) | 337,856 | `2a5906835b76b9e842895b11ef0bcc14c9a71565138837150e1d4d4b647459d9` |
| [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/browser-video.png) | 183,643 | `9e463c627bc09c0359742df6521f27d24780f43658d4d63eb1b03b68df629084` |
| [studio-evaluation.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/studio-evaluation.png) | 101,087 | `b59406d1b25ca84380514e687a997c45492a7f7b418a112b4f65884c8e1087df` |
| [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/comparison.mp4) | 2,275,553 | `7e0ee67139bf014e8f513d7fe4c4c325c4a84bd68f7d15ec59760106310cac64` |
| [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit/api-video.mp4) | 2,155,746 | `dee5b2805efcd8339f9e2389c78fbc6bcd4d095e4caa693d2f2e872bdfeb54ea` |

### Reproduction commands

Run all stages from the repository root with the Studio server at `http://127.0.0.1:63317`. Replace `YYYYMMDD-HHMMSS` consistently with a fresh timestamp; base and continuation runs must remain distinct.

Swing v3 repeats bootstrap PPO with a 524,288-transition budget, not a continuation of the failed v2 final policy. The v2 524k checkpoint passed while its 1M final failed; this shorter schedule still requires a new held-out audit and is not a predeclared pass.

This recipe requires a teacher-assisted initializer. Reproduce the BC/DAgger stage first; it uses privileged state only to create training labels, never as an inference input to the exported actor. Preserve the initializer's own verdict.

```bash
rlx/.venv-microduck/bin/python rlx/scripts/bootstrap_swing_e2e.py --source docs/remaining-scenarios-e2e/recipes/swing-teacher.json --output /tmp/swing-bootstrap-reproduction
mkdir -p rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS
test ! -e rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS/swing.safetensors
cp /tmp/swing-bootstrap-reproduction/swing.safetensors /tmp/swing-bootstrap-reproduction/swing.safetensors.json rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS/
```

Run from the repository root after starting the Studio server at `http://127.0.0.1:63317`:

```bash
node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute --base-url http://127.0.0.1:63317 --experiment swing --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run swing-reproduction-YYYYMMDD-HHMMSS --report rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-api.json --timeout-seconds 7200
rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-audit --seeds 501 502 503 504 505 --render
node duck-viewer/scripts/verify-rlx-video.mjs --recipe-json docs/remaining-scenarios-e2e/recipes/swing.json --run rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS --output rlx/artifacts/scenarios-e2e-20260907/swing-reproduction-YYYYMMDD-HHMMSS-audit --base-url http://127.0.0.1:63317 --playwright-package "${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}"
```

The audit command exits nonzero when the strict skill gate fails. That nonzero exit is expected evidence for a failed attempt, not permission to rewrite the result as successful.

## 8. Historical Running: `running-v1`

**Recorded skill status: NOT RUN.** Not a selected final policy.


Full raw evaluations, historical curves, and videos remain linked below:

- [running-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v1-api.json)

## 9. Historical Running: `running-v2`

**Recorded skill status: FAIL.** Not a selected final policy.

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 2,930 |
| Optimizer minibatches | 46,880 |
| Observed transitions | 6,000,640 |
| Requested transitions | 6,000,640 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 1.2958 |
| Last-100 mean_loss | -0.0089 |
| Last-100 policy_loss | -0.0097 |
| Last-100 value_loss | 0.0008 |
| Last-100 entropy | 0.2441 |
| Last-100 approximate_kl | 0.1124 |
| Last-100 clip_fraction | 0.5453 |
| Last-100 explained_variance | 0.8010 |

Recorded per-episode failures:

- **seed 101, episode 0:** aerial fraction below running target; command-directed displacement below target
- **seed 102, episode 0:** aerial fraction below running target; command-directed displacement below target
- **seed 103, episode 0:** aerial fraction below running target; command-directed displacement below target
- **seed 104, episode 0:** aerial fraction below running target; command-directed displacement below target
- **seed 105, episode 0:** aerial fraction below running target; command-directed displacement below target

Full raw evaluations, historical curves, and videos remain linked below:

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-audit/video-validation.json)
- [running-v2-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-api.json)
- [running-v2-api.log](../../rlx/artifacts/scenarios-e2e-20260907/running-v2-api.log)

## 10. Historical Running: `running-v3`

**Recorded skill status: FAIL.** Not a selected final policy.

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 2,930 |
| Optimizer minibatches | 46,880 |
| Observed transitions | 6,000,640 |
| Requested transitions | 6,000,640 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 1.2383 |
| Last-100 mean_loss | -0.0051 |
| Last-100 policy_loss | -0.0073 |
| Last-100 value_loss | 0.0022 |
| Last-100 entropy | 2.0482 |
| Last-100 approximate_kl | 0.1647 |
| Last-100 clip_fraction | 0.5357 |
| Last-100 explained_variance | 0.7433 |

Recorded per-episode failures:

- **seed 301, episode 0:** command-directed displacement below target
- **seed 302, episode 0:** none
- **seed 303, episode 0:** command-directed displacement below target
- **seed 304, episode 0:** command-directed displacement below target
- **seed 305, episode 0:** none

Full raw evaluations, historical curves, and videos remain linked below:

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-audit/video-validation.json)
- [running-v3-api.json](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-api.json)
- [running-v3-api.log](../../rlx/artifacts/scenarios-e2e-20260907/running-v3-api.log)

## 11. Historical Stilt Walking: `stilts-v1`

**Recorded skill status: TRAIN FAILED.** Not a selected final policy.


Training stopped at 1,980,416 transitions: RuntimeError: PPO gradients contains non-finite values: gradient tensors=critic.layers.0.weight, critic.layers.0.bias, critic.layers.2.weight, critic.layers.2.bias; observations[-7.21088..10; nonfinite=0], actions[-1.39449..3.84149; nonfinite=0], old_log_probabilities[-11.5659..8.13558; nonfinite=0], old_values[-1.64562..3.35344; nonfinite=0], advantages[-1.87452..1.62534; nonfinite=0], returns[-0.430656..3.33284; nonfinite=0]

Full raw evaluations, historical curves, and videos remain linked below:

- [stilts-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v1-api.json)

## 12. Historical Stilt Walking: `stilts-v2`

**Recorded skill status: FAIL.** Not a selected final policy.

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 2,930 |
| Optimizer minibatches | 46,880 |
| Observed transitions | 6,000,640 |
| Requested transitions | 6,000,640 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.7728 |
| Last-100 mean_loss | -0.0045 |
| Last-100 policy_loss | -0.0100 |
| Last-100 value_loss | 0.0055 |
| Last-100 entropy | 1.9163 |
| Last-100 approximate_kl | 0.0814 |
| Last-100 clip_fraction | 0.4561 |
| Last-100 explained_variance | 0.0249 |

Recorded per-episode failures:

- **seed 101, episode 0:** command-directed displacement below target
- **seed 102, episode 0:** command-directed displacement below target
- **seed 103, episode 0:** command-directed displacement below target
- **seed 104, episode 0:** command-directed displacement below target
- **seed 105, episode 0:** none

Full raw evaluations, historical curves, and videos remain linked below:

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-audit/video-validation.json)
- [stilts-v2-api.json](../../rlx/artifacts/scenarios-e2e-20260907/stilts-v2-api.json)

## 13. Historical Swing: `swing-v1`

**Recorded skill status: FAIL.** Not a selected final policy.

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 512 |
| Optimizer minibatches | 6,144 |
| Observed transitions | 1,048,576 |
| Requested transitions | 1,048,576 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.0664 |
| Last-100 mean_loss | -0.0711 |
| Last-100 policy_loss | -0.0074 |
| Last-100 value_loss | 9.801e-06 |
| Last-100 entropy | 12.7553 |
| Last-100 approximate_kl | 0.0034 |
| Last-100 clip_fraction | 0.1767 |
| Last-100 explained_variance | -4.0719 |

Recorded per-episode failures:

- **seed 101, episode 0:** bidirectional span below target
- **seed 102, episode 0:** bidirectional span below target
- **seed 103, episode 0:** bidirectional span below target
- **seed 104, episode 0:** bidirectional span below target
- **seed 105, episode 0:** bidirectional span below target

Full raw evaluations, historical curves, and videos remain linked below:

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-audit/video-validation.json)
- [swing-v1-api.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v1-api.json)

## 14. Historical Swing: `swing-v2`

**Recorded skill status: FAIL.** Not a selected final policy.

| Training evidence | Value |
| --- | --- |
| Finite optimizer metrics | yes |
| PPO rollout/update cycles | 256 |
| Optimizer minibatches | 2,048 |
| Observed transitions | 1,048,576 |
| Requested transitions | 1,048,576 |
| Monotonic update steps | yes |
| Training evidence gate | PASS |
| Maximum approximate KL | 0.0080 |
| Last-100 mean_loss | 0.3409 |
| Last-100 policy_loss | -0.0005 |
| Last-100 value_loss | 0.3415 |
| Last-100 entropy | -12.3716 |
| Last-100 approximate_kl | 0.0004 |
| Last-100 clip_fraction | 0.0724 |
| Last-100 explained_variance | -2.5277 |

Recorded per-episode failures:

- **seed 301, episode 0:** invalid swing geometry; one or both strings lost positive spring tension
- **seed 302, episode 0:** invalid swing geometry; one or both strings lost positive spring tension
- **seed 303, episode 0:** invalid swing geometry; one or both strings lost positive spring tension
- **seed 304, episode 0:** invalid swing geometry; one or both strings lost positive spring tension
- **seed 305, episode 0:** invalid swing geometry; one or both strings lost positive spring tension

Full raw evaluations, historical curves, and videos remain linked below:

- [comparison_sheet.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/comparison_sheet.png)
- [physical-tracking.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/physical-tracking.png)
- [reward-learning.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/reward-learning.png)
- [ppo-losses.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/ppo-losses.png)
- [browser-video.png](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/browser-video.png)
- [comparison.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/comparison.mp4)
- [api-video.mp4](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/api-video.mp4)
- [audit.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/audit.json)
- [video-validation.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-audit/video-validation.json)
- [swing-v2-api.json](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-api.json)
- [swing-v2-api.log](../../rlx/artifacts/scenarios-e2e-20260907/swing-v2-api.log)

## 15. Stilt training failure and ELU diagnosis

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

## 16. Tracked reproduction recipes

These copies are tracked with the report so a fresh checkout does not depend on the gitignored evidence directory:

- [Running](recipes/running.json)
- [Running base](recipes/running-base.json)
- [Stilt Walking](recipes/stilts.json)
- [Stilt Walking base](recipes/stilts-base.json)
- [Swing](recipes/swing.json)

## 17. Interpretation

### What was tested, and what was not

This extends the Dance workflow to the Studio **Running**, **Stilt Walking**, and **Swing** recipes. The executable chain is real HTTP training, checkpoint export, deterministic skill evaluation, MuJoCo rendering, artifact download, H.264 decoding, and Chromium playback/seek. The separate audit uses the exported ONNX actor, not a scripted motion replay. Its comparison video reconstructs the recorded physics states without resetting a fallen control. The displayed zero-action failure is retained, not replaced by a new episode.

All final tests use the nominal XML actuator model. They do not establish BAM actuator equivalence, randomized robustness, arbitrary-speed command following, or hardware readiness. The locomotion recipes use fixed, observable forward commands; the stilt result applies to the selected **2 cm, blend 0, 0.014 kg-per-foot morphology**, not every height offered by the application. Swing uses planar action restrictions as declared in the recipe, starts at zero angle and angular rate, and must retain positive tension in both compliant strings at every 50 Hz sample.

The five final audit seeds are additional reset checks, not five independently trained policies. Swing's nominal reset is effectively identical across seeds, so its repeated passes demonstrate execution reproducibility, **not five independent robustness trials**. Locomotion resets vary, but all share the stated nominal configuration. Checkpoint-development seeds were used for tuning; final audit seed lists and all failed attempts remain visible in the evidence tables.

### Why the early locomotion attempts failed

Running v1 learned standing. Running v2 learned fast grounded stepping but did not satisfy the aerial-phase gate. Running v3 acquired an actual aerial gait around 0.7 m/s, but its trajectories often curved far enough to fail net intended displacement. A high body-frame speed alone was therefore insufficient. The evaluator was strengthened before final selection to measure signed progress against the intended command heading, rather than crediting accumulated travel around a circle. The acceptance floors were not reduced to accommodate these failures.

The optional `flight` term is zero by default and rewards both feet being airborne only with upright posture and command-directed speed. Its bounded form prevents stationary hopping or a falling robot from collecting full credit. Running's optional `yaw_tracking` term isolates yaw-rate error from the existing combined roll/pitch/yaw Gaussian: `exp(-((gyro_z - commanded_yaw_rate) / 0.25)^2)`. This gives a yaw-specific signal while leaving the native gyro, gravity, joint, prior-action, and command observations unchanged. It does not expose absolute world heading to the actor or reward an unobservable heading target. The same opt-in term is supported for stilts, but the selected Stilt v3 recipe does **not** enable it.

### Running: an aerial gait with verified forward progress

Running v4 resumes the 6,000,640-transition v3 actor for **2,097,152 additional PPO transitions**, with learning rate 0.0001, yaw-only weight 8, and combined angular-tracking weight reduced from 8 to 2. The forward command remains 0.75 m/s and the flight term remains 4. This changes the learning objective, not the evaluation criteria or the robot's physics.

All 16 API episodes and all five final audit seeds pass. The latter sustain **0.738–0.775 m/s**, **3.848–9.013 m** signed intended displacement in twelve seconds, **35.0–39.8%** both-feet airborne samples, and **167–170** alternating-support switches, with 100% upright coverage. On seed 501, raw deterministic return under the same final reward configuration improves by about **1,683** versus the resumed initializer. The unnormalized checkpoint curve rises overall and dips slightly at the final checkpoint; this dip is retained. World-path plots still show curvature. The result establishes running and the declared progress floor, not perfect straight-line regulation or hardware-quality gait smoothness.

### Stilt Walking: PPO acquisition followed by refinement

The Stilt v2 base was trained from random initialization for 6,000,640 transitions. It learned moving, alternating-foot balance on the selected physical stilt geometry, but many episodes circled and failed intended progress. Stilt v3 resumes that base for 1,048,576 additional PPO transitions and increases the existing angular-velocity tracking weight from 0.5 to 4. The inherited exploration standard deviation is loaded from the checkpoint; the recipe's initial-standard-deviation field does not reset a resumed actor.

The final API passes all 16 episodes. The separate seeds 501–505 pass all five episodes with mean forward speeds 0.230–0.242 m/s and intended displacements 1.281–2.289 m over ten seconds, without a fall. On comparison seed 501, deterministic raw return rises by about 758 under the **same final reward configuration**. This is stronger evidence of beneficial refinement than merely showing changed weights. The final gait still has measurable curvature; passing the defined progress floor does not mean perfectly straight tracking.

### Swing: distinguish teacher acquisition from PPO improvement

Fresh PPO in Swing v1 did not discover the large-amplitude motion: its strict still-start span was only about 2.23 degrees. Rather than crediting assisted resets as success, a bounded controller feasibility search and behavior-cloning/DAgger initializer were used to place useful motion in the policy's reachable distribution. Privileged physical state supplies teacher labels during data collection only. The exported actor still receives the ordinary **61 inputs and produces 14 actions**; no teacher controller is called at evaluation time. The bootstrap has `ppo_steps=0` and must not be described as PPO learning.

The selected stage-three initializer already passes the Swing skill gate. PPO therefore refines an acquired skill; it does not acquire Swing from scratch in the successful branch. Observation statistics are frozen for this continuation so changing normalization cannot silently alter the competent baseline. The critic and actor are still optimized, rewards remain normalized with separately updated statistics, and all normal PPO finite-value guards remain active.

Swing v2 showed why a final checkpoint cannot be accepted solely because training finished: its 524,288-step checkpoint passed, but continued optimization to 1,048,576 steps caused geometry violations and one sampled loss of string tension. Swing v3 repeats the initialization and PPO recipe with the shorter **524,288-transition budget** selected from that development evidence. Its API result has approximately **162.32 degrees** of symmetric span, 100% valid geometry and tension, and at least **0.209 N** sampled spring tension. This is a model-selection decision with an explicitly retained failed longer run, not a smoothed graph or relaxed threshold. Alignment approaches the 0.05 limit, so the pass is nominal and has limited margin.

The independent ONNX audit reproduces the pass and increases seed-501 raw return from **6,126.39 to 7,016.89**, about **890.50** or **14.5%**, compared with the saved initial checkpoint. This supports a beneficial PPO refinement of a teacher-acquired skill. It does not justify relabeling the bootstrap's behavior cloning as PPO or hiding the subsequent failed longer continuation.

### How to read the reward and loss figures

Each figure comes from saved JSONL telemetry or actual deterministic checkpoint rollouts. No reward or loss series is invented, made monotonic, or truncated to remove startup spikes. The reward figure deliberately separates deterministic unnormalized checkpoint return, deterministic survival, normalized stochastic collection reward, and raw stochastic completed-episode return. Those are different measurements: normalizer scale evolves, training actions are sampled, and occasional exploratory falls shorten episodes. A downward normalized training curve is not by itself evidence that the exported deterministic policy got worse.

For example, Stilt v3's deterministic checkpoint return increases while its normalized stochastic reward decreases during part of training. Its occasional raw training-return drops coincide with shortened exploratory episodes; the independent deterministic gait is the acceptance measurement. PPO policy loss is a batch-relative clipped surrogate, not a supervised prediction-error curve. The reports retain value loss, approximate KL, clip fraction, entropy, explained variance, and update duration so a small scalar loss cannot conceal an unstable or uninformative critic. In particular, poor or negative explained variance in Swing is disclosed rather than presented as healthy value-function convergence.

### Numerical repair and evidence integrity

Stilt v1 aborted on nonfinite critic gradients. A separate installed-MLX probe reproduced finite ELU forward outputs but NaN gradients for large positive inputs. Bounding the inactive exponential branch preserves the ELU function and ONNX layout while eliminating that reproduced gradient-overflow mechanism. Regression tests cover eager and compiled gradients, forward parity, checkpoint compatibility, and the unchanged observation/action interface. The exact failed training minibatch was not saved, so the probe establishes a consistent failure mechanism rather than an exact replay of the original failure.

The evidence retains unsuccessful runs and distinguishes a completed pipeline from a learned skill. Null controls must fail, exported ONNX must match the checkpoint, final episodes must meet their physical gates, and video validation must establish actual downloadable and playable files. The native tests, viewer tests, lint/build results, and known broader-suite failures are reported separately; optional GPU-stack collection failures are not recast as passing CPU coverage. Nothing in this report certifies deployment to a real robot.

The browser test also visits the actual Studio page, selects each of the three experiments, loads the final run by name, verifies its **Accepted** skill verdict, and plays the embedded video at the expected 12/10/24-second duration. Those screenshots complement the direct artifact-download and standalone Chromium playback checks. No UI test clicks Train, changes the completed policy, or treats the disconnected live lab as a policy failure.

That test caught the rollout player's two-column minimum widths overflowing the narrow evaluation card. The player now stacks inside the card, its copy refers to the saved rollout rather than always claiming 24 seconds, and the browser check asserts zero horizontal overflow and scroll offset for all three final runs.

### Implementation map and reproduction boundaries

| File or area | Change and purpose |
| --- | --- |
| `rlx/rlx/models/microduck.py` | Stable ELU activation; same model topology, checkpoint keys, and ONNX function. |
| `rlx/rlx/environments/locomotion_evaluation.py` | Fail-closed per-episode locomotion acceptance, including net intended displacement and contact/aerial evidence. |
| `rlx/rlx/environments/microduck_recipes.py` | Physical locomotion metrics, optional fixed command, bounded flight/yaw shaping, and existing morphology/physics reuse. |
| `rlx/rlx/environments/microduck.py` | Optional observation-statistics freeze for a competent warm start; reward statistics remain separate. |
| `rlx/examples/ppo_microduck_studio.py` | Native PPO training/evaluation integration, continuation provenance, normalization settings, and saved skill assessments. |
| `duck-viewer/lib/rlx-job.ts` and `experiments.ts` | Recipe/API plumbing, correct full evaluation horizons, optional reward weights, and retention of only matching saved evaluations. |
| `duck-viewer/lib/evaluation.ts` | Explicit skill status required for all built-in scenarios; pipeline success is not task acceptance. |
| `duck-viewer/scripts/rlx-dance-api-e2e.mjs` | Shared four-scenario HTTP driver; preserves failed-skill render/export evidence but exits nonzero. |
| `rlx/scripts/audit_scenarios.py` | Independent deterministic ONNX/control evaluation, all-checkpoint curves, PPO telemetry, hashes, and no-reset physics video. |
| `duck-viewer/scripts/verify-rlx-video.mjs` | Hash-linked artifact bytes, range requests, H.264 frame counts, duration, and Chromium playback/seek. |
| `rlx/scripts/bootstrap_swing_e2e.py` | Bounded, explicitly non-PPO teacher BC/DAgger initializer; teacher labels never become privileged inference inputs. |
| `docs/remaining-scenarios-e2e` | Tracked selected/base/teacher recipes, guarded reproduction, report generator, interpretation, and PDF build. |

The existing RLX PPO optimizer and rollout infrastructure are reused, not replaced with an animation or a teacher controller at inference. No new dependencies were installed. The successful locomotion continuations require their base checkpoints; the reproduction commands train those bases and copy both safetensors and normalization sidecars only after checking successful training. A base skill failure is retained and allowed only as the specifically recognized failure of a completed training run. Reusing a stale or failed-training checkpoint is not allowed. Swing likewise requires the disclosed bootstrap before the PPO recipe.

The generated artifacts live in the gitignored `rlx/artifacts/scenarios-e2e-20260907` and `rlx/runs/studio` directories. They are present locally and linked from this report, but a fresh clone does not automatically contain the MP4s, checkpoints, or raw traces. Retain those directories with the report when sharing a complete evidence package, or rerun the tracked reproduction recipes. The unchanged Dance evidence remains a separate report.

### Primary references for interpreting PPO

These references explain the optimization and evaluation concepts, not the outcome of these particular runs. This project executes RLX PPO; the SB3 implementation is cited for metric definitions and scaling cautions, not as the engine used here.

1. Schulman et al., *Proximal Policy Optimization Algorithms* (2017), original clipped-surrogate method: <https://arxiv.org/abs/1707.06347>.
2. OpenAI Spinning Up, *Proximal Policy Optimization*, stochastic exploration, policy/value objectives and approximate-KL diagnostics: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>.
3. Stable-Baselines3 2.6.0 PPO source documentation, policy/value/entropy losses, approximate KL, clipping fraction and reward-scaling caveat: <https://stable-baselines3.readthedocs.io/en/v2.6.0/_modules/stable_baselines3/ppo/ppo.html>.
4. Stable-Baselines3, *Reinforcement Learning Tips and Tricks*, separate evaluation of stochastic training policies: <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>.

## 18. Final conclusion

**Report result: PASS.**

All three selected scenarios have matching successful API jobs, passing independent audits, passing video validation, and current artifact hashes. This remains simulation-only evidence under the stated XML conditions; Swing uses a teacher-assisted initializer before PPO.
