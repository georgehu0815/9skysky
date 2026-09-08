# Complete Studio restart verification

Author: George Hu  
Date: September 8, 2026

## What changed

`restart-lab.sh` now manages the complete local application rather than delegating
to a legacy script that globally killed matching processes, reset the lab roster,
and then installed npm packages on every invocation.

The service topology is:

```text
restart-lab.sh
  ├─ preflight: installed RLX Python → MLX + four MuJoCo environments
  ├─ duck-lab :8788 → simulation/WebSocket API
  └─ Next.js :63317 → Studio UI + /api/rlx
                        └─ on-demand Python workers
                           train → eval → render → export
```

There is no independent, permanently listening PPO daemon. Restarting Next.js
restarts its API job manager. Workers use the existing
`rlx/.venv-microduck/bin/python` when available; the original isolated `uv`
launcher remains a fallback. Preflight uses the same runtime as subsequent API
jobs and checks MLX execution, all four environment resets/transitions, finite
observations/rewards, and the 61-observation/14-action contract.

### Safety and failure behavior

- A filesystem lock prevents simultaneous restarts and rejects new RLX launches
  while restart/preflight/verification is in progress. Read-only evidence routes
  remain available.
- Existing active jobs cause a default restart to fail before stopping services.
  `--stop-jobs` explicitly permits cancellation and process-tree termination;
  it also permits recovery when the API cannot report its job state.
- Listener ownership is checked using the process working directory and command.
  Foreign listeners are not killed. Shutdown targets captured service trees,
  not a machine-wide process-name match.
- The script preserves the roster, checkpoints, evaluations, training journals,
  and rendered media. It does not run `npm install` or pass `--fresh`.
- Runtime preflight failures leave existing services running. Failed readiness
  exits nonzero and leaves logs for diagnosis; merely opening a port is not a pass.
- Interrupted restarts can leave a stale lock after an uncatchable interruption.
  Inspect `.restart-lab/lock/pid` before removing it.

## Reproduce

```bash
./restart-lab.sh
./restart-lab.sh --check

# For a new workspace that has not trained the four skills yet:
./restart-lab.sh --readiness-only

# Only when deliberately terminating active/unknown work:
./restart-lab.sh --stop-jobs

node --test scripts/restart-lab.test.mjs
cd duck-viewer
npm test
npm run lint
npx tsc --noEmit
npm run build
STUDIO_EVIDENCE_DIR=../.restart-lab/browser npm run e2e:studio
```

`PORT`, `LAB_PORT`, and `MICRODUCK_RESTART_TIMEOUT` configure the listening ports
and startup wait. `MICRODUCK_STUDIO_PYTHON_DIRECT` selects an installed Python
environment; `MICRODUCK_STUDIO_PYTHON` selects a system Python for the original
uv-based launcher. See the root README for defaults and prerequisites.

## Recorded evidence

The repaired script completed a real cold restart of the existing services, then
a repeat restart using the installed RLX environment without dependency downloads.
The configured corporate package index did not support the attempted offline uv
resolution, so the implementation does **not** rely on an offline-cache assumption.

Strict verification rediscovered these existing accepted policies:

| Scenario | Saved run | Restored selected-stage rewards | PPO history segments |
|---|---|---:|---:|
| Dance | `dance-e2e-20260907-low-noise` | 1,954 | 1 |
| Swing | `swing-e2e-20260907-v3` | 128 | 1 |
| Running | `running-e2e-20260907-v4` | 1,024 | 2 |
| Stilt Walking | `stilts-e2e-20260907-v3` | 512 | 2 |

The checks require saved evaluation success, matching saved recipes,
`renderVerified`, persisted reward/loss samples in each PPO stage, and valid MP4
HTTP responses. The selected-stage sample count is **not** the cumulative count;
Running and Stilts also restore their ancestor stages. Swing's teacher
initialization is separate from its PPO segment.

Fresh, separate runs named `restart-smoke-20260908-<scenario>` each executed a
four-transition PPO smoke job, evaluation, rendering, and ONNX export through the
restarted API. **All 16 operations succeeded.** All four newly generated MP4s
passed full FFmpeg decoding. These tiny jobs verify execution and artifact
generation, not acquisition of a skill.

Local machine-readable evidence is kept in ignored `.restart-lab/`:

- `verification.json`: strict four-scenario saved-evidence checks.
- `rlx-preflight.log`: MLX and four-environment runtime checks.
- `restart-repeat.log`: repeat restart output.
- `smoke-<scenario>.json`: actual API submissions and operation results.
- `smoke-verification.json`: all 16 outcomes and full video-decoding checks.
- `browser/`: desktop/mobile screenshots and browser verification receipt.
- `node-tests.log`, `shell-tests.log`, `lint.log`, `typecheck.log`: regression logs.

Regression validation: **118 Studio Node tests and 13 shell lifecycle tests
passed**; ESLint, strict TypeScript, and the production build passed. The build
has no dynamic executable tracing warning: the installed interpreter is passed
as an argument through the system `env` executable, without invoking a shell.
The final launcher also passed a live repeat evaluation/render/export against
the disposable Dance checkpoint (`final-runtime-smoke.json`).

The browser regression passed **4/4 desktop/mobile scenarios with zero page
errors**. It played each saved video (Dance 8 seconds, Swing 24 seconds, Running
12 seconds, Stilts 10 seconds), captured full-lifecycle reward/loss charts,
checked mobile overflow, and verified that switching runs clears stale evidence
and resets the visual-review download gate. A final process inspection found
one listener per configured port and no leftover PPO trainer processes.

## Scope of the result

The restart restores previously verified **simulation** policies. It does not
retrain them, claim fresh learning from four transitions, or establish hardware
safety. Dance is the existing short excerpt; Swing uses teacher initialization
followed by PPO refinement. The original policy audits, reward/loss plots, and
simulation limitations remain documented in `REPORT.md` and its PDF.
