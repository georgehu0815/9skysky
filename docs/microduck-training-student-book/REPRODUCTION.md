# Complete reproduction appendix

These guarded procedures are imported verbatim from the authoritative
remaining-scenario report, so the book contains the base-training and copy
guards rather than merely linking to them. Execute each procedure in order;
do not start with its final continuation block. Replace every
`YYYYMMDD-HHMMSS` with the same fresh identifier for that experiment. Existing
checkpoints must never be overwritten. The Dance chapter contains its full
fresh-PPO procedure.

On a clean setup, substitute your actual Python environment for
`rlx/.venv-microduck/bin/python`. Set `PLAYWRIGHT_PACKAGE` to the absolute
`package.json` path of your installed Playwright package; the recorded default
is specific to this workstation. Keep Studio running in a separate terminal.
These commands launch substantial real training, not book generation.

Copy commands from the Markdown edition. The PDF adds visual continuation
arrows when a long line wraps; those arrows are typesetting, not shell syntax.

The Swing bootstrap output under `/tmp` must also be new: replace
`/tmp/swing-bootstrap-reproduction` consistently with a fresh path before
repeating that section. Its successful checkpoint is an imitation initializer,
not a zero-cost scratch PPO result.

## Running: all stages

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

## Stilt Walking: all stages

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

## Swing: all stages

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