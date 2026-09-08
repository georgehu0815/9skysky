# Training Microduck: student book

**Author: George Hu.** Four cases: Dance imitation, Self-pumped Swing,
Running, and Stilt Walking. This is a new book; the older `docs/student-book`
is unchanged.

- Read `microduck-training-student-book.md` or the matching PDF.
- Edit the ordered files in `chapters/`; the combined Markdown is generated.
- `REPRODUCTION.md` imports the complete guarded base/continuation procedures.
- `asset-manifest.json` identifies original measured plots and real cover frames.
- `assets/ui-capture.json` records fresh saved-run browser checks.
- `parameter-coverage.json` maps all API recipe fields; the validator also
  checks the documented reward entries against the source catalog.
- `validation.json` records book-build checks, not new training results.
- `chapters/02a-ppo-code-lab.md` follows tensor shapes, GAE, clipping and sample budgets through real RLX code.
- `chapters/02b-gae-workshop.md` explains why PPO uses GAE, timeout/reset masks, and a corrected three-transition numerical example.
- `learning_lab.py` runs deterministic arithmetic and reads all four bundled audits without training.
- `chapters/09-sim-to-real-practicum.md` provides controlled experiments, portability checks and a separately gated hardware research plan.
- `learning-assets-manifest.json` records the six new vector teaching figures, separate from measured plots.

## Start with the runnable student lab

```bash
python3 docs/microduck-training-student-book/learning_lab.py
python3 docs/microduck-training-student-book/learning_lab.py --evidence
python3 docs/microduck-training-student-book/learning_lab.py --gae
rlx/.venv-microduck/bin/python docs/microduck-training-student-book/test_book.py
```

These commands calculate example objectives and verify saved sample counts;
they do not launch a new PPO run, simulate a robot, or certify hardware safety.
Read the four expanded chapters for acquisition, continuation, dataset and
evaluation details before using the substantial training commands.

## Rebuild from bundled assets

From the workspace root, using the existing Python environment with Pillow:

```bash
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/build-book.py
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/validate-book.py
```

Requires the already used Pandoc, XeLaTeX, TeX packages/fonts, and Poppler
utilities. No dependency is installed by these scripts. TeX log files remain
under `build/` for diagnostics. `--assemble-only` builds just Markdown and crops.

The PDF can be rebuilt from the bundled assets. The stricter validator also
checks historical local evidence links: a fresh clone without the gitignored
source videos/artifacts will report missing links rather than pretending those
experiments were reproduced. The arithmetic/evidence lab uses the bundled audit
copies and remains usable without those original run directories.

To regenerate only the new explanatory diagrams (requires existing
`rsvg-convert`, not original run artifacts):

```bash
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/generate-learning-assets.py
```

## Refresh original evidence assets

Only needed when intentionally refreshing the evidence edition. Requires the
original gitignored run artifacts and the already running Studio server:

```bash
rlx/.venv-microduck/bin/python \
  docs/microduck-training-student-book/prepare-assets.py
node docs/microduck-training-student-book/capture-ui.mjs
```

`prepare-assets.py` copies unaltered measured plots, decodes actual video frames,
and creates vector SVG/PDF diagrams and cover. It does not run training.
`capture-ui.mjs` opens saved runs without submitting train/eval/render jobs.
Configure `STUDIO_URL` and `PLAYWRIGHT_PACKAGE` for another workstation.

The book's Swing outcome is explicitly **teacher initialization + PPO
refinement**, not scratch-PPO acquisition. All four outcomes are nominal
MuJoCo simulation evidence, not hardware or BAM transfer certification.
