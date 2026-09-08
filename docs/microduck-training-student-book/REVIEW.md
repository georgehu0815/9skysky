# Editorial and technical review

The book is new documentation built from existing training evidence. No
training, API behavior, reward, or PPO implementation was changed to produce it.

## Source-grounded corrections made before publication

- Pin setup to Python 3.12 rather than inheriting RLX's 3.13 version file.
- Distinguish help-command inspection from an actual HTTP/environment smoke test.
- Correct Dance's unbounded actor output and applied `[-4,4]` target clipping.
- Document Swing's distinct action projection, seated target and observation slots.
- Separate core PPO defaults, effective Studio defaults, and recorded E2E overrides.
- Explain timeout-aware GAE, actual Huber critic loss and value coefficient 1.0.
- Include energy-proxy clipping and moving-joint subsets in the task equations.
- Make Swing chapter run/report/audit/bootstrap names consistently unique.
- Explain that the base checkpoint guard accepts success or a narrowly recognized
  skill failure; it never forces a new run to reproduce a historical failure.
- Correct the Running exercise: speed 0.30 fails the absolute floor, while its
  ratio 0.40 passes the ratio gate.
- Split tall UI captures into overlapping, unscaled crops for readable print.
- Remove the Markdown-cover placeholder figure from the PDF body; the PDF uses
  the full-page vector cover instead.

## Validation layers

`test_book.py` contains focused regressions for intentional bad syntax,
missing links, exact rollout budgets, sign-correct clipping, timeout boundaries
and all four saved audit budgets. `validate-book.py` checks local links, all 37 recipe
fields, all 49 reward-catalog entries, source/PDF build identity, original asset
hashes, example syntax, original audit verdicts, current saved-run UI playback
receipts, author metadata, typesetting errors, page bounds, and body-image DPI.
It also executes the standard-library evidence lab and book tests, verifies
twelve generated teaching-asset hashes, and binds every UI image to the PDF build.

## Expanded-edition review

- Added source-traced acquisition, input/output, reward, sample accounting,
  checkpoint handoff, failure and evaluation walkthroughs for all four cases.
- Added the runnable PPO arithmetic/evidence lab and a separate sim-to-real
  practicum. These do not replace the native trainer or authorize hardware use.
- Independently reviewed all four cases, the code laboratory, transfer chapter
  and lab script against source and saved evidence.
- Corrected BC mean reduction to `1/(14N)`, distinguished Swing's base offset
  from its final joint target, and fixed minibatch advantage-normalization scope.
- Strengthened Dance and Swing reproduction blocks to stop on failed freshness
  checks rather than allowing later commands to overwrite or mix artifacts.
- Refreshed four saved-run browser/playback receipts with timestamps and added
  focused verification-panel screenshots. Crops remove blank grid space without
  changing measured values or marking motion reviewed.
- Added six vector teaching diagrams, explicitly separate from measured
  reward/loss/physical plots, and print wrapping for long code paths/hashes.

## GAE workshop review

- Clarified that PPO requires advantage estimates, not GAE specifically, and
  located GAE between rollout collection and actor/critic optimization.
- Separated value bootstrapping from stopping the recursive advantage trace
  across resets, including actual final observations, rollout cutoffs,
  termination precedence and task-defined finite horizons.
- Corrected the supplied example's arithmetic: `0.99 * 0.9 - 1.1 = -0.209`,
  yielding timeout advantages `[0.58683558775, -0.1075645, -0.209]`.
- Added a runnable `--gae` workshop with critic targets, terminal comparison,
  lambda endpoints and focused regression tests. The example is synthetic;
  no policy training or hardware evaluation is implied.
- Qualified the sign of timeout-bootstrap bias and the lambda-one endpoint;
  a nonterminal segment still bootstraps even when lambda is one.

Representative pages are visually inspected for the cover, mathematics,
architecture, parameter tables, UI controls, reward/loss plots, physical traces,
and reproduction commands. Machine checks alone do not prove readable diagrams
or correct scientific interpretation.

The final machine-readable results are in `validation.json`. The original
edition recorded CLI help smokes for the API driver, Dance preparation,
scenario audit, Swing bootstrap and Studio entry point; the expanded-edition
checks are identified separately in the current validation report. They do not rerun the original training and
do not certify hardware deployment. The older book and reports are unchanged.
