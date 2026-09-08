"""Generate vector teaching figures without changing measured experiment plots."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


BOOK = Path(__file__).resolve().parent
ASSETS = BOOK / "assets"


def load_helpers():
    specification = importlib.util.spec_from_file_location(
        "prepare_assets", BOOK / "prepare-assets.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def clipping_figure(helpers):
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="700" viewBox="0 0 1040 700">',
        '<rect width="1040" height="700" rx="18" fill="#f4f8fa"/>',
        helpers.svg_text(
            38, 52, "PPO clipping depends on advantage sign", 30, weight="bold"
        ),
        helpers.svg_text(
            38,
            89,
            "Synthetic scalar objective; clip = 0.2; maximize the teal curve",
            22,
        ),
    ]
    for index, advantage in enumerate((2, -2)):
        left = 85 + 495 * index
        top = 160

        def point(ratio, value):
            horizontal = left + (ratio - 0.5) * 360
            vertical = top + (3.2 - value) * 55
            return f"{horizontal:.2f},{vertical:.2f}"

        for boundary in (0.8, 1.2):
            horizontal = left + (boundary - 0.5) * 360
            elements.append(
                f'<path d="M{horizontal} {top}v352" stroke="#b8c8d0" stroke-dasharray="6 5"/>'
            )
        elements.append(
            f'<path d="M{left} {top}v352h360" fill="none" stroke="#183448" stroke-width="2"/>'
        )
        raw = " ".join(point(ratio, advantage * ratio) for ratio in (0.5, 1.5))
        clipped = " ".join(
            point(ratio, min(advantage * ratio, advantage * min(max(ratio, 0.8), 1.2)))
            for ratio in (0.5, 0.8, 1.2, 1.5)
        )
        elements.append(
            f'<polyline points="{raw}" fill="none" stroke="#d47a35" stroke-width="4" stroke-dasharray="8 6"/>'
        )
        elements.append(
            f'<polyline points="{clipped}" fill="none" stroke="#087b79" stroke-width="5"/>'
        )
        elements.append(
            helpers.svg_text(
                left, 135, f"Advantage = {advantage:+d}", 24, weight="bold"
            )
        )
        for ratio in (0.5, 0.8, 1.0, 1.2, 1.5):
            elements.append(
                helpers.svg_text(left + (ratio - 0.5) * 360 - 16, 545, str(ratio), 18)
            )
        for value in (-3, 0, 3):
            elements.append(
                helpers.svg_text(
                    left - 36, top + (3.2 - value) * 55 + 7, str(value), 18
                )
            )
        elements.append(
            helpers.svg_text(left + 76, 579, "new / old action probability", 18)
        )
    elements.extend(
        [
            helpers.svg_text(
                38,
                627,
                "Teal: clipped surrogate     Orange dashed: unclipped advantage × ratio",
                22,
            ),
            helpers.svg_text(
                38,
                665,
                "Clipping limits the reward for helpful probability changes, not every harmful change.",
                20,
            ),
            "</svg>",
        ]
    )
    target = ASSETS / "clipped-objective.svg"
    target.write_text("\n".join(elements))
    subprocess.run(
        [
            "rsvg-convert",
            "--format",
            "pdf",
            str(target),
            "--output",
            str(target.with_suffix(".pdf")),
        ],
        check=True,
    )


def main():
    helpers = load_helpers()
    diagrams = {
        "gae-boundaries": (
            "Timeout-aware GAE: two boundary decisions",
            [
                (
                    "Collect the rollout",
                    "Store rewards, old critic values, terminal flags and final observations",
                ),
                (
                    "Select the bootstrap value",
                    "Terminal: zero; timeout: actual final value; continuing: next value",
                ),
                (
                    "Decide whether the trace continues",
                    "Stop across either kind of reset; never borrow the next episode's tail",
                ),
                (
                    "Sweep backward through each world",
                    "TD residual + gamma × lambda × continuing advantage tail",
                ),
                (
                    "Prepare PPO learning targets",
                    "Actor: advantages; critic: advantages + old values; then optimize",
                ),
            ],
            "Teaching schematic. Bootstrap at a timeout, but do not propagate across its reset.",
        ),
        "data-routes": (
            "Different data, different learning jobs",
            [
                (
                    "Dance reference",
                    "Time + joint targets → reward and tracking baseline, not BC labels",
                ),
                (
                    "Swing teacher",
                    "Privileged simulator state → labels paired with raw 61D observations",
                ),
                (
                    "BC and DAgger",
                    "Fit actor means; student-visited states expand supervised coverage",
                ),
                (
                    "PPO in all four cases",
                    "Current stochastic actor → fresh transitions → GAE → actor + critic",
                ),
                (
                    "Independent evaluation",
                    "Frozen ONNX mean → full physical episode → gates and saved video",
                ),
            ],
            "Teaching schematic. Supervised labels are not counted as PPO environment transitions.",
        ),
        "tensor-batch": (
            "Follow a Swing batch through RLX",
            [
                (
                    "One policy call",
                    "16 worlds: observations (16, 61) → actions (16, 14), values (16, 1)",
                ),
                (
                    "Collect 256 ticks",
                    "Observations (256, 16, 61); actions (256, 16, 14)",
                ),
                (
                    "Compute GAE",
                    "Rewards, flags, values → advantages and returns (256, 16)",
                ),
                (
                    "Flatten time and world",
                    "4096 aligned samples; observations (4096, 61), actions (4096, 14)",
                ),
                (
                    "Shuffle and optimize",
                    "4 minibatches of 1024 × 2 epochs = 8 parameter updates",
                ),
                (
                    "Collect again",
                    "Discard old rollout; new actor generates the next 4096 transitions",
                ),
            ],
            "Derived from the selected Swing recipe, not a measurement of throughput.",
        ),
        "transition-accounting": (
            "One Swing refinement, three counters",
            [
                ("One rollout", "16 environments × 256 steps = 4096 new transitions"),
                ("Whole PPO phase", "128 rollouts × 4096 = 524,288 new transitions"),
                (
                    "Optimizer work",
                    "128 rollouts × 2 epochs × 4 minibatches = 1024 updates",
                ),
                (
                    "Sample reuse",
                    "524,288 transitions × 2 epochs = 1,048,576 presentations",
                ),
                (
                    "Separate acquisition ledger",
                    "38,400 teacher labels at selected DAgger stage 3; PPO steps = 0",
                ),
            ],
            "Counters are not interchangeable. Earlier failed runs and evaluations are excluded.",
        ),
        "transfer-gates": (
            "From local learning to a hardware research plan",
            [
                (
                    "Established here",
                    "Nominal CPU MuJoCo + RLX actor: independent simulated skill gates",
                ),
                (
                    "Reproduce first",
                    "Match raw observations, action semantics, morphology and checkpoint",
                ),
                (
                    "Proposed robustness study",
                    "Matched tests of actuator dynamics, delay, noise and perturbations",
                ),
                (
                    "Official training stack",
                    "Port task to mjlab; retrain with the upstream BAM sim-to-real recipe",
                ),
                (
                    "Separate hardware review",
                    "Confirm mechanism, sensing, timing, limits and emergency stop",
                ),
                (
                    "Only then: controlled trials",
                    "Qualified supervision + approved safety process + recorded outcomes",
                ),
            ],
            "This is a proposed progression. No hardware trial is claimed by this book.",
        ),
    }
    for name, (title, steps, footer) in diagrams.items():
        helpers.diagram(name, title, steps, footer)
    clipping_figure(helpers)
    names = [*diagrams, "clipped-objective"]
    manifest = {
        "scope": "Generated teaching diagrams; not measured reward or performance data",
        "generator": "generate-learning-assets.py",
        "assets": {
            f"assets/{name}.{extension}": hashlib.sha256(
                (ASSETS / f"{name}.{extension}").read_bytes()
            ).hexdigest()
            for name in names
            for extension in ("svg", "pdf")
        },
    }
    (BOOK / "learning-assets-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"generated_diagrams": len(names)}))


if __name__ == "__main__":
    main()
