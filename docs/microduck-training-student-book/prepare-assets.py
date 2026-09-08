"""Collect original experiment figures and create vector teaching diagrams."""

import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image


BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]
ASSETS = BOOK / "assets"
SOURCES = {
    "dance": ROOT / "rlx/artifacts/dance-e2e-20260907/final-audit",
    "running": ROOT / "rlx/artifacts/scenarios-e2e-20260907/running-v4-audit",
    "stilts": ROOT / "rlx/artifacts/scenarios-e2e-20260907/stilts-v3-audit",
    "swing": ROOT / "rlx/artifacts/scenarios-e2e-20260907/swing-v3-audit",
}


def svg_text(horizontal, vertical, text, size=22, color="#183448", weight="normal"):
    return (
        f'<text x="{horizontal}" y="{vertical}" font-family="DejaVu Sans" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"{html.escape(text)}</text>"
    )


def diagram(name, title, steps, footer):
    height = 115 + len(steps) * 116 + 75
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{height}" viewBox="0 0 1040 {height}">',
        f'<rect width="1040" height="{height}" rx="18" fill="#f4f8fa"/>',
        svg_text(38, 52, title, 31, weight="bold"),
    ]
    for index, (heading, detail) in enumerate(steps):
        top = 88 + index * 116
        elements.append(
            f'<rect x="38" y="{top}" width="964" height="90" rx="12" fill="white" stroke="#adc5d0" stroke-width="2"/>'
        )
        elements.append(
            svg_text(60, top + 34, f"{index + 1:02d}  {heading}", 24, "#087b79", "bold")
        )
        elements.append(svg_text(60, top + 65, detail, 20))
        if index < len(steps) - 1:
            elements.append(
                f'<path d="M520 {top + 91}v20m-7-7 7 7 7-7" fill="none" stroke="#087b79" stroke-width="3"/>'
            )
    elements.append(svg_text(38, height - 30, footer, 18))
    elements.append("</svg>")
    target = ASSETS / f"{name}.svg"
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


def cover():
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="816" height="1056" viewBox="0 0 816 1056">',
        '<rect width="816" height="1056" fill="#102b3c"/>',
        '<rect x="46" y="46" width="72" height="7" fill="#51d8bc"/>',
        svg_text(46, 91, "THE STUDENT LAB · EVIDENCE EDITION", 16, "#79dbce", "bold"),
        svg_text(46, 160, "Training Microduck", 46, "white", "bold"),
        svg_text(46, 215, "From Setup to Skill", 42, "white", "bold"),
        svg_text(46, 258, "Dance · Swing · Running · Stilt Walking", 23, "#d5e8ef"),
        svg_text(46, 298, "Author: George Hu", 24, "#79dbce", "bold"),
    ]
    captions = {
        "dance": ("DANCE IMITATION", "8-second Bachata excerpt · fresh PPO"),
        "swing": ("SELF-PUMPED SWING", "162.32° span · teacher + PPO"),
        "running": ("RUNNING", "Aerial phases + forward progress"),
        "stilts": ("STILT WALKING", "2 cm extensions · fresh PPO + refinement"),
    }
    for index, scenario in enumerate(("dance", "swing", "running", "stilts")):
        horizontal = 46 + (index % 2) * 370
        vertical = 348 + (index // 2) * 259
        encoded = base64.b64encode(
            (ASSETS / f"{scenario}-frame.png").read_bytes()
        ).decode()
        elements.append(
            f'<image x="{horizontal}" y="{vertical}" width="354" height="199.125" xlink:href="data:image/png;base64,{encoded}"/>'
        )
        title, caption = captions[scenario]
        elements.append(
            svg_text(horizontal, vertical + 220, title, 17, "white", "bold")
        )
        elements.append(svg_text(horizontal, vertical + 239, caption, 12, "#b9d1dc"))
    elements.extend(
        [
            svg_text(
                46,
                884,
                "Real frames from accepted MuJoCo policy rollouts",
                19,
                "#79dbce",
                "bold",
            ),
            svg_text(
                46,
                919,
                "Mathematics • Code • Studio UI • Reward and loss evidence",
                17,
                "white",
            ),
            svg_text(
                46,
                951,
                "Build, measure, reproduce—and design the next experiment.",
                17,
                "#d5e8ef",
            ),
            svg_text(
                46,
                1010,
                "September 2026 | Local simulation, not hardware certification",
                15,
                "#b9d1dc",
            ),
            "</svg>",
        ]
    )
    target = ASSETS / "cover.svg"
    target.write_text("\n".join(elements))
    subprocess.run(
        [
            "rsvg-convert",
            "--format",
            "pdf",
            str(target),
            "--output",
            str(ASSETS / "cover.pdf"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsvg-convert",
            "--width",
            "1224",
            str(target),
            "--output",
            str(ASSETS / "cover-preview.png"),
        ],
        check=True,
    )


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scenario, source in SOURCES.items():
        plots = {
            "reward": "reward-learning.png",
            "loss": "ppo-losses.png",
            "tracking": "joint-tracking.png"
            if scenario == "dance"
            else "physical-tracking.png",
        }
        for kind, filename in plots.items():
            original = source / filename
            target = ASSETS / f"{scenario}-{kind}.png"
            shutil.copy2(original, target)
            with Image.open(target) as image:
                dimensions = image.size
            manifest.append(
                {
                    "asset": str(target.relative_to(BOOK)),
                    "source": str(original.relative_to(ROOT)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "pixels": dimensions,
                    "transformation": "byte-identical original plot; no rescaling",
                }
            )
        video = source / "api-video.mp4"
        frame = ASSETS / f"{scenario}-frame.png"
        seconds = {"dance": 3, "running": 6, "stilts": 5, "swing": 18}[scenario]
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(seconds),
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(frame),
            ],
            check=True,
        )
        manifest.append(
            {
                "asset": str(frame.relative_to(BOOK)),
                "source": str(video.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                "timestamp_s": seconds,
                "pixels": Image.open(frame).size,
                "transformation": "decoded native-resolution frame; no AI imagery",
            }
        )
        for filename in ("audit.json", "video-validation.json"):
            target = ASSETS / f"{scenario}-{filename}"
            shutil.copy2(source / filename, target)
    diagram(
        "architecture",
        "One experiment, two connected loops",
        [
            (
                "Specify a task",
                "Clip or command + morphology + versioned recipe + seed",
            ),
            ("Studio / API", "Validated request → job runner → RLX Studio entry point"),
            (
                "Collect experience",
                "CPU MuJoCo workers: 61 observations → 14 actions at 50 Hz",
            ),
            (
                "Learn with PPO",
                "MLX/Metal actor + critic updates; repeat collection and learning",
            ),
            (
                "Freeze and export",
                "Checkpoint + normalizer sidecar → deterministic ONNX",
            ),
            (
                "Test independently",
                "Skill gates + controls + full MP4 + physical traces + hashes",
            ),
        ],
        "Feedback: failed physical tests revise the next experiment, never the acceptance gate.",
    )
    diagram(
        "ppo-loop",
        "Inside one PPO iteration",
        [
            (
                "Sample",
                "N parallel worlds × T control steps; save old log probabilities",
            ),
            (
                "Estimate",
                "Timeout-aware critic bootstrap → GAE advantages and return targets",
            ),
            (
                "Optimize",
                "E epochs × M minibatches; clipped policy + robust value loss",
            ),
            (
                "Diagnose",
                "Raw return, normalized reward, KL, clip fraction, entropy, finiteness",
            ),
            (
                "Repeat / evaluate",
                "Next rollout uses the new actor; held-out test uses its mean action",
            ),
        ],
        "A decreasing optimizer loss is not a physical skill test.",
    )
    flows = {
        "dance": (
            "Dance: turn choreography into a testable target",
            [
                ("Input", "Bachata JSON → hashed first 8 seconds; 14 joint targets"),
                (
                    "Observe",
                    "Proprioception + task phase in the existing 61-slot layout",
                ),
                (
                    "Optimize",
                    "Fresh PPO; precise pose kernel; physical balance maintained",
                ),
                (
                    "Evaluate",
                    "Pose and leg tracking vs static baselines; balance; full coverage",
                ),
                ("Output", "ONNX + 8-second audit + 16-second two-cycle video"),
            ],
        ),
        "stilts": (
            "Stilts: adapt the gait to a changed body",
            [
                (
                    "Input",
                    "2 cm extensions; blend 0; 0.014 kg per foot; command 0.25 m/s",
                ),
                ("Acquire", "Fresh PPO base: 6,000,640 environment transitions"),
                ("Refine", "Resume actor + normalizers; strengthen angular tracking"),
                (
                    "Evaluate",
                    "10 seconds; signed progress; alternating contacts; upright",
                ),
                ("Output", "Selected continuation + morphology-specific ONNX and MP4"),
            ],
        ),
        "running": (
            "Running: reward speed without rewarding a shuffle",
            [
                ("Input", "Observable forward command 0.75 m/s; nominal robot"),
                ("Acquire", "Fresh PPO base; require both-feet aerial phases"),
                ("Refine", "Yaw-rate tracking reduces circular reward exploitation"),
                ("Evaluate", "12 seconds; displacement ≥ 2 m; aerial fraction ≥ 3%"),
                ("Output", "ONNX + speed/contact traces + full physical rollout"),
            ],
        ),
        "swing": (
            "Swing: teach a reachable skill, then refine with PPO",
            [
                ("Input", "Two-string mechanism at rest; nominal XML actuators"),
                (
                    "Establish feasibility",
                    "Privileged teacher generates labels, not inference observations",
                ),
                (
                    "Initialize",
                    "BC / DAgger stage 3; no PPO transitions credited to imitation",
                ),
                (
                    "Refine",
                    "Frozen observation statistics; 524,288 conservative PPO steps",
                ),
                (
                    "Evaluate",
                    "24 seconds; symmetric span ≥ 150°; both strings always taut",
                ),
            ],
        ),
    }
    for scenario, (title, steps) in flows.items():
        diagram(
            f"{scenario}-flow",
            title,
            steps,
            "All measured outcomes are nominal simulation evidence; proposed extensions are untested.",
        )
    cover()
    (BOOK / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
