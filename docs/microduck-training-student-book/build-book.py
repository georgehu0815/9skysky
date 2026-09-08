"""Assemble the textbook and typeset native mathematics and vector diagrams."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image


BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]


def reproduction_appendix():
    report = (ROOT / "docs/remaining-scenarios-e2e/REPORT.md").read_text()
    blocks = re.findall(r"### Reproduction commands\n(.*?)(?=\n## |\Z)", report, re.S)
    if len(blocks) != 3:
        raise ValueError(
            "Expected exactly three selected-scenario reproduction sections"
        )
    title = """# Complete reproduction appendix

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

"""
    return title + "\n\n".join(
        f"## {name}: all stages\n\n{block.strip()}"
        for name, block in zip(("Running", "Stilt Walking", "Swing"), blocks)
    )


def prepare_ui_crops():
    capture = json.loads((BOOK / "assets/ui-capture.json").read_text())
    running = next(
        receipt for receipt in capture["receipts"] if receipt["scenario"] == "running"
    )
    for panel in ("parameters", "evaluation"):
        source = BOOK / f"assets/studio-{panel}.png"
        with Image.open(source) as image:
            if panel == "evaluation" and "contentBottomPixels" in running:
                image = image.crop(
                    (
                        0,
                        0,
                        image.width,
                        min(image.height, running["contentBottomPixels"]),
                    )
                )
            midpoint = image.height // 2
            image.crop((0, 0, image.width, midpoint + 45)).save(
                BOOK / f"assets/studio-{panel}-top.png"
            )
            image.crop((0, midpoint - 45, image.width, image.height)).save(
                BOOK / f"assets/studio-{panel}-bottom.png"
            )


def assemble():
    chapters = sorted((BOOK / "chapters").glob("*.md"))
    metadata = """---
title: "Training Microduck: From Setup to Skill"
subtitle: "Dance, Swing, Running, and Stilt Walking"
author: "George Hu"
date: "September 2026 · Evidence edition"
lang: en-US
documentclass: article
papersize: letter
fontsize: 11pt
geometry:
  - margin=0.78in
mainfont: Times New Roman
sansfont: Arial
monofont: Menlo
colorlinks: true
toc-title: "Contents and learning path"
---

![Four accepted simulated scenarios. Author: George Hu.](assets/cover.svg){.book-cover}

"""
    appendix = reproduction_appendix()
    (BOOK / "REPRODUCTION.md").write_text(appendix)
    target = BOOK / "microduck-training-student-book.md"
    target.write_text(
        metadata
        + "\n\n".join(path.read_text().strip() for path in chapters)
        + "\n\n"
        + appendix
        + "\n"
    )
    prepare_ui_crops()
    return target


def build_pdf(source):
    build = BOOK / "build"
    build.mkdir(exist_ok=True)
    command = [
        "pandoc",
        source.name,
        "--from=markdown+tex_math_dollars+raw_tex",
        "--to=latex",
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--syntax-highlighting=tango",
        "--lua-filter=print-filter.lua",
        "--include-in-header=print-header.tex",
        "--output=build/book.tex",
    ]
    subprocess.run(command, cwd=BOOK, check=True)
    for index in range(3):
        process = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=build",
                "build/book.tex",
            ],
            cwd=BOOK,
            capture_output=True,
            text=True,
        )
        (build / f"xelatex-{index + 1}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(
                f"LaTeX failed; see {build / f'xelatex-{index + 1}.log'}\n{process.stdout[-3500:]}"
            )
    target = BOOK / "microduck-training-student-book.pdf"
    target.write_bytes((build / "book.pdf").read_bytes())
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    source = assemble()
    result = {"markdown": str(source), "words": len(source.read_text().split())}
    if not args.assemble_only:
        pdf = build_pdf(source)
        result["pdf"] = str(pdf)
        manifest = {
            "markdown_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "chapters": {
                str(path.relative_to(BOOK)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((BOOK / "chapters").glob("*.md"))
            },
            "ui_assets": {
                str(path.relative_to(BOOK)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((BOOK / "assets").glob("studio-*.png"))
            },
            "note": "Source revision identifies the inspected checkout; this book can be uncommitted.",
        }
        (BOOK / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
