"""Validate the book's build, local evidence, parameter coverage and examples."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image


BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]


def local_links(markdown):
    document = json.loads(
        subprocess.check_output(
            ["pandoc", "--from=markdown", "--to=json"], input=markdown, text=True
        )
    )
    links = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("t") in ("Image", "Link"):
                links.append(value["c"][-1][0])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(document)
    missing = []
    for link in links:
        parsed = urlsplit(link)
        if (
            not parsed.scheme
            and parsed.path
            and not (BOOK / unquote(parsed.path)).exists()
        ):
            missing.append(link)
    return links, missing


def validate_examples(markdown):
    counts = {"bash": 0, "python": 0, "json": 0}
    for language, source in re.findall(
        r"```(bash|python|json)\n(.*?)\n```", markdown, re.S
    ):
        if language == "bash":
            subprocess.run(
                ["bash", "-n"], input=source, text=True, check=True, capture_output=True
            )
        elif language == "python":
            ast.parse(source)
        else:
            json.loads(source)
        counts[language] += 1
    return counts


def validate_pdf(pdf):
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert "George Hu" in info
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    assert pages >= 40
    log = (BOOK / "build/xelatex-3.log").read_text()
    bad = [
        line
        for line in log.splitlines()
        if any(
            token in line
            for token in (
                "Overfull",
                "Missing character",
                "Float too large",
                "! LaTeX Error",
            )
        )
    ]
    assert not bad, bad
    with tempfile.TemporaryDirectory() as temporary:
        bbox = Path(temporary) / "bbox.html"
        subprocess.run(["pdftotext", "-bbox", str(pdf), str(bbox)], check=True)
        document = ET.parse(bbox)
        namespace = {"x": "http://www.w3.org/1999/xhtml"}
        outside = []
        for number, page in enumerate(document.findall(".//x:page", namespace), 1):
            width, height = float(page.attrib["width"]), float(page.attrib["height"])
            for word in page.findall(".//x:word", namespace):
                bounds = {
                    key: float(word.attrib[key])
                    for key in ("xMin", "xMax", "yMin", "yMax")
                }
                if (
                    bounds["xMin"] < 0
                    or bounds["xMax"] > width + 1
                    or bounds["yMin"] < 0
                    or bounds["yMax"] > height + 1
                ):
                    outside.append(
                        {"page": number, "word": word.text, "bounds": bounds}
                    )
        assert not outside, outside[:10]
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    for required in (
        "George Hu",
        "162.32",
        "4,001,792",
        "Huber",
        "Complete reproduction appendix",
    ):
        assert required in text, required
    assert "\\frac" not in text and "$$" not in text
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    image_rows = subprocess.check_output(["pdfimages", "-list", str(pdf)], text=True)
    body_resolutions = [
        min(int(columns[12]), int(columns[13]))
        for line in image_rows.splitlines()[2:]
        if (columns := line.split()) and columns[2] == "image" and int(columns[0]) > 1
    ]
    assert body_resolutions and min(body_resolutions) >= 250
    distorted_images = [
        line
        for line in image_rows.splitlines()[2:]
        if (columns := line.split())
        and columns[2] == "image"
        and int(columns[0]) > 1
        and abs(int(columns[12]) - int(columns[13])) > 1
    ]
    assert not distorted_images, distorted_images
    return {
        "pages": pages,
        "out_of_page_words": 0,
        "typesetting_errors": bad,
        "math_rendered_without_raw_delimiters": True,
        "fonts": fonts,
        "bytes": pdf.stat().st_size,
        "body_raster_minimum_ppi": min(body_resolutions),
        "body_raster_aspect_ratios_preserved": True,
    }


def main():
    markdown = (BOOK / "microduck-training-student-book.md").read_text()
    links, missing = local_links(markdown)
    assert not missing, missing
    coverage = json.loads((BOOK / "parameter-coverage.json").read_text())
    source = (ROOT / "duck-viewer/lib/rlx-job.ts").read_text()
    declaration = source.split("export interface RlxRecipe {", 1)[1].split("\n}", 1)[0]
    fields = re.findall(r"^\s+(\w+)\??:", declaration, re.M)
    assert (
        set(fields)
        == set(coverage["source_recipe_fields"])
        == set(coverage["covered_fields"])
    )
    assert all(f"`{field}`" in markdown for field in fields)
    reward_source = (ROOT / "duck-viewer/lib/experiments.ts").read_text()
    reward_catalog = reward_source.split("const REWARDS:", 1)[1].split("\n};", 1)[0]
    reward_keys = re.findall(r'key: "(\w+)"', reward_catalog)
    assert len(reward_keys) == 49
    assert all(f"`{key}`" in markdown for key in reward_keys)
    manifest = json.loads((BOOK / "build-manifest.json").read_text())
    for kind, filename in (
        ("markdown", "microduck-training-student-book.md"),
        ("pdf", "microduck-training-student-book.pdf"),
    ):
        assert (
            hashlib.sha256((BOOK / filename).read_bytes()).hexdigest()
            == manifest[f"{kind}_sha256"]
        )
    for chapter, digest in manifest["chapters"].items():
        assert hashlib.sha256((BOOK / chapter).read_bytes()).hexdigest() == digest
    for relative, digest in manifest["ui_assets"].items():
        assert hashlib.sha256((BOOK / relative).read_bytes()).hexdigest() == digest
    assets = json.loads((BOOK / "asset-manifest.json").read_text())
    for asset in assets:
        target = BOOK / asset["asset"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == asset["sha256"]
        with Image.open(target) as image:
            assert list(image.size) == asset["pixels"]
    captures = json.loads((BOOK / "assets/ui-capture.json").read_text())
    assert len(captures["receipts"]) == 4
    assert all(
        receipt["accepted"] and receipt["playback"]["error"] is None
        for receipt in captures["receipts"]
    )
    teaching_assets = json.loads((BOOK / "learning-assets-manifest.json").read_text())
    for relative, digest in teaching_assets["assets"].items():
        assert hashlib.sha256((BOOK / relative).read_bytes()).hexdigest() == digest
    lab = json.loads(
        subprocess.check_output(
            [sys.executable, str(BOOK / "learning_lab.py"), "--evidence"], text=True
        )
    )
    assert set(lab) == {"dance", "running", "stilts", "swing"}
    subprocess.run([sys.executable, str(BOOK / "test_book.py")], check=True)
    audits = {}
    for scenario in ("dance", "running", "stilts", "swing"):
        audit = json.loads((BOOK / f"assets/{scenario}-audit.json").read_text())
        trials = audit["controls"]["trained"]
        assert len(trials) == 5 and all(trial["passed"] for trial in trials)
        audits[scenario] = {"saved_passes": len(trials), "new_training": False}
    examples = validate_examples(markdown)
    for script in BOOK.glob("*.py"):
        ast.parse(script.read_text())
    subprocess.run(["node", "--check", str(BOOK / "capture-ui.mjs")], check=True)
    result = {
        "passed": True,
        "author": "George Hu",
        "words": len(markdown.split()),
        "links_checked": len(links),
        "missing_links": missing,
        "recipe_fields_covered": len(fields),
        "reward_catalog_entries_checked": len(reward_keys),
        "build_manifest_matches_source_and_pdf": True,
        "original_assets_hash_checked": len(assets),
        "teaching_assets_hash_checked": len(teaching_assets["assets"]),
        "ui_assets_hash_checked": len(manifest["ui_assets"]),
        "runnable_evidence_lab_cases": len(lab),
        "book_regression_tests_passed": True,
        "example_syntax_checks": examples,
        "saved_studio_runs_accepted_and_played": 4,
        "saved_audit_checks": audits,
        "pdf": validate_pdf(BOOK / "microduck-training-student-book.pdf"),
        "scope": "Book validation and read-only saved-run UI playback, not a new training or simulation audit.",
    }
    (BOOK / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "pdf"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
