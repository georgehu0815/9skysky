#!/usr/bin/env python3
"""Generate an evidence-led report for Swing, Running, and Stilt Walking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCENARIOS = (
    ("running", "Running"),
    ("stilts", "Stilt Walking"),
    ("swing", "Swing"),
)
DEFAULT_EVIDENCE_ROOT = Path("rlx/artifacts/scenarios-e2e-20260907")
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAMES = (
    "comparison_sheet.png",
    "physical-tracking.png",
    "reward-learning.png",
    "ppo-losses.png",
    "browser-video.png",
    "comparison.mp4",
    "api-video.mp4",
    "audit.json",
    "video-validation.json",
)
DIAGNOSTIC_FILES = (
    "activation-probe.json",
    "activation-before.log",
    "activation-after.log",
    "activation_probe.py",
)


@dataclass
class Attempt:
    scenario: str
    label: str
    identifier: str
    recipe_path: Path
    audit_dir: Path
    audit_path: Path
    video_path: Path
    api_path: Path
    progress_path: Path
    tracked_recipe_path: Path
    recipe: dict[str, Any] | None
    audit: dict[str, Any] | None
    video: dict[str, Any] | None
    api: dict[str, Any] | None
    errors: list[str]

    @property
    def status(self) -> str:
        if self.audit is not None:
            return "PASS" if self.audit.get("passed") is True else "FAIL"
        if api_training_failed(self.api):
            return "TRAIN FAILED"
        if api_training_succeeded(self.api):
            return "TRAINED / UNAUDITED"
        if progress_from_log(self.progress_path) is not None:
            return "TRAINING ACTIVE"
        return "NOT RUN"

    @property
    def evidence_complete(self) -> bool:
        return (
            self.audit is not None
            and self.audit.get("passed") is True
            and self.video is not None
            and self.video.get("passed") is True
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT,
        help=f"Evidence directory (default: {DEFAULT_EVIDENCE_ROOT})",
    )
    parser.add_argument("--running", default="running-v3", help="Running attempt ID")
    parser.add_argument("--stilts", default="stilts-v3", help="Stilt attempt ID")
    parser.add_argument("--swing", default="swing-v2", help="Swing attempt ID")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("REPORT.md"),
        help="Markdown output path",
    )
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: expected a JSON object")
        return None
    return value


def load_attempt(
    evidence_root: Path, scenario: str, label: str, identifier: str
) -> Attempt:
    errors: list[str] = []
    recipe_path = evidence_root / f"{identifier}.json"
    audit_dir = evidence_root / f"{identifier}-audit"
    audit_path = audit_dir / "audit.json"
    video_path = audit_dir / "video-validation.json"
    api_path = evidence_root / f"{identifier}-api.json"
    progress_path = evidence_root / f"{identifier}-api.log"
    tracked_recipe_path = Path(__file__).with_name("recipes") / f"{scenario}.json"
    artifact_recipe = load_json(recipe_path, errors)
    tracked_recipe = load_json(tracked_recipe_path, errors)
    return Attempt(
        scenario=scenario,
        label=label,
        identifier=identifier,
        recipe_path=recipe_path,
        audit_dir=audit_dir,
        audit_path=audit_path,
        video_path=video_path,
        api_path=api_path,
        progress_path=progress_path,
        tracked_recipe_path=tracked_recipe_path,
        recipe=artifact_recipe or tracked_recipe,
        audit=load_json(audit_path, errors),
        video=load_json(video_path, errors),
        api=load_json(api_path, errors),
        errors=errors,
    )


def api_operations(api: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not api:
        return []
    value = api.get("operations", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def api_operation_state(
    api: dict[str, Any] | None, action: str
) -> dict[str, Any] | None:
    for operation in api_operations(api):
        if operation.get("action") == action and isinstance(operation.get("state"), dict):
            return operation["state"]
    return None


def api_training_failed(api: dict[str, Any] | None) -> bool:
    state = api_operation_state(api, "train")
    return bool(state and state.get("phase") == "failed")


def api_training_succeeded(api: dict[str, Any] | None) -> bool:
    state = api_operation_state(api, "train")
    return bool(state and state.get("phase") == "succeeded")


def training_failure_summary(api: dict[str, Any] | None) -> dict[str, Any] | None:
    state = api_operation_state(api, "train")
    if not state or state.get("phase") != "failed":
        return None
    failure = api.get("failure") if api and isinstance(api.get("failure"), dict) else {}
    logs = state.get("logs") if isinstance(state.get("logs"), list) else []
    last_log = next(
        (line for line in reversed(logs) if isinstance(line, str) and line.strip()),
        None,
    )
    return {
        "steps": state.get("trainingSteps"),
        "total": state.get("trainingTotal"),
        "exit_code": state.get("exitCode"),
        "started": state.get("startedAt"),
        "finished": state.get("finishedAt"),
        "artifacts": state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {},
        "message": failure.get("message") or last_log or "training failed",
        "last_log": last_log,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_link(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def markdown_link(label: str, path: Path, output_dir: Path) -> str:
    return f"[{label}]({relative_link(path, output_dir)})"


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if value != 0 and abs(value) < 0.0001:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return " x ".join(str(item) for item in value)
    return str(value)


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    header_list = list(headers)
    output = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    output.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return output


def flatten_episodes(control: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = control.get("episodes")
    if isinstance(episodes, list) and episodes:
        return [episode for episode in episodes if isinstance(episode, dict)]
    return [control]


def controls(audit: dict[str, Any] | None, name: str) -> list[dict[str, Any]]:
    if not audit:
        return []
    value = audit.get("controls", {}).get(name, [])
    return value if isinstance(value, list) else []


def episode_rows(attempt: Attempt, control_name: str) -> tuple[list[str], list[list[str]]]:
    entries = controls(attempt.audit, control_name)
    if attempt.scenario == "swing":
        headers = [
            "Seed",
            "Episode",
            "Steps",
            "Complete",
            "Bidirectional span",
            "Negative peak",
            "Positive peak",
            "Tensioned",
            "Valid geometry",
            "Verdict",
        ]
        metric_keys = (
            "bidirectional_span_deg",
            "negative_peak_deg",
            "positive_peak_deg",
            "both_strings_tensioned_fraction",
            "valid_geometry_fraction",
        )
    else:
        headers = [
            "Seed",
            "Episode",
            "Steps",
            "Complete",
            "Upright",
            "Commanded",
            "Directed speed",
            "Displacement",
            "Speed ratio",
            "Verdict",
        ]
        metric_keys = ()
    rows: list[list[str]] = []
    for control in entries:
        for episode in flatten_episodes(control):
            verdict = "PASS" if episode.get("passed", control.get("passed")) is True else "FAIL"
            prefix = [
                format_value(control.get("seed")),
                format_value(episode.get("episode_index", 0)),
                format_value(episode.get("steps", control.get("steps"))),
                format_value(episode.get("complete")),
            ]
            if attempt.scenario == "swing":
                values = [format_value(episode.get(key, control.get(key))) for key in metric_keys]
            else:
                values = [
                    format_value(episode.get("upright_fraction")),
                    format_value(episode.get("commanded_fraction")),
                    format_value(episode.get("mean_command_directed_speed_m_s")),
                    format_value(episode.get("command_directed_displacement_m")),
                    format_value(episode.get("command_speed_tracking_ratio")),
                ]
            rows.append(prefix + values + [verdict])
    return headers, rows


def episode_physics_rows(
    attempt: Attempt, control_name: str
) -> tuple[list[str], list[list[str]]]:
    entries = controls(attempt.audit, control_name)
    if attempt.scenario == "swing":
        headers = [
            "Seed",
            "Episode",
            "Max lateral",
            "Max alignment",
            "Min/max string length",
            "Min spring tension",
        ]
    else:
        headers = [
            "Seed",
            "Episode",
            "L/R contact",
            "L/R air",
            "Aerial",
            "Support switches",
            "L/R liftoffs",
            "L/R touchdowns",
        ]
    rows: list[list[str]] = []
    for control in entries:
        for episode in flatten_episodes(control):
            prefix = [
                format_value(control.get("seed")),
                format_value(episode.get("episode_index", 0)),
            ]
            if attempt.scenario == "swing":
                values = [
                    format_value(episode.get("max_abs_lateral_m")),
                    format_value(episode.get("max_alignment")),
                    (
                        f"{format_value(episode.get('min_string_length_m'))} / "
                        f"{format_value(episode.get('max_string_length_m'))}"
                    ),
                    format_value(episode.get("min_spring_tension_n")),
                ]
            else:
                values = [
                    (
                        f"{format_value(episode.get('left_contact_fraction'))} / "
                        f"{format_value(episode.get('right_contact_fraction'))}"
                    ),
                    (
                        f"{format_value(episode.get('left_air_fraction'))} / "
                        f"{format_value(episode.get('right_air_fraction'))}"
                    ),
                    format_value(episode.get("aerial_fraction")),
                    format_value(episode.get("alternating_support_switches")),
                    (
                        f"{format_value(episode.get('left_liftoffs'))} / "
                        f"{format_value(episode.get('right_liftoffs'))}"
                    ),
                    (
                        f"{format_value(episode.get('left_touchdowns'))} / "
                        f"{format_value(episode.get('right_touchdowns'))}"
                    ),
                ]
            rows.append(prefix + values)
    return headers, rows


def episode_failure_lines(attempt: Attempt, control_name: str) -> list[str]:
    lines: list[str] = []
    for control in controls(attempt.audit, control_name):
        for episode in flatten_episodes(control):
            failures = episode.get("failures") or control.get("failures") or []
            label = (
                f"seed {format_value(control.get('seed'))}, "
                f"episode {format_value(episode.get('episode_index', 0))}"
            )
            lines.append(f"- **{label}:** {'; '.join(map(str, failures)) or 'none'}")
    return lines


def metric_means_rows(attempt: Attempt) -> list[list[str]]:
    rows: list[list[str]] = []
    for control_name in ("trained", "zero", "initial"):
        for control in controls(attempt.audit, control_name):
            means = control.get("metric_means")
            if not isinstance(means, dict):
                continue
            for key, value in sorted(means.items()):
                rows.append(
                    [
                        control_name,
                        format_value(control.get("seed")),
                        key,
                        format_value(value),
                    ]
                )
    return rows


def history_rows(attempt: Attempt) -> list[list[str]]:
    if not attempt.audit or not isinstance(attempt.audit.get("history"), list):
        return []
    rows = []
    for entry in attempt.audit["history"]:
        if not isinstance(entry, dict):
            continue
        rows.append(
            [
                format_value(entry.get("training_steps")),
                format_value(entry.get("steps")),
                format_value(entry.get("raw_return")),
                "PASS" if entry.get("passed") is True else "FAIL",
                "; ".join(map(str, entry.get("failures") or [])) or "none",
            ]
        )
    return rows


def training_rows(attempt: Attempt) -> list[list[str]]:
    training = attempt.audit.get("training", {}) if attempt.audit else {}
    rows = [
        ["Finite optimizer metrics", format_value(training.get("finite"))],
        ["PPO rollout/update cycles", format_value(training.get("updates"))],
        ["Optimizer minibatches", format_value(training.get("optimizer_steps"))],
        ["Observed transitions", format_value(training.get("observed_transitions"))],
        ["Requested transitions", format_value(training.get("requested_transitions"))],
        ["Monotonic update steps", format_value(training.get("monotonic_update_steps"))],
        ["Training evidence gate", "PASS" if training.get("passed") is True else "FAIL"],
        ["Maximum approximate KL", format_value(training.get("max_approximate_kl"))],
    ]
    last = training.get("last_100_means")
    if isinstance(last, dict):
        rows.extend([f"Last-100 {key}", format_value(value)] for key, value in last.items())
    return rows


def tracked_recipe_matches(attempt: Attempt) -> bool:
    return (
        attempt.recipe is not None
        and attempt.tracked_recipe_path.is_file()
        and load_json(attempt.tracked_recipe_path, []) == attempt.recipe
    )


def source_rows(attempt: Attempt, output_dir: Path) -> list[list[str]]:
    paths = [
        *([attempt.tracked_recipe_path] if tracked_recipe_matches(attempt) else []),
        attempt.recipe_path,
        attempt.audit_path,
        attempt.video_path,
        attempt.api_path,
        attempt.progress_path,
    ]
    paths.extend(attempt.audit_dir / name for name in ARTIFACT_NAMES)
    seen: set[Path] = set()
    rows = []
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        rows.append(
            [
                markdown_link(path.name, path, output_dir),
                format_value(path.stat().st_size),
                f"`{sha256(path)}`",
            ]
        )
    return rows


def artifact_lines(attempt: Attempt, output_dir: Path) -> list[str]:
    lines: list[str] = []
    for name in ARTIFACT_NAMES:
        path = attempt.audit_dir / name
        if path.is_file():
            lines.append(f"- {markdown_link(name, path, output_dir)}")
    if attempt.api_path.is_file():
        lines.append(f"- {markdown_link(attempt.api_path.name, attempt.api_path, output_dir)}")
    if attempt.progress_path.is_file():
        lines.append(
            f"- {markdown_link(attempt.progress_path.name, attempt.progress_path, output_dir)}"
        )
    return lines


def api_summary(api: dict[str, Any] | None) -> list[list[str]]:
    if not api:
        return []
    rows: list[list[str]] = []
    for operation in api.get("operations", []):
        if not isinstance(operation, dict):
            continue
        state = operation.get("state") if isinstance(operation.get("state"), dict) else {}
        rows.append(
            [
                operation.get("action", "unknown"),
                format_value(operation.get("accepted", {}).get("accepted")),
                state.get("phase", "unknown"),
                format_value(state.get("exitCode")),
                state.get("finishedAt", "n/a"),
            ]
        )
    return rows


def video_rows(video: dict[str, Any] | None, output_dir: Path) -> list[list[str]]:
    if not video:
        return []
    rows = []
    for entry in video.get("videos", []):
        if not isinstance(entry, dict):
            continue
        file_value = entry.get("file")
        file_path = Path(file_value) if isinstance(file_value, str) else None
        probe = entry.get("probe", {})
        stream = next(
            (
                item
                for item in probe.get("streams", [])
                if isinstance(item, dict) and item.get("codec_type") == "video"
            ),
            {},
        )
        display = (
            markdown_link(file_path.name, file_path, output_dir)
            if file_path and file_path.is_file()
            else str(file_value or "unknown")
        )
        rows.append(
            [
                display,
                stream.get("codec_name", "n/a"),
                f"{stream.get('width', 'n/a')} x {stream.get('height', 'n/a')}",
                stream.get("avg_frame_rate", "n/a"),
                stream.get("nb_read_frames", stream.get("nb_frames", "n/a")),
                probe.get("format", {}).get("duration", "n/a"),
                f"`{entry.get('sha256', 'n/a')}`",
            ]
        )
    return rows


def progress_from_log(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    matches = re.findall(
        r"train:\s+running\s+([0-9]+)\s*/\s*([0-9]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        return None
    current, total = matches[-1]
    return int(current), int(total)


def discovered_attempts(
    evidence_root: Path,
) -> list[tuple[str, str, str, Path | None]]:
    identifiers: set[str] = set()
    for path in evidence_root.glob("*.json"):
        if re.match(r"^(running|stilts|swing)-v\d+\.json$", path.name):
            identifiers.add(path.stem)
    identifiers.update(
        path.name[: -len("-audit")]
        for path in evidence_root.glob("*-audit")
        if path.is_dir()
    )
    identifiers.update(
        path.name[: -len("-api.json")]
        for path in evidence_root.glob("*-api.json")
        if re.match(r"^(running|stilts|swing)-v\d+-api\.json$", path.name)
    )
    identifiers.update(
        path.name[: -len("-api.log")]
        for path in evidence_root.glob("*-api.log")
        if re.match(r"^(running|stilts|swing)-v\d+-api\.log$", path.name)
    )
    rows = []
    for identifier in sorted(identifiers):
        audit_path = evidence_root / f"{identifier}-audit" / "audit.json"
        api_path = evidence_root / f"{identifier}-api.json"
        progress_path = evidence_root / f"{identifier}-api.log"
        audit: dict[str, Any] = {}
        api: dict[str, Any] = {}
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            audit = {}
        try:
            api = json.loads(api_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            api = {}
        recipe = audit.get("recipe") if isinstance(audit, dict) else {}
        if not recipe and isinstance(api, dict):
            recipe = api.get("requestedRecipe")
        scenario = recipe.get("experimentId") if isinstance(recipe, dict) else None
        if not scenario:
            scenario = identifier.split("-", 1)[0]
        if audit:
            status = "PASS" if audit.get("passed") is True else "FAIL"
            evidence_path: Path | None = audit_path
        elif api_training_failed(api):
            failure = training_failure_summary(api)
            steps = format_value(failure.get("steps")) if failure else "n/a"
            total = format_value(failure.get("total")) if failure else "n/a"
            status = f"TRAIN FAILED at {steps}/{total}"
            evidence_path = api_path
        elif api_training_succeeded(api):
            status = "TRAINED / UNAUDITED"
            evidence_path = api_path
        elif (progress := progress_from_log(progress_path)) is not None:
            status = f"TRAINING LOG at {progress[0]:,}/{progress[1]:,}"
            evidence_path = progress_path
        else:
            status = "RECIPE ONLY"
            evidence_path = evidence_root / f"{identifier}.json"
        rows.append((str(scenario), identifier, status, evidence_path))
    return rows


def log_status(evidence_root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    checks = (
        ("RLX native suite", "rlx-native-tests-final.log", r"(\d+ passed, \d+ skipped)"),
        ("Broader microduck_local suite", "microduck-contract-tests.log", r"(\d+ failed, \d+ passed, \d+ skipped)"),
        ("Full RLX collection", "rlx-tests.log", r"(\d+ errors? in [^\n]+)"),
        ("Viewer Node tests", "node-tests.log", r"tests (\d+).*pass (\d+).*fail (\d+)"),
        ("Viewer build", "node-build.log", r"(Compiled successfully)"),
        ("Viewer lint", "node-lint.log", r"(eslint)"),
    )
    for label, filename, pattern in checks:
        path = evidence_root / filename
        if not path.is_file():
            rows.append([label, "not captured", filename])
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(pattern, text, re.DOTALL)
        summary = "captured"
        if match:
            summary = " / ".join(group.strip() for group in match.groups())
        if filename == "rlx-tests.log" and "No module named 'mjlab'" in text:
            summary = "collection blocked: optional `mjlab` package missing"
        rows.append([label, summary, filename])
    return rows


def render_activation_diagnosis(
    evidence_root: Path, output_dir: Path, section: int
) -> list[str]:
    probe_path = evidence_root / "activation-probe.json"
    before_path = evidence_root / "activation-before.log"
    after_path = evidence_root / "activation-after.log"
    probe_errors: list[str] = []
    probe = load_json(probe_path, probe_errors)
    lines = [
        f"## {section}. Stilt training failure and ELU diagnosis",
        "",
        "The retained `stilts-v1` API record establishes the observed failure: PPO "
        "aborted at 1,980,416 of 4,001,792 transitions because critic-layer gradients "
        "became non-finite. The logged observations, actions, old log probabilities, "
        "old values, advantages, and returns were finite, and no final checkpoint or "
        "ONNX artifact was produced.",
        "",
        "A separate installed-MLX activation probe supplies a plausible mechanism. "
        "The stock `mlx.nn.elu` produced finite forward outputs for large positive "
        "inputs 90 and 1000 but non-finite gradients. This is consistent with the "
        "critic-only gradient failure because actor and critic use the same hidden "
        "activation structure while receiving different objectives and gradient paths. "
        "It is **not proof of the exact failing minibatch**, which was not saved.",
        "",
        "The local model now uses a stable ELU expression:",
        "",
        "```python",
        "where(x >= 0, x, exp(min(x, 0)) - 1)",
        "```",
        "",
        "This keeps the same 512-256-128 hidden-layer topology, checkpoint tensor "
        "layout, and ELU forward behavior used by exported ONNX graphs. The PPO "
        "finite-value guards remain fail-closed: a non-finite loss, gradient, or model "
        "parameter still raises; bad updates are not silently skipped.",
        "",
    ]
    if probe:
        lines.extend(
            table(
                ["Probe field", "Captured value"],
                [
                    ["Inputs", json.dumps(probe.get("inputs"))],
                    ["Outputs", json.dumps(probe.get("outputs"))],
                    ["Gradients", json.dumps(probe.get("gradients"))],
                    ["Finite outputs", format_value(probe.get("finite_outputs"))],
                    ["Finite gradients", format_value(probe.get("finite_gradients"))],
                    ["Implementation", probe.get("implementation", "n/a")],
                ],
            )
        )
        lines.append("")
    lines.extend(
        table(
            ["Validation artifact", "Result", "Link"],
            [
                [
                    "Activation test before fix",
                    "3 failed because `stable_elu` did not yet exist",
                    markdown_link(before_path.name, before_path, output_dir)
                    if before_path.is_file()
                    else "missing",
                ],
                [
                    "Activation test after fix",
                    "3 passed: eager/compiled large-input gradient boundary and forward parity",
                    markdown_link(after_path.name, after_path, output_dir)
                    if after_path.is_file()
                    else "missing",
                ],
                [
                    "Installed MLX probe",
                    "finite forward, non-finite gradients at large positive inputs",
                    markdown_link(probe_path.name, probe_path, output_dir)
                    if probe_path.is_file()
                    else "missing",
                ],
            ],
        )
    )
    lines.extend(["", "Source hashes:", ""])
    rows = []
    for path in (probe_path, before_path, after_path, evidence_root / "activation_probe.py"):
        if path.is_file():
            rows.append(
                [
                    markdown_link(path.name, path, output_dir),
                    format_value(path.stat().st_size),
                    f"`{sha256(path)}`",
                ]
            )
    if rows:
        lines.extend(table(["Artifact", "Bytes", "SHA-256"], rows))
    else:
        lines.append("No activation-diagnosis artifacts are available.")
    lines.append("")
    return lines


def render_attempt(attempt: Attempt, output_dir: Path) -> list[str]:
    lines = [f"## {attempt.label}: `{attempt.identifier}`", ""]
    lines.append(f"**Scenario status:** **{attempt.status}**")
    lines.append("")
    if attempt.audit is None:
        failure = training_failure_summary(attempt.api)
        progress = progress_from_log(attempt.progress_path)
        if failure:
            artifacts = failure.get("artifacts", {})
            lines.extend(
                [
                    f"No `{attempt.audit_path.name}` exists because training failed before "
                    "a final checkpoint and deterministic audit were produced.",
                    "",
                ]
            )
            lines.extend(
                table(
                    ["Training failure evidence", "Value"],
                    [
                        ["Transitions reached", format_value(failure.get("steps"))],
                        ["Requested transitions", format_value(failure.get("total"))],
                        ["Exit code", format_value(failure.get("exit_code"))],
                        ["Started", failure.get("started", "n/a")],
                        ["Finished", failure.get("finished", "n/a")],
                        ["Final checkpoint present", format_value(artifacts.get("checkpoint"))],
                        ["Final ONNX present", format_value(artifacts.get("onnx"))],
                    ],
                )
            )
            lines.extend(["", f"Recorded failure: `{failure.get('last_log') or failure.get('message')}`", ""])
        elif progress:
            lines.extend(
                [
                    f"No `{attempt.audit_path.name}` exists because training is active. "
                    f"The retained log most recently recorded **{progress[0]:,}/{progress[1]:,}** "
                    "transitions. The report remains incomplete until training, evaluation, "
                    "audit, and video validation finish.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"No `{attempt.audit_path.name}` exists at `{repository_path(attempt.audit_dir)}`. "
                    "The recipe may be queued. No result is inferred from a recipe file, "
                    "checkpoint name, API state, or partial artifact.",
                    "",
                ]
            )
    if attempt.errors:
        lines.append("Input errors:")
        lines.extend(f"- `{error}`" for error in attempt.errors)
        lines.append("")

    lines.extend(["### Recipe and effective audit environment", ""])
    if attempt.recipe is not None:
        lines.append(
            "The exact requested recipe follows. These values describe training; "
            "they do not override the deterministic audit environment."
        )
        lines.extend(["", "```json", json.dumps(attempt.recipe, indent=2, sort_keys=True), "```", ""])
    else:
        lines.extend([f"Recipe file missing: `{attempt.recipe_path}`.", ""])

    effective = attempt.audit.get("effective_audit_environment") if attempt.audit else None
    if isinstance(effective, dict):
        lines.extend(table(["Audit setting", "Actual value"], sorted(
            (key, format_value(value)) for key, value in effective.items()
        )))
        lines.append("")
    else:
        lines.extend(
            [
                "No effective-environment object is available in the selected audit. "
                "The audit implementation fixes XML actuators, disables domain randomization, "
                "observation noise, action delay, and random yaw, and forces Swing audit starts "
                "to `0 degrees` and `0 rad/s`.",
                "",
            ]
        )

    if attempt.scenario == "stilts":
        recipe = attempt.recipe or {}
        lines.append(
            f"The requested morphology is **{format_value(recipe.get('stiltHeightCm'))} cm** "
            f"stilts, blend **{format_value(recipe.get('stiltBlend'))}**, mass "
            f"**{format_value(recipe.get('stiltMassKg'))} kg**, with a fixed forward command "
            f"of **{format_value(recipe.get('locomotionForwardCommand'))} m/s**."
        )
        lines.append("")
    elif attempt.scenario == "running":
        command = (attempt.recipe or {}).get("locomotionForwardCommand")
        lines.append(
            "The requested fixed forward command is "
            f"**{format_value(command)} m/s**. If this value is `n/a`, the environment's "
            "native command sequence is evaluated rather than a fixed command."
        )
        lines.append("")
    else:
        recipe = attempt.recipe or {}
        lines.append(
            "Training assistance is explicit in the recipe: initial angle "
            f"**{format_value(recipe.get('swingInitialAngleDeg'))} degrees**, initial rate "
            f"**{format_value(recipe.get('swingInitialRateRadS'))} rad/s**, planar actions "
            f"**{format_value(recipe.get('swingPlanarActions'))}**. The independent audit "
            "must start at rest (`0 degrees`, `0 rad/s`); assisted training starts never count "
            "as acceptance evidence."
        )
        lines.append("")

    lines.extend(["### Per-episode skill evidence", ""])
    headers, rows = episode_rows(attempt, "trained")
    if rows:
        lines.extend(table(headers, rows))
        physics_headers, physics_rows = episode_physics_rows(attempt, "trained")
        lines.extend(["", "#### Per-episode physics details", ""])
        lines.extend(table(physics_headers, physics_rows))
        lines.extend(["", "#### Per-episode failures", ""])
        lines.extend(episode_failure_lines(attempt, "trained"))
    else:
        lines.append("No trained-policy episode records are available.")
    lines.append("")

    lines.extend(["### Controls", ""])
    control_names = ["zero", "initial"]
    if attempt.audit and "random" in attempt.audit.get("controls", {}):
        control_names = ["zero", "random", "initial"]
    for control_name in control_names:
        headers, rows = episode_rows(attempt, control_name)
        role = "pretrained baseline" if control_name == "initial" and "random" in control_names else "control"
        lines.append(f"#### {control_name.title()} {role}")
        lines.append("")
        if rows:
            lines.extend(table(headers, rows))
            physics_headers, physics_rows = episode_physics_rows(attempt, control_name)
            lines.extend(["", "Physical details:", ""])
            lines.extend(table(physics_headers, physics_rows))
            lines.extend(["", "Failures:", ""])
            lines.extend(episode_failure_lines(attempt, control_name))
        else:
            lines.append("No records available.")
        lines.append("")

    metric_rows = metric_means_rows(attempt)
    lines.extend(["### Physics metric means", ""])
    if metric_rows:
        lines.extend(table(["Control", "Seed", "Metric", "Mean"], metric_rows))
    else:
        lines.append("No physical metric aggregates are available.")
    lines.append("")

    lines.extend(["### PPO transitions and losses", ""])
    if attempt.audit:
        lines.extend(table(["Training evidence", "Value"], training_rows(attempt)))
        lines.append("")
        history = history_rows(attempt)
        if history:
            lines.extend(
                table(
                    ["Transitions", "Steps", "Raw return", "Skill", "Failures"],
                    history,
                )
            )
            lines.append("")
    else:
        lines.append("No training audit summary is available.")
        lines.append("")

    lines.extend(["### Initialization and PPO contribution", ""])
    initialization = (attempt.audit or {}).get("initialization", {})
    if initialization.get("kind") == "checkpoint":
        loaded = initialization.get("loaded_metadata", {})
        lines.extend(table(["Initialization evidence", "Value"], [
            ["Source checkpoint", format_value(initialization.get("source_checkpoint"))],
            ["Source SHA-256", format_value(initialization.get("source_sha256"))],
            ["Teacher-assisted", format_value(loaded.get("teacher_assisted"))],
            ["Initial policy role", (attempt.audit or {}).get("initial_policy_role", "pretrained baseline")],
            ["Comparison seed", format_value((attempt.audit or {}).get("comparison_seed"))],
            ["PPO raw-return change vs initialization", format_value((attempt.audit or {}).get("ppo_raw_return_change"))],
        ]))
        lines.extend(["", "This run continues a pretrained policy. Its initial skill is not credited to PPO. "
                      "A nonzero parameter change proves optimization occurred, not that it improved the initializer. "
                      "Compare the baseline and final physical outcomes and raw returns before attributing any benefit to PPO.", ""])
    else:
        lines.extend(["No pretrained initialization is recorded for this attempt; the initial actor is the random-initialization control.", ""])

    lines.extend(["### ONNX, controls, and source identity", ""])
    if attempt.audit:
        onnx_contract = attempt.audit.get("onnx_contract", {})
        lines.extend(
            table(
                ["Check", "Value"],
                [
                    ["Policy SHA-256", f"`{attempt.audit.get('policy_sha256', 'n/a')}`"],
                    ["ONNX input", format_value(onnx_contract.get("input"))],
                    ["ONNX output", format_value(onnx_contract.get("output"))],
                    ["ONNX contract", "PASS" if onnx_contract.get("passed") is True else "FAIL"],
                    [
                        "Native/ONNX max absolute error",
                        format_value(attempt.audit.get("onnx_parity_max_abs_error")),
                    ],
                    ["Parameter L2 change", format_value(attempt.audit.get("parameter_l2_change"))],
                    [
                        "Negative controls failed as required",
                        format_value(attempt.audit.get("negative_controls_failed")),
                    ],
                    ["Audit scope", attempt.audit.get("scope", "n/a")],
                ],
            )
        )
        lines.append("")
    else:
        lines.append("No ONNX or control audit is available.")
        lines.append("")

    api_rows = api_summary(attempt.api)
    lines.extend(["### API workflow", ""])
    if api_rows:
        lines.extend(table(["Action", "Accepted", "Phase", "Exit code", "Finished"], api_rows))
        failure = attempt.api.get("failure") if attempt.api else None
        if isinstance(failure, dict) and failure.get("message"):
            lines.extend(["", f"Recorded API failure: **{failure['message']}**"])
    else:
        lines.append("No API workflow JSON is available for this attempt.")
    lines.append("")

    lines.extend(["### Video and graph evidence", ""])
    video = attempt.video
    if video:
        lines.extend(
            table(
                ["Video validation", "Value"],
                [
                    ["Validation status", "PASS" if video.get("passed") is True else "FAIL"],
                    ["API bytes equal local render", format_value(video.get("download_matches_local"))],
                    ["HTTP byte range", format_value(video.get("byte_range_passed"))],
                    ["Browser playback", json.dumps(video.get("playback"), sort_keys=True)],
                ],
            )
        )
        lines.append("")
        rows = video_rows(video, output_dir)
        if rows:
            lines.extend(
                table(
                    ["File", "Codec", "Dimensions", "FPS", "Frames", "Seconds", "SHA-256"],
                    rows,
                )
            )
            lines.append("")
    else:
        lines.extend(
            [
                "No `video-validation.json` is available. Images or MP4 files, if present, "
                "are linked below but are not described as browser/transport verified.",
                "",
            ]
        )
    links = artifact_lines(attempt, output_dir)
    lines.extend(links or ["- No audit artifacts are present."])
    lines.append("")

    image = attempt.audit_dir / "comparison_sheet.png"
    if image.is_file():
        lines.extend(
            [
                f"![{attempt.label} trained-policy and control comparison]({relative_link(image, output_dir)})",
                "",
            ]
        )
    for graph_name, alt in (
        ("physical-tracking.png", "Physical tracking"),
        ("reward-learning.png", "Reward and checkpoint learning"),
        ("ppo-losses.png", "PPO loss history"),
    ):
        graph = attempt.audit_dir / graph_name
        if graph.is_file():
            lines.extend([f"![{attempt.label}: {alt}]({relative_link(graph, output_dir)})", ""])

    lines.extend(["### Source hashes", ""])
    rows = source_rows(attempt, output_dir)
    if rows:
        lines.extend(table(["Source/artifact", "Bytes", "SHA-256"], rows))
    else:
        lines.append("No source files are available to hash.")
    lines.append("")

    lines.extend(["### Reproduction commands", ""])
    if attempt.recipe:
        scenario = attempt.recipe.get("experimentId", attempt.scenario)
        uses_tracked_recipe = tracked_recipe_matches(attempt)
        reproduction_recipe = (
            attempt.tracked_recipe_path if uses_tracked_recipe else attempt.recipe_path
        )
        recipe_rel = repository_path(reproduction_recipe)
        unique_run = f"{scenario}-reproduction-YYYYMMDD-HHMMSS"
        api_rel = f"rlx/artifacts/scenarios-e2e-20260907/{unique_run}-api.json"
        audit_rel = f"rlx/artifacts/scenarios-e2e-20260907/{unique_run}-audit"
        run_rel = f"rlx/runs/studio/{scenario}/{unique_run}"
        if scenario == "swing" and attempt.recipe.get("resumeFromCheckpoint"):
            lines.extend([
                "This recipe requires a teacher-assisted initializer. Reproduce the BC/DAgger "
                "stage first; it uses privileged state only to create training labels, never "
                "as an inference input to the exported actor. Preserve the initializer's own verdict.",
                "", "```bash",
                "rlx/.venv-microduck/bin/python rlx/scripts/bootstrap_swing_e2e.py "
                "--source docs/remaining-scenarios-e2e/recipes/swing-teacher.json "
                "--output /tmp/swing-bootstrap-reproduction",
                f"mkdir -p {run_rel}",
                f"test ! -e {run_rel}/swing.safetensors",
                f"cp /tmp/swing-bootstrap-reproduction/swing.safetensors "
                f"/tmp/swing-bootstrap-reproduction/swing.safetensors.json {run_rel}/",
                "```", "",
            ])
        elif scenario == "stilts" and attempt.recipe.get("resumeFromCheckpoint"):
            base_recipe_rel = (
                "docs/remaining-scenarios-e2e/recipes/stilts-base.json"
            )
            base_run = "stilts-base-reproduction-YYYYMMDD-HHMMSS"
            base_api_rel = (
                "rlx/artifacts/scenarios-e2e-20260907/"
                f"{base_run}-api.json"
            )
            base_run_rel = f"rlx/runs/studio/stilts/{base_run}"
            lines.extend([
                "This recipe is a continuation of the Stilt v2 base policy. First train the "
                "tracked base recipe under its own unique run name. Preserve that base run and "
                "its API verdict as separate evidence; the continuation does not retroactively "
                "change it.",
                "",
                "```bash",
                (
                    "node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute "
                    "--base-url http://127.0.0.1:63317 --experiment stilts "
                    f"--recipe-json {base_recipe_rel} --run {base_run} "
                    f"--report {base_api_rel} --timeout-seconds 7200"
                ),
                f"mkdir -p {run_rel}",
                f"test ! -e {run_rel}/stilts.safetensors",
                f"test ! -e {run_rel}/stilts.safetensors.json",
                f"cp {base_run_rel}/stilts.safetensors "
                f"{base_run_rel}/stilts.safetensors.json {run_rel}/",
                "```",
                "",
                "The copy seeds the target run with both files required by the Studio API. "
                "The continuation runner below then resumes from that target-local checkpoint.",
                "",
            ])
        lines.extend(
            [
                "Run from the repository root after starting the Studio server at "
                "`http://127.0.0.1:63317`:",
                "",
                "```bash",
                (
                    "node duck-viewer/scripts/rlx-dance-api-e2e.mjs --execute "
                    f"--base-url http://127.0.0.1:63317 --experiment {scenario} "
                    f"--recipe-json {recipe_rel} --run {unique_run} "
                    f"--report {api_rel} --timeout-seconds 7200"
                ),
                (
                    "rlx/.venv-microduck/bin/python rlx/scripts/audit_scenarios.py "
                    f"--recipe-json {recipe_rel} --run {run_rel} --output {audit_rel} "
                    "--seeds 101 102 103 104 105 --render"
                ),
                (
                    "node duck-viewer/scripts/verify-rlx-video.mjs "
                    f"--recipe-json {recipe_rel} --run {run_rel} --output {audit_rel} "
                    "--base-url http://127.0.0.1:63317 "
                    "--playwright-package \"${PLAYWRIGHT_PACKAGE:-/Users/ghu/community/package.json}\""
                ),
                "```",
                "",
                "The audit command exits nonzero when the strict skill gate fails. "
                "That nonzero exit is expected evidence for a failed attempt, not permission "
                "to rewrite the result as successful.",
                "",
            ]
        )
        if not uses_tracked_recipe:
            lines.extend(
                [
                    f"This historical command uses the retained artifact recipe "
                    f"`{recipe_rel}` because it does not match the tracked selected "
                    f"`recipes/{attempt.scenario}.json`. It is not a fresh-checkout "
                    "reproduction unless that historical artifact recipe is separately retained.",
                    "",
                ]
            )
    else:
        lines.append("Unavailable until the recipe JSON exists.")
        lines.append("")
    return lines


def render_report(
    evidence_root: Path, attempts: list[Attempt], output: Path
) -> str:
    output_dir = output.parent
    overall_pass = all(attempt.evidence_complete for attempt in attempts)
    overall = "PASS" if overall_pass else "FAIL / INCOMPLETE"
    discovered = discovered_attempts(evidence_root)
    lines = [
        "# Microduck Remaining Scenarios: End-to-End Evidence",
        "",
        "**Workflow dates:** September 7-8, 2026  ",
        f"**Evidence root:** `{repository_path(evidence_root)}`  ",
        f"**Report result:** **{overall}**",
        "",
        "## 1. Executive status",
        "",
        "This report is generated from saved JSON and artifacts. It does not infer a "
        "successful skill from training completion, reward magnitude, finite output, "
        "an exported ONNX file, or a playable video. A selected scenario is complete "
        "only when its independent audit passes and its video validation passes.",
        "",
    ]
    rows = []
    for attempt in attempts:
        trained = controls(attempt.audit, "trained")
        passed = sum(1 for item in trained if item.get("passed") is True)
        rows.append(
            [
                attempt.label,
                f"`{attempt.identifier}`",
                attempt.status,
                f"{passed}/{len(trained)}" if trained else "not available",
                (
                    "PASS"
                    if attempt.video and attempt.video.get("passed") is True
                    else "missing/fail"
                ),
                "complete" if attempt.evidence_complete else "incomplete",
            ]
        )
    lines.extend(
        table(
            ["Scenario", "Selected attempt", "Skill audit", "Held-out controls", "Video", "E2E"],
            rows,
        )
    )
    lines.extend(
        [
            "",
            "> The aggregate result remains failed/incomplete if any selected scenario is "
            "missing or failed. This snapshot must not be described as \"all verified.\"",
            "",
            "## 2. Evidence boundaries",
            "",
            "- Training and deterministic audit are separate. A recipe may use assisted "
            "Swing resets, but evaluation and audit start at rest.",
            "- The effective audit environment uses the XML actuator model only. It is "
            "nominal local MuJoCo evidence, not BAM, mjlab GPU, hardware, or sim-to-real certification.",
            "- Stilt morphology is part of the experiment identity. The selected recipes "
            "request 2 cm morphology; reports must not substitute the API default of 10 cm.",
            "- Fixed locomotion commands are taken from the selected recipe and reported as "
            "actual numeric values. They are not reconstructed from reward or displacement.",
            "- The built-in recipe metrics wrapper captures the active locomotion command "
            "before `env.step()` and attaches physical metrics after the transition. This "
            "fixes the command/metric boundary when commands resample on the step boundary.",
            "- Locomotion evaluator version 2 anchors intended heading at each twist-command "
            "change and advances it only by commanded yaw. Signed displacement therefore "
            "measures net progress along the intended command path; a closed circle cannot "
            "pass by accumulating path length while returning to its start.",
            "- API outcome handling accepts an explicitly failed skill evaluation long enough "
            "to collect render/export evidence. A later successful render or export does not "
            "change the skill verdict.",
            "- Outcome inference is deliberately limited: without `audit.json`, no skill verdict "
            "exists. API JSON and logs may establish TRAIN FAILED, TRAINING ACTIVE, TRAINED / "
            "UNAUDITED, or NOT RUN, but none is a pass. No pass is inferred from filenames, "
            "checkpoint existence, finite telemetry, HTTP success, screenshots, or videos.",
            "",
            "## 3. Historical attempts discovered",
            "",
        ]
    )
    if discovered:
        lines.extend(
            table(
                ["Scenario", "Attempt", "Evidence status", "Evidence"],
                [
                    [
                        scenario,
                        f"`{identifier}`",
                        status,
                        markdown_link(path.name, path, output_dir)
                        if path and path.is_file()
                        else "missing",
                    ]
                    for scenario, identifier, status, path in discovered
                ],
            )
        )
    else:
        lines.append("No `*-audit` directories were discovered.")
    lines.extend(["", "Only directories containing `audit.json` carry a skill verdict.", ""])

    lines.extend(
        [
            "## 4. Repository-wide validation context",
            "",
            "These are saved workflow logs, not newly rerun suites. They remain relevant "
            "limitations on the evidence package:",
            "",
        ]
    )
    lines.extend(table(["Check", "Saved result", "Log"], log_status(evidence_root)))
    lines.extend(
        [
            "",
            "- The full RLX collection is blocked because the optional `mjlab` package is "
            "not installed in the captured environment. Native non-mjlab RLX tests passed separately.",
            "- The broader `microduck_local` run has 12 existing failures: 11 BAM/step "
            "pre-optimization golden mismatches and one symmetry drift assertion. These "
            "failures are not hidden or reclassified by this report.",
            "",
        ]
    )

    section = 5
    for attempt in attempts:
        rendered = render_attempt(attempt, output_dir)
        rendered[0] = f"## {section}. {attempt.label}: `{attempt.identifier}`"
        lines.extend(rendered)
        section += 1

    selected_ids = {attempt.identifier for attempt in attempts}
    historical_attempts: list[Attempt] = []
    for scenario, label in SCENARIOS:
        identifiers = {
            identifier
            for found_scenario, identifier, _, _ in discovered
            if found_scenario == scenario and identifier not in selected_ids
        }
        historical_attempts.extend(
            load_attempt(evidence_root, scenario, label, identifier)
            for identifier in sorted(identifiers)
            if (evidence_root / f"{identifier}-audit" / "audit.json").is_file()
            or (evidence_root / f"{identifier}-api.json").is_file()
        )
    for attempt in historical_attempts:
        rendered = render_attempt(attempt, output_dir)
        rendered[0] = f"## {section}. Historical {attempt.label}: `{attempt.identifier}`"
        lines.extend(rendered)
        section += 1

    lines.extend(render_activation_diagnosis(evidence_root, output_dir, section))
    section += 1

    lines.extend(
        [
            f"## {section}. Tracked reproduction recipes",
            "",
            "These copies are tracked with the report so a fresh checkout does not depend "
            "on the gitignored evidence directory:",
            "",
        ]
    )
    for scenario, label in SCENARIOS:
        path = Path(__file__).with_name("recipes") / f"{scenario}.json"
        lines.append(f"- {markdown_link(label, path, output_dir)}")
        if scenario == "stilts":
            base_path = Path(__file__).with_name("recipes") / "stilts-base.json"
            lines.append(
                f"- {markdown_link('Stilt Walking base', base_path, output_dir)}"
            )
    lines.append("")
    section += 1

    lines.extend(
        [
            f"## {section}. Final conclusion",
            "",
            f"**Report result: {overall}.**",
            "",
        ]
    )
    if overall_pass:
        lines.append(
            "All three selected scenarios have passing independent audits and passing video "
            "validation. This remains simulation-only evidence under the stated XML conditions."
        )
    else:
        failing = ", ".join(
            f"{attempt.label} ({attempt.status.lower()})"
            for attempt in attempts
            if not attempt.evidence_complete
        )
        lines.append(
            f"The evidence package is not fully verified. Outstanding selected scenarios: {failing}. "
            "Regenerate this report after their audit and video-validation artifacts exist; "
            "the generator will update the verdict from the files rather than from expectations."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    evidence_root = (
        args.evidence_root.resolve()
        if args.evidence_root.is_absolute()
        else (REPO_ROOT / args.evidence_root).resolve()
    )
    output = args.output.resolve()
    attempts = [
        load_attempt(evidence_root, scenario, label, getattr(args, scenario))
        for scenario, label in SCENARIOS
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(evidence_root, attempts, output)
    output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "report_status": (
                    "PASS" if all(attempt.evidence_complete for attempt in attempts)
                    else "FAIL / INCOMPLETE"
                ),
                "attempts": {
                    attempt.scenario: {
                        "identifier": attempt.identifier,
                        "status": attempt.status,
                        "video_validated": bool(
                            attempt.video and attempt.video.get("passed") is True
                        ),
                    }
                    for attempt in attempts
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
