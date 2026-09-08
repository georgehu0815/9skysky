from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("generate-report.py")
SPEC = importlib.util.spec_from_file_location("remaining_report", MODULE_PATH)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


class ReportGeneratorTests(unittest.TestCase):
    def passing_evidence(self, root):
        audit_dir = root / "running-v4-audit"
        audit_dir.mkdir()
        policy = root / "running.onnx"
        policy.write_bytes(b"policy")
        policy_hash = REPORT.sha256(policy)
        recipe = {"experimentId": "running", "runName": "run", "locomotionForwardCommand": 0.75}
        audit = {"passed": True, "run_name": "run", "policy_sha256": policy_hash, "recipe": recipe}
        (root / "running-v4.json").write_text(json.dumps(recipe))
        (audit_dir / "audit.json").write_text(json.dumps(audit))
        operations = []
        for action in ("train", "eval", "render", "export"):
            operations.append({"action": action, "state": {
                "phase": "succeeded", "experimentId": "running", "runName": "run",
                "evaluation": {"passed": True, "skill_status": "passed"},
                "result": {"source_type": "policy", "source_sha256": policy_hash, "source": str(policy)},
            }})
        api = {"failure": None, "normalizedRecipe": recipe, "operations": operations}
        (root / "running-v4-api.json").write_text(json.dumps(api))
        videos = []
        for name in ("api-video.mp4", "comparison.mp4"):
            path = audit_dir / name
            path.write_bytes(name.encode())
            videos.append({"sha256": REPORT.sha256(path)})
        video = {"passed": True, "experiment": "running", "run": "run", "policy_sha256": policy_hash,
                 "audit_sha256": REPORT.sha256(audit_dir / "audit.json"), "videos": videos}
        (audit_dir / "video-validation.json").write_text(json.dumps(video))
        return REPORT.load_attempt(root, "running", "Running", "running-v4")

    def test_complete_evidence_requires_successful_matching_api_and_current_files(self):
        mutations = (
            lambda attempt: setattr(attempt, "api", None),
            lambda attempt: attempt.api.update(failure={"message": "failed"}),
            lambda attempt: attempt.api["operations"].pop(),
            lambda attempt: attempt.api["operations"][0]["state"].update(phase="failed"),
            lambda attempt: attempt.api["operations"][1]["state"]["evaluation"].update(skill_status="not_assessed"),
            lambda attempt: attempt.api["operations"][2]["state"]["result"].update(source_sha256="wrong"),
            lambda attempt: attempt.api["normalizedRecipe"].update(runName="other"),
            lambda attempt: attempt.api["normalizedRecipe"].update(locomotionForwardCommand=0.1),
            lambda attempt: attempt.video.update(policy_sha256="wrong"),
            lambda attempt: attempt.video.update(run="other"),
            lambda attempt: attempt.audit_path.write_text("{}"),
            lambda attempt: (attempt.audit_dir / "api-video.mp4").write_bytes(b"changed"),
            lambda attempt: Path(attempt.api["operations"][1]["state"]["result"]["source"]).write_bytes(b"changed"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                attempt = self.passing_evidence(Path(directory))
                self.assertTrue(attempt.evidence_complete)
                mutate(attempt)
                self.assertFalse(attempt.evidence_complete)

    def test_default_attempts_select_current_continuations(self) -> None:
        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = REPORT.parse_args()

        self.assertEqual(args.running, "running-v4")
        self.assertEqual(args.stilts, "stilts-v3")
        self.assertEqual(args.swing, "swing-v3")

    def test_missing_selected_attempts_make_report_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = [
                REPORT.load_attempt(root, scenario, label, identifier)
                for (scenario, label), identifier in zip(
                    REPORT.SCENARIOS, ("running-v2", "stilts-v2", "swing-v1")
                )
            ]
            text = REPORT.render_report(root, attempts, root / "REPORT.md")
            self.assertIn("**Report result:** **FAIL / INCOMPLETE**", text)
            self.assertEqual(text.count("**Scenario status:** **NOT RUN**"), 3)
            self.assertNotIn('**Report result:** **PASS**', text)

    def test_failed_audit_is_not_promoted_by_valid_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "running-v1-audit").mkdir()
            (root / "running-v1.json").write_text(
                json.dumps(
                    {
                        "experimentId": "running",
                        "runName": "run",
                        "locomotionForwardCommand": 0.6,
                    }
                )
            )
            (root / "running-v1-audit" / "audit.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "recipe": {"experimentId": "running"},
                        "controls": {"trained": [{"seed": 101, "passed": False}]},
                    }
                )
            )
            (root / "running-v1-audit" / "video-validation.json").write_text(
                json.dumps({"passed": True})
            )
            attempt = REPORT.load_attempt(root, "running", "Running", "running-v1")
            self.assertEqual(attempt.status, "FAIL")
            self.assertFalse(attempt.evidence_complete)

    def test_failed_training_without_audit_is_reported_as_train_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stilts-v1-api.json").write_text(
                json.dumps(
                    {
                        "operations": [
                            {
                                "action": "train",
                                "state": {
                                    "phase": "failed",
                                    "trainingSteps": 1980416,
                                    "trainingTotal": 4001792,
                                    "exitCode": 1,
                                    "logs": ["RuntimeError: non-finite critic gradients"],
                                    "artifacts": {"checkpoint": False, "onnx": False},
                                },
                            }
                        ]
                    }
                )
            )
            attempt = REPORT.load_attempt(root, "stilts", "Stilt Walking", "stilts-v1")
            self.assertEqual(attempt.status, "TRAIN FAILED")
            text = "\n".join(REPORT.render_attempt(attempt, root))
            self.assertIn("1,980,416", text)
            self.assertIn("Final checkpoint present", text)

    def test_relative_links_are_computed_from_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "docs" / "remaining"
            artifact = root / "rlx" / "artifacts" / "audit.json"
            report_dir.mkdir(parents=True)
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}")
            self.assertEqual(
                REPORT.relative_link(artifact, report_dir),
                "../../rlx/artifacts/audit.json",
            )

    def test_stilt_recipe_reports_actual_two_centimeter_morphology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {
                "experimentId": "stilts",
                "runName": "stilts-v2",
                "stiltHeightCm": 2,
                "stiltBlend": 0,
                "stiltMassKg": 0.014,
                "locomotionForwardCommand": 0.25,
            }
            (root / "stilts-v2.json").write_text(json.dumps(recipe))
            attempt = REPORT.load_attempt(root, "stilts", "Stilt Walking", "stilts-v2")
            text = "\n".join(REPORT.render_attempt(attempt, root))
            self.assertIn("**2 cm**", text)
            self.assertIn("**0.2500 m/s**", text)
            self.assertNotIn("10 cm", text)

    def test_stilt_v3_reproduction_trains_and_copies_tracked_base_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {
                "experimentId": "stilts",
                "runName": "stilts-e2e-20260907-v3",
                "resumeFromCheckpoint": True,
                "stiltHeightCm": 2,
                "stiltBlend": 0,
                "stiltMassKg": 0.014,
                "locomotionForwardCommand": 0.25,
            }
            (root / "stilts-v3.json").write_text(json.dumps(recipe))
            attempt = REPORT.load_attempt(root, "stilts", "Stilt Walking", "stilts-v3")
            text = "\n".join(REPORT.render_attempt(attempt, root))

            base_command = (
                "--recipe-json docs/remaining-scenarios-e2e/recipes/stilts-base.json "
                "--run stilts-base-reproduction-YYYYMMDD-HHMMSS"
            )
            copy_command = (
                "cp rlx/runs/studio/stilts/"
                "stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors "
                "rlx/runs/studio/stilts/"
                "stilts-base-reproduction-YYYYMMDD-HHMMSS/stilts.safetensors.json "
                "rlx/runs/studio/stilts/stilts-reproduction-YYYYMMDD-HHMMSS/"
            )
            resume_command = (
                "--recipe-json "
                f"{REPORT.repository_path(root / 'stilts-v3.json')} "
                "--run stilts-reproduction-YYYYMMDD-HHMMSS"
            )

            self.assertIn(base_command, text)
            self.assertIn(copy_command, text)
            self.assertIn(resume_command, text)
            self.assertLess(text.index(base_command), text.index(copy_command))
            self.assertLess(text.index(copy_command), text.index(resume_command))
            self.assertIn("test ! -e", text)
            self.assertIn("stilts.safetensors.json", text)

    def test_tracked_stilt_base_recipe_matches_retained_v2_hash(self) -> None:
        tracked = MODULE_PATH.with_name("recipes") / "stilts-base.json"

        self.assertEqual(
            hashlib.sha256(tracked.read_bytes()).hexdigest(),
            "cab5e12bed6e85c866d891ec09a563f7d7cfc19a01d337a64b9354bc02a68e09",
        )

    def test_tracked_running_base_recipe_matches_retained_v3_hash(self) -> None:
        tracked = MODULE_PATH.with_name("recipes") / "running-base.json"
        self.assertEqual(
            hashlib.sha256(tracked.read_bytes()).hexdigest(),
            "66a2be10f548531cee3bb487393cea785d4a16834d38c7b177567e27b4c40f70",
        )
        recipe = json.loads(tracked.read_text())
        self.assertFalse(recipe["resumeFromCheckpoint"])
        self.assertEqual(recipe["totalTimesteps"], 6000640)
        self.assertEqual(recipe["rewardWeights"]["track_turn"], 8)
        self.assertNotIn("yaw_tracking", recipe["rewardWeights"])

    def test_running_reproduction_preserves_optional_yaw_weights_and_order(self) -> None:
        for yaw_weights in ({}, {"yaw_tracking": 8}):
            with self.subTest(yaw_weights=yaw_weights), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe = {
                    "experimentId": "running",
                    "runName": "running-e2e-20260907-v4",
                    "resumeFromCheckpoint": True,
                    "rewardWeights": {"track_turn": 2, **yaw_weights},
                }
                (root / "running-v4.json").write_text(json.dumps(recipe))
                attempt = REPORT.load_attempt(root, "running", "Running", "running-v4")
                text = "\n".join(REPORT.render_attempt(attempt, root))
                base_command = (
                    "--recipe-json docs/remaining-scenarios-e2e/recipes/running-base.json "
                    "--run running-base-reproduction-YYYYMMDD-HHMMSS"
                )
                copy_command = (
                    "cp rlx/runs/studio/running/"
                    "running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors "
                    "rlx/runs/studio/running/"
                    "running-base-reproduction-YYYYMMDD-HHMMSS/running.safetensors.json "
                    "rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS/"
                )
                resume_command = (
                    f"--recipe-json {REPORT.repository_path(root / 'running-v4.json')} "
                    "--run running-reproduction-YYYYMMDD-HHMMSS"
                )
                self.assertLess(text.index(base_command), text.index(copy_command))
                self.assertLess(text.index(copy_command), text.index(resume_command))
                self.assertIn('"track_turn": 2', text)
                self.assertEqual('"yaw_tracking": 8' in text, bool(yaw_weights))
                self.assertIn("completed training but failed its skill gate", text)
                self.assertNotIn("|| true", text)

    def test_stilts_uses_the_same_tested_fail_closed_base_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocks = {}
            for scenario in ("running", "stilts"):
                identifier = f"{scenario}-v3"
                (root / f"{identifier}.json").write_text(json.dumps({
                    "experimentId": scenario, "resumeFromCheckpoint": True,
                }))
                attempt = REPORT.load_attempt(root, scenario, scenario, identifier)
                text = "\n".join(REPORT.render_attempt(attempt, root))
                blocks[scenario] = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)[0]
            self.assertEqual(
                blocks["running"].replace("running", "stilts").replace("Running", "Stilts"),
                blocks["stilts"],
            )

    def test_running_base_copy_guard_executes_fail_closed(self) -> None:
        runner_stub = '''node() {
python3 - "$@" <<'FAKE_RUNNER'
import json
import os
import sys
from pathlib import Path
report_path = Path(sys.argv[sys.argv.index('--report') + 1])
run_name = sys.argv[sys.argv.index('--run') + 1]
run_dir = Path('rlx/runs/studio/running') / run_name
run_dir.mkdir(parents=True)
for name in ('running.safetensors', 'running.safetensors.json'):
    if name != os.environ.get('MISSING_FILE'):
        (run_dir / name).write_text('retained base checkpoint')
if os.environ.get('WRITE_REPORT') == 'yes':
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(os.environ['BASE_REPORT'])
sys.exit(int(os.environ['RUNNER_EXIT']))
FAKE_RUNNER
}
'''
        expected_failure = (
            "Full Running skill evaluation failed; render and export evidence were collected."
        )
        cases = (
            "success", "skill_failed", "train_failed", "train_missing", "other_error",
            "missing_checkpoint", "missing_sidecar", "missing_report", "invalid_report",
            "wrong_run", "export_failed", "contradictory_status",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "running-v4.json").write_text(json.dumps({
                    "experimentId": "running", "resumeFromCheckpoint": True,
                }))
                attempt = REPORT.load_attempt(root, "running", "Running", "running-v4")
                text = "\n".join(REPORT.render_attempt(attempt, root))
                block = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)[0]
                operations = [
                    {"action": action, "state": {"phase": "succeeded"}, "error": None}
                    for action in ("train", "eval", "render", "export")
                ]
                operations[1]["state"]["evaluation"] = {"skill_status": "failed"}
                report = {
                    "requestedRecipe": {
                        "experimentId": "running",
                        "runName": "running-base-reproduction-YYYYMMDD-HHMMSS",
                    },
                    "operations": operations,
                    "failure": {"message": expected_failure},
                }
                if case == "success":
                    report["failure"] = None
                    operations[1]["state"]["evaluation"]["skill_status"] = "passed"
                elif case == "train_failed":
                    operations[0]["state"]["phase"] = "failed"
                elif case == "train_missing":
                    operations.pop(0)
                elif case == "other_error":
                    report["failure"]["message"] = "render timed out"
                elif case == "wrong_run":
                    report["requestedRecipe"]["runName"] = "unrelated-run"
                elif case == "export_failed":
                    operations[3]["state"]["phase"] = "failed"
                missing_file = {
                    "missing_checkpoint": "running.safetensors",
                    "missing_sidecar": "running.safetensors.json",
                }.get(case, "")
                result = subprocess.run(
                    ["bash", "-c", runner_stub + block], cwd=root,
                    env={
                        **os.environ,
                        "BASE_REPORT": "invalid json" if case == "invalid_report" else json.dumps(report),
                        "RUNNER_EXIT": "0" if case in ("success", "contradictory_status") else "1",
                        "WRITE_REPORT": "no" if case == "missing_report" else "yes",
                        "MISSING_FILE": missing_file,
                    },
                    text=True, capture_output=True, timeout=15,
                )
                target = root / "rlx/runs/studio/running/running-reproduction-YYYYMMDD-HHMMSS"
                if case in ("success", "skill_failed"):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for name in ("running.safetensors", "running.safetensors.json"):
                        self.assertEqual((target / name).read_text(), "retained base checkpoint")
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertFalse(target.exists(), result.stderr)

    def test_swing_v3_reboots_teacher_with_shorter_budget_and_new_audit_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swing-v3.json").write_text(json.dumps({
                "experimentId": "swing", "resumeFromCheckpoint": True,
                "totalTimesteps": 524288,
            }))
            attempt = REPORT.load_attempt(root, "swing", "Swing", "swing-v3")
            text = "\n".join(REPORT.render_attempt(attempt, root))
            self.assertIn('"totalTimesteps": 524288', text)
            self.assertIn("not a continuation of the failed v2 final policy", text)
            self.assertLess(text.index("bootstrap_swing_e2e.py"), text.index("rlx-dance-api-e2e.mjs"))
            self.assertIn("swing.safetensors.json", text)
            self.assertIn("--seeds 501 502 503 504 505 --render", text)
            self.assertNotIn("--seeds 101", text)

    def test_optional_interpretation_precedes_conclusion_with_contiguous_numbering(self) -> None:
        for include in (False, True):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                interpretation = root / "INTERPRETATION.md"
                content = "### Learning\n\nAuthored interpretation.\n\n### Limitations\n\nStill incomplete.\n"
                if include:
                    interpretation.write_text(content, encoding="utf-8")
                attempt = REPORT.load_attempt(root, "running", "Running", "running-v4")
                with patch.object(REPORT, "INTERPRETATION_PATH", interpretation):
                    text = REPORT.render_report(root, [attempt], root / "REPORT.md")
                headings = re.findall(r"^## (\d+)\. (.+)$", text, re.MULTILINE)
                self.assertEqual([int(number) for number, _ in headings], list(range(1, len(headings) + 1)))
                self.assertEqual(headings[-1][1], "Final conclusion")
                if include:
                    self.assertEqual(headings[-2][1], "Interpretation")
                    self.assertEqual(text.count(content.strip()), 1)
                else:
                    self.assertEqual(headings[-2][1], "Tracked reproduction recipes")
                    self.assertNotIn(". Interpretation", text)
                self.assertIn("**Report result:** **FAIL / INCOMPLETE**", text)
                self.assertIn("[Running base]", text)
                self.assertFalse((root / "REPORT.md").exists())

    def test_swing_episode_uses_evaluator_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_dir = root / "swing-v1-audit"
            audit_dir.mkdir()
            (audit_dir / "audit.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "controls": {
                            "trained": [
                                {
                                    "seed": 101,
                                    "episodes": [
                                        {
                                            "episode_index": 0,
                                            "steps": 1200,
                                            "bidirectional_span_deg": 151.25,
                                            "negative_peak_deg": -76.0,
                                            "positive_peak_deg": 78.0,
                                            "both_strings_tensioned_fraction": 1.0,
                                            "valid_geometry_fraction": 1.0,
                                            "max_abs_lateral_m": 0.003,
                                            "max_alignment": 0.001,
                                            "passed": True,
                                            "failures": [],
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                )
            )
            attempt = REPORT.load_attempt(root, "swing", "Swing", "swing-v1")
            _, rows = REPORT.episode_rows(attempt, "trained")
            self.assertIn("151.2500", rows[0])
            self.assertIn("-76.0000", rows[0])
            self.assertIn("1.0000", rows[0])
            _, physics = REPORT.episode_physics_rows(attempt, "trained")
            self.assertIn("0.0030", physics[0])
            self.assertIn("0.0010", physics[0])

    def test_historical_attempt_does_not_claim_selected_tracked_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "docs" / "remaining-scenarios-e2e"
            recipes_dir = report_dir / "recipes"
            recipes_dir.mkdir(parents=True)
            artifact_recipe = {
                "experimentId": "running",
                "runName": "running-v1",
                "locomotionForwardCommand": 0.4,
            }
            tracked_recipe = {
                "experimentId": "running",
                "runName": "running-v2",
                "locomotionForwardCommand": 0.6,
            }
            (root / "running-v1.json").write_text(json.dumps(artifact_recipe))
            tracked_path = recipes_dir / "running.json"
            tracked_path.write_text(json.dumps(tracked_recipe))
            attempt = REPORT.load_attempt(root, "running", "Running", "running-v1")
            attempt.tracked_recipe_path = tracked_path

            source_text = "\n".join(
                " | ".join(row) for row in REPORT.source_rows(attempt, report_dir)
            )
            rendered = "\n".join(REPORT.render_attempt(attempt, report_dir))

            self.assertNotIn("running.json", source_text)
            self.assertIn("running-v1.json", source_text)
            self.assertIn("not a fresh-checkout reproduction", rendered)

    def test_missing_audit_keeps_training_state_without_skill_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "running-v2-api.log").write_text(
                "train: running 4096 / 6000640\n"
            )
            attempt = REPORT.load_attempt(root, "running", "Running", "running-v2")
            report = REPORT.render_report(root, [attempt], root / "REPORT.md")

            self.assertEqual(attempt.status, "TRAINING ACTIVE")
            self.assertIn("without `audit.json`, no skill verdict exists", report)
            self.assertIn("**Report result:** **FAIL / INCOMPLETE**", report)


if __name__ == "__main__":
    unittest.main()
