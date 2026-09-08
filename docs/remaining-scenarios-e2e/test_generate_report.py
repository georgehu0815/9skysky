from __future__ import annotations

import hashlib
import importlib.util
import json
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
    def test_default_attempts_select_current_continuations(self) -> None:
        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = REPORT.parse_args()

        self.assertEqual(args.running, "running-v3")
        self.assertEqual(args.stilts, "stilts-v3")
        self.assertEqual(args.swing, "swing-v2")

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
