import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parent


def load_script(name):
    specification = importlib.util.spec_from_file_location(
        name.replace("-", "_"), DIRECTORY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class BookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_script("build-book")
        cls.validator = load_script("validate-book")
        cls.lab = load_script("learning_lab")

    def test_swing_budget_counts_transitions_not_gradient_reuse(self):
        budget = self.lab.rollout_budget(524288, 16, 256, 4, 2)
        self.assertEqual(budget["iterations"], 128)
        self.assertEqual(budget["optimizer_steps"], 1024)
        self.assertEqual(budget["actual_transitions"], 524288)
        self.assertEqual(budget["sample_presentations"], 1048576)

    def test_dance_full_rollout_overshoot(self):
        budget = self.lab.rollout_budget(4000000, 16, 128, 4, 4)
        self.assertEqual(budget["actual_transitions"], 4001792)
        self.assertEqual(budget["overshoot"], 1792)
        self.assertEqual(budget["optimizer_steps"], 31264)

    def test_invalid_budget_is_rejected(self):
        for values in ((0, 16, 128, 4, 4), (1, 1, 7, 4, 4), (True, 16, 128, 4, 4)):
            with self.assertRaises(ValueError):
                self.lab.rollout_budget(*values)

    def test_clipping_respects_advantage_sign(self):
        self.assertAlmostEqual(self.lab.clipped_surrogate(2.0, 1.3), 2.4)
        self.assertAlmostEqual(self.lab.clipped_surrogate(-2.0, 1.3), -2.6)
        self.assertAlmostEqual(self.lab.clipped_surrogate(-2.0, 0.7), -1.6)

    def test_terminal_has_precedence_over_timeout(self):
        result = self.lab.scalar_gae([1.0], [2.0], [True], [True], [5.0], 99.0)
        self.assertEqual(result, [-1.0])

    def test_timeout_bootstraps_terminal_observation_not_reset(self):
        result = self.lab.scalar_gae([1.0], [2.0], [False], [True], [5.0], 99.0)
        self.assertAlmostEqual(result[0], 3.95)

    def test_gae_stops_advantage_propagation_across_reset(self):
        result = self.lab.scalar_gae(
            [1.0, 100.0],
            [2.0, 0.0],
            [False, True],
            [True, False],
            [5.0, 0.0],
            0.0,
        )
        self.assertAlmostEqual(result[0], 3.95)
        self.assertEqual(result[1], 100.0)

    def test_three_step_timeout_gae_matches_worked_example(self):
        example = self.lab.gae_walkthrough()
        for actual, expected in zip(
            example["timeout"]["advantages"], (0.58683558775, -0.1075645, -0.209)
        ):
            self.assertAlmostEqual(actual, expected, places=10)
        for actual, expected in zip(
            example["timeout"]["value_targets"], (1.58683558775, 1.0924355, 0.891)
        ):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_three_step_true_terminal_differs_from_timeout(self):
        example = self.lab.gae_walkthrough()
        for actual, expected in zip(
            example["terminal"]["advantages"], (-0.201289775, -0.94555, -1.1)
        ):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_lambda_zero_is_one_step_td(self):
        example = self.lab.gae_walkthrough()
        for actual, expected in zip(
            example["lambda_comparison"]["0.0"], example["timeout"]["td_residuals"]
        ):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_lambda_one_retains_nonterminal_boundary_bootstrap(self):
        example = self.lab.gae_walkthrough()
        bootstrapped_return = 0.5 + 0.99 * 0.2 + 0.99**3 * 0.9
        self.assertAlmostEqual(
            example["lambda_comparison"]["1.0"][0], bootstrapped_return - 1.0
        )
        terminal = self.lab.scalar_gae(
            [0.5, 0.2, 0.0],
            [1.0, 1.2, 1.1],
            [False, False, True],
            [False, False, False],
            [0.0, 0.0, 0.9],
            99.0,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(terminal[0], 0.5 + 0.99 * 0.2 - 1.0)

    def test_unfinished_rollout_uses_last_value(self):
        advantages = self.lab.scalar_gae(
            [0.5, 0.2, 0.0],
            [1.0, 1.2, 1.1],
            [False] * 3,
            [False] * 3,
            [0.0] * 3,
            0.9,
        )
        for actual, expected in zip(
            advantages, self.lab.gae_walkthrough()["timeout"]["advantages"]
        ):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_dropped_negative_timeout_value_biases_upward(self):
        correct = self.lab.scalar_gae([0.0], [0.0], [False], [True], [-0.9], 99.0)
        mistaken = self.lab.scalar_gae([0.0], [0.0], [True], [False], [-0.9], 99.0)
        self.assertAlmostEqual(mistaken[0] - correct[0], 0.891)

    def test_saved_audit_budgets_match_all_four_cases(self):
        summaries = self.lab.evidence_summary()
        self.assertEqual(set(summaries), {"dance", "running", "stilts", "swing"})
        for summary in summaries.values():
            self.assertEqual(summary["saved_passes"], 5)
            self.assertEqual(summary["saved_trials"], 5)

    def assert_existing_directory_blocks_command(self, chapter, command, existing):
        markdown = (DIRECTORY / "chapters" / chapter).read_text()
        block = next(
            source
            for source in re.findall(r"```bash\n(.*?)\n```", markdown, re.S)
            if command in source
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / existing).mkdir(parents=True)
            interpreter = root / "rlx/.venv-microduck/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/bin/sh\nprintf invoked > training-was-invoked\n")
            interpreter.chmod(0o755)
            block = block.replace(
                "/tmp/swing-bootstrap-reproduction-YYYYMMDD-HHMMSS",
                str(root / "bootstrap-output"),
            )
            process = subprocess.run(
                ["bash"], input=block, cwd=root, text=True, capture_output=True
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse((root / "training-was-invoked").exists())

    def test_dance_freshness_guard_stops_before_generator(self):
        self.assert_existing_directory_blocks_command(
            "04-dance.md",
            "--source dance-clip/bachata_microduck_v2.clip.json",
            "rlx/artifacts/dance-reproduction-YYYYMMDD-HHMMSS",
        )

    def test_swing_freshness_guard_rejects_entire_existing_run(self):
        self.assert_existing_directory_blocks_command(
            "07-swing.md",
            "rlx/scripts/bootstrap_swing_e2e.py \\",
            "rlx/runs/studio/swing/swing-reproduction-YYYYMMDD-HHMMSS",
        )

    def test_reproduction_includes_all_three_prerequisite_chains(self):
        appendix = self.builder.reproduction_appendix()
        for token in (
            "running-base.json",
            "stilts-base.json",
            "bootstrap_swing_e2e.py",
            "swing-teacher.json",
        ):
            self.assertIn(token, appendix)
        self.assertEqual(appendix.count("refusing checkpoint copy"), 8)

    def test_reproduction_preserves_expected_failure_guards(self):
        appendix = self.builder.reproduction_appendix()
        self.assertIn("Base API report does not match this run", appendix)
        self.assertIn("Runner status contradicts API failure", appendix)
        self.assertIn("skill_status", appendix)
        self.assertNotIn("|| true", appendix)

    def test_complete_book_examples_parse(self):
        markdown = (DIRECTORY / "microduck-training-student-book.md").read_text()
        counts = self.validator.validate_examples(markdown)
        self.assertGreater(counts["bash"], 10)
        self.assertGreater(counts["python"], 3)

    def test_invalid_shell_example_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.validator.validate_examples("```bash\nif then\n```")

    def test_invalid_python_example_is_rejected(self):
        with self.assertRaises(SyntaxError):
            self.validator.validate_examples("```python\ndef broken(\n```")

    def test_missing_local_asset_is_rejected(self):
        links, missing = self.validator.local_links(
            "![missing](assets/not-a-real-asset.png)"
        )
        self.assertEqual(links, ["assets/not-a-real-asset.png"])
        self.assertEqual(missing, links)

    def test_real_link_and_fragment_are_accepted(self):
        links, missing = self.validator.local_links("[read](README.md) [section](#one)")
        self.assertEqual(len(links), 2)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
