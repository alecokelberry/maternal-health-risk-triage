"""Command-line output and exit codes."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from triage.cli import main
from triage.predict import ModelNotFoundError

VITALS = [
    "--age", "25",
    "--systolic", "130",
    "--diastolic", "80",
    "--bs", "15",
    "--temp", "98",
    "--heart-rate", "86",
]  # fmt: skip


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exit_:
            code = int(exit_.code or 0)
    return code, out.getvalue(), err.getvalue()


class PredictCommandTests(unittest.TestCase):
    def test_prints_the_label_then_probabilities(self) -> None:
        code, out, err = run("predict", *VITALS)
        self.assertEqual(code, 0)
        label, scores = out.splitlines()
        self.assertEqual(label, "high risk")
        self.assertTrue(scores.strip().startswith("high risk"))
        self.assertEqual(err, "")

    def test_range_warning_goes_to_stderr(self) -> None:
        argv = [*VITALS[:-3], "37", *VITALS[-2:]]
        code, out, err = run("predict", *argv)
        self.assertEqual(code, 0)
        self.assertIn("warning: BodyTemp=37", err)
        self.assertNotIn("warning", out)

    def test_bad_numbers_are_usage_errors(self) -> None:
        for bad in ["abc", "-5", "0", "nan", "inf"]:
            with self.subTest(bad=bad):
                code, _, err = run("predict", *VITALS[:-1], bad)
                self.assertEqual(code, 2)
                self.assertIn("--heart-rate", err)

    def test_a_missing_vital_is_a_usage_error(self) -> None:
        code, _, _ = run("predict", *VITALS[:-2])
        self.assertEqual(code, 2)

    def test_a_missing_model_exits_1_with_a_hint(self) -> None:
        with mock.patch(
            "triage.predict.load_artifacts",
            side_effect=ModelNotFoundError("no trained model; run train"),
        ):
            code, out, err = run("predict", *VITALS)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("error: no trained model", err)


class TopLevelTests(unittest.TestCase):
    def test_a_command_is_required(self) -> None:
        code, _, _ = run()
        self.assertEqual(code, 2)

    def test_version(self) -> None:
        code, out, _ = run("--version")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^triage \d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
