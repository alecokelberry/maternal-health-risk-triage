"""Scoring with the shipped model, range warnings, and loading errors."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from triage.predict import (
    Artifacts,
    ModelNotFoundError,
    load_artifacts,
    predict_row,
    range_warnings,
)

# The first row of the public CSV, labeled high risk.
FIRST_ROW = {
    "Age": 25,
    "SystolicBP": 130,
    "DiastolicBP": 80,
    "BS": 15,
    "BodyTemp": 98,
    "HeartRate": 86,
}


class ShippedModelTests(unittest.TestCase):
    artifacts: Artifacts

    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = load_artifacts()

    def test_scores_a_known_high_risk_row(self) -> None:
        result = predict_row(FIRST_ROW, self.artifacts)
        self.assertEqual(result.label, "high risk")
        self.assertEqual(
            set(result.probabilities), {"low risk", "mid risk", "high risk"}
        )
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        self.assertEqual(
            result.label,
            max(result.probabilities, key=result.probabilities.__getitem__),
        )
        self.assertEqual(result.warnings, [])

    def test_warns_when_a_vital_is_outside_the_training_range(self) -> None:
        result = predict_row({**FIRST_ROW, "BodyTemp": 37}, self.artifacts)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("BodyTemp=37 is outside the training range", result.warnings[0])

    def test_warns_when_scikit_learn_versions_differ(self) -> None:
        stale = Artifacts(
            self.artifacts.model,
            self.artifacts.encoder,
            {**self.artifacts.metadata, "sklearn_version": "0.0.1"},
        )
        result = predict_row(FIRST_ROW, stale)
        self.assertTrue(any("scikit-learn 0.0.1" in w for w in result.warnings))

    def test_rejects_missing_or_non_finite_vitals(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing vitals: SystolicBP"):
            predict_row({"Age": 25}, self.artifacts)
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_row({**FIRST_ROW, "BS": float("nan")}, self.artifacts)


class RangeWarningTests(unittest.TestCase):
    def test_bounds_are_inclusive(self) -> None:
        ranges = {name: {"min": 1.0, "max": 2.0} for name in FIRST_ROW}
        inside = dict.fromkeys(FIRST_ROW, 1.0)
        self.assertEqual(range_warnings(inside, ranges), [])
        self.assertEqual(len(range_warnings({**inside, "Age": 2.5}, ranges)), 1)


class LoadArtifactsTests(unittest.TestCase):
    def test_missing_files_raise_a_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "metadata.json").write_text(json.dumps({}))
            with self.assertRaisesRegex(ModelNotFoundError, "random_forest.joblib"):
                load_artifacts(Path(folder))


if __name__ == "__main__":
    unittest.main()
