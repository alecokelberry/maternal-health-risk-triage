"""Cleaning rules and a prediction from the saved model."""

from __future__ import annotations

import unittest

from triage.data import FEATURE_COLS, RAW_CSV, load_raw, preprocess
from triage.predict import predict_row


class PreprocessTests(unittest.TestCase):
    def test_public_csv_keeps_three_risk_levels(self) -> None:
        raw = load_raw(RAW_CSV)
        cleaned, encoder, notes = preprocess(raw)
        self.assertEqual(notes["raw_rows"], 1014)
        self.assertEqual(notes["rows_after_cleaning"], 451)
        self.assertEqual(list(cleaned.columns[:6]), FEATURE_COLS)
        self.assertEqual(set(encoder.classes_), {"high risk", "low risk", "mid risk"})
        self.assertEqual(notes["class_counts"]["low risk"], 233)
        self.assertEqual(notes["class_counts"]["mid risk"], 106)
        self.assertEqual(notes["class_counts"]["high risk"], 112)

    def test_blank_vital_is_dropped(self) -> None:
        raw = load_raw(RAW_CSV).head(5).copy()
        raw.loc[0, "HeartRate"] = None
        cleaned, _, notes = preprocess(raw)
        self.assertEqual(notes["rows_after_cleaning"], 4)
        self.assertEqual(len(cleaned), 4)


class PredictTests(unittest.TestCase):
    def test_known_high_risk_row_returns_a_label_and_probabilities(self) -> None:
        label, probabilities, warnings = predict_row(
            {
                "Age": 25,
                "SystolicBP": 130,
                "DiastolicBP": 80,
                "BS": 15,
                "BodyTemp": 98,
                "HeartRate": 86,
            }
        )
        self.assertIn(label, {"low risk", "mid risk", "high risk"})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
        self.assertEqual(warnings, [])

    def test_missing_vital_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            predict_row({"Age": 25})


if __name__ == "__main__":
    unittest.main()
