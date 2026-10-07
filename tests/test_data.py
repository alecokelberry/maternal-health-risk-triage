"""Loading, cleaning, and splitting the vitals table."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from triage.data import FEATURE_COLS, TARGET_COL, TEST_SIZE, load_raw, preprocess, split

ROW = {
    "Age": 25,
    "SystolicBP": 130,
    "DiastolicBP": 80,
    "BS": 15.0,
    "BodyTemp": 98.0,
    "HeartRate": 86,
    "RiskLevel": "high risk",
}


def frame(*overrides: dict[str, object]) -> pd.DataFrame:
    """One row per override, each starting from ROW with a distinct age."""
    rows = [{**ROW, "Age": 20 + i, **change} for i, change in enumerate(overrides)]
    return pd.DataFrame(rows)


class PublicCsvTests(unittest.TestCase):
    def test_cleaning_counts_on_the_public_csv(self) -> None:
        cleaned, encoder, notes = preprocess(load_raw())
        self.assertEqual(notes["raw_rows"], 1014)
        self.assertEqual(notes["incomplete_rows_dropped"], 0)
        self.assertEqual(notes["implausible_rows_dropped"], 2)
        self.assertEqual(notes["duplicate_rows_dropped"], 561)
        self.assertEqual(notes["rows_after_cleaning"], 451)
        self.assertEqual(len(cleaned), 451)
        self.assertEqual(
            notes["class_counts"], {"low risk": 233, "high risk": 112, "mid risk": 106}
        )
        self.assertEqual(list(encoder.classes_), ["high risk", "low risk", "mid risk"])
        self.assertEqual(list(cleaned.columns[:6]), FEATURE_COLS)

    def test_split_shares_no_rows_between_train_and_test(self) -> None:
        cleaned, _, _ = preprocess(load_raw())
        x_train, x_test, y_train, y_test = split(cleaned)
        train_rows = set(map(tuple, x_train.assign(y=y_train).to_numpy()))
        test_rows = set(map(tuple, x_test.assign(y=y_test).to_numpy()))
        self.assertEqual(train_rows & test_rows, set())
        # scikit-learn rounds the test share up.
        self.assertEqual(len(x_test), math.ceil(len(cleaned) * TEST_SIZE))

    def test_split_is_stratified_and_repeatable(self) -> None:
        cleaned, _, _ = preprocess(load_raw())
        _, x_test_a, y_train, y_test = split(cleaned)
        _, x_test_b, _, _ = split(cleaned)
        self.assertTrue(x_test_a.equals(x_test_b))
        train_share = y_train.value_counts(normalize=True)
        test_share = y_test.value_counts(normalize=True)
        for label in train_share.index:
            self.assertAlmostEqual(train_share[label], test_share[label], delta=0.02)


class PreprocessTests(unittest.TestCase):
    def test_drops_blank_and_non_numeric_vitals(self) -> None:
        cleaned, _, notes = preprocess(
            frame({}, {"HeartRate": None}, {"BS": "n/a"}, {"RiskLevel": None})
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(notes["incomplete_rows_dropped"], 3)

    def test_drops_exact_duplicates_but_keeps_conflicting_labels(self) -> None:
        raw = pd.concat(
            [
                frame({}),
                frame({}),
                frame({"RiskLevel": "low risk"}),
            ],
            ignore_index=True,
        )
        cleaned, _, notes = preprocess(raw)
        self.assertEqual(notes["duplicate_rows_dropped"], 1)
        self.assertEqual(notes["conflicting_label_rows"], 2)
        self.assertEqual(len(cleaned), 2)

    def test_drops_an_implausible_heart_rate(self) -> None:
        cleaned, _, notes = preprocess(frame({}, {"HeartRate": 7}, {"HeartRate": 300}))
        self.assertEqual(notes["implausible_rows_dropped"], 2)
        self.assertEqual(cleaned["HeartRate"].tolist(), [86])

    def test_normalizes_label_case_and_spacing(self) -> None:
        cleaned, encoder, _ = preprocess(
            frame({"RiskLevel": " High Risk "}, {"RiskLevel": "LOW RISK"})
        )
        self.assertEqual(list(encoder.classes_), ["high risk", "low risk"])
        self.assertEqual(cleaned[TARGET_COL].tolist(), ["high risk", "low risk"])

    def test_rejects_an_unknown_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected risk labels"):
            preprocess(frame({"RiskLevel": "severe"}))


class LoadRawTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "vitals.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_a_byte_order_mark_and_padded_headers(self) -> None:
        header = " Age, SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate,RiskLevel"
        path = self.write(f"﻿{header}\n25,130,80,15,98,86,high risk\n")
        self.assertEqual(list(load_raw(path).columns), [*FEATURE_COLS, TARGET_COL])

    def test_names_missing_columns(self) -> None:
        path = self.write("Age,SystolicBP\n25,130\n")
        with self.assertRaisesRegex(ValueError, "DiastolicBP"):
            load_raw(path)


if __name__ == "__main__":
    unittest.main()
