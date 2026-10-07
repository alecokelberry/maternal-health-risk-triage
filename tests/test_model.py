"""Scoring helpers and one end-to-end training run with a reduced search."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

from triage.data import FEATURE_COLS
from triage.model import (
    cross_validate,
    holdout_metrics,
    importances,
    majority_baseline,
    make_forest,
)
from triage.predict import load_artifacts, predict_row
from triage.train import run


def separable(n_per_class: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Three classes that blood sugar alone separates."""
    rows, labels = [], []
    for label, sugar in enumerate([6.0, 10.0, 15.0]):
        for i in range(n_per_class):
            rows.append([20 + i, 120, 80, sugar + i * 0.01, 98.0, 75])
            labels.append(label)
    return pd.DataFrame(rows, columns=FEATURE_COLS), pd.Series(labels)


class MetricTests(unittest.TestCase):
    def test_holdout_metrics_on_a_perfect_model(self) -> None:
        x, y = separable()
        model = make_forest({"n_estimators": 10}).fit(x, y)
        metrics = holdout_metrics(model, x, y, ["high", "low", "mid"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_weighted"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"], [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
        )
        self.assertTrue(metrics["meets_target_weighted_f1"])

    def test_importances_sum_to_one_largest_first(self) -> None:
        x, y = separable()
        model = make_forest({"n_estimators": 10}).fit(x, y)
        weights = importances(model)
        self.assertEqual(set(weights), set(FEATURE_COLS))
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(list(weights.values()), sorted(weights.values(), reverse=True))

    def test_majority_baseline_predicts_the_most_common_class(self) -> None:
        x, y = separable()
        y_test = pd.Series([0, 0, 0, 1])
        scores = majority_baseline(x, y.replace({2: 0}), x.head(4), y_test)
        self.assertEqual(scores["accuracy"], 0.75)

    def test_cross_validate_reports_five_folds(self) -> None:
        x, y = separable()
        cv = cross_validate(make_forest({"n_estimators": 10}), x, y)
        self.assertEqual(cv["n_folds"], 5)
        self.assertEqual(len(cv["scores"]), 5)
        self.assertGreater(cv["mean"], 0.9)


class TrainRunTests(unittest.TestCase):
    def test_writes_a_model_that_predict_can_load(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)

        with (
            mock.patch("triage.model.SEARCH_ITERATIONS", 2),
            redirect_stdout(StringIO()),
        ):
            code = run(
                models_dir=root / "models",
                results_dir=root / "results",
                processed_dir=root / "processed",
            )

        self.assertEqual(code, 0)
        metrics = json.loads((root / "results" / "metrics.json").read_text())
        self.assertEqual(metrics["n_train"] + metrics["n_test"], 451)
        self.assertIn(metrics["chosen_model"], {"baseline", "tuned"})
        self.assertLess(
            metrics["majority_class_holdout"]["f1_weighted"],
            metrics["chosen_holdout"]["f1_weighted"],
        )
        summary = (root / "results" / "evaluation_summary.txt").read_text()
        self.assertIn("duplicates dropped:   561", summary)
        self.assertTrue((root / "processed" / "test.csv").exists())

        artifacts = load_artifacts(root / "models")
        vitals = dict(zip(FEATURE_COLS, [25, 130, 80, 15, 98, 86], strict=True))
        self.assertIn(
            predict_row(vitals, artifacts).label, artifacts.metadata["labels"]
        )

    def test_missing_csv_exits_1(self) -> None:
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new=StringIO()):
            self.assertEqual(run(raw_csv=Path("/nonexistent.csv")), 1)


if __name__ == "__main__":
    unittest.main()
