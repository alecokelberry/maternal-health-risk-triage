"""Fit a baseline forest and a searched forest; keep the cross-validation winner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import sklearn

from triage.data import (
    FEATURE_COLS,
    MODELS_DIR,
    PLAUSIBLE_RANGES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_CSV,
    RESULTS_DIR,
    TARGET_COL,
    TEST_SIZE,
    load_raw,
    preprocess,
    split,
    training_ranges,
    write_tables,
)
from triage.model import (
    TARGET_WEIGHTED_F1,
    cross_validate,
    holdout_metrics,
    importances,
    majority_baseline,
    make_forest,
    tune,
)

MODEL_FILE = "random_forest.joblib"
ENCODER_FILE = "label_encoder.joblib"
METADATA_FILE = "metadata.json"


def run(
    raw_csv: Path = RAW_CSV,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> int:
    if not raw_csv.exists():
        print(f"error: missing raw CSV at {raw_csv}", file=sys.stderr)
        return 1

    processed, encoder, notes = preprocess(load_raw(raw_csv))
    x_train, x_test, y_train, y_test = split(processed)
    write_tables(processed, x_train, x_test, y_train, y_test, encoder, processed_dir)
    class_names = [str(label) for label in encoder.classes_]

    baseline = make_forest()
    baseline_cv = cross_validate(make_forest(), x_train, y_train)
    baseline.fit(x_train, y_train)
    tuned, tuning = tune(x_train, y_train)

    # Choose on cross-validation only. Holdout scores are computed after.
    if baseline_cv["mean"] >= tuning["cv"]["mean"]:
        chosen_name, chosen, chosen_cv = "baseline", baseline, baseline_cv
    else:
        chosen_name, chosen, chosen_cv = "tuned", tuned, tuning["cv"]
    holdout = holdout_metrics(chosen, x_test, y_test, class_names)

    metadata = {
        "features": FEATURE_COLS,
        "labels": class_names,
        "chosen_model": chosen_name,
        "training_ranges": training_ranges(x_train),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
    }
    metrics = {
        "dataset": {
            "name": "Maternal Health Risk",
            "source": "UCI Machine Learning Repository ID 863",
            "doi": "10.24432/C5DP5D",
            "license": "CC BY 4.0",
        },
        "disclaimer": (
            "A portfolio demo. It is not a diagnostic device and does not "
            "recommend treatment."
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "plausible_ranges": PLAUSIBLE_RANGES,
        "preprocess": notes,
        "success_target": {"metric": "weighted_f1", "threshold": TARGET_WEIGHTED_F1},
        "model_selection": (
            "The shipped model is whichever of the baseline and the search had the "
            "higher cross-validated weighted F1. The holdout is scored once, after "
            "that choice."
        ),
        "baseline_cv": baseline_cv,
        "tuning": tuning,
        "chosen_model": chosen_name,
        "chosen_cv": chosen_cv,
        "chosen_holdout": holdout,
        "majority_class_holdout": majority_baseline(x_train, y_train, x_test, y_test),
        "feature_importances": importances(chosen),
    }

    save_artifacts(chosen, encoder, metadata, models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "evaluation_summary.txt").write_text(
        summarize(metrics), encoding="utf-8"
    )
    print(
        f"Shipped {chosen_name} model. "
        f"Cross-validated weighted F1 {chosen_cv['mean']:.3f}. "
        f"Holdout weighted F1 {holdout['f1_weighted']:.3f}."
    )
    return 0


def save_artifacts(
    model: Any, encoder: Any, metadata: dict[str, Any], models_dir: Path
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / MODEL_FILE, compress=3)
    joblib.dump(encoder, models_dir / ENCODER_FILE)
    (models_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def summarize(metrics: dict[str, Any]) -> str:
    """Plain-text report of the cleaning, model choice, and holdout scores."""
    notes = metrics["preprocess"]
    holdout = metrics["chosen_holdout"]
    report = holdout["classification_report"]
    majority = metrics["majority_class_holdout"]
    lines = [
        "Maternal health risk triage. Not a diagnosis.",
        "",
        f"Raw rows: {notes['raw_rows']}",
        f"  incomplete dropped:   {notes['incomplete_rows_dropped']}",
        f"  implausible dropped:  {notes['implausible_rows_dropped']}",
        f"  duplicates dropped:   {notes['duplicate_rows_dropped']}",
        f"Rows kept: {notes['rows_after_cleaning']} "
        f"(train {metrics['n_train']} / test {metrics['n_test']})",
        "",
        "Cross-validated weighted F1 (5 folds, training rows only)",
        f"  baseline forest: {_mean_std(metrics['baseline_cv'])}",
        f"  searched forest: {_mean_std(metrics['tuning']['cv'])}",
        f"  shipped: {metrics['chosen_model']}",
        "",
        "Holdout, scored once after the choice",
        f"  accuracy:    {holdout['accuracy']:.3f}",
        f"  weighted F1: {holdout['f1_weighted']:.3f} "
        f"(target {metrics['success_target']['threshold']:.2f})",
        f"  macro F1:    {holdout['f1_macro']:.3f}",
        f"  majority-class weighted F1, for scale: {majority['f1_weighted']:.3f}",
        "",
        f"  {'class':<10} {'precision':>9} {'recall':>7} {'f1':>6} {'rows':>5}",
        *[
            f"  {label:<10} {report[label]['precision']:>9.3f} "
            f"{report[label]['recall']:>7.3f} {report[label]['f1-score']:>6.3f} "
            f"{int(report[label]['support']):>5}"
            for label in holdout["class_labels"]
        ],
        "",
        "Feature importance (impurity-based)",
        *[
            f"  {name}: {weight:.3f}"
            for name, weight in metrics["feature_importances"].items()
        ],
    ]
    return "\n".join(lines) + "\n"


def _mean_std(cv: dict[str, Any]) -> str:
    return f"{cv['mean']:.3f} +/- {cv['std']:.3f}"
