"""Fit a baseline forest and a searched forest. Keep the one that wins cross-validation."""

from __future__ import annotations

import json
import statistics
import sys

import joblib

from triage.data import (
    FEATURE_COLS,
    MODELS_DIR,
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
    make_forest,
    tune,
)


def run() -> int:
    if not RAW_CSV.exists():
        print(f"Missing raw CSV at {RAW_CSV}", file=sys.stderr)
        return 1

    raw = load_raw()
    processed, encoder, notes = preprocess(raw)
    x_train, x_test, y_train, y_test = split(processed)
    write_tables(processed, x_train, x_test, y_train, y_test, encoder)
    class_names = [str(label) for label in encoder.classes_]

    baseline = make_forest()
    baseline_cv = cross_validate(make_forest(), x_train, y_train)
    baseline.fit(x_train, y_train)
    baseline_holdout = holdout_metrics(
        baseline, x_test, y_test, class_names, "baseline_holdout"
    )

    tuned, tune_info = tune(x_train, y_train)
    tuned_cv = {
        "n_folds": tune_info["cv_folds"],
        "scoring": "f1_weighted",
        "scores": tune_info["best_fold_scores"],
        "mean": tune_info["best_cv_score_weighted_f1"],
        "std": statistics.pstdev(tune_info["best_fold_scores"]),
    }
    tuned_holdout = holdout_metrics(tuned, x_test, y_test, class_names, "tuned_holdout")

    # The holdout scores above are reported. They are not used to pick a winner.
    if baseline_cv["mean"] >= tuned_cv["mean"]:
        chosen = baseline
        chosen_name = "baseline"
        chosen_holdout = baseline_holdout
        chosen_cv = baseline_cv
    else:
        chosen = tuned
        chosen_name = "tuned"
        chosen_holdout = tuned_holdout
        chosen_cv = tuned_cv

    weights = importances(chosen)
    ranges = training_ranges(x_train)
    _save(
        model=chosen,
        encoder=encoder,
        chosen_name=chosen_name,
        notes=notes,
        baseline_cv=baseline_cv,
        baseline_holdout=baseline_holdout,
        tune_info=tune_info,
        tuned_cv=tuned_cv,
        tuned_holdout=tuned_holdout,
        chosen_cv=chosen_cv,
        chosen_holdout=chosen_holdout,
        weights=weights,
        ranges=ranges,
        n_train=len(x_train),
        n_test=len(x_test),
    )
    print(
        f"Shipped {chosen_name} model. "
        f"Cross-validated weighted F1 {chosen_cv['mean']:.4f}. "
        f"Holdout weighted F1 {chosen_holdout['f1_weighted']:.4f}."
    )
    return 0


def _save(**details) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = details["model"]
    encoder = details["encoder"]
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    metadata = {
        "features": FEATURE_COLS,
        "labels": [str(label) for label in encoder.classes_],
        "chosen_model": details["chosen_name"],
        "training_ranges": details["ranges"],
        "random_state": RANDOM_STATE,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    payload = {
        "dataset": {
            "name": "Maternal Health Risk",
            "source": "UCI Machine Learning Repository ID 863",
            "doi": "10.24432/C5DP5D",
            "license": "CC BY 4.0",
        },
        "disclaimer": (
            "A care-coordination demo. It ranks follow-up priority from routine "
            "vitals. It is not a diagnostic device and it does not recommend treatment."
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": details["n_train"],
        "n_test": details["n_test"],
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "success_target": {"metric": "weighted_f1", "threshold": TARGET_WEIGHTED_F1},
        "model_selection": (
            "The shipped model is whichever of the baseline and the search "
            "had the higher cross-validated weighted F1. Holdout scores are reported after that choice."
        ),
        "chosen_model": details["chosen_name"],
        "preprocess": details["notes"],
        "baseline_cv": details["baseline_cv"],
        "baseline_holdout": details["baseline_holdout"],
        "tuning": details["tune_info"],
        "tuned_cv": details["tuned_cv"],
        "tuned_holdout": details["tuned_holdout"],
        "chosen_cv": details["chosen_cv"],
        "chosen_holdout": details["chosen_holdout"],
        "feature_importances": details["weights"],
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    chosen = details["chosen_holdout"]
    summary = RESULTS_DIR / "evaluation_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Maternal health risk triage",
                "Ranks follow-up priority. Not a diagnosis.",
                "",
                f"Train / test rows: {details['n_train']} / {details['n_test']}",
                f"Shipped model: {details['chosen_name']}",
                f"Selection metric: cross-validated weighted F1 (target >= {TARGET_WEIGHTED_F1})",
                "",
                "Baseline cross-validation "
                f"{details['baseline_cv']['mean']:.4f} +/- {details['baseline_cv']['std']:.4f}",
                "Searched cross-validation "
                f"{details['tuned_cv']['mean']:.4f} +/- {details['tuned_cv']['std']:.4f}",
                "",
                "Holdout of the shipped model (not used for selection)",
                f"  accuracy:           {chosen['accuracy']:.4f}",
                f"  precision weighted: {chosen['precision_weighted']:.4f}",
                f"  recall weighted:    {chosen['recall_weighted']:.4f}",
                f"  f1 weighted:        {chosen['f1_weighted']:.4f}",
                f"  f1 macro:           {chosen['f1_macro']:.4f}",
                "",
                "Feature influence on the shipped forest",
                *[
                    f"  {name}: {weight:.4f}"
                    for name, weight in details["weights"].items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
