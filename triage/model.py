"""Random forest training and holdout scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from triage.data import FEATURE_COLS, RANDOM_STATE

N_CV_FOLDS = 5
TARGET_WEIGHTED_F1 = 0.80
SEARCH_ITERATIONS = 24
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 8, 12, 16, 20],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced_subsample", "balanced"],
}


def make_forest(overrides: dict[str, Any] | None = None) -> RandomForestClassifier:
    params: dict[str, Any] = {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    if overrides:
        params.update(overrides)
    return RandomForestClassifier(**params)


def _folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cross_validate(
    model: RandomForestClassifier, x_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Any]:
    """Weighted F1 on the training rows only. The holdout stays unseen."""
    scores = cross_val_score(
        model, x_train, y_train, cv=_folds(), scoring="f1_weighted", n_jobs=1
    )
    return _cv_summary([float(score) for score in scores])


def tune(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Search tree settings by cross-validated weighted F1, then refit the winner."""
    search = RandomizedSearchCV(
        estimator=make_forest(),
        param_distributions=SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_weighted",
        cv=_folds(),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best = int(search.best_index_)
    fold_scores = [
        float(search.cv_results_[f"split{fold}_test_score"][best])
        for fold in range(N_CV_FOLDS)
    ]
    info = {
        "search": "RandomizedSearchCV",
        "n_iter": SEARCH_ITERATIONS,
        "best_params": {
            key: _jsonable(value) for key, value in search.best_params_.items()
        },
        "cv": _cv_summary(fold_scores),
    }
    return search.best_estimator_, info


def majority_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Scores of always predicting the most common training class, for scale."""
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    predicted = dummy.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_weighted": float(
            f1_score(y_test, predicted, average="weighted", zero_division=0)
        ),
    }


def holdout_metrics(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> dict[str, Any]:
    predicted = model.predict(x_test)
    labels = list(range(len(class_names)))
    weighted_f1 = float(
        f1_score(y_test, predicted, average="weighted", zero_division=0)
    )
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_weighted": float(
            precision_score(y_test, predicted, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predicted, average="weighted", zero_division=0)
        ),
        "f1_weighted": weighted_f1,
        "f1_macro": float(
            f1_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            predicted,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "class_labels": class_names,
        "meets_target_weighted_f1": weighted_f1 >= TARGET_WEIGHTED_F1,
    }


def importances(model: RandomForestClassifier) -> dict[str, float]:
    """Impurity-based feature importances, largest first."""
    ranked = zip(FEATURE_COLS, model.feature_importances_, strict=True)
    return {
        name: float(weight)
        for name, weight in sorted(ranked, key=lambda item: item[1], reverse=True)
    }


def _cv_summary(scores: list[float]) -> dict[str, Any]:
    return {
        "n_folds": len(scores),
        "scoring": "f1_weighted",
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
