"""Random forest training and holdout scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def make_forest(overrides: dict | None = None) -> RandomForestClassifier:
    params: dict = {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    if overrides:
        params.update(overrides)
    return RandomForestClassifier(**params)


def cross_validate(
    model: RandomForestClassifier, x_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """Weighted F1 on the training rows only. The holdout stays unseen."""
    folder = StratifiedKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=folder,
        scoring="f1_weighted",
        n_jobs=1,
    )
    return {
        "n_folds": N_CV_FOLDS,
        "scoring": "f1_weighted",
        "scores": [float(score) for score in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def tune(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[RandomForestClassifier, dict]:
    """Search tree settings by cross-validated weighted F1, then refit the winner."""
    search = RandomizedSearchCV(
        estimator=make_forest(),
        param_distributions={
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced_subsample", "balanced"],
        },
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_weighted",
        cv=StratifiedKFold(
            n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_index = int(search.best_index_)
    fold_scores = [
        float(search.cv_results_[f"split{fold}_test_score"][best_index])
        for fold in range(N_CV_FOLDS)
    ]
    info = {
        "search": "RandomizedSearchCV",
        "n_iter": SEARCH_ITERATIONS,
        "scoring": "f1_weighted",
        "cv_folds": N_CV_FOLDS,
        "best_params": {
            key: _jsonable(value) for key, value in search.best_params_.items()
        },
        "best_cv_score_weighted_f1": float(search.best_score_),
        "best_fold_scores": fold_scores,
    }
    return search.best_estimator_, info


def holdout_metrics(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    tag: str,
) -> dict:
    predicted = model.predict(x_test)
    labels = list(range(len(class_names)))
    weighted_f1 = float(
        f1_score(y_test, predicted, average="weighted", zero_division=0)
    )
    return {
        "tag": tag,
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
        "meets_target_weighted_f1_ge_0_80": weighted_f1 >= TARGET_WEIGHTED_F1,
    }


def importances(model: RandomForestClassifier) -> dict[str, float]:
    ranked = {
        FEATURE_COLS[index]: float(model.feature_importances_[index])
        for index in range(len(FEATURE_COLS))
    }
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
