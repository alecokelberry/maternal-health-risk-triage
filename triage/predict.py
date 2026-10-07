"""Score one set of vitals with the shipped forest."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from triage.data import FEATURE_COLS, MODELS_DIR
from triage.train import ENCODER_FILE, METADATA_FILE, MODEL_FILE


class ModelNotFoundError(FileNotFoundError):
    """The models directory is missing a file that predict needs."""


@dataclass(frozen=True)
class Artifacts:
    model: Any
    encoder: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]
    warnings: list[str]


def load_artifacts(models_dir: Path = MODELS_DIR) -> Artifacts:
    """Load the model, label encoder, and metadata written by `triage train`."""
    paths = [models_dir / name for name in (MODEL_FILE, ENCODER_FILE, METADATA_FILE)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise ModelNotFoundError(
            f"no trained model in {models_dir} (missing {', '.join(missing)}); "
            "run `python -m triage train` first"
        )
    with warnings.catch_warnings():
        # A version mismatch is reported once, through Prediction.warnings.
        warnings.simplefilter("ignore", category=UserWarning)
        model = joblib.load(paths[0])
        encoder = joblib.load(paths[1])
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    return Artifacts(model, encoder, metadata)


def range_warnings(
    vitals: Mapping[str, float], ranges: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """One message per vital outside the range seen in training."""
    messages = []
    for column in FEATURE_COLS:
        value, span = vitals[column], ranges[column]
        if not span["min"] <= value <= span["max"]:
            messages.append(
                f"{column}={value:g} is outside the training range "
                f"{span['min']:g}-{span['max']:g}; treat this score with caution"
            )
    return messages


def predict_row(
    vitals: Mapping[str, float], artifacts: Artifacts | None = None
) -> Prediction:
    """Return the risk label, class probabilities, and any warnings."""
    missing = [column for column in FEATURE_COLS if column not in vitals]
    if missing:
        raise ValueError(f"missing vitals: {', '.join(missing)}")
    row = {column: float(vitals[column]) for column in FEATURE_COLS}
    bad = [column for column, value in row.items() if not math.isfinite(value)]
    if bad:
        raise ValueError(f"vitals must be finite numbers: {', '.join(bad)}")

    artifacts = artifacts or load_artifacts()
    notes = range_warnings(row, artifacts.metadata["training_ranges"])
    trained_with = artifacts.metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        notes.append(
            f"model was trained with scikit-learn {trained_with}, "
            f"running {sklearn.__version__}; install requirements.txt to match"
        )

    frame = pd.DataFrame([row], columns=FEATURE_COLS)
    scores = artifacts.model.predict_proba(frame)[0]
    class_names = artifacts.encoder.inverse_transform(artifacts.model.classes_)
    probabilities = {
        str(name): float(score) for name, score in zip(class_names, scores, strict=True)
    }
    label = max(probabilities, key=probabilities.__getitem__)
    return Prediction(label, probabilities, notes)
