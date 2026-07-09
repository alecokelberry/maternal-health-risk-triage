"""Score one set of vitals with the shipped forest."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from triage.data import FEATURE_COLS, MODELS_DIR


def load_artifacts():
    model = joblib.load(MODELS_DIR / "random_forest.joblib")
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, encoder, metadata


def predict_row(vitals: dict[str, float]) -> tuple[str, dict[str, float], list[str]]:
    """Return the risk label, class probabilities, and any range warnings."""
    missing = [column for column in FEATURE_COLS if column not in vitals]
    if missing:
        raise ValueError(f"missing vitals: {', '.join(missing)}")

    model, encoder, metadata = load_artifacts()
    warnings = []
    for column in FEATURE_COLS:
        value = float(vitals[column])
        span = metadata["training_ranges"][column]
        if value < span["min"] or value > span["max"]:
            warnings.append(
                f"{column}={value:g} is outside the training range {span['min']:g}–{span['max']:g}"
            )

    frame = pd.DataFrame([{column: float(vitals[column]) for column in FEATURE_COLS}])
    encoded = model.predict(frame)[0]
    label = str(encoder.inverse_transform([encoded])[0])
    probabilities = {
        str(class_name): float(probability)
        for class_name, probability in zip(
            encoder.classes_,
            model.predict_proba(frame)[0],
            strict=True,
        )
    }
    return label, probabilities, warnings
