"""Load and clean the maternal health risk table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Maternal_Health_Risk_Data_Set.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FEATURE_COLS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
TARGET_COL = "RiskLevel"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV, including a file that starts with a byte-order mark."""
    frame = pd.read_csv(path or RAW_CSV, encoding="utf-8-sig")
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict]:
    """Coerce vitals to numbers, drop incomplete rows, and encode the risk label."""
    notes: dict = {
        "raw_rows": int(len(frame)),
        "missing_before": {column: int(frame[column].isna().sum()) for column in frame.columns},
    }
    work = frame.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    for column in FEATURE_COLS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=FEATURE_COLS)
    work[TARGET_COL] = work[TARGET_COL].astype(str).str.strip().str.lower()
    notes["rows_after_cleaning"] = int(len(work))
    notes["class_counts"] = {str(k): int(v) for k, v in work[TARGET_COL].value_counts().items()}

    encoder = LabelEncoder()
    work["RiskLevelEncoded"] = encoder.fit_transform(work[TARGET_COL])
    notes["labels"] = [str(label) for label in encoder.classes_]
    cleaned = work[FEATURE_COLS + [TARGET_COL, "RiskLevelEncoded"]].reset_index(drop=True)
    return cleaned, encoder, notes


def split(
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. The test rows are not used to choose the model."""
    features = processed[FEATURE_COLS]
    target = processed["RiskLevelEncoded"]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def write_tables(
    processed: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoder: LabelEncoder,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DIR / "maternal_health_risk_preprocessed.csv", index=False)

    def _side(features: pd.DataFrame, labels: pd.Series, name: str) -> None:
        table = features.copy()
        table["RiskLevelEncoded"] = labels.to_numpy()
        table["RiskLevel"] = encoder.inverse_transform(labels)
        table.to_csv(PROCESSED_DIR / name, index=False)

    _side(x_train, y_train, "train.csv")
    _side(x_test, y_test, "test.csv")


def training_ranges(x_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min and max of each vital on the training rows, for a range warning at predict time."""
    return {
        column: {"min": float(x_train[column].min()), "max": float(x_train[column].max())}
        for column in FEATURE_COLS
    }
