"""Load and clean the maternal health risk table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
ENCODED_COL = "RiskLevelEncoded"
RISK_LEVELS = {"low risk", "mid risk", "high risk"}

# A reading outside these bounds is a recording error, not a patient. The
# public CSV has one: a heart rate of 7 bpm, listed twice.
PLAUSIBLE_RANGES = {"HeartRate": (30.0, 220.0)}

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV, including a file that starts with a byte-order mark."""
    frame = pd.read_csv(path or RAW_CSV, encoding="utf-8-sig")
    frame.columns = [column.strip() for column in frame.columns]
    missing = [col for col in [*FEATURE_COLS, TARGET_COL] if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    return frame


def preprocess(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, dict[str, Any]]:
    """Clean the vitals and encode the risk label.

    Steps, in order, each counted in the returned notes:
    1. drop rows with a blank or non-numeric vital, or a blank label
    2. drop physiologically implausible readings (PLAUSIBLE_RANGES)
    3. drop exact duplicate rows

    Step 3 matters: over half of the public CSV is repeated rows. Left in, the
    same patient lands in both the training and the test split, and the
    holdout score measures memory rather than generalization.
    """
    notes: dict[str, Any] = {
        "raw_rows": len(frame),
        "missing_before": {
            column: int(frame[column].isna().sum()) for column in frame.columns
        },
    }
    work = frame.dropna(subset=[*FEATURE_COLS, TARGET_COL]).copy()
    for column in FEATURE_COLS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=FEATURE_COLS)
    work[TARGET_COL] = work[TARGET_COL].astype(str).str.strip().str.lower()
    unknown = set(work[TARGET_COL]) - RISK_LEVELS
    if unknown:
        raise ValueError(f"unexpected risk labels: {sorted(unknown)}")
    notes["incomplete_rows_dropped"] = len(frame) - len(work)

    plausible = pd.Series(True, index=work.index)
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        plausible &= work[column].between(low, high)
    notes["implausible_rows_dropped"] = int((~plausible).sum())
    work = work[plausible]

    before = len(work)
    work = work.drop_duplicates(subset=[*FEATURE_COLS, TARGET_COL])
    notes["duplicate_rows_dropped"] = before - len(work)
    notes["conflicting_label_rows"] = int(
        work.duplicated(FEATURE_COLS, keep=False).sum()
    )

    notes["rows_after_cleaning"] = len(work)
    notes["class_counts"] = {
        str(label): int(count)
        for label, count in work[TARGET_COL].value_counts().items()
    }

    encoder = LabelEncoder()
    work[ENCODED_COL] = encoder.fit_transform(work[TARGET_COL])
    notes["labels"] = [str(label) for label in encoder.classes_]
    cleaned = work[[*FEATURE_COLS, TARGET_COL, ENCODED_COL]].reset_index(drop=True)
    return cleaned, encoder, notes


def split(
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. The test rows are not used to choose the model."""
    x_train, x_test, y_train, y_test = train_test_split(
        processed[FEATURE_COLS],
        processed[ENCODED_COL],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=processed[ENCODED_COL],
    )
    return x_train, x_test, y_train, y_test


def write_tables(
    processed: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoder: LabelEncoder,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Write the cleaned table and both splits as CSV, for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_dir / "maternal_health_risk_preprocessed.csv", index=False)
    for features, labels, name in [
        (x_train, y_train, "train.csv"),
        (x_test, y_test, "test.csv"),
    ]:
        table = features.copy()
        table[ENCODED_COL] = labels.to_numpy()
        table[TARGET_COL] = encoder.inverse_transform(labels)
        table.to_csv(out_dir / name, index=False)


def training_ranges(x_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min and max of each vital in the training rows, for range warnings."""
    return {
        column: {
            "min": float(x_train[column].min()),
            "max": float(x_train[column].max()),
        }
        for column in FEATURE_COLS
    }
