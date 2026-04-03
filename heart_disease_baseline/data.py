"""Reads the CSV dataset, splits features and target, and builds summary statistics."""

from __future__ import annotations
from typing import Any
from heart_disease_baseline.config import DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN
import pandas as pd

EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path=DATA_FILE) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    # missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    # if missing_columns:
    #     raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    # dataframe = dataframe[EXPECTED_COLUMNS].copy()

    # if dataframe[TARGET_COLUMN].nunique() != 2:
    #     raise ValueError("The target column must be binary for this baseline.")

    return dataframe


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()
    return features, target


def build_dataset_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    class_balance = dataframe[TARGET_COLUMN].value_counts().sort_index().to_dict()
    missing_values = dataframe.isna().sum().to_dict()

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_balance": {str(key): int(value) for key, value in class_balance.items()},
        "missing_values": {key: int(value) for key, value in missing_values.items()},
        "suspicious_values": {
            "cholesterol_zero": int((dataframe["Cholesterol"] == 0).sum()),
            "resting_bp_zero": int((dataframe["RestingBP"] == 0).sum()),
            "oldpeak_negative": int((dataframe["Oldpeak"] < 0).sum()),
        },
    }
