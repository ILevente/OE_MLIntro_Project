"""Builds the sklearn baseline models compared during training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart_disease_baseline.config import RANDOM_STATE


def replace_invalid_measurements_with_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace clearly invalid zero-valued measurements before tuned-model imputation."""
    cleaned = dataframe.copy()
    cleaned.loc[cleaned["Cholesterol"] == 0, "Cholesterol"] = np.nan
    cleaned.loc[cleaned["RestingBP"] == 0, "RestingBP"] = np.nan
    return cleaned


def build_tuned_preprocessor(include_scaler: bool) -> list[tuple[str, object]]:
    """Build the tuned-only preprocessing steps applied before the estimator."""
    steps: list[tuple[str, object]] = [
        (
            "invalid_measurement_cleanup",
            FunctionTransformer(replace_invalid_measurements_with_nan, validate=False),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_model_registry() -> dict[str, object]:
    """Return the fixed baseline estimators used for the untuned comparison."""
    # These are the fixed baseline configurations used for the first untuned comparison.
    return {
        "Logistic Regression": Pipeline(
            [
                # Scale-sensitive models get preprocessing bundled into the same pipeline.
                # During fit/predict, sklearn runs these steps in order on the same input data.
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_tuning_registry() -> dict[str, dict[str, object]]:
    """Return fresh estimators plus parameter grids for the tuning workflow."""
    # Each entry provides a fresh estimator plus the search space used by GridSearchCV.
    return {
        "Logistic Regression": {
            "estimator": Pipeline(
                build_tuned_preprocessor(include_scaler=True)
                + [
                    (
                        "model",
                        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                # Tune regularization strength and solver choice for the linear baseline.
                # The `model__` prefix targets the named pipeline step above.
                "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "primary_complexity_param": "model__C",
            "primary_complexity_label": "C",
            "validation_curve_values": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
        "Decision Tree": {
            "estimator": Pipeline(
                build_tuned_preprocessor(include_scaler=False)
                + [("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]
            ),
            "param_grid": {
                "model__max_depth": [2, 3, 4, 5, 6, 7, 8, None],
                "model__class_weight": [None, "balanced"],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "primary_complexity_param": "model__max_depth",
            "primary_complexity_label": "Max Depth",
            "validation_curve_values": [2, 3, 4, 5, 6, 7, 8, None],
        },
        "Random Forest": {
            "estimator": Pipeline(
                build_tuned_preprocessor(include_scaler=False)
                + [("model", RandomForestClassifier(random_state=RANDOM_STATE))]
            ),
            "param_grid": {
                # Explore tree count and tree complexity for the ensemble model.
                "model__class_weight": [None, "balanced"],
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [2, 3, 4, 5, 6],
            },
            "primary_complexity_param": "model__min_samples_leaf",
            "primary_complexity_label": "Min Samples Leaf",
            "validation_curve_values": [2, 3, 4, 5, 6],
        },
        "KNN": {
            "estimator": Pipeline(
                build_tuned_preprocessor(include_scaler=True) + [("model", KNeighborsClassifier())]
            ),
            "param_grid": {
                # KNN tuning focuses on neighborhood size, weighting, and distance metric.
                # `p=1` means Manhattan distance, while `p=2` means Euclidean distance.
                "model__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "primary_complexity_param": "model__n_neighbors",
            "primary_complexity_label": "N Neighbors",
            "validation_curve_values": [3, 5, 7, 11, 15, 21, 31],
        },
        "SVM": {
            "estimator": Pipeline(
                build_tuned_preprocessor(include_scaler=True)
                + [("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]
            ),
            "param_grid": {
                # For the RBF SVM, C and gamma control the margin and kernel shape.
                # These also use the `model__` prefix because the classifier sits inside a pipeline.
                "model__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "model__class_weight": [None, "balanced"],
                "model__gamma": ["scale", "auto"],
            },
            "primary_complexity_param": "model__C",
            "primary_complexity_label": "C",
            "validation_curve_values": [0.25, 0.5, 1.0, 2.0, 4.0],
        },
    }
