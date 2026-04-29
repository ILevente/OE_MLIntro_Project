"""Builds the sklearn baseline models compared during training and evaluation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart_disease_baseline.config import RANDOM_STATE


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
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                # Tune regularization strength and solver choice for the linear baseline.
                # The `model__` prefix targets the named pipeline step above.
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "max_depth": [3, 5, 7, None],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                # Explore tree count and tree complexity for the ensemble model.
                "n_estimators": [200, 300],
                "max_depth": [None, 8, 12],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "KNN": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                # KNN tuning focuses on neighborhood size, weighting, and distance metric.
                # `p=1` means Manhattan distance, while `p=2` means Euclidean distance.
                "model__n_neighbors": [5, 11, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "SVM": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                # For the RBF SVM, C and gamma control the margin and kernel shape.
                # These also use the `model__` prefix because the classifier sits inside a pipeline.
                "model__C": [0.5, 1.0, 2.0],
                "model__gamma": ["scale", "auto"],
            },
        },
    }
