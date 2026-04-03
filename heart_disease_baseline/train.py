"""Runs the train-test experiment, ranks models, and saves metrics, plots, and the best model."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from heart_disease_baseline.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    MODEL_SELECTION_PRIORITY,
    RANDOM_STATE,
    TEST_SIZE,
)
from heart_disease_baseline.data import load_dataset, split_features_target
from heart_disease_baseline.models import build_model_registry


def select_best_model(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        by=["false_negatives", "recall", "f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def run_baseline_experiment() -> pd.DataFrame:
    dataframe = load_dataset()
    features, target = split_features_target(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    fitted_models: dict[str, object] = {}

    for model_name, model in build_model_registry().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )
        confusion_matrices[model_name] = confusion_matrix(y_test, predictions).tolist()
        fitted_models[model_name] = model

    metrics = select_best_model(pd.DataFrame(metrics_rows))
    metrics.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
    (ARTIFACTS_DIR / "confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )

    best_model_name = metrics.iloc[0]["model"]
    selection_summary = {
        "selection_goal": "Minimize false negatives and maximize sensitivity for heart disease detection.",
        "selection_priority": MODEL_SELECTION_PRIORITY,
        "best_model": best_model_name,
        "best_model_metrics": metrics.iloc[0].to_dict(),
    }
    (ARTIFACTS_DIR / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )
    joblib.dump(fitted_models[best_model_name], ARTIFACTS_DIR / "best_model.joblib")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics, x="recall", y="model", orient="h")
    plt.title("Model Comparison by Recall")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_recall_scores.png", dpi=200)
    plt.close()

    for model_name, matrix in confusion_matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name}.png", dpi=200)
        plt.close()

    print(metrics.to_string(index=False))
    print("Best model selection rule: minimize false negatives, then maximize recall.")
    print(f"Best model: {best_model_name}")
    print(f"Saved training outputs to: {ARTIFACTS_DIR}")

    return metrics


if __name__ == "__main__":
    run_baseline_experiment()