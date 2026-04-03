"""Creates the dataset summary JSON and the baseline EDA plots saved under artifacts."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import seaborn as sns

from heart_disease_baseline.config import ARTIFACTS_DIR, FIGURES_DIR, TARGET_COLUMN
from heart_disease_baseline.data import build_dataset_summary, load_dataset


def save_eda_outputs() -> None:
    dataframe = load_dataset()
    summary = build_dataset_summary(dataframe)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = ARTIFACTS_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=dataframe, x=TARGET_COLUMN)
    plt.title("Heart Disease Class Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=dataframe, x="Age", bins=20, kde=True)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Saved EDA outputs to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    save_eda_outputs()
