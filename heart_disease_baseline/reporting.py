"""Exports metrics, summaries, tables, and figures for the training runs."""

from __future__ import annotations

import json
from collections.abc import Mapping

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease_baseline.config import ARTIFACTS_DIR, FIGURES_DIR, MODEL_SELECTION_PRIORITY


def annotate_horizontal_bars(ax: plt.Axes, fmt: str, padding: int = 4) -> None:
    """Add numeric labels to horizontal bars and widen the axis if needed."""
    # Extend the axis a bit so the numeric labels fit just beyond the bar ends.
    max_width = 0.0
    for container in ax.containers:
        if not container:
            continue
        ax.bar_label(container, fmt=fmt, padding=padding)
        container_max = max(bar.get_width() for bar in container)
        max_width = max(max_width, container_max)

    if max_width > 0:
        ax.set_xlim(0, max(ax.get_xlim()[1], max_width * 1.15))


def save_experiment_outputs(
    metrics: pd.DataFrame,
    confusion_matrices: Mapping[str, list[list[int]]],
    fitted_models: Mapping[str, object],
    *,
    prefix: str = "",
    selection_summary_extra: dict[str, object] | None = None,
) -> str:
    """Save metrics, summaries, the best model, and the main figures for one run."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save the tabular outputs first so they can be reused in the report.
    metrics.to_csv(ARTIFACTS_DIR / f"{prefix}metrics.csv", index=False)
    (ARTIFACTS_DIR / f"{prefix}confusion_matrices.json").write_text(
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
    # Tuned runs inject extra metadata here, such as CV settings and refit metric.
    if selection_summary_extra:
        selection_summary.update(selection_summary_extra)

    (ARTIFACTS_DIR / f"{prefix}selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )
    # Persist the top-ranked fitted estimator so it can be loaded later without retraining.
    joblib.dump(fitted_models[best_model_name], ARTIFACTS_DIR / f"{prefix}best_model.joblib")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=metrics, x="recall", y="model", orient="h")
    annotate_horizontal_bars(ax, fmt="%.3f")
    title_prefix = prefix.replace("_", " ").strip().title()
    chart_title = "Model Comparison by Recall"
    if title_prefix:
        chart_title = f"{title_prefix} {chart_title}"
    plt.title(chart_title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}model_recall_scores.png", dpi=200)
    plt.close()

    for model_name, matrix in confusion_matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        title = f"Confusion Matrix - {model_name}"
        if title_prefix:
            title = f"{title_prefix} {title}"
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"{prefix}confusion_matrix_{safe_name}.png", dpi=200)
        plt.close()

    print(metrics.to_string(index=False))
    print("Best model selection rule: minimize false negatives, then maximize recall.")
    print(f"Best model: {best_model_name}")
    print(f"Saved training outputs to: {ARTIFACTS_DIR}")

    return best_model_name


def save_tuning_results(tuning_results: Mapping[str, dict[str, object]]) -> None:
    """Persist the chosen hyperparameters and fold-level summary scores for tuned runs."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "tuned_cv_results.json").write_text(
        json.dumps(tuning_results, indent=2),
        encoding="utf-8",
    )


def save_overfitting_outputs(tuned_metrics: pd.DataFrame, tuning_results: Mapping[str, dict[str, object]]) -> None:
    """Save train-vs-test gap summaries and model-complexity diagnostics for tuned models."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gap_columns = [
        "model",
        "train_accuracy",
        "accuracy",
        "accuracy_gap",
        "train_recall",
        "recall",
        "recall_gap",
    ]
    gap_summary = tuned_metrics[gap_columns].copy()
    gap_summary = gap_summary.rename(
        columns={
            "accuracy": "test_accuracy",
            "recall": "test_recall",
        }
    )
    gap_summary.to_csv(ARTIFACTS_DIR / "tuned_overfitting_summary.csv", index=False)

    recall_plot_data = tuned_metrics[["model", "train_recall", "recall"]].melt(
        id_vars="model",
        var_name="split",
        value_name="score",
    )
    recall_plot_data["split"] = recall_plot_data["split"].map(
        {
            "train_recall": "Train",
            "recall": "Test",
        }
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=recall_plot_data,
        x="score",
        y="model",
        hue="split",
        orient="h",
    )
    annotate_horizontal_bars(ax, fmt="%.3f")
    plt.title("Tuned Models: Train vs Test Recall")
    plt.xlabel("Recall")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tuned_train_vs_test_recall.png", dpi=200)
    plt.close()

    for model_name, result in tuning_results.items():
        primary_param = result.get("primary_complexity_param")
        primary_label = result.get("primary_complexity_label", primary_param)
        diagnostics = result.get("complexity_curve", [])
        if not primary_param or not diagnostics:
            continue

        diagnostics_frame = pd.DataFrame(diagnostics)

        metric_panels = [
            ("mean_train_accuracy", "mean_validation_accuracy", "Accuracy"),
            ("mean_train_recall", "mean_validation_recall", "Recall"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (train_metric, validation_metric, metric_label) in zip(axes, metric_panels):
            panel = diagnostics_frame.melt(
                id_vars=["complexity_value_label"],
                value_vars=[train_metric, validation_metric],
                var_name="split",
                value_name="score",
            )
            panel["split"] = panel["split"].map(
                {
                    train_metric: "Train CV",
                    validation_metric: "Validation CV",
                }
            )
            sns.lineplot(
                data=panel,
                x="complexity_value_label",
                y="score",
                hue="split",
                style="split",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.set_title(f"{model_name} {metric_label} vs Complexity")
            ax.set_xlabel(primary_label)
            ax.set_ylabel(metric_label)
            ax.tick_params(axis="x", rotation=20)

        handles, labels = axes[0].get_legend_handles_labels()
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()
        if axes[1].get_legend() is not None:
            axes[1].get_legend().remove()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{model_name} Overfitting Diagnostic", y=0.98)
        plt.tight_layout(rect=(0, 0.06, 1, 0.95))
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"tuned_overfitting_{safe_name}.png", dpi=200)
        plt.close()


def build_comparison_table(baseline_metrics: pd.DataFrame, tuned_metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine baseline and tuned metrics into one side-by-side comparison table."""
    # Join the two experiment tracks by model name so the report can compare them directly.
    baseline_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_negatives",
    ]
    tuned_columns = baseline_columns + ["cv_recall"]

    comparison = baseline_metrics[["model", *baseline_columns]].merge(
        tuned_metrics[["model", *tuned_columns]],
        on="model",
        suffixes=("_baseline", "_tuned"),
    )

    # Compute explicit deltas so improvements can be reported without recalculating them elsewhere.
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        comparison[f"delta_{metric_name}"] = (
            comparison[f"{metric_name}_tuned"] - comparison[f"{metric_name}_baseline"]
        )

    comparison["delta_false_negatives"] = (
        comparison["false_negatives_tuned"] - comparison["false_negatives_baseline"]
    )

    return comparison.sort_values(
        by=["false_negatives_tuned", "recall_tuned", "f1_tuned", "accuracy_tuned"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def save_comparison_outputs(baseline_metrics: pd.DataFrame, tuned_metrics: pd.DataFrame) -> pd.DataFrame:
    """Save the baseline-vs-tuned table and the comparison figure."""
    comparison = build_comparison_table(baseline_metrics, tuned_metrics)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(ARTIFACTS_DIR / "baseline_vs_tuned_comparison.csv", index=False)

    # Reshape the comparison table so seaborn can draw side-by-side bars for each experiment track.
    # `melt` turns the wide table into a long format with one row per bar.
    recall_plot_data = comparison[
        ["model", "recall_baseline", "recall_tuned"]
    ].melt(id_vars="model", var_name="experiment", value_name="recall")
    recall_plot_data["experiment"] = recall_plot_data["experiment"].map(
        {
            "recall_baseline": "Baseline",
            "recall_tuned": "Tuned",
        }
    )

    fn_plot_data = comparison[
        ["model", "false_negatives_baseline", "false_negatives_tuned"]
    ].melt(id_vars="model", var_name="experiment", value_name="false_negatives")
    fn_plot_data["experiment"] = fn_plot_data["experiment"].map(
        {
            "false_negatives_baseline": "Baseline",
            "false_negatives_tuned": "Tuned",
        }
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=recall_plot_data,
        x="recall",
        y="model",
        hue="experiment",
        orient="h",
        ax=axes[0],
    )
    annotate_horizontal_bars(axes[0], fmt="%.3f")
    axes[0].set_title("Baseline vs Tuned Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Model")

    sns.barplot(
        data=fn_plot_data,
        x="false_negatives",
        y="model",
        hue="experiment",
        orient="h",
        ax=axes[1],
    )
    annotate_horizontal_bars(axes[1], fmt="%.0f")
    axes[1].set_title("Baseline vs Tuned False Negatives")
    axes[1].set_xlabel("False Negatives")
    axes[1].set_ylabel("")

    left_legend = axes[0].get_legend()
    handles = left_legend.legend_handles
    labels = [text.get_text() for text in left_legend.get_texts()]
    left_legend.remove()
    axes[1].get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.suptitle("Baseline vs Tuned Model Comparison", y=0.98)
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.savefig(FIGURES_DIR / "baseline_vs_tuned_comparison.png", dpi=200)
    plt.close()

    return comparison