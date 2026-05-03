"""Runs the train-test experiment, ranks models, and saves metrics, plots, and the best model."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, validation_curve

from heart_disease_baseline.config import (
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    TUNING_REFIT_METRIC,
)
from heart_disease_baseline.data import load_dataset, split_features_target
from heart_disease_baseline.models import build_model_registry, build_tuning_registry
from heart_disease_baseline.reporting import (
    save_comparison_outputs,
    save_experiment_outputs,
    save_overfitting_outputs,
    save_tuning_results,
)


def build_validation_curve(
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    primary_param: str,
    primary_label: str,
    ordered_values: list[object],
) -> tuple[str, str, list[dict[str, float | str | int | None]]]:
    """Build a denser validation curve for one model while keeping other params fixed."""
    train_accuracy_scores, validation_accuracy_scores = validation_curve(
        estimator=clone(estimator),
        X=X_train,
        y=y_train,
        param_name=primary_param,
        param_range=ordered_values,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    train_recall_scores, validation_recall_scores = validation_curve(
        estimator=clone(estimator),
        X=X_train,
        y=y_train,
        param_name=primary_param,
        param_range=ordered_values,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
    )
    rows: list[dict[str, float | str | int | None]] = []

    for index, value in enumerate(ordered_values):
        rows.append(
            {
                "complexity_value": value,
                "complexity_value_label": str(value),
                "mean_train_accuracy": float(train_accuracy_scores[index].mean()),
                "mean_validation_accuracy": float(validation_accuracy_scores[index].mean()),
                "mean_train_recall": float(train_recall_scores[index].mean()),
                "mean_validation_recall": float(validation_recall_scores[index].mean()),
            }
        )

    return primary_param, primary_label, rows


def select_best_model(metrics: pd.DataFrame) -> pd.DataFrame:
    """Sort models using the project priority: few false negatives, then higher recall."""
    # Rank models using the screening-oriented priority defined in config.py.
    return metrics.sort_values(
        by=["false_negatives", "recall", "f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the single stratified train-test split shared by all experiments."""
    dataframe = load_dataset()
    features, target = split_features_target(dataframe)

    # This is the single outer split used for final evaluation.
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    predictions: pd.Series | pd.Index | list[int],
) -> tuple[dict[str, float | str], list[list[int]]]:
    """Compute the standard classification metrics and confusion matrix for one model."""
    # Build all downstream metrics from one confusion matrix + one prediction vector.
    matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()

    return (
        {
            "model": model_name,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions),
            "recall": recall_score(y_true, predictions),
            "f1": f1_score(y_true, predictions),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        matrix.tolist(),
    )


def run_baseline_experiment(
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None = None,
) -> pd.DataFrame:
    """Train the fixed baseline models once and evaluate them on the holdout test set."""
    if split_data is None:
        split_data = build_train_test_split()

    X_train, X_test, y_train, y_test = split_data

    metrics_rows: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    fitted_models: dict[str, object] = {}

    # The baseline path trains each model once with its fixed settings and scores it on the holdout test set.
    for model_name, model in build_model_registry().items():
        # No cross-validation here: fit once on X_train, then evaluate once on X_test.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics_row, matrix = evaluate_predictions(model_name, y_test, predictions)
        metrics_rows.append(metrics_row)
        confusion_matrices[model_name] = matrix
        fitted_models[model_name] = model

    metrics = select_best_model(pd.DataFrame(metrics_rows))
    save_experiment_outputs(metrics, confusion_matrices, fitted_models)

    return metrics


def run_tuned_experiment(
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None = None,
) -> pd.DataFrame:
    """Tune each model with cross-validation, then score the best version on the test set."""
    if split_data is None:
        split_data = build_train_test_split()

    X_train, X_test, y_train, y_test = split_data
    # Cross-validation happens only inside the training split; the test split stays untouched.
    # Stratification keeps the class ratio similar in each fold.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    metrics_rows: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    fitted_models: dict[str, object] = {}
    tuning_results: dict[str, dict[str, object]] = {}

    for model_name, search_config in build_tuning_registry().items():
        estimator = search_config["estimator"]
        param_grid = search_config["param_grid"]
        primary_param = search_config["primary_complexity_param"]
        primary_label = search_config["primary_complexity_label"]
        curve_values = search_config["validation_curve_values"]
        # Evaluate each hyperparameter combination across the same stratified folds.
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            # Store several scoring views, but pick the best parameters by recall.
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            refit=TUNING_REFIT_METRIC,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        # After CV finishes, sklearn refits the winning configuration on the full training split.
        best_model = search.best_estimator_
        train_predictions = best_model.predict(X_train)
        predictions = best_model.predict(X_test)
        metrics_row, matrix = evaluate_predictions(model_name, y_test, predictions)
        train_metrics, _ = evaluate_predictions(model_name, y_train, train_predictions)
        primary_param, primary_label, complexity_curve = build_validation_curve(
            best_model,
            X_train,
            y_train,
            cv,
            primary_param,
            primary_label,
            curve_values,
        )
        # Keep both the final test metrics and the best cross-validation scores for reporting.
        metrics_row.update(
            {
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_recall": float(train_metrics["recall"]),
                "accuracy_gap": float(train_metrics["accuracy"] - metrics_row["accuracy"]),
                "recall_gap": float(train_metrics["recall"] - metrics_row["recall"]),
                "cv_accuracy": float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
                "cv_precision": float(search.cv_results_["mean_test_precision"][search.best_index_]),
                "cv_recall": float(search.cv_results_["mean_test_recall"][search.best_index_]),
                "cv_f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            }
        )
        metrics_rows.append(metrics_row)
        confusion_matrices[model_name] = matrix
        fitted_models[model_name] = best_model
        tuning_results[model_name] = {
            "best_params": search.best_params_,
            # `best_index_` points to the winning row inside cv_results_.
            "best_cv_scores": {
                "accuracy": float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
                "precision": float(search.cv_results_["mean_test_precision"][search.best_index_]),
                "recall": float(search.cv_results_["mean_test_recall"][search.best_index_]),
                "f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            },
            "primary_complexity_param": primary_param,
            "primary_complexity_label": primary_label,
            "complexity_curve": complexity_curve,
        }

    metrics = select_best_model(pd.DataFrame(metrics_rows))
    save_experiment_outputs(
        metrics,
        confusion_matrices,
        fitted_models,
        prefix="tuned_",
        selection_summary_extra={
            "cv_folds": CV_FOLDS,
            "refit_metric": TUNING_REFIT_METRIC,
        },
    )
    # Store the chosen hyperparameters and fold-averaged scores separately from the final test metrics.
    save_tuning_results(tuning_results)
    save_overfitting_outputs(metrics, tuning_results)

    return metrics


def run_all_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the baseline run, the tuned run, and the final comparison export."""
    # Reuse one outer split so the baseline and tuned runs are compared on the same test set.
    split_data = build_train_test_split()
    baseline_metrics = run_baseline_experiment(split_data)
    tuned_metrics = run_tuned_experiment(split_data)
    save_comparison_outputs(baseline_metrics, tuned_metrics)
    return baseline_metrics, tuned_metrics


if __name__ == "__main__":
    run_all_experiments()