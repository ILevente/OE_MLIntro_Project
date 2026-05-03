# Implementation Summary

This document summarizes the current heart disease classification project implementation.

## 1. Dataset inspection completed

The project is built around the dataset file [UCI_Heart_Disease_Dataset_Combined.csv](./UCI_Heart_Disease_Dataset_Combined.csv).

Verified dataset facts:

- Rows: 2,943
- Columns: 11
- Target column: `HeartDisease`
- Target distribution: 1,329 negative cases and 1,614 positive cases
- Feature columns:
  - `Age`
  - `Sex`
  - `ChestPainType`
  - `RestingBP`
  - `Cholesterol`
  - `FastingBS`
  - `RestingECG`
  - `MaxHR`
  - `ExerciseAngina`
  - `Oldpeak`

## 2. Project scaffold created

The current project structure includes:

- [README.md](./README.md)
- [requirements.txt](./requirements.txt)
- [heart_disease_baseline/config.py](./heart_disease_baseline/config.py)
- [heart_disease_baseline/data.py](./heart_disease_baseline/data.py)
- [heart_disease_baseline/eda.py](./heart_disease_baseline/eda.py)
- [heart_disease_baseline/models.py](./heart_disease_baseline/models.py)
- [heart_disease_baseline/reporting.py](./heart_disease_baseline/reporting.py)
- [heart_disease_baseline/train.py](./heart_disease_baseline/train.py)

Purpose of each core file:

| File | Purpose |
| --- | --- |
| [heart_disease_baseline/config.py](./heart_disease_baseline/config.py) | Shared experiment constants, paths, and model-selection priority |
| [heart_disease_baseline/data.py](./heart_disease_baseline/data.py) | Dataset loading, validation, splitting, and raw-data summaries |
| [heart_disease_baseline/eda.py](./heart_disease_baseline/eda.py) | EDA plots and dataset summary artifact generation |
| [heart_disease_baseline/models.py](./heart_disease_baseline/models.py) | Baseline registries, tuned registries, and validation-curve metadata |
| [heart_disease_baseline/reporting.py](./heart_disease_baseline/reporting.py) | Metrics export, plots, comparison charts, and tuned diagnostics |
| [heart_disease_baseline/train.py](./heart_disease_baseline/train.py) | Baseline/tuned training orchestration and CV-based evaluation |

## 3. Baseline ML pipeline implemented

The baseline workflow includes:

- binary classification using `HeartDisease` as the target
- a stratified train/test split
- scaling for scale-sensitive models
- five baseline models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Machine
- evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- best-model selection prioritized by false negatives, recall, F1, then accuracy

This remains the clean untuned comparison path.

## 4. Follow-up optimization pipeline implemented

The tuned workflow includes:

- the same stratified outer split as the baseline run
- 5-fold stratified cross-validation on the training portion only
- `GridSearchCV` tuning for all five models
- tuned-only preprocessing that converts invalid `Cholesterol == 0` and `RestingBP == 0` values to missing values and imputes with the training median
- `class_weight` tuning for Logistic Regression, Decision Tree, Random Forest, and SVM
- a stricter Random Forest `min_samples_leaf` search of `[2, 3, 4]`
- refit based on recall so tuning stays aligned with the screening objective
- separate tuned metrics, confusion matrices, best-model export, and figures

Current limitation: some categorical variables are still label-coded and treated numerically by the models rather than being explicitly encoded as categorical features. The clearest examples are `ChestPainType` and `RestingECG`, and the binary-coded fields `Sex`, `FastingBS`, and `ExerciseAngina` are also currently passed through as numeric values.

The project now preserves both baseline and tuned outputs in the same training run.

## 5. EDA and training scripts verified

The implemented scripts were executed successfully in the local virtual environment.

Commands used:

```powershell
python -m heart_disease_baseline.eda
python -m heart_disease_baseline.train
```

Both commands generated outputs under [artifacts](./artifacts).

## 6. Generated outputs

Created output files:

- [artifacts/dataset_summary.json](./artifacts/dataset_summary.json)
- [artifacts/metrics.csv](./artifacts/metrics.csv)
- [artifacts/confusion_matrices.json](./artifacts/confusion_matrices.json)
- [artifacts/selection_summary.json](./artifacts/selection_summary.json)
- [artifacts/best_model.joblib](./artifacts/best_model.joblib)
- [artifacts/baseline_vs_tuned_comparison.csv](./artifacts/baseline_vs_tuned_comparison.csv)
- [artifacts/tuned_metrics.csv](./artifacts/tuned_metrics.csv)
- [artifacts/tuned_confusion_matrices.json](./artifacts/tuned_confusion_matrices.json)
- [artifacts/tuned_selection_summary.json](./artifacts/tuned_selection_summary.json)
- [artifacts/tuned_cv_results.json](./artifacts/tuned_cv_results.json)
- [artifacts/tuned_best_model.joblib](./artifacts/tuned_best_model.joblib)
- [artifacts/tuned_train_vs_test_gap_summary.csv](./artifacts/tuned_train_vs_test_gap_summary.csv)

Created figures:

- [artifacts/figures/target_distribution.png](./artifacts/figures/target_distribution.png)
- [artifacts/figures/age_distribution.png](./artifacts/figures/age_distribution.png)
- [artifacts/figures/correlation_heatmap.png](./artifacts/figures/correlation_heatmap.png)
- [artifacts/figures/model_recall_scores.png](./artifacts/figures/model_recall_scores.png)
- [artifacts/figures/confusion_matrix_logistic_regression.png](./artifacts/figures/confusion_matrix_logistic_regression.png)
- [artifacts/figures/confusion_matrix_decision_tree.png](./artifacts/figures/confusion_matrix_decision_tree.png)
- [artifacts/figures/confusion_matrix_random_forest.png](./artifacts/figures/confusion_matrix_random_forest.png)
- [artifacts/figures/confusion_matrix_knn.png](./artifacts/figures/confusion_matrix_knn.png)
- [artifacts/figures/confusion_matrix_svm.png](./artifacts/figures/confusion_matrix_svm.png)
- [artifacts/figures/baseline_vs_tuned_comparison.png](./artifacts/figures/baseline_vs_tuned_comparison.png)
- [artifacts/figures/tuned_model_recall_scores.png](./artifacts/figures/tuned_model_recall_scores.png)
- [artifacts/figures/tuned_refit_train_vs_test_recall.png](./artifacts/figures/tuned_refit_train_vs_test_recall.png)
- [artifacts/figures/tuned_confusion_matrix_logistic_regression.png](./artifacts/figures/tuned_confusion_matrix_logistic_regression.png)
- [artifacts/figures/tuned_confusion_matrix_decision_tree.png](./artifacts/figures/tuned_confusion_matrix_decision_tree.png)
- [artifacts/figures/tuned_confusion_matrix_random_forest.png](./artifacts/figures/tuned_confusion_matrix_random_forest.png)
- [artifacts/figures/tuned_confusion_matrix_knn.png](./artifacts/figures/tuned_confusion_matrix_knn.png)
- [artifacts/figures/tuned_confusion_matrix_svm.png](./artifacts/figures/tuned_confusion_matrix_svm.png)
- [artifacts/figures/tuned_cv_complexity_logistic_regression.png](./artifacts/figures/tuned_cv_complexity_logistic_regression.png)
- [artifacts/figures/tuned_cv_complexity_decision_tree.png](./artifacts/figures/tuned_cv_complexity_decision_tree.png)
- [artifacts/figures/tuned_cv_complexity_random_forest.png](./artifacts/figures/tuned_cv_complexity_random_forest.png)
- [artifacts/figures/tuned_cv_complexity_knn.png](./artifacts/figures/tuned_cv_complexity_knn.png)
- [artifacts/figures/tuned_cv_complexity_svm.png](./artifacts/figures/tuned_cv_complexity_svm.png)

## 7. Current baseline results

The current baseline results are:

| Model | Accuracy | Precision | Recall | F1-score | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8812 | 0.8667 | 0.9257 | 0.8952 | 24 |
| Decision Tree | 0.7691 | 0.7664 | 0.8328 | 0.7982 | 54 |
| SVM | 0.7861 | 0.7906 | 0.8297 | 0.8097 | 55 |
| KNN | 0.7844 | 0.7988 | 0.8111 | 0.8049 | 61 |
| Logistic Regression | 0.7216 | 0.7304 | 0.7802 | 0.7545 | 71 |

Current best baseline model: **Random Forest**

## 8. Current tuned results

The tuned training run produced the following evaluation results on the held-out test split:

| Model | Accuracy | Precision | Recall | F1-score | False negatives | CV Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8829 | 0.8713 | 0.9226 | 0.8962 | 25 | 0.9009 |
| KNN | 0.8540 | 0.8516 | 0.8885 | 0.8697 | 36 | 0.8854 |
| SVM | 0.7810 | 0.7803 | 0.8359 | 0.8072 | 53 | 0.8180 |
| Decision Tree | 0.7674 | 0.7642 | 0.8328 | 0.7970 | 54 | 0.8257 |
| Logistic Regression | 0.7216 | 0.7304 | 0.7802 | 0.7545 | 71 | 0.7862 |

Current best tuned model: **Random Forest**

Best hyperparameters found during cross-validation:

| Model | Best parameters |
| --- | --- |
| Logistic Regression | `C=0.1`, solver=`lbfgs` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=4` |
| Random Forest | `n_estimators=300`, `max_depth=None`, `min_samples_leaf=2` |
| KNN | `n_neighbors=21`, `weights=distance`, `p=1` |
| SVM | `C=0.5`, `gamma=scale` |

`class_weight` was added to the tuned search space for all eligible models, but the best configuration for each of those models still selected `class_weight=None`.

## 9. Underfitting and overfitting diagnostics

The tuned workflow also exports explicit fit diagnostics:

- [artifacts/tuned_train_vs_test_gap_summary.csv](./artifacts/tuned_train_vs_test_gap_summary.csv), which compares final refit train and test accuracy/recall for each tuned model
- [artifacts/figures/tuned_refit_train_vs_test_recall.png](./artifacts/figures/tuned_refit_train_vs_test_recall.png), a summary figure for final refit train-vs-test recall gaps
- one per-model cross-validated complexity plot showing the best GridSearchCV result found at each primary complexity value

Those per-model complexity plots now reflect the actual `GridSearchCV` search space on the primary parameter rather than a separate post-hoc sweep.

Current train-vs-test accuracy gaps for the tuned models:

| Model | Train Accuracy | Test Accuracy | Accuracy Gap |
| --- | ---: | ---: | ---: |
| KNN | 1.0000 | 0.8540 | 0.1460 |
| Random Forest | 0.9809 | 0.8829 | 0.0980 |
| Logistic Regression | 0.7438 | 0.7216 | 0.0223 |
| Decision Tree | 0.7782 | 0.7674 | 0.0108 |
| SVM | 0.7910 | 0.7810 | 0.0100 |

The strongest overfitting signal remains in KNN and Random Forest.

## 10. Baseline vs tuned comparison table

The training workflow exports a side-by-side comparison table in [artifacts/baseline_vs_tuned_comparison.csv](./artifacts/baseline_vs_tuned_comparison.csv) and a companion figure in [artifacts/figures/baseline_vs_tuned_comparison.png](./artifacts/figures/baseline_vs_tuned_comparison.png).

Key comparison values:

| Model | Baseline Recall | Tuned Recall | Recall Delta | Baseline FN | Tuned FN | FN Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9226 | 0.9226 | +0.0000 | 25 | 25 | 0 |
| KNN | 0.8266 | 0.8885 | +0.0619 | 56 | 36 | -20 |
| SVM | 0.7864 | 0.8359 | +0.0495 | 69 | 53 | -16 |
| Decision Tree | 0.7926 | 0.8328 | +0.0402 | 67 | 54 | -13 |
| Logistic Regression | 0.7802 | 0.7802 | +0.0000 | 71 | 71 | 0 |

## 11. Partner follow-up changes

Two follow-up ideas were implemented only in the tuned workflow:

1. The 225 zero-cholesterol rows and the single `RestingBP == 0` row are treated as invalid values only for tuned-model preprocessing, then imputed with the training-set median.
2. Negative `Oldpeak` values are currently left unchanged, because the source UCI field describes `Oldpeak` as ST depression induced by exercise relative to rest, so negative values are not clearly invalid from the dataset documentation.
3. The Random Forest tuning grid restricts `min_samples_leaf` to `[2, 3, 4]` instead of allowing `1`.

Measured effect on the tuned results, compared with the previous tuned setup:

| Model | Previous Tuned Recall | New Tuned Recall | Recall Delta | Previous FN | New FN | FN Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9319 | 0.9257 | -0.0062 | 22 | 24 | +2 |
| KNN | 0.9009 | 0.8916 | -0.0093 | 32 | 35 | +3 |
| Decision Tree | 0.8607 | 0.8328 | -0.0279 | 45 | 54 | +9 |
| SVM | 0.8080 | 0.8359 | +0.0279 | 62 | 53 | -9 |
| Logistic Regression | 0.7740 | 0.7802 | +0.0062 | 73 | 71 | -2 |

## 12. What is ready for the course project

The repo now provides:

- a verified dataset loader
- a reproducible experiment structure
- starter exploratory plots
- baseline model comparison
- tuned model comparison with cross-validation
- tuned-only preprocessing for invalid cholesterol values
- baseline-vs-tuned comparison outputs
- tuned fit diagnostics
- exported metrics and confusion matrices
- report-ready figures

## 13. Recommended next steps

1. Whether the label-coded categorical variables should be encoded explicitly as categorical features instead of staying numeric. The main candidates are `ChestPainType` and `RestingECG`, with `Sex`, `FastingBS`, and `ExerciseAngina` also worth reviewing.
2. Whether the negative `Oldpeak` values should remain as-is or be justified explicitly in the report as clinically plausible rather than invalid.
3. Which baseline and tuned tables should be shown side by side in the report.
4. Whether to add one extra evaluation view such as ROC-AUC.
