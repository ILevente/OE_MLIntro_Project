# Implementation Summary

This document summarizes the baseline Python project implemented so far for the heart disease prediction assignment.

## 1. Dataset inspection completed

The project is built around the dataset file [UCI_Heart_Disease_Dataset_Combined.csv](./UCI_Heart_Disease_Dataset_Combined.csv).

Verified dataset facts:

- Rows: 2,943
- Columns: 11
- Target column: `HeartDisease`
- Target distribution: 1,329 negative cases and 1,614 positive cases
- Missing blank cells: none detected

Detected data caveats worth discussing in the report:

- `Cholesterol == 0` appears 225 times
- `RestingBP == 0` appears 1 time
- `Oldpeak < 0` appears 13 times

The dataset columns currently used by the baseline are:

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
- `HeartDisease`

## 2. Project scaffold created

The following baseline Python project structure was added:

- [README.md](./README.md)
- [requirements.txt](./requirements.txt)
- [.gitignore](./.gitignore)
- [heart_disease_baseline/config.py](./heart_disease_baseline/config.py)
- [heart_disease_baseline/data.py](./heart_disease_baseline/data.py)
- [heart_disease_baseline/eda.py](./heart_disease_baseline/eda.py)
- [heart_disease_baseline/models.py](./heart_disease_baseline/models.py)
- [heart_disease_baseline/reporting.py](./heart_disease_baseline/reporting.py)
- [heart_disease_baseline/train.py](./heart_disease_baseline/train.py)

Purpose of each file:

| File | Purpose |
|---|---|
| [heart_disease_baseline/config.py](./heart_disease_baseline/config.py) | Central configuration for paths, column names, and experiment constants |
| [heart_disease_baseline/data.py](./heart_disease_baseline/data.py) | Dataset loading, schema validation, feature-target split, and dataset summary creation |
| [heart_disease_baseline/eda.py](./heart_disease_baseline/eda.py) | Basic exploratory analysis and figure generation |
| [heart_disease_baseline/models.py](./heart_disease_baseline/models.py) | Baseline model definitions and hyperparameter search spaces |
| [heart_disease_baseline/reporting.py](./heart_disease_baseline/reporting.py) | Metrics export, JSON/CSV output writing, and model comparison visualizations |
| [heart_disease_baseline/train.py](./heart_disease_baseline/train.py) | Train/test split, baseline training, cross-validated tuning, and evaluation orchestration |

## 3. Baseline ML pipeline implemented

The current baseline includes:

- Binary classification using `HeartDisease` as the target
- Train/test split with stratification
- `StandardScaler` for models that need scaling
- Five baseline models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Machine
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Best-model selection rule focused on minimizing false negatives and maximizing recall

This is intentionally a clean intro-level baseline, not a tuned final system.

## 4. Follow-up optimization pipeline implemented

The training workflow now also includes a separate tuned experiment that preserves the baseline outputs.

The tuned workflow includes:

- the same stratified train/test split used by the baseline experiment
- 5-fold stratified cross-validation on the training portion only
- `GridSearchCV` hyperparameter tuning for all five models
- refit based on recall so model selection stays aligned with the heart-disease screening goal
- separate tuned metrics, confusion matrices, best-model export, and figure files

This means one training run now produces:

- baseline outputs for the original reportable comparison
- tuned outputs for the optimized comparison

## 5. EDA and training scripts verified

The implemented scripts were executed successfully in the local virtual environment.

Commands used:

```powershell
python -m heart_disease_baseline.eda
python -m heart_disease_baseline.train
```

Both ran successfully and generated outputs in the [artifacts](./artifacts) folder.

The training command now executes both the baseline comparison and the tuned cross-validated comparison in sequence.

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
- [artifacts/figures/tuned_confusion_matrix_logistic_regression.png](./artifacts/figures/tuned_confusion_matrix_logistic_regression.png)
- [artifacts/figures/tuned_confusion_matrix_decision_tree.png](./artifacts/figures/tuned_confusion_matrix_decision_tree.png)
- [artifacts/figures/tuned_confusion_matrix_random_forest.png](./artifacts/figures/tuned_confusion_matrix_random_forest.png)
- [artifacts/figures/tuned_confusion_matrix_knn.png](./artifacts/figures/tuned_confusion_matrix_knn.png)
- [artifacts/figures/tuned_confusion_matrix_svm.png](./artifacts/figures/tuned_confusion_matrix_svm.png)

## 7. Current baseline results

The baseline training run produced the following evaluation results:

| Model | Accuracy | Precision | Recall | F1-score | False negatives |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8812 | 0.8688 | 0.9226 | 0.8949 | 25 |
| KNN | 0.7895 | 0.7970 | 0.8266 | 0.8116 | 56 |
| Decision Tree | 0.7810 | 0.8050 | 0.7926 | 0.7988 | 67 |
| SVM | 0.7725 | 0.7962 | 0.7864 | 0.7913 | 69 |
| Logistic Regression | 0.7131 | 0.7225 | 0.7740 | 0.7474 | 73 |

Current best baseline model: **Random Forest**

The best model is now selected using a heart-disease-oriented rule:

1. Lowest false negatives
2. Highest recall
3. Highest F1-score
4. Highest accuracy

This makes the selection better aligned with a screening scenario where missing a patient with disease is the most important error to reduce.

## 8. Current tuned results

The tuned training run produced the following evaluation results on the held-out test split:

| Model | Accuracy | Precision | Recall | F1-score | False negatives | CV Recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8896 | 0.8750 | 0.9319 | 0.9025 | 22 | 0.9086 |
| KNN | 0.8659 | 0.8609 | 0.9009 | 0.8805 | 32 | 0.8792 |
| Decision Tree | 0.8489 | 0.8634 | 0.8607 | 0.8620 | 45 | 0.8528 |
| SVM | 0.7708 | 0.7814 | 0.8080 | 0.7945 | 62 | 0.8211 |
| Logistic Regression | 0.7131 | 0.7225 | 0.7740 | 0.7474 | 73 | 0.7816 |

Current best tuned model: **Random Forest**

Best hyperparameters found during cross-validation:

| Model | Best parameters |
|---|---|
| Logistic Regression | `C=0.1`, solver=`lbfgs` |
| Decision Tree | `max_depth=None`, `min_samples_leaf=1` |
| Random Forest | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=1` |
| KNN | `n_neighbors=21`, `weights=distance`, `p=1` |
| SVM | `C=0.5`, `gamma=scale` |

Compared with the baseline, tuning improved Random Forest, KNN, and Decision Tree noticeably on recall and false negatives, while Logistic Regression stayed unchanged because the best cross-validated setting effectively matched the baseline behavior on the held-out split.

## 9. Baseline vs tuned comparison table

The training workflow now exports a direct side-by-side comparison table in [artifacts/baseline_vs_tuned_comparison.csv](./artifacts/baseline_vs_tuned_comparison.csv).

It also exports a visual comparison chart in [artifacts/figures/baseline_vs_tuned_comparison.png](./artifacts/figures/baseline_vs_tuned_comparison.png), showing baseline vs tuned values for recall and false negatives for each model.

Key comparison values from the generated table:

| Model | Baseline Recall | Tuned Recall | Recall Delta | Baseline FN | Tuned FN | FN Delta |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9226 | 0.9319 | +0.0093 | 25 | 22 | -3 |
| KNN | 0.8266 | 0.9009 | +0.0743 | 56 | 32 | -24 |
| Decision Tree | 0.7926 | 0.8607 | +0.0681 | 67 | 45 | -22 |
| SVM | 0.7864 | 0.8080 | +0.0217 | 69 | 62 | -7 |
| Logistic Regression | 0.7740 | 0.7740 | +0.0000 | 73 | 73 | 0 |

This table is useful for the report because it shows not only which tuned model is best, but also which models actually benefited the most from tuning.

## 10. What is ready for the course project

The project already provides a usable starting point for the assignment:

- a verified dataset loader
- a reproducible experiment structure
- starter exploratory plots
- baseline model comparison
- tuned model comparison with cross-validation
- baseline-vs-tuned comparison table
- exported metrics and confusion matrices
- figures that can be reused in the report

## 11. Recommended next steps

Before turning this into the final submission, the team should decide:

1. Whether encoded categorical features should stay numeric or be one-hot encoded.
2. How to handle suspicious values such as zero cholesterol rows.
3. Which baseline and tuned tables should be shown side by side in the 10-15 page report.
4. Whether to add one extra evaluation view such as ROC-AUC or a baseline-vs-tuned comparison chart.
