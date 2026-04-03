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
- [heart_disease_baseline/train.py](./heart_disease_baseline/train.py)

Purpose of each file:

| File | Purpose |
|---|---|
| [heart_disease_baseline/config.py](./heart_disease_baseline/config.py) | Central configuration for paths, column names, and experiment constants |
| [heart_disease_baseline/data.py](./heart_disease_baseline/data.py) | Dataset loading, schema validation, feature-target split, and dataset summary creation |
| [heart_disease_baseline/eda.py](./heart_disease_baseline/eda.py) | Basic exploratory analysis and figure generation |
| [heart_disease_baseline/models.py](./heart_disease_baseline/models.py) | Baseline model definitions |
| [heart_disease_baseline/train.py](./heart_disease_baseline/train.py) | Train/test split, model training, evaluation, and artifact export |

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

## 4. EDA and training scripts verified

The implemented scripts were executed successfully in the local virtual environment.

Commands used:

```powershell
python -m heart_disease_baseline.eda
python -m heart_disease_baseline.train
```

Both ran successfully and generated outputs in the [artifacts](./artifacts) folder.

## 5. Generated outputs

Created output files:

- [artifacts/dataset_summary.json](./artifacts/dataset_summary.json)
- [artifacts/metrics.csv](./artifacts/metrics.csv)
- [artifacts/confusion_matrices.json](./artifacts/confusion_matrices.json)
- [artifacts/selection_summary.json](./artifacts/selection_summary.json)
- [artifacts/best_model.joblib](./artifacts/best_model.joblib)

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

## 6. Current baseline results

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

## 7. What is ready for the course project

The project already provides a usable starting point for the assignment:

- a verified dataset loader
- a reproducible experiment structure
- starter exploratory plots
- baseline model comparison
- exported metrics and confusion matrices
- figures that can be reused in the report

## 8. Recommended next steps

Before turning this into the final submission, the team should decide:

1. Whether encoded categorical features should stay numeric or be one-hot encoded.
2. How to handle suspicious values such as zero cholesterol rows.
3. Which plots and tables should be included in the 10-15 page report.
4. When to introduce cross-validation and hyperparameter tuning after the baseline report structure is stable.
